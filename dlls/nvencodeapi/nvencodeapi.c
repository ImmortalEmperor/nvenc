/*
 * Copyright (C) 2015 Michael Müller
 * Copyright (C) 2025 Sveinar Søpler
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301, USA
 */

#include "config.h"
#include <dlfcn.h>

#include <stdarg.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define COBJMACROS
#include "windef.h"
#include "winbase.h"
#include "wine/debug.h"

#include <d3d11.h>
#include <d3d12.h>

#include "nvencodeapi.h"

WINE_DEFAULT_DEBUG_CHANNEL(nvencodeapi);

/* Windows apps pass D3D11 and D3D12 resources to NVENC, but the Linux driver
 * can't use them. DXVK and vkd3d-proton back them with Vulkan memory, so export
 * that as an fd and import it into CUDA. */

static void *libnvidia_encode_handle = NULL;
static LINUX_NV_ENCODE_API_FUNCTION_LIST origFunctions;

static NVENCSTATUS (*pNvEncodeAPICreateInstance)(LINUX_NV_ENCODE_API_FUNCTION_LIST *functionList);
static NVENCSTATUS (*pNvEncodeAPIGetMaxSupportedVersion)(uint32_t* version);

/* Declared here to avoid the full cuda.h. Loaded at runtime from nvcuda.dll. */

typedef int CUresult_local;
#define CUDA_SUCCESS_LOCAL 0

typedef int CUdevice_local;
typedef struct CUctx_st *CUcontext_local;
typedef struct CUarray_st *CUarray_local;
typedef struct CUmipmappedArray_st *CUmipmappedArray_local;
typedef struct CUextMemory_st *CUexternalMemory_local;

typedef struct { char bytes[16]; } CUuuid_local;

typedef enum {
    CU_AD_FORMAT_UNSIGNED_INT8_LOCAL = 0x01,
    CU_AD_FORMAT_SIGNED_INT8_LOCAL   = 0x08,
} CUarray_format_local;

#define CUDA_ARRAY3D_SURFACE_LDST_LOCAL  0x02

typedef struct {
    size_t Width;
    size_t Height;
    size_t Depth;
    CUarray_format_local Format;
    unsigned int NumChannels;
    unsigned int Flags;
} CUDA_ARRAY3D_DESCRIPTOR_local;

typedef enum {
    CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_LOCAL = 1,
} CUexternalMemoryHandleType_local;

typedef struct {
    CUexternalMemoryHandleType_local type;
    union {
        int fd;
        struct { void *handle; const void *name; } win32;
        const void *nvSciBufObject;
    } handle;
    unsigned long long size;
    unsigned int flags;
    unsigned int reserved[16];
} CUDA_EXTERNAL_MEMORY_HANDLE_DESC_local;

typedef struct {
    unsigned long long offset;
    CUDA_ARRAY3D_DESCRIPTOR_local arrayDesc;
    unsigned int numLevels;
    unsigned int reserved[16];
} CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_local;

typedef CUresult_local (WINAPI *PFN_cuInit)(unsigned int Flags);
typedef CUresult_local (WINAPI *PFN_cuDeviceGet)(CUdevice_local *device, int ordinal);
typedef CUresult_local (WINAPI *PFN_cuDeviceGetCount)(int *count);
typedef CUresult_local (WINAPI *PFN_cuDeviceGetUuid)(CUuuid_local *uuid, CUdevice_local dev);
typedef CUresult_local (WINAPI *PFN_cuCtxCreate)(CUcontext_local *pctx, unsigned int flags, CUdevice_local dev);
typedef CUresult_local (WINAPI *PFN_cuCtxDestroy)(CUcontext_local ctx);
typedef CUresult_local (WINAPI *PFN_cuCtxPushCurrent)(CUcontext_local ctx);
typedef CUresult_local (WINAPI *PFN_cuCtxPopCurrent)(CUcontext_local *pctx);
typedef CUresult_local (WINAPI *PFN_cuImportExternalMemory)(CUexternalMemory_local *out, const CUDA_EXTERNAL_MEMORY_HANDLE_DESC_local *desc);
typedef CUresult_local (WINAPI *PFN_cuDestroyExternalMemory)(CUexternalMemory_local em);
typedef CUresult_local (WINAPI *PFN_cuExternalMemoryGetMappedMipmappedArray)(CUmipmappedArray_local *out, CUexternalMemory_local em, const CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_local *desc);
typedef CUresult_local (WINAPI *PFN_cuMipmappedArrayGetLevel)(CUarray_local *out, CUmipmappedArray_local mip, unsigned int level);
typedef CUresult_local (WINAPI *PFN_cuMipmappedArrayDestroy)(CUmipmappedArray_local mip);

static HMODULE g_nvcuda_handle = NULL;
static BOOL    g_cuda_available = FALSE;
static BOOL    g_cuda_initialised = FALSE;

static PFN_cuInit                                 pCuInit = NULL;
static PFN_cuDeviceGet                            pCuDeviceGet = NULL;
static PFN_cuDeviceGetCount                       pCuDeviceGetCount = NULL;
static PFN_cuDeviceGetUuid                        pCuDeviceGetUuid = NULL;
static PFN_cuCtxCreate                            pCuCtxCreate = NULL;
static PFN_cuCtxDestroy                           pCuCtxDestroy = NULL;
static PFN_cuCtxPushCurrent                       pCuCtxPushCurrent = NULL;
static PFN_cuCtxPopCurrent                        pCuCtxPopCurrent = NULL;
static PFN_cuImportExternalMemory                 pCuImportExternalMemory = NULL;
static PFN_cuDestroyExternalMemory                pCuDestroyExternalMemory = NULL;
static PFN_cuExternalMemoryGetMappedMipmappedArray pCuExternalMemoryGetMappedMipmappedArray = NULL;
static PFN_cuMipmappedArrayGetLevel               pCuMipmappedArrayGetLevel = NULL;
static PFN_cuMipmappedArrayDestroy                pCuMipmappedArrayDestroy = NULL;

static BOOL load_cuda(void)
{
    if (g_cuda_initialised)
        return g_cuda_available;
    g_cuda_initialised = TRUE;

    g_nvcuda_handle = LoadLibraryA("nvcuda.dll");
    if (!g_nvcuda_handle)
    {
        WARN("nvcuda.dll not available; DX→CUDA interop disabled.\n");
        return FALSE;
    }

#define LOAD(sym, var) do { var = (void *)GetProcAddress(g_nvcuda_handle, sym); if (!var) { \
        WARN("nvcuda.dll missing symbol %s; interop disabled.\n", sym); \
        FreeLibrary(g_nvcuda_handle); g_nvcuda_handle = NULL; return FALSE; } } while (0)

    LOAD("cuInit",                                 pCuInit);
    LOAD("cuDeviceGet",                            pCuDeviceGet);
    LOAD("cuDeviceGetCount",                       pCuDeviceGetCount);
    LOAD("cuDeviceGetUuid",                        pCuDeviceGetUuid);
    LOAD("cuCtxCreate",                            pCuCtxCreate);
    LOAD("cuCtxDestroy",                           pCuCtxDestroy);
    LOAD("cuCtxPushCurrent",                       pCuCtxPushCurrent);
    LOAD("cuCtxPopCurrent",                        pCuCtxPopCurrent);
    LOAD("cuImportExternalMemory",                 pCuImportExternalMemory);
    LOAD("cuDestroyExternalMemory",                pCuDestroyExternalMemory);
    LOAD("cuExternalMemoryGetMappedMipmappedArray", pCuExternalMemoryGetMappedMipmappedArray);
    LOAD("cuMipmappedArrayGetLevel",               pCuMipmappedArrayGetLevel);
    LOAD("cuMipmappedArrayDestroy",                pCuMipmappedArrayDestroy);
#undef LOAD

    if (pCuInit(0) != CUDA_SUCCESS_LOCAL)
    {
        WARN("cuInit failed; interop disabled.\n");
        FreeLibrary(g_nvcuda_handle);
        g_nvcuda_handle = NULL;
        return FALSE;
    }

    g_cuda_available = TRUE;
    return TRUE;
}

/* Vtable layouts must match DXVK's dxgi_interfaces.h and vkd3d-proton's
 * vkd3d_device_vkd3d_ext.idl. Slots we don't call are void*. */

static const IID IID_IDXGIVkInteropAdapter_local =
    { 0x3a6d8f2c, 0xb0e8, 0x4ab4, { 0xb4, 0xdc, 0x4f, 0xd2, 0x48, 0x91, 0xbf, 0xa5 } };

static const IID IID_IDXGIVkInteropSurface_local =
    { 0x5546cf8c, 0x77e7, 0x4341, { 0xb0, 0x5d, 0x8d, 0x4d, 0x50, 0x00, 0xe7, 0x7d } };

static const IID IID_ID3D12DXVKInteropDevice4_local =
    { 0x7bf3d1e2, 0x9ca5, 0x4f32, { 0x8e, 0x8a, 0xd5, 0xb6, 0xf4, 0xa9, 0x0c, 0x11 } };

/* Defined here to avoid linking Wine's d3d11 and d3d12 stubs. */
static const IID IID_ID3D11Texture2D_local =
    { 0x6f15aaf2, 0xd208, 0x4e89, { 0x9a, 0xb4, 0x48, 0x95, 0x35, 0xd3, 0x4f, 0x9c } };

static const IID IID_ID3D12Resource_local =
    { 0x696442be, 0xa72e, 0x4059, { 0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad } };

typedef struct IDXGIVkInteropAdapter_local IDXGIVkInteropAdapter_local;
typedef struct IDXGIVkInteropAdapterVtbl_local
{
    HRESULT (STDMETHODCALLTYPE *QueryInterface)(IDXGIVkInteropAdapter_local *self, REFIID riid, void **out);
    ULONG   (STDMETHODCALLTYPE *AddRef)(IDXGIVkInteropAdapter_local *self);
    ULONG   (STDMETHODCALLTYPE *Release)(IDXGIVkInteropAdapter_local *self);
    void   *GetVulkanHandles;
    void    (STDMETHODCALLTYPE *GetAdapterUUID)(IDXGIVkInteropAdapter_local *self, char *uuid);
} IDXGIVkInteropAdapterVtbl_local;

struct IDXGIVkInteropAdapter_local { const IDXGIVkInteropAdapterVtbl_local *lpVtbl; };

typedef struct IDXGIVkInteropSurface_local IDXGIVkInteropSurface_local;
typedef struct IDXGIVkInteropSurfaceVtbl_local
{
    HRESULT (STDMETHODCALLTYPE *QueryInterface)(IDXGIVkInteropSurface_local *self, REFIID riid, void **out);
    ULONG   (STDMETHODCALLTYPE *AddRef)(IDXGIVkInteropSurface_local *self);
    ULONG   (STDMETHODCALLTYPE *Release)(IDXGIVkInteropSurface_local *self);
    void   *GetDevice;
    void   *GetVulkanImageInfo;
    HRESULT (STDMETHODCALLTYPE *GetVulkanFdMemoryInfo)(IDXGIVkInteropSurface_local *self, int *fd, size_t *offset, size_t *length);
} IDXGIVkInteropSurfaceVtbl_local;

struct IDXGIVkInteropSurface_local { const IDXGIVkInteropSurfaceVtbl_local *lpVtbl; };

typedef struct ID3D12DXVKInteropDevice4_local ID3D12DXVKInteropDevice4_local;
typedef struct ID3D12DXVKInteropDevice4Vtbl_local
{
    /* IUnknown */
    HRESULT (STDMETHODCALLTYPE *QueryInterface)(ID3D12DXVKInteropDevice4_local *self, REFIID riid, void **out);
    ULONG   (STDMETHODCALLTYPE *AddRef)(ID3D12DXVKInteropDevice4_local *self);
    ULONG   (STDMETHODCALLTYPE *Release)(ID3D12DXVKInteropDevice4_local *self);
    /* ID3D12DXVKInteropDevice */
    void *GetDXGIAdapter;
    void *GetInstanceExtensions;
    void *GetDeviceExtensions;
    void *GetDeviceFeatures;
    void *GetVulkanHandles;
    void *GetVulkanQueueInfo;
    void *GetVulkanImageLayout;
    void *GetVulkanResourceInfo;
    void *LockCommandQueue;
    void *UnlockCommandQueue;
    /* ID3D12DXVKInteropDevice1 */
    void *GetVulkanResourceInfo1;
    void *CreateInteropCommandQueue;
    void *CreateInteropCommandAllocator;
    void *BeginVkCommandBufferInterop;
    void *EndVkCommandBufferInterop;
    /* ID3D12DXVKInteropDevice2 */
    void *LockVulkanQueue;
    void *UnlockVulkanQueue;
    /* ID3D12DXVKInteropDevice3 */
    void *GetVulkanHeapInfo;
    /* ID3D12DXVKInteropDevice4 */
    HRESULT (STDMETHODCALLTYPE *GetAdapterUUID)(ID3D12DXVKInteropDevice4_local *self, BYTE *uuid);
    HRESULT (STDMETHODCALLTYPE *GetVulkanResourceFdMemoryInfo)(ID3D12DXVKInteropDevice4_local *self, ID3D12Resource *resource, int *out_fd, UINT64 *out_offset, UINT64 *out_size);
} ID3D12DXVKInteropDevice4Vtbl_local;

struct ID3D12DXVKInteropDevice4_local { const ID3D12DXVKInteropDevice4Vtbl_local *lpVtbl; };

typedef enum {
    INTEROP_NONE = 0,
    INTEROP_D3D11,
    INTEROP_D3D12,
} interop_kind;

typedef struct encoder_state
{
    struct encoder_state *next;
    void                 *encoder;
    interop_kind          kind;
    IUnknown             *interop_iface; /* IDXGIVkInteropAdapter or ID3D12DXVKInteropDevice4 */
    CUcontext_local       cuda_ctx;      /* Shared; free with release_cuda_ctx() */
} encoder_state;

/* One context per device, released with its last encoder. */
typedef struct cuda_ctx_cache_entry
{
    struct cuda_ctx_cache_entry *next;
    CUdevice_local               device;
    CUcontext_local              ctx;
    unsigned int                 refcount;
} cuda_ctx_cache_entry;

/* CUDA objects to destroy on unregister. The CUarray belongs to the mip and
 * isn't destroyed separately. */
typedef struct registered_resource
{
    struct registered_resource *next;
    void                       *encoder;
    NV_ENC_REGISTERED_PTR       registered;
    CUcontext_local             ctx;
    CUexternalMemory_local      ext_mem;
    CUmipmappedArray_local      mip;
} registered_resource;

static CRITICAL_SECTION      g_state_lock;
static BOOL                  g_state_lock_inited = FALSE;
static encoder_state        *g_states     = NULL;
static cuda_ctx_cache_entry *g_ctx_cache  = NULL;
static registered_resource  *g_registered = NULL;

static void state_lock_init(void)
{
    if (!g_state_lock_inited)
    {
        InitializeCriticalSection(&g_state_lock);
        g_state_lock_inited = TRUE;
    }
}

static void release_cuda_ctx(CUcontext_local ctx);

static encoder_state *state_find_locked(void *encoder)
{
    encoder_state *s;
    for (s = g_states; s; s = s->next)
        if (s->encoder == encoder)
            return s;
    return NULL;
}

static void state_remove(void *encoder)
{
    encoder_state **slink, *s = NULL;
    registered_resource **rlink, *r;
    registered_resource *orphans = NULL;

    EnterCriticalSection(&g_state_lock);
    for (slink = &g_states; (s = *slink); slink = &s->next)
    {
        if (s->encoder == encoder)
        {
            *slink = s->next;
            break;
        }
    }
    rlink = &g_registered;
    while ((r = *rlink))
    {
        if (r->encoder == encoder)
        {
            *rlink = r->next;
            r->next = orphans;
            orphans = r;
        }
        else
        {
            rlink = &r->next;
        }
    }
    LeaveCriticalSection(&g_state_lock);

    /* Free these before release_cuda_ctx() below, while their context is still valid. */
    while (orphans)
    {
        registered_resource *next = orphans->next;
        if (orphans->ctx)
            pCuCtxPushCurrent(orphans->ctx);
        if (orphans->mip)
            pCuMipmappedArrayDestroy(orphans->mip);
        if (orphans->ext_mem)
            pCuDestroyExternalMemory(orphans->ext_mem);
        if (orphans->ctx)
            pCuCtxPopCurrent(NULL);
        free(orphans);
        orphans = next;
    }

    if (s)
    {
        if (s->interop_iface)
            IUnknown_Release(s->interop_iface);
        if (s->cuda_ctx)
            release_cuda_ctx(s->cuda_ctx);
        free(s);
    }
}

/* Caller holds g_state_lock. Takes a reference on the cached context. */
static CUcontext_local cache_get_locked(CUdevice_local device)
{
    cuda_ctx_cache_entry *e;
    for (e = g_ctx_cache; e; e = e->next)
    {
        if (e->device == device)
        {
            e->refcount++;
            return e->ctx;
        }
    }
    return NULL;
}

/* Caller holds g_state_lock. */
static BOOL cache_insert_locked(CUdevice_local device, CUcontext_local ctx)
{
    cuda_ctx_cache_entry *e = calloc(1, sizeof(*e));
    if (!e)
        return FALSE;
    e->device   = device;
    e->ctx      = ctx;
    e->refcount = 1;
    e->next     = g_ctx_cache;
    g_ctx_cache = e;
    return TRUE;
}

/* Use the primary context, not cuCtxCreate: cuImportExternalMemory can reject a
 * cuCtxCreate context with CUDA_ERROR_INVALID_CONTEXT. */
static CUcontext_local create_or_fetch_cuda_ctx_for_uuid(const char *vk_uuid)
{
    int count = 0, i;

    if (pCuDeviceGetCount(&count) != CUDA_SUCCESS_LOCAL || count <= 0)
        return NULL;

    for (i = 0; i < count; i++)
    {
        CUdevice_local dev;
        CUuuid_local cu_uuid;
        CUcontext_local ctx, raced;

        if (pCuDeviceGet(&dev, i) != CUDA_SUCCESS_LOCAL)
            continue;
        if (pCuDeviceGetUuid(&cu_uuid, dev) != CUDA_SUCCESS_LOCAL)
            continue;
        if (memcmp(vk_uuid, cu_uuid.bytes, 16) != 0)
            continue;

        EnterCriticalSection(&g_state_lock);
        ctx = cache_get_locked(dev);
        LeaveCriticalSection(&g_state_lock);
        if (ctx)
            return ctx;

        /* Don't hold g_state_lock across the driver call. */
        if (pCuCtxCreate(&ctx, 0, dev) != CUDA_SUCCESS_LOCAL)
            return NULL;

        EnterCriticalSection(&g_state_lock);
        raced = cache_get_locked(dev);
        if (raced)
        {
            LeaveCriticalSection(&g_state_lock);
            pCuCtxDestroy(ctx);
            return raced;
        }
        if (!cache_insert_locked(dev, ctx))
        {
            LeaveCriticalSection(&g_state_lock);
            pCuCtxDestroy(ctx);
            return NULL;
        }
        LeaveCriticalSection(&g_state_lock);
        return ctx;
    }
    return NULL;
}

static void release_cuda_ctx(CUcontext_local ctx)
{
    cuda_ctx_cache_entry **link, *e;

    if (!ctx)
        return;

    EnterCriticalSection(&g_state_lock);
    for (link = &g_ctx_cache; (e = *link); link = &e->next)
    {
        if (e->ctx == ctx)
        {
            if (--e->refcount == 0)
            {
                *link = e->next;
                LeaveCriticalSection(&g_state_lock);
                if (pCuCtxDestroy)
                    pCuCtxDestroy(e->ctx);
                free(e);
                return;
            }
            LeaveCriticalSection(&g_state_lock);
            return;
        }
    }
    LeaveCriticalSection(&g_state_lock);

    /* Can't release a primary context without its device, which only the cache holds. */
    WARN("release_cuda_ctx: ctx %p not in cache; destroying directly\n", ctx);
    if (pCuCtxDestroy)
        pCuCtxDestroy(ctx);
}

static unsigned int dxgi_format_channel_count(DXGI_FORMAT format)
{
    switch (format)
    {
        case DXGI_FORMAT_R8G8B8A8_UNORM:
        case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
        case DXGI_FORMAT_R8G8B8A8_UINT:
        case DXGI_FORMAT_R8G8B8A8_SNORM:
        case DXGI_FORMAT_R8G8B8A8_SINT:
        case DXGI_FORMAT_B8G8R8A8_UNORM:
        case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
            return 4;
        default:
            return 0;
    }
}

static CUarray_format_local dxgi_format_to_cuda(DXGI_FORMAT format)
{
    switch (format)
    {
        case DXGI_FORMAT_R8G8B8A8_SNORM:
        case DXGI_FORMAT_R8G8B8A8_SINT:
            return CU_AD_FORMAT_SIGNED_INT8_LOCAL;
        default:
            return CU_AD_FORMAT_UNSIGNED_INT8_LOCAL;
    }
}

static NVENCSTATUS register_via_cuda(encoder_state *st,
                                     int fd, uint64_t offset, uint64_t length,
                                     uint32_t width, uint32_t height,
                                     unsigned int channels, CUarray_format_local fmt,
                                     NV_ENC_REGISTER_RESOURCE *params)
{
    CUDA_EXTERNAL_MEMORY_HANDLE_DESC_local mem_desc;
    CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_local mip_desc;
    CUexternalMemory_local ext_mem = NULL;
    CUmipmappedArray_local mip = NULL;
    CUarray_local arr = NULL;
    NVENCSTATUS result;

    pCuCtxPushCurrent(st->cuda_ctx);

    memset(&mem_desc, 0, sizeof(mem_desc));
    mem_desc.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_LOCAL;
    mem_desc.handle.fd = fd;
    mem_desc.size = offset + length;

    if (pCuImportExternalMemory(&ext_mem, &mem_desc) != CUDA_SUCCESS_LOCAL)
    {
        pCuCtxPopCurrent(NULL);
        ERR("cuImportExternalMemory failed for fd=%d size=%llu\n", fd,
            (unsigned long long)(offset + length));
        return NV_ENC_ERR_UNSUPPORTED_PARAM;
    }

    memset(&mip_desc, 0, sizeof(mip_desc));
    mip_desc.offset = offset;
    mip_desc.numLevels = 1;
    mip_desc.arrayDesc.Width = width;
    mip_desc.arrayDesc.Height = height;
    mip_desc.arrayDesc.Depth = 0;
    mip_desc.arrayDesc.NumChannels = channels;
    mip_desc.arrayDesc.Format = fmt;
    mip_desc.arrayDesc.Flags = CUDA_ARRAY3D_SURFACE_LDST_LOCAL;

    if (pCuExternalMemoryGetMappedMipmappedArray(&mip, ext_mem, &mip_desc) != CUDA_SUCCESS_LOCAL)
    {
        pCuDestroyExternalMemory(ext_mem);
        pCuCtxPopCurrent(NULL);
        ERR("cuExternalMemoryGetMappedMipmappedArray failed\n");
        return NV_ENC_ERR_UNSUPPORTED_PARAM;
    }

    if (pCuMipmappedArrayGetLevel(&arr, mip, 0) != CUDA_SUCCESS_LOCAL)
    {
        pCuMipmappedArrayDestroy(mip);
        pCuDestroyExternalMemory(ext_mem);
        pCuCtxPopCurrent(NULL);
        ERR("cuMipmappedArrayGetLevel failed\n");
        return NV_ENC_ERR_UNSUPPORTED_PARAM;
    }

    pCuCtxPopCurrent(NULL);

    params->resourceType = NV_ENC_INPUT_RESOURCE_TYPE_CUDAARRAY;
    params->resourceToRegister = arr;
    params->pitch = width * channels;

    result = origFunctions.nvEncRegisterResource(st->encoder, params);
    if (result != NV_ENC_SUCCESS)
    {
        /* Destroying mip also frees the CUarray. */
        pCuCtxPushCurrent(st->cuda_ctx);
        pCuMipmappedArrayDestroy(mip);
        pCuDestroyExternalMemory(ext_mem);
        pCuCtxPopCurrent(NULL);
        return result;
    }

    {
        registered_resource *r = calloc(1, sizeof(*r));
        if (!r)
        {
            origFunctions.nvEncUnregisterResource(st->encoder, params->registeredResource);
            pCuCtxPushCurrent(st->cuda_ctx);
            pCuMipmappedArrayDestroy(mip);
            pCuDestroyExternalMemory(ext_mem);
            pCuCtxPopCurrent(NULL);
            return NV_ENC_ERR_UNSUPPORTED_PARAM;
        }
        r->encoder    = st->encoder;
        r->registered = params->registeredResource;
        r->ctx        = st->cuda_ctx;
        r->ext_mem    = ext_mem;
        r->mip        = mip;

        EnterCriticalSection(&g_state_lock);
        r->next      = g_registered;
        g_registered = r;
        LeaveCriticalSection(&g_state_lock);
    }
    return result;
}

static NVENCSTATUS open_session_d3d11(NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS *params, void **encoder)
{
    NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS local_params = *params;
    IDXGIVkInteropAdapter_local *adapter = NULL;
    char vk_uuid[16];
    CUcontext_local ctx;
    encoder_state *st;
    NVENCSTATUS status;
    HRESULT hr;

    hr = IUnknown_QueryInterface((IUnknown *)params->device,
                                 &IID_IDXGIVkInteropAdapter_local,
                                 (void **)&adapter);
    if (FAILED(hr) || !adapter)
        return NV_ENC_ERR_UNSUPPORTED_PARAM;

    adapter->lpVtbl->GetAdapterUUID(adapter, vk_uuid);

    ctx = create_or_fetch_cuda_ctx_for_uuid(vk_uuid);
    if (!ctx)
    {
        IUnknown_Release((IUnknown *)adapter);
        ERR("No CUDA device matching DXVK adapter UUID\n");
        return NV_ENC_ERR_UNSUPPORTED_PARAM;
    }

    local_params.deviceType = NV_ENC_DEVICE_TYPE_CUDA;
    local_params.device = ctx;

    status = origFunctions.nvEncOpenEncodeSessionEx(&local_params, encoder);
    if (status != NV_ENC_SUCCESS)
    {
        release_cuda_ctx(ctx);
        IUnknown_Release((IUnknown *)adapter);
        return status;
    }

    st = calloc(1, sizeof(*st));
    if (!st)
    {
        origFunctions.nvEncDestroyEncoder(*encoder);
        release_cuda_ctx(ctx);
        IUnknown_Release((IUnknown *)adapter);
        return NV_ENC_ERR_UNSUPPORTED_PARAM;
    }
    st->encoder = *encoder;
    st->kind = INTEROP_D3D11;
    st->interop_iface = (IUnknown *)adapter;
    st->cuda_ctx = ctx;

    EnterCriticalSection(&g_state_lock);
    st->next = g_states;
    g_states = st;
    LeaveCriticalSection(&g_state_lock);

    return NV_ENC_SUCCESS;
}

static NVENCSTATUS register_resource_d3d11(encoder_state *st, NV_ENC_REGISTER_RESOURCE *params)
{
    ID3D11Texture2D *tex = NULL;
    IDXGIVkInteropSurface_local *surface = NULL;
    D3D11_TEXTURE2D_DESC desc;
    int fd = -1;
    size_t offset = 0, length = 0;
    HRESULT hr;
    NVENCSTATUS rc;

    hr = IUnknown_QueryInterface((IUnknown *)params->resourceToRegister,
                                 &IID_ID3D11Texture2D_local, (void **)&tex);
    if (FAILED(hr) || !tex)
        return NV_ENC_ERR_UNSUPPORTED_PARAM;

    ID3D11Texture2D_GetDesc(tex, &desc);

    hr = IUnknown_QueryInterface((IUnknown *)params->resourceToRegister,
                                 &IID_IDXGIVkInteropSurface_local, (void **)&surface);
    if (FAILED(hr) || !surface)
    {
        ID3D11Texture2D_Release(tex);
        return NV_ENC_ERR_UNSUPPORTED_PARAM;
    }

    hr = surface->lpVtbl->GetVulkanFdMemoryInfo(surface, &fd, &offset, &length);
    if (FAILED(hr) || fd < 0)
    {
        surface->lpVtbl->Release(surface);
        ID3D11Texture2D_Release(tex);
        ERR("GetVulkanFdMemoryInfo failed for D3D11 texture\n");
        return NV_ENC_ERR_UNSUPPORTED_PARAM;
    }

    rc = register_via_cuda(st, fd, offset, length, desc.Width, desc.Height,
                           dxgi_format_channel_count(desc.Format),
                           dxgi_format_to_cuda(desc.Format), params);

    surface->lpVtbl->Release(surface);
    ID3D11Texture2D_Release(tex);
    return rc;
}

static NVENCSTATUS open_session_d3d12(NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS *params, void **encoder)
{
    NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS local_params = *params;
    ID3D12DXVKInteropDevice4_local *iface = NULL;
    BYTE vk_uuid[16];
    CUcontext_local ctx;
    encoder_state *st;
    NVENCSTATUS status;
    HRESULT hr;

    hr = IUnknown_QueryInterface((IUnknown *)params->device,
                                 &IID_ID3D12DXVKInteropDevice4_local,
                                 (void **)&iface);
    if (FAILED(hr) || !iface)
        return NV_ENC_ERR_UNSUPPORTED_PARAM;

    if (FAILED(iface->lpVtbl->GetAdapterUUID(iface, vk_uuid)))
    {
        iface->lpVtbl->Release(iface);
        return NV_ENC_ERR_UNSUPPORTED_PARAM;
    }

    ctx = create_or_fetch_cuda_ctx_for_uuid((const char *)vk_uuid);
    if (!ctx)
    {
        iface->lpVtbl->Release(iface);
        ERR("No CUDA device matching vkd3d-proton adapter UUID\n");
        return NV_ENC_ERR_UNSUPPORTED_PARAM;
    }

    local_params.deviceType = NV_ENC_DEVICE_TYPE_CUDA;
    local_params.device = ctx;

    status = origFunctions.nvEncOpenEncodeSessionEx(&local_params, encoder);
    if (status != NV_ENC_SUCCESS)
    {
        release_cuda_ctx(ctx);
        iface->lpVtbl->Release(iface);
        return status;
    }

    st = calloc(1, sizeof(*st));
    if (!st)
    {
        origFunctions.nvEncDestroyEncoder(*encoder);
        release_cuda_ctx(ctx);
        iface->lpVtbl->Release(iface);
        return NV_ENC_ERR_UNSUPPORTED_PARAM;
    }
    st->encoder = *encoder;
    st->kind = INTEROP_D3D12;
    st->interop_iface = (IUnknown *)iface;
    st->cuda_ctx = ctx;

    EnterCriticalSection(&g_state_lock);
    st->next = g_states;
    g_states = st;
    LeaveCriticalSection(&g_state_lock);

    return NV_ENC_SUCCESS;
}

static NVENCSTATUS register_resource_d3d12(encoder_state *st, NV_ENC_REGISTER_RESOURCE *params)
{
    ID3D12DXVKInteropDevice4_local *iface = (ID3D12DXVKInteropDevice4_local *)st->interop_iface;
    ID3D12Resource *res = NULL;
    D3D12_RESOURCE_DESC desc;
    int fd = -1;
    UINT64 offset = 0, length = 0;
    HRESULT hr;
    NVENCSTATUS rc;
    unsigned int channels;
    CUarray_format_local fmt;

    hr = IUnknown_QueryInterface((IUnknown *)params->resourceToRegister,
                                 &IID_ID3D12Resource_local, (void **)&res);
    if (FAILED(hr) || !res)
        return NV_ENC_ERR_UNSUPPORTED_PARAM;

    desc = ID3D12Resource_GetDesc(res);

    hr = iface->lpVtbl->GetVulkanResourceFdMemoryInfo(iface, res, &fd, &offset, &length);
    if (FAILED(hr) || fd < 0)
    {
        ID3D12Resource_Release(res);
        ERR("GetVulkanResourceFdMemoryInfo failed for D3D12 resource\n");
        return NV_ENC_ERR_UNSUPPORTED_PARAM;
    }

    channels = dxgi_format_channel_count(desc.Format);
    fmt = dxgi_format_to_cuda(desc.Format);

    rc = register_via_cuda(st, fd, offset, length, (uint32_t)desc.Width, desc.Height,
                           channels, fmt, params);

    ID3D12Resource_Release(res);
    return rc;
}

static NVENCSTATUS WINAPI NvEncOpenEncodeSession(void *device, uint32_t deviceType, void **encoder)
{
    TRACE("(%p, %u, %p)\n", device, deviceType, encoder);
    return origFunctions.nvEncOpenEncodeSession(device, deviceType, encoder);
}

static NVENCSTATUS WINAPI NvEncGetEncodeGUIDCount(void *encoder, uint32_t *encodeGUIDCount)
{
    TRACE("(%p, %p)\n", encoder, encodeGUIDCount);
    return origFunctions.nvEncGetEncodeGUIDCount(encoder, encodeGUIDCount);
}

static NVENCSTATUS WINAPI NvEncGetEncodeGUIDs(void *encoder, GUID *GUIDs, uint32_t guidArraySize, uint32_t *GUIDCount)
{
    TRACE("(%p, %p, %u, %p)\n", encoder, GUIDs, guidArraySize, GUIDCount);
    return origFunctions.nvEncGetEncodeGUIDs(encoder, GUIDs, guidArraySize, GUIDCount);
}

static NVENCSTATUS WINAPI NvEncGetEncodeProfileGUIDCount(void *encoder, GUID encodeGUID, uint32_t *encodeProfileGUIDCount)
{
    TRACE("(%p, %s, %p)\n", encoder, debugstr_guid(&encodeGUID), encodeProfileGUIDCount);
    return origFunctions.nvEncGetEncodeProfileGUIDCount(encoder, encodeGUID, encodeProfileGUIDCount);
}

static NVENCSTATUS WINAPI NvEncGetEncodeProfileGUIDs(void *encoder, GUID encodeGUID, GUID *profileGUIDs,
                                                     uint32_t guidArraySize, uint32_t *GUIDCount)
{
    TRACE("(%p, %s, %p, %u, %p)\n", encoder, debugstr_guid(&encodeGUID), profileGUIDs, guidArraySize, GUIDCount);
    return origFunctions.nvEncGetEncodeProfileGUIDs(encoder, encodeGUID, profileGUIDs, guidArraySize, GUIDCount);
}

static NVENCSTATUS WINAPI NvEncGetInputFormatCount(void *encoder, GUID encodeGUID, uint32_t *inputFmtCount)
{
    TRACE("(%p, %s, %p)\n", encoder, debugstr_guid(&encodeGUID), inputFmtCount);
    return origFunctions.nvEncGetInputFormatCount(encoder, encodeGUID, inputFmtCount);
}

static NVENCSTATUS WINAPI NvEncGetInputFormats(void *encoder, GUID encodeGUID, NV_ENC_BUFFER_FORMAT *inputFmts,
                                               uint32_t inputFmtArraySize, uint32_t *inputFmtCount)
{
    TRACE("(%p, %s, %p)\n", encoder, debugstr_guid(&encodeGUID), inputFmtCount);
    return origFunctions.nvEncGetInputFormatCount(encoder, encodeGUID, inputFmtCount);
}

static NVENCSTATUS WINAPI NvEncGetEncodeCaps(void *encoder, GUID encodeGUID, NV_ENC_CAPS_PARAM *capsParam, int *capsVal)
{
    TRACE("(%p, %s, %p, %p)\n", encoder, debugstr_guid(&encodeGUID), capsParam, capsVal);
    return origFunctions.nvEncGetEncodeCaps(encoder, encodeGUID, capsParam, capsVal);
}

static NVENCSTATUS WINAPI NvEncGetEncodePresetCount(void *encoder, GUID encodeGUID, uint32_t *encodePresetGUIDCount)
{
    TRACE("(%p, %s, %p)\n", encoder, debugstr_guid(&encodeGUID), encodePresetGUIDCount);
    return origFunctions.nvEncGetEncodePresetCount(encoder, encodeGUID, encodePresetGUIDCount);
}

static NVENCSTATUS WINAPI NvEncGetEncodePresetGUIDs(void *encoder, GUID encodeGUID, GUID *presetGUIDs, uint32_t guidArraySize,
                                                    uint32_t *encodePresetGUIDCount)
{
    TRACE("(%p, %s, %p, %u, %p)\n", encoder, debugstr_guid(&encodeGUID), presetGUIDs, guidArraySize, encodePresetGUIDCount);
    return origFunctions.nvEncGetEncodePresetGUIDs(encoder, encodeGUID, presetGUIDs, guidArraySize, encodePresetGUIDCount);
}

static NVENCSTATUS WINAPI NvEncGetEncodePresetConfig(void *encoder, GUID encodeGUID, GUID presetGUID, NV_ENC_PRESET_CONFIG *presetConfig)
{
    TRACE("(%p, %s, %s, %p)\n", encoder, debugstr_guid(&encodeGUID), debugstr_guid(&presetGUID), presetConfig);
    return origFunctions.nvEncGetEncodePresetConfig(encoder, encodeGUID, presetGUID, presetConfig);
}

static NVENCSTATUS WINAPI NvEncInitializeEncoder(void *encoder, NV_ENC_INITIALIZE_PARAMS *createEncodeParams)
{
    NV_ENC_INITIALIZE_PARAMS linux_encode_params;

    TRACE("(%p, %p)\n", encoder, createEncodeParams);

    if (!createEncodeParams)
        return NV_ENC_ERR_INVALID_PTR;

    if (createEncodeParams->enableEncodeAsync)
    {
        FIXME("Async encoding is not supported by the linux NVIDIA driver.\n");
        FIXME("Trying to emulate async mode, but this might not work for all applications.\n");

        /* Forward modified information to the linux library. */
        linux_encode_params = *createEncodeParams;
        createEncodeParams = &linux_encode_params;
        linux_encode_params.enableEncodeAsync = 0;
    }

    return origFunctions.nvEncInitializeEncoder(encoder, createEncodeParams);
}

static NVENCSTATUS WINAPI NvEncCreateInputBuffer(void *encoder, NV_ENC_CREATE_INPUT_BUFFER *createInputBufferParams)
{
    TRACE("(%p, %p)\n", encoder, createInputBufferParams);
    return origFunctions.nvEncCreateInputBuffer(encoder, createInputBufferParams);
}

static NVENCSTATUS WINAPI NvEncDestroyInputBuffer(void *encoder, NV_ENC_INPUT_PTR inputBuffer)
{
    TRACE("(%p, %p)\n", encoder, inputBuffer);
    return origFunctions.nvEncDestroyInputBuffer(encoder, inputBuffer);
}

static NVENCSTATUS WINAPI NvEncCreateBitstreamBuffer(void *encoder, NV_ENC_CREATE_BITSTREAM_BUFFER *createBitstreamBufferParams)
{
    TRACE("(%p, %p)\n", encoder, createBitstreamBufferParams);
    return origFunctions.nvEncCreateBitstreamBuffer(encoder, createBitstreamBufferParams);
}

static NVENCSTATUS WINAPI NvEncDestroyBitstreamBuffer(void *encoder, NV_ENC_OUTPUT_PTR bitstreamBuffer)
{
    TRACE("(%p, %p)\n", encoder, bitstreamBuffer);
    return origFunctions.nvEncDestroyBitstreamBuffer(encoder, bitstreamBuffer);
}

static NVENCSTATUS WINAPI NvEncEncodePicture(void *encoder, NV_ENC_PIC_PARAMS *encodePicParams)
{
    NVENCSTATUS result;
    TRACE("(%p, %p)\n", encoder, encodePicParams);

    result = origFunctions.nvEncEncodePicture(encoder, encodePicParams);

    if (encodePicParams->completionEvent)
        SetEvent(encodePicParams->completionEvent);

    return result;
}

static NVENCSTATUS WINAPI NvEncLockBitstream(void *encoder, NV_ENC_LOCK_BITSTREAM *lockBitstreamBufferParams)
{
    TRACE("(%p, %p)\n", encoder, lockBitstreamBufferParams);
    return origFunctions.nvEncLockBitstream(encoder, lockBitstreamBufferParams);
}

static NVENCSTATUS WINAPI NvEncUnlockBitstream(void *encoder, NV_ENC_OUTPUT_PTR bitstreamBuffer)
{
    TRACE("(%p, %p)\n", encoder, bitstreamBuffer);
    return origFunctions.nvEncUnlockBitstream(encoder, bitstreamBuffer);
}

static NVENCSTATUS WINAPI NvEncLockInputBuffer(void *encoder, NV_ENC_LOCK_INPUT_BUFFER *lockInputBufferParams)
{
    TRACE("(%p, %p)\n", encoder, lockInputBufferParams);
    return origFunctions.nvEncLockInputBuffer(encoder, lockInputBufferParams);
}

static NVENCSTATUS WINAPI NvEncUnlockInputBuffer(void *encoder, NV_ENC_INPUT_PTR inputBuffer)
{
    TRACE("(%p, %p)\n", encoder, inputBuffer);
    return origFunctions.nvEncUnlockInputBuffer(encoder, inputBuffer);
}

static NVENCSTATUS WINAPI NvEncGetEncodeStats(void *encoder, NV_ENC_STAT *encodeStats)
{
    TRACE("(%p, %p)\n", encoder, encodeStats);
    return origFunctions.nvEncGetEncodeStats(encoder, encodeStats);
}

static NVENCSTATUS WINAPI NvEncGetSequenceParams(void *encoder, NV_ENC_SEQUENCE_PARAM_PAYLOAD *sequenceParamPayload)
{
    TRACE("(%p, %p)\n", encoder, sequenceParamPayload);
    return origFunctions.nvEncGetSequenceParams(encoder, sequenceParamPayload);
}

static NVENCSTATUS WINAPI NvEncRegisterAsyncEvent(void *encoder, NV_ENC_EVENT_PARAMS *eventParams)
{
    TRACE("(%p, %p)\n", encoder, eventParams);
    /* This function will always fail as the linux NVIDIA driver doesn't support async mode */
    /* return origFunctions.nvEncRegisterAsyncEvent(encoder, eventParams); */
    return NV_ENC_SUCCESS;
}

static NVENCSTATUS WINAPI NvEncUnregisterAsyncEvent(void *encoder, NV_ENC_EVENT_PARAMS *eventParams)
{
    TRACE("(%p, %p)\n", encoder, eventParams);
    /* This function will always fail as the linux NVIDIA driver doesn't support async mode */
    /* return origFunctions.nvEncUnregisterAsyncEvent(encoder, eventParams); */
    return NV_ENC_SUCCESS;
}

static NVENCSTATUS WINAPI NvEncMapInputResource(void *encoder, NV_ENC_MAP_INPUT_RESOURCE *mapInputResParams)
{
    TRACE("(%p, %p)\n", encoder, mapInputResParams);
    return origFunctions.nvEncMapInputResource(encoder, mapInputResParams);
}

static NVENCSTATUS WINAPI NvEncUnmapInputResource(void *encoder, NV_ENC_INPUT_PTR mappedInputBuffer)
{
    TRACE("(%p, %p)\n", encoder, mappedInputBuffer);
    return origFunctions.nvEncUnmapInputResource(encoder, mappedInputBuffer);
}

static NVENCSTATUS WINAPI NvEncDestroyEncoder(void *encoder)
{
    NVENCSTATUS result;
    TRACE("(%p)\n", encoder);

    result = origFunctions.nvEncDestroyEncoder(encoder);
    state_remove(encoder);
    return result;
}

static NVENCSTATUS WINAPI NvEncInvalidateRefFrames(void *encoder, uint64_t invalidRefFrameTimeStamp)
{
    TRACE("(%p, %s)\n", encoder, wine_dbgstr_longlong(invalidRefFrameTimeStamp));
    return origFunctions.nvEncInvalidateRefFrames(encoder, invalidRefFrameTimeStamp);
}

static NVENCSTATUS WINAPI NvEncOpenEncodeSessionEx(NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS *openSessionExParams, void **encoder)
{
    TRACE("(%p, %p)\n", openSessionExParams, encoder);

    if (!openSessionExParams || !encoder)
        return NV_ENC_ERR_INVALID_PTR;

    if (openSessionExParams->deviceType == NV_ENC_DEVICE_TYPE_DIRECTX &&
        openSessionExParams->device && load_cuda())
    {
        ID3D12DXVKInteropDevice4_local *d12 = NULL;
        IDXGIVkInteropAdapter_local    *d11 = NULL;

        if (SUCCEEDED(IUnknown_QueryInterface((IUnknown *)openSessionExParams->device,
                                              &IID_ID3D12DXVKInteropDevice4_local,
                                              (void **)&d12)))
        {
            d12->lpVtbl->Release(d12);
            return open_session_d3d12(openSessionExParams, encoder);
        }
        if (SUCCEEDED(IUnknown_QueryInterface((IUnknown *)openSessionExParams->device,
                                              &IID_IDXGIVkInteropAdapter_local,
                                              (void **)&d11)))
        {
            d11->lpVtbl->Release(d11);
            return open_session_d3d11(openSessionExParams, encoder);
        }
        WARN("DIRECTX device not recognised by DXVK/vkd3d-proton interop; passing through.\n");
    }

    return origFunctions.nvEncOpenEncodeSessionEx(openSessionExParams, encoder);
}

static NVENCSTATUS WINAPI NvEncRegisterResource(void *encoder, NV_ENC_REGISTER_RESOURCE *registerResParams)
{
    encoder_state *st;
    NVENCSTATUS rc;

    TRACE("(%p, %p)\n", encoder, registerResParams);

    if (!registerResParams)
        return NV_ENC_ERR_INVALID_PTR;

    EnterCriticalSection(&g_state_lock);
    st = state_find_locked(encoder);
    LeaveCriticalSection(&g_state_lock);

    if (st && registerResParams->resourceType == NV_ENC_INPUT_RESOURCE_TYPE_DIRECTX)
    {
        NV_ENC_REGISTER_RESOURCE local_params = *registerResParams;
        if (st->kind == INTEROP_D3D12)
            rc = register_resource_d3d12(st, &local_params);
        else
            rc = register_resource_d3d11(st, &local_params);

        if (rc == NV_ENC_SUCCESS)
            registerResParams->registeredResource = local_params.registeredResource;
        return rc;
    }

    return origFunctions.nvEncRegisterResource(encoder, registerResParams);
}

static NVENCSTATUS WINAPI NvEncUnregisterResource(void *encoder, NV_ENC_REGISTERED_PTR registeredResource)
{
    registered_resource **link, *r = NULL;
    NVENCSTATUS status;

    TRACE("(%p, %p)\n", encoder, registeredResource);

    EnterCriticalSection(&g_state_lock);
    for (link = &g_registered; (r = *link); link = &r->next)
    {
        if (r->registered == registeredResource && r->encoder == encoder)
        {
            *link = r->next;
            break;
        }
    }
    LeaveCriticalSection(&g_state_lock);

    status = origFunctions.nvEncUnregisterResource(encoder, registeredResource);

    if (r)
    {
        if (r->ctx)
            pCuCtxPushCurrent(r->ctx);
        if (r->mip)
            pCuMipmappedArrayDestroy(r->mip);
        if (r->ext_mem)
            pCuDestroyExternalMemory(r->ext_mem);
        if (r->ctx)
            pCuCtxPopCurrent(NULL);
        free(r);
    }
    return status;
}

static NVENCSTATUS WINAPI NvEncReconfigureEncoder(void *encoder, NV_ENC_RECONFIGURE_PARAMS *reInitEncodeParams)
{
    TRACE("(%p, %p)\n", encoder, reInitEncodeParams);
    return origFunctions.nvEncReconfigureEncoder(encoder, reInitEncodeParams);
}

static NVENCSTATUS WINAPI NvEncCreateMVBuffer(void *encoder, NV_ENC_CREATE_MV_BUFFER *createMVBufferParams)
{
    TRACE("(%p, %p)\n", encoder, createMVBufferParams);
    return origFunctions.nvEncCreateMVBuffer(encoder, createMVBufferParams);
}

static NVENCSTATUS WINAPI NvEncDestroyMVBuffer(void *encoder, NV_ENC_OUTPUT_PTR MVBuffer)
{
    TRACE("(%p, %p)\n", encoder, MVBuffer);
    return origFunctions.nvEncDestroyMVBuffer(encoder, MVBuffer);
}

static NVENCSTATUS WINAPI NvEncRunMotionEstimationOnly(void *encoder, NV_ENC_MEONLY_PARAMS *MEOnlyParams)
{
    TRACE("(%p, %p)\n", encoder, MEOnlyParams);
    return origFunctions.nvEncRunMotionEstimationOnly(encoder, MEOnlyParams);
}

static const char *WINAPI NvEncGetLastErrorString(void* encoder)
{
    TRACE("(%p)\n", encoder);
    return origFunctions.nvEncGetLastErrorString(encoder);
}

static NVENCSTATUS WINAPI NvEncSetIOCudaStreams(void* encoder, NV_ENC_CUSTREAM_PTR inputStream, NV_ENC_CUSTREAM_PTR outputStream)
{
    TRACE("(%p, %p, %p)\n", encoder, inputStream, outputStream);
    return origFunctions.nvEncSetIOCudaStreams(encoder, inputStream, outputStream);
}

static NVENCSTATUS WINAPI NvEncGetEncodePresetConfigEx(void* encoder, GUID encodeGUID, GUID presetGUID, NV_ENC_TUNING_INFO tuningInfo, NV_ENC_PRESET_CONFIG *presetConfig)
{
    TRACE("(%p, %s, %s, %u, %p)\n", encoder, debugstr_guid(&encodeGUID), debugstr_guid(&presetGUID), tuningInfo, presetConfig);
    return origFunctions.nvEncGetEncodePresetConfigEx(encoder, encodeGUID, presetGUID, tuningInfo, presetConfig);
}

static NVENCSTATUS WINAPI NvEncGetSequenceParamEx(void* encoder, NV_ENC_INITIALIZE_PARAMS* encInitParams, NV_ENC_SEQUENCE_PARAM_PAYLOAD *sequenceParamPayload)
{
    TRACE("(%p, %p, %p)\n", encoder, encInitParams, sequenceParamPayload);
    return origFunctions.nvEncGetSequenceParamEx(encoder, encInitParams, sequenceParamPayload);
}

static NVENCSTATUS WINAPI NvEncRestoreEncoderState(void* encoder, NV_ENC_RESTORE_ENCODER_STATE_PARAMS *restoreState)
{
    TRACE("(%p, %p)\n", encoder, restoreState);
    return origFunctions.nvEncRestoreEncoderState(encoder, restoreState);
}

static NVENCSTATUS WINAPI NvEncLookaheadPicture(void* encoder, NV_ENC_LOOKAHEAD_PIC_PARAMS *lookaheadParams)
{
    TRACE("(%p, %p)\n", encoder, lookaheadParams);
    return origFunctions.nvEncLookaheadPicture(encoder, lookaheadParams);
}

NVENCSTATUS WINAPI NvEncodeAPIGetMaxSupportedVersion(uint32_t* version)
{
    TRACE("(%p)\n", version);
    return pNvEncodeAPIGetMaxSupportedVersion(version);
}

NVENCSTATUS WINAPI NvEncodeAPICreateInstance(NV_ENCODE_API_FUNCTION_LIST *functionList)
{
    NVENCSTATUS status;

    TRACE("(%p)\n", functionList);

    if (!functionList)
        return NV_ENC_ERR_INVALID_PTR;

    memset(&origFunctions, 0, sizeof(origFunctions));
    origFunctions.version = functionList->version;
    status = pNvEncodeAPICreateInstance(&origFunctions);
    if (status)
    {
        FIXME("Failed to create native encoder for version %x\n", functionList->version);
        return status;
    }

    /* set all function points and reserved values to zero */
    memset(functionList, 0, sizeof(*functionList));
    functionList->version = origFunctions.version;

    #define SET_FUNCPTR(f) if (origFunctions.nv##f) functionList->nv##f = &Nv##f

    SET_FUNCPTR(EncOpenEncodeSession);
    SET_FUNCPTR(EncGetEncodeGUIDCount);
    SET_FUNCPTR(EncGetEncodeProfileGUIDCount);
    SET_FUNCPTR(EncGetEncodeProfileGUIDs);
    SET_FUNCPTR(EncGetEncodeGUIDs);
    SET_FUNCPTR(EncGetInputFormatCount);
    SET_FUNCPTR(EncGetInputFormats);
    SET_FUNCPTR(EncGetEncodeCaps);
    SET_FUNCPTR(EncGetEncodePresetCount);
    SET_FUNCPTR(EncGetEncodePresetGUIDs);
    SET_FUNCPTR(EncGetEncodePresetConfig);
    SET_FUNCPTR(EncInitializeEncoder);
    SET_FUNCPTR(EncCreateInputBuffer);
    SET_FUNCPTR(EncDestroyInputBuffer);
    SET_FUNCPTR(EncCreateBitstreamBuffer);
    SET_FUNCPTR(EncDestroyBitstreamBuffer);
    SET_FUNCPTR(EncEncodePicture);
    SET_FUNCPTR(EncLockBitstream);
    SET_FUNCPTR(EncUnlockBitstream);
    SET_FUNCPTR(EncLockInputBuffer);
    SET_FUNCPTR(EncUnlockInputBuffer);
    SET_FUNCPTR(EncGetEncodeStats);
    SET_FUNCPTR(EncGetSequenceParams);
    SET_FUNCPTR(EncRegisterAsyncEvent);
    SET_FUNCPTR(EncUnregisterAsyncEvent);
    SET_FUNCPTR(EncMapInputResource);
    SET_FUNCPTR(EncUnmapInputResource);
    SET_FUNCPTR(EncDestroyEncoder);
    SET_FUNCPTR(EncInvalidateRefFrames);
    SET_FUNCPTR(EncOpenEncodeSessionEx);
    SET_FUNCPTR(EncRegisterResource);
    SET_FUNCPTR(EncUnregisterResource);
    SET_FUNCPTR(EncReconfigureEncoder);
    SET_FUNCPTR(EncCreateMVBuffer);             /* available since 6.0 */
    SET_FUNCPTR(EncDestroyMVBuffer);            /* available since 6.0 */
    SET_FUNCPTR(EncRunMotionEstimationOnly);    /* available since 6.0 */
    SET_FUNCPTR(EncGetLastErrorString);         /* 10.0 */
    SET_FUNCPTR(EncSetIOCudaStreams);           /* 10.0 */
    SET_FUNCPTR(EncGetEncodePresetConfigEx);    /* 10.0 */
    SET_FUNCPTR(EncGetSequenceParamEx);         /* 10.0 */
    SET_FUNCPTR(EncRestoreEncoderState);        /* 12.0 */
    SET_FUNCPTR(EncLookaheadPicture);           /* 12.0 */

    #undef SET_FUNCPTR

    return NV_ENC_SUCCESS;
}

static BOOL load_nvencode(void)
{
    libnvidia_encode_handle = dlopen("libnvidia-encode.so.1", RTLD_NOW);

    if (!libnvidia_encode_handle)
    {
        FIXME("Wine cannot find the libnvidia-encode.so.1 library, NVIDIA video encoding support disabled.\n");
        return FALSE;
    }

    pNvEncodeAPICreateInstance = dlsym(libnvidia_encode_handle, "NvEncodeAPICreateInstance");
    if (!pNvEncodeAPICreateInstance)
    {
        FIXME("Can't find symbol NvEncodeAPICreateInstance.\n");
        return FALSE;
    }

    pNvEncodeAPIGetMaxSupportedVersion = dlsym(libnvidia_encode_handle, "NvEncodeAPIGetMaxSupportedVersion");
    if (!pNvEncodeAPIGetMaxSupportedVersion)
    {
        FIXME("Can't find symbol NvEncodeAPIGetMaxSupportedVersion.\n");
        return FALSE;
    }

    return TRUE;
}

BOOL WINAPI DllMain(HINSTANCE instance, DWORD reason, LPVOID reserved)
{
    TRACE("(%p, %u, %p)\n", instance, reason, reserved);

    switch (reason)
    {
        case DLL_PROCESS_ATTACH:
            DisableThreadLibraryCalls(instance);
            state_lock_init();
            if (!load_nvencode()) return FALSE;
            break;
        case DLL_PROCESS_DETACH:
            if (reserved) break;
            while (g_states)
                state_remove(g_states->encoder);
            if (g_nvcuda_handle)
            {
                FreeLibrary(g_nvcuda_handle);
                g_nvcuda_handle = NULL;
            }
            if (libnvidia_encode_handle)
                dlclose(libnvidia_encode_handle);
            if (g_state_lock_inited)
            {
                DeleteCriticalSection(&g_state_lock);
                g_state_lock_inited = FALSE;
            }
            break;
    }

    return TRUE;
}
