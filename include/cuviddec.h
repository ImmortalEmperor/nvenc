/*
 * Copyright (C) 2015 Michael Müller
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

#ifndef __WINE_CUVIDDEC_H
#define __WINE_CUVIDDEC_H

#include "cuda.h"

typedef void *CUvideodecoder;
typedef void *CUvideoctxlock;
typedef long long CUvideotimestamp;

/* the following structures are documented but we don't need to know the content */
typedef struct _CUVIDPROCPARAMS CUVIDPROCPARAMS;
typedef struct _CUVIDDECODECAPS CUVIDDECODECAPS;
typedef struct _CUVIDGETDECODESTATUS CUVIDGETDECODESTATUS;
typedef struct _CUVIDDECODESTATSHEADER CUVIDDECODESTATSHEADER;
typedef struct _CUVIDDECODESTATS CUVIDDECODESTATS;
typedef struct _CUVIDREGISTERDECODESURFACESINFO CUVIDREGISTERDECODESURFACESINFO;
typedef struct _CUVIDPROCPARAMSEXT CUVIDPROCPARAMSEXT;

typedef enum cuvidMaxNumRegisterDecodeSurfaces_enum {
    MAX_NUM_REGISTERED_DECODE_SURFACES = 32
} cuvidMaxNumRegisterDecodeSurfaces;

typedef enum cudaVideoCodec_enum
{
    cudaVideoCodec_MPEG1=0,
    cudaVideoCodec_MPEG2,
    cudaVideoCodec_MPEG4,
    cudaVideoCodec_VC1,
    cudaVideoCodec_H264,
    cudaVideoCodec_JPEG,
    cudaVideoCodec_H264_SVC,
    cudaVideoCodec_H264_MVC,
    cudaVideoCodec_HEVC,
    cudaVideoCodec_VP8,
    cudaVideoCodec_VP9,
    cudaVideoCodec_AV1,
    cudaVideoCodec_NumCodecs,
    cudaVideoCodec_YUV420 = (('I'<<24)|('Y'<<16)|('U'<<8)|('V')),
    cudaVideoCodec_YV12   = (('Y'<<24)|('V'<<16)|('1'<<8)|('2')),
    cudaVideoCodec_NV12   = (('N'<<24)|('V'<<16)|('1'<<8)|('2')),
    cudaVideoCodec_YUYV   = (('Y'<<24)|('U'<<16)|('Y'<<8)|('V')),
    cudaVideoCodec_UYVY   = (('U'<<24)|('Y'<<16)|('V'<<8)|('Y'))
} cudaVideoCodec;

typedef enum cudaVideoChromaFormat_enum
{
    cudaVideoChromaFormat_Monochrome=0,
    cudaVideoChromaFormat_420,
    cudaVideoChromaFormat_422,
    cudaVideoChromaFormat_444
} cudaVideoChromaFormat;

typedef enum cudaVideoSurfaceFormat_enum
{
    cudaVideoSurfaceFormat_NV12=0,
    cudaVideoSurfaceFormat_P016=1,
    cudaVideoSurfaceFormat_YUV444=2,
    cudaVideoSurfaceFormat_YUV444_16Bit=3,
    cudaVideoSurfaceFormat_NV16=4,
    cudaVideoSurfaceFormat_P216=5,
    cudaVideoSurfaceFormat_NV12_Opaque=6,
    cudaVideoSurfaceFormat_P016_Opaque=7,
    cudaVideoSurfaceFormat_YUV444_Opaque=8,
    cudaVideoSurfaceFormat_YUV444_16Bit_Opaque=9,
    cudaVideoSurfaceFormat_NV16_Opaque=10,
    cudaVideoSurfaceFormat_P216_Opaque=11
} cudaVideoSurfaceFormat;

typedef enum cudaVideoDeinterlaceMode_enum
{
    cudaVideoDeinterlaceMode_Weave=0,
    cudaVideoDeinterlaceMode_Bob,
    cudaVideoDeinterlaceMode_Adaptive
} cudaVideoDeinterlaceMode;

typedef enum cudaVideoCreateFlags_enum
{
    cudaVideoCreate_Default     = 0x00,
    cudaVideoCreate_PreferCUDA  = 0x01,
    cudaVideoCreate_PreferDXVA  = 0x02,
    cudaVideoCreate_PreferCUVID = 0x04
} cudaVideoCreateFlags;

typedef enum cuvidDecodeStatus_enum
{
    cuvidDecodeStatus_Invalid         = 0,
    cuvidDecodeStatus_InProgress      = 1,
    cuvidDecodeStatus_Success         = 2,
    cuvidDecodeStatus_Error           = 8,
    cuvidDecodeStatus_Error_Concealed = 9,
} cuvidDecodeStatus;

typedef enum cuvidDecodeFeature_enum
{
    cuvidDecodeFeature_DecStats         = 0,
} cuvidDecodeFeature;

typedef struct _LINUX_CUVIDDECODECREATEINFO
{
    unsigned long ulWidth;
    unsigned long ulHeight;
    unsigned long ulNumDecodeSurfaces;
    cudaVideoCodec CodecType;
    cudaVideoChromaFormat ChromaFormat;
    unsigned long ulCreationFlags;
    unsigned long bitDepthMinus8;
    unsigned long ulIntraDecodeOnly;
    unsigned long ulMaxWidth;
    unsigned long ulMaxHeight;
    unsigned long Reserved1;
    struct
    {
        short left;
        short top;
        short right;
        short bottom;
    } display_area;
    cudaVideoSurfaceFormat OutputFormat;
    cudaVideoDeinterlaceMode DeinterlaceMode;
    unsigned long ulTargetWidth;
    unsigned long ulTargetHeight;
    unsigned long ulNumOutputSurfaces;
    CUvideoctxlock vidLock;
    struct
    {
        short left;
        short top;
        short right;
        short bottom;
    } target_rect;
    unsigned long enableHistogram;
    unsigned long enableDecodeFeatures;
    unsigned long Reserved2[3];
} LINUX_CUVIDDECODECREATEINFO;

typedef struct _CUVIDDECODECREATEINFO
{
    ULONG ulWidth;
    ULONG ulHeight;
    ULONG ulNumDecodeSurfaces;
    cudaVideoCodec CodecType;
    cudaVideoChromaFormat ChromaFormat;
    ULONG ulCreationFlags;
    ULONG bitDepthMinus8;
    ULONG ulIntraDecodeOnly;
    ULONG ulMaxWidth;
    ULONG ulMaxHeight;
    ULONG Reserved1;
    struct
    {
        short left;
        short top;
        short right;
        short bottom;
    } display_area;
    cudaVideoSurfaceFormat OutputFormat;
    cudaVideoDeinterlaceMode DeinterlaceMode;
    ULONG ulTargetWidth;
    ULONG ulTargetHeight;
    ULONG ulNumOutputSurfaces;
    CUvideoctxlock vidLock;
    struct
    {
        short left;
        short top;
        short right;
        short bottom;
    } target_rect;
    ULONG enableHistogram;
    ULONG enableDecodeFeatures;
    ULONG Reserved2[3];
} CUVIDDECODECREATEINFO;

#endif /* __WINE_CUVIDDEC_H */
