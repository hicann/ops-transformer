/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file vf_dequant_quant_compressor.h
 * \brief HiFP8 matmul结果反量化：result = input * descale
 */

#ifndef VF_DEQUANT_QUANT_COMPRESSOR_H
#define VF_DEQUANT_QUANT_COMPRESSOR_H

#include "kernel_operator.h"
#include "../quant_compressor_comm.h"
using namespace AscendC;
using namespace QuantCompressor;

// ================================ Coff=1 ===================================
// total = col, descale = col（register一一对应）

__simd_vf__ void DequantVfCoff1BaseImpl(__ubuf__ float *outputAddr, __ubuf__ float *inputAddr,
                                        __ubuf__ float *descaleAddr, uint32_t row, uint32_t col, uint32_t actualCol)
{
    MicroAPI::RegTensor<float> vregInput;
    MicroAPI::RegTensor<float> vregDescale;
    MicroAPI::RegTensor<float> vregMul;

    MicroAPI::MaskReg mask = MicroAPI::UpdateMask<float>(col);
    MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(vregDescale, descaleAddr);

    for (uint16_t i = 0; i < row; i++) {
        uint64_t offset = i * actualCol;
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(vregInput, inputAddr + offset);
        MicroAPI::Mul(vregMul, vregInput, vregDescale, mask);
        MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM>(outputAddr + offset, vregMul, mask);
    }
}

__simd_vf__ void DequantVfCoff1D128Impl(__ubuf__ float *outputAddr, __ubuf__ float *inputAddr,
                                        __ubuf__ float *descaleAddr, uint32_t row, uint32_t col, uint32_t actualCol)
{
    MicroAPI::RegTensor<float> vregInput[2];
    MicroAPI::RegTensor<float> vregDescale[2];
    MicroAPI::RegTensor<float> vregMul[2];

    MicroAPI::MaskReg mask = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();

    MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(vregDescale[0], descaleAddr);
    MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(vregDescale[1], descaleAddr + VF_FLOAT_REP_SIZE);

    for (uint16_t i = 0; i < row; i++) {
        uint64_t offset = i * actualCol;

        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(vregInput[0], inputAddr + offset);
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(vregInput[1], inputAddr + offset + VF_FLOAT_REP_SIZE);

        MicroAPI::Mul(vregMul[0], vregInput[0], vregDescale[0], mask);
        MicroAPI::Mul(vregMul[1], vregInput[1], vregDescale[1], mask);

        MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM>(outputAddr + offset, vregMul[0], mask);
        MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM>(outputAddr + offset + VF_FLOAT_REP_SIZE, vregMul[1],
                                                                    mask);
    }
}

__simd_vf__ void DequantVfCoff1D256Impl(__ubuf__ float *outputAddr, __ubuf__ float *inputAddr,
                                        __ubuf__ float *descaleAddr, uint32_t row, uint32_t col, uint32_t actualCol)
{
    MicroAPI::RegTensor<float> vregInput[4];
    MicroAPI::RegTensor<float> vregDescale[4];
    MicroAPI::RegTensor<float> vregMul[4];

    MicroAPI::MaskReg mask = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();

    MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(vregDescale[0], descaleAddr);
    MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(vregDescale[1], descaleAddr + VF_FLOAT_REP_SIZE);
    MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(vregDescale[2], descaleAddr + 2 * VF_FLOAT_REP_SIZE);
    MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(vregDescale[3], descaleAddr + 3 * VF_FLOAT_REP_SIZE);

    for (uint16_t i = 0; i < row; i++) {
        uint64_t offset = i * actualCol;

        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(vregInput[0], inputAddr + offset);
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(vregInput[1], inputAddr + offset + VF_FLOAT_REP_SIZE);
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(vregInput[2],
                                                                  inputAddr + offset + 2 * VF_FLOAT_REP_SIZE);
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(vregInput[3],
                                                                  inputAddr + offset + 3 * VF_FLOAT_REP_SIZE);

        MicroAPI::Mul(vregMul[0], vregInput[0], vregDescale[0], mask);
        MicroAPI::Mul(vregMul[1], vregInput[1], vregDescale[1], mask);
        MicroAPI::Mul(vregMul[2], vregInput[2], vregDescale[2], mask);
        MicroAPI::Mul(vregMul[3], vregInput[3], vregDescale[3], mask);

        MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM>(outputAddr + offset, vregMul[0], mask);
        MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM>(outputAddr + offset + VF_FLOAT_REP_SIZE, vregMul[1],
                                                                    mask);
        MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM>(outputAddr + offset + 2 * VF_FLOAT_REP_SIZE,
                                                                    vregMul[2], mask);
        MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM>(outputAddr + offset + 3 * VF_FLOAT_REP_SIZE,
                                                                    vregMul[3], mask);
    }
}

__simd_vf__ void DequantVfCoff1D512Impl(__ubuf__ float *outputAddr, __ubuf__ float *inputAddr,
                                        __ubuf__ float *descaleAddr, uint32_t row, uint32_t col, uint32_t actualCol)
{
    MicroAPI::RegTensor<float> vregInput[8];
    MicroAPI::RegTensor<float> vregDescale[8];
    MicroAPI::RegTensor<float> vregMul[8];

    MicroAPI::MaskReg mask = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();

    MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(vregDescale[0], descaleAddr);
    MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(vregDescale[1], descaleAddr + VF_FLOAT_REP_SIZE);
    MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(vregDescale[2], descaleAddr + 2 * VF_FLOAT_REP_SIZE);
    MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(vregDescale[3], descaleAddr + 3 * VF_FLOAT_REP_SIZE);
    MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(vregDescale[4], descaleAddr + 4 * VF_FLOAT_REP_SIZE);
    MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(vregDescale[5], descaleAddr + 5 * VF_FLOAT_REP_SIZE);
    MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(vregDescale[6], descaleAddr + 6 * VF_FLOAT_REP_SIZE);
    MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(vregDescale[7], descaleAddr + 7 * VF_FLOAT_REP_SIZE);

    for (uint16_t i = 0; i < row; i++) {
        uint64_t offset = i * actualCol;

        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(vregInput[0], inputAddr + offset);
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(vregInput[1], inputAddr + offset + VF_FLOAT_REP_SIZE);
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(vregInput[2],
                                                                  inputAddr + offset + 2 * VF_FLOAT_REP_SIZE);
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(vregInput[3],
                                                                  inputAddr + offset + 3 * VF_FLOAT_REP_SIZE);
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(vregInput[4],
                                                                  inputAddr + offset + 4 * VF_FLOAT_REP_SIZE);
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(vregInput[5],
                                                                  inputAddr + offset + 5 * VF_FLOAT_REP_SIZE);
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(vregInput[6],
                                                                  inputAddr + offset + 6 * VF_FLOAT_REP_SIZE);
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(vregInput[7],
                                                                  inputAddr + offset + 7 * VF_FLOAT_REP_SIZE);

        MicroAPI::Mul(vregMul[0], vregInput[0], vregDescale[0], mask);
        MicroAPI::Mul(vregMul[1], vregInput[1], vregDescale[1], mask);
        MicroAPI::Mul(vregMul[2], vregInput[2], vregDescale[2], mask);
        MicroAPI::Mul(vregMul[3], vregInput[3], vregDescale[3], mask);
        MicroAPI::Mul(vregMul[4], vregInput[4], vregDescale[4], mask);
        MicroAPI::Mul(vregMul[5], vregInput[5], vregDescale[5], mask);
        MicroAPI::Mul(vregMul[6], vregInput[6], vregDescale[6], mask);
        MicroAPI::Mul(vregMul[7], vregInput[7], vregDescale[7], mask);

        MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM>(outputAddr + offset, vregMul[0], mask);
        MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM>(outputAddr + offset + VF_FLOAT_REP_SIZE, vregMul[1],
                                                                    mask);
        MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM>(outputAddr + offset + 2 * VF_FLOAT_REP_SIZE,
                                                                    vregMul[2], mask);
        MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM>(outputAddr + offset + 3 * VF_FLOAT_REP_SIZE,
                                                                    vregMul[3], mask);
        MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM>(outputAddr + offset + 4 * VF_FLOAT_REP_SIZE,
                                                                    vregMul[4], mask);
        MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM>(outputAddr + offset + 5 * VF_FLOAT_REP_SIZE,
                                                                    vregMul[5], mask);
        MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM>(outputAddr + offset + 6 * VF_FLOAT_REP_SIZE,
                                                                    vregMul[6], mask);
        MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM>(outputAddr + offset + 7 * VF_FLOAT_REP_SIZE,
                                                                    vregMul[7], mask);
    }
}

// ================================ Coff=2 ===================================
// total = 2 * col, 每行 = [coff0_input | coff1_input]，各col个元素
// descale: coff0_descale 从offset0加载, coff1_descale 从headDim偏移加载

__simd_vf__ void DequantVfCoff2BaseImpl(__ubuf__ float *outputAddr, __ubuf__ float *inputAddr,
                                        __ubuf__ float *descaleAddr, uint32_t row, uint32_t col, uint32_t headDim,
                                        uint32_t actualCol)
{
    MicroAPI::RegTensor<float> vregInputCoff0;
    MicroAPI::RegTensor<float> vregInputCoff1;
    MicroAPI::RegTensor<float> vregDescaleCoff0;
    MicroAPI::RegTensor<float> vregDescaleCoff1;
    MicroAPI::RegTensor<float> vregMulCoff0;
    MicroAPI::RegTensor<float> vregMulCoff1;

    MicroAPI::MaskReg mask = MicroAPI::UpdateMask<float>(col);

    MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(vregDescaleCoff0, descaleAddr);
    MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(vregDescaleCoff1, descaleAddr + headDim);

    for (uint16_t i = 0; i < row; i++) {
        uint64_t offset = i * 2 * actualCol;

        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(vregInputCoff0, inputAddr + offset);
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(vregInputCoff1, inputAddr + offset + actualCol);

        MicroAPI::Mul(vregMulCoff0, vregInputCoff0, vregDescaleCoff0, mask);
        MicroAPI::Mul(vregMulCoff1, vregInputCoff1, vregDescaleCoff1, mask);

        MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM>(outputAddr + offset, vregMulCoff0, mask);
        MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM>(outputAddr + offset + actualCol, vregMulCoff1,
                                                                    mask);
    }
}

// total = 256, coff0: 2reg input + 2reg descale, coff1: 2reg input + 2reg descale
__simd_vf__ void DequantVfCoff2D128Impl(__ubuf__ float *outputAddr, __ubuf__ float *inputAddr,
                                        __ubuf__ float *descaleAddr, uint32_t row, uint32_t col, uint32_t headDim,
                                        uint32_t actualCol)
{
    MicroAPI::RegTensor<float> vregInputCoff0[2];
    MicroAPI::RegTensor<float> vregInputCoff1[2];
    MicroAPI::RegTensor<float> vregDescaleCoff0[2];
    MicroAPI::RegTensor<float> vregDescaleCoff1[2];
    MicroAPI::RegTensor<float> vregMulCoff0[2];
    MicroAPI::RegTensor<float> vregMulCoff1[2];

    MicroAPI::MaskReg mask = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();

    MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(vregDescaleCoff0[0], descaleAddr);
    MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(vregDescaleCoff0[1], descaleAddr + VF_FLOAT_REP_SIZE);
    MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(vregDescaleCoff1[0], descaleAddr + headDim);
    MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(vregDescaleCoff1[1],
                                                              descaleAddr + headDim + VF_FLOAT_REP_SIZE);

    for (uint16_t i = 0; i < row; i++) {
        uint64_t offset = i * 2 * actualCol;

        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(vregInputCoff0[0], inputAddr + offset);
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(vregInputCoff0[1],
                                                                  inputAddr + offset + VF_FLOAT_REP_SIZE);
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(vregInputCoff1[0], inputAddr + offset + actualCol);
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(vregInputCoff1[1],
                                                                  inputAddr + offset + actualCol + VF_FLOAT_REP_SIZE);

        MicroAPI::Mul(vregMulCoff0[0], vregInputCoff0[0], vregDescaleCoff0[0], mask);
        MicroAPI::Mul(vregMulCoff0[1], vregInputCoff0[1], vregDescaleCoff0[1], mask);
        MicroAPI::Mul(vregMulCoff1[0], vregInputCoff1[0], vregDescaleCoff1[0], mask);
        MicroAPI::Mul(vregMulCoff1[1], vregInputCoff1[1], vregDescaleCoff1[1], mask);

        MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM>(outputAddr + offset, vregMulCoff0[0], mask);
        MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM>(outputAddr + offset + VF_FLOAT_REP_SIZE,
                                                                    vregMulCoff0[1], mask);
        MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM>(outputAddr + offset + actualCol, vregMulCoff1[0],
                                                                    mask);
        MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM>(outputAddr + offset + actualCol + VF_FLOAT_REP_SIZE,
                                                                    vregMulCoff1[1], mask);
    }
}

// total = 512, coff0: 4reg input + 4reg descale, coff1: 4reg input + 4reg descale
__simd_vf__ void DequantVfCoff2D256Impl(__ubuf__ float *outputAddr, __ubuf__ float *inputAddr,
                                        __ubuf__ float *descaleAddr, uint32_t row, uint32_t col, uint32_t headDim,
                                        uint32_t actualCol)
{
    MicroAPI::RegTensor<float> vregInputCoff0[4];
    MicroAPI::RegTensor<float> vregInputCoff1[4];
    MicroAPI::RegTensor<float> vregDescaleCoff0[4];
    MicroAPI::RegTensor<float> vregDescaleCoff1[4];
    MicroAPI::RegTensor<float> vregMulCoff0[4];
    MicroAPI::RegTensor<float> vregMulCoff1[4];

    MicroAPI::MaskReg mask = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();

    MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(vregDescaleCoff0[0], descaleAddr);
    MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(vregDescaleCoff0[1], descaleAddr + VF_FLOAT_REP_SIZE);
    MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(vregDescaleCoff0[2], descaleAddr + 2 * VF_FLOAT_REP_SIZE);
    MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(vregDescaleCoff0[3], descaleAddr + 3 * VF_FLOAT_REP_SIZE);
    MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(vregDescaleCoff1[0], descaleAddr + headDim);
    MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(vregDescaleCoff1[1],
                                                              descaleAddr + headDim + VF_FLOAT_REP_SIZE);
    MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(vregDescaleCoff1[2],
                                                              descaleAddr + headDim + 2 * VF_FLOAT_REP_SIZE);
    MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(vregDescaleCoff1[3],
                                                              descaleAddr + headDim + 3 * VF_FLOAT_REP_SIZE);

    for (uint16_t i = 0; i < row; i++) {
        uint64_t offset = i * 2 * actualCol;

        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(vregInputCoff0[0], inputAddr + offset);
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(vregInputCoff0[1],
                                                                  inputAddr + offset + VF_FLOAT_REP_SIZE);
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(vregInputCoff0[2],
                                                                  inputAddr + offset + 2 * VF_FLOAT_REP_SIZE);
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(vregInputCoff0[3],
                                                                  inputAddr + offset + 3 * VF_FLOAT_REP_SIZE);
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(vregInputCoff1[0], inputAddr + offset + actualCol);
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(vregInputCoff1[1],
                                                                  inputAddr + offset + actualCol + VF_FLOAT_REP_SIZE);
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(
            vregInputCoff1[2], inputAddr + offset + actualCol + 2 * VF_FLOAT_REP_SIZE);
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(
            vregInputCoff1[3], inputAddr + offset + actualCol + 3 * VF_FLOAT_REP_SIZE);

        MicroAPI::Mul(vregMulCoff0[0], vregInputCoff0[0], vregDescaleCoff0[0], mask);
        MicroAPI::Mul(vregMulCoff0[1], vregInputCoff0[1], vregDescaleCoff0[1], mask);
        MicroAPI::Mul(vregMulCoff0[2], vregInputCoff0[2], vregDescaleCoff0[2], mask);
        MicroAPI::Mul(vregMulCoff0[3], vregInputCoff0[3], vregDescaleCoff0[3], mask);
        MicroAPI::Mul(vregMulCoff1[0], vregInputCoff1[0], vregDescaleCoff1[0], mask);
        MicroAPI::Mul(vregMulCoff1[1], vregInputCoff1[1], vregDescaleCoff1[1], mask);
        MicroAPI::Mul(vregMulCoff1[2], vregInputCoff1[2], vregDescaleCoff1[2], mask);
        MicroAPI::Mul(vregMulCoff1[3], vregInputCoff1[3], vregDescaleCoff1[3], mask);

        MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM>(outputAddr + offset, vregMulCoff0[0], mask);
        MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM>(outputAddr + offset + VF_FLOAT_REP_SIZE,
                                                                    vregMulCoff0[1], mask);
        MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM>(outputAddr + offset + 2 * VF_FLOAT_REP_SIZE,
                                                                    vregMulCoff0[2], mask);
        MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM>(outputAddr + offset + 3 * VF_FLOAT_REP_SIZE,
                                                                    vregMulCoff0[3], mask);
        MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM>(outputAddr + offset + actualCol, vregMulCoff1[0],
                                                                    mask);
        MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM>(outputAddr + offset + actualCol + VF_FLOAT_REP_SIZE,
                                                                    vregMulCoff1[1], mask);
        MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM>(
            outputAddr + offset + actualCol + 2 * VF_FLOAT_REP_SIZE, vregMulCoff1[2], mask);
        MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM>(
            outputAddr + offset + actualCol + 3 * VF_FLOAT_REP_SIZE, vregMulCoff1[3], mask);
    }
}

// total = 1024, coff0: 8reg input + 8reg descale, coff1: 8reg input + 8reg descale
__simd_vf__ void DequantVfCoff2D512Impl(__ubuf__ float *outputAddr, __ubuf__ float *inputAddr,
                                        __ubuf__ float *descaleAddr, uint32_t row, uint32_t col, uint32_t headDim,
                                        uint32_t actualCol)
{
    MicroAPI::RegTensor<float> vregInputCoff0[8];
    MicroAPI::RegTensor<float> vregInputCoff1[8];
    MicroAPI::RegTensor<float> vregDescaleCoff0[8];
    MicroAPI::RegTensor<float> vregDescaleCoff1[8];
    MicroAPI::RegTensor<float> vregMulCoff0[8];
    MicroAPI::RegTensor<float> vregMulCoff1[8];

    MicroAPI::MaskReg mask = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();

    MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(vregDescaleCoff0[0], descaleAddr);
    MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(vregDescaleCoff0[1], descaleAddr + VF_FLOAT_REP_SIZE);
    MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(vregDescaleCoff0[2], descaleAddr + 2 * VF_FLOAT_REP_SIZE);
    MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(vregDescaleCoff0[3], descaleAddr + 3 * VF_FLOAT_REP_SIZE);
    MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(vregDescaleCoff0[4], descaleAddr + 4 * VF_FLOAT_REP_SIZE);
    MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(vregDescaleCoff0[5], descaleAddr + 5 * VF_FLOAT_REP_SIZE);
    MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(vregDescaleCoff0[6], descaleAddr + 6 * VF_FLOAT_REP_SIZE);
    MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(vregDescaleCoff0[7], descaleAddr + 7 * VF_FLOAT_REP_SIZE);
    MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(vregDescaleCoff1[0], descaleAddr + headDim);
    MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(vregDescaleCoff1[1],
                                                              descaleAddr + headDim + VF_FLOAT_REP_SIZE);
    MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(vregDescaleCoff1[2],
                                                              descaleAddr + headDim + 2 * VF_FLOAT_REP_SIZE);
    MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(vregDescaleCoff1[3],
                                                              descaleAddr + headDim + 3 * VF_FLOAT_REP_SIZE);
    MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(vregDescaleCoff1[4],
                                                              descaleAddr + headDim + 4 * VF_FLOAT_REP_SIZE);
    MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(vregDescaleCoff1[5],
                                                              descaleAddr + headDim + 5 * VF_FLOAT_REP_SIZE);
    MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(vregDescaleCoff1[6],
                                                              descaleAddr + headDim + 6 * VF_FLOAT_REP_SIZE);
    MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(vregDescaleCoff1[7],
                                                              descaleAddr + headDim + 7 * VF_FLOAT_REP_SIZE);

    for (uint16_t i = 0; i < row; i++) {
        uint64_t offset = i * 2 * actualCol;

        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(vregInputCoff0[0], inputAddr + offset);
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(vregInputCoff0[1],
                                                                  inputAddr + offset + VF_FLOAT_REP_SIZE);
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(vregInputCoff0[2],
                                                                  inputAddr + offset + 2 * VF_FLOAT_REP_SIZE);
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(vregInputCoff0[3],
                                                                  inputAddr + offset + 3 * VF_FLOAT_REP_SIZE);
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(vregInputCoff0[4],
                                                                  inputAddr + offset + 4 * VF_FLOAT_REP_SIZE);
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(vregInputCoff0[5],
                                                                  inputAddr + offset + 5 * VF_FLOAT_REP_SIZE);
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(vregInputCoff0[6],
                                                                  inputAddr + offset + 6 * VF_FLOAT_REP_SIZE);
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(vregInputCoff0[7],
                                                                  inputAddr + offset + 7 * VF_FLOAT_REP_SIZE);
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(vregInputCoff1[0], inputAddr + offset + actualCol);
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(vregInputCoff1[1],
                                                                  inputAddr + offset + actualCol + VF_FLOAT_REP_SIZE);
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(
            vregInputCoff1[2], inputAddr + offset + actualCol + 2 * VF_FLOAT_REP_SIZE);
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(
            vregInputCoff1[3], inputAddr + offset + actualCol + 3 * VF_FLOAT_REP_SIZE);
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(
            vregInputCoff1[4], inputAddr + offset + actualCol + 4 * VF_FLOAT_REP_SIZE);
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(
            vregInputCoff1[5], inputAddr + offset + actualCol + 5 * VF_FLOAT_REP_SIZE);
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(
            vregInputCoff1[6], inputAddr + offset + actualCol + 6 * VF_FLOAT_REP_SIZE);
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(
            vregInputCoff1[7], inputAddr + offset + actualCol + 7 * VF_FLOAT_REP_SIZE);

        MicroAPI::Mul(vregMulCoff0[0], vregInputCoff0[0], vregDescaleCoff0[0], mask);
        MicroAPI::Mul(vregMulCoff0[1], vregInputCoff0[1], vregDescaleCoff0[1], mask);
        MicroAPI::Mul(vregMulCoff0[2], vregInputCoff0[2], vregDescaleCoff0[2], mask);
        MicroAPI::Mul(vregMulCoff0[3], vregInputCoff0[3], vregDescaleCoff0[3], mask);
        MicroAPI::Mul(vregMulCoff0[4], vregInputCoff0[4], vregDescaleCoff0[4], mask);
        MicroAPI::Mul(vregMulCoff0[5], vregInputCoff0[5], vregDescaleCoff0[5], mask);
        MicroAPI::Mul(vregMulCoff0[6], vregInputCoff0[6], vregDescaleCoff0[6], mask);
        MicroAPI::Mul(vregMulCoff0[7], vregInputCoff0[7], vregDescaleCoff0[7], mask);
        MicroAPI::Mul(vregMulCoff1[0], vregInputCoff1[0], vregDescaleCoff1[0], mask);
        MicroAPI::Mul(vregMulCoff1[1], vregInputCoff1[1], vregDescaleCoff1[1], mask);
        MicroAPI::Mul(vregMulCoff1[2], vregInputCoff1[2], vregDescaleCoff1[2], mask);
        MicroAPI::Mul(vregMulCoff1[3], vregInputCoff1[3], vregDescaleCoff1[3], mask);
        MicroAPI::Mul(vregMulCoff1[4], vregInputCoff1[4], vregDescaleCoff1[4], mask);
        MicroAPI::Mul(vregMulCoff1[5], vregInputCoff1[5], vregDescaleCoff1[5], mask);
        MicroAPI::Mul(vregMulCoff1[6], vregInputCoff1[6], vregDescaleCoff1[6], mask);
        MicroAPI::Mul(vregMulCoff1[7], vregInputCoff1[7], vregDescaleCoff1[7], mask);

        MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM>(outputAddr + offset, vregMulCoff0[0], mask);
        MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM>(outputAddr + offset + VF_FLOAT_REP_SIZE,
                                                                    vregMulCoff0[1], mask);
        MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM>(outputAddr + offset + 2 * VF_FLOAT_REP_SIZE,
                                                                    vregMulCoff0[2], mask);
        MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM>(outputAddr + offset + 3 * VF_FLOAT_REP_SIZE,
                                                                    vregMulCoff0[3], mask);
        MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM>(outputAddr + offset + 4 * VF_FLOAT_REP_SIZE,
                                                                    vregMulCoff0[4], mask);
        MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM>(outputAddr + offset + 5 * VF_FLOAT_REP_SIZE,
                                                                    vregMulCoff0[5], mask);
        MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM>(outputAddr + offset + 6 * VF_FLOAT_REP_SIZE,
                                                                    vregMulCoff0[6], mask);
        MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM>(outputAddr + offset + 7 * VF_FLOAT_REP_SIZE,
                                                                    vregMulCoff0[7], mask);
        MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM>(outputAddr + offset + actualCol, vregMulCoff1[0],
                                                                    mask);
        MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM>(outputAddr + offset + actualCol + VF_FLOAT_REP_SIZE,
                                                                    vregMulCoff1[1], mask);
        MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM>(
            outputAddr + offset + actualCol + 2 * VF_FLOAT_REP_SIZE, vregMulCoff1[2], mask);
        MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM>(
            outputAddr + offset + actualCol + 3 * VF_FLOAT_REP_SIZE, vregMulCoff1[3], mask);
        MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM>(
            outputAddr + offset + actualCol + 4 * VF_FLOAT_REP_SIZE, vregMulCoff1[4], mask);
        MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM>(
            outputAddr + offset + actualCol + 5 * VF_FLOAT_REP_SIZE, vregMulCoff1[5], mask);
        MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM>(
            outputAddr + offset + actualCol + 6 * VF_FLOAT_REP_SIZE, vregMulCoff1[6], mask);
        MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM>(
            outputAddr + offset + actualCol + 7 * VF_FLOAT_REP_SIZE, vregMulCoff1[7], mask);
    }
}

// ================================ 入口 ===================================
/**
 * @brief DequantVf 对matmul输出做反量化：output = input * descale
 * @param outputLocal 输出tensor [row, coff * actualCol], 支持原地更新
 * @param inputLocal 输入tensor（matmul输出）[row, coff * actualCol]
 * @param descale 反量化权重 (xDescale * wDescale) [coff * headDim], coff间以headDim偏移
 * @param row 行数
 * @param col 有效列数
 * @param headDim 实际列数，用于计算coff间偏移
 * @param actualCol input/output的行偏移列数（含padding）
 */
template <COFF Coff>
__aicore__ inline void DequantVf(const LocalTensor<float> &outputLocal, const LocalTensor<float> &inputLocal,
                                 const LocalTensor<float> &descale, uint32_t row, uint32_t col, uint32_t headDim,
                                 uint32_t actualCol)
{
    __ubuf__ float *inputAddr = (__ubuf__ float *)inputLocal.GetPhyAddr();
    __ubuf__ float *descaleAddr = (__ubuf__ float *)descale.GetPhyAddr();
    __ubuf__ float *outputAddr = (__ubuf__ float *)outputLocal.GetPhyAddr();

    if constexpr (Coff == COFF::DISABLE) {
        if (col <= VF_D_SIZE_64) {
            DequantVfCoff1BaseImpl(outputAddr, inputAddr, descaleAddr, row, col, actualCol);
        } else if (col == VF_D_SIZE_128) {
            DequantVfCoff1D128Impl(outputAddr, inputAddr, descaleAddr, row, col, actualCol);
        } else if (col == VF_D_SIZE_256) {
            DequantVfCoff1D256Impl(outputAddr, inputAddr, descaleAddr, row, col, actualCol);
        } else {
            DequantVfCoff1D512Impl(outputAddr, inputAddr, descaleAddr, row, col, actualCol);
        }
    } else {
        if (col <= VF_D_SIZE_64) {
            DequantVfCoff2BaseImpl(outputAddr, inputAddr, descaleAddr, row, col, headDim, actualCol);
        } else if (col == VF_D_SIZE_128) {
            DequantVfCoff2D128Impl(outputAddr, inputAddr, descaleAddr, row, col, headDim, actualCol);
        } else if (col == VF_D_SIZE_256) {
            DequantVfCoff2D256Impl(outputAddr, inputAddr, descaleAddr, row, col, headDim, actualCol);
        } else {
            DequantVfCoff2D512Impl(outputAddr, inputAddr, descaleAddr, row, col, headDim, actualCol);
        }
    }
}

#endif
