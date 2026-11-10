/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#if __CCE_AICORE__ == 310
#include "arc35/block_sparse_attention_grad_arch35.h"
#include "arc35/vector_api/non_zero_big_mask_dim1.h"
#else
#include "arc22/block_sparse_attention_grad_interface.h"
#endif
// ============================================================================
// Kernel Entry Point
// ============================================================================

#if __CCE_AICORE__ == 310
namespace {
// Compact a small mask on one AIV. Other AIVs may already clear gradient
// buffers and compute rowsums; their existing preparation SyncAll publishes
// these indices before any AIC/AIV consumes them.
template <bool GROUPED = false>
__aicore__ inline void SmallMaskIndices(GM_ADDR mask, GM_ADDR indices, GM_ADDR shape, uint32_t count, TPipe* pipe,
                                        uint32_t qBlocks = 0, uint32_t kBlocks = 0, uint32_t qGroup = 1)
{
    TBuf<TPosition::VECCALC> storage;
    uint32_t aligned = (count + 255) / 256 * 256;
    pipe->InitBuffer(storage, aligned * 5 + 288);
    auto x = storage.GetWithOffset<uint8_t>(aligned, 0);
    auto y = storage.GetWithOffset<int32_t>(aligned + 64, aligned);
    auto outShape = storage.GetWithOffset<uint64_t>(4, aligned * 5 + 256);
    GlobalTensor<uint8_t> maskGm;
    maskGm.SetGlobalBuffer((__gm__ uint8_t*)mask);
    GlobalTensor<int32_t> indexGm;
    indexGm.SetGlobalBuffer((__gm__ int32_t*)indices);
    GlobalTensor<uint64_t> shapeGm;
    shapeGm.SetGlobalBuffer((__gm__ uint64_t*)shape);
    DataCopyPad(x, maskGm, {1, count, 0, 0, 0}, {false, 0, 0, 0});
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    auto src = (__ubuf__ uint8_t*)x.GetPhyAddr();
    auto dst = (__ubuf__ int32_t*)y.GetPhyAddr();
    uint16_t loops = (count + 255) / 256;
    if (qBlocks > 0) {
        uint32_t heads = count / (qBlocks * kBlocks);
        const uint32_t groupSize = GROUPED ? qGroup : 1;
        uint16_t qLoops = (qBlocks + 127) / 128;
        __VEC_SCOPE__
        {
            Reg::RegTensor<uint16_t> offsets, readOffsets;
            Reg::RegTensor<int16_t> values;
            Reg::RegTensor<int32_t> ids, packed;
            Reg::MaskReg all16 = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
            Reg::MaskReg all32 = Reg::CreateMask<int32_t, Reg::MaskPattern::ALL>();
            Reg::MaskReg valid, selected, low, high;
            Reg::UnalignReg state;
            Reg::ClearSpr<SpecialPurposeReg::AR>();
            Reg::Arange((Reg::RegTensor<int16_t>&)offsets, (int16_t)0);
            Reg::Muls(offsets, offsets, (uint16_t)kBlocks, all16);
            for (uint16_t kvHead = 0; kvHead < heads / groupSize; ++kvHead) {
                for (uint16_t k = 0; k < kBlocks; ++k) {
                    for (uint16_t g = 0; g < groupSize; ++g) {
                        uint16_t h = kvHead * groupSize + g;
                        uint32_t left = qBlocks;
                        for (uint16_t q = 0; q < qLoops; ++q) {
                            valid = Reg::UpdateMask<int16_t>(left);
                            uint16_t base = h * qBlocks * kBlocks + k + q * 128 * kBlocks;
                            Reg::Adds(readOffsets, offsets, base, all16);
                            Reg::Gather(values, src, readOffsets, valid);
                            Reg::CompareScalar<int16_t, CMPMODE::NE>(selected, values, (int16_t)0, valid);
                            Reg::MaskUnPack<Reg::HighLowPart::LOWEST>(low, selected);
                            Reg::Arange(ids, (int32_t)(h * qBlocks * kBlocks + k * qBlocks + q * 128));
                            Reg::GatherMask<int32_t, Reg::GatherMaskMode::STORE_REG>(packed, ids, low);
                            Reg::DataCopyUnAlign<int32_t, Reg::PostLiteral::POST_MODE_UPDATE>(dst, packed, state);
                            if (qBlocks > 64) {
                                Reg::MaskUnPack<Reg::HighLowPart::HIGHEST>(high, selected);
                                Reg::Adds(ids, ids, 64, all32);
                                Reg::GatherMask<int32_t, Reg::GatherMaskMode::STORE_REG>(packed, ids, high);
                                Reg::DataCopyUnAlign<int32_t, Reg::PostLiteral::POST_MODE_UPDATE>(dst, packed, state);
                            }
                        }
                    }
                }
            }
            Reg::DataCopyUnAlignPost(dst, state);
        }
    } else {
        __VEC_SCOPE__
        {
            Reg::RegTensor<uint8_t> value;
            Reg::RegTensor<int32_t> ids, packed;
            Reg::MaskReg maskAll = Reg::CreateMask<int32_t, Reg::MaskPattern::ALL>();
            Reg::MaskReg valid, selected, low, high, m0, m1, m2, m3;
            Reg::UnalignReg state;
            Reg::ClearSpr<SpecialPurposeReg::AR>();
            Reg::Arange(ids, 0);
            uint32_t remaining = count;
            for (uint16_t i = 0; i < loops; ++i) {
                valid = Reg::UpdateMask<uint8_t>(remaining);
                Reg::DataCopy(value, src + i * 256);
                Reg::CompareScalar<uint8_t, CMPMODE::NE>(selected, value, 0, valid);
                Reg::MaskUnPack<Reg::HighLowPart::LOWEST>(low, selected);
                Reg::MaskUnPack<Reg::HighLowPart::HIGHEST>(high, selected);
                Reg::MaskUnPack<Reg::HighLowPart::LOWEST>(m0, low);
                Reg::MaskUnPack<Reg::HighLowPart::HIGHEST>(m1, low);
                Reg::MaskUnPack<Reg::HighLowPart::LOWEST>(m2, high);
                Reg::MaskUnPack<Reg::HighLowPart::HIGHEST>(m3, high);
                Reg::GatherMask<int32_t, Reg::GatherMaskMode::STORE_REG>(packed, ids, m0);
                Reg::DataCopyUnAlign<int32_t, Reg::PostLiteral::POST_MODE_UPDATE>(dst, packed, state);
                Reg::Adds(ids, ids, 64, maskAll);
                Reg::GatherMask<int32_t, Reg::GatherMaskMode::STORE_REG>(packed, ids, m1);
                Reg::DataCopyUnAlign<int32_t, Reg::PostLiteral::POST_MODE_UPDATE>(dst, packed, state);
                Reg::Adds(ids, ids, 64, maskAll);
                Reg::GatherMask<int32_t, Reg::GatherMaskMode::STORE_REG>(packed, ids, m2);
                Reg::DataCopyUnAlign<int32_t, Reg::PostLiteral::POST_MODE_UPDATE>(dst, packed, state);
                Reg::Adds(ids, ids, 64, maskAll);
                Reg::GatherMask<int32_t, Reg::GatherMaskMode::STORE_REG>(packed, ids, m3);
                Reg::DataCopyUnAlign<int32_t, Reg::PostLiteral::POST_MODE_UPDATE>(dst, packed, state);
                Reg::Adds(ids, ids, 64, maskAll);
            }
            Reg::DataCopyUnAlignPost(dst, state);
        }
    }
    uint32_t nonzero = Reg::GetSpr<SpecialPurposeReg::AR>() / 4;
    outShape.SetValue(0, 2);
    outShape.SetValue(1, 1);
    outShape.SetValue(2, nonzero);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
    if (nonzero > 0)
        DataCopyPad(indexGm, y, {1, nonzero * 4, 0, 0, 0});
    DataCopyPad(shapeGm, outShape, {1, 24, 0, 0, 0});
    SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
}

template <typename BSA_TYPE_T>
__aicore__ inline void RunBsagByNonZeroIndex(
    GM_ADDR dout, GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR out, GM_ADDR softmaxLse, GM_ADDR blockSparseMask,
    GM_ADDR blockShape, GM_ADDR attentionMask, GM_ADDR actualSeqLengths, GM_ADDR actualSeqLengthsKv, GM_ADDR qDescale,
    GM_ADDR kDescale, GM_ADDR vDescale, GM_ADDR doDescale, GM_ADDR pScale, GM_ADDR dsScale, GM_ADDR dq, GM_ADDR dk,
    GM_ADDR dv, GM_ADDR user, const BlockSparseAttentionGradTilingDataArch35* tilingDataPtr, TPipe* tPipe)
{
    if ASCEND_IS_AIV {
        const auto& nz = tilingDataPtr->mNonZeroTilingData;
        uint64_t elements = nz.realCoreNum * nz.numPerCore + nz.numTailCore;
        if (BSA_TYPE_T::hif8_enable && elements <= 32768) {
            if (GetBlockIdx() == 0) {
                uint32_t qBlocks = 0, kBlocks = 0;
                if (tilingDataPtr->BlockX == 128 && tilingDataPtr->BlockY == 128) {
                    uint64_t maxQ = tilingDataPtr->qSeqLen, maxK = tilingDataPtr->kvSeqLen;
                    if constexpr (BSA_TYPE_T::input_layout == BSA_ARC35::TND) {
                        maxQ = 0;
                        maxK = 0;
                        int64_t lastQ = 0, lastK = 0;
                        for (uint32_t b = 0; b < tilingDataPtr->batchNum; ++b) {
                            int64_t endQ = ((__gm__ int64_t*)actualSeqLengths)[b];
                            int64_t endK = ((__gm__ int64_t*)actualSeqLengthsKv)[b];
                            maxQ = maxQ > endQ - lastQ ? maxQ : endQ - lastQ;
                            maxK = maxK > endK - lastK ? maxK : endK - lastK;
                            lastQ = endQ;
                            lastK = endK;
                        }
                    }
                    qBlocks = (maxQ + 127) / 128;
                    kBlocks = (maxK + 127) / 128;
                }
                if (qBlocks > 0 &&
                    (qBlocks <= 64 ||
                     (qBlocks <= 128 && tilingDataPtr->headDim == 128 && *((__gm__ float*)dsScale) > 0.0f)) &&
                    tilingDataPtr->qGroup > 1) {
                    SmallMaskIndices<true>(blockSparseMask, user + tilingDataPtr->indexWorkspaceOffset,
                                           user + tilingDataPtr->indexShapeWorkspaceOffset, elements, tPipe, qBlocks,
                                           kBlocks, tilingDataPtr->qGroup);
                } else {
                    SmallMaskIndices<false>(blockSparseMask, user + tilingDataPtr->indexWorkspaceOffset,
                                            user + tilingDataPtr->indexShapeWorkspaceOffset, elements, tPipe, qBlocks,
                                            kBlocks);
                }
            }
        } else {
            NonZero::NonZeroBigMaskDim1<int8_t, int32_t> indexOp;
            indexOp.Init(blockSparseMask, user + tilingDataPtr->indexWorkspaceOffset,
                         user + tilingDataPtr->indexShapeWorkspaceOffset, user + tilingDataPtr->nonZeroWorkspaceOffset,
                         &tilingDataPtr->mNonZeroTilingData, tPipe);
            indexOp.Process<NonZero::NonZeroBigMaskDim1<int8_t, int32_t>,
                            &NonZero::NonZeroBigMaskDim1<int8_t, int32_t>::ComputeOutput>(&indexOp);
        }
    }
    tPipe->Destroy();
    tPipe->Init();
    BSA_ARC35::BlockSparseAttentionGradArch35<BSA_TYPE_T> op;
    op.Process(dout, query, key, value, out, softmaxLse, blockSparseMask, blockShape, attentionMask, actualSeqLengths,
               actualSeqLengthsKv, qDescale, kDescale, vDescale, doDescale, pScale, dsScale, dq, dk, dv, user,
               tilingDataPtr, tPipe);
}
} // namespace
#endif

/**
 * @brief BlockSparseAttentionGrad算子的kernel入口函数
 *
 * 这是NPU上执行的入口点，使用extern "C"以避免C++名称修饰，
 * 使用__global__和__aicore__修饰符标识为全局kernel函数
 */
extern "C" __global__ __aicore__ void block_sparse_attention_grad(
    __gm__ uint8_t* dout, __gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* value, __gm__ uint8_t* out,
    __gm__ uint8_t* softmaxLse, __gm__ uint8_t* blockSparseMask, __gm__ uint8_t* attentionMask,
    __gm__ uint8_t* blockShape, __gm__ uint8_t* actualSeqLengths, __gm__ uint8_t* actualSeqLengthsKv,
    __gm__ uint8_t* qDescale, __gm__ uint8_t* kDescale, __gm__ uint8_t* vDescale, __gm__ uint8_t* doDescale,
    __gm__ uint8_t* pScale, __gm__ uint8_t* dsScale, __gm__ uint8_t* dq, __gm__ uint8_t* dk, __gm__ uint8_t* dv,
    __gm__ uint8_t* workspace, __gm__ uint8_t* tiling)
{
    __gm__ uint8_t* user = AscendC::GetUserWorkspace(workspace);

#if __CCE_AICORE__ == 310
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    GET_TILING_DATA_WITH_STRUCT(BlockSparseAttentionGradTilingDataArch35, tiling_data_in, tiling);
    const BlockSparseAttentionGradTilingDataArch35* __restrict tilingDataPtrArch35 = &tiling_data_in;
    using ARC35_TILING_CLASS = BlockSparseAttentionGradTilingDataArch35;
    TPipe tPipe;
    if (TILING_KEY_IS(1000)) {
        using bsa_type = BSA_ARC35::BSA_TYPE<bfloat16_t, BSA_ARC35::BSND, ARC35_TILING_CLASS, false, true>;
        RunBsagByNonZeroIndex<bsa_type>(dout, query, key, value, out, softmaxLse, blockSparseMask, blockShape,
                                        attentionMask, actualSeqLengths, actualSeqLengthsKv, qDescale, kDescale,
                                        vDescale, doDescale, pScale, dsScale, dq, dk, dv, user, tilingDataPtrArch35,
                                        &tPipe);
    } else if (TILING_KEY_IS(1001)) {
        using bsa_type = BSA_ARC35::BSA_TYPE<half, BSA_ARC35::BSND, ARC35_TILING_CLASS, false, true>;
        RunBsagByNonZeroIndex<bsa_type>(dout, query, key, value, out, softmaxLse, blockSparseMask, blockShape,
                                        attentionMask, actualSeqLengths, actualSeqLengthsKv, qDescale, kDescale,
                                        vDescale, doDescale, pScale, dsScale, dq, dk, dv, user, tilingDataPtrArch35,
                                        &tPipe);
    } else if (TILING_KEY_IS(1002)) {
        using bsa_type = BSA_ARC35::BSA_TYPE<bfloat16_t, BSA_ARC35::BNSD, ARC35_TILING_CLASS, false, true>;
        RunBsagByNonZeroIndex<bsa_type>(dout, query, key, value, out, softmaxLse, blockSparseMask, blockShape,
                                        attentionMask, actualSeqLengths, actualSeqLengthsKv, qDescale, kDescale,
                                        vDescale, doDescale, pScale, dsScale, dq, dk, dv, user, tilingDataPtrArch35,
                                        &tPipe);
    } else if (TILING_KEY_IS(1003)) {
        using bsa_type = BSA_ARC35::BSA_TYPE<half, BSA_ARC35::BNSD, ARC35_TILING_CLASS, false, true>;
        RunBsagByNonZeroIndex<bsa_type>(dout, query, key, value, out, softmaxLse, blockSparseMask, blockShape,
                                        attentionMask, actualSeqLengths, actualSeqLengthsKv, qDescale, kDescale,
                                        vDescale, doDescale, pScale, dsScale, dq, dk, dv, user, tilingDataPtrArch35,
                                        &tPipe);
    } else if (TILING_KEY_IS(1004)) {
        using bsa_type = BSA_ARC35::BSA_TYPE<bfloat16_t, BSA_ARC35::TND, ARC35_TILING_CLASS, false, true>;
        RunBsagByNonZeroIndex<bsa_type>(dout, query, key, value, out, softmaxLse, blockSparseMask, blockShape,
                                        attentionMask, actualSeqLengths, actualSeqLengthsKv, qDescale, kDescale,
                                        vDescale, doDescale, pScale, dsScale, dq, dk, dv, user, tilingDataPtrArch35,
                                        &tPipe);
    } else if (TILING_KEY_IS(1005)) {
        using bsa_type = BSA_ARC35::BSA_TYPE<half, BSA_ARC35::TND, ARC35_TILING_CLASS, false, true>;
        RunBsagByNonZeroIndex<bsa_type>(dout, query, key, value, out, softmaxLse, blockSparseMask, blockShape,
                                        attentionMask, actualSeqLengths, actualSeqLengthsKv, qDescale, kDescale,
                                        vDescale, doDescale, pScale, dsScale, dq, dk, dv, user, tilingDataPtrArch35,
                                        &tPipe);
    } else if (TILING_KEY_IS(2000)) {
        using bsa_type = BSA_ARC35::BSA_TYPE<hifloat8_t, BSA_ARC35::BSND, ARC35_TILING_CLASS, false, true, bfloat16_t,
                                             bfloat16_t, true>;
        RunBsagByNonZeroIndex<bsa_type>(dout, query, key, value, out, softmaxLse, blockSparseMask, blockShape,
                                        attentionMask, actualSeqLengths, actualSeqLengthsKv, qDescale, kDescale,
                                        vDescale, doDescale, pScale, dsScale, dq, dk, dv, user, tilingDataPtrArch35,
                                        &tPipe);
    } else if (TILING_KEY_IS(2002)) {
        using bsa_type = BSA_ARC35::BSA_TYPE<hifloat8_t, BSA_ARC35::BNSD, ARC35_TILING_CLASS, false, true, bfloat16_t,
                                             bfloat16_t, true>;
        RunBsagByNonZeroIndex<bsa_type>(dout, query, key, value, out, softmaxLse, blockSparseMask, blockShape,
                                        attentionMask, actualSeqLengths, actualSeqLengthsKv, qDescale, kDescale,
                                        vDescale, doDescale, pScale, dsScale, dq, dk, dv, user, tilingDataPtrArch35,
                                        &tPipe);
    } else if (TILING_KEY_IS(2004)) {
        using bsa_type = BSA_ARC35::BSA_TYPE<hifloat8_t, BSA_ARC35::TND, ARC35_TILING_CLASS, false, true, bfloat16_t,
                                             bfloat16_t, true>;
        RunBsagByNonZeroIndex<bsa_type>(dout, query, key, value, out, softmaxLse, blockSparseMask, blockShape,
                                        attentionMask, actualSeqLengths, actualSeqLengthsKv, qDescale, kDescale,
                                        vDescale, doDescale, pScale, dsScale, dq, dk, dv, user, tilingDataPtrArch35,
                                        &tPipe);
    } else if (TILING_KEY_IS(1008)) {
        using bsa_type = BSA_ARC35::BSA_TYPE<bfloat16_t, BSA_ARC35::BSND, ARC35_TILING_CLASS, true>;
        BSA_ARC35::BlockSparseAttentionGradArch35<bsa_type> op;
        op.Process(dout, query, key, value, out, softmaxLse, blockSparseMask, blockShape, attentionMask,
                   actualSeqLengths, actualSeqLengthsKv, qDescale, kDescale, vDescale, doDescale, pScale, dsScale, dq,
                   dk, dv, user, tilingDataPtrArch35, &tPipe);
    } else if (TILING_KEY_IS(1009)) {
        using bsa_type = BSA_ARC35::BSA_TYPE<half, BSA_ARC35::BSND, ARC35_TILING_CLASS, true>;
        BSA_ARC35::BlockSparseAttentionGradArch35<bsa_type> op;
        op.Process(dout, query, key, value, out, softmaxLse, blockSparseMask, blockShape, attentionMask,
                   actualSeqLengths, actualSeqLengthsKv, qDescale, kDescale, vDescale, doDescale, pScale, dsScale, dq,
                   dk, dv, user, tilingDataPtrArch35, &tPipe);
    } else if (TILING_KEY_IS(1010)) {
        using bsa_type = BSA_ARC35::BSA_TYPE<bfloat16_t, BSA_ARC35::BNSD, ARC35_TILING_CLASS, true>;
        BSA_ARC35::BlockSparseAttentionGradArch35<bsa_type> op;
        op.Process(dout, query, key, value, out, softmaxLse, blockSparseMask, blockShape, attentionMask,
                   actualSeqLengths, actualSeqLengthsKv, qDescale, kDescale, vDescale, doDescale, pScale, dsScale, dq,
                   dk, dv, user, tilingDataPtrArch35, &tPipe);
    } else if (TILING_KEY_IS(1011)) {
        using bsa_type = BSA_ARC35::BSA_TYPE<half, BSA_ARC35::BNSD, ARC35_TILING_CLASS, true>;
        BSA_ARC35::BlockSparseAttentionGradArch35<bsa_type> op;
        op.Process(dout, query, key, value, out, softmaxLse, blockSparseMask, blockShape, attentionMask,
                   actualSeqLengths, actualSeqLengthsKv, qDescale, kDescale, vDescale, doDescale, pScale, dsScale, dq,
                   dk, dv, user, tilingDataPtrArch35, &tPipe);
    } else if (TILING_KEY_IS(1012)) {
        using bsa_type = BSA_ARC35::BSA_TYPE<bfloat16_t, BSA_ARC35::TND, ARC35_TILING_CLASS, true>;
        BSA_ARC35::BlockSparseAttentionGradArch35<bsa_type> op;
        op.Process(dout, query, key, value, out, softmaxLse, blockSparseMask, blockShape, attentionMask,
                   actualSeqLengths, actualSeqLengthsKv, qDescale, kDescale, vDescale, doDescale, pScale, dsScale, dq,
                   dk, dv, user, tilingDataPtrArch35, &tPipe);
    } else if (TILING_KEY_IS(1013)) {
        using bsa_type = BSA_ARC35::BSA_TYPE<half, BSA_ARC35::TND, ARC35_TILING_CLASS, true>;
        BSA_ARC35::BlockSparseAttentionGradArch35<bsa_type> op;
        op.Process(dout, query, key, value, out, softmaxLse, blockSparseMask, blockShape, attentionMask,
                   actualSeqLengths, actualSeqLengthsKv, qDescale, kDescale, vDescale, doDescale, pScale, dsScale, dq,
                   dk, dv, user, tilingDataPtrArch35, &tPipe);
    } else if (TILING_KEY_IS(2008)) {
        using bsa_type = BSA_ARC35::BSA_TYPE<hifloat8_t, BSA_ARC35::BSND, ARC35_TILING_CLASS, true, false, bfloat16_t,
                                             bfloat16_t, true>;
        BSA_ARC35::BlockSparseAttentionGradArch35<bsa_type> op;
        op.Process(dout, query, key, value, out, softmaxLse, blockSparseMask, blockShape, attentionMask,
                   actualSeqLengths, actualSeqLengthsKv, qDescale, kDescale, vDescale, doDescale, pScale, dsScale, dq,
                   dk, dv, user, tilingDataPtrArch35, &tPipe);
    } else if (TILING_KEY_IS(2010)) {
        using bsa_type = BSA_ARC35::BSA_TYPE<hifloat8_t, BSA_ARC35::BNSD, ARC35_TILING_CLASS, true, false, bfloat16_t,
                                             bfloat16_t, true>;
        BSA_ARC35::BlockSparseAttentionGradArch35<bsa_type> op;
        op.Process(dout, query, key, value, out, softmaxLse, blockSparseMask, blockShape, attentionMask,
                   actualSeqLengths, actualSeqLengthsKv, qDescale, kDescale, vDescale, doDescale, pScale, dsScale, dq,
                   dk, dv, user, tilingDataPtrArch35, &tPipe);
    } else if (TILING_KEY_IS(2012)) {
        using bsa_type = BSA_ARC35::BSA_TYPE<hifloat8_t, BSA_ARC35::TND, ARC35_TILING_CLASS, true, false, bfloat16_t,
                                             bfloat16_t, true>;
        BSA_ARC35::BlockSparseAttentionGradArch35<bsa_type> op;
        op.Process(dout, query, key, value, out, softmaxLse, blockSparseMask, blockShape, attentionMask,
                   actualSeqLengths, actualSeqLengthsKv, qDescale, kDescale, vDescale, doDescale, pScale, dsScale, dq,
                   dk, dv, user, tilingDataPtrArch35, &tPipe);
    }

#else
    __gm__ BlockSparseAttentionGradTilingData* tilingDataPtr =
        reinterpret_cast<__gm__ BlockSparseAttentionGradTilingData*>(tiling);
    if (TILING_KEY_IS(0)) {
        BSA::BlockSparseAttentionGradInfer<half, 0>(dout, query, key, value, out, softmaxLse, blockSparseMask,
                                                    blockShape, attentionMask, actualSeqLengths, actualSeqLengthsKv, dq,
                                                    dk, dv, user, tiling);
    } else if (TILING_KEY_IS(1)) {
        BSA::BlockSparseAttentionGradInfer<half, 1>(dout, query, key, value, out, softmaxLse, blockSparseMask,
                                                    blockShape, attentionMask, actualSeqLengths, actualSeqLengthsKv, dq,
                                                    dk, dv, user, tiling);
    } else if (TILING_KEY_IS(10)) {
        BSA::BlockSparseAttentionGradInfer<bfloat16_t, 0>(dout, query, key, value, out, softmaxLse, blockSparseMask,
                                                          blockShape, attentionMask, actualSeqLengths,
                                                          actualSeqLengthsKv, dq, dk, dv, user, tiling);
    } else if (TILING_KEY_IS(11)) {
        BSA::BlockSparseAttentionGradInfer<bfloat16_t, 1>(dout, query, key, value, out, softmaxLse, blockSparseMask,
                                                          blockShape, attentionMask, actualSeqLengths,
                                                          actualSeqLengthsKv, dq, dk, dv, user, tiling);
    }
#endif
}
