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
 * \file kv_quant_sparse_attn_sharedkv_scfa_block_vector.h
 * \brief
 */
#ifndef KV_QUANT_SPARSE_ATTN_SHAREDKV_SCFA_BLOCK_VECTOR_H
#define KV_QUANT_SPARSE_ATTN_SHAREDKV_SCFA_BLOCK_VECTOR_H

#include "util_regbase.h"
#include "kv_quant_sparse_attn_sharedkv_common_arch35.h"
#include "common/buffers_policy.h"
#include "common/buffer_manager.h"
#include "common/buffer.h"
#include "kernel_operator_list_tensor_intf.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"

#include "vf/vf_mul_sel_softmaxflashv2_cast_nz_scfa.h"
#include "vf/vf_flashupdate_new_scfa.h"

using namespace AscendC;
using namespace SCFaVectorApi;
using namespace AscendC::Impl::Detail;
using namespace optiling;
using namespace optiling::detail;
using namespace regbaseutil;
using namespace matmul;

namespace BaseApi {
constexpr uint16_t KV_PHY_ADDR_SYNC_FLAG_ID = 8;

TEMPLATES_DEF
class SCFABlockVec {
public:
    // BUFFER的字节数
    static constexpr uint32_t BUFFER_SIZE_BYTE_32B = 32;
    /* =================编译期常量的基本块信息================= */
    static constexpr uint32_t s1BaseSize = KvQuantSparseAttnSharedkv::ScfaUbConfig::S1_BASE_SIZE;
    static constexpr uint32_t s2BaseSize = 128;
    static constexpr uint32_t vec1Srcstride = (s1BaseSize >> 1) + 1;
    static constexpr uint32_t dVTemplateType = KvQuantSparseAttnSharedkv::ScfaUbConfig::D_TEMPLATE_SIZE;
    static constexpr uint32_t dTemplateAlign64 = KvQuantSparseAttnSharedkv::ScfaUbConfig::D_TEMPLATE_ALIGN64;
    static constexpr uint32_t dVTemplateTypeInput = KvQuantSparseAttnSharedkv::ScfaUbConfig::D_INPUT_SIZE;
    static constexpr uint32_t v0ProcessRowNum = KvQuantSparseAttnSharedkv::ScfaUbConfig::V0_PROCESS_ROW_NUM;
    static constexpr uint32_t combineKvInputBytes = dVTemplateTypeInput;
    static constexpr uint32_t combineKvInputAlignBytes =
        ((combineKvInputBytes + BUFFER_SIZE_BYTE_32B - 1) / BUFFER_SIZE_BYTE_32B) * BUFFER_SIZE_BYTE_32B;
    static constexpr uint32_t sparseCopyTokenNum = 8;
    static constexpr float R0 = 1.0f;

    // buffer ID 是 get_buf/rls_buf 使用的同步槽，不是 TBuf 数组下标。同步槽与物理 UB 的映射为：
    //   0/1: stage0InBuf[0/1]    2/3: stage0OutBuf[0/1]
    //   4/5: stage1OutBuf[0/1]  6:   stage2OutBuf       7: sinksBuf
    // ping-pong 下标只能是 0/1，叠加对应基址后得到该物理 UB 唯一的同步槽。
    static constexpr uint32_t stage0InBufBaseId = 0;
    static constexpr uint32_t stage0OutBufBaseId = 2;
    static constexpr uint32_t stage1OutBufBaseId = 4;
    static constexpr uint32_t stage2OutBufBufId = 6;                   // stage2OutBuf 的主流程同步槽。
    static constexpr uint32_t blkTableBufId = stage0OutBufBaseId;      // blkTableBuf 复用 stage0OutBuf[0]。
    static constexpr uint32_t sparseIdxBufId = stage0OutBufBaseId + 1; // sparseIdxBuf 复用 stage0OutBuf[1]。
    static constexpr uint32_t kvPhyAddrBufId = stage2OutBufBufId;      // kvPhyAddrBuf 复用 stage2OutBuf。
    static constexpr uint32_t syncSinksBufId = 7; // sinksBuf 使用独立同步槽，避免与流水缓冲区冲突。

    // ==================== Functions ======================
    __aicore__ inline SCFABlockVec(){};
    __aicore__ inline void InitVecBlock(TPipe *pipe, __gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *sequsedKv)
    {
        if ASCEND_IS_AIV {
            tPipe = pipe;
            if (cuSeqlensQ != nullptr) {
                cuSeqlensQGm.SetGlobalBuffer((__gm__ int32_t *)cuSeqlensQ);
            }
            if (sequsedKv != nullptr) {
                actualSeqLengthsKVGm.SetGlobalBuffer((__gm__ int32_t *)sequsedKv);
            }
            this->GetExtremeValue(this->negativeFloatScalar);
        }
    }

    // 初始化LocalTensor
    __aicore__ inline void InitLocalBuffer(TPipe *pipe, ConstInfo &constInfo);
    // 初始化attentionOutGM
    __aicore__ inline void CleanOutput(__gm__ uint8_t *attentionOut, ConstInfo &constInfo);
    __aicore__ inline void InitGlobalBuffer(__gm__ uint8_t *oriKV, __gm__ uint8_t *cmpKV,
                                            __gm__ uint8_t *cmpSparseIndices, __gm__ uint8_t *oriBlockTable,
                                            __gm__ uint8_t *cmpBlockTable, __gm__ uint8_t *sequsedQ,
                                            __gm__ uint8_t *sinks);
    __aicore__ inline void InitOutputSingleCore(ConstInfo &constInfo);
    __aicore__ inline void ProcessVec0(Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
                                       const RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline void ProcessVec1(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputBuf,
                                       Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf,
                                       RunInfo &runInfo, ConstInfo &constInfo);
    using mm2ResPos = Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>;
    __aicore__ inline void ProcessVec2(mm2ResPos &bmm2ResBuf, RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline void GetKVPhyAddr(uint32_t hasLoad, uint32_t bN2StartIdx, uint32_t bN2EndIdx,
                                        uint32_t gS1StartIdx, uint32_t nextGs1Idx, __gm__ int32_t *actualSeqQlenAddr,
                                        __gm__ int32_t *cuSeqlensQAddr, __gm__ int32_t *actualSeqKvlenAddr,
                                        __gm__ uint8_t *workspace, ConstInfo &constInfo);
    __aicore__ inline void SetKvPhyAddrReady();
    __aicore__ inline void WaitKvPhyAddrBeforeRead();

private:
    __aicore__ inline void ProcessSparseKv(Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
                                           const RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline void ProcessNotSparseKv(Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
                                              const RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline void CalProcSize(const RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline int64_t GetkeyOffset(int64_t s2Idx, const RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline void GetRealCmpS2Idx(int64_t *tokenData, int64_t s2IdxInBase, uint64_t topkBaseIdx,
                                           uint64_t topkKBase, uint32_t sparseBlockCount, int64_t procS2EndLocal);
    __aicore__ inline void GetRealS2Addr(int64_t *tokenData, int64_t s2IdxInBase, uint64_t topkBaseIdx,
                                         uint64_t topkKBase, uint32_t sparseBlockCount, int64_t procS2EndLocal);
    __aicore__ inline void CopyInKvNotSparse(LocalTensor<KV_T> kvMergUb, int64_t dealRow, int64_t s2StartIdx,
                                             const RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline uint32_t CopyInKvSparse(LocalTensor<KV_T> kvInUb, int64_t startRow, int64_t *tokenData,
                                              const RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline void DequantKv(LocalTensor<Q_T> antiKvTensorAsB16, LocalTensor<KV_T> srcTensor, int64_t dealRow,
                                     ConstInfo &constInfo);
    __aicore__ inline void CopyOutKvUb2Gm(Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
                                          LocalTensor<Q_T> antiKvTensorAsB16, int64_t dealRow, int64_t s2StartIdx,
                                          const RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline void CopyInSingleKv(LocalTensor<KV_T> kvInUb, int64_t startRow, int64_t keyOffset);
    /* VEC2_RES_T 表示bmm2ResUb当前的类型，VEC2_RES_T = Q_T那么不需要做Cast。另外，无效行场景当前默认需要做Cast */
    using VEC2_RES_T = T;
    template <typename VEC2_RES_T>
    __aicore__ inline void Bmm2DataCopyOut(RunInfo &runInfo, ConstInfo &constInfo, LocalTensor<VEC2_RES_T> &vec2ResUb,
                                           int64_t vec2S1Idx, int64_t vec2CalcSize = 0);
    template <typename VEC2_RES_T>
    __aicore__ inline void CopyOutAttentionOut(RunInfo &runInfo, ConstInfo &constInfo,
                                               LocalTensor<VEC2_RES_T> &vec2ResUb, int64_t vec2S1Idx,
                                               int64_t vec2CalcSize);
    __aicore__ inline void SoftmaxInitBuffer();
    __aicore__ inline void GetExtremeValue(T &negativeScalar);
    __aicore__ inline void InitSinksBuffer(ConstInfo &constInfo);
    __aicore__ inline void CopyPhyAddrToGm(LocalTensor<uint32_t> kvPhyAddrUb, int64_t bS1Idx, int64_t s1Idx,
                                           int64_t validS2, int64_t alignNum, ConstInfo &constInfo);
    __aicore__ inline void CopyPaTableToUb(LocalTensor<int32_t> blkTableUb, int64_t bIdx, ConstInfo &constInfo);
    __aicore__ inline void CopySparseIdxToUb(LocalTensor<int32_t> sparseIdxUb, int64_t bS1Idx, int64_t s1Idx,
                                             int64_t validS2, ConstInfo &constInfo);
    __aicore__ inline int32_t GetActualS1Size(uint32_t bIdx, __gm__ int32_t *actualSeqQlenAddr,
                                              __gm__ int32_t *cuSeqlensQAddr, const ConstInfo &constInfo);
    __aicore__ inline int32_t GetActualS2Size(uint32_t bIdx, __gm__ int32_t *actualSeqKvlenAddr,
                                              const ConstInfo &constInfo);

    TPipe *tPipe;

    GlobalTensor<OUTPUT_T> attentionOutGm;
    GlobalTensor<KV_T> oriKVGm;
    GlobalTensor<KV_T> cmpKVGm;
    GlobalTensor<KV_T> keyGm;
    GlobalTensor<int32_t> cmpSparseIndicesGm;
    GlobalTensor<int32_t> oriBlockTableGm;
    GlobalTensor<int32_t> cmpBlockTableGm;
    GlobalTensor<int32_t> blockTableGm;
    GlobalTensor<T> sinksGm;
    GlobalTensor<int32_t> cuSeqlensQGm;
    GlobalTensor<int32_t> actualSeqLengthsKVGm;
    GlobalTensor<uint32_t> kvPhyAddrGm;

    TBuf<> commonTBuf; // common的复用空间
    TBuf<> sinksBuf;
    TBuf<> stage1OutBuf[2]; // 2份表示可能存在pingpong
    TBuf<> stage2OutBuf;
    TBuf<> softmaxMaxBuf[2];
    TBuf<> softmaxSumBuf[2];
    TBuf<> softmaxExpBuf[2];
    TBuf<> dequantScaleBuff;
    TBuf<> stage0InBuf[2];
    TBuf<> stage0OutBuf[2];
    uint32_t pingPongV0 = 0;

    T negativeFloatScalar;
    bool isSinks = false;
    bool isKvPhyAddrSynced = false;
    uint32_t maxBlockNumPerBatch;
    uint32_t blockSize;
    int64_t procSize;
    int64_t procS2Start;
    int64_t procS2End;
};

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::GetRealCmpS2Idx(int64_t *tokenData, int64_t s2IdxInBase,
                                                                    uint64_t topkBaseIdx, uint64_t topkKBase,
                                                                    uint32_t sparseBlockCount, int64_t procS2EndLocal)
{
    uint64_t topkKIdx = s2IdxInBase + topkKBase;
    for (uint64_t i = 0; i < sparseCopyTokenNum; ++i) {
        uint64_t idx = topkBaseIdx + s2IdxInBase + i;
        if (likely((topkKIdx + i < sparseBlockCount) && (s2IdxInBase + i < procS2EndLocal))) {
            tokenData[i] = cmpSparseIndicesGm.GetValue(idx);
        } else {
            break;
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::SetKvPhyAddrReady()
{
    if constexpr (TEMPLATE_MODE == SASTemplateMode::SCFA_TEMPLATE_MODE && IS_VEC_S2PHYADDR) {
        CrossCoreSetFlag<0, PIPE_MTE3>(KV_PHY_ADDR_SYNC_FLAG_ID);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::WaitKvPhyAddrBeforeRead()
{
    if constexpr (TEMPLATE_MODE == SASTemplateMode::SCFA_TEMPLATE_MODE && IS_VEC_S2PHYADDR) {
        if (unlikely(!isKvPhyAddrSynced)) {
            CrossCoreWaitFlag<0, PIPE_S>(KV_PHY_ADDR_SYNC_FLAG_ID);
            isKvPhyAddrSynced = true;
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::GetRealS2Addr(int64_t *tokenData, int64_t s2IdxInBase,
                                                                  uint64_t topkBaseIdx, uint64_t topkKBase,
                                                                  uint32_t sparseBlockCount, int64_t procS2EndLocal)
{
    uint64_t topkKIdx = s2IdxInBase + topkKBase;
    GlobalTensor<int64_t> kvPhyAddrGm64 = kvPhyAddrGm.template ReinterpretCast<int64_t>();
    for (uint64_t i = 0; i < sparseCopyTokenNum; ++i) {
        uint64_t idx = topkBaseIdx + s2IdxInBase + i;
        if (likely((topkKIdx + i < sparseBlockCount) && (s2IdxInBase + i < procS2EndLocal))) {
            tokenData[i] = kvPhyAddrGm64.GetValue(idx);
        } else {
            break;
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline int64_t SCFABlockVec<TEMPLATE_ARGS>::GetkeyOffset(int64_t s2Idx, const RunInfo &runInfo,
                                                                    ConstInfo &constInfo)
{
    if (s2Idx < 0) {
        return -1;
    }
    int64_t realkeyOffset = 0;
    if constexpr (isPa) {
        int64_t blkTableIdx = s2Idx / blockSize;
        int64_t blkTableOffset = s2Idx % blockSize;
        int64_t paBlockStride = runInfo.isCmp ? constInfo.cmpKvStride : constInfo.oriKvStride;
        realkeyOffset = blockTableGm.GetValue(runInfo.boIdx * maxBlockNumPerBatch + blkTableIdx) * paBlockStride +
                        blkTableOffset * combineKvInputBytes; // BlockNum, BlockSize, N(1), D
    } else {
        realkeyOffset = runInfo.boIdx * constInfo.s2Size + s2Idx; // BSN(1)D
    }
    return realkeyOffset;
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::CopyInSingleKv(LocalTensor<KV_T> kvInUb, int64_t startRow,
                                                                   int64_t keyOffset)
{
    if (keyOffset < 0) {
        return;
    }
    DataCopyExtParams intriParams;

    intriParams.blockCount = 1;
    intriParams.dstStride = 0;
    intriParams.srcStride = 0;
    DataCopyPadExtParams<KV_T> padParams;
    constexpr uint32_t combineBytes = combineKvInputBytes;
    constexpr uint32_t combineDim = combineKvInputBytes / sizeof(KV_T);
    constexpr uint32_t combineDimAlign = combineKvInputAlignBytes / sizeof(KV_T);
    intriParams.blockLen = combineBytes;
    padParams.isPad = true;
    padParams.leftPadding = 0;
    padParams.rightPadding = combineDimAlign - combineDim;
    padParams.paddingValue = 0;
    DataCopyPad(kvInUb[startRow * combineDimAlign], keyGm[keyOffset], intriParams, padParams);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline uint32_t SCFABlockVec<TEMPLATE_ARGS>::CopyInKvSparse(LocalTensor<KV_T> kvInUb, int64_t startRow,
                                                                       int64_t *tokenData, const RunInfo &runInfo,
                                                                       ConstInfo &constInfo)
{
    // 当前仅支持COMBINE模式
    constexpr uint32_t combineBytes = combineKvInputBytes;
    constexpr uint32_t combineDim = combineKvInputBytes / sizeof(KV_T);
    constexpr uint32_t combineDimAlign = combineKvInputAlignBytes / sizeof(KV_T);
    const uint32_t sparseBlockSize = constInfo.sparseBlockSize;
    DataCopyExtParams intriParams;
    intriParams.blockLen = combineBytes;
    intriParams.dstStride = 0;
    DataCopyPadExtParams<KV_T> padParams;
    padParams.isPad = true;
    padParams.leftPadding = 0;
    padParams.rightPadding = combineDimAlign - combineDim;
    padParams.paddingValue = 0;
    uint32_t dealRow = 0;
    for (uint32_t i = 0; i < sparseCopyTokenNum; i += 2) {
        int64_t keyOffset0 = 0;
        int64_t keyOffset1 = 0;
        if constexpr (IS_VEC_S2PHYADDR) {
            keyOffset0 = tokenData[i];
            keyOffset1 = tokenData[i + 1];
        } else {
            keyOffset0 = GetkeyOffset(tokenData[i], runInfo, constInfo);
            keyOffset1 = GetkeyOffset(tokenData[i + 1], runInfo, constInfo);
        }

        if (unlikely(keyOffset0 < 0 && keyOffset1 < 0)) {
            return dealRow;
        }
        int64_t keySrcStride =
            (keyOffset0 > keyOffset1 ? (keyOffset0 - keyOffset1) : (keyOffset1 - keyOffset0)) - combineBytes;
        if (unlikely(keySrcStride >= INT32_MAX || keySrcStride < 0) || sparseBlockSize > 1) {
            // stride溢出、stride为负数、s2超长等异常场景，还原成2条搬运指令
            CopyInSingleKv(kvInUb, startRow, keyOffset0);
            CopyInSingleKv(kvInUb, startRow + 1, keyOffset1);
        } else {
            intriParams.blockCount = (keyOffset0 >= 0) + (keyOffset1 >= 0);
            intriParams.srcStride = keySrcStride;

            int64_t keyOffset = keyOffset0 > -1 ? keyOffset0 : keyOffset1;
            if (keyOffset1 > -1 && keyOffset1 < keyOffset0) {
                keyOffset = keyOffset1;
            }

            DataCopyPad(kvInUb[startRow * combineDimAlign], keyGm[keyOffset], intriParams, padParams);
        }
        dealRow += (keyOffset0 >= 0) + (keyOffset1 >= 0);
        startRow += 2;
    }
    return dealRow;
}

// fp8->fp32
static constexpr MicroAPI::CastTrait castTraitFp8_1 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                       MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
// fp8->fp32
static constexpr MicroAPI::CastTrait castTraitFp8_2 = {MicroAPI::RegLayout::ONE, MicroAPI::SatMode::UNKNOWN,
                                                       MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
// fp32->fp16
static constexpr MicroAPI::CastTrait castTraitFp8_3 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                                                       MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
// fp32->fp16
static constexpr MicroAPI::CastTrait castTraitFp8_4 = {MicroAPI::RegLayout::ONE, MicroAPI::SatMode::NO_SAT,
                                                       MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
template <typename Q_T, typename KV_T>
__simd_vf__ void CastScaleImpl(__ubuf__ float *ubDstAddr, __ubuf__ int8_t *ubSrcAddr, uint32_t dealRowCount)
{
    MicroAPI::RegTensor<fp8_e8m0_t> vScale0;
    MicroAPI::RegTensor<fp8_e8m0_t> vScale1;
    MicroAPI::RegTensor<bfloat16_t> vScalebf16Res0;
    MicroAPI::RegTensor<bfloat16_t> vScalebf16Res1;
    MicroAPI::RegTensor<float> vScalefp32Res0;
    MicroAPI::RegTensor<float> vScalefp32Res1;
    __ubuf__ int8_t *ubScaleSrcAddrTemp = ubSrcAddr;
    __ubuf__ float *ubDstAddrTmp = ubDstAddr;
    MicroAPI::MaskReg bf16TypeMaskAll = MicroAPI::CreateMask<bfloat16_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg fp32MaskAll = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    for (uint16_t i = 0; i < static_cast<uint16_t>(dealRowCount); i++) {
        // load scale
        MicroAPI::LoadAlign<int8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
            (MicroAPI::RegTensor<int8_t> &)vScale0, ubScaleSrcAddrTemp, 640);

        MicroAPI::Cast<bfloat16_t, fp8_e8m0_t, castTraitFp8_1>(vScalebf16Res0, vScale0, bf16TypeMaskAll);
        MicroAPI::Cast<float, bfloat16_t, castTraitFp8_1>(vScalefp32Res0, vScalebf16Res0, fp32MaskAll);

        MicroAPI::StoreAlign<float, MicroAPI::PostLiteral::POST_MODE_UPDATE>(ubDstAddrTmp, vScalefp32Res0, 64,
                                                                             bf16TypeMaskAll);
    }
}

template <typename Q_T, typename KV_T>
__aicore__ inline void CastScale(LocalTensor<float> &outputUb, LocalTensor<KV_T> &inputUb, uint32_t dealRowCount)
{
    __ubuf__ float *ubDstAddr = (__ubuf__ float *)(outputUb.GetPhyAddr());
    __ubuf__ int8_t *ubScaleAddr =
        (__ubuf__ int8_t *)(inputUb[448 + 64 * 2].GetPhyAddr()); // 448 for nope, 64 for rope, 2 for sizeof(bf16)

    CastScaleImpl<Q_T, KV_T>(ubDstAddr, ubScaleAddr, dealRowCount);
}

template <typename Q_T, typename KV_T>
__simd_vf__ void AntiquantVFImplFp8D448(__ubuf__ Q_T *ubKRopeNzAddr, __ubuf__ int8_t *ubSrcAddr,
                                        __ubuf__ Q_T *ubDstAddr, __ubuf__ bfloat16_t *ubScaleSrcAddr,
                                        __ubuf__ int8_t *ubKRopeAddr, uint32_t dealRowCount)
{
    MicroAPI::RegTensor<KV_T> vKvData0;
    MicroAPI::RegTensor<KV_T> vKvData1;
    MicroAPI::RegTensor<Q_T> vScale0Bf16;
    MicroAPI::RegTensor<Q_T> vScale1Bf16;
    MicroAPI::RegTensor<half> vKvDataHalf0;
    MicroAPI::RegTensor<half> vKvDataHalf1;
    MicroAPI::RegTensor<float> vCastFp32Res0;
    MicroAPI::RegTensor<float> vCastFp32Res1;
    MicroAPI::RegTensor<Q_T> vMulRes0;
    MicroAPI::RegTensor<Q_T> vMulRes1;
    MicroAPI::RegTensor<float> vScale0;
    MicroAPI::RegTensor<float> vScale1;
    MicroAPI::RegTensor<Q_T> vCastRes0;
    MicroAPI::RegTensor<Q_T> vCastRes1;
    MicroAPI::RegTensor<Q_T> vCastResPack0;
    MicroAPI::RegTensor<Q_T> vCastResPack1;
    MicroAPI::RegTensor<int8_t> vKvRope;

    MicroAPI::MaskReg kvTypeMaskAll = MicroAPI::CreateMask<KV_T, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg kvRopeTypeMaskAll = MicroAPI::CreateMask<Q_T, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg kvRopeTypeMaskHalf = MicroAPI::CreateMask<Q_T, MicroAPI::MaskPattern::H>();
    MicroAPI::MaskReg fp32MaskAll = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg fp16MaskAll = MicroAPI::CreateMask<bfloat16_t, MicroAPI::MaskPattern::ALL>();
    uint32_t combineDim = 640;     // 128对齐
    uint32_t scaleRowStride = 320; // combineDim / 2，scale数据的行步长
    uint32_t blockStride = 17;     // +1 to solve bank confict
    uint32_t repeatStride = 1;
    const uint32_t nopeDim = 448;
    const uint32_t kvNumPerLoop = 128;
    const uint32_t scaleNumPerLoop = 2;
    const uint32_t tileSize = 64;

    // tilesize is 64, deal 128 b8 kv, deal 2 fp32 scale
    for (uint16_t j = 0; j < (nopeDim / kvNumPerLoop); j++) {
        __ubuf__ int8_t *ubSrcTemp = ubSrcAddr + j * kvNumPerLoop;
        __ubuf__ Q_T *ubScaleSrcAddrTemp = ubScaleSrcAddr + j * scaleNumPerLoop;
        __ubuf__ Q_T *ubDstAddrTmp = ubDstAddr + j * kvNumPerLoop * blockStride;
        for (uint16_t i = 0; i < static_cast<uint16_t>(dealRowCount); i++) {
            // load scale
            MicroAPI::LoadAlign<int8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                (MicroAPI::RegTensor<int8_t> &)vKvData0, ubSrcTemp, tileSize);
            MicroAPI::LoadAlign<int8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
                (MicroAPI::RegTensor<int8_t> &)vKvData1, ubSrcTemp, combineDim - tileSize);

            // bf16 brc 128
            MicroAPI::LoadAlign<Q_T, MicroAPI::LoadDist::DIST_BRC_B16>((MicroAPI::RegTensor<Q_T> &)vScale0Bf16,
                                                                       ubScaleSrcAddrTemp + i * scaleRowStride);
            MicroAPI::LoadAlign<Q_T, MicroAPI::LoadDist::DIST_BRC_B16>((MicroAPI::RegTensor<Q_T> &)vScale1Bf16,
                                                                       ubScaleSrcAddrTemp + 1 + i * scaleRowStride);

            MicroAPI::Cast<float, KV_T, castTraitFp8_1>(vCastFp32Res0, vKvData0, fp32MaskAll);
            MicroAPI::Cast<float, KV_T, castTraitFp8_1>(vCastFp32Res1, vKvData1, fp32MaskAll);

            MicroAPI::Cast<Q_T, float, castTraitFp8_3>(vCastRes0, vCastFp32Res0, fp16MaskAll);
            MicroAPI::Cast<Q_T, float, castTraitFp8_3>(vCastRes1, vCastFp32Res1, fp16MaskAll);

            MicroAPI::Mul<Q_T, MicroAPI::MaskMergeMode::ZEROING>(vMulRes0, vCastRes0, vScale0Bf16, fp16MaskAll);
            MicroAPI::Mul<Q_T, MicroAPI::MaskMergeMode::ZEROING>(vMulRes1, vCastRes1, vScale1Bf16, fp16MaskAll);

            MicroAPI::DeInterleave(vCastResPack0, vCastResPack1, vMulRes0, vMulRes1);

            MicroAPI::StoreAlign<Q_T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                ubDstAddrTmp, vCastResPack0, blockStride, repeatStride, kvRopeTypeMaskAll);
        }
    }

    uint16_t lastLoopOffset = nopeDim / kvNumPerLoop; // 偏移已经处理的循环次数
    __ubuf__ int8_t *ubSrcTemp = ubSrcAddr + lastLoopOffset * kvNumPerLoop;
    __ubuf__ Q_T *ubScaleSrcAddrTemp = ubScaleSrcAddr + lastLoopOffset * scaleNumPerLoop;
    __ubuf__ Q_T *ubDstAddrTmp = ubDstAddr + lastLoopOffset * kvNumPerLoop * blockStride;
    MicroAPI::Duplicate(vCastRes1, 0.0);
    for (uint16_t i = 0; i < static_cast<uint16_t>(dealRowCount); i++) {
        MicroAPI::LoadAlign<int8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_NORM>(
            (MicroAPI::RegTensor<int8_t> &)vKvRope, ubKRopeAddr, combineDim);
        // load scale
        MicroAPI::LoadAlign<int8_t, MicroAPI::PostLiteral::POST_MODE_UPDATE, MicroAPI::LoadDist::DIST_UNPACK4_B8>(
            (MicroAPI::RegTensor<int8_t> &)vKvData0, ubSrcTemp, combineDim);
        MicroAPI::LoadAlign<Q_T, MicroAPI::LoadDist::DIST_BRC_B16>((MicroAPI::RegTensor<Q_T> &)vScale0Bf16,
                                                                   ubScaleSrcAddrTemp + i * scaleRowStride);
        MicroAPI::Cast<float, KV_T, castTraitFp8_1>(vCastFp32Res0, vKvData0, fp32MaskAll);
        MicroAPI::Cast<Q_T, float, castTraitFp8_3>(vCastRes0, vCastFp32Res0, fp16MaskAll);
        MicroAPI::Mul<Q_T, MicroAPI::MaskMergeMode::ZEROING>(vMulRes0, vCastRes0, vScale0Bf16, fp16MaskAll);
        MicroAPI::DeInterleave(vCastResPack0, vCastResPack1, vMulRes0, vCastRes1);

        MicroAPI::StoreAlign<Q_T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            ubDstAddrTmp, vCastResPack0, blockStride, repeatStride, kvRopeTypeMaskHalf);
        MicroAPI::StoreAlign<Q_T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            ubKRopeNzAddr, (MicroAPI::RegTensor<Q_T> &)vKvRope, blockStride, repeatStride, kvRopeTypeMaskHalf);
    }
}

template <typename Q_T, typename KV_T>
__aicore__ inline void AntiquantVFFp8D448(LocalTensor<Q_T> &kRopeUbNz, LocalTensor<Q_T> &outputUb,
                                          LocalTensor<KV_T> &inputUb, LocalTensor<Q_T> &scaleUb,
                                          LocalTensor<int8_t> &kRopeUb, uint32_t dealRowCount)
{
    __ubuf__ int8_t *ubSrcAddr = (__ubuf__ int8_t *)(inputUb[64 * sizeof(Q_T)].GetPhyAddr());
    __ubuf__ Q_T *ubScaleAddr = (__ubuf__ Q_T *)(scaleUb[64 + 448 / 2].GetPhyAddr());
    __ubuf__ int8_t *ubKRopeAddr = (__ubuf__ int8_t *)(kRopeUb.GetPhyAddr());
    __ubuf__ Q_T *ubDstAddr = (__ubuf__ Q_T *)(outputUb.GetPhyAddr());
    __ubuf__ Q_T *ubKRopeNzAddr = (__ubuf__ Q_T *)(kRopeUbNz.GetPhyAddr());

    AntiquantVFImplFp8D448<Q_T, KV_T>(ubKRopeNzAddr, ubSrcAddr, ubDstAddr, ubScaleAddr, ubKRopeAddr, dealRowCount);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::DequantKv(LocalTensor<Q_T> antiKvTensorAsB16,
                                                              LocalTensor<KV_T> srcTensor, int64_t dealRow,
                                                              ConstInfo &constInfo)
{
    // srcTensor是rope(448) + nope(64) + scale + pad, dstTensor是nope(448) + rope(64)
    LocalTensor<int8_t> kRopeUb = srcTensor.template ReinterpretCast<int8_t>();
    LocalTensor<Q_T> kRopeUbNz = antiKvTensorAsB16[constInfo.dSizeNope * (16 + 1)]; // V0单次处理16行数据
    LocalTensor<Q_T> scaleUb = srcTensor.template ReinterpretCast<bfloat16_t>();
    AntiquantVFFp8D448<Q_T, KV_T>(kRopeUbNz, antiKvTensorAsB16, srcTensor, scaleUb, kRopeUb, dealRow);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::CopyOutKvUb2Gm(
    Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm, LocalTensor<Q_T> antiKvTensorAsB16,
    int64_t dealRow, int64_t s2StartIdx, const RunInfo &runInfo, ConstInfo &constInfo)
{
    GlobalTensor<Q_T> v0ResGmTensor = v0ResGm.template GetTensor<Q_T>();
    uint64_t blockElementNum = 16;
    const int64_t s2RealSize = runInfo.s2RealSize;
    DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = (constInfo.dSizeNope + constInfo.dSizeRope) / blockElementNum;
    dataCopyParams.blockLen = dealRow;
    dataCopyParams.srcGap = blockElementNum + 1 - dealRow;
    dataCopyParams.dstGap = Align16Func(s2RealSize) - dealRow;
    DataCopy(v0ResGmTensor[s2StartIdx * blockElementNum], antiKvTensorAsB16, dataCopyParams);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::ProcessNotSparseKv(
    Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm, const RunInfo &runInfo, ConstInfo &constInfo)
{
    const int64_t procSizeLocal = procSize;
    if (procSizeLocal == 0) {
        return;
    }
    int64_t v0ProcessBase = 16;
    int64_t v0ProcessSize = v0ProcessBase;
    int64_t loopTimes = (procSizeLocal + v0ProcessBase - 1) / v0ProcessBase;
    for (uint32_t i = 0; i < loopTimes; i++) {
        int64_t s2StartIdx = procS2Start + i * v0ProcessBase;
        if (i == loopTimes - 1) {
            v0ProcessSize = procSizeLocal - i * v0ProcessBase;
        }

        const uint32_t pp = pingPongV0;
        const uint32_t stage0InId = stage0InBufBaseId + pp;
        const uint32_t stage0OutId = stage0OutBufBaseId + pp;
        get_buf(PIPE_MTE2, stage0InId, false);
        rls_buf(PIPE_MTE2, stage0InId, false);
        LocalTensor<KV_T> kvInUb = stage0InBuf[pp].Get<KV_T>();
        CopyInKvNotSparse(kvInUb, v0ProcessSize, s2StartIdx, runInfo, constInfo);
        get_buf(PIPE_MTE2, stage0InId, true);
        rls_buf(PIPE_MTE2, stage0InId, true);
        get_buf(PIPE_V, stage0InId, false);
        rls_buf(PIPE_V, stage0InId, false);
        // 2、dequant by vf
        get_buf(PIPE_V, stage0OutId, false);
        rls_buf(PIPE_V, stage0OutId, false);
        LocalTensor<Q_T> kvDequantOutUb = stage0OutBuf[pp].Get<Q_T>();
        DequantKv(kvDequantOutUb, kvInUb, v0ProcessSize, constInfo);
        get_buf(PIPE_V, stage0InId, true);
        rls_buf(PIPE_V, stage0InId, true);

        // 3、copy kv out, ub -> l1
        get_buf(PIPE_V, stage0OutId, true);
        rls_buf(PIPE_V, stage0OutId, true);
        get_buf(PIPE_MTE3, stage0OutId, false);
        rls_buf(PIPE_MTE3, stage0OutId, false);
        CopyOutKvUb2Gm(v0ResGm, kvDequantOutUb, v0ProcessSize, s2StartIdx, runInfo, constInfo);
        get_buf(PIPE_MTE3, stage0OutId, true);
        rls_buf(PIPE_MTE3, stage0OutId, true);
        pingPongV0 ^= 1;
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::CopyInKvNotSparse(LocalTensor<KV_T> kvMergUb, int64_t dealRow,
                                                                      int64_t s2StartOffset, const RunInfo &runInfo,
                                                                      ConstInfo &constInfo)
{
    const int64_t runS2LoopCount = runInfo.s2LoopCount;
    const int64_t oriKvLoopEndIdx = runInfo.oriKvLoopEndIdx;
    const int64_t s2BaseSizeLocal = constInfo.s2BaseSize;
    const int64_t s2StartIdx = runInfo.s2StartIdx;
    int64_t s2LoopCount = (runS2LoopCount >= oriKvLoopEndIdx) ? (runS2LoopCount - oriKvLoopEndIdx) : runS2LoopCount;
    int64_t s2Idx = s2StartOffset + s2LoopCount * s2BaseSizeLocal + s2StartIdx;
    constexpr uint32_t combineBytes = combineKvInputBytes;
    constexpr uint32_t combineDim = combineKvInputBytes / sizeof(KV_T);
    constexpr uint32_t combineDimAlign = combineKvInputAlignBytes / sizeof(KV_T);
    DataCopyExtParams intriParams;
    intriParams.blockCount = dealRow;
    intriParams.blockLen = combineBytes;
    intriParams.dstStride = 0;
    intriParams.srcStride = 0;
    DataCopyPadExtParams<KV_T> padParams;
    padParams.isPad = true;
    padParams.leftPadding = 0;
    padParams.rightPadding = combineDimAlign - combineDim;
    padParams.paddingValue = 0;
    if constexpr (isPa) {
        const uint64_t blockTableBaseOffset = runInfo.boIdx * maxBlockNumPerBatch;
        const uint32_t blockSizeLocal = blockSize;
        const uint64_t n2Size = constInfo.n2Size;
        const uint64_t n2oIdx = runInfo.n2oIdx;
        const int64_t paBlockStride = runInfo.isCmp ? constInfo.cmpKvStride : constInfo.oriKvStride;
        uint64_t dstOffset = 0;
        uint32_t copyFinishElmenCnt = 0;
        uint32_t curSequence = s2Idx;
        while (copyFinishElmenCnt < dealRow) {
            uint64_t blockIdOffset = curSequence / blockSizeLocal;
            uint64_t remainElmenCnt = curSequence % blockSizeLocal;
            uint64_t idInBlockTable = blockTableGm.GetValue(blockTableBaseOffset + blockIdOffset);
            uint32_t copyElmenCnt = blockSizeLocal - remainElmenCnt;
            if (copyElmenCnt + copyFinishElmenCnt > dealRow) {
                copyElmenCnt = dealRow - copyFinishElmenCnt;
            }
            uint64_t srcOffset = idInBlockTable * paBlockStride + remainElmenCnt * n2Size * combineBytes +
                                 static_cast<uint64_t>(n2oIdx * combineBytes); // BlockNum, BlockSize, N, D
            intriParams.blockCount = copyElmenCnt;                             // base s2 size
            DataCopyPad(kvMergUb[dstOffset * combineDimAlign], keyGm[srcOffset], intriParams, padParams);
            dstOffset += copyElmenCnt;
            copyFinishElmenCnt += copyElmenCnt;
            curSequence += copyElmenCnt;
        }
    } else {
        DataCopyPad(kvMergUb, keyGm[s2Idx * combineDim], intriParams, padParams);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::CalProcSize(const RunInfo &runInfo, ConstInfo &constInfo)
{
    if constexpr (IS_SPLIT_G) {
        uint32_t aicIdx = constInfo.aivIdx >> 1U;
        uint32_t v0S2SizeFirstCore = CeilDiv(runInfo.s2RealSize, 2);
        uint32_t v0S2SizeSecondCore = runInfo.s2RealSize - v0S2SizeFirstCore;
        int32_t vecCnt = (aicIdx % 2U == 0) ? (GetSubBlockIdx() == 0 ? 0 : 1) : (GetSubBlockIdx() == 0 ? 2 : 3);
        if (aicIdx % 2U == 0) {
            if (GetSubBlockIdx() == 0) {
                procSize = CeilDiv(v0S2SizeFirstCore, 2);
                procS2Start = 0;
            } else {
                procSize = v0S2SizeFirstCore - CeilDiv(v0S2SizeFirstCore, 2);
                procS2Start = CeilDiv(v0S2SizeFirstCore, 2);
            }
        } else {
            if (GetSubBlockIdx() == 0) {
                procSize = CeilDiv(v0S2SizeSecondCore, 2);
                procS2Start = v0S2SizeFirstCore;
            } else {
                procSize = v0S2SizeSecondCore - CeilDiv(v0S2SizeSecondCore, 2);
                procS2Start = v0S2SizeFirstCore + CeilDiv(v0S2SizeSecondCore, 2);
            }
        }
        procS2End = procS2Start + procSize;
    } else {
        int64_t s2PerVecLoop = 2LL;
        int64_t vecNum = 2LL;
        int64_t s2Loops = CeilDiv(CeilDiv(runInfo.s2RealSize, vecNum), s2PerVecLoop);
        procS2Start = GetSubBlockIdx() == 0 ? 0 : Min(s2Loops * s2PerVecLoop, runInfo.s2RealSize);
        procS2End = GetSubBlockIdx() == 0 ? Min(s2Loops * s2PerVecLoop, runInfo.s2RealSize) : runInfo.s2RealSize;
        procSize = procS2End - procS2Start;
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::ProcessVec0(
    Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm, const RunInfo &runInfo, ConstInfo &constInfo)
{
    bool isCmp = runInfo.s2LoopCount >= runInfo.oriKvLoopEndIdx;
    if (isCmp) {
        keyGm = cmpKVGm;
        blockTableGm = cmpBlockTableGm;
        blockSize = constInfo.cmpBlockSize;
        maxBlockNumPerBatch = constInfo.cmpMaxBlockNumPerBatch;
    } else {
        keyGm = oriKVGm;
        blockTableGm = oriBlockTableGm;
        blockSize = constInfo.oriBlockSize;
        maxBlockNumPerBatch = constInfo.oriMaxBlockNumPerBatch;
    }
    CalProcSize(runInfo, constInfo);
    if constexpr (TEMPLATE_MODE == SASTemplateMode::SCFA_TEMPLATE_MODE) {
        if (isCmp) {
            ProcessSparseKv(v0ResGm, runInfo, constInfo);
        } else {
            ProcessNotSparseKv(v0ResGm, runInfo, constInfo);
        }
    } else {
        ProcessNotSparseKv(v0ResGm, runInfo, constInfo);
    }

    v0ResGm.SetCrossCore();
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::ProcessSparseKv(
    Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm, const RunInfo &runInfo, ConstInfo &constInfo)
{
    // 所有进入压缩KV路径的AIV在提前返回前消费一次地址就绪flag。
    WaitKvPhyAddrBeforeRead();
    const int64_t procSizeLocal = procSize;
    if (procSizeLocal == 0) {
        return;
    }

    // Left-closed, right-open interval
    // 4x = 2x + 2x
    // 4x + 1 = (2x + 2) + (2x - 1)
    // 4x + 2 = (2x + 2) + (2x)
    // 4x + 3 = (2x + 2) + (2x + 1)
    bool meetEnd = false;
    const int64_t procS2EndLocal = procS2End;
    int64_t s2Start = procS2Start;
    int64_t s2 = procS2Start;
    const uint32_t sparseBlockCount = constInfo.sparseBlockCount;
    const int64_t s2LoopCount = runInfo.s2LoopCount;
    const int64_t oriKvLoopEndIdx = runInfo.oriKvLoopEndIdx;
    const uint64_t topkKBase =
        static_cast<uint64_t>(s2LoopCount - oriKvLoopEndIdx) * static_cast<uint64_t>(constInfo.s2BaseSize);
    uint64_t topkBS1Idx = 0;
    if constexpr (LAYOUT_T == SAS_LAYOUT::TND) {
        uint64_t actualSeqQPrefixSum = cuSeqlensQGm.GetValue(runInfo.boIdx);
        topkBS1Idx += (actualSeqQPrefixSum + runInfo.s1oIdx) * sparseBlockCount; // T, N2(1), K
    } else {
        topkBS1Idx +=
            runInfo.boIdx * constInfo.s1Size * sparseBlockCount + runInfo.s1oIdx * sparseBlockCount; // B, S1, N2(1), K
    }
    const uint64_t topkBaseIdx = topkBS1Idx + runInfo.s2StartIdx + topkKBase;
    // 处理一个s2的base块
    while ((s2 < procS2EndLocal) && !meetEnd) { // 拷贝到s2End或者遇到-1
        int64_t dealRow = 0;
        const uint32_t pp = pingPongV0;
        const uint32_t stage0InId = stage0InBufBaseId + pp;
        const uint32_t stage0OutId = stage0OutBufBaseId + pp;
        // 1、copy kv in, gm ->ub
        get_buf(PIPE_MTE2, stage0InId, false);
        rls_buf(PIPE_MTE2, stage0InId, false);
        LocalTensor<KV_T> kvInUb = stage0InBuf[pp].Get<KV_T>();
        while (dealRow < Min(16, procSizeLocal) && s2 < procS2EndLocal) { // 拷贝满16行或者遇到-1
            int64_t tokenData[8] = {-1, -1, -1, -1, -1, -1, -1, -1};      // 拷贝进入的8个token的index
            if constexpr (IS_VEC_S2PHYADDR) {
                GetRealS2Addr(tokenData, s2, topkBaseIdx, topkKBase, sparseBlockCount, procS2EndLocal);
            } else {
                GetRealCmpS2Idx(tokenData, s2, topkBaseIdx, topkKBase, sparseBlockCount, procS2EndLocal);
            }
            s2 += 8; // 每次搬运8行
            if (tokenData[0] == -1 && tokenData[1] == -1 && tokenData[2] == -1 && tokenData[3] == -1 &&
                tokenData[4] == -1 && tokenData[5] == -1 && tokenData[6] == -1 && tokenData[7] == -1) {
                meetEnd = true;
                break;
            }
            dealRow += CopyInKvSparse(kvInUb, dealRow, tokenData, runInfo, constInfo);
            if (tokenData[7] == -1) {
                meetEnd = true;
                break;
            }
        }
        if (dealRow == 0) {
            get_buf(PIPE_V, stage0InId, true);
            rls_buf(PIPE_V, stage0InId, true);
            return;
        }
        get_buf(PIPE_MTE2, stage0InId, true);
        rls_buf(PIPE_MTE2, stage0InId, true);
        get_buf(PIPE_V, stage0InId, false);
        rls_buf(PIPE_V, stage0InId, false);
        // 2、dequant by vf
        get_buf(PIPE_V, stage0OutId, false);
        rls_buf(PIPE_V, stage0OutId, false);
        LocalTensor<Q_T> kvDequantOutUb = stage0OutBuf[pp].Get<Q_T>();
        DequantKv(kvDequantOutUb, kvInUb, dealRow, constInfo);
        get_buf(PIPE_V, stage0InId, true);
        rls_buf(PIPE_V, stage0InId, true);
        // 3、copy kv out, ub -> l1
        get_buf(PIPE_V, stage0OutId, true);
        rls_buf(PIPE_V, stage0OutId, true);
        get_buf(PIPE_MTE3, stage0OutId, false);
        rls_buf(PIPE_MTE3, stage0OutId, false);
        CopyOutKvUb2Gm(v0ResGm, kvDequantOutUb, dealRow, s2Start, runInfo, constInfo);
        get_buf(PIPE_MTE3, stage0OutId, true);
        rls_buf(PIPE_MTE3, stage0OutId, true);
        s2Start += dealRow;
        pingPongV0 ^= 1;
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::ProcessVec1(
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputBuf,
    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf, RunInfo &runInfo, ConstInfo &constInfo)
{
    bmm1ResBuf.WaitCrossCore();

    const uint32_t softmaxOffset = runInfo.multiCoreIdxMod2;
    const int64_t s2LoopCount = runInfo.s2LoopCount;
    const int64_t s2RealSize = runInfo.s2RealSize;
    const int64_t halfMRealSize = runInfo.halfMRealSize;
    const T softmaxScale = static_cast<T>(constInfo.softmaxScale);
    LocalTensor<float> sumUb = this->softmaxSumBuf[softmaxOffset].template Get<float>();
    LocalTensor<float> maxUb = this->softmaxMaxBuf[softmaxOffset].template Get<float>();
    const uint32_t stage1Offset = runInfo.taskIdMod2;
    // SetRunInfo 将 taskIdMod2 设为 taskId & 1，因此 stage1Offset 只会是0/1；叠加基址后，
    // stage1OutBuf[0/1] 分别使用同步槽4/5。
    const uint32_t stage1BufId = stage1OutBufBaseId + stage1Offset;
    LocalTensor<float> expUb = this->softmaxExpBuf[stage1Offset].template Get<T>();
    get_buf(PIPE_V, stage1BufId, false);
    rls_buf(PIPE_V, stage1BufId, false);
    LocalTensor<Q_T> stage1CastTensor = this->stage1OutBuf[stage1Offset].template Get<Q_T>();

    LocalTensor<T> apiTmpBuffer = this->commonTBuf.template Get<T>();
    LocalTensor<T> mmRes = bmm1ResBuf.template GetTensor<T>();

    // loopCount = 0 但传入sinks时走update分支，maxUb通过sinks初始化，sumUb初始化为1.0
    if (s2LoopCount == 0 && !isSinks) {
        if (likely(s2RealSize == 128)) { // s2RealSize等于128分档, VF内常量化减少if判断
            ProcessVec1Vf<T, Q_T, false, s1BaseSize, s2BaseSize, SCFaVectorApi::OriginNRange::EQ_128_SCFA>(
                stage1CastTensor, mmRes, sumUb, maxUb, maxUb, apiTmpBuffer, halfMRealSize, s2RealSize, softmaxScale,
                negativeFloatScalar);
        } else if (s2RealSize <= 64) { // s2RealSize小于等于64分档, VF内常量化减少if判断
            ProcessVec1Vf<T, Q_T, false, s1BaseSize, s2BaseSize, SCFaVectorApi::OriginNRange::GT_0_AND_LTE_64_SCFA>(
                stage1CastTensor, mmRes, sumUb, maxUb, maxUb, apiTmpBuffer, halfMRealSize, s2RealSize, softmaxScale,
                negativeFloatScalar);
        } else if (s2RealSize < 128) { // s2RealSize小于128分档, VF内常量化减少if判断
            ProcessVec1Vf<T, Q_T, false, s1BaseSize, s2BaseSize, SCFaVectorApi::OriginNRange::GT_64_AND_LTE_128_SCFA>(
                stage1CastTensor, mmRes, sumUb, maxUb, maxUb, apiTmpBuffer, halfMRealSize, s2RealSize, softmaxScale,
                negativeFloatScalar);
        }
    } else {
        if (s2LoopCount == 0 && isSinks) {
            // s1切1,vec0: 0 ~ halfMRealSize - 1, vec1: gSize - halfMRealSize ~ gSize
            int64_t sinksOffset = 0;
            if constexpr (!IS_SPLIT_G) {
                sinksOffset = GetBlockIdx() % 2 == 0 ? 0 : constInfo.gSize - halfMRealSize;
            } else {
                switch (constInfo.aivIdx % 4) {
                    case 0:
                        sinksOffset = 0;
                        break;
                    case 1:
                        sinksOffset = 32;
                        break;
                    case 2:
                        sinksOffset = 64;
                        break;
                    case 3:
                        sinksOffset = 96;
                        break;
                    default:
                        break;
                }
            }
            LocalTensor<T> sinksUb = this->sinksBuf.template Get<T>();
            DataCopy(maxUb, sinksUb[sinksOffset], halfMRealSize);
            DuplicateSumWithR0<T>(sumUb, R0, halfMRealSize);
        }
        if (likely(s2RealSize == 128)) { // s2RealSize等于128分档, VF内常量化减少if判断
            ProcessVec1Vf<T, Q_T, true, s1BaseSize, s2BaseSize, SCFaVectorApi::OriginNRange::EQ_128_SCFA>(
                stage1CastTensor, mmRes, sumUb, maxUb, maxUb, apiTmpBuffer, halfMRealSize, s2RealSize, softmaxScale,
                negativeFloatScalar);
        } else if (s2RealSize <= 64) { // s2RealSize小于等于64分档, VF内常量化减少if判断
            ProcessVec1Vf<T, Q_T, true, s1BaseSize, s2BaseSize, SCFaVectorApi::OriginNRange::GT_0_AND_LTE_64_SCFA>(
                stage1CastTensor, mmRes, sumUb, maxUb, maxUb, apiTmpBuffer, halfMRealSize, s2RealSize, softmaxScale,
                negativeFloatScalar);
        } else if (s2RealSize < 128) { // s2RealSize小于128分档, VF内常量化减少if判断
            ProcessVec1Vf<T, Q_T, true, s1BaseSize, s2BaseSize, SCFaVectorApi::OriginNRange::GT_64_AND_LTE_128_SCFA>(
                stage1CastTensor, mmRes, sumUb, maxUb, maxUb, apiTmpBuffer, halfMRealSize, s2RealSize, softmaxScale,
                negativeFloatScalar);
        }
    }
    bmm1ResBuf.SetCrossCore();

    // ===================DataCopy to L1 ====================
    get_buf(PIPE_V, stage1BufId, true);
    rls_buf(PIPE_V, stage1BufId, true);
    get_buf(PIPE_MTE3, stage1BufId, false);
    rls_buf(PIPE_MTE3, stage1BufId, false);

    outputBuf.WaitCrossCore();
    LocalTensor<Q_T> mm2AL1Tensor = outputBuf.GetTensor<Q_T>();
    if (likely(halfMRealSize != 0)) {
        DataCopy(mm2AL1Tensor[constInfo.subBlockIdx * (BLOCK_BYTE / sizeof(Q_T)) * (runInfo.mRealSize - halfMRealSize)],
                 stage1CastTensor,
                 {s2BaseSize / 16, (uint16_t)halfMRealSize, (uint16_t)(vec1Srcstride - halfMRealSize),
                  (uint16_t)(s1BaseSize - halfMRealSize)});
    }

    get_buf(PIPE_MTE3, stage1BufId, true);
    rls_buf(PIPE_MTE3, stage1BufId, true);

    outputBuf.SetCrossCore();
    // ======================================================
    if (s2LoopCount != 0 || (s2LoopCount == 0 && isSinks)) {
        SCFAUpdateExpSumAndExpMax<T>(sumUb, maxUb, expUb, sumUb, maxUb, apiTmpBuffer, halfMRealSize);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::ProcessVec2(
    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm2ResBuf, RunInfo &runInfo, ConstInfo &constInfo)
{
    bmm2ResBuf.WaitCrossCore();
    if (unlikely(runInfo.vec2MBaseSize == 0)) {
        bmm2ResBuf.SetCrossCore();
        return;
    }

    const int64_t s2LoopCount = runInfo.s2LoopCount;
    const int64_t s2LoopLimit = runInfo.s2LoopLimit;
    const int64_t vec2MBaseSize = runInfo.vec2MBaseSize;
    runInfo.vec2S1RealSize = runInfo.vec2S1BaseSize;
    runInfo.vec2MRealSize = vec2MBaseSize;
    int64_t vec2CalcSize = vec2MBaseSize * dTemplateAlign64;

    LocalTensor<T> vec2ResUb = this->stage2OutBuf.template Get<T>();
    LocalTensor<T> mmRes = bmm2ResBuf.template GetTensor<T>();
    get_buf(PIPE_V, stage2OutBufBufId, false);
    rls_buf(PIPE_V, stage2OutBufBufId, false);
    if (unlikely(s2LoopCount == 0)) {
        DataCopy(vec2ResUb, mmRes, vec2CalcSize);
    } else {
        const uint32_t softmaxExpOffset = runInfo.taskIdMod2;
        LocalTensor<T> expUb = softmaxExpBuf[softmaxExpOffset].template Get<T>();
        if (s2LoopCount < s2LoopLimit) {
            FlashUpdateNew<T, Q_T, OUTPUT_T, dTemplateAlign64>(vec2ResUb, mmRes, vec2ResUb, expUb, vec2MBaseSize);
        } else {
            LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.multiCoreIdxMod2].template Get<float>();
            FlashUpdateLastNew<T, Q_T, OUTPUT_T, dTemplateAlign64>(vec2ResUb, mmRes, vec2ResUb, expUb, sumUb,
                                                                   vec2MBaseSize);
        }
    }

    bmm2ResBuf.SetCrossCore();
    if (s2LoopCount == s2LoopLimit) {
        if (unlikely(s2LoopCount == 0)) {
            LocalTensor<float> sumUb = this->softmaxSumBuf[runInfo.multiCoreIdxMod2].template Get<float>();
            LastDivNew<T, Q_T, OUTPUT_T, dTemplateAlign64>(vec2ResUb, vec2ResUb, sumUb, vec2MBaseSize);
        }

        this->CopyOutAttentionOut(runInfo, constInfo, vec2ResUb, 0, vec2CalcSize);
    }
    get_buf(PIPE_MTE3, stage2OutBufBufId, true);
    rls_buf(PIPE_MTE3, stage2OutBufBufId, true);
}

TEMPLATES_DEF_NO_DEFAULT
template <typename VEC2_RES_T>
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::Bmm2DataCopyOut(RunInfo &runInfo, ConstInfo &constInfo,
                                                                    LocalTensor<VEC2_RES_T> &vec2ResUb,
                                                                    int64_t vec2S1Idx, int64_t vec2CalcSize)
{
    LocalTensor<OUTPUT_T> attenOut;
    int64_t dSizeAligned64 = (int64_t)dTemplateAlign64;

    attenOut.SetAddr(vec2ResUb.address_);
    Cast(attenOut, vec2ResUb, RoundMode::CAST_ROUND, vec2CalcSize);
    get_buf(PIPE_V, stage2OutBufBufId, true);
    rls_buf(PIPE_V, stage2OutBufBufId, true);
    get_buf(PIPE_MTE3, stage2OutBufBufId, false);
    rls_buf(PIPE_MTE3, stage2OutBufBufId, false);

    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockLen = constInfo.dSizeV * sizeof(OUTPUT_T);
    dataCopyParams.srcStride = (dSizeAligned64 - constInfo.dSizeV) >> 4; // 以32B为单位偏移，bf16类型即偏移16个数，右移4
    dataCopyParams.dstStride = constInfo.attentionOutStride;
    dataCopyParams.blockCount = runInfo.vec2MRealSize;

    DataCopyPad(this->attentionOutGm[runInfo.attentionOutOffset], attenOut, dataCopyParams);
}

TEMPLATES_DEF_NO_DEFAULT
template <typename VEC2_RES_T>
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::CopyOutAttentionOut(RunInfo &runInfo, ConstInfo &constInfo,
                                                                        LocalTensor<VEC2_RES_T> &vec2ResUb,
                                                                        int64_t vec2S1Idx, int64_t vec2CalcSize)
{
    this->Bmm2DataCopyOut(runInfo, constInfo, vec2ResUb, vec2S1Idx, vec2CalcSize);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::InitOutputSingleCore(ConstInfo &constInfo)
{
    uint32_t coreNum = GetBlockNum();
    uint64_t totalOutputSize = 0;

    // n2 = 1, n1 = gn2 = gSize
    if (LAYOUT_T == SAS_LAYOUT::BSND) {
        totalOutputSize = constInfo.bSize * constInfo.gSize * constInfo.s1Size * constInfo.dSizeV;
    } else if (LAYOUT_T == SAS_LAYOUT::TND) {
        totalOutputSize = constInfo.s1Size * constInfo.gSize * constInfo.dSizeV;
    }

    if (coreNum != 0) {
        uint64_t singleCoreSize = (totalOutputSize + (CV_RATIO * coreNum) - 1) / (CV_RATIO * coreNum);
        uint64_t tailSize = totalOutputSize - constInfo.aivIdx * singleCoreSize;
        uint64_t singleInitOutputSize = tailSize < singleCoreSize ? tailSize : singleCoreSize;
        if (singleInitOutputSize > 0) {
            matmul::InitOutput<OUTPUT_T>(this->attentionOutGm[constInfo.aivIdx * singleCoreSize], singleInitOutputSize,
                                         0);
        }
    }
    SyncAll();
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::CleanOutput(__gm__ uint8_t *attentionOut, ConstInfo &constInfo)
{
    if ASCEND_IS_AIV {
        this->attentionOutGm.SetGlobalBuffer((__gm__ OUTPUT_T *)attentionOut);
        if (constInfo.needInit == 1) {
            InitOutputSingleCore(constInfo);
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline int32_t SCFABlockVec<TEMPLATE_ARGS>::GetActualS1Size(uint32_t bIdx, __gm__ int32_t *actualSeqQlenAddr,
                                                                       __gm__ int32_t *cuSeqlensQAddr,
                                                                       const ConstInfo &constInfo)
{
    int32_t actualS1Size = 0;
    if constexpr (LAYOUT_T == SAS_LAYOUT::TND) {
        actualS1Size = (actualSeqQlenAddr == nullptr) ? (cuSeqlensQAddr[bIdx + 1] - cuSeqlensQAddr[bIdx]) :
                                                        actualSeqQlenAddr[bIdx];
    } else {
        actualS1Size = (actualSeqQlenAddr == nullptr) ? constInfo.s1Size : actualSeqQlenAddr[bIdx];
    }
    return actualS1Size;
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline int32_t SCFABlockVec<TEMPLATE_ARGS>::GetActualS2Size(uint32_t bIdx,
                                                                       __gm__ int32_t *actualSeqKvlenAddr,
                                                                       const ConstInfo &constInfo)
{
    int32_t actualS2Size = 0;
    if (constInfo.isActualLenDimsKVNull) {
        actualS2Size = constInfo.s2Size;
    } else {
        actualS2Size = actualSeqKvlenAddr[bIdx];
    }
    return actualS2Size;
}

template <typename T>
__simd_vf__ void GetKVPhyAddrVFImpl(__ubuf__ uint32_t *kvPhyAddrUb, __ubuf__ int32_t *sparseIdxUb,
                                    __ubuf__ int32_t *blkTableUb, const uint16_t s2Loop, uint32_t s2Tail,
                                    const uint32_t blockSize, const int16_t shiftRightNum,
                                    const uint32_t sparseBlockSize, const uint32_t kvDim, const uint32_t kvStride)
{
    static const uint16_t s2NumPerLoop = 128;
    static const uint16_t s2NumPerReg = 64;
    static const uint16_t outOffsetPerLoop = 256;
    static const uint16_t outOffsetPerReg = 128;
    static const uint32_t inValidValue = 0xFFFFFFFF;
    MicroAPI::MaskReg preg_all_b32 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg add_carry_l_1;
    MicroAPI::MaskReg add_carry_h_1;
    MicroAPI::MaskReg add_carry_l_2;
    MicroAPI::MaskReg add_carry_h_2;
    MicroAPI::MaskReg preg_tail_neg_1_b32;
    MicroAPI::MaskReg preg_tail_neg_2_b32;

    MicroAPI::RegTensor<uint32_t> vreg_kvStride;
    MicroAPI::RegTensor<uint32_t> vreg_sparse_idx_1;
    MicroAPI::RegTensor<uint32_t> vreg_sparse_idx_2;
    MicroAPI::RegTensor<uint32_t> vreg_blockSize;
    MicroAPI::RegTensor<uint32_t> vreg_shift_rights_num;
    MicroAPI::RegTensor<uint32_t> vreg_pa_blk_idx_1;
    MicroAPI::RegTensor<uint32_t> vreg_pa_blk_idx_2;
    MicroAPI::RegTensor<uint32_t> vreg_pa_tmp_1;
    MicroAPI::RegTensor<uint32_t> vreg_pa_tmp_2;
    MicroAPI::RegTensor<uint32_t> vreg_pa_offset_1;
    MicroAPI::RegTensor<uint32_t> vreg_pa_offset_2;
    MicroAPI::RegTensor<uint32_t> vreg_phy_offset_1;
    MicroAPI::RegTensor<uint32_t> vreg_phy_offset_2;
    MicroAPI::RegTensor<uint32_t> vreg_phy_blk_idx_1;
    MicroAPI::RegTensor<uint32_t> vreg_phy_blk_idx_2;

    MicroAPI::RegTensor<uint32_t> vreg_blk_id_mul_stride_H_1;
    MicroAPI::RegTensor<uint32_t> vreg_blk_id_mul_stride_tmp_H_1;
    MicroAPI::RegTensor<uint32_t> vreg_blk_id_mul_stride_L_1;
    MicroAPI::RegTensor<uint32_t> vreg_mul_overflow_L_1;
    MicroAPI::RegTensor<uint32_t> vreg_total_offset_L_1;
    MicroAPI::RegTensor<uint32_t> vreg_total_offset_H_1;

    MicroAPI::RegTensor<uint32_t> vreg_blk_id_mul_stride_H_2;
    MicroAPI::RegTensor<uint32_t> vreg_blk_id_mul_stride_tmp_H_2;
    MicroAPI::RegTensor<uint32_t> vreg_blk_id_mul_stride_L_2;
    MicroAPI::RegTensor<uint32_t> vreg_mul_overflow_L_2;
    MicroAPI::RegTensor<uint32_t> vreg_total_offset_L_2;
    MicroAPI::RegTensor<uint32_t> vreg_total_offset_H_2;

    MicroAPI::RegTensor<uint32_t> vreg_zero;
    MicroAPI::Duplicate(vreg_zero, 0);
    MicroAPI::Duplicate(vreg_kvStride, kvStride);

    for (; s2Loop > 1;) {
        for (uint16_t i = 0; i < s2Loop - 1; i++) {
            MicroAPI::LoadAlign<int32_t, MicroAPI::LoadDist::DIST_NORM>(
                (MicroAPI::RegTensor<int32_t> &)vreg_sparse_idx_1, sparseIdxUb + i * s2NumPerLoop);
            MicroAPI::LoadAlign<int32_t, MicroAPI::LoadDist::DIST_NORM>(
                (MicroAPI::RegTensor<int32_t> &)vreg_sparse_idx_2, sparseIdxUb + s2NumPerReg + i * s2NumPerLoop);
            // * sparseBlockSize
            MicroAPI::Muls(vreg_sparse_idx_1, vreg_sparse_idx_1, sparseBlockSize, preg_all_b32);
            MicroAPI::Muls(vreg_sparse_idx_2, vreg_sparse_idx_2, sparseBlockSize, preg_all_b32);
            // 计算右移位数
            // 右移 -> 除blockSize 得到paBlockIdx，vreg_sparse_idx - pa_idx * blocksize -> pa offset
            MicroAPI::ShiftRights(vreg_pa_blk_idx_1, vreg_sparse_idx_1, shiftRightNum, preg_all_b32);
            MicroAPI::ShiftRights(vreg_pa_blk_idx_2, vreg_sparse_idx_2, shiftRightNum, preg_all_b32);

            MicroAPI::Muls(vreg_pa_tmp_1, vreg_pa_blk_idx_1, blockSize, preg_all_b32);
            MicroAPI::Muls(vreg_pa_tmp_2, vreg_pa_blk_idx_2, blockSize, preg_all_b32);
            // offset
            MicroAPI::Sub(vreg_pa_offset_1, vreg_sparse_idx_1, vreg_pa_tmp_1, preg_all_b32);
            MicroAPI::Sub(vreg_pa_offset_2, vreg_sparse_idx_2, vreg_pa_tmp_2, preg_all_b32);
            // 物理页内offset
            MicroAPI::Muls(vreg_phy_offset_1, vreg_pa_offset_1, kvDim, preg_all_b32);
            MicroAPI::Muls(vreg_phy_offset_2, vreg_pa_offset_2, kvDim, preg_all_b32);

            // int32 paBlockId -> 物理id
            DataCopyGather(vreg_phy_blk_idx_1, blkTableUb, vreg_pa_blk_idx_1, preg_all_b32);
            DataCopyGather(vreg_phy_blk_idx_2, blkTableUb, vreg_pa_blk_idx_2, preg_all_b32);

            // 分高低32位计算int64物理地址 -- 乘 stride
            // 低位乘 带进位
            MicroAPI::Mull(vreg_blk_id_mul_stride_L_1, vreg_mul_overflow_L_1, vreg_phy_blk_idx_1, vreg_kvStride,
                           preg_all_b32);
            MicroAPI::Mull(vreg_blk_id_mul_stride_L_2, vreg_mul_overflow_L_2, vreg_phy_blk_idx_2, vreg_kvStride,
                           preg_all_b32);

            // 分高低32位计算int64物理地址 -- 加 offset
            MicroAPI::Add(add_carry_l_1, vreg_total_offset_L_1, vreg_blk_id_mul_stride_L_1, vreg_phy_offset_1,
                          preg_all_b32);
            MicroAPI::Add(add_carry_l_2, vreg_total_offset_L_2, vreg_blk_id_mul_stride_L_2, vreg_phy_offset_2,
                          preg_all_b32);

            MicroAPI::AddC(add_carry_h_1, vreg_total_offset_H_1, vreg_mul_overflow_L_1, vreg_zero, add_carry_l_1,
                           preg_all_b32);
            MicroAPI::AddC(add_carry_h_2, vreg_total_offset_H_2, vreg_mul_overflow_L_2, vreg_zero, add_carry_l_2,
                           preg_all_b32);

            // 搬出 由于拆分为了int32类型，元素个数翻倍
            MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_INTLV_B32>(
                kvPhyAddrUb + i * outOffsetPerLoop, vreg_total_offset_L_1, vreg_total_offset_H_1, preg_all_b32);
            MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_INTLV_B32>(
                kvPhyAddrUb + outOffsetPerReg + i * outOffsetPerLoop, vreg_total_offset_L_2, vreg_total_offset_H_2,
                preg_all_b32);
        }
        break;
    }

    for (uint16_t i = s2Loop - 1; i < s2Loop; i++) {
        MicroAPI::MaskReg preg_tail_1_b32 = MicroAPI::UpdateMask<int32_t>(s2Tail);
        MicroAPI::MaskReg preg_tail_2_b32 = MicroAPI::UpdateMask<int32_t>(s2Tail);
        MicroAPI::Not(preg_tail_neg_1_b32, preg_tail_1_b32, preg_all_b32);
        MicroAPI::Not(preg_tail_neg_2_b32, preg_tail_2_b32, preg_all_b32);

        MicroAPI::LoadAlign<int32_t, MicroAPI::LoadDist::DIST_NORM>((MicroAPI::RegTensor<int32_t> &)vreg_sparse_idx_1,
                                                                    sparseIdxUb + i * s2NumPerLoop);
        MicroAPI::LoadAlign<int32_t, MicroAPI::LoadDist::DIST_NORM>((MicroAPI::RegTensor<int32_t> &)vreg_sparse_idx_2,
                                                                    sparseIdxUb + s2NumPerReg + i * s2NumPerLoop);
        // * sparseBlockSize
        MicroAPI::Muls(vreg_sparse_idx_1, vreg_sparse_idx_1, sparseBlockSize, preg_tail_1_b32);
        MicroAPI::Muls(vreg_sparse_idx_2, vreg_sparse_idx_2, sparseBlockSize, preg_tail_2_b32);
        // 计算右移位数
        // 右移 -> 除blockSize 得到paBlockIdx，vreg_sparse_idx - pa_idx * blocksize -> pa offset
        MicroAPI::ShiftRights(vreg_pa_blk_idx_1, vreg_sparse_idx_1, shiftRightNum, preg_tail_1_b32);
        MicroAPI::ShiftRights(vreg_pa_blk_idx_2, vreg_sparse_idx_2, shiftRightNum, preg_tail_2_b32);

        MicroAPI::Muls(vreg_pa_tmp_1, vreg_pa_blk_idx_1, blockSize, preg_tail_1_b32);
        MicroAPI::Muls(vreg_pa_tmp_2, vreg_pa_blk_idx_2, blockSize, preg_tail_2_b32);
        // offset
        MicroAPI::Sub(vreg_pa_offset_1, vreg_sparse_idx_1, vreg_pa_tmp_1, preg_tail_1_b32);
        MicroAPI::Sub(vreg_pa_offset_2, vreg_sparse_idx_2, vreg_pa_tmp_2, preg_tail_2_b32);
        // 物理页内offset
        MicroAPI::Muls(vreg_phy_offset_1, vreg_pa_offset_1, kvDim, preg_tail_1_b32);
        MicroAPI::Muls(vreg_phy_offset_2, vreg_pa_offset_2, kvDim, preg_tail_2_b32);

        // int32 paBlockId -> 物理id
        DataCopyGather(vreg_phy_blk_idx_1, blkTableUb, vreg_pa_blk_idx_1, preg_tail_1_b32);
        DataCopyGather(vreg_phy_blk_idx_2, blkTableUb, vreg_pa_blk_idx_2, preg_tail_2_b32);

        // 分高低32位计算int64物理地址 -- 乘 stride
        // 低位乘 带进位
        MicroAPI::Mull(vreg_blk_id_mul_stride_L_1, vreg_mul_overflow_L_1, vreg_phy_blk_idx_1, vreg_kvStride,
                       preg_tail_1_b32);
        MicroAPI::Mull(vreg_blk_id_mul_stride_L_2, vreg_mul_overflow_L_2, vreg_phy_blk_idx_2, vreg_kvStride,
                       preg_tail_2_b32);

        // 分高低32位计算int64物理地址 -- 加 offset
        MicroAPI::Add(add_carry_l_1, vreg_total_offset_L_1, vreg_blk_id_mul_stride_L_1, vreg_phy_offset_1,
                      preg_tail_1_b32);
        MicroAPI::Add(add_carry_l_2, vreg_total_offset_L_2, vreg_blk_id_mul_stride_L_2, vreg_phy_offset_2,
                      preg_tail_2_b32);

        MicroAPI::AddC(add_carry_h_1, vreg_total_offset_H_1, vreg_mul_overflow_L_1, vreg_zero, add_carry_l_1,
                       preg_tail_1_b32);
        MicroAPI::AddC(add_carry_h_2, vreg_total_offset_H_2, vreg_mul_overflow_L_2, vreg_zero, add_carry_l_2,
                       preg_tail_2_b32);

        // 无效值填充-1(0xFFFFFFFF)
        MicroAPI::Duplicate<uint32_t, MicroAPI::MaskMergeMode::MERGING>(vreg_total_offset_L_1, inValidValue,
                                                                        preg_tail_neg_1_b32);
        MicroAPI::Duplicate<uint32_t, MicroAPI::MaskMergeMode::MERGING>(vreg_total_offset_H_1, inValidValue,
                                                                        preg_tail_neg_1_b32);
        MicroAPI::Duplicate<uint32_t, MicroAPI::MaskMergeMode::MERGING>(vreg_total_offset_L_2, inValidValue,
                                                                        preg_tail_neg_2_b32);
        MicroAPI::Duplicate<uint32_t, MicroAPI::MaskMergeMode::MERGING>(vreg_total_offset_H_2, inValidValue,
                                                                        preg_tail_neg_2_b32);
        MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_INTLV_B32>(
            kvPhyAddrUb + i * outOffsetPerLoop, vreg_total_offset_L_1, vreg_total_offset_H_1, preg_all_b32);
        MicroAPI::StoreAlign<uint32_t, MicroAPI::StoreDist::DIST_INTLV_B32>(
            kvPhyAddrUb + outOffsetPerReg + i * outOffsetPerLoop, vreg_total_offset_L_2, vreg_total_offset_H_2,
            preg_all_b32);
    }
}

template <typename T>
__aicore__ inline void GetKVPhyAddrVF(LocalTensor<uint32_t> kvPhyAddrTensor, LocalTensor<int32_t> sparseIdxTensor,
                                      LocalTensor<int32_t> blkTableTensor, const uint16_t s2Loop, const uint32_t s2Tail,
                                      const uint32_t blockSize, const int16_t shiftRightNum,
                                      const uint32_t sparseBlockSize, const uint32_t kvDim, const uint32_t kvStride)
{
    __ubuf__ uint32_t *kvPhyAddrUb = (__ubuf__ uint32_t *)(kvPhyAddrTensor.GetPhyAddr());
    __ubuf__ int32_t *sparseIdxUb = (__ubuf__ int32_t *)(sparseIdxTensor.GetPhyAddr());
    __ubuf__ int32_t *blkTableUb = (__ubuf__ int32_t *)(blkTableTensor.GetPhyAddr());
    GetKVPhyAddrVFImpl<uint32_t>(kvPhyAddrUb, sparseIdxUb, blkTableUb, s2Loop, s2Tail, blockSize, shiftRightNum,
                                 sparseBlockSize, kvDim, kvStride);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::CopyPhyAddrToGm(LocalTensor<uint32_t> kvPhyAddrUb, int64_t bS1Idx,
                                                                    int64_t s1Idx, int64_t validS2, int64_t alignNum,
                                                                    ConstInfo &constInfo)
{
    static constexpr int64_t numPerBlock = 32;
    DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = 1U; // 每次处理1行
    dataCopyParams.blockLen = ((validS2 + alignNum - 1) / alignNum * alignNum) * sizeof(int64_t) / numPerBlock;
    dataCopyParams.srcGap = 0U;
    dataCopyParams.dstGap = 0U;
    // gm数据类型int32，实际元素类型int64，字节长度为2倍
    DataCopy(this->kvPhyAddrGm[(bS1Idx + s1Idx) * constInfo.sparseBlockCount * 2], kvPhyAddrUb, dataCopyParams);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::CopyPaTableToUb(LocalTensor<int32_t> blkTableUb, int64_t bIdx,
                                                                    ConstInfo &constInfo)
{
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = 1U; // 每次处理1行
    dataCopyParams.blockLen = constInfo.cmpMaxBlockNumPerBatch * sizeof(int32_t);
    dataCopyParams.srcStride = 0U;
    dataCopyParams.dstStride = 0U;
    DataCopyPadExtParams<int32_t> padParams;
    DataCopyPad(blkTableUb, this->cmpBlockTableGm[bIdx * constInfo.cmpMaxBlockNumPerBatch], dataCopyParams, padParams);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::CopySparseIdxToUb(LocalTensor<int32_t> sparseIdxUb, int64_t bS1Idx,
                                                                      int64_t s1Idx, int64_t validS2,
                                                                      ConstInfo &constInfo)
{
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = 1U;
    dataCopyParams.blockLen = validS2 * sizeof(int32_t); // TODO
    dataCopyParams.srcStride = 0U;
    dataCopyParams.dstStride = 0U;
    DataCopyPadExtParams<int32_t> padParams;
    // TODO offset计算
    DataCopyPad(sparseIdxUb, this->cmpSparseIndicesGm[(bS1Idx + s1Idx) * constInfo.sparseBlockCount], dataCopyParams,
                padParams);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::GetKVPhyAddr(uint32_t hasLoad, uint32_t bN2StartIdx,
                                                                 uint32_t bN2EndIdx, uint32_t gS1StartIdx,
                                                                 uint32_t nextGs1Idx, __gm__ int32_t *actualSeqQlenAddr,
                                                                 __gm__ int32_t *cuSeqlensQAddr,
                                                                 __gm__ int32_t *actualSeqKvlenAddr,
                                                                 __gm__ uint8_t *workspace, ConstInfo &constInfo)
{
    if (hasLoad == 0) {
        return;
    }
    static constexpr uint16_t s2NumPerLoop = 128;
    int32_t blkSize = constInfo.cmpBlockSize;
    // N1=128时相邻4个v核分核数据一致，N1=64时两个v核分核数据一致
    static constexpr uint32_t vecCoreNum = IS_SPLIT_G ? 4 : 2;
    uint32_t vecCoreIdx = IS_SPLIT_G ? constInfo.aivIdx % 4 : constInfo.aivIdx % 2;

    // 计算右移位数
    int16_t shiftRightNum = 0;
    while (blkSize > 1) {
        blkSize >>= 1;
        shiftRightNum++;
    }

    // GM分配
    int64_t v0TotalOffset = 0;
    uint32_t v0ResSize = constInfo.s2BaseSize * constInfo.dSize * sizeof(Q_T);
    if constexpr (IS_SPLIT_G) {
        v0TotalOffset = v0ResSize * 3 * (GetBlockNum() >> 1U); // 3buffer
    } else {
        v0TotalOffset = v0ResSize * 3 * GetBlockNum(); // 3buffer
    }

    this->kvPhyAddrGm.SetGlobalBuffer((__gm__ uint32_t *)(workspace + v0TotalOffset));

    const uint32_t kvStride = static_cast<uint32_t>(constInfo.cmpKvStride);

    LocalTensor<int32_t> blkTableUb = stage0OutBuf[0].template Get<int32_t>();
    LocalTensor<int32_t> sparseIdxUb = stage0OutBuf[1].template Get<int32_t>(); // 逐行处理
    LocalTensor<uint32_t> kvPhyAddrUb = stage2OutBuf.template Get<uint32_t>();

    int64_t totalValidS1 = 0;
    uint32_t tmpGS1Start = gS1StartIdx;
    for (uint32_t bIdx = bN2StartIdx; bIdx < bN2EndIdx; ++bIdx) {
        bool lastBN = (bIdx == bN2EndIdx - 1);
        int32_t actualS1Size = GetActualS1Size(bIdx, actualSeqQlenAddr, cuSeqlensQAddr, constInfo);

        int32_t s1End = actualS1Size;
        if (lastBN && nextGs1Idx != 0) {
            s1End = nextGs1Idx;
        }

        int32_t actualS2Size = GetActualS2Size(bIdx, actualSeqKvlenAddr, constInfo);

        for (int32_t s1Idx = tmpGS1Start; s1Idx < s1End; ++s1Idx) {
            int32_t numerator = actualS2Size - actualS1Size + 1 + s1Idx;
            int32_t curValidS2 =
                (numerator > 0) ? Min((int32_t)constInfo.sparseBlockCount, numerator / (int32_t)constInfo.cmpRatio) : 0;
            if (curValidS2 > 0) {
                totalValidS1++;
            }
        }
        tmpGS1Start = 0;
    }

    int64_t s1PerVecCore = totalValidS1 / vecCoreNum;
    int64_t s1Tail = totalValidS1 % vecCoreNum;
    int64_t curStart = s1PerVecCore * vecCoreIdx + Min((int64_t)vecCoreIdx, s1Tail);
    int64_t curCount = s1PerVecCore + (vecCoreIdx < (uint32_t)s1Tail ? 1 : 0); // 尾块均分到前几个vec

    if (curCount == 0) {
        return;
    }

    int64_t validCounter = 0;
    int64_t processedCount = 0;
    tmpGS1Start = gS1StartIdx;
    bool done = false;

    for (uint32_t bIdx = bN2StartIdx; bIdx < bN2EndIdx && !done; ++bIdx) {
        bool lastBN = (bIdx == bN2EndIdx - 1);

        int32_t actualS1Size = GetActualS1Size(bIdx, actualSeqQlenAddr, cuSeqlensQAddr, constInfo);
        int64_t bS1Idx = 0;
        if constexpr (LAYOUT_T == SAS_LAYOUT::TND) {
            bS1Idx = (actualSeqQlenAddr == nullptr) ? cuSeqlensQAddr[bIdx] : constInfo.s1Size * bIdx;
        } else {
            bS1Idx = constInfo.s1Size * bIdx;
        }

        int32_t s1End = actualS1Size;
        if (lastBN && nextGs1Idx != 0) {
            s1End = nextGs1Idx;
        }

        int32_t actualS2Size = GetActualS2Size(bIdx, actualSeqKvlenAddr, constInfo);

        get_buf(PIPE_MTE2, blkTableBufId, false);
        rls_buf(PIPE_MTE2, blkTableBufId, false);
        CopyPaTableToUb(blkTableUb, bIdx, constInfo);
        get_buf(PIPE_MTE2, blkTableBufId, true);
        rls_buf(PIPE_MTE2, blkTableBufId, true);
        get_buf(PIPE_V, blkTableBufId, false);
        rls_buf(PIPE_V, blkTableBufId, false);

        for (int32_t s1Idx = tmpGS1Start; s1Idx < s1End; ++s1Idx) {
            int32_t numerator = actualS2Size - actualS1Size + 1 + s1Idx;
            int32_t curValidS2 =
                (numerator > 0) ? Min((int32_t)constInfo.sparseBlockCount, numerator / (int32_t)constInfo.cmpRatio) : 0;
            if (curValidS2 <= 0) {
                continue;
            }

            if (validCounter < curStart || validCounter >= curStart + curCount) {
                validCounter++;
                continue;
            }
            validCounter++;

            uint16_t s2Loop = (curValidS2 + s2NumPerLoop - 1) / s2NumPerLoop;
            int32_t s2Tail = curValidS2 - (s2Loop - 1) * s2NumPerLoop;
            get_buf(PIPE_MTE2, sparseIdxBufId, false);
            rls_buf(PIPE_MTE2, sparseIdxBufId, false);
            CopySparseIdxToUb(sparseIdxUb, bS1Idx, s1Idx, curValidS2, constInfo);
            get_buf(PIPE_MTE2, sparseIdxBufId, true);
            rls_buf(PIPE_MTE2, sparseIdxBufId, true);

            get_buf(PIPE_V, sparseIdxBufId, false);
            rls_buf(PIPE_V, sparseIdxBufId, false);
            get_buf(PIPE_V, kvPhyAddrBufId, false);
            rls_buf(PIPE_V, kvPhyAddrBufId, false);
            GetKVPhyAddrVF<uint32_t>(kvPhyAddrUb, sparseIdxUb, blkTableUb, s2Loop, s2Tail, constInfo.cmpBlockSize,
                                     shiftRightNum, constInfo.sparseBlockSize, dVTemplateTypeInput, kvStride);
            get_buf(PIPE_V, sparseIdxBufId, true);
            rls_buf(PIPE_V, sparseIdxBufId, true);
            get_buf(PIPE_V, kvPhyAddrBufId, true);
            rls_buf(PIPE_V, kvPhyAddrBufId, true);
            get_buf(PIPE_MTE3, kvPhyAddrBufId, false);
            rls_buf(PIPE_MTE3, kvPhyAddrBufId, false);
            CopyPhyAddrToGm(kvPhyAddrUb, bS1Idx, s1Idx, curValidS2, s2NumPerLoop, constInfo);
            get_buf(PIPE_MTE3, kvPhyAddrBufId, true);
            rls_buf(PIPE_MTE3, kvPhyAddrBufId, true);

            processedCount++;
            if (processedCount >= curCount) {
                done = true;
                break;
            }
        }
        get_buf(PIPE_V, blkTableBufId, true);
        rls_buf(PIPE_V, blkTableBufId, true);
        tmpGS1Start = 0;
    }

    get_buf(PIPE_MTE2, blkTableBufId, false);
    rls_buf(PIPE_MTE2, blkTableBufId, false);
    get_buf(PIPE_MTE2, sparseIdxBufId, false);
    rls_buf(PIPE_MTE2, sparseIdxBufId, false);
    get_buf(PIPE_V, kvPhyAddrBufId, false);
    rls_buf(PIPE_V, kvPhyAddrBufId, false);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::InitGlobalBuffer(__gm__ uint8_t *oriKV, __gm__ uint8_t *cmpKV,
                                                                     __gm__ uint8_t *cmpSparseIndices,
                                                                     __gm__ uint8_t *oriBlockTable,
                                                                     __gm__ uint8_t *cmpBlockTable,
                                                                     __gm__ uint8_t *sequsedQ, __gm__ uint8_t *sinks)
{
    oriKVGm.SetGlobalBuffer((__gm__ KV_T *)(oriKV));
    oriBlockTableGm.SetGlobalBuffer((__gm__ int32_t *)oriBlockTable);

    if constexpr (TEMPLATE_MODE != SASTemplateMode::SWA_TEMPLATE_MODE) {
        cmpKVGm.SetGlobalBuffer((__gm__ KV_T *)cmpKV);
        cmpBlockTableGm.SetGlobalBuffer((__gm__ int32_t *)cmpBlockTable);
    }

    if constexpr (TEMPLATE_MODE == SASTemplateMode::SCFA_TEMPLATE_MODE) {
        cmpSparseIndicesGm.SetGlobalBuffer((__gm__ int32_t *)cmpSparseIndices);
    }

    if (sinks != nullptr) {
        sinksGm.SetGlobalBuffer((__gm__ T *)sinks);
        this->isSinks = true;
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::SoftmaxInitBuffer()
{
    constexpr uint32_t softmaxBufSize = 256; // VF单次操作256Byte
    tPipe->InitBuffer(softmaxSumBuf[0], softmaxBufSize);
    tPipe->InitBuffer(softmaxSumBuf[1], softmaxBufSize);
    tPipe->InitBuffer(softmaxMaxBuf[0], softmaxBufSize);
    tPipe->InitBuffer(softmaxMaxBuf[1], softmaxBufSize);
    tPipe->InitBuffer(softmaxExpBuf[0], softmaxBufSize);
    tPipe->InitBuffer(softmaxExpBuf[1], softmaxBufSize);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::InitSinksBuffer(ConstInfo &constInfo)
{
    LocalTensor<T> sinksUb = this->sinksBuf.template Get<T>();
    const uint32_t maxN = constInfo.gSize; // N最大支持128, sink shape是[N]
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = 1U;
    dataCopyParams.blockLen = maxN * sizeof(T);
    dataCopyParams.srcStride = 0U;
    dataCopyParams.dstStride = 0U;
    DataCopyPadExtParams<T> padParams;
    DataCopyPad(sinksUb, this->sinksGm, dataCopyParams, padParams);
    get_buf(PIPE_MTE2, syncSinksBufId, true);
    rls_buf(PIPE_MTE2, syncSinksBufId, true);
    get_buf(PIPE_V, syncSinksBufId, false);
    rls_buf(PIPE_V, syncSinksBufId, false);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::InitLocalBuffer(TPipe *pipe, ConstInfo &constInfo)
{
    constexpr uint32_t stage0OutBufferSize = KvQuantSparseAttnSharedkv::ScfaUbConfig::TypedBufferSize<Q_T>::STAGE0_OUT;
    constexpr uint32_t stage2OutBufferSize = KvQuantSparseAttnSharedkv::ScfaUbConfig::TypedBufferSize<T>::STAGE2_OUT;

    // ub buffer
    // V0阶段解量化scale，双buffer
    pipe->InitBuffer(dequantScaleBuff, s1BaseSize * v0ProcessRowNum * 2 * sizeof(float));

    SoftmaxInitBuffer();

    tPipe->InitBuffer(commonTBuf, 512); // commonTBuf内存申请512B
    tPipe->InitBuffer(sinksBuf, 512);   // sinksBuf内存申请512B

    // V0阶段输入，双buffer
    tPipe->InitBuffer(stage0InBuf[0], dVTemplateTypeInput * v0ProcessRowNum * sizeof(KV_T));
    tPipe->InitBuffer(stage0InBuf[1], dVTemplateTypeInput * v0ProcessRowNum * sizeof(KV_T));
    // V0阶段输出额外预留一行，双buffer
    tPipe->InitBuffer(stage0OutBuf[0], stage0OutBufferSize);
    tPipe->InitBuffer(stage0OutBuf[1], stage0OutBufferSize);

    tPipe->InitBuffer(stage1OutBuf[0], vec1Srcstride * s2BaseSize * sizeof(Q_T));
    tPipe->InitBuffer(stage1OutBuf[1], vec1Srcstride * s2BaseSize * sizeof(Q_T));
    tPipe->InitBuffer(stage2OutBuf, stage2OutBufferSize);

    if (this->isSinks) {
        InitSinksBuffer(constInfo);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockVec<TEMPLATE_ARGS>::GetExtremeValue(T &negativeScalar)
{
    uint32_t tmp1 = NEGATIVE_MIN_VAULE_FP32;
    negativeScalar = *((float *)&tmp1);
}

TEMPLATES_DEF
class SCFABlockVecDummy {
public:
    __aicore__ inline SCFABlockVecDummy(){};
    __aicore__ inline void CleanOutput(__gm__ uint8_t *attentionOut, ConstInfo &constInfo) {}
    __aicore__ inline void InitGlobalBuffer(__gm__ uint8_t *oriKV, __gm__ uint8_t *cmpKV,
                                            __gm__ uint8_t *cmpSparseIndices, __gm__ uint8_t *oriBlockTable,
                                            __gm__ uint8_t *cmpBlockTable, __gm__ uint8_t *sequsedQ,
                                            __gm__ uint8_t *sinks)
    {}
    __aicore__ inline void InitVecBlock(TPipe *pipe, __gm__ uint8_t *cuSeqlensQ, __gm__ uint8_t *sequsedKv) {};
    __aicore__ inline void InitLocalBuffer(TPipe *pipe, ConstInfo &constInfo) {}
    __aicore__ inline void SetKvPhyAddrReady() {}
    __aicore__ inline void WaitKvPhyAddrBeforeRead() {}
    __aicore__ inline void ProcessVec1(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &outputBuf,
                                       Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf,
                                       RunInfo &runInfo, ConstInfo &constInfo)
    {}

    using mm2ResPos = Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>;
    __aicore__ inline void ProcessVec2(mm2ResPos &bmm2ResBuf, RunInfo &runInfo, ConstInfo &constInfo) {}
};
} // namespace BaseApi
#endif // KV_QUANT_SPARSE_ATTN_SHAREDKV_SCFA_BLOCK_VECTOR_H
