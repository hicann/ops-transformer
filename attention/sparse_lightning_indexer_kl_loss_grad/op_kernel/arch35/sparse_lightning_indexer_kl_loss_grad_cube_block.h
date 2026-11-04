/* *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file sparse_lightning_indexer_kl_loss_grad_cube_block.h
 * \brief
 */
#ifndef SPARSE_LIGHTNING_INDEXER_KL_LOSS_GRAD_CUBE_BLOCK_H
#define SPARSE_LIGHTNING_INDEXER_KL_LOSS_GRAD_CUBE_BLOCK_H

#include "../../../common/op_kernel/matmul.h"
#include "../../../common/op_kernel/FixpipeOut.h"
#include "sparse_lightning_indexer_kl_loss_grad_regbase_common.h"
#include "c_api/cube_datamove/cube_datamove.h"
#include "c_api/cache_ctrl/cache_ctrl.h"
#include "c_api/sys_var/sys_var.h"
namespace Slig {
using T = float;

template <typename GatherT, bool Enabled>
struct SligAicGatherState {};

template <typename GatherT>
struct SligAicGatherState<GatherT, true> {
    GM_ADDR aicGatherKeyAddr_;
    GM_ADDR aicGatherTopKAddr_;
    event_t gatherReadDoneEvent_;
    event_t gatherWriteDoneEvent_;
};

TEMPLATES_DEF
class SligBlockCube : private SligAicGatherState<INPUT_T, !Deterministic> {
public:
    using Shape = SligShapeTraits<ResidentPipeline>;
    TPipe* pipe;
    constexpr static uint32_t QUERY_INDEX_BUFFER_SIZE = 128 * 128 * 2;
    constexpr static uint32_t MM4_GATHER_BUFFER_SIZE = 128 * 128 * 2;
    constexpr static uint32_t L0C_SINGLE_BUFFER_SIZE = 64 * 1024;
    constexpr static uint32_t L0_SINGLE_BUFFER_SIZE = 32 * 1024;
    Buffer<BufferType::L1> sYQL1Buffer;
    LocalTensor<INPUT_T> sYQL1Tensor;

    static constexpr uint32_t M_SPLIT_SIZE = 128;     // m方向切分
    static constexpr uint32_t N_SPLIT_SIZE = 128;     // n方向切分
    static constexpr uint32_t K_SPLIT_SIZE = 128;     // k方向切分
    static constexpr uint32_t QUERY_INDEX_SIZE = 128; // k方向切分
    static constexpr uint32_t DROPE_SIZE = 64;

    static constexpr uint8_t SYNC_MODE = 2;
    static constexpr uint32_t C0_SIZE = 16;

    // matmul global tensor
    GlobalTensor<INPUT_T> queryGm;
    GlobalTensor<INPUT_T> qRopeGm;
    GlobalTensor<INPUT_T> queryIndexGm;
    GlobalTensor<T> mm2ResGm;
    GlobalTensor<INPUT_T> gatherSYResGm;
    LocalTensor<INPUT_T> gatherAllL1Tensor;
    GlobalTensor<OUT_T> mm4ResGm;
    GlobalTensor<T> mm3DeterResGm;
    /* =====================LocalBuffer变量==================== */

    BufferManager<BufferType::L1>* l1BufferManagerPtr;
    BuffersPolicySingleBuffer<BufferType::L1> sYQueryL1Buf;
    BuffersPolicySingleBuffer<BufferType::L1> sYGatherResBuf; // mm4右矩阵用

    BufferManager<BufferType::L0A> l0aBufferManager;
    BufferManager<BufferType::L0B> l0bBufferManager;
    BuffersPolicyDB<BufferType::L0A> l0aBuf;
    BuffersPolicyDB<BufferType::L0B> l0bBuf;

    BufferManager<BufferType::L0C> l0cBufferManager;
    BuffersPolicy3buff<BufferType::L0C> commonL0CBuf;
    BuffersPolicySingleBuffer<BufferType::L0C> mm4L0CSpecialBuf;

    SLIGradConstInfo constInfo{};

    __aicore__ inline SligBlockCube(){};
    __aicore__ inline void SetCubeBlockParams(TPipe* tPipe, BufferManager<BufferType::L1>* l1BuffMgr);
    __aicore__ inline void InitCubeBuffers(bool gatherL1En);
    __aicore__ inline void InitAicGather(GM_ADDR key, GM_ADDR sparseIndices, bool gatherL1En);
    __aicore__ inline void GatherAicTile(const SLIGradRunInfo& runInfo, const SLIGradConstInfo& info,
                                         const SLIGradKRunInfo& syRunInfo);
    __aicore__ inline void SetGatherAllL1(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_BOTH>& buf)
    {
        this->gatherAllL1Tensor = buf.GetTensor<INPUT_T>();
    }
    __aicore__ inline void InitGlobalBuffer(GM_ADDR query, GM_ADDR dQuery, GlobalTensor<T>& bmm2Res,
                                            GlobalTensor<INPUT_T>& gatherSYResGm, GlobalTensor<T>& bmm3Res);
    __aicore__ inline void ComputeMmSy(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>& bmm2ResBuf,
                                       BuffersPolicyDB<BufferType::L1, SyncType::CROSS_CORE_SYNC_BOTH>& sYL1Buf,
                                       SLIGradRunInfo& runInfo, SLIGradConstInfo& constInfo,
                                       SLIGradKRunInfo& syRunInfo);
    __aicore__ inline void ComputeMm3(BuffersPolicyDB<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>& bmm3Buf,
                                      Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_BOTH>& reluGradResL1Buf,
                                      SLIGradRunInfo& runInfo, SLIGradConstInfo& constInfo, SLIGradKRunInfo& pRunInfo);
    __aicore__ inline void ComputeMm4(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_BOTH>& reluGradResL1Buf,
                                      SLIGradRunInfo& runInfo, SLIGradConstInfo& constInfo, SLIGradKRunInfo& pRunInfo,
                                      bool isFixOut);

private:
    __aicore__ inline bool TryGatherAicTileRaw(const GlobalTensor<INPUT_T>& keyGm, const SLIGradRunInfo& runInfo,
                                               const SLIGradConstInfo& info, const SLIGradKRunInfo& syRunInfo);
    __aicore__ inline void MatmulMm4Resident(const LocalTensor<INPUT_T>& aL1Tensor,
                                             const LocalTensor<INPUT_T>& bL1Tensor, const LocalTensor<T>& cL0Tensor,
                                             const MMParam& param);
    __aicore__ inline void CopyGmToL1(LocalTensor<INPUT_T>& l1Tensor, GlobalTensor<INPUT_T>& gmSrcTensor, uint32_t srcN,
                                      uint32_t srcD, uint32_t srcDstride);
    __aicore__ inline void CopyInMm1AToL1(LocalTensor<INPUT_T>& l1Tensor, const SLIGradRunInfo& info,
                                          SLIGradConstInfo& constInfo);
    __aicore__ inline void CopyInMm1ARopeToL1(LocalTensor<INPUT_T>& l1Tensor, const SLIGradRunInfo& info,
                                              SLIGradConstInfo& constInfo);
    __aicore__ inline void CopyInMm2AToL1(LocalTensor<INPUT_T>& l1Tensor, const SLIGradRunInfo& info,
                                          SLIGradConstInfo& constInfo);
    __aicore__ inline void CopyInMm4BToL1(LocalTensor<INPUT_T>& l1Tensor, uint64_t gatherOffset, uint32_t mSizeAct,
                                          uint32_t realDSize, uint32_t headOffset);
};

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline bool SligBlockCube<TEMPLATE_ARGS>::TryGatherAicTileRaw(const GlobalTensor<INPUT_T>& keyGm,
                                                                         const SLIGradRunInfo& runInfo,
                                                                         const SLIGradConstInfo& info,
                                                                         const SLIGradKRunInfo& syRunInfo)
{
#if (!defined(ASCENDC_CPU_DEBUG) || ASCENDC_CPU_DEBUG != 1) && (!defined(ASCENDC_OOM) || ASCENDC_OOM != 1) && \
    !defined(ASCENDC_TIME_STAMP_ON)
    if constexpr (ResidentPipeline && !Deterministic) {
        if (syRunInfo.kProcessSize != VEC_SY_BASESIZE) {
            return false;
        }
        constexpr uint32_t dValue = 128;
        constexpr uint32_t dValueBytes = dValue * sizeof(INPUT_T);
        constexpr uint32_t c0 = 16;
        constexpr int64_t gapOverflowAt = (static_cast<int64_t>(INT32_MAX) + dValueBytes - 1) / dValueBytes + 1;
        __gm__ INPUT_T* keyPtr = const_cast<__gm__ INPUT_T*>(keyGm.GetPhyAddr());
        const uint64_t keyAddress = reinterpret_cast<uint64_t>(keyPtr);
        if (keyAddress > AscendC::L2_CACHE_OFFSET_MASK) {
            return false;
        }
        const uint64_t maxRawRows = (AscendC::L2_CACHE_OFFSET_MASK - keyAddress + 1U) / dValueBytes;
        const uint8_t cache = AscendC::ExtractCacheMode(keyGm);
        if (static_cast<uint64_t>(info.totalCost) > maxRawRows ||
            (cache != 0U && cache != 1U && cache != 2U && cache != 4U)) {
            return false;
        }
        const int64_t keyRowBase = Shape::AccumS2Idx(runInfo);
        keyPtr += keyRowBase * dValue;
        const uint32_t tileBase = static_cast<uint32_t>(syRunInfo.kTaskId) * (VEC_SY_BASESIZE * dValue);
        __cbuf__ INPUT_T* tilePtr = reinterpret_cast<__cbuf__ INPUT_T*>(gatherAllL1Tensor.GetPhyAddr()) + tileBase;
        const __gm__ int32_t* indexPtr = reinterpret_cast<const __gm__ int32_t*>(this->aicGatherTopKAddr_) +
                                         runInfo.topkGmBaseOffset + syRunInfo.kTaskId * VEC_SY_BASESIZE;
        if ((reinterpret_cast<uint64_t>(indexPtr) & 63U) == 0) {
            __gm__ uint64_t* indexCache = reinterpret_cast<__gm__ uint64_t*>(const_cast<__gm__ int32_t*>(indexPtr));
            if (syRunInfo.kTaskId == 0) {
#pragma unroll
                for (uint32_t offset = 0; offset < COOP_AIC_GATHER_ROWS * sizeof(int32_t); offset += 64) {
                    asc_datacache_preload(indexCache, static_cast<int64_t>(offset));
                }
            }
            if ((syRunInfo.kTaskId + 2) * VEC_SY_BASESIZE <= runInfo.s2RealSize) {
#pragma unroll
                for (uint32_t offset = 0; offset < COOP_AIC_GATHER_ROWS * sizeof(int32_t); offset += 64) {
                    asc_datacache_preload(indexCache, static_cast<int64_t>(VEC_SY_BASESIZE * sizeof(int32_t) + offset));
                }
            }
        }
        Nd2NzParams single;
        single.ndNum = 1;
        single.nValue = 1;
        single.dValue = dValue;
        single.srcDValue = dValue;
        single.srcNdMatrixStride = 0;
        single.dstNzC0Stride = VEC_SY_BASESIZE;
        single.dstNzNStride = 1;
        single.dstNzMatrixStride = c0;
        constexpr uint64_t pairNzConfig =
            (uint64_t(1) << 48) | (uint64_t(VEC_SY_BASESIZE) << 32) | (uint64_t(1) << 16) | uint64_t(2);
        ::asc_set_gm2l1_nz_para(pairNzConfig);
        const uint32_t rowBegin = CoopAicGatherBegin(static_cast<uint32_t>(syRunInfo.kTaskId));
        const uint32_t rowEnd = rowBegin + COOP_AIC_GATHER_ROWS;
        for (uint32_t row = rowBegin; row < rowEnd; row += 2) {
            int64_t index0 = indexPtr[row];
            int64_t index1 = indexPtr[row + 1];
            SortSligPair(index0, index1);
            const int64_t gapRows = index1 - index0;
            if (gapRows > 0 && gapRows < gapOverflowAt) {
                ::asc_copy_gm2l1_nd2nz(tilePtr + row * c0, keyPtr + index0 * dValue, static_cast<uint64_t>(dValueBytes),
                                       cache, 1, dValue, static_cast<uint64_t>(gapRows) * dValueBytes, false);
            } else {
                DataCopy(gatherAllL1Tensor[tileBase + row * c0], keyGm[(keyRowBase + index0) * dValue], single);
                DataCopy(gatherAllL1Tensor[tileBase + (row + 1) * c0], keyGm[(keyRowBase + index1) * dValue], single);
                ::asc_set_gm2l1_nz_para(pairNzConfig);
            }
        }
        return true;
    }
#endif
    return false;
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SligBlockCube<TEMPLATE_ARGS>::InitAicGather(GM_ADDR key, GM_ADDR sparseIndices, bool gatherL1En)
{
    if constexpr (!Deterministic) {
        if (Shape::GatherResident(gatherL1En)) {
            this->aicGatherKeyAddr_ = key;
            this->aicGatherTopKAddr_ = sparseIndices;
            this->gatherReadDoneEvent_ = static_cast<event_t>(pipe->FetchEventID(HardEvent::MTE1_MTE2));
            this->gatherWriteDoneEvent_ = static_cast<event_t>(pipe->FetchEventID(HardEvent::MTE2_MTE1));
        }
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SligBlockCube<TEMPLATE_ARGS>::GatherAicTile(const SLIGradRunInfo& runInfo,
                                                                   const SLIGradConstInfo& info,
                                                                   const SLIGradKRunInfo& syRunInfo)
{
    if constexpr (!Deterministic) {
        if (!Shape::GatherResident(info)) {
            return;
        }
        if constexpr (ResidentPipeline) {
            if (syRunInfo.kProcessSize != VEC_SY_BASESIZE) {
                return;
            }
        } else if (!IsAicGatherTile(static_cast<uint32_t>(syRunInfo.kTaskId), runInfo.s2LoopTimes)) {
            return;
        }
        GlobalTensor<INPUT_T> aicGatherKeyGm;
        aicGatherKeyGm.SetGlobalBuffer((__gm__ INPUT_T*)this->aicGatherKeyAddr_);
        constexpr uint32_t dValue = 128;
        constexpr uint32_t c0 = 16;
        constexpr uint32_t dValueBytes = dValue * sizeof(INPUT_T);
        constexpr int64_t gapOverflowAt = (static_cast<int64_t>(INT32_MAX) + dValueBytes - 1) / dValueBytes + 1;
        const int64_t topkBase = runInfo.topkGmBaseOffset + syRunInfo.kTaskId * info.syKBaseSize;
        const int64_t keyRowBase = Shape::AccumS2Idx(runInfo);
        const __gm__ int32_t* topkPtr = (const __gm__ int32_t*)this->aicGatherTopKAddr_;
        const uint32_t rows = syRunInfo.kProcessSize;
        const uint32_t rowBegin = ResidentPipeline ? CoopAicGatherBegin(static_cast<uint32_t>(syRunInfo.kTaskId)) : 0U;
        const uint32_t rowEnd = ResidentPipeline ? rowBegin + COOP_AIC_GATHER_ROWS : rows;
        const uint32_t pairEnd = rowEnd & ~1U;
        const uint32_t tileBase = static_cast<uint32_t>(syRunInfo.kTaskId) * (VEC_SY_BASESIZE * dValue);

        if (syRunInfo.kTaskId == 0) {
            if constexpr (ResidentPipeline) {
                CrossCoreSetFlag<2, PIPE_MTE1>(SYNC_COOP_GATHER_REUSE_FLAG);
            }
            SetFlag<HardEvent::MTE1_MTE2>(this->gatherReadDoneEvent_);
            WaitFlag<HardEvent::MTE1_MTE2>(this->gatherReadDoneEvent_);
        }

        bool rawGatherDone = false;
        if constexpr (ResidentPipeline) {
            rawGatherDone = TryGatherAicTileRaw(aicGatherKeyGm, runInfo, info, syRunInfo);
        }
        if (!rawGatherDone) {
            Nd2NzParams copy;
            copy.nValue = 1;
            copy.dValue = dValue;
            copy.srcDValue = dValue;
            copy.dstNzC0Stride = VEC_SY_BASESIZE;
            copy.dstNzNStride = 1;
            copy.dstNzMatrixStride = c0;
            const bool pairSortedTile = Shape::PairSortedTile(syRunInfo);
            for (uint32_t row = rowBegin; row < pairEnd; row += 2) {
                int64_t index0 = topkPtr[topkBase + row];
                int64_t index1 = topkPtr[topkBase + row + 1];
                if constexpr (ResidentPipeline) {
                    if (pairSortedTile) {
                        SortSligPair(index0, index1);
                    }
                }
                const int64_t keyOffset0 = (keyRowBase + index0) * dValue;
                const int64_t keyOffset1 = (keyRowBase + index1) * dValue;
                const int64_t gapRows = index1 - index0;
                if (gapRows > 0 && gapRows < gapOverflowAt) {
                    copy.ndNum = 2;
                    copy.srcNdMatrixStride = static_cast<uint64_t>(gapRows) * dValue;
                    DataCopy(gatherAllL1Tensor[tileBase + row * c0], aicGatherKeyGm[keyOffset0], copy);
                } else {
                    copy.ndNum = 1;
                    copy.srcNdMatrixStride = 0;
                    DataCopy(gatherAllL1Tensor[tileBase + row * c0], aicGatherKeyGm[keyOffset0], copy);
                    DataCopy(gatherAllL1Tensor[tileBase + (row + 1) * c0], aicGatherKeyGm[keyOffset1], copy);
                }
            }
            if ((rows & 1U) != 0) {
                const int64_t index = topkPtr[topkBase + pairEnd];
                copy.ndNum = 1;
                copy.srcNdMatrixStride = 0;
                DataCopy(gatherAllL1Tensor[tileBase + pairEnd * c0], aicGatherKeyGm[(keyRowBase + index) * dValue],
                         copy);
            }
        }
        SetFlag<HardEvent::MTE2_MTE1>(this->gatherWriteDoneEvent_);
        WaitFlag<HardEvent::MTE2_MTE1>(this->gatherWriteDoneEvent_);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SligBlockCube<TEMPLATE_ARGS>::SetCubeBlockParams(TPipe* tPipe,
                                                                        BufferManager<BufferType::L1>* l1BuffMgr)
{
    this->pipe = tPipe;
    this->l1BufferManagerPtr = l1BuffMgr;
};

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SligBlockCube<TEMPLATE_ARGS>::InitCubeBuffers(bool gatherL1En)
{
    // init l1 buffer
    sYQueryL1Buf.Init(*l1BufferManagerPtr, QUERY_INDEX_BUFFER_SIZE);
    if constexpr (Deterministic) {
        sYGatherResBuf.Init(*l1BufferManagerPtr, MM4_GATHER_BUFFER_SIZE);
    } else if (!Shape::GatherResident(gatherL1En)) {
        sYGatherResBuf.Init(*l1BufferManagerPtr, MM4_GATHER_BUFFER_SIZE);
    }

    // init l0a l0b buffer
    l0aBufferManager.Init(pipe, L0_MAX_SIZE);
    l0bBufferManager.Init(pipe, L0_MAX_SIZE);
    l0aBuf.Init(l0aBufferManager, L0_SINGLE_BUFFER_SIZE);
    l0bBuf.Init(l0bBufferManager, L0_SINGLE_BUFFER_SIZE);

    // init l0c buffer
    l0cBufferManager.Init(pipe, L0C_MAX_SIZE);
    commonL0CBuf.Init(l0cBufferManager, L0C_SINGLE_BUFFER_SIZE);
    mm4L0CSpecialBuf.Init(l0cBufferManager, ResidentPipeline ? L0C_SINGLE_BUFFER_SIZE : L0_SINGLE_BUFFER_SIZE);
};

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SligBlockCube<TEMPLATE_ARGS>::InitGlobalBuffer(GM_ADDR query, GM_ADDR dQuery,
                                                                      GlobalTensor<T>& bmm2Res,
                                                                      GlobalTensor<INPUT_T>& gatherSYResGm,
                                                                      GlobalTensor<T>& bmm3Res)
{
    queryIndexGm.SetGlobalBuffer((__gm__ INPUT_T*)query);
    mm4ResGm.SetGlobalBuffer((__gm__ OUT_T*)dQuery);
    this->mm2ResGm = bmm2Res;
    this->gatherSYResGm = gatherSYResGm;
    this->mm3DeterResGm = bmm3Res;
};

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SligBlockCube<TEMPLATE_ARGS>::CopyGmToL1(LocalTensor<INPUT_T>& l1Tensor,
                                                                GlobalTensor<INPUT_T>& gmSrcTensor, uint32_t srcN,
                                                                uint32_t srcD, uint32_t srcDstride)
{
    Nd2NzParams nd2nzPara;
    nd2nzPara.ndNum = 1;
    nd2nzPara.nValue = srcN;
    nd2nzPara.dValue = srcD;
    nd2nzPara.srcDValue = srcDstride;
    nd2nzPara.dstNzC0Stride = (srcN + 15) / 16 * 16; // 对齐到16 单位block
    nd2nzPara.dstNzNStride = 1;
    nd2nzPara.srcNdMatrixStride = 0;
    nd2nzPara.dstNzMatrixStride = 0;
    DataCopy(l1Tensor, gmSrcTensor, nd2nzPara);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SligBlockCube<TEMPLATE_ARGS>::CopyInMm2AToL1(LocalTensor<INPUT_T>& l1Tensor,
                                                                    const SLIGradRunInfo& info,
                                                                    SLIGradConstInfo& constInfo)
{
    auto srcGm = queryIndexGm[info.queryIndexTensorOffset];
    CopyGmToL1(l1Tensor, srcGm, Shape::GSize(constInfo), D_SIZE_QUERY_INDEX, D_SIZE_QUERY_INDEX);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SligBlockCube<TEMPLATE_ARGS>::CopyInMm4BToL1(LocalTensor<INPUT_T>& l1Tensor,
                                                                    uint64_t gatherOffset, uint32_t mSizeAct,
                                                                    uint32_t realDSize, uint32_t headOffset)
{
    auto srcGm = gatherSYResGm[gatherOffset];
    CopyGmToL1(l1Tensor, srcGm, mSizeAct, realDSize, headOffset);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SligBlockCube<TEMPLATE_ARGS>::ComputeMmSy(
    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>& bmm2ResBuf,
    BuffersPolicyDB<BufferType::L1, SyncType::CROSS_CORE_SYNC_BOTH>& sYL1Buf, SLIGradRunInfo& runInfo,
    SLIGradConstInfo& constInfo, SLIGradKRunInfo& syRunInfo)
{
    uint32_t kSize = syRunInfo.kProcessSize; // 2048 / 256
    const uint32_t kLoopTimes =
        ResidentPipeline ? 1U : CeilDiv(kSize, VEC_SY_BASESIZE);         // 循环次数，循环累计kSize行搬出UB
    uint32_t tailLoopKSize = kSize - (kLoopTimes - 1) * VEC_SY_BASESIZE; // 尾块大小
    LocalTensor<T> outTensor = bmm2ResBuf.template GetTensor<T>();
    MMParam mmParam = {Shape::GSize(constInfo),
                       VEC_SY_BASESIZE,
                       D_SIZE_QUERY_INDEX,
                       false,
                       true,
                       true,
                       true,
                       ENABLE_UNITFLAG_MM2 ? UNITFLAG_EN_OUTER_LAST : UNITFLAG_DISABLE};
    if (syRunInfo.kTaskId == 0) { // 首次搬运query_index
        sYQL1Buffer = sYQueryL1Buf.Get();
        sYQL1Buffer.Wait<HardEvent::MTE1_MTE2>();
        sYQL1Tensor = sYQL1Buffer.GetTensor<INPUT_T>();
        CopyInMm2AToL1(sYQL1Tensor, runInfo, constInfo);
        sYQL1Buffer.Set<HardEvent::MTE2_MTE1>();
        sYQL1Buffer.Wait<HardEvent::MTE2_MTE1>();
    }
    Buffer<BufferType::L0C> mm2L0CBuffer = commonL0CBuf.Get();
    LocalTensor<T> mm2L0CTensor = mm2L0CBuffer.GetTensor<T>();
    uint64_t gmOffset = syRunInfo.kTaskId * constInfo.syKBaseSize;
    mm2L0CBuffer.Wait<HardEvent::FIX_M>(); // 反向同步
    for (uint32_t kIdx = 0; kIdx < kLoopTimes; kIdx++) {
        Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_BOTH> gatherResL1Buf = sYL1Buf.Get();
        // 右矩阵
        CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(SYNC_GATHER_TO_MM12_FLAG[syRunInfo.kTaskIdMod2]);
        LocalTensor<INPUT_T> gatherL1Tensor;
        if constexpr (Deterministic) {
            gatherL1Tensor = gatherResL1Buf.GetTensor<INPUT_T>();
        } else if (Shape::GatherResident(constInfo)) {
            gatherL1Tensor = gatherAllL1Tensor[static_cast<uint32_t>(syRunInfo.kTaskId) * (VEC_SY_BASESIZE * 128U)];
        } else {
            gatherL1Tensor = gatherResL1Buf.GetTensor<INPUT_T>();
        }
        uint64_t l0cOffset = kIdx * VEC_SY_BASESIZE * Shape::GSize(constInfo);
        MatmulBase<INPUT_T, INPUT_T, T, QUERY_INDEX_SIZE, N_SPLIT_SIZE, K_SPLIT_SIZE, ABLayout::MK, ABLayout::KN>(
            sYQL1Tensor, gatherL1Tensor, l0aBuf, l0bBuf, mm2L0CTensor[l0cOffset], mmParam);
        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(SYNC_GATHER_TO_MM12_FLAG[syRunInfo.kTaskIdMod2]);
    }
    if constexpr (!ENABLE_UNITFLAG_MM2) {
        mm2L0CBuffer.Set<HardEvent::M_FIX>();
        mm2L0CBuffer.Wait<HardEvent::M_FIX>();
    }
    // fix2ub
    CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(SYNC_MM2_TO_V1_FLAG[syRunInfo.kTaskIdMod2]);
    FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipe2UbParams;
    fixpipe2UbParams.nSize = (kSize + 7) >> 3 << 3;
    fixpipe2UbParams.mSize = (Shape::GSize(constInfo) + 1) >> 1 << 1;
    fixpipe2UbParams.srcStride = AlignTo(fixpipe2UbParams.mSize, static_cast<uint16_t>(C0_SIZE));
    // mmResUb上两行之间的间隔，单位：element。
    fixpipe2UbParams.dstStride = AlignTo(constInfo.syKBaseSize, static_cast<int32_t>(C0_SIZE));
    // 双目标模式，按M维度拆分，M / 2 * N写入每个UB, M必须为2的倍数
    fixpipe2UbParams.dualDstCtl = 1;
    fixpipe2UbParams.params.ndNum = 1;
    fixpipe2UbParams.params.srcNdStride = 0;
    fixpipe2UbParams.params.dstNdStride = 0;
    fixpipe2UbParams.unitFlag = ENABLE_UNITFLAG_MM2 ? UNITFLAG_EN_OUTER_LAST : UNITFLAG_DISABLE;
    Fixpipe<T, T, PFA_CFG_ROW_MAJOR_UB>(outTensor, mm2L0CTensor, fixpipe2UbParams);
    CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SYNC_MM2_TO_V1_FLAG[syRunInfo.kTaskIdMod2]);
    // fix2gm
    FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipe2GmParams;
    fixpipe2GmParams.nSize = kSize;
    fixpipe2GmParams.mSize = Shape::GSize(constInfo);
    fixpipe2GmParams.srcStride = AlignTo(fixpipe2GmParams.mSize, static_cast<uint16_t>(C0_SIZE));
    fixpipe2GmParams.dstStride = AlignTo(constInfo.kSizeAlign128, static_cast<uint32_t>(C0_SIZE));
    fixpipe2GmParams.reluEn = true;
    fixpipe2GmParams.dualDstCtl = 0;
    fixpipe2GmParams.params.ndNum = 1;
    fixpipe2GmParams.params.srcNdStride = 0;
    fixpipe2GmParams.params.dstNdStride = 0;
    Fixpipe<T, T, PFA_CFG_ROW_MAJOR_GM>(mm2ResGm[gmOffset], mm2L0CTensor, fixpipe2GmParams);
    if constexpr (ResidentPipeline) {
        if (syRunInfo.kTaskId == 0) {
            CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SYNC_FIRST_RELU_TO_V6_FLAG);
        }
    }
    mm2L0CBuffer.Set<HardEvent::FIX_M>();
};

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SligBlockCube<TEMPLATE_ARGS>::ComputeMm3(
    BuffersPolicyDB<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>& bmm3Buf,
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_BOTH>& reluGradResL1Buf, SLIGradRunInfo& runInfo,
    SLIGradConstInfo& constInfo, SLIGradKRunInfo& pRunInfo)
{
    uint32_t kSize = pRunInfo.kProcessSize; // 2048
    const uint32_t kLoopTimes =
        ResidentPipeline ? 1U : CeilDiv(kSize, VEC_P_BASESIZE);         // 循环次数，循环累计kSize行搬出UB
    uint32_t tailLoopKSize = kSize - (kLoopTimes - 1) * VEC_P_BASESIZE; // 尾块大小
    LocalTensor<INPUT_T> reluGradL1Tensor;
    constexpr bool mm3UnitFlagEn = ENABLE_UNITFLAG_MM3 && !Deterministic;
    MMParam mmParam = {M_SPLIT_SIZE,
                       D_SIZE_QUERY_INDEX,
                       Shape::GSize(constInfo),
                       true,
                       false,
                       true,
                       true,
                       mm3UnitFlagEn ? UNITFLAG_EN_OUTER_LAST : UNITFLAG_DISABLE};
    CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(SYNC_V6_TO_C3_FLAG);
    for (uint32_t kIdx = 0; kIdx < kLoopTimes; kIdx++) {
        if (kIdx == kLoopTimes - 1) {
            mmParam.singleM = tailLoopKSize;
        }
        Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> bmm3ResBuf = bmm3Buf.Get();
        LocalTensor<T> outTensor = bmm3ResBuf.GetTensor<T>();
        reluGradL1Tensor = reluGradResL1Buf.GetTensor<INPUT_T>();
        int64_t reluGradOffset =
            kIdx * VEC_P_BASESIZE * AlignTo(Shape::GSize(constInfo), static_cast<uint32_t>(C0_SIZE));
        // 拷贝左矩阵
        Buffer<BufferType::L0C> mm3L0CBuffer = commonL0CBuf.Get();
        mm3L0CBuffer.Wait<HardEvent::FIX_M>();
        MatmulBase<INPUT_T, INPUT_T, T, M_SPLIT_SIZE, N_SPLIT_SIZE, QUERY_INDEX_SIZE, ABLayout::MK, ABLayout::KN>(
            reluGradL1Tensor[reluGradOffset], sYQL1Tensor, l0aBuf, l0bBuf, mm3L0CBuffer.GetTensor<T>(), mmParam);
        if constexpr (!mm3UnitFlagEn) {
            mm3L0CBuffer.Set<HardEvent::M_FIX>();
            mm3L0CBuffer.Wait<HardEvent::M_FIX>();
        }
        // fix2ub
        if constexpr (!Deterministic) {
            CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(SYNC_C3_TO_V7_FLAG[pRunInfo.kTaskIdMod2]);
            FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipe2UbParams;
            fixpipe2UbParams.nSize = (mmParam.singleN + 7) >> 3 << 3;
            fixpipe2UbParams.mSize = (mmParam.singleM + 1) >> 1 << 1;
            fixpipe2UbParams.srcStride = AlignTo(fixpipe2UbParams.mSize, static_cast<uint16_t>(C0_SIZE));
            // mmResUb上两行之间的间隔，单位：element。
            fixpipe2UbParams.dstStride = AlignTo(D_SIZE_QUERY_INDEX, static_cast<uint32_t>(C0_SIZE));
            // 双目标模式，按M维度拆分，M / 2 * N写入每个UB, M必须为2的倍数
            fixpipe2UbParams.dualDstCtl = 1;
            fixpipe2UbParams.params.ndNum = 1;
            fixpipe2UbParams.params.srcNdStride = 0;
            fixpipe2UbParams.params.dstNdStride = 0;
            fixpipe2UbParams.unitFlag = mm3UnitFlagEn ? UNITFLAG_EN_OUTER_LAST : UNITFLAG_DISABLE;
            Fixpipe<T, T, PFA_CFG_ROW_MAJOR_UB>(outTensor, mm3L0CBuffer.GetTensor<T>(), fixpipe2UbParams);
            mm3L0CBuffer.Set<HardEvent::FIX_M>();
            CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SYNC_C3_TO_V7_FLAG[pRunInfo.kTaskIdMod2]);
        } else {
            int64_t gmOffset = (runInfo.taskIdMod2 * constInfo.kSize + pRunInfo.kTaskId * constInfo.pKBaseSize +
                                kIdx * VEC_P_BASESIZE) *
                               D_SIZE_QUERY_INDEX;
            FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipe2GmParams;
            fixpipe2GmParams.nSize = mmParam.singleN;
            fixpipe2GmParams.mSize = mmParam.singleM;
            fixpipe2GmParams.srcStride = AlignTo(fixpipe2GmParams.mSize, static_cast<uint16_t>(C0_SIZE));
            fixpipe2GmParams.dstStride = AlignTo(D_SIZE_QUERY_INDEX, static_cast<uint32_t>(C0_SIZE));
            fixpipe2GmParams.dualDstCtl = 0;
            fixpipe2GmParams.params.ndNum = 1;
            fixpipe2GmParams.params.srcNdStride = 0;
            fixpipe2GmParams.params.dstNdStride = 0;
            Fixpipe<T, T, PFA_CFG_ROW_MAJOR_GM>(mm3DeterResGm[constInfo.bmm3BaseOffset + gmOffset],
                                                mm3L0CBuffer.GetTensor<T>(), fixpipe2GmParams);
            mm3L0CBuffer.Set<HardEvent::FIX_M>();
        }
    }
    if (pRunInfo.kTaskId == runInfo.s2LoopTimes - 1) {
        sYQL1Buffer.Set<HardEvent::MTE1_MTE2>();
    }
};

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SligBlockCube<TEMPLATE_ARGS>::MatmulMm4Resident(const LocalTensor<INPUT_T>& aL1Tensor,
                                                                       const LocalTensor<INPUT_T>& bL1Tensor,
                                                                       const LocalTensor<T>& cL0Tensor,
                                                                       const MMParam& param)
{
    auto l0aBuffer = l0aBuf.Get();
    l0aBuffer.template Wait<HardEvent::M_MTE1>();
    LocalTensor<INPUT_T> aL0Tensor = l0aBuffer.template GetTensor<INPUT_T>();
    LoadDataToL0A(aL0Tensor, aL1Tensor, param, 0, param.singleK, param.singleM);
    l0aBuffer.template Set<HardEvent::MTE1_M>();

    auto l0bBuffer = l0bBuf.Get();
    l0bBuffer.template Wait<HardEvent::M_MTE1>();
    LocalTensor<INPUT_T> bL0Tensor = l0bBuffer.template GetTensor<INPUT_T>();
    MMParam bLoadParam = param;
    bLoadParam.singleK = K_SPLIT_SIZE;
    LoadDataToL0B(bL0Tensor, bL1Tensor, bLoadParam, 0, param.singleK, param.singleN);
    l0bBuffer.template Set<HardEvent::MTE1_M>();

    l0aBuffer.template Wait<HardEvent::MTE1_M>();
    l0bBuffer.template Wait<HardEvent::MTE1_M>();
    MmadParams madParam;
    madParam.m = param.singleM == 1 ? C0_SIZE : param.singleM;
    madParam.n = param.singleN;
    madParam.k = param.singleK;
    madParam.cmatrixInitVal = param.isOutKFisrt;
    madParam.cmatrixSource = false;
    madParam.unitFlag = param.unitFlag;
    Mmad(cL0Tensor, aL0Tensor, bL0Tensor, madParam);
    l0aBuffer.template Set<HardEvent::M_MTE1>();
    l0bBuffer.template Set<HardEvent::M_MTE1>();
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SligBlockCube<TEMPLATE_ARGS>::ComputeMm4(
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_BOTH>& reluGradResL1Buf, SLIGradRunInfo& runInfo,
    SLIGradConstInfo& constInfo, SLIGradKRunInfo& pRunInfo, bool isFixOut)
{
    uint32_t kSize = pRunInfo.kProcessSize; // 2048
    const uint32_t kLoopTimes =
        ResidentPipeline ? 1U : CeilDiv(kSize, VEC_P_BASESIZE);         // 循环次数，循环累计kSize行搬出UB
    uint32_t tailLoopKSize = kSize - (kLoopTimes - 1) * VEC_P_BASESIZE; // 尾块大小
    constexpr uint32_t dSize = D_SIZE_QUERY_INDEX;

    LocalTensor<INPUT_T> reluGradL1Tensor;
    Buffer<BufferType::L1> sYGatherL1Buffer;
    LocalTensor<INPUT_T> sYGatherResTensor;
    MMParam mmParam = {Shape::GSize(constInfo), dSize, K_SPLIT_SIZE, false, false, true, true, UNITFLAG_DISABLE};
    Buffer<BufferType::L0C> mm4L0CBuffer = mm4L0CSpecialBuf.Get();
    bool useResidentL1 = false;
    if constexpr (!Deterministic) {
        useResidentL1 = Shape::GatherResident(constInfo);
    }
    if constexpr (!ResidentPipeline) {
        if (!useResidentL1) {
            sYGatherL1Buffer = sYGatherResBuf.Get();
            sYGatherResTensor = sYGatherL1Buffer.GetTensor<INPUT_T>();
        }
    }
    for (uint32_t kIdx = 0; kIdx < kLoopTimes; kIdx++) { // 2048 / 128
        if (kIdx == kLoopTimes - 1) {
            mmParam.singleK = tailLoopKSize;
        }

        reluGradL1Tensor = reluGradResL1Buf.GetTensor<INPUT_T>();
        // 拷贝左矩阵
        // 拷贝右矩阵
        int64_t reluGradOffset =
            kIdx * VEC_P_BASESIZE * AlignTo(Shape::GSize(constInfo), static_cast<uint32_t>(C0_SIZE));
        int64_t gatherWorkspaceOffset = (pRunInfo.kTaskId * constInfo.pKBaseSize + kIdx * VEC_P_BASESIZE) * dSize;

        LocalTensor<INPUT_T> sYGatherResTensorL1;
        if (useResidentL1) {
            sYGatherResTensorL1 = gatherAllL1Tensor[gatherWorkspaceOffset];
        } else {
            sYGatherL1Buffer.Wait<HardEvent::MTE1_MTE2>();
            CopyInMm4BToL1(sYGatherResTensor, gatherWorkspaceOffset, mmParam.singleK, dSize, dSize);
            sYGatherL1Buffer.Set<HardEvent::MTE2_MTE1>();
            sYGatherL1Buffer.Wait<HardEvent::MTE2_MTE1>();
            sYGatherResTensorL1 = sYGatherResTensor;
        }
        if ((pRunInfo.kTaskId == 0) && (kIdx == 0)) {
            mm4L0CBuffer.Wait<HardEvent::FIX_M>();
            mmParam.isOutKFisrt = true;
        } else {
            mmParam.isOutKFisrt = false;
        }
        if (useResidentL1) {
            MatmulMm4Resident(reluGradL1Tensor[reluGradOffset], sYGatherResTensorL1, mm4L0CBuffer.GetTensor<T>(),
                              mmParam);
        } else {
            MatmulBase<INPUT_T, INPUT_T, T, QUERY_INDEX_SIZE, N_SPLIT_SIZE, K_SPLIT_SIZE, ABLayout::MK, ABLayout::KN>(
                reluGradL1Tensor[reluGradOffset], sYGatherResTensorL1, l0aBuf, l0bBuf, mm4L0CBuffer.GetTensor<T>(),
                mmParam);
        }
        if (!useResidentL1) {
            sYGatherL1Buffer.Set<HardEvent::MTE1_MTE2>();
        }
    }
    mm4L0CBuffer.Set<HardEvent::M_FIX>();
    mm4L0CBuffer.Wait<HardEvent::M_FIX>();
    CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(SYNC_V6_TO_C3_FLAG);
    if (isFixOut) {
        // fix2gm
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipe2GmParams;
        fixpipe2GmParams.nSize = dSize;
        fixpipe2GmParams.mSize = Shape::GSize(constInfo);
        fixpipe2GmParams.srcStride = AlignTo(fixpipe2GmParams.mSize, static_cast<uint16_t>(C0_SIZE));
        fixpipe2GmParams.dstStride = AlignTo(dSize, static_cast<uint32_t>(C0_SIZE));
        fixpipe2GmParams.dualDstCtl = 0;
        fixpipe2GmParams.params.ndNum = 1;
        fixpipe2GmParams.params.srcNdStride = 0;
        fixpipe2GmParams.params.dstNdStride = 0;
        if constexpr (std::is_same<OUT_T, half>::value) {
            fixpipe2GmParams.quantPre = QuantMode_t::F322F16;
        } else {
            fixpipe2GmParams.quantPre = QuantMode_t::F322BF16;
        }
        Fixpipe<OUT_T, T, PFA_CFG_ROW_MAJOR_GM>(mm4ResGm[runInfo.queryIndexTensorOffset], mm4L0CBuffer.GetTensor<T>(),
                                                fixpipe2GmParams);
        mm4L0CBuffer.Set<HardEvent::FIX_M>();
    }
};

TEMPLATES_DEF
class SligBlockCubeDummy {
public:
    __aicore__ inline SligBlockCubeDummy(){};
    __aicore__ inline void SetCubeBlockParams(TPipe* tPipe, BufferManager<BufferType::L1>* l1BuffMgr) {};
    __aicore__ inline void InitCubeBuffers(bool gatherL1En) {};
    __aicore__ inline void SetGatherAllL1(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_BOTH>& buf) {};
    __aicore__ inline void InitGlobalBuffer(GM_ADDR query, GM_ADDR dQuery, GlobalTensor<T>& bmm2Res,
                                            GlobalTensor<INPUT_T>& gatherSYResGm, GlobalTensor<T>& bmm3Res) {};
    __aicore__ inline void ComputeMmSy(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>& bmm2ResBuf,
                                       BuffersPolicyDB<BufferType::L1, SyncType::CROSS_CORE_SYNC_BOTH>& sYL1Buf,
                                       SLIGradRunInfo& runInfo, SLIGradConstInfo& constInfo,
                                       SLIGradKRunInfo& syRunInfo) {};
    __aicore__ inline void ComputeMm3(BuffersPolicyDB<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>& bmm3Buf,
                                      Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_BOTH>& reluGradResL1Buf,
                                      SLIGradRunInfo& runInfo, SLIGradConstInfo& constInfo,
                                      SLIGradKRunInfo& pRunInfo) {};
    __aicore__ inline void ComputeMm4(Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_BOTH>& reluGradResL1Buf,
                                      SLIGradRunInfo& runInfo, SLIGradConstInfo& constInfo, SLIGradKRunInfo& pRunInfo,
                                      bool isFixOut) {};
};

template <typename T>
struct CubeBlockTraits; // 声明

/* 生成CubeBlockTraits */
#define GEN_TRAIT_TYPE(name, ...) using name##_TRAITS = name;
#define GEN_TRAIT_CONST(name, type, ...) static constexpr type name##Traits = name;

#define DEFINE_CUBE_BLOCK_TRAITS(CUBE_BLOCK_CLASS) \
    TEMPLATES_DEF_NO_DEFAULT \
    struct CubeBlockTraits<CUBE_BLOCK_CLASS<TEMPLATE_ARGS>> { \
        CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_TRAIT_TYPE) \
        CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_TRAIT_CONST) \
    };

DEFINE_CUBE_BLOCK_TRAITS(SligBlockCube);
DEFINE_CUBE_BLOCK_TRAITS(SligBlockCubeDummy);

// /* 生成Arg Traits, kernel中只需要调用ARGS_TRAITS就可以获取所有CubeBlock中的模板参数 */
#define GEN_ARGS_TYPE(name, ...) using name = typename CubeBlockTraits<CubeBlockType>::name##_TRAITS;
#define GEN_ARGS_CONST(name, type, ...) static constexpr type name = CubeBlockTraits<CubeBlockType>::name##Traits;
#define ARGS_TRAITS \
    CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_ARGS_TYPE) \
    CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_ARGS_CONST)
} // namespace Slig
#endif
