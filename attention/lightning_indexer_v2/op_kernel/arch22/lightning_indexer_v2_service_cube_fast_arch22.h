/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file lightning_indexer_v2_service_cube_fast_arch22.h
 * \brief matmul cal for cube fill vector
 */
#ifndef LIGHTNING_INDEXER_V2_SERVICE_CUBE_FAST_ARCH22_H
#define LIGHTNING_INDEXER_V2_SERVICE_CUBE_FAST_ARCH22_H

#include "kernel_operator.h"
#include "c_api/asc_simd.h"
#include "c_api/utils/union.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "lightning_indexer_v2_common_fast_arch22.h"

// Cube 补 Vector 流程：K×Q^T 在 L0C 生成 KQT，转存 L1 后与对角 Weight 相乘，Score 写入 GM ring。
// L1 同时保存双 phase KQT、常驻 Q/Weight 和四槽 K，以流水重叠 MM1、MM2 与搬运。
namespace LIV2Kernel {
using namespace AscendC;
using namespace LIV2Common;

template <typename LIT>
class LightningIndexerV2ServiceCubeFast {
public:
    using Q_T = typename LIT::queryType;
    using K_T = typename LIT::keyType;

    __aicore__ inline void InitBuffers(TPipe* pipe);
    __aicore__ inline void InitMm1GlobalTensor(const GlobalTensor<int32_t>& blkTableGm, const GlobalTensor<K_T>& keyGm,
                                               const GlobalTensor<Q_T>& queryGm, const GlobalTensor<float>& mm1ResGm);
    __aicore__ inline void InitDiagGlobalTensor(const GlobalTensor<float>& weightsGm,
                                                const GlobalTensor<float>& scoreGm);
    __aicore__ inline void InitParams(const LIV2FastCommon::ConstInfo& constInfo);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
    __aicore__ inline void ComputeCubeStages(const LIV2FastCommon::RunInfo& runInfo);

    static constexpr uint32_t QUERY_MTE1_MTE2_EVENT = EVENT_ID5;
    static constexpr uint32_t MTE2_MTE1_EVENT = EVENT_ID2;
    static constexpr uint32_t MTE1_M_EVENT = EVENT_ID2;
    static constexpr uint32_t CFV_W_L1_EVENT = EVENT_ID1;
    static constexpr IsResetLoad3dConfig LOAD3DV2_CONFIG = {true, true};

    static constexpr uint64_t D_BASIC_BLOCK_L0 = 128;
    static constexpr uint64_t S2_BASIC_BLOCK_L0 = 128;
    static constexpr uint64_t L0AB_BUFFER_OFFSET = 128 * 128;
    static constexpr uint64_t CFV_PAD_ROWS = 2;
    static constexpr uint64_t CFV_DIAG_N_STORE = 16;
    static constexpr uint64_t CFV_QK_CHUNK = 32;
    static constexpr uint64_t CFV_QK_CHUNKS = 6;
    static constexpr uint64_t CFV_K_HALF = 64;
    static constexpr uint64_t CFV_QK_C_SLOT_ELEMENTS = S2_BASIC_BLOCK_L0 * CFV_QK_CHUNK + BLOCK_CUBE * BLOCK_CUBE;
    static constexpr uint64_t CFV_QK_C_SLOT_NUM = 4;
    static constexpr uint64_t CFV_OUTPUT_C_BASE = CFV_QK_C_SLOT_NUM * CFV_QK_C_SLOT_ELEMENTS;
    static constexpr uint64_t CFV_OUTPUT_C_SLOT_ELEMENTS = 2 * S2_BASIC_BLOCK_L0 * CFV_DIAG_N_STORE;
    static constexpr uint64_t CFV_OUTPUT_TILE_ELEMENTS = S2_BASIC_BLOCK_L0 * CFV_DIAG_N_STORE;
    // KQT的L1分块向上对齐到512
    static constexpr uint64_t KQT_L1_ALIGNMENT_BYTES = 512U;
    static constexpr uint64_t CFV_QK_L1_CHUNK_BYTES =
        (((S2_BASIC_BLOCK_L0 + CFV_PAD_ROWS) * CFV_QK_CHUNK * sizeof(float) + KQT_L1_ALIGNMENT_BYTES - 1U) /
         KQT_L1_ALIGNMENT_BYTES) *
        KQT_L1_ALIGNMENT_BYTES;
    static constexpr uint64_t CFV_QK_L1_CHUNK_ELEMENTS = CFV_QK_L1_CHUNK_BYTES / sizeof(float);
    static constexpr uint64_t CFV_QK_L1_PHASE_ELEMENTS = CFV_QK_CHUNKS * CFV_QK_L1_CHUNK_ELEMENTS;
    // L1 固定分区：[KQT phase0][Q][Weight][4 个 K slot][KQT phase1]。
    static constexpr uint64_t CFV_QK_L1_PHASE0_BYTES = 0;
    static constexpr uint64_t CFV_Q_L1_BASE_BYTES = CFV_QK_L1_PHASE_ELEMENTS * sizeof(float);
    static constexpr uint64_t CFV_W_L1_BASE_BYTES = CFV_Q_L1_BASE_BYTES + 48 * 1024;
    static constexpr uint64_t CFV_K_L1_BASE_BYTES = 192 * 1024;
    static constexpr uint64_t CFV_QK_L1_PHASE1_BYTES = 320 * 1024;
    static constexpr uint64_t CFV_L1_BYTES = 464 * 1024;
    static constexpr uint64_t CFV_K_L1_SLOT_ELEMENTS = S2_BASIC_BLOCK_L0 * D_BASIC_BLOCK_L0;
    static constexpr uint64_t CFV_K_L0A_HALF_ELEMENTS = S2_BASIC_BLOCK_L0 * CFV_K_HALF;
    static constexpr uint64_t CFV_QK_L0A_SLOT_ELEMENTS = S2_BASIC_BLOCK_L0 * CFV_QK_CHUNK;
    static constexpr uint64_t CFV_W_L0B_OFFSET = CFV_QK_CHUNKS * CFV_QK_CHUNK * D_BASIC_BLOCK_L0;
    static constexpr uint32_t CFV_K_L1_EVENT[4] = {EVENT_ID0, EVENT_ID3, EVENT_ID4, EVENT_ID7};
    static constexpr uint64_t CFV_WT_L0B_ELEMENTS = 256 * CFV_DIAG_N_STORE;

private:
    __aicore__ inline void CfvLoadQueryAndWeight(const LIV2FastCommon::RunInfo& runInfo);
    template <uint32_t Slot>
    __aicore__ inline void CfvLoadKToL1(uint64_t keyGmOffset, const LIV2FastCommon::RunInfo& runInfo);
    template <uint32_t Slot>
    __aicore__ inline void CfvPrefetchKToL0a();
    template <uint32_t Phase, uint32_t Chunk>
    __aicore__ inline void CfvStartQkChunk();
    template <uint32_t Phase, uint32_t Chunk>
    __aicore__ inline void CfvFinishQkChunk();
    template <uint32_t Phase, uint32_t Chunk>
    __aicore__ inline void CfvProduceQkChunk();
    template <uint32_t Phase>
    __aicore__ inline void CfvProduceQkTile();
    template <uint32_t Phase, uint32_t Chunk, uint32_t L0aSlot>
    __aicore__ inline void CfvPrefetchQkChunk();
    template <uint32_t Chunk, uint32_t L0aSlot, uint32_t OutputTile, uint32_t OutputSlot>
    __aicore__ inline void CfvConsumeQkChunk();
    template <uint32_t CurrentPhase, uint32_t OutputTile, uint32_t OutputSlot, uint32_t FutureKSlot>
    __aicore__ inline void CfvStepWithFuture();
    template <uint32_t CurrentPhase, uint32_t OutputTile, uint32_t OutputSlot>
    __aicore__ inline void CfvStepWithoutFuture();
    template <uint32_t Phase, uint32_t OutputTile, uint32_t OutputSlot>
    __aicore__ inline void CfvConsumeFinalTile();
    template <uint32_t OutputSlot>
    __aicore__ inline void CfvStoreOutputPair(uint64_t scoreGmOffset);
    __aicore__ inline void LoadWtDiagToL0b(const LIV2FastCommon::RunInfo& runInfo);

    GlobalTensor<int32_t> blkTableGm_;
    GlobalTensor<K_T> keyGm_;
    GlobalTensor<Q_T> queryGm_;
    GlobalTensor<float> scoreGm_;
    GlobalTensor<float> weightsGmForDiag_;

    TBuf<TPosition::A1> bufCfvL1_;
    LocalTensor<uint8_t> cfvL1_;
    TBuf<TPosition::A2> bufQL0_;
    LocalTensor<Q_T> queryL0_;
    TBuf<TPosition::B2> bufKeyL0_;
    LocalTensor<K_T> keyL0_;
    TBuf<TPosition::CO1> bufL0C_;
    LocalTensor<float> cL0_;
    LocalTensor<float> rtL0a_;
    LocalTensor<float> wtL0b_;

    bool cfvStreamPrimed_ = false;
    // 本 tile 实际参与 MM1/MM2 的 QK chunk 数: M<192 时跳过纯零填充 chunk, 减 Cube 无效功
    uint32_t cfvActiveChunks_ = static_cast<uint32_t>(CFV_QK_CHUNKS);
    LIV2FastCommon::ConstInfo constInfo_;
    static constexpr bool PAGE_ATTENTION = LIT::pageAttention;
};

template <typename LIT>
__aicore__ inline void LightningIndexerV2ServiceCubeFast<LIT>::InitParams(const LIV2FastCommon::ConstInfo& constInfo)
{
    constInfo_ = constInfo;
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2ServiceCubeFast<LIT>::InitBuffers(TPipe* pipe)
{
    pipe->InitBuffer(bufCfvL1_, CFV_L1_BYTES);
    cfvL1_ = bufCfvL1_.Get<uint8_t>();
    pipe->InitBuffer(bufQL0_, 2 * L0AB_BUFFER_OFFSET * sizeof(Q_T));
    queryL0_ = bufQL0_.Get<Q_T>();
    pipe->InitBuffer(bufKeyL0_, 2 * L0AB_BUFFER_OFFSET * sizeof(K_T));
    keyL0_ = bufKeyL0_.Get<K_T>();
    pipe->InitBuffer(bufL0C_, 2 * L0AB_BUFFER_OFFSET * sizeof(float));
    cL0_ = bufL0C_.Get<float>();
    rtL0a_ = queryL0_[L0AB_BUFFER_OFFSET].template ReinterpretCast<float>();
    wtL0b_ = keyL0_[CFV_W_L0B_OFFSET].template ReinterpretCast<float>();
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2ServiceCubeFast<LIT>::InitMm1GlobalTensor(
    const GlobalTensor<int32_t>& blkTableGm, const GlobalTensor<K_T>& keyGm, const GlobalTensor<Q_T>& queryGm,
    const GlobalTensor<float>&)
{
    blkTableGm_ = blkTableGm;
    keyGm_ = keyGm;
    queryGm_ = queryGm;
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2ServiceCubeFast<LIT>::InitDiagGlobalTensor(
    const GlobalTensor<float>& weightsGm, const GlobalTensor<float>& scoreGm)
{
    weightsGmForDiag_ = weightsGm;
    scoreGm_ = scoreGm;
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2ServiceCubeFast<LIT>::AllocEventID()
{
    SetMMLayoutTransform(true);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID7);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(CFV_W_L1_EVENT);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(QUERY_MTE1_MTE2_EVENT + 0);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(QUERY_MTE1_MTE2_EVENT + 1);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_FIX>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_FIX>(EVENT_ID1);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_FIX>(EVENT_ID2);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_FIX>(EVENT_ID3);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID2);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID4);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID5);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID6);
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID2);
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID3);
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID4);
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID5);
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID6);
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID7);
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2ServiceCubeFast<LIT>::FreeEventID()
{
    SetMMLayoutTransform(false);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID7);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(CFV_W_L1_EVENT);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(QUERY_MTE1_MTE2_EVENT + 0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(QUERY_MTE1_MTE2_EVENT + 1);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_FIX>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_FIX>(EVENT_ID1);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_FIX>(EVENT_ID2);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_FIX>(EVENT_ID3);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID2);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID4);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID5);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID6);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID2);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID3);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID4);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID5);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID6);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID7);
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2ServiceCubeFast<LIT>::LoadWtDiagToL0b(const LIV2FastCommon::RunInfo& runInfo)
{
    uint64_t s1Rows = runInfo.actMBaseSize / constInfo_.gSize;
    uint64_t colOffset = (CFV_DIAG_N_STORE - s1Rows) / 2;

    // Weight 需要按 b32s split 布局写入 L1；C API 可直接表达该布局。
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(CFV_W_L1_EVENT);
    __cbuf__ float* wL1Ptr = reinterpret_cast<__cbuf__ float*>(CFV_W_L1_BASE_BYTES);
    asc_fill_value_config fillConfig{};
    fillConfig.repeat = 1;
    fillConfig.blk_num = (CFV_WT_L0B_ELEMENTS + 256) * sizeof(float) / 32;
    asc_fill_l1(wL1Ptr, static_cast<uint32_t>(0), fillConfig);
    AscendC::PipeBarrier<PIPE_MTE2>();
    uint64_t dstMatrixStride = 8 + (constInfo_.gSize / 8) * 128;
    uint64_t weightGmOffset = runInfo.tensorWeightsOffset + runInfo.gS1Idx * constInfo_.mBaseSize;
    asc_copy_gm2l1_nd2nz(wL1Ptr + colOffset * 8,
                         const_cast<__gm__ float*>(weightsGmForDiag_[weightGmOffset].GetPhyAddr()),
                         static_cast<uint16_t>(s1Rows), 1, static_cast<uint16_t>(constInfo_.gSize),
                         static_cast<uint16_t>(constInfo_.gSize), static_cast<uint16_t>(constInfo_.gSize), 16, 1,
                         static_cast<uint16_t>(dstMatrixStride));
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT);

    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
    asc_copy_l12l0b(reinterpret_cast<__cb__ float*>(wtL0b_.GetPhyAddr()), wL1Ptr, 0,
                    static_cast<uint8_t>(runInfo.actMBaseSize / 8), 1, 0);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(MTE1_M_EVENT);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(MTE1_M_EVENT);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(CFV_W_L1_EVENT);
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2ServiceCubeFast<LIT>::CfvLoadQueryAndWeight(
    const LIV2FastCommon::RunInfo& runInfo)
{
    constexpr uint64_t Q_SLOT = 0;
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(static_cast<event_t>(QUERY_MTE1_MTE2_EVENT + Q_SLOT));

    // L1 固定为六个 [32,128] NZ 面板; 尾块按有效行分段填入, 未填充列由 Weight 零列屏蔽
    if (runInfo.actMBaseSize == cfvActiveChunks_ * CFV_QK_CHUNK && constInfo_.kHeadNum == 1) {
        AscendC::Nd2NzParams qCopyParams;
        qCopyParams.ndNum = static_cast<uint16_t>(cfvActiveChunks_);
        qCopyParams.nValue = CFV_QK_CHUNK;
        qCopyParams.dValue = D_BASIC_BLOCK_L0;
        qCopyParams.srcNdMatrixStride = CFV_QK_CHUNK * D_BASIC_BLOCK_L0;
        qCopyParams.srcDValue = D_BASIC_BLOCK_L0;
        qCopyParams.dstNzC0Stride = CFV_QK_CHUNK;
        qCopyParams.dstNzNStride = 1;
        qCopyParams.dstNzMatrixStride = CFV_QK_CHUNK * D_BASIC_BLOCK_L0;
        AscendC::DataCopy(cfvL1_[CFV_Q_L1_BASE_BYTES].template ReinterpretCast<Q_T>(),
                          queryGm_[runInfo.tensorQueryOffset], qCopyParams);
    } else {
        AscendC::InitConstValueParams<Q_T> zeroParams;
        zeroParams.repeatTimes = D_BASIC_BLOCK_L0 / BLOCK_CUBE;
        zeroParams.blockNum = CFV_QK_CHUNK;
        zeroParams.dstGap = 0;
        zeroParams.initValue = 0;
        for (uint32_t chunk = 0; chunk < cfvActiveChunks_; ++chunk) {
            AscendC::InitConstValue(cfvL1_[CFV_Q_L1_BASE_BYTES + chunk * CFV_QK_CHUNK * D_BASIC_BLOCK_L0 * sizeof(Q_T)]
                                        .template ReinterpretCast<Q_T>(),
                                    zeroParams);
        }
        AscendC::PipeBarrier<PIPE_MTE2>();
        uint64_t copied = 0;
        while (copied < runInfo.actMBaseSize) {
            uint64_t chunk = copied / CFV_QK_CHUNK;
            uint64_t rowInChunk = copied % CFV_QK_CHUNK;
            uint64_t rowInToken = copied % constInfo_.gSize;
            uint64_t copyRows =
                Min(runInfo.actMBaseSize - copied, Min(CFV_QK_CHUNK - rowInChunk, constInfo_.gSize - rowInToken));
            uint64_t token = copied / constInfo_.gSize;
            uint64_t qGmOffset = runInfo.tensorQueryOffset + token * constInfo_.qHeadNum * D_BASIC_BLOCK_L0 +
                                 rowInToken * D_BASIC_BLOCK_L0;
            AscendC::Nd2NzParams qCopyParams;
            qCopyParams.ndNum = 1;
            qCopyParams.nValue = copyRows;
            qCopyParams.dValue = D_BASIC_BLOCK_L0;
            qCopyParams.srcNdMatrixStride = 0;
            qCopyParams.srcDValue = D_BASIC_BLOCK_L0;
            qCopyParams.dstNzC0Stride = CFV_QK_CHUNK;
            qCopyParams.dstNzNStride = 1;
            qCopyParams.dstNzMatrixStride = 0;
            AscendC::DataCopy(cfvL1_[CFV_Q_L1_BASE_BYTES + chunk * CFV_QK_CHUNK * D_BASIC_BLOCK_L0 * sizeof(Q_T) +
                                     rowInChunk * BLOCK_CUBE * sizeof(Q_T)]
                                  .template ReinterpretCast<Q_T>(),
                              queryGm_[qGmOffset], qCopyParams);
            copied += copyRows;
        }
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(static_cast<event_t>(MTE2_MTE1_EVENT));
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(static_cast<event_t>(MTE2_MTE1_EVENT));

    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(static_cast<event_t>(EVENT_ID0));
    AscendC::LoadData2DParams qParams;
    qParams.startIndex = 0;
    qParams.repeatTimes = (CFV_QK_CHUNK / BLOCK_CUBE) * (D_BASIC_BLOCK_L0 / BLOCK_CUBE);
    qParams.srcStride = 1;
    qParams.dstGap = 0;
    qParams.ifTranspose = false;
    for (uint32_t chunk = 0; chunk < cfvActiveChunks_; ++chunk) {
        AscendC::LoadData(keyL0_[chunk * CFV_QK_CHUNK * D_BASIC_BLOCK_L0],
                          cfvL1_[CFV_Q_L1_BASE_BYTES + chunk * CFV_QK_CHUNK * D_BASIC_BLOCK_L0 * sizeof(K_T)]
                              .template ReinterpretCast<K_T>(),
                          qParams);
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(static_cast<event_t>(MTE1_M_EVENT));
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(static_cast<event_t>(MTE1_M_EVENT));

    LoadWtDiagToL0b(runInfo);
}

template <typename LIT>
template <uint32_t Slot>
__aicore__ inline void LightningIndexerV2ServiceCubeFast<LIT>::CfvLoadKToL1(uint64_t rowOffset,
                                                                            const LIV2FastCommon::RunInfo& runInfo)
{
    // 四个 K slot 独立同步复用；PA 场景需按 cache block 边界拆分搬运。
    static_assert(Slot < 4);
    constexpr uint32_t K_SLOT_EVENT = CFV_K_L1_EVENT[Slot];
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(static_cast<event_t>(K_SLOT_EVENT));
    uint64_t logicalRow = runInfo.s2Idx * constInfo_.s2BaseSize + rowOffset;
    uint64_t validRows = (logicalRow < runInfo.actS2Size) ? Min(S2_BASIC_BLOCK_L0, runInfo.actS2Size - logicalRow) : 0;
    if (validRows < S2_BASIC_BLOCK_L0) {
        AscendC::InitConstValueParams<K_T> zeroParams;
        zeroParams.repeatTimes = D_BASIC_BLOCK_L0 / BLOCK_CUBE;
        zeroParams.blockNum = S2_BASIC_BLOCK_L0;
        zeroParams.dstGap = 0;
        zeroParams.initValue = 0;
        AscendC::InitConstValue(
            cfvL1_[CFV_K_L1_BASE_BYTES + Slot * CFV_K_L1_SLOT_ELEMENTS * sizeof(K_T)].template ReinterpretCast<K_T>(),
            zeroParams);
        AscendC::PipeBarrier<PIPE_MTE2>();
    }
    uint64_t copied = 0;
    while (copied < validRows) {
        uint64_t copyRows = validRows - copied;
        uint64_t keyGmOffset;
        if constexpr (PAGE_ATTENTION) {
            uint64_t currentRow = logicalRow + copied;
            uint64_t blockId = currentRow / constInfo_.kCacheBlockSize;
            uint64_t blockRow = currentRow % constInfo_.kCacheBlockSize;
            copyRows = Min(copyRows, constInfo_.kCacheBlockSize - blockRow);
            keyGmOffset =
                static_cast<uint64_t>(blkTableGm_.GetValue(runInfo.bIdx * constInfo_.maxBlockNumPerBatch + blockId)) *
                    constInfo_.kCacheStride +
                blockRow * constInfo_.kHeadNum * D_BASIC_BLOCK_L0 + runInfo.n2Idx * D_BASIC_BLOCK_L0;
        } else {
            keyGmOffset = runInfo.tensorKeyOffset + (rowOffset + copied) * constInfo_.kHeadNum * D_BASIC_BLOCK_L0;
        }
        AscendC::Nd2NzParams params;
        params.ndNum = 1;
        params.nValue = copyRows;
        params.dValue = D_BASIC_BLOCK_L0;
        params.srcDValue = constInfo_.kHeadNum * D_BASIC_BLOCK_L0;
        params.dstNzC0Stride = S2_BASIC_BLOCK_L0;
        params.dstNzNStride = 1;
        params.srcNdMatrixStride = 0;
        params.dstNzMatrixStride = 0;
        // 将copyRows行K从GM搬到当前L1的NZ布局
        // keyGmOffset以K_T元素计，cfvL1_以uint8_t字节计
        // copied表示槽内已写入行数；每行在当前NZ布局内步进BLOCK_CUBE个K_T元素
        AscendC::DataCopy(cfvL1_[CFV_K_L1_BASE_BYTES + Slot * CFV_K_L1_SLOT_ELEMENTS * sizeof(K_T) +
                                 copied * BLOCK_CUBE * sizeof(K_T)]
                              .template ReinterpretCast<K_T>(),
                          keyGm_[keyGmOffset], params);
        copied += copyRows;
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(static_cast<event_t>(K_SLOT_EVENT));
}

template <typename LIT>
template <uint32_t Slot>
__aicore__ inline void LightningIndexerV2ServiceCubeFast<LIT>::CfvPrefetchKToL0a()
{
    // 一个 128×128 K tile 按 K=64 拆入双 L0A 区，分别参与 MM1 的两次累加。
    static_assert(Slot < 4);
    constexpr uint32_t K_SLOT_EVENT = CFV_K_L1_EVENT[Slot];
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(static_cast<event_t>(K_SLOT_EVENT));
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(static_cast<event_t>(EVENT_ID2));
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(static_cast<event_t>(EVENT_ID3));

    LocalTensor<K_T> src =
        cfvL1_[CFV_K_L1_BASE_BYTES + Slot * CFV_K_L1_SLOT_ELEMENTS * sizeof(K_T)].template ReinterpretCast<K_T>();
    AscendC::LoadData3DParamsV2<K_T> params;
    params.l1H = 1;
    params.l1W = S2_BASIC_BLOCK_L0;
    params.channelSize = CFV_K_HALF;
    params.padList[0] = 0;
    params.padList[1] = 0;
    params.padList[2] = 0;
    params.padList[3] = 0;
    params.mExtension = S2_BASIC_BLOCK_L0;
    params.kExtension = CFV_K_HALF;
    params.mStartPt = 0;
    params.kStartPt = 0;
    params.strideW = 1;
    params.strideH = 1;
    params.filterW = 1;
    params.filterSizeW = 0;
    params.filterH = 1;
    params.filterSizeH = 0;
    params.dilationFilterW = 1;
    params.dilationFilterH = 1;
    params.enTranspose = 0;
    params.fMatrixCtrl = 0;
    AscendC::LoadData<K_T, LOAD3DV2_CONFIG>(queryL0_[0], src, params);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(static_cast<event_t>(EVENT_ID2));

    AscendC::LoadData<K_T, LOAD3DV2_CONFIG>(queryL0_[CFV_K_L0A_HALF_ELEMENTS], src[CFV_K_L0A_HALF_ELEMENTS], params);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(static_cast<event_t>(EVENT_ID3));
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(static_cast<event_t>(K_SLOT_EVENT));
}

template <typename LIT>
template <uint32_t Phase, uint32_t Chunk>
__aicore__ inline void LightningIndexerV2ServiceCubeFast<LIT>::CfvStartQkChunk()
{
    static_assert(Phase < 2 && Chunk < CFV_QK_CHUNKS);
    if (Chunk >= cfvActiveChunks_) {
        return;
    }
    constexpr uint32_t C_SLOT = (Phase * CFV_QK_CHUNKS + Chunk) & 3;
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(static_cast<event_t>(C_SLOT));
    AscendC::MmadParams params;
    params.m = S2_BASIC_BLOCK_L0;
    params.n = CFV_QK_CHUNK;
    params.k = CFV_K_HALF;
    params.cmatrixInitVal = true;
    params.cmatrixSource = false;
    params.unitFlag = 0;
    AscendC::Mmad(cL0_[C_SLOT * CFV_QK_C_SLOT_ELEMENTS], queryL0_[0], keyL0_[Chunk * CFV_QK_CHUNK * D_BASIC_BLOCK_L0],
                  params);
}

template <typename LIT>
template <uint32_t Phase, uint32_t Chunk>
__aicore__ inline void LightningIndexerV2ServiceCubeFast<LIT>::CfvFinishQkChunk()
{
    static_assert(Phase < 2 && Chunk < CFV_QK_CHUNKS);
    if (Chunk >= cfvActiveChunks_) {
        return;
    }
    constexpr uint32_t C_SLOT = (Phase * CFV_QK_CHUNKS + Chunk) & 3;
    AscendC::MmadParams params;
    params.m = S2_BASIC_BLOCK_L0;
    params.n = CFV_QK_CHUNK;
    params.k = CFV_K_HALF;
    params.cmatrixInitVal = false;
    params.cmatrixSource = false;
    params.unitFlag = 0;
    AscendC::Mmad(cL0_[C_SLOT * CFV_QK_C_SLOT_ELEMENTS], queryL0_[CFV_K_L0A_HALF_ELEMENTS],
                  keyL0_[Chunk * CFV_QK_CHUNK * D_BASIC_BLOCK_L0 + CFV_K_HALF * CFV_QK_CHUNK], params);
    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(static_cast<event_t>(C_SLOT));
    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(static_cast<event_t>(C_SLOT));

    constexpr uint64_t PHASE_BASE =
        Phase == 0 ? CFV_QK_L1_PHASE0_BYTES / sizeof(float) : CFV_QK_L1_PHASE1_BYTES / sizeof(float);
    __cbuf__ void* dst = reinterpret_cast<__cbuf__ void*>(PHASE_BASE * sizeof(float) + Chunk * CFV_QK_L1_CHUNK_BYTES);
    __cc__ float* src = reinterpret_cast<__cc__ float*>(cL0_[C_SLOT * CFV_QK_C_SLOT_ELEMENTS].GetPhyAddr());
    // void* 重载保持 FP32 原格式；多写 2 行以完整回写 L0C 尾块。
    asc_copy_l0c2l1(dst, src, static_cast<uint16_t>(CFV_QK_CHUNK),
                    static_cast<uint16_t>(S2_BASIC_BLOCK_L0 + CFV_PAD_ROWS), static_cast<uint32_t>(S2_BASIC_BLOCK_L0),
                    static_cast<uint16_t>(S2_BASIC_BLOCK_L0), 0, 0, 1, true, false);
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(static_cast<event_t>(C_SLOT));
}

template <typename LIT>
template <uint32_t Phase, uint32_t Chunk>
__aicore__ inline void LightningIndexerV2ServiceCubeFast<LIT>::CfvProduceQkChunk()
{
    CfvStartQkChunk<Phase, Chunk>();
    CfvFinishQkChunk<Phase, Chunk>();
}

template <typename LIT>
template <uint32_t Phase>
__aicore__ inline void LightningIndexerV2ServiceCubeFast<LIT>::CfvProduceQkTile()
{
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(static_cast<event_t>(EVENT_ID2));
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_FIX>(static_cast<event_t>(Phase));
    CfvStartQkChunk<Phase, 0>();
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(static_cast<event_t>(EVENT_ID3));
    CfvFinishQkChunk<Phase, 0>();
    CfvProduceQkChunk<Phase, 1>();
    AscendC::SetFlag<AscendC::HardEvent::FIX_MTE1>(static_cast<event_t>(Phase));

    AscendC::WaitFlag<AscendC::HardEvent::MTE1_FIX>(static_cast<event_t>(EVENT_ID2 + Phase));
    CfvProduceQkChunk<Phase, 2>();
    CfvProduceQkChunk<Phase, 3>();
    CfvProduceQkChunk<Phase, 4>();
    CfvProduceQkChunk<Phase, 5>();
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(static_cast<event_t>(EVENT_ID2));
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(static_cast<event_t>(EVENT_ID3));
    AscendC::SetFlag<AscendC::HardEvent::FIX_MTE1>(static_cast<event_t>(EVENT_ID2 + Phase));
}

template <typename LIT>
template <uint32_t Phase, uint32_t Chunk, uint32_t L0aSlot>
__aicore__ inline void LightningIndexerV2ServiceCubeFast<LIT>::CfvPrefetchQkChunk()
{
    static_assert(Phase < 2 && Chunk < CFV_QK_CHUNKS && L0aSlot < 2);
    if (Chunk >= cfvActiveChunks_) {
        return;
    }
    constexpr uint32_t QK_L0A_EVENT = EVENT_ID4 + L0aSlot;
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(static_cast<event_t>(QK_L0A_EVENT));
    constexpr uint64_t PHASE_BASE_BYTES = Phase == 0 ? CFV_QK_L1_PHASE0_BYTES : CFV_QK_L1_PHASE1_BYTES;
    LocalTensor<float> src = cfvL1_[PHASE_BASE_BYTES + Chunk * CFV_QK_L1_CHUNK_BYTES].template ReinterpretCast<float>();
    AscendC::LoadData3DParamsV2<float> params;
    params.l1H = 1;
    params.l1W = S2_BASIC_BLOCK_L0;
    params.channelSize = CFV_QK_CHUNK;
    params.padList[0] = 0;
    params.padList[1] = 0;
    params.padList[2] = 0;
    params.padList[3] = 0;
    params.mExtension = S2_BASIC_BLOCK_L0;
    params.kExtension = CFV_QK_CHUNK;
    params.mStartPt = 0;
    params.kStartPt = 0;
    params.strideW = 1;
    params.strideH = 1;
    params.filterW = 1;
    params.filterSizeW = 0;
    params.filterH = 1;
    params.filterSizeH = 0;
    params.dilationFilterW = 1;
    params.dilationFilterH = 1;
    params.enTranspose = 0;
    params.fMatrixCtrl = 0;
    AscendC::LoadData<float, LOAD3DV2_CONFIG>(rtL0a_[L0aSlot * CFV_QK_L0A_SLOT_ELEMENTS], src, params);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(static_cast<event_t>(QK_L0A_EVENT));
}

template <typename LIT>
template <uint32_t Chunk, uint32_t L0aSlot, uint32_t OutputTile, uint32_t OutputSlot>
__aicore__ inline void LightningIndexerV2ServiceCubeFast<LIT>::CfvConsumeQkChunk()
{
    static_assert(Chunk < CFV_QK_CHUNKS && L0aSlot < 2 && OutputTile < 2 && OutputSlot < 2);
    if (Chunk >= cfvActiveChunks_) {
        return;
    }
    constexpr uint32_t QK_L0A_EVENT = EVENT_ID4 + L0aSlot;
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(static_cast<event_t>(QK_L0A_EVENT));
    AscendC::MmadParams params;
    params.m = S2_BASIC_BLOCK_L0;
    params.n = CFV_DIAG_N_STORE;
    params.k = CFV_QK_CHUNK;
    params.cmatrixInitVal = Chunk == 0;
    params.cmatrixSource = false;
    params.unitFlag = 0;
    AscendC::Mmad(
        cL0_[CFV_OUTPUT_C_BASE + OutputSlot * CFV_OUTPUT_C_SLOT_ELEMENTS + OutputTile * CFV_OUTPUT_TILE_ELEMENTS],
        rtL0a_[L0aSlot * CFV_QK_L0A_SLOT_ELEMENTS], wtL0b_[Chunk * CFV_QK_CHUNK * CFV_DIAG_N_STORE], params);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(static_cast<event_t>(QK_L0A_EVENT));
}

template <typename LIT>
template <uint32_t CurrentPhase, uint32_t OutputTile, uint32_t OutputSlot, uint32_t FutureKSlot>
__aicore__ inline void LightningIndexerV2ServiceCubeFast<LIT>::CfvStepWithFuture()
{
    // 同时推进下一 KQT phase、当前 phase 的 MM2 和未来 K tile 预取。
    constexpr uint32_t NEXT_PHASE = CurrentPhase ^ 1;
    AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE1>(static_cast<event_t>(CurrentPhase));
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(static_cast<event_t>(EVENT_ID2));
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_FIX>(static_cast<event_t>(NEXT_PHASE));

    CfvPrefetchQkChunk<CurrentPhase, 0, 0>();
    CfvPrefetchQkChunk<CurrentPhase, 1, 1>();
    AscendC::SetFlag<AscendC::HardEvent::MTE1_FIX>(static_cast<event_t>(CurrentPhase));

    CfvStartQkChunk<NEXT_PHASE, 0>();
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(static_cast<event_t>(EVENT_ID3));
    CfvFinishQkChunk<NEXT_PHASE, 0>();
    CfvProduceQkChunk<NEXT_PHASE, 1>();
    AscendC::SetFlag<AscendC::HardEvent::FIX_MTE1>(static_cast<event_t>(NEXT_PHASE));

    CfvConsumeQkChunk<0, 0, OutputTile, OutputSlot>();
    AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE1>(static_cast<event_t>(EVENT_ID2 + CurrentPhase));
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_FIX>(static_cast<event_t>(EVENT_ID2 + NEXT_PHASE));

    CfvPrefetchQkChunk<CurrentPhase, 2, 0>();
    CfvConsumeQkChunk<1, 1, OutputTile, OutputSlot>();
    CfvPrefetchQkChunk<CurrentPhase, 3, 1>();
    CfvProduceQkChunk<NEXT_PHASE, 2>();
    CfvProduceQkChunk<NEXT_PHASE, 3>();
    CfvConsumeQkChunk<2, 0, OutputTile, OutputSlot>();

    CfvPrefetchQkChunk<CurrentPhase, 4, 0>();
    CfvProduceQkChunk<NEXT_PHASE, 4>();
    CfvConsumeQkChunk<3, 1, OutputTile, OutputSlot>();
    CfvPrefetchQkChunk<CurrentPhase, 5, 1>();
    CfvProduceQkChunk<NEXT_PHASE, 5>();

    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(static_cast<event_t>(EVENT_ID2));
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(static_cast<event_t>(EVENT_ID3));
    AscendC::SetFlag<AscendC::HardEvent::FIX_MTE1>(static_cast<event_t>(EVENT_ID2 + NEXT_PHASE));
    AscendC::SetFlag<AscendC::HardEvent::MTE1_FIX>(static_cast<event_t>(EVENT_ID2 + CurrentPhase));
    CfvPrefetchKToL0a<FutureKSlot>();

    CfvConsumeQkChunk<4, 0, OutputTile, OutputSlot>();
    CfvConsumeQkChunk<5, 1, OutputTile, OutputSlot>();
}

template <typename LIT>
template <uint32_t CurrentPhase, uint32_t OutputTile, uint32_t OutputSlot>
__aicore__ inline void LightningIndexerV2ServiceCubeFast<LIT>::CfvStepWithoutFuture()
{
    constexpr uint32_t NEXT_PHASE = CurrentPhase ^ 1;
    AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE1>(static_cast<event_t>(CurrentPhase));
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(static_cast<event_t>(EVENT_ID2));
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_FIX>(static_cast<event_t>(NEXT_PHASE));
    CfvPrefetchQkChunk<CurrentPhase, 0, 0>();
    CfvPrefetchQkChunk<CurrentPhase, 1, 1>();
    AscendC::SetFlag<AscendC::HardEvent::MTE1_FIX>(static_cast<event_t>(CurrentPhase));
    CfvStartQkChunk<NEXT_PHASE, 0>();
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(static_cast<event_t>(EVENT_ID3));
    CfvFinishQkChunk<NEXT_PHASE, 0>();
    CfvProduceQkChunk<NEXT_PHASE, 1>();
    AscendC::SetFlag<AscendC::HardEvent::FIX_MTE1>(static_cast<event_t>(NEXT_PHASE));
    CfvConsumeQkChunk<0, 0, OutputTile, OutputSlot>();
    AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE1>(static_cast<event_t>(EVENT_ID2 + CurrentPhase));
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_FIX>(static_cast<event_t>(EVENT_ID2 + NEXT_PHASE));
    CfvPrefetchQkChunk<CurrentPhase, 2, 0>();
    CfvConsumeQkChunk<1, 1, OutputTile, OutputSlot>();
    CfvPrefetchQkChunk<CurrentPhase, 3, 1>();
    CfvProduceQkChunk<NEXT_PHASE, 2>();
    CfvProduceQkChunk<NEXT_PHASE, 3>();
    CfvConsumeQkChunk<2, 0, OutputTile, OutputSlot>();
    CfvPrefetchQkChunk<CurrentPhase, 4, 0>();
    CfvProduceQkChunk<NEXT_PHASE, 4>();
    CfvConsumeQkChunk<3, 1, OutputTile, OutputSlot>();
    CfvPrefetchQkChunk<CurrentPhase, 5, 1>();
    CfvProduceQkChunk<NEXT_PHASE, 5>();
    AscendC::SetFlag<AscendC::HardEvent::FIX_MTE1>(static_cast<event_t>(EVENT_ID2 + NEXT_PHASE));
    AscendC::SetFlag<AscendC::HardEvent::MTE1_FIX>(static_cast<event_t>(EVENT_ID2 + CurrentPhase));
    CfvConsumeQkChunk<4, 0, OutputTile, OutputSlot>();
    CfvConsumeQkChunk<5, 1, OutputTile, OutputSlot>();
}

template <typename LIT>
template <uint32_t Phase, uint32_t OutputTile, uint32_t OutputSlot>
__aicore__ inline void LightningIndexerV2ServiceCubeFast<LIT>::CfvConsumeFinalTile()
{
    AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE1>(static_cast<event_t>(Phase));
    CfvPrefetchQkChunk<Phase, 0, 0>();
    CfvPrefetchQkChunk<Phase, 1, 1>();
    AscendC::SetFlag<AscendC::HardEvent::MTE1_FIX>(static_cast<event_t>(Phase));
    CfvConsumeQkChunk<0, 0, OutputTile, OutputSlot>();
    AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE1>(static_cast<event_t>(EVENT_ID2 + Phase));
    CfvPrefetchQkChunk<Phase, 2, 0>();
    CfvConsumeQkChunk<1, 1, OutputTile, OutputSlot>();
    CfvPrefetchQkChunk<Phase, 3, 1>();
    CfvConsumeQkChunk<2, 0, OutputTile, OutputSlot>();
    CfvPrefetchQkChunk<Phase, 4, 0>();
    CfvConsumeQkChunk<3, 1, OutputTile, OutputSlot>();
    CfvPrefetchQkChunk<Phase, 5, 1>();
    AscendC::SetFlag<AscendC::HardEvent::MTE1_FIX>(static_cast<event_t>(EVENT_ID2 + Phase));
    CfvConsumeQkChunk<4, 0, OutputTile, OutputSlot>();
    CfvConsumeQkChunk<5, 1, OutputTile, OutputSlot>();
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(static_cast<event_t>(EVENT_ID2));
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(static_cast<event_t>(EVENT_ID3));
}

template <typename LIT>
template <uint32_t OutputSlot>
__aicore__ inline void LightningIndexerV2ServiceCubeFast<LIT>::CfvStoreOutputPair(uint64_t scoreGmOffset)
{
    // 两个 128×16 输出 tile 连续写为 256 行，GM 中按低/高 8 列分给两颗 AIV。
    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(static_cast<event_t>(EVENT_ID4 + OutputSlot));
    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(static_cast<event_t>(EVENT_ID4 + OutputSlot));
    AscendC::DataCopyCO12DstParams params;
    params.mSize = 2 * S2_BASIC_BLOCK_L0;
    params.nSize = CFV_DIAG_N_STORE;
    params.dstStride = constInfo_.s2BaseSize;
    params.srcStride = 2 * S2_BASIC_BLOCK_L0;
    params.quantPre = QuantMode_t::NoQuant;
    params.nz2ndEn = false;
    params.channelSplit = true;
    params.reluPre = 0;
    AscendC::DataCopy(scoreGm_[scoreGmOffset], cL0_[CFV_OUTPUT_C_BASE + OutputSlot * CFV_OUTPUT_C_SLOT_ELEMENTS],
                      params);
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(static_cast<event_t>(EVENT_ID4 + OutputSlot));
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2ServiceCubeFast<LIT>::ComputeCubeStages(const LIV2FastCommon::RunInfo& runInfo)
{
    // causal 可见范围可能早于实际 K 序列结束，预取不得越过当前 gS1 的 s2LoopEnd
    uint64_t remainingRows = (runInfo.s2LoopEnd - runInfo.s2Idx + 1) * constInfo_.s2BaseSize;

    // 本 tile 的 M 行数决定实际 chunk 数, 尾块向上取整
    cfvActiveChunks_ = static_cast<uint32_t>((runInfo.actMBaseSize + CFV_QK_CHUNK - 1U) / CFV_QK_CHUNK);

    if (runInfo.isFirstS2InnerLoop) {
        CfvLoadQueryAndWeight(runInfo);
    }
    // 首块灌入 K/Q/KQT；后续块沿用常驻状态并交替两个 KQT phase。
    if (!cfvStreamPrimed_) {
        CfvLoadKToL1<0>(0, runInfo);
        CfvLoadKToL1<2>(S2_BASIC_BLOCK_L0, runInfo);
        CfvLoadKToL1<1>(2 * S2_BASIC_BLOCK_L0, runInfo);
        CfvLoadKToL1<3>(3 * S2_BASIC_BLOCK_L0, runInfo);
        CfvPrefetchKToL0a<0>();
        CfvProduceQkTile<0>();
        if (4 * S2_BASIC_BLOCK_L0 < remainingRows) {
            CfvLoadKToL1<0>(4 * S2_BASIC_BLOCK_L0, runInfo);
        }
        CfvPrefetchKToL0a<2>();
        if (5 * S2_BASIC_BLOCK_L0 < remainingRows) {
            CfvLoadKToL1<2>(5 * S2_BASIC_BLOCK_L0, runInfo);
        }
        cfvStreamPrimed_ = true;
    }

    uint64_t scoreBase =
        (runInfo.loop % LIV2FastCommon::ConstInfo::SCORE_RING_DEPTH) * constInfo_.s2BaseSize * CFV_DIAG_N_STORE;

    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(static_cast<event_t>(EVENT_ID4));
    CfvStepWithFuture<0, 0, 0, 1>();
    if (6 * S2_BASIC_BLOCK_L0 < remainingRows) {
        CfvLoadKToL1<1>(6 * S2_BASIC_BLOCK_L0, runInfo);
    }
    CfvStepWithFuture<1, 1, 0, 3>();
    if (7 * S2_BASIC_BLOCK_L0 < remainingRows) {
        CfvLoadKToL1<3>(7 * S2_BASIC_BLOCK_L0, runInfo);
    }
    CfvStoreOutputPair<0>(scoreBase);

    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(static_cast<event_t>(EVENT_ID5));
    if (!runInfo.isLastS2InnerLoop) {
        CfvStepWithFuture<0, 0, 1, 0>();
        if (8 * S2_BASIC_BLOCK_L0 < remainingRows) {
            CfvLoadKToL1<0>(8 * S2_BASIC_BLOCK_L0, runInfo);
        }
        CfvStepWithFuture<1, 1, 1, 2>();
        if (9 * S2_BASIC_BLOCK_L0 < remainingRows) {
            CfvLoadKToL1<2>(9 * S2_BASIC_BLOCK_L0, runInfo);
        }
    } else {
        CfvStepWithoutFuture<0, 0, 1>();
        CfvConsumeFinalTile<1, 1, 1>();
    }
    CfvStoreOutputPair<1>(scoreBase + 2 * S2_BASIC_BLOCK_L0 * (CFV_DIAG_N_STORE / 2));

    if (runInfo.isLastS2InnerLoop) {
        cfvStreamPrimed_ = false;
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(static_cast<event_t>(EVENT_ID0));
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(static_cast<event_t>(EVENT_ID1));
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(static_cast<event_t>(QUERY_MTE1_MTE2_EVENT));
    }
}

} // namespace LIV2Kernel
#endif
