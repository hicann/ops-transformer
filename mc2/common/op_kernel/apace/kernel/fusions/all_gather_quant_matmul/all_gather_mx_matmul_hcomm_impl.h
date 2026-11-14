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
 * \file all_gather_mx_matmul_hcomm_impl.h
 * \brief AllGather + QuantMatmul fusion kernel implementation (Hcomm/CCU variant)
 *
 * Init():
 *   AIC: hccl_.InitV2()/SetCcTilingV2()，批量下发 AllGather<true>（scale + dataHead + dataTail），
 *        GetBufferSyncMgr().state_ = &commState_ 绑定 WaitPolicy。
 *
 * Run():
 *   AIC: 执行 FragmentTensor kernel，所有核 Acquire 完成后再统一 Finalize → hccl_.Finalize。
 *        kernel 内逐 tile bufferSyncMgr_.Acquire(dependTileIdx) → state_->hccl_.Wait(handle)。
 * hcomm 通道由 HCCL 自管 buffer，不支持 win 复用：enableBufferReuse=false，Release 为空实现。
 */

#pragma once

// 低版本 devkit(如 9.2.0) 的编译环境不含 hifloat4x2_t/AscendC::hif4_scale，
// 以 devkit 版本守卫 HIF4 kernel 代码（CANN >= 9.3 才启用，与 op_host 侧
// ENABLE_ALL_GATHER_MATMUL_DT_HIFLOAT4 守卫配套、阈值一致）
#if ASC_DEVKIT_MAJOR > 9 || (ASC_DEVKIT_MAJOR == 9 && ASC_DEVKIT_MINOR >= 3)
#define APACE_HIF4_TYPE_SUPPORTED 1
#else
#define APACE_HIF4_TYPE_SUPPORTED 0
#endif

#include "include/tensor_api/tensor.h"
#include "lib/hccl/hccl.h"
#include "blaze/gemm/utils/common_utils.h"
#include "apace/kernel/matmul/quant_batch_matmul/all_gather_qbmm_mx_kernel.h"
#include "apace/kernel/fusions/all_gather_quant_matmul/all_gather_mx_matmul_tiling_data.h"
#include "apace/tiling/matmul_tiling/quant_matmul_tiling_data.h"

namespace Apace {

using namespace AscendC;

template <typename T>
constexpr uint64_t GetHcclDataType()
{
    if constexpr (AscendC::IsSameType<T, float8_e5m2_t>::value) {
        return AscendC::HCCL_DATA_TYPE_FP8E5M2;
    } else if constexpr (AscendC::IsSameType<T, fp4x2_e2m1_t>::value) {
        return AscendC::HCCL_DATA_TYPE_UINT8;
#if APACE_HIF4_TYPE_SUPPORTED
    } else if constexpr (AscendC::IsSameType<T, hifloat4x2_t>::value) {
        return AscendC::HCCL_DATA_TYPE_UINT8;
#endif
    } else {
        return AscendC::HCCL_DATA_TYPE_FP8E4M3;
    }
}

template <AscendC::HcclServerType ServerType>
struct HcommCommState {
    AscendC::Hccl<ServerType> hccl_;
    AscendC::HcclHandle scaleHandle_{0};
    AscendC::HcclHandle dataHeadHandle_{0};
    AscendC::HcclHandle dataTailHandle_{0};
    uint32_t tileCnt_{0};
};

template <AscendC::HcclServerType ServerType>
struct HcommCommWaitPolicy {
    HcommCommState<ServerType>* state_{nullptr};

    __aicore__ inline void Acquire(uint32_t tileIdx)
    {
        // dependTileIdx=0 (HEAD): 本 rank 数据，无需等待通信
        if (tileIdx == 0) {
            return;
        }
        // dependTileIdx=1 (MAIN r0): 首次远端等待，额外等 scale（全量）
        if (tileIdx == 1) {
            state_->hccl_.Wait(state_->scaleHandle_);
        }
        // dependTileIdx=1..tileCnt (MAIN): 等 data head（repeat=tileCnt，每次 Wait 消费一轮）
        if (tileIdx <= state_->tileCnt_) {
            state_->hccl_.Wait(state_->dataHeadHandle_);
        } else {
            // dependTileIdx>tileCnt (TAIL): 等 data tail
            state_->hccl_.Wait(state_->dataTailHandle_);
        }
    }

    __aicore__ inline void Release(uint32_t tileIdx)
    {
        (void)tileIdx; // hcomm 通道由 HCCL 自管 buffer，无 win 复用，无需释放
    }
};

template <typename AType, typename BType, typename CType, AscendC::HcclServerType ServerType, bool IsMxFp4, bool TransB>
class AllGatherMxQuantMatmulHcommImpl {
public:
    explicit __aicore__ inline AllGatherMxQuantMatmulHcommImpl(Apace::hcommAllGatherMatmulTilingData* tilingData)
        : tilingData_(tilingData)
    {}

    __aicore__ inline void Init(GM_ADDR aGM, GM_ADDR aScaleGM, GM_ADDR bGM, GM_ADDR bScaleGM, GM_ADDR biasGM,
                                GM_ADDR cGM, GM_ADDR gatherOut, GM_ADDR workspaceGM);
    __aicore__ inline void Run();

    // Layout 定义
    using LayoutA = asc::te::nd_ext_layout_ptn;
    // MX 场景恒为转置, HiF4 场景转置/非转置均支持
    using LayoutB = AscendC::Std::conditional_t<TransB, asc::te::dn_ext_layout_ptn, asc::te::nd_ext_layout_ptn>;
    using LayoutC = asc::te::nd_ext_layout_ptn;
    using ProblemShape = asc::te::shape<int64_t, int64_t, int64_t, int64_t>;

    // 组件定义
    using BlockMmadFrag = Blaze::Gemm::Block::QmmMxBlockMmadFragment<0, false, AType, LayoutA, BType, LayoutB, CType,
                                                                     LayoutC, float, LayoutC>;
    // Reuse MXFP4's B4 scheduler geometry for HiFP4; MMAD keeps the real type.
#if APACE_HIF4_TYPE_SUPPORTED
    using SchedulerAType =
        AscendC::Std::conditional_t<AscendC::IsSameType<AType, hifloat4x2_t>::value, fp4x2_e2m1_t, AType>;
#else
    using SchedulerAType = AType;
#endif
    using BlockScheduler =
        Blaze::Gemm::Block::BlockSchedulerQuantBatchMatmulV3<ProblemShape, 0, LayoutA, LayoutB, SchedulerAType>;
    using QuantMatmulKernelImpl =
        AllGatherQbmmMxKernel<BlockMmadFrag, BlockScheduler, HcommCommWaitPolicy<ServerType>, false>;

    // 参数类型
    using Params = typename QuantMatmulKernelImpl::Params;
    using BlockSchedulerParams = typename QuantMatmulKernelImpl::BlockSchedulerParams;
    using FragL1Params = typename QuantMatmulKernelImpl::FragL1Params;

    QuantMatmulKernelImpl quantMatmulKernelImpl_;
    HcommCommState<ServerType> commState_;
    Mc2Kernel::OpStateDump opStateDump_;

private:
    __aicore__ inline void InitBaseParams();
    __aicore__ inline void CommitAllGather();
    __aicore__ inline void SetupKernelParams(Params& params);

    Apace::hcommAllGatherMatmulTilingData* tilingData_;

    GM_ADDR aGM_{};
    GM_ADDR aScaleGM_{};
    GM_ADDR bGM_{};
    GM_ADDR bScaleGM_{};
    GM_ADDR biasGM_{};
    GM_ADDR cGM_{};
    GM_ADDR workspaceGM_{};
    GM_ADDR gatherDataAddr_{};
    GM_ADDR gatherScaleAddr_{};

    uint32_t rankId_{};
    uint32_t rankSize_{};
    uint32_t m_{};
    uint32_t k_{};
    uint32_t n_{};
    uint32_t tileCnt_{};
    uint32_t tileM_{};
    uint32_t tailCnt_{};
    uint32_t tailM_{};
    uint32_t paddedTailM_{};
    uint32_t commTurn_{};
    uint64_t headRows_{};
    uint64_t scaleKLen_{};
    uint64_t scaleKGroups_{};
    uint64_t kForComm_{}; // 通信用 K 元素数（FP4 时 = CeilDiv(K,2)）
    uint64_t dataBytesPerMRow_{};
    uint64_t scaleBytesPerMRow_{};
    uint64_t cBytesPerM_{};
    uint64_t dataRegionBytes_{};
    uint64_t scaleRegionBytes_{};

#if APACE_HIF4_TYPE_SUPPORTED
    static constexpr bool IS_HIF4 = AscendC::IsSameType<AType, hifloat4x2_t>::value;
    // HiF4 scale 为 4 字节(uint32)/64K 组，MXFP8/FP4 为 1 字节/组；ScaleType 决定每组字节数
    using ScaleType = AscendC::Std::conditional_t<IS_HIF4, AscendC::hif4_scale, AscendC::fp8_e8m0_t>;
#else
    static constexpr bool IS_HIF4 = false;
    using ScaleType = AscendC::fp8_e8m0_t;
#endif
    static constexpr int32_t SCALE_C0 = IS_HIF4 ? 1 : 2;
    static constexpr uint32_t kPaddingLength = 16;
    // Scale 通信数据类型: MX 为 E8M0; HiF4 的 hif4_scale 为完整 4 字节元数据(E6M2+E1_8+E1_16),
    // HCCL 未提供专用类型, E8M0和hif4_scale 均按 FP8E8M0 传输, 实际传输时乘数据类型(对齐 all_to_all impl 口径)
    static constexpr uint64_t scaleDataType_ = HCCL_DATA_TYPE_FP8E8M0;
    static constexpr uint64_t dataDataType_ = GetHcclDataType<AType>();
    static constexpr uint64_t MXFP_DATA_NUM_PER_BYTE = 2UL;
};

template <typename AType, typename BType, typename CType, AscendC::HcclServerType ServerType, bool IsMxFp4, bool TransB>
__aicore__ inline void
AllGatherMxQuantMatmulHcommImpl<AType, BType, CType, ServerType, IsMxFp4, TransB>::InitBaseParams()
{
    const auto& ct = tilingData_->commTile;
    tileCnt_ = static_cast<uint32_t>(ct.splitAxisTileCnt);
    tileM_ = static_cast<uint32_t>(ct.splitAxisTileSize);
    tailCnt_ = static_cast<uint32_t>(ct.splitAxisTailCnt);
    tailM_ = static_cast<uint32_t>(ct.splitAxisTailSize);
    k_ = tilingData_->mmTile.k;
    n_ = tilingData_->mmTile.n;
    m_ = static_cast<uint32_t>(ct.splitAxisTileSize * ct.splitAxisTileCnt + ct.splitAxisTailSize * ct.splitAxisTailCnt);
    commTurn_ = tileCnt_ + tailCnt_;
    paddedTailM_ = (tailM_ > 0) ? ((tailM_ + kPaddingLength - 1) / kPaddingLength * kPaddingLength) : 0U;
    headRows_ = static_cast<uint64_t>(tileCnt_) * tileM_;

    scaleKGroups_ = Blaze::Gemm::CeilDiv(static_cast<uint64_t>(k_), Blaze::Gemm::MXFP_DIVISOR_SIZE);
    scaleKLen_ = scaleKGroups_ * static_cast<uint64_t>(SCALE_C0);
    // FP4 / HiF4 (fp4x2 / hifloat4x2) 1 字节装 2 个元素：每行字节数 = k/2；FP8 等 = k * sizeof(AType)（与 URMA impl
    // 契约一致） 以类模板参数 IsMxFp4 或 IS_HIF4 为判断源（与 kForComm_ 一致），保证所有实例化场景 AType 与 4bit
    // 类型配对
    dataBytesPerMRow_ =
        (IsMxFp4 || IS_HIF4) ? (static_cast<uint64_t>(k_) >> 1) : static_cast<uint64_t>(k_) * sizeof(AType);
    scaleBytesPerMRow_ = scaleKGroups_ * static_cast<uint64_t>(SCALE_C0) * sizeof(ScaleType);
    cBytesPerM_ = static_cast<uint64_t>(n_) * sizeof(CType);
    // FP4 / HiF4: 2 个元素打包为 1 字节，通信 count/stride 需用 CeilDiv(K, 2) 减半
    kForComm_ = static_cast<uint64_t>(k_);
    if constexpr (IsMxFp4 || IS_HIF4) {
        kForComm_ = Blaze::Gemm::CeilDiv(kForComm_, MXFP_DATA_NUM_PER_BYTE);
    }
}

template <typename AType, typename BType, typename CType, AscendC::HcclServerType ServerType, bool IsMxFp4, bool TransB>
__aicore__ inline void AllGatherMxQuantMatmulHcommImpl<AType, BType, CType, ServerType, IsMxFp4, TransB>::Init(
    GM_ADDR aGM, GM_ADDR aScaleGM, GM_ADDR bGM, GM_ADDR bScaleGM, GM_ADDR biasGM, GM_ADDR cGM, GM_ADDR gatherOut,
    GM_ADDR workspaceGM)
{
    aGM_ = aGM;
    aScaleGM_ = aScaleGM;
    bGM_ = bGM;
    bScaleGM_ = bScaleGM;
    biasGM_ = biasGM;
    cGM_ = cGM;
    workspaceGM_ = workspaceGM;

    InitBaseParams();

    // HCCL initialization
    commState_.hccl_.InitV2(AscendC::GetHcclContext<0>(), &(tilingData_->mc2InitTiling));
    commState_.hccl_.SetCcTilingV2(static_cast<uint64_t>(offsetof(Apace::hcommAllGatherMatmulTilingData, mc2CcTiling)));
    rankId_ = commState_.hccl_.GetRankId();
    rankSize_ = commState_.hccl_.GetRankDim();

    // Gather buffer: scale 始终用 workspace; data 用 gatherOut（当可用）或 workspace
    scaleRegionBytes_ = static_cast<uint64_t>(rankSize_) * m_ * scaleBytesPerMRow_;
    dataRegionBytes_ = static_cast<uint64_t>(rankSize_) * m_ * dataBytesPerMRow_;
    gatherScaleAddr_ = workspaceGM_;
    if (tilingData_->gatherLen == 0 && gatherOut != 0) {
        gatherDataAddr_ = gatherOut;
    } else {
        gatherDataAddr_ = workspaceGM_ + scaleRegionBytes_;
    }

    // Submit AllGather (non-blocking <true>)
    CommitAllGather();

#if MC2_DFX_ENABLE
    opStateDump_.Init(workspaceGM_, &tilingData_->dumpInfo.workspaceLayout, tilingData_->mmTile.usedCoreNum);
#endif
    // Bind BufferSyncManager to CommState（hcomm 非 win 复用路径，slotNum 无效）
    quantMatmulKernelImpl_.GetBufferSyncMgr().state_ = &commState_;
    commState_.tileCnt_ = tileCnt_;
}

template <typename AType, typename BType, typename CType, AscendC::HcclServerType ServerType, bool IsMxFp4, bool TransB>
__aicore__ inline void
AllGatherMxQuantMatmulHcommImpl<AType, BType, CType, ServerType, IsMxFp4, TransB>::CommitAllGather()
{
    // Scale: 全量一次性提交（repeat=1）；hif4_scale 按 E8M0(1B) 传输，count/stride 为字节当量
    // （= scale 元素数 * sizeof(ScaleType)，与 mc2CcTiling 的 1B dtype 口径一致；总字节数不变）
    commState_.scaleHandle_ = commState_.hccl_.template AllGather<true>(
        aScaleGM_, gatherScaleAddr_, static_cast<uint64_t>(m_) * scaleKLen_ * sizeof(ScaleType),
        static_cast<AscendC::HcclDataType>(scaleDataType_), static_cast<uint64_t>(m_) * scaleKLen_ * sizeof(ScaleType),
        static_cast<uint8_t>(1));

    // Data Head: 批量提交所有 head tile（repeat=tileCnt）；纯尾块路径（tileCnt=0）跳过
    // count/stride 用 kForComm_（FP4 = K/2 打包字节），地址偏移用 dataBytesPerMRow_（数值与 kForComm_ 一致）
    if (tileCnt_ > 0) {
        commState_.dataHeadHandle_ = commState_.hccl_.template AllGather<true>(
            aGM_, gatherDataAddr_, static_cast<uint64_t>(tileM_) * kForComm_,
            static_cast<AscendC::HcclDataType>(dataDataType_), static_cast<uint64_t>(m_) * kForComm_,
            static_cast<uint8_t>(tileCnt_));
    }

    // Data Tail: 批量提交所有 tail tile（repeat=tailCnt）
    if (tailCnt_ > 0) {
        commState_.dataTailHandle_ = commState_.hccl_.template AllGather<true>(
            aGM_ + headRows_ * dataBytesPerMRow_, gatherDataAddr_ + headRows_ * dataBytesPerMRow_,
            static_cast<uint64_t>(tailM_) * kForComm_, static_cast<AscendC::HcclDataType>(dataDataType_),
            static_cast<uint64_t>(m_) * kForComm_, static_cast<uint8_t>(tailCnt_));
    }
}

template <typename AType, typename BType, typename CType, AscendC::HcclServerType ServerType, bool IsMxFp4, bool TransB>
__aicore__ inline void
AllGatherMxQuantMatmulHcommImpl<AType, BType, CType, ServerType, IsMxFp4, TransB>::SetupKernelParams(Params& params)
{
    const auto& mmT = tilingData_->mmTile;
    params.schParams = BlockSchedulerParams{
        mmT.baseM,     mmT.baseN,    mmT.mTailTile, mmT.nTailTile, mmT.mBaseTailSplitCnt, mmT.nBaseTailSplitCnt,
        mmT.mTailMain, mmT.nTailMain};
    params.qbmmParams = {mmT.baseM, mmT.baseN, mmT.baseK, mmT.dbL0c};
    params.l1Params = FragL1Params{static_cast<uint64_t>(mmT.stepK) * mmT.baseK, mmT.scaleKL1, mmT.nBufferNum};
    params.fragParams = {tileCnt_,
                         tileM_,
                         tailCnt_,
                         tailM_,
                         paddedTailM_,
                         commTurn_,
                         headRows_,
                         rankId_,
                         rankSize_,
                         m_,
                         static_cast<uint64_t>(k_),
                         static_cast<uint64_t>(n_),
                         scaleKLen_};
    params.mmadParams = {aGM_, bGM_, cGM_, biasGM_, aScaleGM_, bScaleGM_};
    params.winDataBase = gatherDataAddr_;
    params.winScaleBase = gatherScaleAddr_;
    params.dataBytesPerMRow = dataBytesPerMRow_;
    params.scaleBytesPerMRow = scaleBytesPerMRow_;
    params.cBytesPerM = cBytesPerM_;
}

template <typename AType, typename BType, typename CType, AscendC::HcclServerType ServerType, bool IsMxFp4, bool TransB>
__aicore__ inline void AllGatherMxQuantMatmulHcommImpl<AType, BType, CType, ServerType, IsMxFp4, TransB>::Run()
{
    Params params;
    SetupKernelParams(params);
    quantMatmulKernelImpl_(params, opStateDump_);
    AscendC::CrossCoreSetFlag<0, PIPE_FIX>(3);
    AscendC::CrossCoreWaitFlag(3);
    commState_.hccl_.Finalize();
}

} // namespace Apace
