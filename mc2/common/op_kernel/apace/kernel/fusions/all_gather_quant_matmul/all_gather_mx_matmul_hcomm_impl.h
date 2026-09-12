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
 *        GetCommPolicy().state_ = &commState_ 绑定 WaitPolicy。
 *
 * Run():
 *   AIC: 执行 FragmentTensor kernel，所有核 WaitTile 完成后再统一 Finalize → hccl_.Finalize。
 *        kernel 内逐 tile commPolicy_.WaitTile(dependTileIdx) → state_->hccl_.Wait(handle)。
 */

#pragma once

#include "include/tensor_api/tensor.h"
#include "lib/hccl/hccl.h"
#include "blaze/gemm/utils/common_utils.h"
#include "apace/kernel/matmul/quant_batch_matmul/all_gather_qbmm_mx_kernel.h"
#include "apace/kernel/fusions/all_gather_quant_matmul/all_gather_mx_matmul_tiling_data.h"
#include "apace/tiling/quant_matmul_tiling_data.h"

namespace Apace {

using namespace AscendC;

template <typename T>
constexpr uint64_t GetHcclDataType()
{
    if constexpr (AscendC::IsSameType<T, float8_e5m2_t>::value) {
        return AscendC::HCCL_DATA_TYPE_FP8E5M2;
    } else if constexpr (AscendC::IsSameType<T, fp4x2_e2m1_t>::value) {
        return AscendC::HCCL_DATA_TYPE_UINT8;
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
    HcommCommState<ServerType> *state_{nullptr};

    __aicore__ inline void WaitTile(uint32_t tileIdx)
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
};

template <typename AType, typename BType, typename CType, AscendC::HcclServerType ServerType, bool IsMxFp4>
class AllGatherMxQuantMatmulHcommImpl {
public:
    explicit __aicore__ inline AllGatherMxQuantMatmulHcommImpl(Apace::hcommAllGatherMatmulTilingData *tilingData)
        : tilingData_(tilingData)
    {}

    __aicore__ inline void Init(GM_ADDR aGM, GM_ADDR aScaleGM, GM_ADDR bGM, GM_ADDR bScaleGM, GM_ADDR biasGM,
                                GM_ADDR cGM, GM_ADDR gatherOut, GM_ADDR workspaceGM);
    __aicore__ inline void Run();

    using QuantMatmulKernelImpl = AllGatherQbmmMxKernel<AType, BType, CType, HcommCommWaitPolicy<ServerType>>;
    using KernelParams = typename QuantMatmulKernelImpl::Params;

    QuantMatmulKernelImpl quantMatmulKernelImpl_;
    HcommCommState<ServerType> commState_;
    Mc2Kernel::OpStateDump opStateDump_;

private:
    __aicore__ inline void InitBaseParams();
    __aicore__ inline void CommitAllGather();
    __aicore__ inline void SetupKernelParams(KernelParams &params);

    Apace::hcommAllGatherMatmulTilingData *tilingData_;

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

    static constexpr uint32_t kPaddingLength = 16;
    static constexpr uint64_t scaleDataType_ = HCCL_DATA_TYPE_FP8E8M0;
    static constexpr uint64_t dataDataType_ = GetHcclDataType<AType>();
    static constexpr uint64_t MXFP_DATA_NUM_PER_BYTE = 2UL;
};

template <typename AType, typename BType, typename CType, AscendC::HcclServerType ServerType, bool IsMxFp4>
__aicore__ inline void AllGatherMxQuantMatmulHcommImpl<AType, BType, CType, ServerType, IsMxFp4>::InitBaseParams()
{
    const auto &ct = tilingData_->commTile;
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
    scaleKLen_ = scaleKGroups_ * static_cast<uint64_t>(Blaze::Gemm::MXFP_MULTI_BASE_SIZE);
    // FP4 (fp4x2) 1 字节装 2 个元素：每行字节数 = k/2；FP8 等 = k * sizeof(AType)（与 URMA impl 契约一致）
    // 以类模板参数 IsMxFp4 为唯一判断源（与 kForComm_ 一致），保证所有实例化场景 AType 与 IsMxFp4 配对
    dataBytesPerMRow_ = IsMxFp4 ? (static_cast<uint64_t>(k_) >> 1) : static_cast<uint64_t>(k_) * sizeof(AType);
    scaleBytesPerMRow_ =
        scaleKGroups_ * static_cast<uint64_t>(Blaze::Gemm::MXFP_MULTI_BASE_SIZE) * sizeof(AscendC::fp8_e8m0_t);
    cBytesPerM_ = static_cast<uint64_t>(n_) * sizeof(CType);
    // FP4: 2 个元素打包为 1 字节，通信 count/stride 需用 CeilDiv(K, 2) 减半
    kForComm_ = static_cast<uint64_t>(k_);
    if constexpr (IsMxFp4) {
        kForComm_ = Blaze::Gemm::CeilDiv(kForComm_, MXFP_DATA_NUM_PER_BYTE);
    }
}

template <typename AType, typename BType, typename CType, AscendC::HcclServerType ServerType, bool IsMxFp4>
__aicore__ inline void AllGatherMxQuantMatmulHcommImpl<AType, BType, CType, ServerType, IsMxFp4>::Init(
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
    // Bind CommPolicy to CommState
    quantMatmulKernelImpl_.GetCommPolicy().state_ = &commState_;
    commState_.tileCnt_ = tileCnt_;
}

template <typename AType, typename BType, typename CType, AscendC::HcclServerType ServerType, bool IsMxFp4>
__aicore__ inline void AllGatherMxQuantMatmulHcommImpl<AType, BType, CType, ServerType, IsMxFp4>::CommitAllGather()
{
    // Scale: 全量一次性提交（repeat=1）
    commState_.scaleHandle_ =
        commState_.hccl_.template AllGather<true>(aScaleGM_, gatherScaleAddr_, static_cast<uint64_t>(m_) * scaleKLen_,
                                                  static_cast<AscendC::HcclDataType>(scaleDataType_),
                                                  static_cast<uint64_t>(m_) * scaleKLen_, static_cast<uint8_t>(1));

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

template <typename AType, typename BType, typename CType, AscendC::HcclServerType ServerType, bool IsMxFp4>
__aicore__ inline void AllGatherMxQuantMatmulHcommImpl<AType, BType, CType, ServerType, IsMxFp4>::SetupKernelParams(
    KernelParams &params)
{
    params.mmTile = &(tilingData_->mmTile);
    params.qbmmParams = {tilingData_->mmTile.baseM, tilingData_->mmTile.baseN, tilingData_->mmTile.baseK,
                         tilingData_->mmTile.dbL0c};
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
    params.aGM = aGM_;
    params.aScaleGM = aScaleGM_;
    params.bGM = bGM_;
    params.bScaleGM = bScaleGM_;
    params.biasGM = biasGM_;
    params.isBias = (biasGM_ != nullptr); // bias optional 输入缺失时框架传 nullptr（同 gatherOut 判空惯例）
    params.cGM = cGM_;
    params.winDataBase = gatherDataAddr_;
    params.winScaleBase = gatherScaleAddr_;
    params.dataBytesPerMRow = dataBytesPerMRow_;
    params.scaleBytesPerMRow = scaleBytesPerMRow_;
    params.cBytesPerM = cBytesPerM_;
}

template <typename AType, typename BType, typename CType, AscendC::HcclServerType ServerType, bool IsMxFp4>
__aicore__ inline void AllGatherMxQuantMatmulHcommImpl<AType, BType, CType, ServerType, IsMxFp4>::Run()
{
    KernelParams params;
    SetupKernelParams(params);
    quantMatmulKernelImpl_(params, opStateDump_);
    commState_.hccl_.Finalize();
}

} // namespace Apace
