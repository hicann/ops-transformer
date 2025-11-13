/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file grouped_mat_mul_allto_allv.h
 * \brief
 */
#ifndef GROUPED_MAT_MUL_ALLTO_ALLV_H
#define GROUPED_MAT_MUL_ALLTO_ALLV_H

#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#include "lib/matmul_intf.h"
#include "grouped_mat_mul_allto_allv_tiling.h"

#if __has_include("../../allto_allv_grouped_mat_mul/op_kernel/allto_allv_gmm.h")
#include "../../allto_allv_grouped_mat_mul/op_kernel/allto_allv_gmm.h"
#else
#include "../allto_allv_grouped_mat_mul/allto_allv_gmm.h"
#endif
#include "grouped_mat_mul_allto_allv_tiling.h"

namespace AscendC {
using namespace ALLTO_ALLV_GMM;
template <typename GMMATAV>
class GroupedMatmulAlltoAllv
{
public:
    __aicore__ inline GroupedMatmulAlltoAllv()
    {}
    __aicore__ inline void Init(
        GM_ADDR gmmxGM, GM_ADDR gmmweightGM, GM_ADDR sendCountsTensorGM, GM_ADDR recvCountsTensorGM, GM_ADDR mmGM,
        GM_ADDR mmweightGM, GM_ADDR yGM, GM_ADDR mmyGM, GM_ADDR workspaceGM, GM_ADDR contextGM,
        const GroupedMatMulAlltoAllvTilingData* tilingData, __gm__ void* hcclInitTiling, __gm__ void* alltoAllvCcTiling,
        TPipe* tPipe);
    __aicore__ inline void Process();
    using X_T = typename GMMATAV::xType;
    static constexpr bool NEED_MM = GMMATAV::isOptionalMm;
    static constexpr bool NEED_GMMW_TRANS = GMMATAV::isGmmWeightTrans;
    static constexpr bool NEED_MMW_TRANS = GMMATAV::isOptWeightTrans;

    using aType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, X_T, false>;
    using gmmBType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, X_T, NEED_GMMW_TRANS>;
    using mmBType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, X_T, NEED_MMW_TRANS>;
    using cType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, X_T, false>;
    using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, X_T>;
    using gmmType = ALLTO_ALLV_GMM::MMImplType<aType, gmmBType, cType, biasType, CFG_MDL>;
    using sharedmmType = ALLTO_ALLV_GMM::MMImplType<aType, mmBType, cType, biasType, CFG_MDL>;

private:
    __aicore__ inline void HcclAlltoAllvPrepare();
    __aicore__ inline void HcclFinalize();
    __aicore__ inline void GmmProcessAlltoallv();
    __aicore__ inline void ShareMatMulCompute();

    GM_ADDR gmmxGM_;
    GM_ADDR gmmweightGM_;
    GM_ADDR mmxGM_;
    GM_ADDR mmweightGM_;
    GM_ADDR yGM_;
    GM_ADDR mmyGM_;
    GM_ADDR allGatherOutGM_;
    GM_ADDR gmmOutGM_;
    GM_ADDR workspaceGM_;
    const GroupedMatMulAlltoAllvTilingData* tilingData_ = nullptr;
    TCubeTiling matmulTiling_;
    TCubeTiling sharedMatmulTiling_;

    TPipe* tPipe_;
    uint64_t rankId_{0};
    uint64_t rankDim_{8};

    uint64_t expertNumInOneRank_ = 0U; // 单卡专家数
    uint64_t expertNumAll_ = 0U;       // 通信域内专家数

    uint64_t axisBsK_ = 0U;
    uint64_t axisH_ = 0U;
    uint64_t axisA_ = 0U;
    uint64_t axisN1_ = 0U;
    uint64_t workSpaceOffset = 0U;

    // alltoallv 流程数据结构
    static constexpr uint64_t MAX_EP_RANK_SIZE = 64U;
    static constexpr uint64_t TOTAL_UBSIZE = static_cast<uint64_t>(190U * 1024U / 2U);
    static constexpr uint64_t MAX_AIV_NUM = 48U;
    static constexpr uint64_t MAX_HANDLE_ID_NUM = 64U;
    Hccl<HCCL_SERVER_TYPE_AICPU> hccl_;
    HcclHandle allGatherHandleId_{INVALID_HANDLE_ID};
    HcclHandle alltoAllvHandleId_[MAX_HANDLE_ID_NUM] = {INVALID_HANDLE_ID};
    HcclDataType hcclDataType_;

    GlobalTensor<int64_t> allGatherOutGMTensor_;
    GlobalTensor<X_T> gmmXGMTensor_;
    GlobalTensor<X_T> gmmWeightGMTensor_;
    GlobalTensor<X_T> gmmOutGMTensor_;
    GlobalTensor<X_T> yOutGMTensor_;
    GlobalTensor<X_T> mmXGMTensor_;
    GlobalTensor<X_T> mmWeightGMTensor_;
    GlobalTensor<X_T> mmOutGMTensor_;
    typename gmmType::MT gmm;
    typename sharedmmType::MT sharedmm;
};
template <typename GMMATAV>
__aicore__ inline void GroupedMatmulAlltoAllv<GMMATAV>::Init(
    GM_ADDR gmmxGM, GM_ADDR gmmweightGM, GM_ADDR sendCountsTensorGM, GM_ADDR recvCountsTensorGM, GM_ADDR mmGM,
    GM_ADDR mmweightGM, GM_ADDR yGM, GM_ADDR mmyGM, GM_ADDR workspaceGM, GM_ADDR contextGM,
    const GroupedMatMulAlltoAllvTilingData* tilingData, __gm__ void* hcclInitTiling, __gm__ void* alltoAllvCcTiling,
    TPipe* tPipe)
{
    gmmxGM_ = gmmxGM;
    gmmweightGM_ = gmmweightGM;
    mmxGM_ = mmGM;
    mmweightGM_ = mmweightGM;
    yGM_ = yGM;
    mmyGM_ = mmyGM;
    gmmOutGM_ = workspaceGM;
    workspaceGM_ = workspaceGM;
    tilingData_ = tilingData;
    tPipe_ = tPipe;

    axisBsK_ = tilingData_->commonTilingInfo.BsK;
    axisH_ = tilingData_->commonTilingInfo.H;
    axisA_ = tilingData_->commonTilingInfo.A;
    axisN1_ = tilingData_->commonTilingInfo.N1;

    hccl_.Init(contextGM, hcclInitTiling);
    hccl_.SetCcTiling(alltoAllvCcTiling);
    rankId_ = hccl_.GetRankId();
    rankDim_ = hccl_.GetRankDim();

    expertNumInOneRank_ = tilingData_->commonTilingInfo.E_ep;
    expertNumAll_ = expertNumInOneRank_ * rankDim_;
    matmulTiling_ = tilingData_->matmulTiling;
    sharedMatmulTiling_ = tilingData_->sharedExpMatmulTiling;

    if (AscendC::IsSameType<X_T, bfloat16_t>::value) {
        hcclDataType_ = HcclDataType::HCCL_DATA_TYPE_BFP16;
    } else {
        hcclDataType_ = HcclDataType::HCCL_DATA_TYPE_FP16;
    }

    gmmXGMTensor_.SetGlobalBuffer((__gm__ X_T*)this->gmmxGM_);
    gmmWeightGMTensor_.SetGlobalBuffer((__gm__ X_T*)this->gmmweightGM_);
    gmmOutGMTensor_.SetGlobalBuffer((__gm__ X_T*)workspaceGM_);

    mmXGMTensor_.SetGlobalBuffer((__gm__ X_T*)this->mmxGM_);
    mmWeightGMTensor_.SetGlobalBuffer((__gm__ X_T*)this->mmweightGM_);
    mmOutGMTensor_.SetGlobalBuffer((__gm__ X_T*)this->mmyGM_);
    yOutGMTensor_.SetGlobalBuffer((__gm__ X_T*)this->yGM_);
}

template <typename GMMATAV>
__aicore__ inline void GroupedMatmulAlltoAllv<GMMATAV>::Process()
{
    HcclAlltoAllvPrepare(); // alltoall prepare
    // gmm + alltoall，做完一块gmm下发一块alltoall
    GmmProcessAlltoallv();
    if ASCEND_IS_AIV {
        if (GetBlockIdx() == 0) {
            for (uint64_t i = 0; i < expertNumInOneRank_; i++) {
                hccl_.Wait(alltoAllvHandleId_[i]);
            }
        }
    }
    HcclFinalize();
}

template <typename GMMATAV>
__aicore__ inline void GroupedMatmulAlltoAllv<GMMATAV>::ShareMatMulCompute()
{
    if (NEED_MM) {
        if ASCEND_IS_AIC {
            uint64_t aSharedOffset = 0U;
            uint64_t bSharedOffset = 0U;
            uint64_t cSharedOffset = 0U;

            sharedmm.Init(&(tilingData_->sharedExpMatmulTiling));
            GMMCompute<sharedmmType> computeOp2(sharedmm);
            computeOp2.Init(mmxGM_, mmweightGM_, mmyGM_);
            GMMProcess<decltype(computeOp2)> mmOp(computeOp2);
            mmOp.Init(tilingData_->sharedExpMatmulTiling.baseM, tilingData_->sharedExpMatmulTiling.baseN, 24);

            uint64_t mmInOffset[1] = {0};
            uint64_t mmOutOffset[1] = {0};
            uint64_t mmWeightOffset[1] = {0};
            uint32_t tokenNum[1] = {static_cast<uint32_t>(tilingData_->commonTilingInfo.Bs)};
            mmOp.Process(
                this->rankId_, tilingData_->commonTilingInfo.sharedMatmulH, tilingData_->commonTilingInfo.N2,
                mmInOffset, mmOutOffset, tokenNum, 1, 0);
        }
    }
}

template <typename GMMATAV>
__aicore__ inline void GroupedMatmulAlltoAllv<GMMATAV>::GmmProcessAlltoallv()
{
    gmm.Init(&(tilingData_->matmulTiling));
    GMMCompute<gmmType> computeOp(gmm);
    computeOp.Init(gmmxGM_, gmmweightGM_, gmmOutGM_);
    GMMProcess<decltype(computeOp)> gmmOp(computeOp);
    gmmOp.Init(tilingData_->matmulTiling.baseM, tilingData_->matmulTiling.baseN, 24);
    auto* sendCnt = &tilingData_->aicpuTilingInfo.sendCnt[0];

    uint64_t mmInOffset[2] = {0};
    uint64_t mmOutOffset[2] = {0};
    uint64_t mmWeightOffset[2] = {0};
    uint32_t tokenNum[2] = {0};
    uint32_t processNum = 1;

    uint64_t expertoffset[MAX_HANDLE_ID_NUM] = {0UL};
    uint64_t gmmTokennum[MAX_HANDLE_ID_NUM] = {0UL};
    for (uint64_t i = 0UL; i < expertNumInOneRank_; i++) {
        uint64_t cur_gmmTokenNum = 0;
        for (uint64_t j = 0; j < rankDim_; j++) {
            cur_gmmTokenNum += static_cast<uint64_t>(sendCnt[i + j * expertNumInOneRank_]);
        }
        gmmTokennum[i] = cur_gmmTokenNum; // 本专家的tokenNum
    }
    for (uint64_t i = 0UL; i < expertNumInOneRank_; i++) {
        if (i >= 1UL) {
            expertoffset[i] = expertoffset[i - 1] + gmmTokennum[i - 1] * axisH_;
        }
    }
    for (uint64_t i = 0UL; i < expertNumInOneRank_; i++) {
        mmInOffset[0] = expertoffset[i];
        mmOutOffset[0] = expertoffset[i] / axisH_ * axisN1_;
        mmWeightOffset[0] = i * axisH_ * axisN1_;
        tokenNum[0] = gmmTokennum[i];
        if ASCEND_IS_AIC {
            gmmOp.Process(this->rankId_, axisH_, axisN1_, mmInOffset, mmOutOffset, tokenNum, processNum, i);
        }
        SyncAll<false>();
        if ASCEND_IS_AIV {
            if (GetBlockIdx() == 0) {
                hccl_.Commit(alltoAllvHandleId_[i]);
            }
        }
    }

    ShareMatMulCompute();
    SyncAll<false>();
}

template <typename GMMATAV>
__aicore__ inline void GroupedMatmulAlltoAllv<GMMATAV>::HcclAlltoAllvPrepare()
{
    if ASCEND_IS_AIV {
        if (GetBlockIdx() != 0) {
            return;
        }

        const auto* sendCnt = &tilingData_->aicpuTilingInfo.sendCnt[0];
        const auto* recvCnt = &tilingData_->aicpuTilingInfo.recvCnt[0];
        for (uint64_t e = 0UL; e < expertNumInOneRank_; e++) {
            uint64_t alltoAllvSendCnt[MAX_EP_RANK_SIZE] = {0UL};
            uint64_t alltoAllvSendOffset[MAX_EP_RANK_SIZE] = {0UL};
            uint64_t alltoAllvRecvCnt[MAX_EP_RANK_SIZE] = {0UL};
            uint64_t alltoAllvRecvOffset[MAX_EP_RANK_SIZE] = {0UL};
            // 计算sendcnts：根据偏移取sendCnt
            for (uint64_t i = 0UL; i < rankDim_; i++) {
                alltoAllvSendCnt[i] = static_cast<uint64_t>(sendCnt[e + i * expertNumInOneRank_]) * axisN1_;
            }
            // 计算sendoffset：本卡上专家数偏移+前面卡token数的和 -- 维护每张卡每个专家总长度
            uint64_t expertOffset = 0UL; // 发送数据卡当前发送到第e个专家，前面e-1专家的偏移
            for (uint64_t i = 0UL; i < e; i++) {            // 遍历专家
                for (uint64_t j = 0UL; j < rankDim_; j++) { // 遍历每个专家所有卡上的token数
                    expertOffset += static_cast<uint64_t>(sendCnt[i + j * expertNumInOneRank_]);
                }
            }
            // 要发送的专家内，发送到第i张卡，把前面i-1张卡偏掉,按卡数累加，每次只要上次的地址+上张卡的token偏移
            alltoAllvSendOffset[0] = expertOffset * axisN1_;
            for (uint64_t i = 1UL; i < rankDim_; i++) {
                alltoAllvSendOffset[i] = alltoAllvSendOffset[i - 1] +
                                         static_cast<uint64_t>(sendCnt[e + (i - 1) * expertNumInOneRank_]) * axisN1_;
            }
            // 计算recvcnts：根据偏移取recvCnt --- 0,4,8,12,...252; 1,5,9,13,...253; 2,6,10,14,...254; 3,7,11,15,...,255
            for (uint64_t i = 0UL; i < rankDim_; i++) {
                alltoAllvRecvCnt[i] = static_cast<uint64_t>(recvCnt[e + i * expertNumInOneRank_]) * axisN1_;
            }
            // 计算rcvoffset：前面专家token数和---确定e轮次，之后每卡向后偏移4个专家的token数
            // 0,4,8,12,...252; 1,5,9,13,...253; 2,6,10,14,...254; 3,7,11,15,...,255
            alltoAllvRecvOffset[0] = 0;
            for (uint64_t i = 0UL; i < e; i++) { // 确定rank0的轮次，0/1/2/3，把前面的专家偏过去，确认起始地址
                alltoAllvRecvOffset[0] += static_cast<uint64_t>(recvCnt[i]) * axisN1_;
            }
            for (uint64_t i = 1UL; i < rankDim_; i++) { // 确定rank1~63的轮次，每次在上次的基础上向后偏4个专家的token数
                alltoAllvRecvOffset[i] = alltoAllvRecvOffset[i - 1];
                for (uint64_t j = 0UL; j < expertNumInOneRank_; j++) {
                    alltoAllvRecvOffset[i] +=
                        static_cast<uint64_t>(recvCnt[e + (i - 1) * expertNumInOneRank_ + j]) * axisN1_;
                }
            }

            alltoAllvHandleId_[e] = hccl_.AlltoAllV<false>(
                (__gm__ uint8_t*)gmmOutGMTensor_.GetPhyAddr(), alltoAllvSendCnt, alltoAllvSendOffset, hcclDataType_,
                (__gm__ uint8_t*)yOutGMTensor_.GetPhyAddr(), alltoAllvRecvCnt, alltoAllvRecvOffset, hcclDataType_);
        }
    }
}

template <typename GMMATAV>
__aicore__ inline void GroupedMatmulAlltoAllv<GMMATAV>::HcclFinalize()
{
    if ASCEND_IS_AIV {
        hccl_.Finalize();
    }
}
} // namespace AscendC
#endif
