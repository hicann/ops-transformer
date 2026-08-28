/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file grouped_matmul_add.cpp
 * \brief
 */

#ifdef __CCE_KT_TEST__
#include "stub_def.h"
#include "stub_fun.h"
#endif

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
#include "arch35/grouped_matmul_add_kernel.h"
#else
#include "grouped_matmul_add_kernel.h"
#endif
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

using namespace AscendC;

/* pour changement co */

namespace AscendC {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
#else
using namespace matmul;
constexpr uint32_t thresholdBlockNum = 8;
constexpr uint32_t thresholdDimM = 5;

template <bool Trans = false, typename Dtype = half>
using xType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, Dtype, Trans>;

template <bool Trans = false, typename Dtype = half>
using weightType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, Dtype, Trans>;

using yType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, float>;

/** @brief store variables for core split configuration
 */
struct MNConfig {
    uint32_t m = 0;
    uint32_t k = 0;
    uint32_t n = 0;
    uint32_t baseM = 0;
    uint32_t baseN = 0;
    uint32_t mIdx = 0;
    uint32_t nIdx = 0;
    uint32_t blockDimM = 0;
    uint32_t blockDimN = 0;
    uint32_t singleM = 0;
    uint32_t singleN = 0;
    uint64_t wBaseOffset = 0;
    uint64_t nAxisBaseOffset = 0;
    uint64_t mAxisBaseOffset = 0;
    uint64_t xBaseOffset = 0;
    uint64_t yBaseOffset = 0;
    uint64_t wOutOffset = 0;
    uint64_t workSpaceOffset = 0;
};

template <typename ComputeType>
class GmmAddProcess {
protected:
    ComputeType &computeOp; // inernal computation operator
    const GmmBaseParams *__restrict gmmBaseParams;
    const TCubeTiling *__restrict mmTilingData;

    uint32_t blockIdx;
    uint32_t coreIdx; // 优化版本是cv融合算子，此处的coreIdx就等于blockIdx
    uint32_t groupNum;
    int32_t preOffset;
    GM_ADDR groupListPtr;
    GlobalTensor<int64_t> groupListGm;
    MNConfig lastMnConfig;
    uint32_t lastGroupIdx;
    uint32_t groupListType;

public:
    __aicore__ inline GmmAddProcess(ComputeType &computeOp_)
        : computeOp(computeOp_)
    {}

    __aicore__ inline void Init(const GmmBaseParams *__restrict gmmBaseParamsIn,
                                const TCubeTiling *__restrict mmTilingDataIn, GM_ADDR groupList, GM_ADDR tiling);

    __aicore__ inline void Process();

    __aicore__ inline void SetMNConfig(const int32_t splitValue, MNConfig &mnConfig);

    __aicore__ inline void SetMKN(const int32_t splitValue, MNConfig &mnConfig);

    __aicore__ inline void UpdateMnConfig(MNConfig &mnConfig);

    __aicore__ inline void MnBlockIdxCompute(MNConfig &mnConfig, const uint32_t curBlock, const uint32_t count,
                                             const uint32_t thresholdM_dimN);
    __aicore__ inline uint32_t GreatestCommonDivisor(uint32_t a, uint32_t b);
    __aicore__ inline uint32_t LeastCommonMultiple(uint32_t a, uint32_t b);
};

template <typename ComputeType>
__aicore__ inline uint32_t GmmAddProcess<ComputeType>::GreatestCommonDivisor(uint32_t a, uint32_t b)
{
    uint32_t c = a;
    if (a < b) {
        a = b;
        b = c;
    }
    while (b != 0) {
        c = a;
        a = b;
        b = c % b;
    }
    return a;
}

template <typename ComputeType>
__aicore__ inline uint32_t GmmAddProcess<ComputeType>::LeastCommonMultiple(uint32_t a, uint32_t b)
{
    return a * b / GreatestCommonDivisor(a, b);
}

template <typename ComputeType>
__aicore__ inline void GmmAddProcess<ComputeType>::Init(const GmmBaseParams *__restrict gmmBaseParamsIn,
                                                        const TCubeTiling *__restrict mmTilingDataIn, GM_ADDR groupList,
                                                        GM_ADDR tiling)
{
    blockIdx = GetBlockIdx();
    coreIdx = blockIdx;
    gmmBaseParams = gmmBaseParamsIn;
    mmTilingData = mmTilingDataIn;
    groupNum = static_cast<uint32_t>(gmmBaseParams->groupNum);
    groupListPtr = groupList;
    groupListGm.SetGlobalBuffer((__gm__ int64_t *)groupList);
    groupListType = static_cast<uint32_t>(gmmBaseParams->groupListType);
}

template <typename ComputeType>
__aicore__ inline void GmmAddProcess<ComputeType>::SetMNConfig(const int32_t splitValue, MNConfig &mnConfig)
{
    SetMKN(splitValue, mnConfig);
    mnConfig.baseM = mmTilingData->baseM;
    mnConfig.baseN = mmTilingData->baseN;
    mnConfig.singleM = mnConfig.baseM;
    mnConfig.singleN = mnConfig.baseN;
}

template <typename ComputeType>
__aicore__ inline void GmmAddProcess<ComputeType>::SetMKN(const int32_t splitValue, MNConfig &mnConfig)
{
    mnConfig.m = mmTilingData->M;
    mnConfig.k = splitValue;
    mnConfig.n = mmTilingData->N;
    return;
}

template <typename ComputeType>
__aicore__ inline void GmmAddProcess<ComputeType>::UpdateMnConfig(MNConfig &mnConfig)
{
    mnConfig.wBaseOffset += static_cast<uint64_t>(mnConfig.k) * mnConfig.n;
    mnConfig.nAxisBaseOffset += mnConfig.n;
    mnConfig.mAxisBaseOffset += mnConfig.m;
    mnConfig.xBaseOffset += static_cast<uint64_t>(mnConfig.m) * mnConfig.k;
    mnConfig.yBaseOffset += static_cast<uint64_t>(mnConfig.m) * mnConfig.n;
}

template <typename ComputeType>
__aicore__ inline void GmmAddProcess<ComputeType>::MnBlockIdxCompute(MNConfig &mnConfig, const uint32_t curBlock,
                                                                     const uint32_t count,
                                                                     const uint32_t thresholdM_dimN)
{
    if (mnConfig.blockDimM <= thresholdDimM || thresholdDimM == 1) {
        mnConfig.mIdx = (curBlock - count) / mnConfig.blockDimN;
        mnConfig.nIdx = (curBlock - count) % mnConfig.blockDimN;
    } else {
        uint32_t relativeBlock = curBlock - count;
        uint32_t curThresholdM = relativeBlock >= AlignDown(mnConfig.blockDimM * mnConfig.blockDimN, thresholdM_dimN) ?
                                     mnConfig.blockDimM % thresholdBlockNum :
                                     thresholdBlockNum;
        uint32_t curThresholdM_thresholdN = curThresholdM * thresholdBlockNum;
        uint32_t curThresholdN =
            relativeBlock % thresholdM_dimN >= AlignDown(curThresholdM * mnConfig.blockDimN, curThresholdM_thresholdN) ?
                mnConfig.blockDimN % thresholdBlockNum :
                thresholdBlockNum;

        uint32_t localRelativeBlock = relativeBlock % thresholdM_dimN % curThresholdM_thresholdN;
        mnConfig.mIdx = localRelativeBlock % curThresholdM + relativeBlock / thresholdM_dimN * thresholdBlockNum;
        mnConfig.nIdx = (localRelativeBlock + localRelativeBlock / LeastCommonMultiple(curThresholdM, curThresholdN)) %
                            curThresholdN +
                        relativeBlock % thresholdM_dimN / curThresholdM_thresholdN * thresholdBlockNum;
    }
}

template <typename ComputeType>
__aicore__ inline void GmmAddProcess<ComputeType>::Process()
{
    MNConfig mnConfig;
    MNConfig mnPreConfig;
    if (gmmBaseParams->groupType != -1) { // -1: no split
        preOffset = 0;
        if (unlikely(groupListPtr == nullptr)) {
            groupNum = 0; // not continue Process
        }
    }

    for (uint32_t groupIdx = 0, count = 0; groupIdx < groupNum; ++groupIdx) {
        int32_t splitValue = GetSplitValueFromGroupList(groupIdx, preOffset, groupListType, groupListGm);
        SetMNConfig(splitValue, mnConfig);
        uint32_t dimM = Ceil(mnConfig.m, mnConfig.singleM);
        uint32_t dimN = Ceil(mnConfig.n, mnConfig.singleN);
        mnConfig.blockDimM = dimM;
        mnConfig.blockDimN = dimN;
        if constexpr (ComputeType::transposeX) {
            if (mnConfig.k == 0) {
                UpdateMnConfig(mnConfig);
                continue;
            }
        }

        uint32_t curCount = count + dimM * dimN;
        uint32_t curBlock = coreIdx >= count ? coreIdx : coreIdx + gmmBaseParams->coreNum;
        uint32_t thresholdM_dimN = thresholdBlockNum * dimN;
        while (curBlock < curCount) {
            MnBlockIdxCompute(mnConfig, curBlock, count, thresholdM_dimN);
            /*先准备数据再计算有助于优化流水*/
            computeOp.MmCompute(lastMnConfig, blockIdx);
            computeOp.MmComputePrepare(groupIdx, mnConfig);
            lastGroupIdx = groupIdx;
            lastMnConfig = mnConfig;
            curBlock += gmmBaseParams->coreNum;
            mnPreConfig = mnConfig;
        }
        UpdateMnConfig(mnConfig);
        count = curCount % gmmBaseParams->coreNum;
    }
    computeOp.MmCompute(lastMnConfig, blockIdx);
}

template <class mmType, bool sync = false>
class GmmAddCompute {
public:
    using AT = typename mmType::AT::T;
    using BT = typename mmType::BT::T;
    using CT = typename mmType::CT::T;
    constexpr static bool transposeX = mmType::AT::isTrans;
    constexpr static bool transposeW = mmType::BT::isTrans;

    /** @brief constructor */
    __aicore__ inline GmmAddCompute(typename mmType::MT &mm_)
        : mm(mm_)
    {}

    __aicore__ inline ~GmmAddCompute()
    {
        queIn.FreeTensor(tensorIn);
        return;
    }

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR groupList, GM_ADDR y,
                                const GmmBaseParams *__restrict gmmBaseParams,
                                const TCubeTiling *__restrict mmTilingData, TPipe *tPipe, GM_ADDR yRef);

    __aicore__ inline void MmComputePrepare(uint32_t groupIdx, MNConfig &mnConfig);

    __aicore__ inline void MmCompute(MNConfig &mnConfig, uint32_t blockIdx);

    __aicore__ inline GlobalTensor<BT> SetGlobalBufferW(uint32_t groupIdx, uint32_t tailN, MNConfig &mnConfig);

protected:
    TPipe *pipe;
    typename mmType::MT &mm; // matmul operator
    bool hasBias = false;
    GM_ADDR xTensorPtr;
    GM_ADDR weightTensorPtr;
    GM_ADDR yTensorPtr;
    GM_ADDR yRefTensorPtr;
    GlobalTensor<AT> xGm;
    GlobalTensor<BT> weightGm;
    GlobalTensor<CT> yGm;
    GlobalTensor<CT> yRefGm;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> queIn; // 创建CopyIn阶段的队列
    AscendC::LocalTensor<CT> tensorIn;
    uint32_t cubeNum; // 核上已准备完成的matmul次数
    uint32_t aicIdx;
    uint32_t coreNum;
    bool mmWaitStatus;
    int32_t eventIDMTE2ToMTE3;
};

template <typename mmType, bool sync>
__aicore__ inline void GmmAddCompute<mmType, sync>::Init(GM_ADDR x, GM_ADDR weight, GM_ADDR groupList, GM_ADDR y,
                                                         const GmmBaseParams *__restrict gmmBaseParams,
                                                         const TCubeTiling *__restrict mmTilingData, TPipe *tPipe,
                                                         GM_ADDR yRef)
{
    xTensorPtr = x;
    weightTensorPtr = weight;
    yTensorPtr = y;
    yRefTensorPtr = yRef;
    pipe = tPipe;
    coreNum = static_cast<uint32_t>(gmmBaseParams->coreNum);
    mmWaitStatus = false;
    cubeNum = 0;
    pipe->InitBuffer(queIn, 1, mmTilingData->baseM * mmTilingData->baseN * sizeof(CT));
    tensorIn = queIn.AllocTensor<CT>();
    eventIDMTE2ToMTE3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_MTE3));
}

template <typename mmType, bool sync>
__aicore__ inline GlobalTensor<typename mmType::BT::T> GmmAddCompute<mmType, sync>::SetGlobalBufferW(uint32_t groupIdx,
                                                                                                     uint32_t tailN,
                                                                                                     MNConfig &mnConfig)
{
    uint64_t wOffset;
    if constexpr (transposeW) {
        wOffset = tailN * mnConfig.k;
    } else {
        wOffset = tailN;
    }
    GlobalTensor<BT> weightGmLocal;
    weightGmLocal.SetGlobalBuffer(reinterpret_cast<__gm__ BT *>(weightTensorPtr) + mnConfig.wBaseOffset + wOffset);
    if (mnConfig.blockDimM == 1) {
        weightGmLocal.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    }
    return weightGmLocal;
}

template <typename mmType, bool sync>
__aicore__ inline void GmmAddCompute<mmType, sync>::MmCompute(MNConfig &mnConfig, uint32_t blockIdx)
{
    if (0 == cubeNum) {
        cubeNum++;
        return;
    }
    uint32_t tailN = mnConfig.nIdx * mnConfig.singleN;
    uint32_t curSingleN = mnConfig.nIdx < mnConfig.blockDimN - 1 ? mnConfig.singleN : mnConfig.n - tailN;
    uint32_t curSingleM =
        mnConfig.mIdx < mnConfig.blockDimM - 1 ? mnConfig.singleM : mnConfig.m - mnConfig.mIdx * mnConfig.singleM;
    uint64_t outOffset = mnConfig.mIdx * mnConfig.singleM * mnConfig.n + tailN;
    yGm.SetGlobalBuffer(reinterpret_cast<__gm__ CT *>(yTensorPtr) + mnConfig.yBaseOffset);
    yRefGm.SetGlobalBuffer(reinterpret_cast<__gm__ CT *>(yRefTensorPtr) + (blockIdx * mnConfig.baseM * mnConfig.baseN));
    mm.template IterateAll<sync>(yRefGm, 0, true, true);
    mm.WaitIterateAll();

    AscendC::DataCopyPadExtParams<CT> copyPadParams{false, 0, 0, 0};
    AscendC::DataCopyExtParams copyParams{
        static_cast<uint16_t>(curSingleM), static_cast<uint32_t>(curSingleN * sizeof(CT)), 0,
        static_cast<uint32_t>((mnConfig.baseN - curSingleN) * sizeof(CT) / UB_BLOCK_UNIT_SIZE), 0};
    AscendC::DataCopyPad<CT>(tensorIn, yRefGm, copyParams, copyPadParams);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventIDMTE2ToMTE3);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventIDMTE2ToMTE3);
    AscendC::SetAtomicAdd<CT>();
    copyParams = {static_cast<uint16_t>(curSingleM), static_cast<uint32_t>(curSingleN * sizeof(CT)),
                  static_cast<uint32_t>((mnConfig.baseN - curSingleN) * sizeof(CT) / UB_BLOCK_UNIT_SIZE),
                  static_cast<uint32_t>((mnConfig.n - curSingleN) * sizeof(CT)), 0};
    AscendC::DataCopyPad<CT>(yGm[outOffset], tensorIn, copyParams);
    AscendC::SetAtomicNone();

    cubeNum++;
}
template <typename mmType, bool sync>
__aicore__ inline void GmmAddCompute<mmType, sync>::MmComputePrepare(uint32_t groupIdx, MNConfig &mnConfig)
{
    uint32_t tailN = mnConfig.nIdx * mnConfig.singleN;
    uint32_t curSingleN = mnConfig.nIdx < mnConfig.blockDimN - 1 ? mnConfig.singleN : mnConfig.n - tailN;
    uint32_t curSingleM =
        mnConfig.mIdx < mnConfig.blockDimM - 1 ? mnConfig.singleM : mnConfig.m - mnConfig.mIdx * mnConfig.singleM;
    uint64_t xOffset = mnConfig.mIdx * mnConfig.singleM * mnConfig.k;
    if constexpr (transposeX) {
        xOffset = mnConfig.mIdx * mnConfig.singleM;
    }
    uint64_t outOffset = mnConfig.mIdx * mnConfig.singleM * mnConfig.n + tailN;
    xGm.SetGlobalBuffer(reinterpret_cast<__gm__ AT *>(xTensorPtr) + mnConfig.xBaseOffset);
    GlobalTensor<BT> weightGmLocal = SetGlobalBufferW(groupIdx, tailN, mnConfig);
    mm.SetSingleShape(curSingleM, curSingleN, mnConfig.k);
    mm.SetTensorA(xGm[xOffset], transposeX);
    mm.SetTensorB(weightGmLocal, transposeW);
}
#endif

extern "C" __global__ __aicore__ void grouped_matmul_add(GM_ADDR x, GM_ADDR weight, GM_ADDR groupList, GM_ADDR y,
                                                         GM_ADDR yRef, GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe tPipe;
    AscendCUtils::SetOverflow(1);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    REGISTER_TILING_DEFAULT(GroupedMatmulAdd::GmmAddTilingDataParams);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    if (TILING_KEY_IS(10000900009000090001UL)) { // split_k
        GroupedMatmulAdd::GroupedMatMulAddKernel<AscendC::Te::DNExtLayoutPtn, AscendC::Te::NDExtLayoutPtn>(
            x, weight, groupList, y, tiling);
    }
#else
#ifndef __CCE_UT_TEST__
    GET_TILING_DATA_MEMBER(GroupedMatmulAddTilingData, gmmBaseParams, gmmBaseParamsObj, tiling);
    GET_TILING_DATA_MEMBER(GroupedMatmulAddTilingData, mmTilingData, mmTilingDataObj, tiling);
#else
    GET_TILING_DATA(GroupedMatmulAddTilingData, tiling);
    auto gmmBaseParamsObj = GroupedMatmulAddTilingData.gmmBaseParams;
    auto mmTilingDataObj = GroupedMatmulAddTilingData.mmTilingData;
#endif
    if (TILING_KEY_IS(0)) {
        using matmulType = MmImplType<xType<true>, weightType<false>, yType>;
        matmulType::MT mm;
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), mm, &mmTilingDataObj);
        GmmAddCompute<matmulType, false> computeOp(mm);
        computeOp.Init(x, weight, groupList, yRef, &gmmBaseParamsObj, &mmTilingDataObj, &tPipe, workspace);
        GmmAddProcess<decltype(computeOp)> op(computeOp);
        op.Init(&gmmBaseParamsObj, &mmTilingDataObj, groupList, tiling);
        op.Process();
    } else if (TILING_KEY_IS(1)) {
        using matmulType = MmImplType<xType<true, bfloat16_t>, weightType<false, bfloat16_t>, yType>;
        matmulType::MT mm;
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), mm, &mmTilingDataObj);
        GmmAddCompute<matmulType, false> computeOp(mm);
        computeOp.Init(x, weight, groupList, yRef, &gmmBaseParamsObj, &mmTilingDataObj, &tPipe, workspace);
        GmmAddProcess<decltype(computeOp)> op(computeOp);
        op.Init(&gmmBaseParamsObj, &mmTilingDataObj, groupList, tiling);
        op.Process();
    }
#endif
}

} // namespace AscendC
