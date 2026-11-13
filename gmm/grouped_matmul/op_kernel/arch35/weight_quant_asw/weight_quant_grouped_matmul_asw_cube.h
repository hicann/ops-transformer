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
 * \file weight_quant_grouped_matmul_asw_cube.h
 * \brief
 */
#pragma once

#include "weight_quant_tool.h"
#include "grouped_matmul_asw_basic_block_config.h"
#include "weight_quant_grouped_matmul_asw_block.h"
#include "weight_quant_grouped_matmul_cube_compute_tools.h"
#include "../../grouped_matmul_utils.h"
#include "../grouped_matmul_tiling_data_apt.h"

using AscendC::DataCopy;
using GMMWeightQuantAswParam = GroupedMatmulTilingData::GMMWeightQuantAswParams;
using WeightQuantTools::B32_IN_B16_NUM;
using WeightQuantTools::B8_IN_B16_NUM;
using WeightQuantTools::BIAS_TABLE_SIZE_KB;
using WeightQuantTools::C0;
using WeightQuantTools::ConfigE8M0ScaleDn2NzParams;
using WeightQuantTools::ConfigHif4ScaleDn2NzParams;
using WeightQuantTools::DOUBLE_BUFFER;
using WeightQuantTools::DataCopyPad2D;
using WeightQuantTools::GetKBUnit;
using WeightQuantTools::GetSplitValueFromGroupList;
using WeightQuantTools::IsMxA8HiF4;
using WeightQuantTools::IsMxA8W4;
using WeightQuantTools::L0A_BUFFER_SIZE_KB;
using WeightQuantTools::L0B_BUFFER_SIZE_KB;
using WeightQuantTools::L1_BUFFER_HALF_SIZE_KB;
using WeightQuantTools::MTE1_MTE2_EVENT_ID;
using WeightQuantTools::MTE1_MTE2_SCALE_EVENT_ID;
using WeightQuantTools::MTE1_M_EVENT_ID;
using WeightQuantTools::MTE2_MTE1_EVENT_ID;
using WeightQuantTools::MX_GROUP_SIZE_32;
using WeightQuantTools::MX_GROUP_SIZE_64;
using WeightQuantTools::MX_SCALE_DIVISOR_SIZE;
using WeightQuantTools::M_MTE1_EVENT_ID;
using WeightQuantTools::Min;
using WeightQuantGroupedMatmul::Arch35::CopyBiasL1ToBT;
using WeightQuantGroupedMatmul::Arch35::FixL0CToDst;
using WeightQuantGroupedMatmul::Arch35::FixL0CToDstParams;
using WeightQuantGroupedMatmul::Arch35::GetBufferConfig;
using WeightQuantGroupedMatmul::Arch35::L0CopyAndCalcParams;
using WeightQuantGroupedMatmul::Arch35::L1BufferCfg;
using WeightQuantGroupedMatmul::Arch35::LoadAAndScaleL1ToL0;
using WeightQuantGroupedMatmul::Arch35::LoadAL1ToL0;
using WeightQuantGroupedMatmul::Arch35::LoadBAndScaleL1ToL0;
using WeightQuantGroupedMatmul::Arch35::MmadCompute;
using WeightQuantGroupedMatmul::Arch35::WeightQuantASWBlock;

#define LOCAL_TEMPLATE_CLASS_PARAMS \
    template <typename xType, typename wType, typename antiquantScaleType, typename pertokenScaleType, \
              typename biasType, typename yType, CubeFormat wFormat, bool xTrans, bool wTrans>
#define LOCAL_TEMPLATE_FUNC_PARAMS \
    xType, wType, antiquantScaleType, pertokenScaleType, biasType, yType, wFormat, xTrans, wTrans

namespace GROUPED_MATMUL {
LOCAL_TEMPLATE_CLASS_PARAMS
class WeightQuantASWMxKernel {
public:
    __aicore__ inline WeightQuantASWMxKernel() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR scale, GM_ADDR antiquantScale,
                                GM_ADDR antiquantOffset, GM_ADDR bias, GM_ADDR groupList, GM_ADDR perTokenScale,
                                GM_ADDR y, const GMMWeightQuantAswParam* __restrict gmmBaseParamsIn,
                                const TCubeTiling* __restrict mmTilingDataIn);
    __aicore__ inline void Process();

protected:
    __aicore__ inline void InitAddrAndParams(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR antiquantScale,
                                             GM_ADDR groupList, GM_ADDR perTokenScale, GM_ADDR y);
    __aicore__ inline void InitCubeBuffers();
    __aicore__ inline void SetMNK(uint32_t groupIdx, uint64_t& mSize, uint64_t& nSize, uint64_t& kSize);
    __aicore__ inline bool EnablePreAlloc();
    __aicore__ inline bool IsLastGroupAndNeedSplit(uint32_t groupIdx);
    __aicore__ inline void UpdateGlobalAddr(uint32_t groupIdx);
    __aicore__ inline bool IsLastGroupAndRound(uint32_t groupIdx, uint64_t roundIdx);
    __aicore__ inline void MatmulCompute(uint64_t roundIndex);
    __aicore__ inline void CopyGmToL1(uint64_t kL1Size, uint64_t kOffset);
    __aicore__ inline void CopyAGmToL1(uint64_t kL1Size, uint64_t kOffset);
    __aicore__ inline void CopyBGmToL1(uint64_t kL1Size, uint64_t kOffset);
    __aicore__ inline void CopyScaleGmToL1(uint64_t aScaleL1GroupNum, uint64_t bScaleL1GroupNum, uint64_t kOffset);
    __aicore__ inline void IterateMatmul(uint64_t kL1Size, uint64_t aScaleL1GroupNum, uint64_t bScaleL1GroupNum,
                                         uint64_t kInnerIdx, uint64_t kOffset);
    __aicore__ inline void GetTensorC(const GlobalTensor<yType>& outTensor);
    __aicore__ inline void InitEventIds();
    __aicore__ inline void EndSync();

    static constexpr L1BufferCfg L1_BUFFER_CFG =
        GetBufferConfig<xType, wType, antiquantScaleType, pertokenScaleType, biasType>();
    static constexpr uint32_t B_L1_OFFSET_ELEM = L1_BUFFER_HALF_SIZE_KB * GetKBUnit<wType>();
    static constexpr uint32_t A_L1_OFFSET_ELEM = L1_BUFFER_HALF_SIZE_KB * GetKBUnit<xType>();
    static constexpr uint32_t A_SCALE_L1_OFFSET_ELEM = L1_BUFFER_HALF_SIZE_KB * GetKBUnit<pertokenScaleType>();
    static constexpr uint32_t B_SCALE_L1_OFFSET_ELEM = L1_BUFFER_HALF_SIZE_KB * GetKBUnit<antiquantScaleType>();
    static constexpr uint32_t BIAS_L1_OFFSET_ELEM = L1_BUFFER_HALF_SIZE_KB * GetKBUnit<biasType>();

    static constexpr uint32_t B_L0_OFFSET_ELEM = (L0B_BUFFER_SIZE_KB >> 1) * GetKBUnit<wType>();
    static constexpr uint32_t A_L0_OFFSET_ELEM = (L0A_BUFFER_SIZE_KB >> 1) * GetKBUnit<xType>();
    static constexpr uint32_t BIAS_L0_OFFSET_ELEM = (BIAS_TABLE_SIZE_KB >> 1) * GetKBUnit<float>();

    static constexpr uint64_t B8_IN_B16_NUM = 2UL;
    static constexpr uint64_t B32_IN_B16_NUM = 2UL;
    static constexpr uint64_t MX_A_SCALE_GROUP_SIZE = MX_GROUP_SIZE_32;
    static constexpr uint64_t MX_B_SCALE_GROUP_SIZE =
        IsSameType<antiquantScaleType, hif4_scale>::value ? MX_GROUP_SIZE_64 : MX_GROUP_SIZE_32;

    uint32_t blockIdx_;
    uint32_t groupNum_;
    uint32_t l0cSizeElem_;
    int8_t groupType_;
    uint8_t groupListType_;
    bool enablePreAlloc_;
    uint64_t preOffset_;
    bool hasBias_;
    const TCubeTiling* __restrict mmTilingData_;
    const GMMWeightQuantAswParam* __restrict gmmWeightQuantParams_;

    using aL0DataType = std::conditional_t<(IsMxA8W4<xType, wType, antiquantScaleType>() ||
                                            IsMxA8HiF4<xType, wType, antiquantScaleType>()),
                                           AscendC::mx_fp8_e4m3_t, xType>;

    GM_ADDR xTensorPtr_;
    GM_ADDR weightTensorPtr_;
    GM_ADDR biasTensorPtr_;
    GM_ADDR antiquantScaleTensorPtr_;
    GM_ADDR groupListPtr_;
    GM_ADDR yTensorPtr_;
    __gm__ half* perTokenScaleGm_; // 使用fp16搬运float8_e8m0

    GlobalTensor<xType> xGlobal_;
    GlobalTensor<wType> wGlobal_;
    GlobalTensor<yType> yGlobal_;
    GlobalTensor<biasType> biasGlobal_;
    GlobalTensor<int64_t> groupListGlobal_;
    GlobalTensor<half> aScaleGlobal_;
    GlobalTensor<half> bScaleGlobal_;

    LocalTensor<xType> aL1Tensor_;
    LocalTensor<wType> bL1Tensor_;
    LocalTensor<pertokenScaleType> aScaleL1Tensor_;
    LocalTensor<antiquantScaleType> bScaleL1Tensor_;
    LocalTensor<biasType> biasL1Tensor_;

    LocalTensor<aL0DataType> aL0Tensor_{AscendC::TPosition::A2, 0, (L0A_BUFFER_SIZE_KB * GetKBUnit<xType>())};
    LocalTensor<wType> bL0Tensor_{AscendC::TPosition::B2, 0, L0B_BUFFER_SIZE_KB* GetKBUnit<wType>()};
    LocalTensor<float> biasBtTensor_{AscendC::TPosition::C2, 0, (BIAS_TABLE_SIZE_KB * GetKBUnit<float>())};
    LocalTensor<float> cL0Tensor_;

    uint64_t roundDbFlag_ = 0; // bias DB倒换Flag
    uint64_t l1DbFlag_ = 0;    // A/B矩阵L1 DB倒换Flag
    uint64_t scaleL1DbFlag_ = 0;
    uint64_t l0DbFlag_ = 0;
    uint64_t kBlockSize_ = 0;
    uint64_t totalKBlocks_ = 0;
    uint64_t l0cOffset_ = 0;

    WeightQuantASWBlock block_;
};

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void WeightQuantASWMxKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::Init(
    GM_ADDR x, GM_ADDR weight, GM_ADDR scale, GM_ADDR antiquantScale, GM_ADDR antiquantOffset, GM_ADDR bias,
    GM_ADDR groupList, GM_ADDR perTokenScale, GM_ADDR y, const GMMWeightQuantAswParam* __restrict gmmBaseParamsIn,
    const TCubeTiling* __restrict mmTilingDataIn)
{
    if ASCEND_IS_AIV {
        return;
    }

    mmTilingData_ = mmTilingDataIn;
    gmmWeightQuantParams_ = gmmBaseParamsIn;
    blockIdx_ = GetBlockIdx();
    kBlockSize_ = mmTilingData_->stepKa * mmTilingData_->baseK;
    totalKBlocks_ = Ops::Base::CeilDiv(static_cast<uint64_t>(mmTilingData_->Ka), kBlockSize_);

    InitAddrAndParams(x, weight, bias, antiquantScale, groupList, perTokenScale, y);
    InitCubeBuffers();
    InitEventIds();
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void WeightQuantASWMxKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::InitAddrAndParams(
    GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR antiquantScale, GM_ADDR groupList, GM_ADDR perTokenScale,
    GM_ADDR y)
{
    groupNum_ = gmmWeightQuantParams_->groupNum;
    l0cSizeElem_ = gmmWeightQuantParams_->l0cSize * GetKBUnit<float>();
    groupType_ = gmmWeightQuantParams_->groupType;
    groupListType_ = gmmWeightQuantParams_->groupListType;
    hasBias_ = mmTilingData_->isBias;
    block_.Init(mmTilingData_, blockIdx_);

    xTensorPtr_ = x;
    weightTensorPtr_ = weight;
    biasTensorPtr_ = bias;
    antiquantScaleTensorPtr_ = antiquantScale;
    groupListPtr_ = groupList;
    yTensorPtr_ = y;
    perTokenScaleGm_ = reinterpret_cast<__gm__ half*>(perTokenScale);

    if (groupList != nullptr) {
        groupListGlobal_.SetGlobalBuffer((__gm__ int64_t*)groupList);
    }
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void WeightQuantASWMxKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::InitCubeBuffers()
{
    cL0Tensor_ = {AscendC::TPosition::CO1, 0, l0cSizeElem_};

    // L1 Buffer
    //
    // 无bias:
    // 0 ~ 256   | bL1 ping | aL1 ping | bScaleL1 ping | aScaleL1 ping |  |
    // 256 ~ 512 | bL1 pong | aL1 pong | bScaleL1 pong | aScaleL1 pong |  |
    //
    // 有bias:
    // 0 ~ 256   | bL1 ping | aL1 ping | bias ping | bScaleL1 ping | aScaleL1 ping |  |
    // 256 ~ 512 | bL1 pong | aL1 pong | bias pong | bScaleL1 pong | aScaleL1 pong |  |
    //
    // 每块buffer的起始地址为ping的起始地址，buffer大小为单块buffer + L1_SIZE/2 (256KB)，覆盖ping的头和pong的尾
    // 通过dbFlag和L1_OFFSET_ELEM (256KB)控制db倒换

    uint64_t l1BufOffset = 0; // Byte

    bL1Tensor_ = AscendC::LocalTensor<wType>(AscendC::TPosition::TSCM, l1BufOffset,
                                             L1_BUFFER_CFG.bL1SizeElem + B_L1_OFFSET_ELEM);
    if constexpr (AscendC::IsSupportB4<wType>()) {
        l1BufOffset += L1_BUFFER_CFG.bL1SizeElem >> 1;
    } else {
        l1BufOffset += L1_BUFFER_CFG.bL1SizeElem * sizeof(wType);
    }

    if (hasBias_) {
        aL1Tensor_ = AscendC::LocalTensor<xType>(AscendC::TPosition::TSCM, l1BufOffset,
                                                 L1_BUFFER_CFG.aL1SizeElemWithBias + A_L1_OFFSET_ELEM);
        l1BufOffset += L1_BUFFER_CFG.aL1SizeElemWithBias * sizeof(xType);

        biasL1Tensor_ = AscendC::LocalTensor<biasType>(AscendC::TPosition::TSCM, l1BufOffset,
                                                       L1_BUFFER_CFG.biasL1SizeElem + BIAS_L1_OFFSET_ELEM);
        l1BufOffset += L1_BUFFER_CFG.biasL1SizeElem * sizeof(biasType);
    } else {
        aL1Tensor_ = AscendC::LocalTensor<xType>(AscendC::TPosition::TSCM, l1BufOffset,
                                                 L1_BUFFER_CFG.aL1SizeElemWithoutBias + A_L1_OFFSET_ELEM);
        l1BufOffset += L1_BUFFER_CFG.aL1SizeElemWithoutBias * sizeof(xType);
    }

    bScaleL1Tensor_ = AscendC::LocalTensor<antiquantScaleType>(AscendC::TPosition::TSCM, l1BufOffset,
                                                               L1_BUFFER_CFG.bScaleL1SizeElem + B_SCALE_L1_OFFSET_ELEM);

    if constexpr (IsMxA8W4<xType, wType, antiquantScaleType>() || IsMxA8HiF4<xType, wType, antiquantScaleType>()) {
        l1BufOffset += L1_BUFFER_CFG.bScaleL1SizeElem * sizeof(antiquantScaleType);
        aScaleL1Tensor_ = AscendC::LocalTensor<pertokenScaleType>(
            AscendC::TPosition::TSCM, l1BufOffset, L1_BUFFER_CFG.aScaleL1SizeElem + A_SCALE_L1_OFFSET_ELEM);
    }
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void WeightQuantASWMxKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::Process()
{
    if ASCEND_IS_AIV {
        return;
    }

    if (groupType_ != -1) { // -1: no split
        if (unlikely(groupListPtr_ == nullptr)) {
            return;
        }
        preOffset_ = 0;
    }

    for (uint32_t groupIdx = 0; groupIdx < groupNum_; ++groupIdx) {
        uint64_t mSize;
        uint64_t nSize;
        uint64_t kSize;
        // 更新group内的M, N, K
        SetMNK(groupIdx, mSize, nSize, kSize);
        block_.template UpdateGroupOffset<wTrans, xType, wType, antiquantScaleType, yType, wFormat>(mSize, nSize, kSize,
                                                                                                    groupIdx);
        if (mSize <= 0 || kSize <= 0 || nSize <= 0) {
            continue;
        }
        block_.template UpdateGroupParams<>(groupIdx == groupNum_ - 1);
        enablePreAlloc_ = EnablePreAlloc();
        // 最后一个group最后一轮是否进一步切分以使用更多的核数
        if (IsLastGroupAndNeedSplit(groupIdx)) {
            block_.template CalcTailTile<>();
            block_.UpdateTailTile();
        }

        UpdateGlobalAddr(groupIdx);
        for (uint64_t roundIdx = 0; roundIdx < block_.params_.round; ++roundIdx) {
            bool isLastGroupRound = IsLastGroupAndRound(groupIdx, roundIdx);
            block_.UpdateBasicIndex(roundIdx, isLastGroupRound);
            // 1. Set single core param
            block_.template UpdateBlockParams<xTrans, wTrans, wFormat>(roundIdx, isLastGroupRound);
            if (block_.params_.singleCoreM <= 0 || block_.params_.singleCoreN <= 0) {
                continue;
            }
            // 2. compute offset
            block_.template CalcGMOffset<wTrans, xType, wType, antiquantScaleType, wFormat>();
            // 3. mmad
            MatmulCompute(roundIdx);
            roundDbFlag_ ^= 1;
        }
    }
    EndSync();
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void WeightQuantASWMxKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::SetMNK(uint32_t groupIdx, uint64_t& mSize,
                                                                                  uint64_t& nSize, uint64_t& kSize)
{
    mSize = GetSplitValueFromGroupList(groupIdx, preOffset_, groupType_, groupListType_, groupListGlobal_);
    kSize = mmTilingData_->Ka;
    nSize = mmTilingData_->N;
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline bool WeightQuantASWMxKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::EnablePreAlloc()
{
    uint64_t kL1Size = WeightQuantTools::Min<uint64_t>(block_.params_.k, kBlockSize_);
    uint64_t aL1Size = block_.params_.baseM * kL1Size * sizeof(xType);
    uint64_t bL1Size = 0UL;
    if constexpr (AscendC::IsSupportB4<wType>()) {
        bL1Size = (block_.params_.splitBaseN * kL1Size) >> 1;
    } else {
        bL1Size = block_.params_.splitBaseN * kL1Size * sizeof(wType);
    }
    // AL1+BL1载入数据量较大(128*1024B，经验值)时，开启prealloc会导致后2轮的mte2抢占前面的ostd，导致前面的带宽跟不上cube计算。
    // 当K较小时，单轮计算量较小，如果不开启prealloc，下一轮MN的计算的mte2无法提前发出，会导致mte2断流。
    return !(aL1Size + bL1Size >= 128 * 1024 && block_.params_.k > 1024);
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline bool WeightQuantASWMxKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::IsLastGroupAndNeedSplit(uint32_t groupIdx)
{
    // 2: 剩一半及以上核数时才考虑尾块切分
    return groupIdx == groupNum_ - 1 && (block_.GetEndBlockIdx() + 1) <= mmTilingData_->usedCoreNum / 2;
}

// 更新每个group的global地址
LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void WeightQuantASWMxKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::UpdateGlobalAddr(uint32_t groupIdx)
{
    xGlobal_.SetGlobalBuffer(GetTensorAddr<xType>(0, xTensorPtr_) + block_.params_.xGroupAddrOffset);
    wGlobal_.SetGlobalBuffer(GetTensorAddr<wType>(0, weightTensorPtr_) + block_.params_.wGroupAddrOffset);
    if constexpr (IsMxA8W4<xType, wType, antiquantScaleType>() || IsMxA8HiF4<xType, wType, antiquantScaleType>()) {
        aScaleGlobal_.SetGlobalBuffer(perTokenScaleGm_ + block_.params_.pertokenScaleGroupAddrOffset / B8_IN_B16_NUM);
    }
    bScaleGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(
        GetTensorAddr<antiquantScaleType>(0, antiquantScaleTensorPtr_) + block_.params_.antiquantScaleGroupAddrOffset));
    yGlobal_.SetGlobalBuffer(GetTensorAddr<yType>(0, yTensorPtr_) + block_.params_.yGroupAddrOffset);

    if (hasBias_) {
        biasGlobal_.SetGlobalBuffer(GetTensorAddr<biasType>(0, biasTensorPtr_) + block_.params_.biasGroupAddrOffset);
    }
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline bool WeightQuantASWMxKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::IsLastGroupAndRound(uint32_t groupIdx,
                                                                                               uint64_t roundIdx)
{
    return groupIdx == groupNum_ - 1 && roundIdx == block_.params_.round - 1 && blockIdx_ <= block_.GetEndBlockIdx();
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void WeightQuantASWMxKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::MatmulCompute(uint64_t roundIndex)
{
    uint64_t kOffset = 0;
    uint64_t aScaleL1GroupNum = 0;
    uint64_t bScaleL1GroupNum = 0;
    for (uint64_t kOutIdx = 0; kOutIdx < totalKBlocks_; kOutIdx++) {
        uint64_t kL1Size = WeightQuantTools::Min<uint64_t>(block_.params_.k - kOffset, kBlockSize_);

        if (kOffset % L1_BUFFER_CFG.scaleKL1Size == 0) {
            if constexpr (IsMxA8W4<xType, wType, antiquantScaleType>() ||
                          IsMxA8HiF4<xType, wType, antiquantScaleType>()) {
                aScaleL1GroupNum = Ops::Base::CeilDiv(
                    WeightQuantTools::Min<uint64_t>(block_.params_.k - kOffset, L1_BUFFER_CFG.scaleKL1Size),
                    MX_A_SCALE_GROUP_SIZE);
            }
            bScaleL1GroupNum = Ops::Base::CeilDiv(
                WeightQuantTools::Min<uint64_t>(block_.params_.k - kOffset, L1_BUFFER_CFG.scaleKL1Size),
                MX_B_SCALE_GROUP_SIZE);
            if constexpr (IsMxA8W4<xType, wType, antiquantScaleType>()) {
                // Copy A scales in pairs of 2 E8M0 values, including the tail.
                aScaleL1GroupNum = Ops::Base::CeilAlign(aScaleL1GroupNum, uint64_t{2});
                // Copy B scales in pairs of 2 E8M0 values, including the tail.
                bScaleL1GroupNum = Ops::Base::CeilAlign(bScaleL1GroupNum, uint64_t{2});
            }
            WaitFlag<HardEvent::MTE1_MTE2>(MTE1_MTE2_SCALE_EVENT_ID + static_cast<uint8_t>(scaleL1DbFlag_));
            CopyScaleGmToL1(aScaleL1GroupNum, bScaleL1GroupNum, kOffset);
        }

        WaitFlag<HardEvent::MTE1_MTE2>(MTE1_MTE2_EVENT_ID + static_cast<uint8_t>(l1DbFlag_));
        CopyGmToL1(kL1Size, kOffset);

        SetFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT_ID + static_cast<uint8_t>(l1DbFlag_));
        WaitFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT_ID + static_cast<uint8_t>(l1DbFlag_));

        uint64_t kInnerBlockCount = Ops::Base::CeilDiv(kL1Size, static_cast<uint64_t>(mmTilingData_->baseK));
        for (uint64_t kInnerIdx = 0; kInnerIdx < kInnerBlockCount; kInnerIdx++) {
            IterateMatmul(kL1Size, aScaleL1GroupNum, bScaleL1GroupNum, kInnerIdx, kOffset);
            kOffset += mmTilingData_->baseK;
        }

        SetFlag<HardEvent::MTE1_MTE2>(MTE1_MTE2_EVENT_ID + static_cast<uint8_t>(l1DbFlag_));
        l1DbFlag_ ^= 1;
        if (kOffset % L1_BUFFER_CFG.scaleKL1Size == 0 || kOffset >= block_.params_.k) {
            SetFlag<HardEvent::MTE1_MTE2>(MTE1_MTE2_SCALE_EVENT_ID + static_cast<uint8_t>(scaleL1DbFlag_));
            scaleL1DbFlag_ ^= 1;
        }
    }
    GetTensorC(yGlobal_[block_.offset_.offsetY]);
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void WeightQuantASWMxKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::CopyGmToL1(uint64_t kL1Size,
                                                                                      uint64_t kOffset)
{
    CopyAGmToL1(kL1Size, kOffset);
    CopyBGmToL1(kL1Size, kOffset);

    if (hasBias_ && kOffset == 0) {
        DataCopyPad2D(biasL1Tensor_[roundDbFlag_ * BIAS_L1_OFFSET_ELEM], biasGlobal_[block_.offset_.offsetBias], 1,
                      block_.params_.singleCoreN, block_.params_.singleCoreN, block_.params_.singleCoreN,
                      enablePreAlloc_);
    }
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void WeightQuantASWMxKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::CopyAGmToL1(uint64_t kL1Size,
                                                                                       uint64_t kOffset)
{
    uint64_t aGmOffset = (block_.offset_.offsetM * block_.params_.k + kOffset);

    AscendC::Nd2NzParams nd2nzParams;
    nd2nzParams.ndNum = 1;
    nd2nzParams.nValue = block_.params_.singleCoreM;
    nd2nzParams.dValue = kL1Size;
    nd2nzParams.srcDValue = block_.params_.k;
    nd2nzParams.srcNdMatrixStride = 0;

    nd2nzParams.dstNzC0Stride = Ops::Base::CeilAlign<uint64_t>(nd2nzParams.nValue, BLOCK_CUBE);
    nd2nzParams.dstNzNStride = 1;
    nd2nzParams.dstNzMatrixStride = 0;

    DataCopy(aL1Tensor_[l1DbFlag_ * A_L1_OFFSET_ELEM], xGlobal_[aGmOffset], nd2nzParams, enablePreAlloc_);
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void WeightQuantASWMxKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::CopyBGmToL1(uint64_t kL1Size,
                                                                                       uint64_t kOffset)
{
    uint64_t nAlign = Ops::Base::CeilAlign(block_.params_.n, static_cast<uint64_t>(BLOCK_CUBE));
    uint64_t weightGmOffset = kOffset * nAlign + block_.offset_.offsetN * C0<wType>::value; // 转置 NZ

    // DataCopyPad不支持直接往L1搬运fp4，转换成B8搬运，步长和数据长度等需要除2
    DataCopyPad2D(
        bL1Tensor_[l1DbFlag_ * B_L1_OFFSET_ELEM].template ReinterpretCast<uint8_t>(),
        wGlobal_[weightGmOffset].template ReinterpretCast<uint8_t>(), Ops::Base::CeilDiv(kL1Size, C0<wType>::value),
        block_.params_.singleCoreN * C0<wType>::value >> 1,
        Ops::Base::CeilAlign(block_.params_.singleCoreN, static_cast<uint64_t>(BLOCK_CUBE)) * C0<wType>::value >> 1,
        nAlign * C0<wType>::value >> 1, enablePreAlloc_);
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void WeightQuantASWMxKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::CopyScaleGmToL1(uint64_t aScaleL1GroupNum,
                                                                                           uint64_t bScaleL1GroupNum,
                                                                                           uint64_t kOffset)
{
    uint64_t kSizeAlign = Ops::Base::CeilAlign(block_.params_.k, MX_SCALE_DIVISOR_SIZE);

    if constexpr (IsMxA8W4<xType, wType, antiquantScaleType>() || IsMxA8HiF4<xType, wType, antiquantScaleType>()) {
        uint64_t aScaleKGmGroupNum = Ops::Base::CeilDiv(kSizeAlign, MX_A_SCALE_GROUP_SIZE);
        Dn2NzParams aScaleDn2NzParams;
        ConfigE8M0ScaleDn2NzParams(block_.params_.singleCoreM, aScaleKGmGroupNum, aScaleL1GroupNum, aScaleDn2NzParams);
        uint64_t aScaleGmOffset =
            (block_.offset_.offsetPerTokenScale + kOffset / MX_A_SCALE_GROUP_SIZE) / B8_IN_B16_NUM;
        DataCopy(aScaleL1Tensor_[scaleL1DbFlag_ * A_SCALE_L1_OFFSET_ELEM].template ReinterpretCast<half>(),
                 aScaleGlobal_[aScaleGmOffset], aScaleDn2NzParams, enablePreAlloc_);
    }

    uint64_t bScaleKGmGroupNum = Ops::Base::CeilDiv(kSizeAlign, MX_B_SCALE_GROUP_SIZE);
    Dn2NzParams bScaleDn2NzParams;
    uint64_t bScaleGmOffset = block_.offset_.offsetAntiquantScale + kOffset / MX_B_SCALE_GROUP_SIZE;
    if constexpr (IsSameType<antiquantScaleType, hif4_scale>::value) {
        ConfigHif4ScaleDn2NzParams(block_.params_.singleCoreN, bScaleKGmGroupNum, bScaleL1GroupNum, bScaleDn2NzParams);
        bScaleGmOffset *= B32_IN_B16_NUM;
    } else {
        ConfigE8M0ScaleDn2NzParams(block_.params_.singleCoreN, bScaleKGmGroupNum, bScaleL1GroupNum, bScaleDn2NzParams);
        bScaleGmOffset /= B8_IN_B16_NUM;
    }
    DataCopy(bScaleL1Tensor_[scaleL1DbFlag_ * B_SCALE_L1_OFFSET_ELEM].template ReinterpretCast<half>(),
             bScaleGlobal_[bScaleGmOffset], bScaleDn2NzParams, enablePreAlloc_);
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void WeightQuantASWMxKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::IterateMatmul(
    uint64_t kL1Size, uint64_t aScaleL1GroupNum, uint64_t bScaleL1GroupNum, uint64_t kInnerIdx, uint64_t kOffset)
{
    L0CopyAndCalcParams l0CopyAndCalcParams;
    l0CopyAndCalcParams.mL1Size = block_.params_.singleCoreM;
    l0CopyAndCalcParams.nL1Size = block_.params_.singleCoreN;
    l0CopyAndCalcParams.mL0Size = l0CopyAndCalcParams.mL1Size;
    l0CopyAndCalcParams.nL0Size = l0CopyAndCalcParams.nL1Size;
    l0CopyAndCalcParams.kL1Size = kL1Size;
    l0CopyAndCalcParams.kL0Size = WeightQuantTools::Min<uint64_t>(block_.params_.k - kOffset, mmTilingData_->baseK);
    l0CopyAndCalcParams.aScaleL1GroupNum = aScaleL1GroupNum;
    l0CopyAndCalcParams.bScaleL1GroupNum = bScaleL1GroupNum;
    l0CopyAndCalcParams.isFirstKLoop = kOffset == 0;
    l0CopyAndCalcParams.isLastKLoop = block_.params_.k - kOffset <= mmTilingData_->baseK;

    uint64_t kL1Offset = kInnerIdx * mmTilingData_->baseK;
    uint64_t aL1Offset =
        kL1Offset * Ops::Base::CeilAlign(l0CopyAndCalcParams.mL1Size, static_cast<uint64_t>(BLOCK_CUBE));
    uint64_t bL1Offset =
        kL1Offset * Ops::Base::CeilAlign(l0CopyAndCalcParams.nL1Size, static_cast<uint64_t>(BLOCK_CUBE));
    uint64_t aScaleL1Offset = (kOffset % L1_BUFFER_CFG.scaleKL1Size) / MX_A_SCALE_GROUP_SIZE * BLOCK_CUBE;
    uint64_t bScaleL1Offset = (kOffset % L1_BUFFER_CFG.scaleKL1Size) / MX_B_SCALE_GROUP_SIZE * BLOCK_CUBE;

    LocalTensor<aL0DataType> aL0Tensor = aL0Tensor_[l0DbFlag_ * A_L0_OFFSET_ELEM];
    LocalTensor<wType> bL0Tensor = bL0Tensor_[l0DbFlag_ * B_L0_OFFSET_ELEM];
    if (l0CopyAndCalcParams.isFirstKLoop) {
        uint64_t nextL0cBufSize = Ops::Base::CeilAlign(l0CopyAndCalcParams.mL0Size, static_cast<uint64_t>(BLOCK_CUBE)) *
                                  Ops::Base::CeilAlign(l0CopyAndCalcParams.nL0Size, static_cast<uint64_t>(BLOCK_CUBE));
        if (l0cOffset_ + nextL0cBufSize > l0cSizeElem_) {
            l0cOffset_ = 0;
        }
    }

    WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT_ID + static_cast<uint8_t>(l0DbFlag_));

    if constexpr (IsMxA8W4<xType, wType, antiquantScaleType>() || IsMxA8HiF4<xType, wType, antiquantScaleType>()) {
        LoadAAndScaleL1ToL0<aL0DataType, xType, pertokenScaleType>(
            aL0Tensor, aL1Tensor_[l1DbFlag_ * A_L1_OFFSET_ELEM + aL1Offset],
            aScaleL1Tensor_[scaleL1DbFlag_ * A_SCALE_L1_OFFSET_ELEM + aScaleL1Offset], l0CopyAndCalcParams);
    } else {
        LoadAL1ToL0<aL0DataType, xType>(aL0Tensor, aL1Tensor_[l1DbFlag_ * A_L1_OFFSET_ELEM + aL1Offset],
                                        l0CopyAndCalcParams);
    }
    LoadBAndScaleL1ToL0<wType, wType, antiquantScaleType>(
        bL0Tensor, bL1Tensor_[l1DbFlag_ * B_L1_OFFSET_ELEM + bL1Offset],
        bScaleL1Tensor_[scaleL1DbFlag_ * B_SCALE_L1_OFFSET_ELEM + bScaleL1Offset], l0CopyAndCalcParams);

    if (l0CopyAndCalcParams.isFirstKLoop && hasBias_) {
        CopyBiasL1ToBT<float, biasType>(biasBtTensor_[roundDbFlag_ * BIAS_L0_OFFSET_ELEM],
                                        biasL1Tensor_[roundDbFlag_ * BIAS_L1_OFFSET_ELEM], l0CopyAndCalcParams.nL0Size);
    }

    SetFlag<HardEvent::MTE1_M>(MTE1_M_EVENT_ID);
    WaitFlag<HardEvent::MTE1_M>(MTE1_M_EVENT_ID);

    if (l0CopyAndCalcParams.isFirstKLoop && hasBias_) {
        MmadCompute<float, aL0DataType, wType, float>(cL0Tensor_[l0cOffset_], aL0Tensor, bL0Tensor,
                                                      biasBtTensor_[roundDbFlag_ * BIAS_L0_OFFSET_ELEM],
                                                      l0CopyAndCalcParams);
    } else {
        MmadCompute<float, aL0DataType, wType>(cL0Tensor_[l0cOffset_], aL0Tensor, bL0Tensor, l0CopyAndCalcParams);
    }

    SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT_ID + static_cast<uint8_t>(l0DbFlag_));

    l0DbFlag_ ^= 1;
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void WeightQuantASWMxKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::GetTensorC(
    const GlobalTensor<yType>& outTensor)
{
    FixL0CToDstParams fixL0CToDstParams;
    fixL0CToDstParams.mL0Size = block_.params_.singleCoreM;
    fixL0CToDstParams.nL0Size = block_.params_.singleCoreN;
    fixL0CToDstParams.outNSize = block_.params_.n;
    FixL0CToDst<yType, float>(outTensor, cL0Tensor_[l0cOffset_], fixL0CToDstParams);
    l0cOffset_ += Ops::Base::CeilAlign(fixL0CToDstParams.mL0Size, static_cast<uint64_t>(BLOCK_CUBE)) *
                  Ops::Base::CeilAlign(fixL0CToDstParams.nL0Size, static_cast<uint64_t>(BLOCK_CUBE));
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void WeightQuantASWMxKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::InitEventIds()
{
    for (uint8_t idx = 0; idx < DOUBLE_BUFFER; ++idx) {
        SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT_ID + idx);
        SetFlag<HardEvent::MTE1_MTE2>(MTE1_MTE2_EVENT_ID + idx);
        SetFlag<HardEvent::MTE1_MTE2>(MTE1_MTE2_SCALE_EVENT_ID + idx);
    }
}

LOCAL_TEMPLATE_CLASS_PARAMS
__aicore__ inline void WeightQuantASWMxKernel<LOCAL_TEMPLATE_FUNC_PARAMS>::EndSync()
{
    for (uint8_t idx = 0; idx < DOUBLE_BUFFER; ++idx) {
        WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT_ID + idx);
        WaitFlag<HardEvent::MTE1_MTE2>(MTE1_MTE2_EVENT_ID + idx);
        WaitFlag<HardEvent::MTE1_MTE2>(MTE1_MTE2_SCALE_EVENT_ID + idx);
    }
}
} // namespace GROUPED_MATMUL
