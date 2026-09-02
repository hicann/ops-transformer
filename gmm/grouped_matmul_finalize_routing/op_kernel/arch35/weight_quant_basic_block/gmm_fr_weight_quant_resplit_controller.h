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
 * \file grouped_matmul_weight_quant_resplit_controller.h
 * \brief
 */
#ifndef GMM_FR_WEIGHT_QUANT_RESPLIT_CONTROLLER_H
#define GMM_FR_WEIGHT_QUANT_RESPLIT_CONTROLLER_H

#include "./gmm_fr_weight_quant_tiling_data.h"
#include "./gmm_fr_weight_quant_vcv_basic_block.h"
#include "weight_quant_tool.h"

using AscendC::CeilAlign;
using AscendC::SyncAll;
using GMMFinalizeRoutingArch35Tiling::GMMFinalizeRoutingWeightQuantTilingData;
using WeightQuantBatchMatmulV2::Arch35::A_L1_MAX_SIZE_WITH_BIAS_QUANT;
using WeightQuantBatchMatmulV2::Arch35::BASIC_BLOCK_PROCESS_NUM;
using WeightQuantBatchMatmulV2::Arch35::BasicBlockControlParam;
using WeightQuantBatchMatmulV2::Arch35::BasicBlockOffsetParam;
using WeightQuantBatchMatmulV2::Arch35::CeilDivide;
using WeightQuantBatchMatmulV2::Arch35::DOUBLE_BUFFER_NUM;
using WeightQuantBatchMatmulV2::Arch35::GetKBUnit;
using WeightQuantBatchMatmulV2::Arch35::GMMFRWeightQuantVcvBasicBlock;
using WeightQuantBatchMatmulV2::Arch35::IsMxA8W4;
using WeightQuantBatchMatmulV2::Arch35::K_ALIGNMENT64;
using WeightQuantBatchMatmulV2::Arch35::QUADRUPLE_BUFFER_NUM;
using WeightQuantBatchMatmulV2::Arch35::QuantType;
using WeightQuantBatchMatmulV2::Arch35::SCALE_FACTOR_B_BIT;
using WeightQuantBatchMatmulV2::Arch35::VecAntiQuantConfig;
using WeightQuantBatchMatmulV2::Arch35::WqmmConfig;
using GMMFRTiling = GMMFinalizeRoutingArch35Tiling::GMMFinalizeRoutingWeightQuantTilingData;

namespace GROUPED_MATMUL_FINALIZE_ROUTING {
#define GMM_FR_WEIGHT_QUANT_RESPLIT_CONTROLLER_TEMPLATE_PARAM \
    template <typename xType, typename wType, typename antiQuantScaleType, typename scaleType, \
              typename perTokenScaleType, typename biasType, typename yType, typename sharedInputDType, \
              typename logitsType, typename rowIndexType, bool hasBias, const WqmmConfig &wqmmConfig, \
              const VecAntiQuantConfig &vecConfig>

#define GMM_FR_WEIGHT_QUANT_RESPLIT_CONTROLLER_CLASS \
    GMMFRWeightQuantResplitController<xType, wType, antiQuantScaleType, scaleType, perTokenScaleType, biasType, yType, \
                                      sharedInputDType, logitsType, rowIndexType, hasBias, wqmmConfig, vecConfig>

GMM_FR_WEIGHT_QUANT_RESPLIT_CONTROLLER_TEMPLATE_PARAM
class GMMFRWeightQuantResplitController {
public:
    __aicore__ inline GMMFRWeightQuantResplitController(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR bias, GM_ADDR groupList,
                                GM_ADDR perTokenScale, GM_ADDR logitsAddr, GM_ADDR rowIndexAddr, GM_ADDR y,
                                GM_ADDR shareInput,
                                const GMMFinalizeRoutingWeightQuantTilingData *__restrict baseTiling);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitOffsetParam(BasicBlockOffsetParam offsetParam[BASIC_BLOCK_PROCESS_NUM]);
    __aicore__ inline void SplitNByMultiCore(BasicBlockOffsetParam offsetParam[BASIC_BLOCK_PROCESS_NUM],
                                             BasicBlockControlParam &ctrlParam, uint64_t basicBlockCount,
                                             uint64_t basicBlockSize);
    __aicore__ inline uint64_t GetSplitValueFromGroupList(uint64_t groupIdx);
    __aicore__ inline void UpdateGmAddr(uint64_t mSize, uint64_t kSize, uint64_t nSize);
    __aicore__ inline uint64_t GetSwitchedProcessId(const BasicBlockControlParam &ctrlParam);

    const GMMFinalizeRoutingWeightQuantTilingData *tiling_;

    __gm__ xType *xGm_;
    __gm__ wType *weightGm_;
    __gm__ antiQuantScaleType *antiquantScaleGm_;
    __gm__ biasType *biasGm_;
    __gm__ yType *yGm_;
    __gm__ perTokenScaleType *perTokenScaleGm_;
    __gm__ sharedInputDType *shareInputAddr_;
    __gm__ logitsType *logitsAddr_;
    __gm__ rowIndexType *rowIndexAddr_;
    GlobalTensor<int64_t> groupListGm_;
    GMMFRWeightQuantVcvBasicBlock<xType, wType, antiQuantScaleType, scaleType, perTokenScaleType, biasType, yType,
                                  sharedInputDType, logitsType, rowIndexType, hasBias, wqmmConfig, vecConfig>
        basicBlock_;

    uint64_t preOffset_ = 0;
    static constexpr uint64_t MX_A8W4_L1_K_CONFIG_256 = 256;
    static constexpr uint64_t MX_A8W4_L1_K_CONFIG_512 = 512;
    static constexpr uint64_t MX_A8W4_L1_K_DYNAMIC_CONFIG_N_THRESHOLD = 128;
    static constexpr uint64_t MX_A8W4_L1_K_DYNAMIC_CONFIG_M_THRESHOLD_256 = 256;
    static constexpr uint64_t MX_A8W4_L1_K_DYNAMIC_CONFIG_M_THRESHOLD_240 = 240;
    static constexpr uint64_t M_L1 = 256;
    static constexpr uint64_t KB_L1_SIZE = 256;
    static constexpr uint64_t N_L1 = 256;
    uint64_t mxA8W4L1KDynamicConfigMThreshold_; // m轴依赖空间划分，无法静态配置
};

GMM_FR_WEIGHT_QUANT_RESPLIT_CONTROLLER_TEMPLATE_PARAM
__aicore__ inline void GMM_FR_WEIGHT_QUANT_RESPLIT_CONTROLLER_CLASS::Init(
    GM_ADDR x, GM_ADDR weight, GM_ADDR antiquantScale, GM_ADDR bias, GM_ADDR groupList, GM_ADDR perTokenScale,
    GM_ADDR logitsAddr, GM_ADDR rowIndexAddr, GM_ADDR y, GM_ADDR shareInput,
    const GMMFinalizeRoutingWeightQuantTilingData *__restrict baseTiling)
{
    tiling_ = baseTiling;

    xGm_ = reinterpret_cast<__gm__ xType *>(x);
    weightGm_ = reinterpret_cast<__gm__ wType *>(weight);
    biasGm_ = reinterpret_cast<__gm__ biasType *>(bias);
    antiquantScaleGm_ = reinterpret_cast<__gm__ antiQuantScaleType *>(antiquantScale);
    perTokenScaleGm_ = reinterpret_cast<__gm__ perTokenScaleType *>(perTokenScale);
    logitsAddr_ = reinterpret_cast<__gm__ logitsType *>(logitsAddr);
    rowIndexAddr_ = reinterpret_cast<__gm__ rowIndexType *>(rowIndexAddr);
    yGm_ = reinterpret_cast<__gm__ yType *>(y);
    shareInputAddr_ = reinterpret_cast<__gm__ sharedInputDType *>(shareInput);
    groupListGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(groupList));
    basicBlock_.Init(tiling_->groupSize, yGm_, tiling_->sharedInputWeight);
    if constexpr (hasBias) {
        mxA8W4L1KDynamicConfigMThreshold_ = MX_A8W4_L1_K_DYNAMIC_CONFIG_M_THRESHOLD_240;
    } else {
        mxA8W4L1KDynamicConfigMThreshold_ = MX_A8W4_L1_K_DYNAMIC_CONFIG_M_THRESHOLD_256;
    }
}

GMM_FR_WEIGHT_QUANT_RESPLIT_CONTROLLER_TEMPLATE_PARAM
__aicore__ inline void GMM_FR_WEIGHT_QUANT_RESPLIT_CONTROLLER_CLASS::Process()
{
    uint32_t cubeBlockIdx = GetBlockIdx();
    if ASCEND_IS_AIV {
        cubeBlockIdx = cubeBlockIdx >> 1;
    }

    BasicBlockOffsetParam offsetParam[BASIC_BLOCK_PROCESS_NUM];
    InitOffsetParam(offsetParam);

    bool isCacheLineUnaligned = offsetParam[0].kSize % 128 != 0; // 缓存大小128B，对应8bit为128个元素
    if constexpr (IsSameType<wType, int4b_t>::value || IsSameType<wType, fp4x2_e2m1_t>::value ||
                  IsSameType<wType, fp4x2_e1m2_t>::value) {
        isCacheLineUnaligned = offsetParam[0].kSize % 256 != 0; // 缓存大小128B，对应4bit为256个元素
    }

    basicBlock_.InitAtomicGm(tiling_->initSize, tiling_->sharedInputOffset * tiling_->nSize,
                             tiling_->sharedInputLen * tiling_->nSize, shareInputAddr_);
    SyncAll();

    BasicBlockControlParam ctrlParam;
    ctrlParam.processId = 0;
    for (uint32_t groupIdx = 0, startBasicBlockId = 0; groupIdx < tiling_->groupNum; ++groupIdx) {
        // groupIdx 根据splitItem对M轴进行切分
        ctrlParam.mSize = GetSplitValueFromGroupList(groupIdx);
        if (ctrlParam.mSize > 0 && offsetParam[ctrlParam.processId].nSize > 0) {
            uint64_t mBlkNum = CeilDivide(ctrlParam.mSize, M_L1);
            // 划分M轴在L1上的大小
            ctrlParam.mL1Size = CeilDivide(ctrlParam.mSize, mBlkNum);
            basicBlock_.UpdateGlobalAddr(xGm_, weightGm_, antiquantScaleGm_, nullptr, perTokenScaleGm_, biasGm_, yGm_,
                                         logitsAddr_, rowIndexAddr_,
                                         ctrlParam.mL1Size < ctrlParam.mSize || isCacheLineUnaligned);
            ctrlParam.curBasicBlockId =
                cubeBlockIdx >= startBasicBlockId ? cubeBlockIdx : cubeBlockIdx + tiling_->coreNum;
            ctrlParam.basicBlockLimit = startBasicBlockId;
            for (ctrlParam.mOffset = 0; ctrlParam.mOffset < ctrlParam.mSize; ctrlParam.mOffset += ctrlParam.mL1Size) {
                ctrlParam.nOffset = 0;
                // 主块
                SplitNByMultiCore(offsetParam, ctrlParam, CeilDivide(offsetParam[0].nSize, N_L1), N_L1);
                ctrlParam.basicBlockLimit += CeilDivide(offsetParam[0].nSize, N_L1);
            }
            startBasicBlockId = ctrlParam.basicBlockLimit % tiling_->coreNum;
        }
        UpdateGmAddr(ctrlParam.mSize, offsetParam[0].kSize, offsetParam[0].nSize);
    }

    basicBlock_.End(offsetParam[GetSwitchedProcessId(ctrlParam)]);
}

GMM_FR_WEIGHT_QUANT_RESPLIT_CONTROLLER_TEMPLATE_PARAM
__aicore__ inline void GMM_FR_WEIGHT_QUANT_RESPLIT_CONTROLLER_CLASS::InitOffsetParam(
    BasicBlockOffsetParam offsetParam[BASIC_BLOCK_PROCESS_NUM])
{
    offsetParam[0].kbL1Size = KB_L1_SIZE;
    offsetParam[0].kaL1Size = offsetParam[0].kbL1Size; // 当前实现a矩阵切分保持b矩阵一致
    offsetParam[0].kSize = tiling_->kSize;
    offsetParam[0].nSize = tiling_->nSize;
    offsetParam[0].kAlign = CeilAlign(tiling_->kSize, static_cast<uint64_t>(BLOCK_CUBE));
    offsetParam[1].kbL1Size = KB_L1_SIZE;
    offsetParam[1].kaL1Size = offsetParam[0].kbL1Size; // 当前实现a矩阵切分保持b矩阵一致
    offsetParam[1].kSize = offsetParam[0].kSize;
    offsetParam[1].nSize = offsetParam[0].nSize;
    offsetParam[1].kAlign = offsetParam[0].kAlign;

    offsetParam[0].nAlign = CeilAlign(tiling_->nSize, static_cast<uint64_t>(BLOCK_CUBE));
    offsetParam[1].nAlign = offsetParam[0].nAlign;
}

GMM_FR_WEIGHT_QUANT_RESPLIT_CONTROLLER_TEMPLATE_PARAM
__aicore__ inline void GMM_FR_WEIGHT_QUANT_RESPLIT_CONTROLLER_CLASS::SplitNByMultiCore(
    BasicBlockOffsetParam offsetParam[BASIC_BLOCK_PROCESS_NUM], BasicBlockControlParam &ctrlParam,
    uint64_t basicBlockCount, uint64_t basicBlockSize)
{
    for (; ctrlParam.curBasicBlockId < ctrlParam.basicBlockLimit + basicBlockCount;
         ctrlParam.curBasicBlockId += tiling_->coreNum) {
        offsetParam[ctrlParam.processId].mSize = ctrlParam.mSize;
        offsetParam[ctrlParam.processId].mOffset = ctrlParam.mOffset;
        offsetParam[ctrlParam.processId].mL1Size = ctrlParam.mOffset + ctrlParam.mL1Size > ctrlParam.mSize ?
                                                       ctrlParam.mSize - ctrlParam.mOffset :
                                                       ctrlParam.mL1Size;
        offsetParam[ctrlParam.processId].nOffset =
            ctrlParam.nOffset +
            ((ctrlParam.curBasicBlockId - ctrlParam.basicBlockLimit) % basicBlockCount) * basicBlockSize;
        offsetParam[ctrlParam.processId].nL1Size =
            offsetParam[ctrlParam.processId].nOffset + basicBlockSize > tiling_->nSize ?
                tiling_->nSize - offsetParam[ctrlParam.processId].nOffset :
                basicBlockSize;
        offsetParam[ctrlParam.processId].yGmAddr = reinterpret_cast<GM_ADDR>(yGm_);
        // MxA8W4场景，切换低阶api，支持kernel内动态调整k轴切分
        offsetParam[ctrlParam.processId].kbL1Size =
            (offsetParam[ctrlParam.processId].mL1Size <= mxA8W4L1KDynamicConfigMThreshold_ &&
             offsetParam[ctrlParam.processId].nL1Size <= MX_A8W4_L1_K_DYNAMIC_CONFIG_N_THRESHOLD) ?
                MX_A8W4_L1_K_CONFIG_512 :
                MX_A8W4_L1_K_CONFIG_256;

        uint64_t mL1Align = CeilAlign(offsetParam[ctrlParam.processId].mL1Size, BLOCK_CUBE);
        // 实际A矩阵通过ND2NZ搬运实现，B矩阵通过MOVE_ALIGN搬运实现，两者效率不同。为了平衡两者的搬运效率，折算系数2简化计算。
        uint64_t kaDepth = CeilDivide(offsetParam[ctrlParam.processId].nL1Size, mL1Align * 2);
        // L1的空间，在有Bias时，预留124 Kb, 其他场景预留128 Kb
        uint64_t aL1Size = 0;
        if constexpr (hasBias) {
            // 124: 分配124 Kb
            aL1Size = 124 * GetKBUnit<xType>();
        } else {
            // 128: 分配128 Kb
            aL1Size = 128 * GetKBUnit<xType>();
        }
        uint64_t maxKaDepth = aL1Size / (mL1Align * offsetParam[ctrlParam.processId].kbL1Size);
        if (kaDepth > maxKaDepth) {
            kaDepth = maxKaDepth;
        }
        offsetParam[ctrlParam.processId].kaL1Size = kaDepth * offsetParam[ctrlParam.processId].kbL1Size;

        basicBlock_.ComputeBasicBlock(offsetParam[ctrlParam.processId], offsetParam[GetSwitchedProcessId(ctrlParam)]);
        ctrlParam.processId = GetSwitchedProcessId(ctrlParam);
    }
    ctrlParam.nOffset += basicBlockSize * basicBlockCount;
}

GMM_FR_WEIGHT_QUANT_RESPLIT_CONTROLLER_TEMPLATE_PARAM
__aicore__ inline void GMM_FR_WEIGHT_QUANT_RESPLIT_CONTROLLER_CLASS::UpdateGmAddr(uint64_t mSize, uint64_t kSize,
                                                                                  uint64_t nSize)
{
    xGm_ += mSize * kSize;
    weightGm_ += (nSize * kSize) >> 1;
    uint64_t kAlignSize = CeilAlign(kSize, K_ALIGNMENT64);
    antiquantScaleGm_ += nSize * CeilDivide(kAlignSize, static_cast<uint64_t>(tiling_->groupSize));

    perTokenScaleGm_ += mSize * CeilDivide(kAlignSize, static_cast<uint64_t>(tiling_->groupSize));

    biasGm_ += nSize;
    yGm_ += mSize * nSize;

    logitsAddr_ += mSize;
    rowIndexAddr_ += mSize;
}

GMM_FR_WEIGHT_QUANT_RESPLIT_CONTROLLER_TEMPLATE_PARAM
__aicore__ inline uint64_t GMM_FR_WEIGHT_QUANT_RESPLIT_CONTROLLER_CLASS::GetSplitValueFromGroupList(uint64_t groupIdx)
{
    uint64_t splitValue = 0;
    if (tiling_->groupListType == 0) {
        uint64_t offset = static_cast<uint64_t>(groupListGm_.GetValue(groupIdx));
        splitValue = offset - preOffset_;
        preOffset_ = offset;
    } else {
        splitValue = static_cast<uint64_t>(groupListGm_.GetValue(groupIdx));
    }
    return splitValue;
}

GMM_FR_WEIGHT_QUANT_RESPLIT_CONTROLLER_TEMPLATE_PARAM
__aicore__ inline uint64_t GMM_FR_WEIGHT_QUANT_RESPLIT_CONTROLLER_CLASS::GetSwitchedProcessId(
    const BasicBlockControlParam &ctrlParam)
{
    return 1 - ctrlParam.processId;
}

} // namespace GROUPED_MATMUL_FINALIZE_ROUTING

#endif // GMM_FR_WEIGHT_QUANT_RESPLIT_CONTROLLER_H
