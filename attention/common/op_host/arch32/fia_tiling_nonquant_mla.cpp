/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fia_tiling_nonquant_mla.cpp
 * \brief
 */

#include <map>
#include <vector>
#include <numeric>
#include <algorithm>
#include <graph/utils/type_utils.h>
#include "err/ops_err.h"
#include "fia_tiling_nonquant_mla.h"
#include "../fia_tiling_templates_registry.h"
#include "../split_core.h"
#include "../../../fused_infer_attention_score/op_kernel/fused_infer_attention_score_template_tilingkey.h"

using namespace ge;
using namespace AscendC;
namespace optiling {

constexpr uint64_t PRE_LOAD_NUM_MLA = 2;

constexpr uint64_t FIA_TILINGKEYOFFSET = uint64_t(100000000000000000UL); // 10^17
constexpr uint64_t FIA_PERF_MODE_TILINGKEYOFFSET = uint64_t(1000000000000000UL); // 10^15

template <typename T> 
inline auto Align(T num, T rnd) -> T
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd) - 1) / (rnd) * (rnd)));
}

constexpr uint64_t RecursiveSum()
{
    return 0;
}

template <typename T, typename... Args> 
constexpr uint64_t RecursiveSum(T templateId, Args... templateIds)
{
    return static_cast<uint64_t>(templateId) + 10U * RecursiveSum(templateIds...);
}
template <typename... Args> 
constexpr uint64_t FIA_GET_TILINGKEY(Args... templateIds)
{
    return RecursiveSum(templateIds...);
}

void FiaTilingNonQuantMla::InitTilingInfo(TilingInfo *tilingInfo)
{
    fiaInfo_ = static_cast<FiaTilingInfo *>(tilingInfo);
}

ge::graphStatus FiaTilingNonQuantMla::GetPlatformInfo()
{
    OP_CHECK_IF(fiaInfo_->platformInfo == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR(fiaInfo_->opName, "GetPlatformInfo is nullptr."), return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(fiaInfo_->platformInfo);
    libapiSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();
    aivNum_ = ascendcPlatform.GetCoreNumAiv();
    aicNum_ = ascendcPlatform.GetCoreNumAic();

    OP_CHECK_IF(aicNum_ == 0 || aivNum_ == 0,
        OPS_REPORT_VECTOR_INNER_ERR(fiaInfo_->opName, "num of core obtained is 0."), return GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

bool FiaTilingNonQuantMla::IsCapable()
{
    if (fiaInfo_ == nullptr) {
        return false;
    }

    ge::DataType qDataType = fiaInfo_->inputQType;
    ge::DataType kDataType = fiaInfo_->inputKvType;

    if (fiaInfo_->ropeMode == RopeMode::ROPE_SPLIT) {
        // MLA非量化
        if ((qDataType == ge::DT_FLOAT16 || qDataType == ge::DT_BF16) && (qDataType == kDataType)) {
            return true;
        }
    }
    return false;
}

void FiaTilingNonQuantMla::GenTilingKey()
{
    uint8_t inputQVal{0}, inputKvVal{0}, outputVal{0};

    const std::map<ge::DataType, uint8_t> typeMap = {
        {ge::DT_FLOAT16, 0U}, {ge::DT_BF16, 2U}, {ge::DT_INT8, 3U}, {ge::DT_INT4, 4U},
    };

    if (typeMap.find(fiaInfo_->inputQType) != typeMap.end()) {
        inputQVal = typeMap.at(fiaInfo_->inputQType);
    }
    if (typeMap.find(fiaInfo_->inputKvType) != typeMap.end()) {
        inputKvVal = typeMap.at(fiaInfo_->inputKvType);
    }
    if (typeMap.find(fiaInfo_->outputType) != typeMap.end()) {
        outputVal = typeMap.at(fiaInfo_->outputType);
    }

    bool isFlashDecode = (kvSplit_ > 0);
    bool isPageAttention = (fiaInfo_->pageAttentionFlag && fiaInfo_->s2Size != 0);
    tilingKey_ = GET_TPL_TILING_KEY(static_cast<uint8_t>(inputQVal), static_cast<uint8_t>(inputKvVal), static_cast<uint8_t>(outputVal), static_cast<uint8_t>(isPageAttention),
                                    static_cast<uint8_t>(fiaInfo_->inputLayout),
                                    static_cast<uint8_t>(fiaInfo_->inputKvLayout), static_cast<uint8_t>(isFlashDecode), static_cast<uint8_t>(fiaInfo_->sysPrefixFlag),
                                    0, 0, 0, 0, 3, 0, 0, 0, 0);
    OP_LOGI(fiaInfo_->opName, "FIA tilingKey_: %lu.", tilingKey_);
}

bool FiaTilingNonQuantMla::IsFlashDecode()
{
    uint32_t tndFDCoreArrLen = tilingData_->fdParams.get_numOfFdHead();
    return tndFDCoreArrLen > 0U;
}

bool FiaTilingNonQuantMla::DealSameSeqEachBatch() const
{
    if (!fiaInfo_->batchContinuousFlag) {
        if (fiaInfo_->actualSeqLenFlag) {
            return fiaInfo_->isSameActualseq;
        } else {
            return fiaInfo_->isSameSeqAllKVTensor;
        }
    } else {
        return fiaInfo_->isSameActualseq;
    }
}

void FiaTilingNonQuantMla::ZeroTensorProcess() const
{
    if (fiaInfo_->s2Size == 0) {
        /*
         * 1024，空tensor场景下，作为默认值完成后续计算
         * 避免matmal tiling  softmax tiling异常
         * kernel计算使用真实的seqSize=0, 与actuseq_len流程归一
         */
        fiaInfo_->s2Size = 1024;
    }
}

void FiaTilingNonQuantMla::InitParams()
{
    perfMode_ = IfaPerfMode::CUBE_VIEW_MM_MLA;
    coreNum_ = aicNum_;

    headDimAlign_ = Align(fiaInfo_->qkHeadDim, BYTE_BLOCK); // 元素个数按照基本块大小对齐
    ZeroTensorProcess();
}

void FiaTilingNonQuantMla::CalcInnerSize(uint32_t seqSize)
{
    sInnerSize_ = 512U;
    // FlashDecode时，如果S2的计算量>=256(确保切分后不小于128)但又不足以分2次计算时，则修改sInnerSize_，均分为2份进行计算，确保Nbuffer=2
    if (splitKVFlag_ && fiaInfo_->inputLayout != TilingKeyLayout::TND) {
        if (seqSize == 256U) {
            sInnerSize_ = 128U;
        } else if (seqSize > 256U && seqSize <= sInnerSize_) {
            sInnerSize_ = (sInnerSize_ + 1U) / 2U;
        }
    }

    // PA特性泛化场景，blockSize可能为112等值，无法被sInnerSize_整除，当step*base跨block时，搬运处理复杂，通过向下对齐避免
    if (fiaInfo_->pageAttentionFlag && fiaInfo_->blockSize != 0) {
        uint32_t blockSize = static_cast<uint32_t>(fiaInfo_->blockSize);
        if ((sInnerSize_ > blockSize) && (sInnerSize_ % blockSize != 0)) {
            sInnerSize_ = (sInnerSize_ / blockSize) * blockSize;
        }
    }

    sInnerLoopTimes_ = (seqSize + sInnerSize_ - 1U) / sInnerSize_;
    sInnerSizeTail_ = seqSize - (sInnerLoopTimes_ - 1U) * sInnerSize_;
    // tiling下沉 && flash decoder场景时，sInnerSize_基块大小不按照真实值修改
    // 否则会导致 tiling下沉 && flash decoder 场景时开辟workspace空间大小小于真实运行时所需的workspace大小
    if (sInnerSize_ > seqSize) {
        sInnerSize_ = seqSize;
    }
    sInnerSizeAlign_ = Align(sInnerSize_, BYTE_BLOCK); // 元素个数按照基本块大小对齐

    CalcMBaseSize();
}

/**
 * @brief 根据shape调整mBaseSize，在一定范围内可以不开FD
 */
void FiaTilingNonQuantMla::CalcMBaseSize()
{
    uint32_t bN2 = fiaInfo_->bSize * fiaInfo_->n2Size;
    uint32_t s1G = fiaInfo_->s1Size * fiaInfo_->gSize;
    if (bN2 == 0U || aicNum_ == 0U) {
        return;
    }
    // 若BN2能整除核数或者KVS不等长，不考虑调整mBaseSize
    if (bN2 % aicNum_ == 0U || !DealSameSeqEachBatch()) {
        return;
    }
    
    // 求B*N2与核数的最小公倍数
    uint32_t originalA = bN2;
    uint32_t originalB = aicNum_;
    // 欧几里得算法计算最大公约数
    while (originalB != 0U) {
        originalA %= originalB;
        std::swap(originalA, originalB);
    }
    
    // 计算最小公倍数
    uint32_t lcm = (bN2 / originalA) * aicNum_;
    uint32_t splitOfS1G = lcm / bN2;    // 求S1G被切多少份，可以不开FD
    // 如果不能等分S1G，则不切S2无法做到负载不均衡，不修改mBaseSize，直接按照最大mBaseSize进行切分
    if (splitOfS1G == 0U || s1G % splitOfS1G != 0U) {
        return;
    }
    uint32_t mBaseSizeTmp = s1G / splitOfS1G;   // 计算可以等分S1G的mBaseSize
    // 128：最小的mBaseSize，计算的理论mBaseSize需要在原定mBaseSize和最小mBaseSize区间内
    if (mBaseSizeTmp > mBaseSize_ || mBaseSizeTmp < 128U) {
        return;
    }

    mBaseSize_ = mBaseSizeTmp;  // 满足以上条件则更新mBaseSize
}

void FiaTilingNonQuantMla::Split()
{
    CalcInnerSize(static_cast<uint32_t>(fiaInfo_->s2Size));

    //构造分核输入参数
    BaseInfo baseInfo;
    baseInfo.bSize = fiaInfo_->bSize;
    baseInfo.n2Size = fiaInfo_->n2Size;
    baseInfo.gSize = fiaInfo_->gSize;
    baseInfo.s2Size = static_cast<uint32_t>(fiaInfo_->s2Size);
    baseInfo.s1Size = fiaInfo_->s1Size;
    baseInfo.actualLenQDims = fiaInfo_->actualLenQDims;
    baseInfo.actualLenKvDims = fiaInfo_->actualLenDims;
    baseInfo.preToken = fiaInfo_->preToken;
    baseInfo.nextToken = fiaInfo_->nextToken;
    baseInfo.slidingFlag = fiaInfo_->slidingFlag;

    if (fiaInfo_->inputLayout == TilingKeyLayout::TND) {
        baseInfo.isAccumSeqS1 = true;
    } else {
        baseInfo.s1Size = fiaInfo_->s1Size;
        baseInfo.s2Size = fiaInfo_->s2Size;
    }
    if (fiaInfo_->opParamInfo.actualSeqLengthsQ.tensor != nullptr) {
        baseInfo.actualSeqS1Size = fiaInfo_->opParamInfo.actualSeqLengthsQ.tensor->GetData<int64_t>();
    }
    if (fiaInfo_->opParamInfo.actualSeqLengths.tensor != nullptr) {
        baseInfo.actualSeqS2Size = fiaInfo_->opParamInfo.actualSeqLengths.tensor->GetData<int64_t>();
    }
    baseInfo.isAccumSeqS2 = false;

    InnerSplitParams innerSplitParams;
    innerSplitParams.s1GBaseSize = mBaseSize_;
    innerSplitParams.s2BaseSize = sInnerSize_;
    tilingData_->innerSplitParams.set_mBaseSize(innerSplitParams.s1GBaseSize);
    tilingData_->innerSplitParams.set_s2BaseSize(innerSplitParams.s2BaseSize);

    //构造分核输出参数
    OuterSplitParams outerSplitParams;
    outerSplitParams.bN2End = tilingData_->outerSplitParams.get_bN2End();
    outerSplitParams.gS1End = tilingData_->outerSplitParams.get_gS1End();
    outerSplitParams.s2End = tilingData_->outerSplitParams.get_s2End();
    FlashDecodeParams fDParams;
    tilingData_->fdParams.set_gS1BaseSizeOfFd(mFdBaseSize_);
    fDParams.bN2IdxOfFdHead = tilingData_->fdParams.get_bN2IdxOfFdHead();
    fDParams.gS1IdxOfFdHead = tilingData_->fdParams.get_gS1IdxOfFdHead();
    fDParams.s2SplitNumOfFdHead = tilingData_->fdParams.get_s2SplitNumOfFdHead();
    fDParams.s2SplitStartIdxOfCore = tilingData_->fdParams.get_s2SplitStartIdxOfCore();
    fDParams.gS1BaseSizeOfFd = tilingData_->fdParams.get_gS1BaseSizeOfFd();
    fDParams.gS1SplitNumOfFdHead = tilingData_->fdParams.get_gS1SplitNumOfFdHead();
    fDParams.gS1LastPartSizeOfFdHead = tilingData_->fdParams.get_gS1LastPartSizeOfFdHead();
    fDParams.gS1IdxEndOfFdHead = tilingData_->fdParams.get_gS1IdxEndOfFdHead();
    fDParams.gS1IdxEndOfFdHeadSplit = tilingData_->fdParams.get_gS1IdxEndOfFdHeadSplit();
    SplitCoreRes res;
    res.numOfFdHead = 0U;
    res.maxS2SplitNum = 1U;
    res.usedCoreNum = aicNum_;
    if (fiaInfo_->slidingFlag) {
        SplitCoreOfBand(baseInfo, innerSplitParams, aicNum_, outerSplitParams, fDParams, res); // band暂时延用优化前的分核
    } else {
        SplitCore(baseInfo, innerSplitParams, aicNum_, outerSplitParams, fDParams, res);
    }
    if (res.numOfFdHead > aicNum_ || res.usedCoreNum > aicNum_ || res.maxS2SplitNum > aicNum_ + 1U) {
        OP_LOGE(fiaInfo_->opName, "used_core_num: %u, num_of_fd_head: %u, max_s2_split_num: %u, aic_num: %u", 
            res.usedCoreNum, res.numOfFdHead, res.maxS2SplitNum, aicNum_);
    }

    tilingData_->fdParams.set_numOfFdHead(res.numOfFdHead);
    usedCoreNum_ = res.usedCoreNum;

    //kvSplitPart_,用于lse out workspace计算
    if (IsFlashDecode()) {
        splitKVFlag_ = true;
        kvSplit_++;
        kvSplitPart_ = res.maxS2SplitNum;
        SplitFD(res, fDParams, usedCoreNum_);
        tilingData_->fdParams.set_usedVecNumOfFd(res.usedVecNumOfFd);
    }
    CalcMmResSize();
}

void FiaTilingNonQuantMla::FillTilingBaseParams()
{
    tilingData_->baseParams.set_bSize(fiaInfo_->bSize);
    tilingData_->baseParams.set_s2Size(fiaInfo_->s2Size);
    tilingData_->baseParams.set_s1Size(fiaInfo_->s1Size);
    tilingData_->baseParams.set_scaleValue(fiaInfo_->scaleValue);
    tilingData_->baseParams.set_gSize(fiaInfo_->n1Size / fiaInfo_->n2Size);
    tilingData_->baseParams.set_actualSeqS1Dims(fiaInfo_->actualLenQDims);
    tilingData_->baseParams.set_actualSeqS2Dims(fiaInfo_->actualLenDims);
    tilingData_->baseParams.set_outputLayout(static_cast<uint32_t>(fiaInfo_->outputLayout));
    tilingData_->baseParams.set_slidingFlag(fiaInfo_->slidingFlag);
    tilingData_->baseParams.set_needInit(fiaInfo_->needInit);
    tilingData_->baseParams.set_usedCoreNum(usedCoreNum_);
}

void FiaTilingNonQuantMla::FillTilingPageAttenParams()
{
    tilingData_->pageAttenParams.set_blockSize(fiaInfo_->blockSize);
    tilingData_->pageAttenParams.set_maxBlockNumPerBatch(fiaInfo_->maxBlockNumPerBatch);
}

void FiaTilingNonQuantMla::FillTilingMaskParams()
{
    tilingData_->maskParams.set_attenMaskFlag(fiaInfo_->attenMaskFlag ? 1 : 0);
    tilingData_->maskParams.set_attenMaskSize(fiaInfo_->attenMaskSize);
    tilingData_->maskParams.set_preToken(fiaInfo_->preToken);
    tilingData_->maskParams.set_nextToken(fiaInfo_->nextToken);
}

void FiaTilingNonQuantMla::FillTilingWorkspaceParams()
{
    uint32_t accumOutElemNumOfOneFdTask = fiaInfo_->n2Size * mBaseSize_ * headDimAlign_;
    // max和sum总共两份
    uint32_t sumAndMaxElemNumOfOneFdTask = 2 * fiaInfo_->n2Size * mBaseSize_ * (BYTE_BLOCK / sizeof(float));
    // 每个核可能有头规约和尾规约，一共两份规约信息
    constexpr uint32_t MAX_FD_TASK_NUMS = 2;
    tilingData_->workspaceParams.set_fdAccumOutSize(aicNum_ * MAX_FD_TASK_NUMS * accumOutElemNumOfOneFdTask);
    tilingData_->workspaceParams.set_fdLogSumExpSize(aicNum_ * MAX_FD_TASK_NUMS * sumAndMaxElemNumOfOneFdTask);
    tilingData_->workspaceParams.set_mm1ResSize(mm1ResSize_);
    tilingData_->workspaceParams.set_mm2ResSize(mm2ResSize_);
}

void FiaTilingNonQuantMla::CalcMmResSize()
{
    int64_t mSize = std::min(fiaInfo_->gSize * fiaInfo_->s1Size, mBaseSize_);
    mSize = Align(mSize, static_cast<int64_t>(16U));
    mm1ResSize_ = static_cast<int64_t>(sInnerSizeAlign_) * mSize;
    mm2ResSize_ = static_cast<int64_t>(headDimAlign_) * mSize;
}

void FiaTilingNonQuantMla::CalcMaxMmResSize()
{
    constexpr int64_t S2_SIZE_MAX = 512;
    mm1ResSize_ = S2_SIZE_MAX * static_cast<int64_t>(mBaseSize_);
    mm2ResSize_ = static_cast<int64_t>(headDimAlign_) * static_cast<int64_t>(mBaseSize_);
}

void FiaTilingNonQuantMla::FillTiling()
{
    FillTilingBaseParams();
    FillTilingPageAttenParams();
    FillTilingMaskParams();
    FillTilingWorkspaceParams();
}

uint32_t FiaTilingNonQuantMla::CalcFlashDecodeParamNums(const uint32_t coreNum) const
{
    return coreNum * 2U * fiaInfo_->n2Size * mBaseSize_; // 每个核可能有头规约和尾规约，一共两份规约信息
}

uint64_t FiaTilingNonQuantMla::CalcNormalWorkspaceSize(uint32_t coreNum, int64_t mm1ResSize,
    int64_t mm2ResSize, uint32_t mBaseSize) const
{
    constexpr uint32_t MM1_RES_ELEM_SIZE = 4;      // 4: fp32
    constexpr uint32_t V1_RES_ELEM_SIZE = 2;       // 2: fp16/bf16
    constexpr uint32_t MM2_RES_ELEM_SIZE = 4;      // 4: fp32
    constexpr uint32_t V2_RES_ELEM_SIZE = 4;       // 4: fp32
    constexpr uint32_t N_UPDATE_ELEM_SIZE = 4;     // 4: int32
    constexpr uint32_t SOFTMAX_SUM_ELEM_SIZE = 4;  // 4: int32

    uint64_t workspaceSize = 0;
    workspaceSize += PRE_LOAD_NUM_MLA * coreNum * mm1ResSize * MM1_RES_ELEM_SIZE;
    workspaceSize += PRE_LOAD_NUM_MLA * coreNum * mm1ResSize * V1_RES_ELEM_SIZE;
    workspaceSize += PRE_LOAD_NUM_MLA * coreNum * mm2ResSize * MM2_RES_ELEM_SIZE;
    workspaceSize += PRE_LOAD_NUM_MLA * coreNum * mm2ResSize * V2_RES_ELEM_SIZE;
    workspaceSize += PRE_LOAD_NUM_MLA * coreNum * mBaseSize * N_UPDATE_ELEM_SIZE; //aMla nUpdate, mBaseSize=128
    workspaceSize += PRE_LOAD_NUM_MLA * coreNum * mBaseSize * SOFTMAX_SUM_ELEM_SIZE; //aMla softmaxSum, mBaseSize=128
    return workspaceSize;
}

uint64_t FiaTilingNonQuantMla::CalcFlashDecodeWorkspace(const uint32_t coreNum) const
{
    uint64_t flashDecodeParamNums = static_cast<uint64_t>(CalcFlashDecodeParamNums(coreNum));
    uint64_t accumOutSize = flashDecodeParamNums * static_cast<uint64_t>(headDimAlign_);
    uint64_t logSumExpSize = 2 * flashDecodeParamNums * (BYTE_BLOCK / sizeof(float)); // log和sum的存储空间一致,需要2份
    uint64_t workspaceSize = (accumOutSize + logSumExpSize) * sizeof(float);
    return workspaceSize;
}

void FiaTilingNonQuantMla::CalcWorkspaceSize()
{
    workspaceSize_ = libapiSize_;
    workspaceSize_ += CalcNormalWorkspaceSize(coreNum_, mm1ResSize_, mm2ResSize_, mBaseSize_);
    if (splitKVFlag_) {
        workspaceSize_ += CalcFlashDecodeWorkspace(coreNum_);
    }
}

void FiaTilingNonQuantMla::CalcMaxWorkspaceSize()
{
    CalcMaxMmResSize();
    workspaceSize_ = libapiSize_;
    workspaceSize_ += CalcNormalWorkspaceSize(coreNum_, mm1ResSize_, mm2ResSize_, mBaseSize_);
    workspaceSize_ += CalcFlashDecodeWorkspace(aicNum_);
}

void FiaTilingNonQuantMla::CalcBlockDim(uint32_t coreNum)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(fiaInfo_->platformInfo);
    auto aicNum = coreNum;
    auto aivNum = 2U * coreNum;  // vec核数量是cube核数量的两倍

    blockDim_ = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum); // 暂时与当前代码一致
    OP_LOGI(fiaInfo_->opName, "FIA block dim: %u aiv Num: %u aic Num: %u.", blockDim_, aivNum, aicNum);
}

ge::graphStatus FiaTilingNonQuantMla::DoOpTiling()
{
    if (GetPlatformInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    InitParams();
    if (fiaInfo_->isMaxWorkspace) {
        CalcMaxWorkspaceSize();
        GenTilingKey();
    } else {
        Split();
        FillTiling();
        CalcBlockDim(usedCoreNum_);
        CalcWorkspaceSize();
        GenTilingKey();
    }

    if ((SetBlockDim(blockDim_) != ge::GRAPH_SUCCESS) ||
        (SetTilingKey(tilingKey_) != ge::GRAPH_SUCCESS) ||
        (SetWorkspaceSize(workspaceSize_) != ge::GRAPH_SUCCESS) 
        //|| (SetTilingData(tilingData_) != ge::GRAPH_SUCCESS)
        ) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

// 值越小表示优先级越高. 对于FIA, 使用3位数表示优先级, 优先级编码含义为:
// 1. 百位代表非量化、伪量化、全量化等场景, 即: 0xx-非量化，1xx-伪量化, 2xx-全量化
// 2. 十位表示gqa、mla、泛化，即: x0x-mla, x1x-gpa, x2x-泛化
// 3. 个位代表特化模板到泛化模板的优先级排序
REGISTER_TILING_TEMPLATE_FIA(FusedInferAttentionScore, FiaTilingNonQuantMla,
    std::vector<int32_t>({static_cast<int32_t>(platform_ascendc::SocVersion::ASCEND910B)}), 1);
} // namespace optiling
