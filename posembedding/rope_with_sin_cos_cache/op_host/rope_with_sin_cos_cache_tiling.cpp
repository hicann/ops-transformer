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
 * \file rope_with_sin_cos_cache.cpp
 * \brief
 */
#include <iostream>
#include <log/log.h>
#include <cstdio>
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "rope_with_sin_cos_cache_tiling.h"

namespace {
constexpr uint32_t INPUT_POSITION_INDEX = 0;
constexpr uint32_t INPUT_QUERY_IN_INDEX = 1;
constexpr uint32_t INPUT_KEY_IN_INDEX = 2;
constexpr uint32_t INPUT_COSSINCACHE_INDEX = 3;

constexpr uint32_t INPUT_NUM_QHEADS_INDEX = 0;
constexpr uint32_t INPUT_NUM_KHEADS_INDEX = 1;
constexpr uint32_t INPUT_HEAD_SIZE_INDEX = 2;
constexpr uint32_t INPUT_MROPE_SECTION_INDEX = 3;
constexpr uint32_t INPUT_QSTRIDE_INDEX = 4;
constexpr uint32_t INPUT_KSTRIDE_INDEX = 5;
constexpr uint32_t INPUT_IS_NEOXSTYLE_INDEX = 6;
constexpr uint32_t INPUT_CACHE_MODE_INDEX = 7;

constexpr uint32_t INDEX_QUERYOUT_OUTPUT_ = 0;
constexpr uint32_t INDEX_KEYOUT_OUTPUT = 1;

static constexpr uint32_t TILING_BF16 = 20;
static constexpr uint32_t TILING_FP16 = 21;
static constexpr uint32_t TILING_FP32 = 22;
static constexpr uint32_t HAlignSize = 64;
static constexpr uint32_t MRopeSectionMaxSize = 4;

constexpr size_t DIM_0 = 0;
constexpr size_t DIM_1 = 1;

constexpr int64_t UB_SIZE = static_cast<int64_t>(192) * 1024;
constexpr uint32_t FP32_DTYPE_SIZE = 4;

struct TilingParams {
    uint64_t core_num_use = 0;
    uint64_t num_tokens = 0;
    uint64_t num_q_heads = 0;
    uint64_t num_kv_heads = 0;
    uint64_t head_size = 0;
    uint64_t rotary_dim = 0;
    uint64_t mrope_section0 = 0;
    uint64_t mrope_section1 = 0;
    uint64_t mrope_section2 = 0;
    uint64_t mrope_section3 = 0;
    uint64_t q_leading_dimension = 0;
    uint64_t k_leading_dimension = 0;
    uint64_t isNeoxStyle = 0;
    uint64_t front_core = 0;
    uint64_t tail_core = 0;
    uint64_t num_tokens_front_core_each_loop = 0;
    uint64_t num_tokens_tail_core_each_loop = 0;
    uint64_t num_tokens_each_front_core = 0;
    uint64_t num_tokens_each_tail_core = 0;
    uint64_t loop_time_each_front_core = 0;
    uint64_t loop_time_each_tail_core = 0;
    uint64_t num_tokens_front_core_last_loop = 0;
    uint64_t num_tokens_tail_core_last_loop = 0;
    uint64_t loop_for_one_token = 0;
    uint64_t num_heads_each_loop = 0;
    uint64_t num_heads_last_loop = 0;
    uint64_t loop_along_qheads = 0;
    uint64_t loop_along_kheads = 0;
    uint64_t num_qheads_each_loop = 0;
    uint64_t num_qheads_last_loop = 0;
    uint64_t num_kheads_each_loop = 0;
    uint64_t num_kheads_last_loop = 0;
    uint64_t tilingKey = 0;
    uint64_t cacheMode = 0;
};
} // namespace

namespace optiling {

inline uint32_t GetLengthByType(int32_t dtype)
{
    switch (dtype) {
        case ge::DT_FLOAT16:
        case ge::DT_INT16:
        case ge::DT_UINT16:
        case ge::DT_BF16:
            return sizeof(int16_t);
        case ge::DT_FLOAT:
        case ge::DT_INT32:
        case ge::DT_UINT32:
            return sizeof(int32_t);
        case ge::DT_DOUBLE:
        case ge::DT_INT64:
        case ge::DT_UINT64:
            return sizeof(int64_t);
        default:
            return 0;
    }
}

static void SetTiling(TilingParams &params, RopeWithSinCosCacheTilingData &tiling)
{
    tiling.set_core_num_use(params.core_num_use);
    tiling.set_num_tokens(params.num_tokens);
    tiling.set_num_q_heads(params.num_q_heads);
    tiling.set_num_kv_heads(params.num_kv_heads);
    tiling.set_head_size(params.head_size);
    tiling.set_rotary_dim(params.rotary_dim);
    tiling.set_mrope_section0(params.mrope_section0);
    tiling.set_mrope_section1(params.mrope_section1);
    tiling.set_mrope_section2(params.mrope_section2);
    tiling.set_mrope_section3(params.mrope_section3);
    tiling.set_q_leading_dimension(params.q_leading_dimension);
    tiling.set_k_leading_dimension(params.k_leading_dimension);
    tiling.set_isNeoxStyle(params.isNeoxStyle);
    tiling.set_cacheMode(params.cacheMode);
    tiling.set_front_core(params.front_core);
    tiling.set_tail_core(params.tail_core);
    tiling.set_num_tokens_front_core_each_loop(params.num_tokens_front_core_each_loop);
    tiling.set_num_tokens_tail_core_each_loop(params.num_tokens_tail_core_each_loop);
    tiling.set_num_tokens_each_front_core(params.num_tokens_each_front_core);
    tiling.set_num_tokens_each_tail_core(params.num_tokens_each_tail_core);
    tiling.set_loop_time_each_front_core(params.loop_time_each_front_core);
    tiling.set_loop_time_each_tail_core(params.loop_time_each_tail_core);
    tiling.set_num_tokens_front_core_last_loop(params.num_tokens_front_core_last_loop);
    tiling.set_num_tokens_tail_core_last_loop(params.num_tokens_tail_core_last_loop);
    tiling.set_loop_for_one_token(params.loop_for_one_token);
    tiling.set_loop_along_qheads(params.loop_along_qheads);
    tiling.set_loop_along_kheads(params.loop_along_kheads);
    tiling.set_num_qheads_each_loop(params.num_qheads_each_loop);
    tiling.set_num_qheads_last_loop(params.num_qheads_last_loop);
    tiling.set_num_kheads_each_loop(params.num_kheads_each_loop);
    tiling.set_num_kheads_last_loop(params.num_kheads_last_loop);
}

static void PrintTilingData(const gert::TilingContext *context, RopeWithSinCosCacheTilingData &tiling)
{
    OP_LOGD(context->GetNodeName(), ">>>>>>>>>> Start to print RopeWithSinCosCache tiling data <<<<<<<<<<");
    OP_LOGD(context->GetNodeName(), "RopeWithSinCosCache_tiling: coreNumUse is %ld.", tiling.get_core_num_use());
    OP_LOGD(context->GetNodeName(), "RopeWithSinCosCache_tiling: num tokens is %ld.", tiling.get_num_tokens());
    OP_LOGD(context->GetNodeName(), "RopeWithSinCosCache_tiling: num_q_heads is %ld.", tiling.get_num_q_heads());
    OP_LOGD(context->GetNodeName(), "RopeWithSinCosCache_tiling: num_kv_heads is %ld.", tiling.get_num_kv_heads());
    OP_LOGD(context->GetNodeName(), "RopeWithSinCosCache_tiling: head_size is %ld.", tiling.get_head_size());
    OP_LOGD(context->GetNodeName(), "RopeWithSinCosCache_tiling: rotary_dim is %ld.", tiling.get_rotary_dim());
    OP_LOGD(context->GetNodeName(), "RopeWithSinCosCache_tiling: mrope_section0 is %ld.", tiling.get_mrope_section0());
    OP_LOGD(context->GetNodeName(), "RopeWithSinCosCache_tiling: mrope_section1 is %ld.", tiling.get_mrope_section1());
    OP_LOGD(context->GetNodeName(), "RopeWithSinCosCache_tiling: mrope_section2 is %ld.", tiling.get_mrope_section2());
    OP_LOGD(context->GetNodeName(), "RopeWithSinCosCache_tiling: mrope_section3 is %ld.", tiling.get_mrope_section3());
    OP_LOGD(context->GetNodeName(), "RopeWithSinCosCache_tiling: q_leading_dimension is %ld.",
            tiling.get_q_leading_dimension());
    OP_LOGD(context->GetNodeName(), "RopeWithSinCosCache_tiling: k_leading_dimension is %ld.",
            tiling.get_k_leading_dimension());
    OP_LOGD(context->GetNodeName(), "RopeWithSinCosCache_tiling: isNeoxStyle is %ld.", tiling.get_isNeoxStyle());
    OP_LOGD(context->GetNodeName(), "RopeWithSinCosCache_tiling: cacheMode is %ld.", tiling.get_cacheMode());
    OP_LOGD(context->GetNodeName(), "RopeWithSinCosCache_tiling: front_core is %ld.", tiling.get_front_core());
    OP_LOGD(context->GetNodeName(), "RopeWithSinCosCache_tiling: tail_core is %ld.", tiling.get_tail_core());
    OP_LOGD(context->GetNodeName(), "RopeWithSinCosCache_tiling: num_tokens_front_core_each_loop is %ld.",
            tiling.get_num_tokens_front_core_each_loop());
    OP_LOGD(context->GetNodeName(), "RopeWithSinCosCache_tiling: num_tokens_tail_core_each_loop is %ld.",
            tiling.get_num_tokens_tail_core_each_loop());
    OP_LOGD(context->GetNodeName(), "RopeWithSinCosCache_tiling: num_tokens_each_front_core is %ld.",
            tiling.get_num_tokens_each_front_core());
    OP_LOGD(context->GetNodeName(), "RopeWithSinCosCache_tiling: num_tokens_each_tail_core is %ld.",
            tiling.get_num_tokens_each_tail_core());
    OP_LOGD(context->GetNodeName(), "RopeWithSinCosCache_tiling: loop_time_each_front_core is %ld.",
            tiling.get_loop_time_each_front_core());
    OP_LOGD(context->GetNodeName(), "RopeWithSinCosCache_tiling: loop_time_each_tail_core is %ld.",
            tiling.get_loop_time_each_tail_core());
    OP_LOGD(context->GetNodeName(), "RopeWithSinCosCache_tiling: num_tokens_front_core_last_loop is %ld.",
            tiling.get_num_tokens_front_core_last_loop());
    OP_LOGD(context->GetNodeName(), "RopeWithSinCosCache_tiling: num_tokens_tail_core_last_loop is %ld.",
            tiling.get_num_tokens_tail_core_last_loop());
    OP_LOGD(context->GetNodeName(), "RopeWithSinCosCache_tiling: loop_for_one_token is %ld.",
            tiling.get_loop_for_one_token());
    OP_LOGD(context->GetNodeName(), "RopeWithSinCosCache_tiling: loop_along_qheads is %ld.",
            tiling.get_loop_along_qheads());
    OP_LOGD(context->GetNodeName(), "RopeWithSinCosCache_tiling: loop_along_kheads is %ld.",
            tiling.get_loop_along_kheads());
    OP_LOGD(context->GetNodeName(), "RopeWithSinCosCache_tiling: num_qheads_each_loop is %ld.",
            tiling.get_num_qheads_each_loop());
    OP_LOGD(context->GetNodeName(), "RopeWithSinCosCache_tiling: num_qheads_last_loop is %ld.",
            tiling.get_num_qheads_last_loop());
    OP_LOGD(context->GetNodeName(), "RopeWithSinCosCache_tiling: num_kheads_each_loop is %ld.",
            tiling.get_num_kheads_each_loop());
    OP_LOGD(context->GetNodeName(), "RopeWithSinCosCache_tiling: num_kheads_last_loop is %ld.",
            tiling.get_num_kheads_last_loop());
}

static ge::graphStatus TilingKeyChose(const gert::TilingContext *context, TilingParams &params)
{
    auto qDtype = context->GetInputDesc(INPUT_QUERY_IN_INDEX)->GetDataType();
    if (qDtype == ge::DT_BF16) {
        params.tilingKey = TILING_BF16;
        return ge::GRAPH_SUCCESS;
    }
    if (qDtype == ge::DT_FLOAT) {
        params.tilingKey = TILING_FP32;
        return ge::GRAPH_SUCCESS;
    }
    if (qDtype == ge::DT_FLOAT16) {
        params.tilingKey = TILING_FP16;
        return ge::GRAPH_SUCCESS;
    }
    return ge::GRAPH_FAILED;
}

static ge::graphStatus TilingCompute(gert::TilingContext *context, TilingParams &params)
{
    uint64_t numTokens =
        static_cast<uint64_t>(context->GetOutputShape(INDEX_QUERYOUT_OUTPUT_)->GetStorageShape().GetDim(DIM_0));
    uint64_t numQheads = params.num_q_heads;
    uint64_t numKheads = params.num_kv_heads;
    uint64_t headSize = params.head_size;

    if (headSize > 0) {
        uint64_t queryLastDim =
            static_cast<uint64_t>(context->GetInputShape(INPUT_QUERY_IN_INDEX)->GetStorageShape().GetDim(DIM_1));
        uint64_t keyLastDim =
            static_cast<uint64_t>(context->GetInputShape(INPUT_KEY_IN_INDEX)->GetStorageShape().GetDim(DIM_1));
        if (queryLastDim % headSize != 0) {
            OP_LOGE(context->GetNodeName(), "queryIn.shape[1] must be divisible by headSize, but got %lu and %lu.",
                    queryLastDim, headSize);
        }
        if (keyLastDim % headSize != 0) {
            OP_LOGE(context->GetNodeName(), "keyIn.shape[1] must be divisible by headSize, but got %lu and %lu.",
                    keyLastDim, headSize);
        }
    }

    auto cosSinSize = context->GetInputShape(INPUT_COSSINCACHE_INDEX)->GetStorageShape().GetDimNum();
    uint64_t rotaryDim = static_cast<uint64_t>(
        context->GetInputShape(INPUT_COSSINCACHE_INDEX)->GetStorageShape().GetDim(cosSinSize - 1));

    int32_t qDtype = context->GetInputDesc(INPUT_QUERY_IN_INDEX)->GetDataType();
    uint32_t queryInTypeLength = GetLengthByType(qDtype);
    OP_CHECK_IF((queryInTypeLength == 0), OP_LOGE(context->GetNodeName(), "queryInTypeLength can't be 0."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF((headSize % (HAlignSize / queryInTypeLength) != 0),
                OP_LOGE(context->GetNodeName(), "headSize must be 64-byte aligned."), return ge::GRAPH_FAILED);

    OP_CHECK_IF((rotaryDim % (HAlignSize / queryInTypeLength) != 0),
                OP_LOGE(context->GetNodeName(), "rotaryDim must be 64-byte aligned."), return ge::GRAPH_FAILED);

    OP_CHECK_IF((headSize < rotaryDim), OP_LOGE(context->GetNodeName(), "headSize can't be smaller than rotaryDim."),
                return ge::GRAPH_FAILED);

    auto platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint64_t coreNum = static_cast<uint64_t>(ascendcPlatform.GetCoreNum());

    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    auto maxUbSize = static_cast<uint64_t>(ubSizePlatForm);

    uint64_t totalDataNum = numTokens;
    uint64_t dataTypeSize = FP32_DTYPE_SIZE;
    uint64_t front_core = totalDataNum % coreNum != 0 ? static_cast<uint64_t>(totalDataNum % coreNum) : coreNum;
    uint64_t tail_core = totalDataNum <= coreNum ? 0 : coreNum - front_core;
    uint64_t numBlocks = front_core + tail_core;

    uint64_t numHeadsMax = numQheads > numKheads ? numQheads : numKheads;
    uint64_t allSize = params.isNeoxStyle == 1UL ?
                           static_cast<uint64_t>(numHeadsMax * (rotaryDim * 5UL + headSize * 2UL) * dataTypeSize) :
                           static_cast<uint64_t>(numHeadsMax * (rotaryDim * 6UL + headSize * 2UL) * dataTypeSize);
    uint64_t maxNPerLoopForUb =
        (maxUbSize - rotaryDim * 10UL * dataTypeSize) / allSize; // ub每次能载入最大行数（包括所有计算数据）;

    uint64_t num_tokens_each_front_core = (totalDataNum + coreNum - 1) / coreNum;
    uint64_t loop_time_each_front_core = maxNPerLoopForUb == 0 ? 0 :
                                                                 (num_tokens_each_front_core + maxNPerLoopForUb - 1UL) /
                                                                     static_cast<uint64_t>(maxNPerLoopForUb);

    uint64_t num_tokens_front_core_each_loop =
        loop_time_each_front_core == 1UL ? num_tokens_each_front_core : maxNPerLoopForUb;
    uint64_t num_tokens_front_core_last_loop =
        loop_time_each_front_core == 1UL ?
            0 :
            num_tokens_each_front_core - num_tokens_front_core_each_loop * (loop_time_each_front_core - 1UL);
    uint64_t num_tokens_each_tail_core = totalDataNum / coreNum;
    uint64_t loop_time_each_tail_core =
        maxNPerLoopForUb == 0 ? 0 : (num_tokens_each_tail_core + maxNPerLoopForUb - 1) / maxNPerLoopForUb;
    uint64_t num_tokens_tail_core_each_loop =
        loop_time_each_tail_core <= 1UL ? num_tokens_each_tail_core : maxNPerLoopForUb;
    uint64_t num_tokens_tail_core_last_loop =
        static_cast<uint64_t>(loop_time_each_front_core) == 1UL ?
            0 :
            num_tokens_each_tail_core - num_tokens_tail_core_each_loop * (loop_time_each_tail_core - 1UL);

    uint64_t numHeadsForUb =
        params.isNeoxStyle == 1UL ?
            (maxUbSize - rotaryDim * 10UL * dataTypeSize) / ((rotaryDim * 5UL + headSize * 2UL) * dataTypeSize) :
            (maxUbSize - rotaryDim * 10UL * dataTypeSize) / ((rotaryDim * 6UL + headSize * 2UL) * dataTypeSize);

    uint64_t loop_for_one_token =
        numHeadsMax > numHeadsForUb ? 1UL : 0UL; // 取1时，对numheads循环，maxNPerLoopForUb == 0
    uint64_t loop_along_qheads = numHeadsForUb == 0 ? 0 : (numQheads + numHeadsForUb - 1UL) / numHeadsForUb;
    uint64_t loop_along_kheads = numHeadsForUb == 0 ? 0 : (numKheads + numHeadsForUb - 1UL) / numHeadsForUb;
    uint64_t num_qheads_each_loop = loop_along_qheads == 1UL ? numQheads : numHeadsForUb;
    uint64_t num_qheads_last_loop =
        loop_along_qheads == 1UL ? 0 : numQheads - num_qheads_each_loop * (loop_along_qheads - 1UL);
    uint64_t num_kheads_each_loop = loop_along_kheads == 1UL ? numKheads : numHeadsForUb;
    uint64_t num_kheads_last_loop =
        loop_along_kheads == 1UL ? 0 : numKheads - num_kheads_each_loop * (loop_along_kheads - 1UL);

    params.num_tokens = numTokens;
    params.rotary_dim = rotaryDim;
    params.core_num_use = numBlocks;
    params.front_core = front_core;
    params.tail_core = tail_core;

    params.num_tokens_each_front_core = num_tokens_each_front_core;
    params.loop_time_each_front_core = loop_time_each_front_core;
    params.num_tokens_front_core_each_loop = num_tokens_front_core_each_loop;
    params.num_tokens_front_core_last_loop = num_tokens_front_core_last_loop;
    params.num_tokens_each_tail_core = num_tokens_each_tail_core;
    params.loop_time_each_tail_core = loop_time_each_tail_core;
    params.num_tokens_tail_core_each_loop = num_tokens_tail_core_each_loop;
    params.num_tokens_tail_core_last_loop = num_tokens_tail_core_last_loop;

    params.loop_for_one_token = loop_for_one_token;
    params.loop_along_qheads = loop_along_qheads;
    params.loop_along_kheads = loop_along_kheads;
    params.num_qheads_each_loop = num_qheads_each_loop;
    params.num_qheads_last_loop = num_qheads_last_loop;
    params.num_kheads_each_loop = num_kheads_each_loop;
    params.num_kheads_last_loop = num_kheads_last_loop;

    context->SetBlockDim(numBlocks);
    context->SetTilingKey(params.tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingRopeWithSinCosCache(gert::TilingContext *context)
{
    TilingParams params;
    RopeWithSinCosCacheTilingData tiling;

    const auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    context->SetBlockDim(totalCoreNum);
    params.core_num_use = context->GetBlockDim();

    auto *attrs = context->GetAttrs();

    const uint64_t *attrNumQHeads = attrs->GetAttrPointer<uint64_t>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrNumQHeads);
    params.num_q_heads = static_cast<uint64_t>(*attrNumQHeads);

    const uint64_t *attrNumKVHeads = attrs->GetAttrPointer<uint64_t>(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrNumKVHeads);
    params.num_kv_heads = static_cast<uint64_t>(*attrNumKVHeads);

    const uint64_t *attrHeadSize = attrs->GetAttrPointer<uint64_t>(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrHeadSize);
    params.head_size = static_cast<uint64_t>(*attrHeadSize);

    const auto attrMRopeSection = attrs->GetAttrPointer<gert::ContinuousVector>(3);
    const uint64_t *attrMRopeSectionData = reinterpret_cast<const uint64_t *>(attrMRopeSection->GetData());
    if (attrMRopeSectionData != nullptr) {
        params.mrope_section0 = attrMRopeSectionData[0];
        params.mrope_section1 = attrMRopeSectionData[1];
        params.mrope_section2 = attrMRopeSectionData[2];
        params.mrope_section3 = attrMRopeSection->GetSize() == MRopeSectionMaxSize ? attrMRopeSectionData[3] : 0;
    }

    const uint64_t *attrQStride = attrs->GetAttrPointer<uint64_t>(4);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrQStride);
    params.q_leading_dimension = static_cast<uint64_t>(*attrQStride);

    const uint64_t *attrKStride = attrs->GetAttrPointer<uint64_t>(5);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrKStride);
    params.k_leading_dimension = static_cast<uint64_t>(*attrKStride);

    const bool *attrIsNeoxStyle = attrs->GetAttrPointer<bool>(6);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrIsNeoxStyle);
    params.isNeoxStyle = static_cast<bool>(*attrIsNeoxStyle);

    const uint64_t *attrCacheMode = attrs->GetAttrPointer<uint64_t>(INPUT_CACHE_MODE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrCacheMode);
    params.cacheMode = static_cast<uint64_t>(*attrCacheMode);
    OP_CHECK_IF((attrMRopeSection->GetSize() == MRopeSectionMaxSize) && (params.cacheMode != 0),
                OP_LOGE(context->GetNodeName(),
                        "cacheMode is only support to be 0 but got %lu when the size of mRopeSection equals %u.",
                        params.cacheMode, MRopeSectionMaxSize),
                return ge::GRAPH_FAILED);

    TilingKeyChose(context, params);
    TilingCompute(context, params);
    SetTiling(params, tiling);
    PrintTilingData(context, tiling);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t *workspaces = context->GetWorkspaceSizes(1);

    size_t UserWorkspaceSize = 0;

    workspaces[0] = sysWorkspaceSize + UserWorkspaceSize;

    return ge::GRAPH_SUCCESS;
}

struct TilingmropeCompileInfo {};

static ge::graphStatus TilingPreparemrope(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(RopeWithSinCosCache)
    .Tiling(TilingRopeWithSinCosCache)
    .TilingParse<TilingmropeCompileInfo>(TilingPreparemrope);
} // namespace optiling
