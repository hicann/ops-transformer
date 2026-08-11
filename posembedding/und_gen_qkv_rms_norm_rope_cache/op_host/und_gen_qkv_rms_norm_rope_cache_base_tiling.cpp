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
 * \file und_gen_qkv_rms_norm_rope_cache_base_tiling.cpp
 * \brief 公共 tiling：平台信息获取、shape/attr 解析与校验、tiling 入口注册
 */

#include "und_gen_qkv_rms_norm_rope_cache_tiling.h"
#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "op_host/tiling_util.h"
#include "graph/utils/type_utils.h"

namespace optiling {

ge::graphStatus UndGenQkvRmsNormRopeCacheTilingBase::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        auto compileInfoPtr = context_->GetCompileInfo<UndGenQkvRmsNormRopeCacheCompileInfo>();
        OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context_->GetNodeName(), "CompileInfo is nullptr."),
                    return ge::GRAPH_FAILED);
        coreNum_ = compileInfoPtr->coreNum;
        ubSize_ = compileInfoPtr->ubSize;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        coreNum_ = ascendcPlatform.GetCoreNumAiv();
        uint64_t ubSize = 0;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
        ubSize_ = static_cast<int64_t>(ubSize);
    }
    OP_CHECK_IF(coreNum_ <= 0 || ubSize_ <= 0,
                OP_LOGE(context_->GetNodeName(), "coreNum(%ld) and ubSize(%ld) must be positive.", coreNum_, ubSize_),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus UndGenQkvRmsNormRopeCacheTilingBase::GetShapeAttrsInfo()
{
    OP_CHECK_IF(context_ == nullptr, OP_LOGE("UndGenQkvRmsNormRopeCache", "context_ can not be nullptr."),
                return ge::GRAPH_FAILED);

    auto undQkvShapePtr = context_->GetInputShape(UND_QKV_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, undQkvShapePtr);
    auto& undQkvShape = undQkvShapePtr->GetStorageShape();
    OP_CHECK_IF(undQkvShape.GetDimNum() != DIM_NUM_THREE,
                OP_LOGE(context_->GetNodeName(), "und_qkv must be 3D tensor [und_len, N, D]."),
                return ge::GRAPH_FAILED);
    undLen_ = undQkvShape.GetDim(DIM_ZERO);
    numHead_ = undQkvShape.GetDim(DIM_ONE);
    headDim_ = undQkvShape.GetDim(DIM_TWO);

    // gen_qkv 为可选输入：不传时退化为纯 prefill（gen_len = 0）
    auto genQkvShapePtr = context_->GetOptionalInputShape(GEN_QKV_INDEX);
    if (genQkvShapePtr != nullptr) {
        auto& genQkvShape = genQkvShapePtr->GetStorageShape();
        OP_CHECK_IF(genQkvShape.GetDimNum() != DIM_NUM_THREE,
                    OP_LOGE(context_->GetNodeName(), "gen_qkv must be 3D tensor [gen_len, N, D]."),
                    return ge::GRAPH_FAILED);
        genLen_ = genQkvShape.GetDim(DIM_ZERO);
        hasGen_ = genLen_ > 0;
    }
    totalTokens_ = undLen_ + genLen_;

    // cat_indices 为可选输入：不传时 src_t = out_t（单序列恒等映射）
    hasCatIndices_ = context_->GetOptionalInputShape(CAT_INDICES_INDEX) != nullptr;

    auto kCacheShapePtr = context_->GetInputShape(K_CACHE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, kCacheShapePtr);
    auto& kCacheShape = kCacheShapePtr->GetStorageShape();
    OP_CHECK_IF(kCacheShape.GetDimNum() != DIM_NUM_FOUR,
                OP_LOGE(context_->GetNodeName(), "k_cache must be 4D tensor [Bn, Bs, Hk, D]."),
                return ge::GRAPH_FAILED);
    blockNum_ = kCacheShape.GetDim(DIM_ZERO);
    blockSize_ = kCacheShape.GetDim(DIM_ONE);

    auto cosSinShapePtr = context_->GetInputShape(COS_SIN_CACHE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, cosSinShapePtr);
    maxPos_ = cosSinShapePtr->GetStorageShape().GetDim(DIM_ZERO);

    OP_CHECK_IF(CheckDtypeValid() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "CheckDtypeValid failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckAttrsValid() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "CheckAttrsValid failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckUndGenQkvValid() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "CheckUndGenQkvValid failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckWeightsValid() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "CheckWeightsValid failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckCosSinCacheValid() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "CheckCosSinCacheValid failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckKvCacheValid() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "CheckKvCacheValid failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckSlotMappingValid() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "CheckSlotMappingValid failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckPositionsValid() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "CheckPositionsValid failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckCatIndicesValid() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "CheckCatIndicesValid failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckOutputShapeValid() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "CheckOutputShapeValid failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckSupportRange() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "CheckSupportRange failed."), return ge::GRAPH_FAILED);

    reciprocal_ = 1.0f / static_cast<float>(headDim_);
    return ge::GRAPH_SUCCESS;
}

namespace {
struct DtypeSpec {
    int64_t index;
    ge::DataType expect;
    const char* name;
    bool optional;
};

constexpr DtypeSpec INPUT_DTYPE_SPECS[] = {
    {UND_QKV_INDEX, ge::DT_BF16, "und_qkv", false},
    {UND_WEIGHTS_Q_INDEX, ge::DT_BF16, "und_weights_q", false},
    {UND_WEIGHTS_K_INDEX, ge::DT_BF16, "und_weights_k", false},
    {COS_SIN_CACHE_INDEX, ge::DT_FLOAT, "cos_sin_cache", false},
    {K_CACHE_INDEX, ge::DT_BF16, "k_cache", false},
    {V_CACHE_INDEX, ge::DT_BF16, "v_cache", false},
    {SLOT_MAPPING_INDEX, ge::DT_INT64, "slot_mapping", false},
    {POSITIONS_INDEX, ge::DT_INT64, "positions", false},
    {GEN_QKV_INDEX, ge::DT_BF16, "gen_qkv", true},
    {GEN_WEIGHTS_Q_INDEX, ge::DT_BF16, "gen_weights_q", true},
    {GEN_WEIGHTS_K_INDEX, ge::DT_BF16, "gen_weights_k", true},
    {CAT_INDICES_INDEX, ge::DT_INT64, "cat_indices", true},
};

constexpr DtypeSpec OUTPUT_DTYPE_SPECS[] = {
    {Q_OUT_INDEX, ge::DT_BF16, "q", false},
    {K_CACHE_OUT_INDEX, ge::DT_BF16, "k_cache(output)", false},
    {V_CACHE_OUT_INDEX, ge::DT_BF16, "v_cache(output)", false},
};
} // namespace

ge::graphStatus UndGenQkvRmsNormRopeCacheTilingBase::CheckDtypeValid()
{
    for (const auto& spec : INPUT_DTYPE_SPECS) {
        // 可选输入必须按 IR 下标取：它未实例化时后面输入的实例化下标会整体前移，
        // 用 GetInputDesc 会取到隔壁输入的 desc
        auto desc = spec.optional ? context_->GetOptionalInputDesc(spec.index) : context_->GetInputDesc(spec.index);
        if (desc == nullptr) {
            OP_CHECK_IF(!spec.optional,
                        OP_LOGE(context_->GetNodeName(), "required input %s desc is nullptr.", spec.name),
                        return ge::GRAPH_FAILED);
            continue;
        }
        auto dtype = desc->GetDataType();
        OP_CHECK_IF(dtype != spec.expect,
                    OP_LOGE(context_->GetNodeName(), "input %s dtype must be %s, got %s.", spec.name,
                            ge::TypeUtils::DataTypeToSerialString(spec.expect).c_str(),
                            ge::TypeUtils::DataTypeToSerialString(dtype).c_str()),
                    return ge::GRAPH_FAILED);
    }
    for (const auto& spec : OUTPUT_DTYPE_SPECS) {
        auto desc = context_->GetOutputDesc(spec.index);
        OP_CHECK_NULL_WITH_CONTEXT(context_, desc);
        auto dtype = desc->GetDataType();
        OP_CHECK_IF(dtype != spec.expect,
                    OP_LOGE(context_->GetNodeName(), "output %s dtype must be %s, got %s.", spec.name,
                            ge::TypeUtils::DataTypeToSerialString(spec.expect).c_str(),
                            ge::TypeUtils::DataTypeToSerialString(dtype).c_str()),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus UndGenQkvRmsNormRopeCacheTilingBase::CheckOutputShapeValid()
{
    // 输出 buffer 由调用方分配（aclnn 单算子路径上 InferShape 不会替调用方重新开），
    // 这里不校验的话，q 开小一行 kernel 就会按 TilingData 里的 T 写满而静默越界。
    auto qShapePtr = context_->GetOutputShape(Q_OUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, qShapePtr);
    auto& qShape = qShapePtr->GetStorageShape();
    OP_CHECK_IF(qShape.GetDimNum() != DIM_NUM_THREE,
                OP_LOGE(context_->GetNodeName(), "output q must be 3D tensor [T, Hq, D], got %zu dims.",
                        qShape.GetDimNum()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(qShape.GetDim(DIM_ZERO) != totalTokens_ || qShape.GetDim(DIM_ONE) != numHeadQ_ ||
                    qShape.GetDim(DIM_TWO) != headDim_,
                OP_LOGE(context_->GetNodeName(), "output q shape must be [%ld, %ld, %ld], got [%ld, %ld, %ld].",
                        totalTokens_, numHeadQ_, headDim_, qShape.GetDim(DIM_ZERO), qShape.GetDim(DIM_ONE),
                        qShape.GetDim(DIM_TWO)),
                return ge::GRAPH_FAILED);

    // k_cache/v_cache 是原地写入的输入输出，输出 shape 必须与对应输入逐维一致
    const int64_t cachePairs[][DIM_NUM_TWO] = {{K_CACHE_INDEX, K_CACHE_OUT_INDEX},
                                               {V_CACHE_INDEX, V_CACHE_OUT_INDEX}};
    const char* cacheNames[] = {"k_cache", "v_cache"};
    for (size_t i = 0; i < sizeof(cacheNames) / sizeof(cacheNames[0]); ++i) {
        auto inShapePtr = context_->GetInputShape(cachePairs[i][DIM_ZERO]);
        OP_CHECK_NULL_WITH_CONTEXT(context_, inShapePtr);
        auto outShapePtr = context_->GetOutputShape(cachePairs[i][DIM_ONE]);
        OP_CHECK_NULL_WITH_CONTEXT(context_, outShapePtr);
        auto& inShape = inShapePtr->GetStorageShape();
        auto& outShape = outShapePtr->GetStorageShape();
        OP_CHECK_IF(outShape.GetDimNum() != inShape.GetDimNum(),
                    OP_LOGE(context_->GetNodeName(),
                            "output %s dim num must match input (%zu), got %zu.", cacheNames[i], inShape.GetDimNum(),
                            outShape.GetDimNum()),
                    return ge::GRAPH_FAILED);
        for (size_t dim = 0; dim < inShape.GetDimNum(); ++dim) {
            OP_CHECK_IF(outShape.GetDim(dim) != inShape.GetDim(dim),
                        OP_LOGE(context_->GetNodeName(),
                                "output %s dim %zu must match input (%ld), got %ld.", cacheNames[i], dim,
                                inShape.GetDim(dim), outShape.GetDim(dim)),
                        return ge::GRAPH_FAILED);
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus UndGenQkvRmsNormRopeCacheTilingBase::CheckSupportRange()
{
    // 本期支持范围：headDim=128、
    // (Hq, Hk, Hv) ∈ {(8,1,1), (16,2,2)}；需要放宽时改 tiling.h 里的常量即可。
    //
    // block_size 不设限：cache 强制连续 BBND，[Bn, Bs, N, D] 展平即 [Bn*Bs, N, D]，
    // kernel 直接拿 slot 当扁平行号写（kCacheGm_[slot * Hk*D]），Bs 不进任何地址计算，
    // 也不进多核/UB 切分。写出粒度 Hk*D*2B = 256*Hk 恒为 32B 整数倍，对齐由 headDim
    // 与 Hk 保证，同样与 Bs 无关。Bs 的唯一约束是容量，由 CheckKvCacheValid 负责。
    // T 只要求为正：不设人为上限，真实上限由 KV Cache 容量（blockNum * blockSize >= T）决定，
    // 见 CheckKvCacheValid。切分与偏移计算全部 int64_t，对 T 的绝对大小无假设。
    OP_CHECK_IF(totalTokens_ <= 0,
                OP_LOGE(context_->GetNodeName(), "total tokens must be positive, got %ld.", totalTokens_),
                return ge::GRAPH_FAILED);

    // 当前实现只支持 4 个可选输入全部提供的场景：
    // gen_qkv / gen_weights_q / gen_weights_k / cat_indices 缺省的退化路径（纯 prefill、
    // 单序列恒等映射）暂不支持。IR/OpDef 保留 OPTIONAL 声明，后续开启时删掉本段校验即可。
    OP_CHECK_IF(!hasGen_,
                OP_LOGE(context_->GetNodeName(),
                        "gen_qkv is currently required and gen_len must be positive "
                        "(prefill-only path is not supported yet), got gen_len=%ld.", genLen_),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->GetOptionalInputShape(GEN_WEIGHTS_Q_INDEX) == nullptr ||
                    context_->GetOptionalInputShape(GEN_WEIGHTS_K_INDEX) == nullptr,
                OP_LOGE(context_->GetNodeName(),
                        "gen_weights_q/gen_weights_k are currently required."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!hasCatIndices_,
                OP_LOGE(context_->GetNodeName(),
                        "cat_indices is currently required "
                        "(identity-mapping path is not supported yet)."),
                return ge::GRAPH_FAILED);

    bool comboValid = false;
    for (int64_t i = 0; i < SUPPORTED_HEAD_COMBO_NUM; ++i) {
        if (numHeadQ_ == SUPPORTED_HEAD_COMBOS[i][DIM_ZERO] && numHeadK_ == SUPPORTED_HEAD_COMBOS[i][DIM_ONE] &&
            numHeadV_ == SUPPORTED_HEAD_COMBOS[i][DIM_TWO]) {
            comboValid = true;
            break;
        }
    }
    OP_CHECK_IF(!comboValid,
                OP_LOGE(context_->GetNodeName(),
                        "(num_heads_q, num_heads_k, num_heads_v) only supports {(8,1,1), (16,2,2)}, got (%ld,%ld,%ld).",
                        numHeadQ_, numHeadK_, numHeadV_),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus UndGenQkvRmsNormRopeCacheTilingBase::CheckAttrsValid()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);

    const int64_t* numHeadsQ = attrs->GetInt(NUM_HEADS_Q_ATTR_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, numHeadsQ);
    const int64_t* numHeadsK = attrs->GetInt(NUM_HEADS_K_ATTR_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, numHeadsK);
    const int64_t* numHeadsV = attrs->GetInt(NUM_HEADS_V_ATTR_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, numHeadsV);
    numHeadQ_ = *numHeadsQ;
    numHeadK_ = *numHeadsK;
    numHeadV_ = *numHeadsV;
    OP_CHECK_IF(numHeadQ_ <= 0 || numHeadK_ <= 0 || numHeadV_ <= 0,
                OP_LOGE(context_->GetNodeName(), "num_heads_q/k/v must be positive, got (%ld, %ld, %ld).", numHeadQ_,
                        numHeadK_, numHeadV_),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(numHeadQ_ + numHeadK_ + numHeadV_ != numHead_,
                OP_LOGE(context_->GetNodeName(),
                        "num_heads_q + num_heads_k + num_heads_v (%ld) must equal und_qkv N dim (%ld).",
                        numHeadQ_ + numHeadK_ + numHeadV_, numHead_),
                return ge::GRAPH_FAILED);

    const float* normEps = attrs->GetFloat(NORM_EPS_ATTR_IDX);
    epsilon_ = (normEps == nullptr) ? 1e-6f : *normEps;
    OP_CHECK_IF(epsilon_ <= 0.0f, OP_LOGE(context_->GetNodeName(), "norm_eps must be positive, got %f.", epsilon_),
                return ge::GRAPH_FAILED);

    // mrope_section 为空时退化为 [D/2, 0, 0]，即标准 RoPE（三轴同源）。
    //
    // 这三个数**不是**对 half 的划分：轴映射（kernel 的 BuildMropeGatherIndex，规则抄自
    // 参考实现 _mrope）只读 sec[1]/sec[2]，sec[0] 从不参与计算，T 是"其余全归它"的兜底轴。
    // 所以 [16,16,16] 实际得到 T/H/W = 32/16/16，[64,16,16] 与 [0,16,16] 的轴映射逐位相同。
    // 下面的 sum <= half 因此只是挡手误的粗筛，不是语义要求——参考实现没有这条约束，
    // 它自己的用例就是 sum=48。要放宽得连 tests/assets/golden.py:mrope_axis_map 的同名 assert 一起改。
    int64_t half = headDim_ / DIM_NUM_TWO;
    mropeSection_[DIM_ZERO] = half;
    mropeSection_[DIM_ONE] = 0;
    mropeSection_[DIM_TWO] = 0;
    const gert::ContinuousVector* mropeSection = attrs->GetAttrPointer<gert::ContinuousVector>(MROPE_SECTION_ATTR_IDX);
    if (mropeSection != nullptr && mropeSection->GetSize() > 0) {
        OP_CHECK_IF(mropeSection->GetSize() != static_cast<size_t>(MROPE_AXIS_NUM),
                    OP_LOGE(context_->GetNodeName(), "mrope_section must be empty or have exactly 3 elements, got %zu.",
                            mropeSection->GetSize()),
                    return ge::GRAPH_FAILED);
        const int64_t* sectionData = reinterpret_cast<const int64_t*>(mropeSection->GetData());
        OP_CHECK_NULL_WITH_CONTEXT(context_, sectionData);
        int64_t sectionSum = 0;
        for (int64_t i = 0; i < MROPE_AXIS_NUM; ++i) {
            OP_CHECK_IF(sectionData[i] < 0,
                        OP_LOGE(context_->GetNodeName(), "mrope_section[%ld] must be non-negative, got %ld.", i,
                                sectionData[i]),
                        return ge::GRAPH_FAILED);
            mropeSection_[i] = sectionData[i];
            sectionSum += sectionData[i];
        }
        OP_CHECK_IF(sectionSum > half,
                    OP_LOGE(context_->GetNodeName(), "sum(mrope_section) (%ld) must not exceed headDim/2 (%ld).",
                            sectionSum, half),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus UndGenQkvRmsNormRopeCacheTilingBase::CheckUndGenQkvValid()
{
    OP_CHECK_IF(undLen_ <= 0, OP_LOGE(context_->GetNodeName(), "und_qkv und_len must be positive, got %ld.", undLen_),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(headDim_ != SUPPORTED_HEAD_DIM,
                OP_LOGE(context_->GetNodeName(), "headDim only supports %ld, got %ld.", SUPPORTED_HEAD_DIM, headDim_),
                return ge::GRAPH_FAILED);

    auto genQkvShapePtr = context_->GetOptionalInputShape(GEN_QKV_INDEX);
    if (genQkvShapePtr != nullptr) {
        auto& genQkvShape = genQkvShapePtr->GetStorageShape();
        OP_CHECK_IF(genQkvShape.GetDim(DIM_ONE) != numHead_ || genQkvShape.GetDim(DIM_TWO) != headDim_,
                    OP_LOGE(context_->GetNodeName(),
                            "gen_qkv N/D dims must match und_qkv, got (%ld, %ld) vs (%ld, %ld).",
                            genQkvShape.GetDim(DIM_ONE), genQkvShape.GetDim(DIM_TWO), numHead_, headDim_),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(genQkvShape.GetDim(DIM_ZERO) < 0,
                    OP_LOGE(context_->GetNodeName(), "gen_qkv gen_len must be non-negative."), return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus UndGenQkvRmsNormRopeCacheTilingBase::CheckWeightsValid()
{
    const int64_t requiredWeightIdx[] = {UND_WEIGHTS_Q_INDEX, UND_WEIGHTS_K_INDEX};
    for (int64_t idx : requiredWeightIdx) {
        auto shapePtr = context_->GetInputShape(idx);
        OP_CHECK_NULL_WITH_CONTEXT(context_, shapePtr);
        auto& shape = shapePtr->GetStorageShape();
        OP_CHECK_IF(shape.GetDimNum() != DIM_NUM_ONE || shape.GetDim(DIM_ZERO) != headDim_,
                    OP_LOGE(context_->GetNodeName(), "weights(input %ld) must be 1D tensor [D=%ld].", idx, headDim_),
                    return ge::GRAPH_FAILED);
    }

    // gen 段权重：有 gen_qkv 时必须成对给出
    const int64_t genWeightIdx[] = {GEN_WEIGHTS_Q_INDEX, GEN_WEIGHTS_K_INDEX};
    for (int64_t idx : genWeightIdx) {
        auto shapePtr = context_->GetOptionalInputShape(idx);
        if (shapePtr == nullptr) {
            OP_CHECK_IF(hasGen_,
                        OP_LOGE(context_->GetNodeName(),
                                "gen_weights_q/gen_weights_k are required when gen_qkv is provided."),
                        return ge::GRAPH_FAILED);
            continue;
        }
        auto& shape = shapePtr->GetStorageShape();
        OP_CHECK_IF(shape.GetDimNum() != DIM_NUM_ONE || shape.GetDim(DIM_ZERO) != headDim_,
                    OP_LOGE(context_->GetNodeName(), "gen weights(input %ld) must be 1D tensor [D=%ld].", idx,
                            headDim_),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus UndGenQkvRmsNormRopeCacheTilingBase::CheckCosSinCacheValid()
{
    auto shapePtr = context_->GetInputShape(COS_SIN_CACHE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, shapePtr);
    auto& shape = shapePtr->GetStorageShape();
    OP_CHECK_IF(shape.GetDimNum() != DIM_NUM_TWO,
                OP_LOGE(context_->GetNodeName(), "cos_sin_cache must be 2D tensor [max_pos, D]."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(shape.GetDim(DIM_ZERO) <= 0,
                OP_LOGE(context_->GetNodeName(), "cos_sin_cache max_pos must be positive."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(shape.GetDim(DIM_ONE) != headDim_,
                OP_LOGE(context_->GetNodeName(), "cos_sin_cache D dimension must be %ld, got %ld.", headDim_,
                        shape.GetDim(DIM_ONE)),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus UndGenQkvRmsNormRopeCacheTilingBase::CheckKvCacheValid()
{
    // k_cache/v_cache 固定为连续 BBND：[Bn, Bs, Hk/Hv, D]
    auto kCacheShapePtr = context_->GetInputShape(K_CACHE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, kCacheShapePtr);
    auto& kCacheShape = kCacheShapePtr->GetStorageShape();
    auto vCacheShapePtr = context_->GetInputShape(V_CACHE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, vCacheShapePtr);
    auto& vCacheShape = vCacheShapePtr->GetStorageShape();

    OP_CHECK_IF(vCacheShape.GetDimNum() != DIM_NUM_FOUR,
                OP_LOGE(context_->GetNodeName(), "v_cache must be 4D tensor [Bn, Bs, Hv, D]."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(blockNum_ <= 0 || blockSize_ <= 0,
                OP_LOGE(context_->GetNodeName(), "k_cache Bn(%ld) and Bs(%ld) must be positive.", blockNum_,
                        blockSize_),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(kCacheShape.GetDim(DIM_TWO) != numHeadK_,
                OP_LOGE(context_->GetNodeName(), "k_cache Hk dimension must be num_heads_k(%ld), got %ld.", numHeadK_,
                        kCacheShape.GetDim(DIM_TWO)),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(kCacheShape.GetDim(DIM_THREE) != headDim_,
                OP_LOGE(context_->GetNodeName(), "k_cache D dimension must be %ld, got %ld.", headDim_,
                        kCacheShape.GetDim(DIM_THREE)),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(vCacheShape.GetDim(DIM_ZERO) != blockNum_ || vCacheShape.GetDim(DIM_ONE) != blockSize_,
                OP_LOGE(context_->GetNodeName(), "v_cache Bn/Bs must match k_cache (%ld, %ld).", blockNum_, blockSize_),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(vCacheShape.GetDim(DIM_TWO) != numHeadV_,
                OP_LOGE(context_->GetNodeName(), "v_cache Hv dimension must be num_heads_v(%ld), got %ld.", numHeadV_,
                        vCacheShape.GetDim(DIM_TWO)),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(vCacheShape.GetDim(DIM_THREE) != headDim_,
                OP_LOGE(context_->GetNodeName(), "v_cache D dimension must be %ld, got %ld.", headDim_,
                        vCacheShape.GetDim(DIM_THREE)),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(blockNum_ * blockSize_ < totalTokens_,
                OP_LOGE(context_->GetNodeName(), "KV Cache capacity Bn*Bs(%ld) is smaller than total tokens(%ld).",
                        blockNum_ * blockSize_, totalTokens_),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus UndGenQkvRmsNormRopeCacheTilingBase::CheckSlotMappingValid()
{
    auto shapePtr = context_->GetInputShape(SLOT_MAPPING_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, shapePtr);
    auto& shape = shapePtr->GetStorageShape();
    OP_CHECK_IF(shape.GetDimNum() != DIM_NUM_ONE,
                OP_LOGE(context_->GetNodeName(), "slot_mapping must be 1D tensor [T]."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(shape.GetDim(DIM_ZERO) != totalTokens_,
                OP_LOGE(context_->GetNodeName(), "slot_mapping length must be und_len + gen_len (%ld), got %ld.",
                        totalTokens_, shape.GetDim(DIM_ZERO)),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus UndGenQkvRmsNormRopeCacheTilingBase::CheckPositionsValid()
{
    auto shapePtr = context_->GetInputShape(POSITIONS_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, shapePtr);
    auto& shape = shapePtr->GetStorageShape();
    OP_CHECK_IF(shape.GetDimNum() != DIM_NUM_TWO,
                OP_LOGE(context_->GetNodeName(), "positions must be 2D tensor [3, T]."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(shape.GetDim(DIM_ZERO) != MROPE_AXIS_NUM,
                OP_LOGE(context_->GetNodeName(), "positions first dimension must be %ld, got %ld.", MROPE_AXIS_NUM,
                        shape.GetDim(DIM_ZERO)),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(shape.GetDim(DIM_ONE) != totalTokens_,
                OP_LOGE(context_->GetNodeName(), "positions T dimension must be %ld, got %ld.", totalTokens_,
                        shape.GetDim(DIM_ONE)),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus UndGenQkvRmsNormRopeCacheTilingBase::CheckCatIndicesValid()
{
    auto shapePtr = context_->GetOptionalInputShape(CAT_INDICES_INDEX);
    if (shapePtr == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    auto& shape = shapePtr->GetStorageShape();
    OP_CHECK_IF(shape.GetDimNum() != DIM_NUM_ONE,
                OP_LOGE(context_->GetNodeName(), "cat_indices must be 1D tensor [T]."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(shape.GetDim(DIM_ZERO) != totalTokens_,
                OP_LOGE(context_->GetNodeName(), "cat_indices length must be und_len + gen_len (%ld), got %ld.",
                        totalTokens_, shape.GetDim(DIM_ZERO)),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

uint64_t UndGenQkvRmsNormRopeCacheTilingBase::GetTilingKey() const
{
    return tilingKey_;
}

ge::graphStatus Tiling4UndGenQkvRmsNormRopeCache(gert::TilingContext* context)
{
    OP_LOGD(context, "Tiling4UndGenQkvRmsNormRopeCache running.");
    return Ops::Transformer::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

ge::graphStatus TilingPrepare4UndGenQkvRmsNormRopeCache(gert::TilingParseContext* context)
{
    OP_LOGD(context, "TilingPrepare4UndGenQkvRmsNormRopeCache running.");
    auto compileInfo = context->GetCompiledInfo<UndGenQkvRmsNormRopeCacheCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(compileInfo->coreNum <= 0, OP_LOGE(context, "coreNum must be greater than 0."),
                return ge::GRAPH_FAILED);
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    compileInfo->ubSize = static_cast<int64_t>(ubSize);
    OP_CHECK_IF(compileInfo->ubSize <= 0, OP_LOGE(context, "ubSize must be greater than 0."), return ge::GRAPH_FAILED);
    OP_LOGD(context, "coreNum: %ld, ubSize: %ld", compileInfo->coreNum, compileInfo->ubSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(UndGenQkvRmsNormRopeCache)
    .Tiling(Tiling4UndGenQkvRmsNormRopeCache)
    .TilingParse<UndGenQkvRmsNormRopeCacheCompileInfo>(TilingPrepare4UndGenQkvRmsNormRopeCache);

} // namespace optiling
