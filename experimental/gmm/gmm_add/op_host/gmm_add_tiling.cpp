/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gmm_add_tiling.h"
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include <unordered_map>
#include "../../common/tiling_utils.h"

using namespace ge;
using namespace AscendC;

namespace optiling {
namespace {
// some indexs
constexpr uint32_t X_INDEX = 0;
constexpr uint32_t WEIGHT_INDEX = 1;
constexpr uint32_t GROUP_LIST_INDEX = 2;
constexpr uint32_t WEIGHT_GRAD_INDEX = 3;
constexpr uint32_t TRANS_A_ATTR_INDEX = 0;
constexpr uint32_t AICORE_NUM_ATTR_INDEX = 1;

// some best knows
constexpr int64_t BEST_L1_PARTA = 256 * 1024;
constexpr int64_t BEST_L1_PARTB = 128 * 1024;
constexpr int64_t BEST_BASEN = 256;
constexpr uint64_t DOUBLE_BUFFER_L0A_L0B = 2;
constexpr uint64_t DOUBLE_BUFFER_STEPKA_STEPKB = 2;
// some tools
constexpr uint32_t FP32_DATATYPE_SIZE = 4;
constexpr uint32_t SYS_WORKSPACE_SIZE = 16 * 1024 * 1024;
constexpr uint32_t ASCENDC_MAMTUL_MAX_STRIDE = 65535;
constexpr uint32_t AIV_AIC_RATIO = 2;
// tiling keys
constexpr uint64_t TILING_KEY_FP16_NONTRANSPOSE_C_32 = 0;
constexpr uint64_t TILING_KEY_FP16_NONTRANSPOSE_C_16 = 1;
constexpr uint64_t TILING_KEY_FP16_TRANSPOSE_C_32 = 2;
constexpr uint64_t TILING_KEY_FP16_TRANSPOSE_C_16 = 3;
constexpr uint64_t TILING_KEY_BF16_NONTRANSPOSE_C_32 = 4;
constexpr uint64_t TILING_KEY_BF16_NONTRANSPOSE_C_16 = 5;
constexpr uint64_t TILING_KEY_BF16_TRANSPOSE_C_32 = 6;
constexpr uint64_t TILING_KEY_BF16_TRANSPOSE_C_16 = 7;

// for tuning:
constexpr int32_t SINGLE_M = 256;
constexpr int32_t DEFAULT_SINGLE_N = 256;
class TilingTuner {
public:
    static const TilingTuner &Instance()
    {
        static const TilingTuner instance;
        return instance;
    }

    int32_t GetBestSingleN(int32_t numExpert, int32_t M, int32_t N) const
    {
        if (numExpert <= GROUP_NUM_THRESHOLD) {
            return DEFAULT_SINGLE_N;
        }
        auto it = singleNMap_.find(MakeKey(M, N));
        return it != singleNMap_.end() ? it->second : DEFAULT_SINGLE_N;
    }

private:
    static constexpr int32_t GROUP_NUM_THRESHOLD = 20;

    static uint64_t MakeKey(int32_t M, int32_t N)
    {
        return (static_cast<uint64_t>(M) << 32) | static_cast<uint64_t>(static_cast<uint32_t>(N));
    }

    TilingTuner()
        : singleNMap_({
              // 26B - gmoe
              {MakeKey(4096, 1536), 768},
              {MakeKey(2048, 1536), 512},
              // 其他配置...
          })
    {}

    TilingTuner(const TilingTuner &) = delete;
    TilingTuner &operator=(const TilingTuner &) = delete;

    const std::unordered_map<uint64_t, int32_t> singleNMap_;
};

uint64_t GmmAddGetSizePlatForm(const platform_ascendc::CoreMemType memType,
                               platform_ascendc::PlatformAscendC ascendcPlatform)
{
    uint64_t size;
    ascendcPlatform.GetCoreMemSize(memType, size);
    return size;
}

struct PlatFormMemSize {
    uint64_t ubSize;
    uint64_t l1Size;
    uint64_t l0CSize;
    uint64_t l0ASize;
    uint64_t l0BSize;

    explicit PlatFormMemSize(platform_ascendc::PlatformAscendC ascendcPlatform)
        : ubSize(GmmAddGetSizePlatForm(platform_ascendc::CoreMemType::UB, ascendcPlatform)),
          l1Size(GmmAddGetSizePlatForm(platform_ascendc::CoreMemType::L1, ascendcPlatform)),
          l0CSize(GmmAddGetSizePlatForm(platform_ascendc::CoreMemType::L0_C, ascendcPlatform)),
          l0ASize(GmmAddGetSizePlatForm(platform_ascendc::CoreMemType::L0_A, ascendcPlatform)),
          l0BSize(GmmAddGetSizePlatForm(platform_ascendc::CoreMemType::L0_B, ascendcPlatform))
    {}
};
} // namespace
class GmmAddTiling {
public:
    GmmAddTilingData tilingData;
    ge::graphStatus Init(const gert::TilingContext *context);
    ge::graphStatus RunFusionKernelTiling(gert::TilingContext *context);

protected:
    ge::graphStatus CalMMTiling(const gert::TilingContext *context, PlatFormMemSize platFormMemSize);
    ge::graphStatus GMMSetMMTiling(const gert::TilingContext *context, uint64_t l1Size, uint64_t l0CSize,
                                   matmul_tiling::DataType matmulDtype, matmul_tiling::DataType TensorCType);

private:
    int32_t TotalK;
    int32_t baseM_;
    int32_t baseN_;
    int32_t baseK_;
    int32_t bestSingleN_;

    bool trans_a;
    int32_t inputAicoreNum_;

    ge::DataType mmDType;
    ge::DataType weightDtype;
    ge::DataType weightGradDtype;
    uint32_t mmDataTypeSize;
};

ge::graphStatus GmmAddTiling::Init(const gert::TilingContext *context)
{
    const gert::Shape &xShape = context->GetInputShape(X_INDEX)->GetStorageShape();
    const gert::Shape &wShape = context->GetInputShape(WEIGHT_INDEX)->GetStorageShape();
    const gert::Shape &group_list_shape = context->GetInputShape(GROUP_LIST_INDEX)->GetStorageShape();
    const gert::Shape &wGradShape = context->GetInputShape(WEIGHT_GRAD_INDEX)->GetStorageShape();

    trans_a = *(context->GetAttrs()->GetAttrPointer<bool>(TRANS_A_ATTR_INDEX));
    inputAicoreNum_ = *(context->GetAttrs()->GetAttrPointer<int>(AICORE_NUM_ATTR_INDEX));
    tilingData.gmmBaseParams.set_trans_a(trans_a);

    if (xShape.GetDimNum() != 2) {
        OP_LOGE(context->GetNodeName(), "dim of x is not 2");
        return ge::GRAPH_FAILED;
    }
    if (wShape.GetDimNum() != 2) {
        OP_LOGE(context->GetNodeName(), "dim of weight is not 2");
        return ge::GRAPH_FAILED;
    }
    if (group_list_shape.GetDimNum() != 1) {
        OP_LOGE(context->GetNodeName(), "dim of group_list is not 1");
        return ge::GRAPH_FAILED;
    }
    if (wGradShape.GetDimNum() != 3) {
        OP_LOGE(context->GetNodeName(), "dim of weightGrad is not 3");
        return ge::GRAPH_FAILED;
    }

    // check shape
    // [K, M].t() x [K, N] + [E, M, N] = [E, M, N]
    // tensor_a x tensor_b + tensor_c = tensor_d
    auto M = trans_a ? xShape.GetDim(1) : xShape.GetDim(0);
    auto Ka = trans_a ? xShape.GetDim(0) : xShape.GetDim(1);
    TotalK = wShape.GetDim(0);
    auto N = wShape.GetDim(1);
    auto Mc = wGradShape.GetDim(1);
    auto Nc = wGradShape.GetDim(2);
    if (Ka != TotalK) {
        OP_LOGE(context->GetNodeName(), "Ka from tensor_a != Kb from tensor_b.");
        return ge::GRAPH_FAILED;
    }
    if (M != Mc) {
        OP_LOGE(context->GetNodeName(), "M from tensor_a != Mc from tensor_c.");
        return ge::GRAPH_FAILED;
    }
    if (N != Nc) {
        OP_LOGE(context->GetNodeName(), "N from tensor_a != Nc from tensor_c.");
        return ge::GRAPH_FAILED;
    }
    auto groupNum = group_list_shape.GetDim(0);
    if (groupNum != wGradShape.GetDim(0)) {
        OP_LOGE(context->GetNodeName(), "group num mismatch between group_list and weightGrad");
        return ge::GRAPH_FAILED;
    }

    // check stride
    // from ascendc mamtul api: the stride should be in range [1, 65535] by default.
    auto strideA = xShape.GetDim(1);
    if (strideA > ASCENDC_MAMTUL_MAX_STRIDE) {
        OP_LOGE(context->GetNodeName(), "strideA is %d, exceeds max stride value limit: %d.", strideA,
                ASCENDC_MAMTUL_MAX_STRIDE);
        return ge::GRAPH_FAILED;
    }
    if (N > ASCENDC_MAMTUL_MAX_STRIDE) {
        OP_LOGE(context->GetNodeName(), "strideB is %d, exceeds max stride value limit: %d.", N,
                ASCENDC_MAMTUL_MAX_STRIDE);
        return ge::GRAPH_FAILED;
    }

    // check dtype
    mmDType = context->GetInputDesc(X_INDEX)->GetDataType();                   // save x dtype
    weightDtype = context->GetInputDesc(WEIGHT_INDEX)->GetDataType();          // save weight dtype
    weightGradDtype = context->GetInputDesc(WEIGHT_GRAD_INDEX)->GetDataType(); // save weightGrad dtype
    if (mmDType != weightDtype) {
        OP_LOGE(context->GetNodeName(), "dtype mismatch between x and weight");
        return ge::GRAPH_FAILED;
    }

    mmDataTypeSize = GetSizeByDataType(mmDType);

    // 确定最优切块配置
    bestSingleN_ = TilingTuner::Instance().GetBestSingleN(groupNum, M, N);

    tilingData.gmmBaseParams.set_groupNum(groupNum);
    tilingData.gmmBaseParams.set_M(M);
    tilingData.gmmBaseParams.set_N(N);
    tilingData.gmmBaseParams.set_TotalK(TotalK);
    tilingData.gmmBaseParams.set_singleM(SINGLE_M);
    tilingData.gmmBaseParams.set_singleN(bestSingleN_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GmmAddTiling::RunFusionKernelTiling(gert::TilingContext *context)
{
    auto platformInfo = context->GetPlatformInfo();
    platform_ascendc::PlatformAscendC ascendcPlatform(context->GetPlatformInfo());
    static const uint32_t coreNum = ascendcPlatform.GetCoreNumAiv();
    static const uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    static const uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    static const PlatFormMemSize platFormMemSize(ascendcPlatform);

    if (coreNum == 0 || platFormMemSize.ubSize == 0 || platFormMemSize.l1Size == 0 || platFormMemSize.l0CSize == 0 ||
        platFormMemSize.l0ASize == 0 || platFormMemSize.l0BSize == 0) {
        OP_LOGE(context->GetNodeName(),
                "platform info is invalid, coreNum=%u, ubSize=%lu, l1Size=%lu, l0CSize=%lu, l0ASize=%lu, l0BSize=%lu",
                coreNum, platFormMemSize.ubSize, platFormMemSize.l1Size, platFormMemSize.l0CSize,
                platFormMemSize.l0ASize, platFormMemSize.l0BSize);
        return ge::GRAPH_FAILED;
    }

    // handle aicore number
    int validAicNum = aicNum;
    if (inputAicoreNum_ > 0 && inputAicoreNum_ < aicNum) {
        validAicNum = inputAicoreNum_;
    }
    context->SetBlockDim(validAicNum);
    tilingData.gmmBaseParams.set_usedCoreNum(validAicNum * AIV_AIC_RATIO);

    if (CalMMTiling(context, platFormMemSize) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "GMM CalMMTiling failed");
        return ge::GRAPH_FAILED;
    }
    if (GMMSetMMTiling(context, platFormMemSize.l1Size, platFormMemSize.l0CSize,
                       static_cast<matmul_tiling::DataType>(mmDType),
                       static_cast<matmul_tiling::DataType>(weightGradDtype)) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "GMM GMMSetMMTiling failed");
        return ge::GRAPH_FAILED;
    }
    // cube tiling的used core num还是设成aicNum
    tilingData.mmTilingData.set_usedCoreNum(validAicNum);
    // set workspsces
    size_t *workspaces = context->GetWorkspaceSizes(1);
    // 用户需要48*128*256*4=6MB的空间，每个core 128KB
    // TensorC如果使用bf16类型，不需要这么多workspace，目前这里先不动，申请多一点没什么问题，总量并不大
    workspaces[0] = SYS_WORKSPACE_SIZE + aivNum * SINGLE_M * bestSingleN_ * sizeof(float);
    // set tilingkey
    if (weightGradDtype == ge::DT_FLOAT) {
        if (mmDType == ge::DT_FLOAT16 && trans_a == false) {
            context->SetTilingKey(TILING_KEY_FP16_NONTRANSPOSE_C_32);
        } else if (mmDType == ge::DT_FLOAT16 && trans_a == true) {
            context->SetTilingKey(TILING_KEY_FP16_TRANSPOSE_C_32);
        } else if (mmDType == ge::DT_BF16 && trans_a == false) {
            context->SetTilingKey(TILING_KEY_BF16_NONTRANSPOSE_C_32);
        } else if (mmDType == ge::DT_BF16 && trans_a == true) {
            context->SetTilingKey(TILING_KEY_BF16_TRANSPOSE_C_32);
        }
    } else {
        if (mmDType == ge::DT_FLOAT16 && trans_a == false) {
            context->SetTilingKey(TILING_KEY_FP16_NONTRANSPOSE_C_16);
        } else if (mmDType == ge::DT_FLOAT16 && trans_a == true) {
            context->SetTilingKey(TILING_KEY_FP16_TRANSPOSE_C_16);
        } else if (mmDType == ge::DT_BF16 && trans_a == false) {
            context->SetTilingKey(TILING_KEY_BF16_NONTRANSPOSE_C_16);
        } else if (mmDType == ge::DT_BF16 && trans_a == true) {
            context->SetTilingKey(TILING_KEY_BF16_TRANSPOSE_C_16);
        }
    }

    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GmmAddTiling::GMMSetMMTiling(const gert::TilingContext *context, uint64_t l1Size, uint64_t l0CSize,
                                             matmul_tiling::DataType matmulDtype, matmul_tiling::DataType TensorCType)
{
    matmul_tiling::MatmulApiTiling mm;
    mm.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmulDtype, trans_a);
    mm.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmulDtype, false);
    // c type需要使用fp32
    mm.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, TensorCType);

    mm.SetOrgShape(tilingData.gmmBaseParams.get_M(), tilingData.gmmBaseParams.get_N(), TotalK);
    // @lg: single K is the k of each expert, not TotalK
    mm.SetShape(SINGLE_M, bestSingleN_, TotalK);
    mm.SetFixSplit(baseM_, baseN_, baseK_);
    mm.SetBufferSpace(l1Size, l0CSize, -1);
    if (mm.GetTiling(tilingData.mmTilingData) == -1) {
        OP_LOGE(context->GetNodeName(), "matmul getTiling failed.");
        return ge::GRAPH_FAILED;
    }

    uint32_t mmStepKa = (BEST_L1_PARTB >> 1) / (baseM_ * baseK_ * mmDataTypeSize);
    uint32_t mmStepKb = (BEST_L1_PARTA >> 1) / (baseN_ * baseK_ * mmDataTypeSize);

    if (mmStepKa > mmStepKb) {
        mmStepKa = mmStepKa / mmStepKb * mmStepKb;
    } else if (mmStepKa < mmStepKb) {
        mmStepKb = mmStepKb / mmStepKa * mmStepKa;
    }
    uint32_t stepM = 1; // 1: stepM set fixed value 1
    uint32_t stepN = 1; // 1: stepN set fixed value 1
    uint32_t mmDepthA1 = mmStepKa * DOUBLE_BUFFER_STEPKA_STEPKB * stepM;
    uint32_t mmDepthB1 = mmStepKb * DOUBLE_BUFFER_STEPKA_STEPKB * stepN;
    tilingData.mmTilingData.set_shareMode(0);
    tilingData.mmTilingData.set_shareL1Size(l1Size);
    tilingData.mmTilingData.set_shareL0CSize(l0CSize);
    tilingData.mmTilingData.set_dbL0C(1);
    tilingData.mmTilingData.set_baseM(baseM_);
    tilingData.mmTilingData.set_baseN(baseN_);
    tilingData.mmTilingData.set_baseK(baseK_);
    tilingData.mmTilingData.set_stepKa(mmStepKa);
    tilingData.mmTilingData.set_depthA1(mmDepthA1);
    tilingData.mmTilingData.set_stepKb(mmStepKb);
    tilingData.mmTilingData.set_depthB1(mmDepthB1);

    tilingData.mmTilingData.set_stepM(stepM);
    tilingData.mmTilingData.set_stepN(stepN);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GmmAddTiling::CalMMTiling(const gert::TilingContext *context, PlatFormMemSize platFormMemSize)
{
    baseN_ = BEST_BASEN;
    // 基于使能double buffer的L0B内存计算baseK
    baseK_ = (platFormMemSize.l0BSize / DOUBLE_BUFFER_L0A_L0B) / (BEST_BASEN * mmDataTypeSize);
    baseK_ = SixteenAlign(baseK_);
    if (baseK_ == 0) {
        OP_LOGE(context->GetNodeName(), "baseK_ canot be 0.");
        return ge::GRAPH_FAILED;
    }
    // 基于使能double buffer的L0A内存和L0C内存计算baseM(cube)
    uint32_t maxBaseM = platFormMemSize.l0CSize / (BEST_BASEN * FP32_DATATYPE_SIZE);
    baseM_ =
        std::min<uint32_t>((platFormMemSize.l0ASize / DOUBLE_BUFFER_L0A_L0B) / (baseK_ * mmDataTypeSize), maxBaseM);
    baseM_ = SixteenAlign(baseM_);
    if (baseM_ == 0) {
        OP_LOGE(context->GetNodeName(), "baseM_ canot be 0.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingGmmAdd(gert::TilingContext *context)
{
    GmmAddTiling GMM_tiling;
    if (GMM_tiling.Init(context) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "GMM tiling init failed.");
        return ge::GRAPH_FAILED;
    }
    return GMM_tiling.RunFusionKernelTiling(context);
}

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    return TilingGmmAdd(context);
}

IMPL_OP_OPTILING(GmmAdd).Tiling(TilingFunc);
} // namespace optiling
