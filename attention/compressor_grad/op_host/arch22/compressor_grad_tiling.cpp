/* *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file compressor_grad_tiling.cpp
 * \brief CompressorGrad arch22 (A3 / ascend910_93) tiling.
 *
 * Validates with the arch35 Check* series plus the A3 implementation's
 * constraints (nblk/nblkIn checks), then computes the schedule:
 * cast/scatter core clamps, segLen 64->32->16 UB downgrade, shape-adaptive
 * gDx split, and the five-block fp16 workspace layout.
 *
 * TH layout (x [T,H] + cu_seqlens, A5 对齐): host 只做 shape 派生
 * (B = cu_seqlens.dim0-1, T = x.dim0, nblkIn_ = d_cmp_kv 行数 = 有效块上界)，
 * 每 batch 实际块数/块路由由 kernel 运行时读 cu_seqlens/seqused/start_pos
 * GM 值决定（与 arch35 PyPTO/A5 及 forward arch22 的 GetValue 模式一致）。
 * BSH 布局的 seqused/start_pos 同为 kernel 端读值（值语义，tiling 只置
 * flags + 安全上界，见 SetSeqStartFlags）。
 */

#include <functional>
#include <algorithm>
#include <cmath>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "register/op_def_registry.h"
#include "compressor_grad_tiling_arch22.h"

using namespace ge;
using namespace AscendC;
namespace optiling {
namespace {

template <typename T>
std::string to_string(const T& value)
{
    if (std::is_same_v<T, bool>) {
        return value ? "true" : "false";
    } else {
        return std::to_string(value);
    }
}

template <typename T>
void LogErrorNumberSupport(const std::vector<T>& expectNumberList, const T& actualValue, const std::string& name,
                           const std::string subName, const char* opName)
{
    std::ostringstream oss;
    for (size_t i = 0; i < expectNumberList.size(); ++i) {
        oss << to_string(expectNumberList[i]);
        if (i < expectNumberList.size() - 1) {
            oss << ", ";
        }
    }

    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, name, to_string(actualValue),
                                          subName + " only supports " + oss.str());
}

template <typename T>
ge::graphStatus CheckFeatureValueSupport(const T* featureValue, const std::vector<T>& expectFeatureValList,
                                         const std::string& name, const char* opName)
{
    if (std::find(expectFeatureValList.begin(), expectFeatureValList.end(), *featureValue) ==
        expectFeatureValList.end()) {
        LogErrorNumberSupport(expectFeatureValList, *featureValue, name, "feature value", opName);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

template <typename T>
ge::graphStatus CheckAttrValueSupportInterval(const T* attrValue, const uint32_t minVal, const uint32_t maxVal,
                                              const std::string& name, const char* opName)
{
    if (attrValue == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    std::string attr_value = "attr value";
    if (*attrValue < minVal || *attrValue > maxVal) {
        std::ostringstream oss;
        oss << "[" << minVal << ", " << maxVal << "]";
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, name, std::to_string(*attrValue),
                                              "attr value only supports value in range " + oss.str());
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

template <typename T>
ge::graphStatus CheckAttrValueSupportList(const T* attrValue, const std::vector<T>& expectAttrValList,
                                          const std::string& name, const char* opName)
{
    if (attrValue == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    if (std::find(expectAttrValList.begin(), expectAttrValList.end(), *attrValue) == expectAttrValList.end()) {
        LogErrorNumberSupport(expectAttrValList, *attrValue, name, "attr value", opName);
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

static std::string DataTypeToSerialString(ge::DataType type)
{
    const auto it = DATATYPE_TO_STRING_MAP.find(type);
    if (it != DATATYPE_TO_STRING_MAP.end()) {
        return it->second;
    } else {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON("CompressorGrad", "datatype", std::to_string(static_cast<int32_t>(type)),
                                              "not support");
        return "UNDEFINED";
    }
}

void LogErrorDtypeSupport(const std::vector<ge::DataType>& expectDtypeList, const ge::DataType& actualDtype,
                          const std::string& name, const char* opName)
{
    std::ostringstream oss;
    for (size_t i = 0; i < expectDtypeList.size(); ++i) {
        oss << DataTypeToSerialString(expectDtypeList[i]);
        if (i < expectDtypeList.size() - 1) {
            oss << ", ";
        }
    }
    OP_LOGE_FOR_INVALID_DTYPE(opName, name, DataTypeToSerialString(actualDtype), oss.str());
}

ge::graphStatus LogErrorShapeConsistency(const std::string& name, const gert::StorageShape* shape,
                                         const uint32_t& dimNum, const std::string& subName, const uint32_t& expectNum,
                                         const char* opName)
{
    if (shape == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    const uint32_t actualNum = shape->GetStorageShape().GetDim(dimNum);
    OP_CHECK_IF(actualNum != expectNum,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    opName, name, "dim " + std::to_string(dimNum) + "=" + std::to_string(actualNum),
                    "should be equal to " + subName + ": " + std::to_string(expectNum)),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

struct ShapeCheckItem {
    const std::string& tensorName; // *_NAME 常量（静态存储期，引用安全）
    const gert::StorageShape* shape;
    uint32_t dimIdx;
    const char* expectName;
    uint32_t expectValue;
};

ge::graphStatus CheckShapeList(const char* opName, const ShapeCheckItem* items, size_t count)
{
    for (size_t i = 0; i < count; i++) {
        if (LogErrorShapeConsistency(items[i].tensorName, items[i].shape, items[i].dimIdx, items[i].expectName,
                                     items[i].expectValue, opName) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

// ── 辅助函数：检查单个 tensor 的 shape 和 desc 均不为空 ──
ge::graphStatus CheckTensorShapeAndDesc(const char* opName, const std::string& tensorName,
                                        const gert::CompileTimeTensorDesc* desc, const gert::StorageShape* shape)
{
    OP_CHECK_IF(shape == nullptr, OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName, tensorName, "shape is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(desc == nullptr, OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName, tensorName, "desc is nullptr"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

struct TensorCheckItem {
    const std::string& name; // *_NAME 常量（静态存储期，引用安全）
    const gert::CompileTimeTensorDesc* desc;
    const gert::StorageShape* shape;
};

ge::graphStatus CheckTensorList(const char* opName, const TensorCheckItem* items, size_t count)
{
    for (size_t i = 0; i < count; i++) {
        if (CheckTensorShapeAndDesc(opName, items[i].name, items[i].desc, items[i].shape) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace

ge::graphStatus CompressorGradTilingA3::ConvertRequiredParams(gert::TilingContext& context,
                                                              CompressorGradContext& compressorGradContext)
{
    compressorGradContext.x.desc = context.GetRequiredInputDesc(TOKEN_X_INPUT_INDEX);
    compressorGradContext.x.shape = context.GetRequiredInputShape(TOKEN_X_INPUT_INDEX);
    OP_CHECK_IF(compressorGradContext.x.shape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(compressorGradContext.opName, X_NAME, "shape is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(compressorGradContext.x.desc == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(compressorGradContext.opName, X_NAME, "desc is nullptr"),
                return ge::GRAPH_FAILED);
    compressorGradContext.wkv.desc = context.GetRequiredInputDesc(WEIGHT_KV_INPUT_INDEX);
    compressorGradContext.wkv.shape = context.GetRequiredInputShape(WEIGHT_KV_INPUT_INDEX);
    compressorGradContext.wgate.desc = context.GetRequiredInputDesc(WEIGHT_WGATE_INPUT_INDEX);
    compressorGradContext.wgate.shape = context.GetRequiredInputShape(WEIGHT_WGATE_INPUT_INDEX);
    compressorGradContext.dCmpKv.desc = context.GetRequiredInputDesc(D_CMP_KV_INPUT_INDEX);
    compressorGradContext.dCmpKv.shape = context.GetRequiredInputShape(D_CMP_KV_INPUT_INDEX);
    compressorGradContext.softmaxScore.desc = context.GetRequiredInputDesc(SOFTMAX_SCORE_INPUT_INDEX);
    compressorGradContext.softmaxScore.shape = context.GetRequiredInputShape(SOFTMAX_SCORE_INPUT_INDEX);
    compressorGradContext.kv.desc = context.GetRequiredInputDesc(KV_INPUT_INDEX);
    compressorGradContext.kv.shape = context.GetRequiredInputShape(KV_INPUT_INDEX);

    compressorGradContext.dX.desc = context.GetOutputDesc(D_X_OUTPUT_INDEX);
    compressorGradContext.dX.shape = context.GetOutputShape(D_X_OUTPUT_INDEX);
    compressorGradContext.dWkv.desc = context.GetOutputDesc(D_WKV_OUTPUT_INDEX);
    compressorGradContext.dWkv.shape = context.GetOutputShape(D_WKV_OUTPUT_INDEX);
    compressorGradContext.dWgate.desc = context.GetOutputDesc(D_WGATE_OUTPUT_INDEX);
    compressorGradContext.dWgate.shape = context.GetOutputShape(D_WGATE_OUTPUT_INDEX);
    compressorGradContext.dApe.desc = context.GetOutputDesc(D_APE_OUTPUT_INDEX);
    compressorGradContext.dApe.shape = context.GetOutputShape(D_APE_OUTPUT_INDEX);

    compressorGradContext.dtype = compressorGradContext.x.desc->GetDataType();
    auto xDimNum = compressorGradContext.x.shape->GetStorageShape().GetDimNum();
    if (xDimNum == COMPRESSOR_GRAD_DIM_NUM_3) {
        compressorGradContext.layout = LayoutType::LAYOUT_BSH;
    } else if (xDimNum == COMPRESSOR_GRAD_DIM_NUM_2) {
        compressorGradContext.layout = LayoutType::LAYOUT_TH;
    } else {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(compressorGradContext.opName, X_NAME, std::to_string(xDimNum),
                                                 "x dimension should be 2 or 3");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

void CompressorGradTilingA3::ConvertOptionalParams(gert::TilingContext& context,
                                                   CompressorGradContext& compressorGradContext)
{
    compressorGradContext.cuSeqlens.desc = context.GetOptionalInputDesc(CU_SEQ_LEN_INPUT_INDEX);
    compressorGradContext.cuSeqlens.shape = context.GetOptionalInputShape(CU_SEQ_LEN_INPUT_INDEX);
    compressorGradContext.seqUsed.desc = context.GetOptionalInputDesc(SEQ_USED_INPUT_INDEX);
    compressorGradContext.seqUsed.shape = context.GetOptionalInputShape(SEQ_USED_INPUT_INDEX);
    compressorGradContext.startPos.desc = context.GetOptionalInputDesc(START_POS_INPUT_INDEX);
    compressorGradContext.startPos.shape = context.GetOptionalInputShape(START_POS_INPUT_INDEX);
}

ge::graphStatus CompressorGradTilingA3::ConvertContext(gert::TilingContext& context,
                                                       CompressorGradContext& compressorGradContext)
{
    if (context.GetNodeName() == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("CompressorGrad", "opName", "got from TilingContext is nullptr");
        return ge::GRAPH_FAILED;
    }

    compressorGradContext.opName = context.GetNodeName();
    compressorGradContext.opType = context.GetNodeType();
    compressorGradContext.platformInfo = context.GetPlatformInfo();
    OP_CHECK_IF(ConvertRequiredParams(context, compressorGradContext) != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
    ConvertOptionalParams(context, compressorGradContext);

    auto attrs = context.GetAttrs();
    OP_CHECK_IF(attrs == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context.GetNodeName(), "attrs", "got from ge is nullptr"),
                return ge::GRAPH_FAILED);
    compressorGradContext.coff = attrs->GetAttrPointer<uint32_t>(COFF_ATTR_INDEX);
    compressorGradContext.cmpRatio = attrs->GetAttrPointer<uint32_t>(CMP_RATIO_ATTR_INDEX);

    OP_CHECK_IF(
        context.GetWorkspaceSizes(1) == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context.GetNodeName(), "workSpaceSize", "got from ge is nullptr"),
        return ge::GRAPH_FAILED);
    compressorGradContext.workSpaces = context.GetWorkspaceSizes(1);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingA3::GetNpuInfo()
{
    OP_CHECK_IF(context_->platformInfo == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->opName, "platformInfo", "is nullptr"),
                return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->platformInfo);
    socVersion_ = ascendcPlatform.GetSocVersion();

    libapiSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();

    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize_);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1Size_);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0cSize_);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, l0bSize_);

    aivNum_ = ascendcPlatform.GetCoreNumAiv();
    aicNum_ = ascendcPlatform.GetCoreNumAic();
    OP_CHECK_IF(
        aicNum_ == 0 || aivNum_ == 0,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->opName, "aicNum/aivNum", "num of core obtained is 0"),
        return GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingA3::SetBaseInfo()
{
    if (context_->layout == LayoutType::LAYOUT_TH) {
        // TH: x [T,H]，batch 边界由 cu_seqlens 前缀和定义（cu 值只能 kernel 运行时读
        // GM，host 只做 shape 派生：B = cu_seqlens.dim0-1，S 无固定值置 0）
        baseParams_.batchSize = context_->cuSeqlens.shape->GetStorageShape().GetDim(COMPRESSOR_GRAD_DIM_INDEX_0) - 1;
        baseParams_.seqSize = 0;
        baseParams_.tokenSize = context_->x.shape->GetStorageShape().GetDim(COMPRESSOR_GRAD_DIM_INDEX_0);
        baseParams_.hiddenSize = context_->x.shape->GetStorageShape().GetDim(COMPRESSOR_GRAD_DIM_INDEX_1);
    } else {
        baseParams_.batchSize = context_->x.shape->GetStorageShape().GetDim(COMPRESSOR_GRAD_DIM_INDEX_0);
        baseParams_.seqSize = context_->x.shape->GetStorageShape().GetDim(COMPRESSOR_GRAD_DIM_INDEX_1);
        baseParams_.hiddenSize = context_->x.shape->GetStorageShape().GetDim(COMPRESSOR_GRAD_DIM_INDEX_2);
        baseParams_.tokenSize = baseParams_.batchSize * baseParams_.seqSize;
    }

    coff_ = context_->coff == nullptr ? COFF_VALUE : static_cast<uint8_t>(*context_->coff);
    baseParams_.headDim = context_->wkv.shape->GetStorageShape().GetDim(COMPRESSOR_GRAD_DIM_INDEX_0) / coff_;
    baseParams_.featureDim = context_->wkv.shape->GetStorageShape().GetDim(COMPRESSOR_GRAD_DIM_INDEX_0);
    baseParams_.cmpRatio = static_cast<uint32_t>(*context_->cmpRatio);
    baseParams_.usedCoreNum = aicNum_;

    // A3 派生量
    tPad_ = (static_cast<int64_t>(baseParams_.tokenSize) + CG_A3_NZ_TILE - 1) / CG_A3_NZ_TILE * CG_A3_NZ_TILE;
    if (context_->layout == LayoutType::LAYOUT_TH) {
        // TH: 每 batch 有效块数依赖 cu_seqlens/seqused/start_pos 值（kernel 读 GM
        // 运行时判定，A5 语义），host 无法静态算 nblk；nblkIn_ 取 d_cmp_kv 行数
        // （有效块数上界 min(T, T/cr+B)，与正向 outputRows 一致），仅作 scatter
        // 分核上界（实际总块数 ≤ nblkIn_，kernel 端走查自然收敛）
        nblk_ = 0;
        nblkIn_ = context_->dCmpKv.shape->GetStorageShape().GetDim(COMPRESSOR_GRAD_DIM_INDEX_0);
    } else {
        // BSH: 先按 shape 推导（无值语义输入时的最终值）；seqused/start_pos 提供
        // 时由 SetSeqStartFlags 置 kernel-only 安全上界（值由设备端读）
        nblk_ = baseParams_.seqSize / baseParams_.cmpRatio; // complete blocks only
        nblkIn_ = context_->dCmpKv.shape->GetStorageShape().GetDim(COMPRESSOR_GRAD_DIM_INDEX_1);
    }
    // fp16 uses HiLo dual-plane dn workspaces (dn = hi + lo); bf16 is
    // single-plane (official A5 strategy).
    dnPlanes_ = (context_->dtype == ge::DT_FLOAT16) ? 2 : 1;

    OP_LOGI(context_->opName, "[TILING][arch22] layout:%s bSize:%u sSize:%u tSize:%u cmpRatio:%u coff:%u",
            context_->layout == LayoutType::LAYOUT_TH ? "TH" : "BSH", baseParams_.batchSize, baseParams_.seqSize,
            baseParams_.tokenSize, baseParams_.cmpRatio, coff_);

    return ge::GRAPH_SUCCESS;
}

// ── seqused / start_pos 值语义 —— kernel-only 设计 ──────────────────────
// tiling 不读张量值：按"张量是否提供"置 seqUsedValid_/startPosValid_ flags，
// nblk_ 取覆盖合法值的安全上界；实际块数 nblkUsed_b=(sp_b+sq_b)/cr 与块锚点
// 由 scatter 在设备端按 GM 值逐 batch 复算。
ge::graphStatus CompressorGradTilingA3::SetSeqStartFlags()
{
    if (context_->layout == LayoutType::LAYOUT_TH) {
        // TH: seqused/start_pos 由 kernel 走查元数据读取（ScatterRangeTh，指针判空），
        // tiling 不置 BSH 专用 flags
        return ge::GRAPH_SUCCESS;
    }
    seqUsedValid_ = (context_->seqUsed.desc != nullptr) ? 1 : 0;
    startPosValid_ = (context_->startPos.desc != nullptr) ? 1 : 0;
    if (!seqUsedValid_ && !startPosValid_) {
        return ge::GRAPH_SUCCESS; // 默认路径：保持 shape 推导
    }
    // 安全上界：任何 sq<=S 都有 (sp+sq)/cr <= S/cr + 1
    nblk_ = std::min(static_cast<int64_t>(baseParams_.seqSize) / baseParams_.cmpRatio + 1, nblkIn_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingA3::CalcSchedule()
{
    const int64_t tLen = baseParams_.tokenSize;
    const int64_t hSize = baseParams_.hiddenSize;
    const int64_t cSize = coff_ * baseParams_.headDim;
    // TH: 静态总块数未知（cu/seqused 值 kernel 运行时读），用 d_cmp_kv 行数
    // （有效块上界）做 scatter 分核；BSH: B * nblk
    const int64_t nbTotal =
        (context_->layout == LayoutType::LAYOUT_TH) ? nblkIn_ : static_cast<int64_t>(baseParams_.batchSize) * nblk_;
    const int64_t nRows = static_cast<int64_t>(coff_) * baseParams_.cmpRatio;

    // ---- cast 核数（A3 host: min(aivNum, max(1, T_pad + 2*C))）----
    int64_t totalRows = tPad_ + 2 * cSize;
    castCoreNum_ = std::min<int64_t>(aivNum_, std::max<int64_t>(1, totalRows));

    // ---- scatter 核数与每核块数 ----
    // nbTotal == 0（S < cr 或全部 batch 无完整块）合法——scatter 不处理任何块，
    // dn/d_ape 已预清零，gemm 输出天然全零（A5 _zero_outputs 语义）。核数下限 1
    // 只是避免除零；blocksPerCore_ = 0 使 kernel 块循环体一次都不进。
    scatterCoreNum_ = std::min<int64_t>(aivNum_, std::max<int64_t>(nbTotal, 1));
    blocksPerCore_ = (nbTotal + scatterCoreNum_ - 1) / scatterCoreNum_;

    // ---- SEG: UB budget 降档 64 -> 32 -> 16（arch2201 TBuf ceiling 184KB）----
    //   bytes(S) = (2*cr + 3*N + 5)*S*4 + (planes-1)*cr*S*4 + S*ioT(2B)
    //              + slots*planes*cr*S*2
    const int64_t planes = dnPlanes_;
    const int64_t slots = (coff_ == 2) ? 2 : 1;
    const int64_t cr = baseParams_.cmpRatio;
    //   TH 额外驻留 cu_seqlens/seqused/start_pos 三个 int32 数组（3*(B+1)*4B，
    //   scatter 逐 batch 元数据走查用，MTE2 一次搬入 UB）
    // Aligned meta size: cu(sqOff)+seqused(spOff-sqOff)+startPos, each 32B-aligned
    const int64_t bSz = static_cast<int64_t>(baseParams_.batchSize);
    const int64_t sqOff = (bSz + 1 + 7) & ~7;
    const int64_t spOff = sqOff + ((bSz + 7) & ~7);
    const int64_t metaBytes = (context_->layout == LayoutType::LAYOUT_TH) ? (spOff + bSz) * 4 : 0;
    auto segBytes = [&](int64_t s) {
        return (2 * cr + 3 * nRows + 5) * s * 4 + (planes - 1) * cr * s * 4 + s * 2 + slots * planes * cr * s * 2 +
               metaBytes;
    };
    segLen_ = 64;
    if (segBytes(64) > static_cast<int64_t>(CG_A3_SCATTER_UB_LIMIT)) {
        segLen_ = 32;
    }
    if (segBytes(segLen_) > static_cast<int64_t>(CG_A3_SCATTER_UB_LIMIT)) {
        segLen_ = 16;
    }
    // final UB guardrail: reject shapes that overflow even after the downgrade
    // (headDim ∈ {128, 512}，CheckFeature 已拦——无 <64 的降档分支)
    OP_CHECK_IF(
        segBytes(16) > static_cast<int64_t>(CG_A3_SCATTER_UB_LIMIT),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->opName, CMP_RATIO_NAME + "/head_dim",
                                              std::to_string(cr) + "/" + std::to_string(baseParams_.headDim),
                                              "scatter UB budget exceeded even at segLen=16; shape not supported"),
        return ge::GRAPH_FAILED);

    // ---- gemm dx/dw 核数划分（离散 makespan 的 argmin 搜索）----
    //   u_dx = a*nCn + b (us/unit, 128M x 2*128N pair)
    //   u_dw = (xResident ? c_l2 : c_hbm) * kT (us/unit, 128x128), kT = T_pad/128
    //   t(g) = max(ceil(nDxU/g)*u_dx, ceil(nDwU/(cores-g))*u_dw), g in [6,16]
    //   并列时取连续平衡点 g_cont = cores*DxT/(DxT+DwT)
    const int64_t gemmCores = aicNum_;
    int64_t gDx = std::max<int64_t>(1, gemmCores / 2);
    {
        const int64_t nM = (tLen + CG_A3_NZ_TILE - 1) / CG_A3_NZ_TILE;
        const int64_t nH = hSize / CG_A3_NZ_TILE;  // H % 128 == 0 (checked)
        const int64_t nCn = cSize / CG_A3_NZ_TILE; // C % 128 == 0
        const int64_t nPairRow = (nH + 1) / 2;
        const int64_t nDxU = nM * nPairRow; // dx units: 128M x 2*128N pairs
        const int64_t nDwU = nCn * nH;      // dw units: 128x128 tiles
        const bool fp16Dtype = (context_->dtype == ge::DT_FLOAT16);
        const double uDx = (fp16Dtype ? 2.68 : 1.63) * static_cast<double>(nCn) + (fp16Dtype ? 0.76 : 0.78);
        const bool xResident = (tPad_ * hSize * 2) <= static_cast<int64_t>(128) * 1024 * 1024;
        const double uDw = (fp16Dtype ? (xResident ? 1.48 : 1.47) : (xResident ? 0.62 : 0.78)) *
                           static_cast<double>(tPad_ / CG_A3_NZ_TILE);
        const double dxT = static_cast<double>(nDxU) * uDx;
        const double dwT = static_cast<double>(nDwU) * uDw;
        const double gCont = static_cast<double>(gemmCores) * dxT / (dxT + dwT);
        int64_t bestG = -1;
        double bestCost = 0.0;
        double bestTie = 0.0;
        for (int64_t g = 6; g <= 16 && g < gemmCores; ++g) {
            const double tG = std::max(std::ceil(static_cast<double>(nDxU) / static_cast<double>(g)) * uDx,
                                       std::ceil(static_cast<double>(nDwU) / static_cast<double>(gemmCores - g)) * uDw);
            const double tie = std::fabs(static_cast<double>(g) - gCont);
            if (bestG < 0 || tG < bestCost - 1e-9 || (tG <= bestCost + 1e-9 && tie < bestTie)) {
                bestG = g;
                bestCost = tG;
                bestTie = tie;
            }
        }
        if (bestG > 0) {
            bestG = std::min(bestG, std::max<int64_t>(1, nM * nPairRow));
            bestG = std::min<int64_t>(std::max<int64_t>(bestG, 1), gemmCores);
            gDx = bestG;
        }
    }
    gemmDxCores_ = gDx;

    OP_LOGI(context_->opName, "[TILING][arch22] castCores:%ld scatterCores:%ld blocksPerCore:%ld segLen:%ld gDx:%ld",
            castCoreNum_, scatterCoreNum_, blocksPerCore_, segLen_, gemmDxCores_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingA3::CalcWorkSpace()
{
    // dn workspaces 为 blocked-NZ：tile (rt,ct) 于 C-strip 主序 ct*(T_pad/128)+rt；
    // 每个 128x128 tile 单元内交错 [hi 16384 | lo 16384]（fp16 双平面；bf16 仅 hi）。
    // 总大小 dnPlanes * T_pad * C。
    // One flat fp16 workspace, five sub-blocks: xF16 | wkvF16 | wgateF16 |
    // dnKv | dnSc. T_pad/H/C are all 128-aligned, so every block keeps 32KB
    // alignment.
    constexpr int64_t f16Size = 2; // fp16 element bytes
    const int64_t hSize = baseParams_.hiddenSize;
    const int64_t cSize = coff_ * baseParams_.headDim;
    const int64_t xBytes = tPad_ * hSize * f16Size;
    const int64_t wBytes = cSize * hSize * f16Size;
    const int64_t dnBytes = dnPlanes_ * tPad_ * cSize * f16Size;

    xF16WsOff_ = 0;
    wkvF16WsOff_ = xF16WsOff_ + xBytes;
    wgateF16WsOff_ = wkvF16WsOff_ + wBytes;
    dnKvWsOff_ = wgateF16WsOff_ + wBytes;
    dnScWsOff_ = dnKvWsOff_ + dnBytes;
    // d_ape per-core partials (fp32): one [cid][cr][C] slot per scatter core,
    // offset rounded up to the 512B alignment kept by the blocks above.
    apePartElems_ = scatterCoreNum_ * baseParams_.cmpRatio * cSize;
    constexpr int64_t wsAlign = 512;
    apePartWsOff_ = (dnScWsOff_ + dnBytes + wsAlign - 1) / wsAlign * wsAlign;
    const int64_t userWsBytes = apePartWsOff_ + apePartElems_ * static_cast<int64_t>(sizeof(float));

    workspaceSize_ = libapiSize_ + static_cast<size_t>(userWsBytes);
    if (context_->workSpaces) {
        context_->workSpaces[0] = workspaceSize_;
    }

    OP_LOGI(context_->opName, "[TILING][arch22] workspaceSize = %zu (x=%ld w=2x%ld dn=2x%ld planes=%ld apePart=%ld)",
            workspaceSize_, xBytes, wBytes, dnBytes, dnPlanes_, apePartElems_ * static_cast<int64_t>(sizeof(float)));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingA3::SetTilingData()
{
    tilingData_->layout = (context_->layout == LayoutType::LAYOUT_TH) ? 1 : 0;
    tilingData_->batchSize = baseParams_.batchSize;
    tilingData_->seqSize = baseParams_.seqSize;
    tilingData_->tLen = baseParams_.tokenSize;
    tilingData_->tPad = tPad_;
    tilingData_->hiddenSize = baseParams_.hiddenSize;
    tilingData_->headDim = baseParams_.headDim;
    tilingData_->totalHeadDim = coff_ * baseParams_.headDim;
    tilingData_->cmpRatio = baseParams_.cmpRatio;
    tilingData_->coff = coff_;
    tilingData_->cmpRowCnt = coff_ * baseParams_.cmpRatio;
    tilingData_->nblk = nblk_;
    tilingData_->nblkIn = nblkIn_;
    tilingData_->seqUsedValid = seqUsedValid_;
    tilingData_->startPosValid = startPosValid_;
    tilingData_->dnPlanes = dnPlanes_;
    tilingData_->castCoreNum = castCoreNum_;
    tilingData_->scatterCoreNum = scatterCoreNum_;
    tilingData_->blocksPerCore = blocksPerCore_;
    tilingData_->segLen = segLen_;
    tilingData_->gemmDxCores = gemmDxCores_;
    tilingData_->xF16WsOff = xF16WsOff_;
    tilingData_->wkvF16WsOff = wkvF16WsOff_;
    tilingData_->wgateF16WsOff = wgateF16WsOff_;
    tilingData_->dnKvWsOff = dnKvWsOff_;
    tilingData_->dnScWsOff = dnScWsOff_;
    tilingData_->apePartWsOff = apePartWsOff_;
    tilingData_->apePartElems = apePartElems_;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingA3::GenTilingKey() const
{
    // 0:BF16, 1:FP16（与 arch35 编码一致）
    uint8_t dtype = 0;
    auto xDtype = context_->x.desc->GetDataType();
    if (xDtype == ge::DT_BF16) {
        dtype = 0;
    } else if (xDtype == ge::DT_FLOAT16) {
        dtype = 1;
    }
    // arch22 模板空间只有 D_TYPE 一个字段（ASCENDC_TPL_UINT_DECL(D_TYPE, 2bit, LIST, 0, 1)）
    context_->tilingKey = GET_TPL_TILING_KEY(dtype);

    OP_LOGI(context_->opName, "CompressorGrad[arch22] dtype:%hhu tilingKey:%lu", dtype, context_->tilingKey);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingA3::CheckDimNumInLayoutSupport(const std::string& layout,
                                                                   const gert::StorageShape* shape,
                                                                   const std::string& name) const
{
    const auto& dimIt = LAYOUT_DIM_MAP.find(layout);
    OP_CHECK_IF(shape->GetStorageShape().GetDimNum() != dimIt->second,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                    context_->opName, name, std::to_string(shape->GetStorageShape().GetDimNum()),
                    "when layout is " + layout + ", dimension should be " + std::to_string(dimIt->second)),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingA3::CheckDtypeSupport(const gert::CompileTimeTensorDesc* desc,
                                                          const std::string& name) const
{
    if (desc != nullptr) {
        const auto& it = DTYPE_SUPPORT_MAP.find(name);
        OP_CHECK_IF(it == DTYPE_SUPPORT_MAP.end(),
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                        context_->opName, name, "datatype support list should be specify in DTYPE_SUPPORT_MAP"),
                    return ge::GRAPH_FAILED);
        auto& expectDtypeList = it->second;
        OP_CHECK_IF(
            std::find(expectDtypeList.begin(), expectDtypeList.end(), desc->GetDataType()) == expectDtypeList.end(),
            LogErrorDtypeSupport(expectDtypeList, desc->GetDataType(), name, context_->opName),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingA3::CheckDimNumSupport(const gert::StorageShape* shape,
                                                           const std::string& name) const
{
    if (shape == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    const auto& it = DIM_NUM_MAP.find(name);
    OP_CHECK_IF(it == DIM_NUM_MAP.end(),
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->opName, name,
                                                         "dim number support list should be specify in DIM_NUM_MAP"),
                return ge::GRAPH_FAILED);
    auto& expectDimNumList = it->second;
    OP_CHECK_IF(std::find(expectDimNumList.begin(), expectDimNumList.end(), shape->GetStorageShape().GetDimNum()) ==
                    expectDimNumList.end(),
                LogErrorNumberSupport(expectDimNumList, static_cast<uint32_t>(shape->GetStorageShape().GetDimNum()),
                                      name, "dimension", context_->opName),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingA3::CheckSingleParaX() const
{
    const std::string layout = (context_->layout == LayoutType::LAYOUT_TH) ? "TH" : "BSH";
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->x.desc, X_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->x.shape, X_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumInLayoutSupport(layout, context_->x.shape, X_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingA3::CheckSingleParaWkv() const
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->wkv.desc, WKV_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->wkv.shape, WKV_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingA3::CheckSingleParaWgate() const
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->wgate.desc, WGATE_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->wgate.shape, WGATE_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingA3::CheckSingleParaDCmpKv() const
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->dCmpKv.desc, D_CMP_KV_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->dCmpKv.shape, D_CMP_KV_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingA3::CheckSingleParaSoftmaxScore() const
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->softmaxScore.desc, SOFTMAX_SCORE_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->softmaxScore.shape, SOFTMAX_SCORE_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingA3::CheckSingleParaKV() const
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->kv.desc, KV_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->kv.shape, KV_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingA3::CheckSingleParaCuSeqlens() const
{
    if (context_->cuSeqlens.desc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->cuSeqlens.desc, CU_SEQLENS_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->cuSeqlens.shape, CU_SEQLENS_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingA3::CheckSingleParaSeqused() const
{
    if (context_->seqUsed.desc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->seqUsed.desc, SEQUSED_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->seqUsed.shape, SEQUSED_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingA3::CheckSingleParaStartPos() const
{
    if (context_->startPos.desc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->startPos.desc, START_POS_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->startPos.shape, START_POS_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingA3::CheckSingleParaDX() const
{
    if (context_->dX.desc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->dX.desc, D_X_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->dX.shape, D_X_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingA3::CheckSingleParaDWkv() const
{
    if (context_->dWkv.desc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->dWkv.desc, D_WKV_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->dWkv.shape, D_WKV_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingA3::CheckSingleParaDWgate() const
{
    if (context_->dWgate.desc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->dWgate.desc, D_WGATE_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->dWgate.shape, D_WGATE_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingA3::CheckSingleParaDApe() const
{
    if (context_->dApe.desc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(context_->dApe.desc, D_APE_NAME) ||
        ge::GRAPH_SUCCESS != CheckDimNumSupport(context_->dApe.shape, D_APE_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingA3::CheckSingleParaCmpRatio() const
{
    if (ge::GRAPH_SUCCESS != CheckAttrValueSupportInterval(context_->cmpRatio, MIN_CMP_RATIO, MAX_CMP_RATIO,
                                                           CMP_RATIO_NAME, context_->opName)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingA3::CheckSingleParaCoff() const
{
    if (ge::GRAPH_SUCCESS != CheckAttrValueSupportList(context_->coff, COFF, COFF_NAME, context_->opName)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingA3::CheckLayoutSupport() const
{
    // arch22 同时支持 BSH 与 TH（TH 与 A5 对齐：x [T,H] + cu_seqlens，块路由由
    // kernel 运行时读 GM 元数据完成）；TH 必须携带 cu_seqlens（存在性在
    // CheckRequiredInOutExistence 校验）
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingA3::CheckFeature() const
{
    if (ge::GRAPH_SUCCESS != CheckFeatureValueSupport(&baseParams_.headDim, HEAD_DIM, "head_dim", context_->opName)) {
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(
        baseParams_.hiddenSize > MAX_HIDDEN_SIZE || baseParams_.hiddenSize < MIN_HIDDEN_SIZE ||
            baseParams_.hiddenSize % ALIGN_FACTOR_HIDDEN_SIZE != 0,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->opName, "hidden_size", std::to_string(baseParams_.hiddenSize),
                                              "should be within [" + std::to_string(MIN_HIDDEN_SIZE) + ", " +
                                                  std::to_string(MAX_HIDDEN_SIZE) + "] and be 512-aligned"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingA3::CheckShapeConsistency() const
{
    uint8_t coff = coff_;
    auto coffD = baseParams_.headDim * coff;
    const char* opName = context_->opName;
    // 公共部分：cuSeqlens/seqUsed/startPos + wkv/wgate 的 hiddenSize 和 coff*headDim
    const ShapeCheckItem commonChecks[] = {
        {CU_SEQLENS_NAME, context_->cuSeqlens.shape, COMPRESSOR_GRAD_DIM_INDEX_0, "batchSize+1",
         baseParams_.batchSize + 1},
        {SEQUSED_NAME, context_->seqUsed.shape, COMPRESSOR_GRAD_DIM_INDEX_0, "batchSize", baseParams_.batchSize},
        {START_POS_NAME, context_->startPos.shape, COMPRESSOR_GRAD_DIM_INDEX_0, "batchSize", baseParams_.batchSize},
        {WKV_NAME, context_->wkv.shape, COMPRESSOR_GRAD_DIM_INDEX_1, "hidden_size", baseParams_.hiddenSize},
        {WGATE_NAME, context_->wgate.shape, COMPRESSOR_GRAD_DIM_INDEX_1, "hidden_size", baseParams_.hiddenSize},
        {WKV_NAME, context_->wkv.shape, COMPRESSOR_GRAD_DIM_INDEX_0, "coff*head_dim", static_cast<uint32_t>(coffD)},
        {WGATE_NAME, context_->wgate.shape, COMPRESSOR_GRAD_DIM_INDEX_0, "coff*head_dim", static_cast<uint32_t>(coffD)},
    };
    if (CheckShapeList(opName, commonChecks, sizeof(commonChecks) / sizeof(commonChecks[0])) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (context_->layout == LayoutType::LAYOUT_TH) {
        // TH 2D 分支：d_cmp_kv [rows, D]，softmax_score/kv [rows, coff*cr, D]；
        // rows = 有效块数上界（值依赖 cu/seqused，静态只查列维与行数一致）。
        // 维数一致性（dimnum == x/x+1）已在 CheckDimNumConsistency 前置校验，
        // 本分支的 GetDim(1)/GetDim(2) 不会越 rank。
        const ShapeCheckItem checks2D[] = {
            {D_CMP_KV_NAME, context_->dCmpKv.shape, COMPRESSOR_GRAD_DIM_INDEX_1, "head_dim", baseParams_.headDim},
            {SOFTMAX_SCORE_NAME, context_->softmaxScore.shape, COMPRESSOR_GRAD_DIM_INDEX_1, "coff*cmp_ratio",
             coff * baseParams_.cmpRatio},
            {SOFTMAX_SCORE_NAME, context_->softmaxScore.shape, COMPRESSOR_GRAD_DIM_INDEX_2, "head_dim",
             baseParams_.headDim},
            {KV_NAME, context_->kv.shape, COMPRESSOR_GRAD_DIM_INDEX_1, "coff*cmp_ratio", coff * baseParams_.cmpRatio},
            {KV_NAME, context_->kv.shape, COMPRESSOR_GRAD_DIM_INDEX_2, "head_dim", baseParams_.headDim},
            {SOFTMAX_SCORE_NAME, context_->softmaxScore.shape, COMPRESSOR_GRAD_DIM_INDEX_0, "d_cmp_kv rows", nblkIn_},
            {KV_NAME, context_->kv.shape, COMPRESSOR_GRAD_DIM_INDEX_0, "d_cmp_kv rows", nblkIn_},
        };
        if (CheckShapeList(opName, checks2D, sizeof(checks2D) / sizeof(checks2D[0])) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
        return ge::GRAPH_SUCCESS;
    }

    // BSH 3D 分支（维数一致性同样已前置，GetDim(3) 不会越 rank）
    const ShapeCheckItem checks3D[] = {
        {D_CMP_KV_NAME, context_->dCmpKv.shape, COMPRESSOR_GRAD_DIM_INDEX_0, "batchSize", baseParams_.batchSize},
        {D_CMP_KV_NAME, context_->dCmpKv.shape, COMPRESSOR_GRAD_DIM_INDEX_2, "head_dim", baseParams_.headDim},
        {SOFTMAX_SCORE_NAME, context_->softmaxScore.shape, COMPRESSOR_GRAD_DIM_INDEX_0, "batchSize",
         baseParams_.batchSize},
        {SOFTMAX_SCORE_NAME, context_->softmaxScore.shape, COMPRESSOR_GRAD_DIM_INDEX_2, "coff*cmp_ratio",
         coff * baseParams_.cmpRatio},
        {SOFTMAX_SCORE_NAME, context_->softmaxScore.shape, COMPRESSOR_GRAD_DIM_INDEX_3, "head_dim",
         baseParams_.headDim},
        {KV_NAME, context_->kv.shape, COMPRESSOR_GRAD_DIM_INDEX_0, "batchSize", baseParams_.batchSize},
        {KV_NAME, context_->kv.shape, COMPRESSOR_GRAD_DIM_INDEX_2, "coff*cmp_ratio", coff * baseParams_.cmpRatio},
        {KV_NAME, context_->kv.shape, COMPRESSOR_GRAD_DIM_INDEX_3, "head_dim", baseParams_.headDim},
    };
    if (CheckShapeList(opName, checks3D, sizeof(checks3D) / sizeof(checks3D[0])) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingA3::CheckDtypeConsistencyX(const gert::CompileTimeTensorDesc* desc,
                                                               const std::string& name) const
{
    const auto actualDtype = desc->GetDataType();
    OP_CHECK_IF(
        actualDtype != context_->dtype,
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context_->opName, name, DataTypeToSerialString(actualDtype),
                                              "should be same with x: " + DataTypeToSerialString(context_->dtype)),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingA3::CheckDtypeConsistency() const
{
    if (CheckDtypeConsistencyX(context_->wkv.desc, WKV_NAME) != ge::GRAPH_SUCCESS ||
        CheckDtypeConsistencyX(context_->wgate.desc, WGATE_NAME) != ge::GRAPH_SUCCESS ||
        CheckDtypeConsistencyX(context_->dCmpKv.desc, D_CMP_KV_NAME) != ge::GRAPH_SUCCESS ||
        CheckDtypeConsistencyX(context_->dX.desc, D_X_NAME) != ge::GRAPH_SUCCESS ||
        CheckDtypeConsistencyX(context_->dWkv.desc, D_WKV_NAME) != ge::GRAPH_SUCCESS ||
        CheckDtypeConsistencyX(context_->dWgate.desc, D_WGATE_NAME) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingA3::CheckDimNumConsistency() const
{
    auto xDimNum = context_->x.shape->GetStorageShape().GetDimNum();
    OP_CHECK_IF(xDimNum != context_->dX.shape->GetStorageShape().GetDimNum(),
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                    context_->opName, D_X_NAME, std::to_string(context_->dX.shape->GetStorageShape().GetDimNum()),
                    "dim num should be equal to x: " + std::to_string(xDimNum)),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        xDimNum != context_->dCmpKv.shape->GetStorageShape().GetDimNum(),
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context_->opName, D_CMP_KV_NAME,
                                                 std::to_string(context_->dCmpKv.shape->GetStorageShape().GetDimNum()),
                                                 "dim num should be equal to x: " + std::to_string(xDimNum)),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(xDimNum != context_->softmaxScore.shape->GetStorageShape().GetDimNum() - 1,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                    context_->opName, SOFTMAX_SCORE_NAME,
                    std::to_string(context_->softmaxScore.shape->GetStorageShape().GetDimNum()),
                    "dim num should be x dim + 1: " + std::to_string(xDimNum + 1)),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(xDimNum != context_->kv.shape->GetStorageShape().GetDimNum() - 1,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                    context_->opName, KV_NAME, std::to_string(context_->kv.shape->GetStorageShape().GetDimNum()),
                    "dim num should be x dim + 1: " + std::to_string(xDimNum + 1)),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingA3::CheckBlockDimConstrain() const
{
    // D_BASE_SIZE = 128 是 arch22 gemm 的 tile 边（BM/BN/BK）；dx/dw 分核要求
    // AIC 核数 >= D/128（D ∈ {128, 512}，即 >= 1 或 >= 4）
    uint32_t minBlockNum = baseParams_.headDim / D_BASE_SIZE;
    OP_CHECK_IF(aicNum_ < minBlockNum,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->opName, "aicNum", std::to_string(aicNum_),
                                                      "should not be less than " + std::to_string(minBlockNum)),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingA3::CheckA3Constraints() const
{
    if (context_->layout == LayoutType::LAYOUT_TH) {
        // TH：块数由 cu_seqlens/seqused 值决定（kernel 运行时判定），合法 0 块
        // （全 batch 不足一个完整块时 dn 全零、gemm 输出全 0，与 A5/golden 一致）；
        // 静态只要求 d_cmp_kv 行数（有效块上界）>= 1
        OP_CHECK_IF(
            nblkIn_ < 1,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->opName, D_CMP_KV_NAME + " rows", std::to_string(nblkIn_),
                                                  "TH layout requires d_cmp_kv rows >= 1"),
            return ge::GRAPH_FAILED);
        // B = cu_seqlens.numel - 1 >= 1（cu numel >= 2）：B=0 属空 batch，文档声明
        // 不支持（numel==0 已在 CheckEmptyTensor 拦截，这里拦 numel==1）
        OP_CHECK_IF(baseParams_.batchSize < 1,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        context_->opName, CU_SEQLENS_NAME + " numel", std::to_string(baseParams_.batchSize + 1),
                        "TH layout requires cu_seqlens numel == B+1 >= 2 (batch of 0 is not supported)"),
                    return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }
    // A3 实现语义：仅处理完整压缩块，不完整尾部块不落 dn。nblk == 0 合法
    // （S < cmp_ratio 或 seqused 全部不足一块）——dn 工作区预清零 + gemm 全行覆盖，
    // 四输出全零，与 A5 _zero_outputs / golden 空循环一致；tiling/kernel 均已防护
    // （CalcSchedule 的 scatter 核数下限、scatter 块循环直接跳过）。nblk_ 源自
    // 无符号除法/min，恒 >= 0，无需再查负。
    OP_CHECK_IF(
        nblkIn_ < nblk_,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->opName, D_CMP_KV_NAME, "dim1=" + std::to_string(nblkIn_),
                                              "d_cmp_kv blocks must cover complete blocks: " + std::to_string(nblk_)),
        return ge::GRAPH_FAILED);
    const int64_t smBlocks = context_->softmaxScore.shape->GetStorageShape().GetDim(COMPRESSOR_GRAD_DIM_INDEX_1);
    OP_CHECK_IF(smBlocks != nblkIn_,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    context_->opName, SOFTMAX_SCORE_NAME, "dim1=" + std::to_string(smBlocks),
                    "softmax_score block dim should be equal to d_cmp_kv: " + std::to_string(nblkIn_)),
                return ge::GRAPH_FAILED);
    const int64_t kvBlocks = context_->kv.shape->GetStorageShape().GetDim(COMPRESSOR_GRAD_DIM_INDEX_1);
    OP_CHECK_IF(
        kvBlocks != nblkIn_,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->opName, KV_NAME, "dim1=" + std::to_string(kvBlocks),
                                              "kv block dim should be equal to d_cmp_kv: " + std::to_string(nblkIn_)),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingA3::CheckMultiParaConsistency() const
{
    // 维数一致性前置：CheckShapeConsistency 的 TH/BSH 分支按 layout 无条件
    // GetDim(1..3)，只有先确认 d_x/d_cmp_kv 与 x 同维、softmax_score/kv 比 x 多
    // 一维（root arch35 的同款 gating），这些取维才不会越 rank
    if (CheckDimNumConsistency() != ge::GRAPH_SUCCESS || CheckShapeConsistency() != ge::GRAPH_SUCCESS ||
        CheckDtypeConsistency() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingA3::CheckSinglePara() const
{
    if (ge::GRAPH_SUCCESS != CheckSingleParaX() || ge::GRAPH_SUCCESS != CheckSingleParaWkv() ||
        ge::GRAPH_SUCCESS != CheckSingleParaWgate() || ge::GRAPH_SUCCESS != CheckSingleParaDCmpKv() ||
        ge::GRAPH_SUCCESS != CheckSingleParaSoftmaxScore() || ge::GRAPH_SUCCESS != CheckSingleParaKV() ||
        ge::GRAPH_SUCCESS != CheckSingleParaCuSeqlens() || ge::GRAPH_SUCCESS != CheckSingleParaSeqused() ||
        ge::GRAPH_SUCCESS != CheckSingleParaStartPos() || ge::GRAPH_SUCCESS != CheckSingleParaDX() ||
        ge::GRAPH_SUCCESS != CheckSingleParaDWkv() || ge::GRAPH_SUCCESS != CheckSingleParaDWgate() ||
        ge::GRAPH_SUCCESS != CheckSingleParaDApe() || ge::GRAPH_SUCCESS != CheckSingleParaCmpRatio() ||
        ge::GRAPH_SUCCESS != CheckSingleParaCoff()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingA3::CheckRequiredInOutExistence() const
{
    const TensorCheckItem requiredTensors[] = {
        {X_NAME, context_->x.desc, context_->x.shape},
        {WKV_NAME, context_->wkv.desc, context_->wkv.shape},
        {WGATE_NAME, context_->wgate.desc, context_->wgate.shape},
        {D_CMP_KV_NAME, context_->dCmpKv.desc, context_->dCmpKv.shape},
        {SOFTMAX_SCORE_NAME, context_->softmaxScore.desc, context_->softmaxScore.shape},
        {KV_NAME, context_->kv.desc, context_->kv.shape},
        {D_X_NAME, context_->dX.desc, context_->dX.shape},
        {D_WKV_NAME, context_->dWkv.desc, context_->dWkv.shape},
        {D_WGATE_NAME, context_->dWgate.desc, context_->dWgate.shape},
        {D_APE_NAME, context_->dApe.desc, context_->dApe.shape},
    };
    if (CheckTensorList(context_->opName, requiredTensors, sizeof(requiredTensors) / sizeof(requiredTensors[0])) !=
        ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    // cuSeqlens 特殊校验（与 A5 一致）：TH 布局必须非空，BSH 布局必须为空
    if (context_->layout == LayoutType::LAYOUT_TH) {
        OP_CHECK_IF(context_->cuSeqlens.desc == nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->opName, CU_SEQLENS_NAME,
                                                             "in TH layout, should not be nullptr"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(context_->cuSeqlens.shape == nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->opName, CU_SEQLENS_NAME,
                                                             "in TH layout, should not be nullptr"),
                    return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(context_->cuSeqlens.desc != nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->opName, CU_SEQLENS_NAME,
                                                             "in BSH layout, must be nullptr"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(context_->cuSeqlens.shape != nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->opName, CU_SEQLENS_NAME,
                                                             "in BSH layout, must be nullptr"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingA3::CheckRequiredAttrExistence() const
{
    OP_CHECK_IF(context_->cmpRatio == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context_->opName, "cmp_ratio", "attr is nullptr"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingA3::CheckRequiredParaExistence() const
{
    if (CheckRequiredInOutExistence() != ge::GRAPH_SUCCESS || CheckRequiredAttrExistence() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingA3::CheckEmptyTensor() const
{
    // CompressorGrad 不支持空 tensor：所有输入/输出 shapeSize 必须 > 0，空则直接拦截；
    // 报错按首个命中的张量报名并给出其真实 shapeSize
    const std::pair<const std::string, const gert::StorageShape*> checkList[] = {
        {X_NAME, context_->x.shape},
        {WKV_NAME, context_->wkv.shape},
        {WGATE_NAME, context_->wgate.shape},
        {D_CMP_KV_NAME, context_->dCmpKv.shape},
        {SOFTMAX_SCORE_NAME, context_->softmaxScore.shape},
        {KV_NAME, context_->kv.shape},
        {D_X_NAME, context_->dX.shape},
        {D_WKV_NAME, context_->dWkv.shape},
        {D_WGATE_NAME, context_->dWgate.shape},
        {D_APE_NAME, context_->dApe.shape},
    };
    for (const auto& item : checkList) {
        const int64_t shapeSize = item.second->GetStorageShape().GetShapeSize();
        OP_CHECK_IF(shapeSize == 0,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        context_->opName, item.first, std::to_string(shapeSize),
                        "CompressorGrad does not support empty tensor: all inputs/outputs shapeSize must be > 0"),
                    return ge::GRAPH_FAILED);
    }
    // cu_seqlens（TH 必传，BSH 必空——存在性已由 CheckRequiredInOutExistence 拦截）：
    // numel==0 会使 batchSize = dim0-1 无符号下溢，结构性拒绝
    if (context_->cuSeqlens.shape != nullptr && context_->cuSeqlens.shape->GetStorageShape().GetShapeSize() == 0) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->opName, CU_SEQLENS_NAME, "0",
                                              "cu_seqlens must not be empty in TH layout: numel must be B+1 >= 2");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CompressorGradTilingA3::RunBigKernelTiling(CompressorGradTilingDataA3* tilingData)
{
    tilingData_ = tilingData;
    using StatusFunction = std::function<ge::graphStatus()>;
    std::vector<StatusFunction> tilingFuncs{std::bind(&CompressorGradTilingA3::GetNpuInfo, this),
                                            std::bind(&CompressorGradTilingA3::CheckRequiredParaExistence, this),
                                            std::bind(&CompressorGradTilingA3::CheckEmptyTensor, this),
                                            std::bind(&CompressorGradTilingA3::CheckLayoutSupport, this),
                                            std::bind(&CompressorGradTilingA3::CheckSinglePara, this),
                                            std::bind(&CompressorGradTilingA3::SetBaseInfo, this),
                                            std::bind(&CompressorGradTilingA3::SetSeqStartFlags, this),
                                            std::bind(&CompressorGradTilingA3::CheckFeature, this),
                                            std::bind(&CompressorGradTilingA3::CheckMultiParaConsistency, this),
                                            std::bind(&CompressorGradTilingA3::CheckBlockDimConstrain, this),
                                            std::bind(&CompressorGradTilingA3::CheckA3Constraints, this),
                                            std::bind(&CompressorGradTilingA3::CalcSchedule, this),
                                            std::bind(&CompressorGradTilingA3::CalcWorkSpace, this),
                                            std::bind(&CompressorGradTilingA3::SetTilingData, this),
                                            std::bind(&CompressorGradTilingA3::GenTilingKey, this)};
    for (const auto& func : tilingFuncs) {
        if (func() != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }

    // 单入口 MIX_AIC_1_2 kernel：blockDim 计 AIC 核数（AIV 自动 x2）
    context_->blockDim = aicNum_;

    OP_LOGI(context_->opName, "Run big kernel (arch22)");

    return ge::GRAPH_SUCCESS;
}

CMP_GRAD_EXTERN_C ge::graphStatus TilingCompressorGradArch22(gert::TilingContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("CompressorGrad", "context", "is nullptr"),
                return ge::GRAPH_FAILED);

    OP_LOGI("Getting Tiling (arch22)");

    CompressorGradContext compressorGradContext{};
    if (CompressorGradTilingA3::ConvertContext(*context, compressorGradContext) != ge::GRAPH_SUCCESS) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            context->GetNodeName(), "context",
            "error occurred while converting tilingContext to CompressorGrad context");
        return ge::GRAPH_FAILED;
    }

    CompressorGradTilingDataA3* tilingData = context->GetTilingData<CompressorGradTilingDataA3>();
    OP_CHECK_IF(tilingData == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(compressorGradContext.opName, "tilingData", "is nullptr"),
                return ge::GRAPH_FAILED);
    // 使用SyncAll，需要设置为batchmode模式，所有核同时启动，否则多流方式下执行可能会卡死
    context->SetScheduleMode(BATCH_MODE_SCHEDULE);

    CompressorGradTilingA3 compressorGradTiling(&compressorGradContext);
    if (compressorGradTiling.RunBigKernelTiling(tilingData) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    context->SetTilingKey(compressorGradContext.tilingKey);
    context->SetBlockDim(compressorGradContext.blockDim);
    OP_LOGI(compressorGradContext.opName, "block dim: %u.", compressorGradContext.blockDim);
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
