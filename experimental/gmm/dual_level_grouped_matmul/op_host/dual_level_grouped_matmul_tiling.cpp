// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// Licensed under CANN Open Software License Agreement Version 2.0.
#include "dual_level_grouped_matmul_tiling.h"
#include <algorithm>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <string>
#include "platform/platform_ascendc.h"
#include "log/log.h"
#include "util/math_util.h"

namespace optiling {
namespace {
using namespace DualLevelGroupedMatmul;
using Ops::Base::CeilDiv;
constexpr int64_t MX_SCALE_K = MX_SCALE_PAIR * DEFAULT_LEVEL1_GROUP_SIZE;
constexpr int64_t NZ_ROWS = 16;
constexpr int64_t NZ_COLS = 64;
constexpr uint64_t FP4_ELEMENTS_PER_BYTE = 2;
constexpr int64_t MIN_K = 2;
constexpr int64_t SPARSE_GROUP_LIST = 2;
constexpr int64_t GROUP_LIST_PAIR_SIZE = 2;
constexpr uint64_t NZ_BLOCK_BYTES = NZ_ROWS * NZ_COLS / FP4_ELEMENTS_PER_BYTE;
constexpr uint64_t KIBIBYTE = 1024;
constexpr uint64_t MIN_L1_BYTES = 512 * KIBIBYTE;
constexpr uint64_t MIN_L0_OPERAND_BYTES = 64 * KIBIBYTE;
constexpr uint64_t MIN_L0C_BYTES = 256 * KIBIBYTE;
constexpr uint64_t AIV_COUNT = 2;

void ReportInvalid(const gert::TilingContext* ctx, const char* parameter, const std::string& actual,
                   const std::string& reason)
{
    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ctx->GetNodeName(), parameter, actual.c_str(), reason.c_str());
}

// Per-AIV UB layout: partial slots, tiled accumulators, scale A window, scale B and bias.
// Match the L0 block geometry and UB layout of the dual-level BlockMmad and Epilogue.
uint64_t CalcEpilogueUbBytes(uint64_t m, uint64_t n)
{
    constexpr uint64_t L0_BASE_M = 128;
    constexpr uint64_t L0_BASE_N = 128;
    constexpr uint64_t PARTIAL_BUFFER_COUNT = 2;
    constexpr uint64_t SCALE_B_AND_BIAS_BUFFERS = 2;
    constexpr uint64_t SCALE0_WINDOW = 32;
    constexpr uint64_t PARTIAL_BYTES = L0_BASE_M / AIV_COUNT * L0_BASE_N * sizeof(float);
    const uint64_t mTiles = CeilDiv(m, L0_BASE_M);
    const uint64_t nTiles = CeilDiv(n, L0_BASE_N);
    const uint64_t accumulatorBytes = mTiles * nTiles * PARTIAL_BYTES;
    const uint64_t scaleABytes = mTiles * (L0_BASE_M / AIV_COUNT) * SCALE0_WINDOW * sizeof(float);
    return PARTIAL_BUFFER_COUNT * PARTIAL_BYTES + accumulatorBytes + scaleABytes +
           n * sizeof(float) * SCALE_B_AND_BIAS_BUFFERS;
}

std::string ShapeString(const gert::Shape& shape)
{
    std::string out = "(";
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        if (i != 0) {
            out += ",";
        }
        out += std::to_string(shape.GetDim(i));
    }
    return out + ")";
}

std::string DimsString(std::initializer_list<int64_t> dims)
{
    std::string out = "(";
    for (auto dim : dims) {
        if (out.size() != 1) {
            out += ",";
        }
        out += std::to_string(dim);
    }
    return out + ")";
}

bool Match(const gert::Shape& s, std::initializer_list<int64_t> dims)
{
    if (s.GetDimNum() != dims.size()) {
        return false;
    }
    size_t i = 0;
    for (const auto d : dims) {
        if (s.GetDim(i++) != d) {
            return false;
        }
    }
    return true;
}

bool NdInput(gert::TilingContext* ctx, size_t i, ge::DataType dtype, std::initializer_list<int64_t> dims)
{
    const auto* desc = ctx->GetInputDesc(i);
    const auto* shape = ctx->GetInputShape(i);
    const bool valid = desc != nullptr && shape != nullptr && desc->GetDataType() == dtype &&
                       desc->GetStorageFormat() == ge::FORMAT_ND && Match(shape->GetOriginShape(), dims) &&
                       Match(shape->GetStorageShape(), dims);
    if (!valid && desc != nullptr && shape != nullptr) {
        ReportInvalid(ctx, "input",
                      "index=" + std::to_string(i) + ", dtype=" + std::to_string(desc->GetDataType()) +
                          ", format=" + std::to_string(desc->GetStorageFormat()) + ", origin=" +
                          ShapeString(shape->GetOriginShape()) + ", storage=" + ShapeString(shape->GetStorageShape()),
                      "expected ND, dtype=" + std::to_string(dtype) + ", shape=" + DimsString(dims));
    }
    if (desc == nullptr || shape == nullptr) {
        ReportInvalid(ctx, "input", "index=" + std::to_string(i), "descriptor and shape must be present");
    }
    return valid;
}

bool ProductFits(std::initializer_list<uint64_t> factors)
{
    uint64_t v = 1;
    for (auto f : factors) {
        if (f != 0 && v > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) / f) {
            return false;
        }
        v *= f;
    }
    return true;
}

} // namespace

ge::graphStatus DualLevelGroupedMatmulTiling::CheckAttributes()
{
    auto* ctx = context_;
    const auto* attrs = ctx->GetAttrs();
    const bool* trans = attrs->GetAttrPointer<bool>(DualLevelAttrIndex::TRANSPOSE_WEIGHT);
    const auto* groupType = attrs->GetAttrPointer<int64_t>(DualLevelAttrIndex::GROUP_LIST_TYPE);
    const auto* tuning = attrs->GetAttrPointer<gert::ContinuousVector>(DualLevelAttrIndex::TUNING_CONFIG);
    const auto* dtype = attrs->GetAttrPointer<int64_t>(DualLevelAttrIndex::OUTPUT_DTYPE);
    const auto* level0Size = attrs->GetAttrNum() > DualLevelAttrIndex::LEVEL0_GROUP_SIZE ?
                                 attrs->GetAttrPointer<int64_t>(DualLevelAttrIndex::LEVEL0_GROUP_SIZE) :
                                 nullptr;
    if (trans == nullptr || groupType == nullptr || dtype == nullptr || tuning == nullptr || *groupType < 0 ||
        *groupType > SPARSE_GROUP_LIST || tuning->GetSize() != 1 || tuning->GetData() == nullptr ||
        static_cast<const int64_t*>(tuning->GetData())[0] != 0) {
        ReportInvalid(ctx, "attributes", "invalid or missing",
                      "expected transpose_weight and dtype, supported "
                      "group_list_type, and tuning_config={0}");
        return ge::GRAPH_FAILED;
    }
    // Older callers can omit the appended group-size attributes.
    const int64_t segmentSize = level0Size == nullptr ? DEFAULT_LEVEL0_GROUP_SIZE : *level0Size;
    if (segmentSize != DEFAULT_LEVEL0_GROUP_SIZE) {
        ReportInvalid(ctx, "level0_group_size", std::to_string(segmentSize),
                      "expected " + std::to_string(DEFAULT_LEVEL0_GROUP_SIZE));
        return ge::GRAPH_FAILED;
    }
    const auto* level1Size = attrs->GetAttrNum() > DualLevelAttrIndex::LEVEL1_GROUP_SIZE ?
                                 attrs->GetAttrPointer<int64_t>(DualLevelAttrIndex::LEVEL1_GROUP_SIZE) :
                                 nullptr;
    if (level1Size != nullptr && *level1Size != DEFAULT_LEVEL1_GROUP_SIZE) {
        ReportInvalid(ctx, "level1_group_size", std::to_string(*level1Size),
                      "expected " + std::to_string(DEFAULT_LEVEL1_GROUP_SIZE));
        return ge::GRAPH_FAILED;
    }
    tilingData_.level0Size = static_cast<uint64_t>(segmentSize);
    transposeWeight_ = *trans;
    groupListType_ = *groupType;
    outputDtype_ = *dtype;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DualLevelGroupedMatmulTiling::GetShapeAttrs()
{
    auto* ctx = context_;
    auto& t = tilingData_;
    const auto* xs = ctx->GetInputShape(DualLevelInputIndex::X);
    const auto* ws = ctx->GetInputShape(DualLevelInputIndex::WEIGHT);
    const auto* wd = ctx->GetInputDesc(DualLevelInputIndex::WEIGHT);
    if (xs == nullptr || ws == nullptr || wd == nullptr || xs->GetOriginShape().GetDimNum() != MATRIX_RANK ||
        ws->GetOriginShape().GetDimNum() != EXPERT_WEIGHT_RANK) {
        ReportInvalid(ctx, "x/weight", "invalid or missing descriptors", "expected rank-two x and rank-three weight");
        return ge::GRAPH_FAILED;
    }
    t.m = xs->GetOriginShape().GetDim(0);
    t.k = xs->GetOriginShape().GetDim(1);
    t.experts = ws->GetOriginShape().GetDim(0);
    t.n = ws->GetOriginShape().GetDim(transposeWeight_ ? 1 : WEIGHT_NK_AXIS);
    const auto wk = ws->GetOriginShape().GetDim(transposeWeight_ ? WEIGHT_NK_AXIS : 1);
    // Packed FP4 rows need an even inner dimension, as in MXFP4 GMM.
    if (t.m <= 0 || t.n <= 1 || t.k <= MIN_K || t.experts <= 0 || wk != t.k || t.k % FP4_ELEMENTS_PER_BYTE != 0 ||
        (!transposeWeight_ && t.n % FP4_ELEMENTS_PER_BYTE != 0)) {
        ReportInvalid(ctx, "x/weight shape", DimsString({t.m, t.n, t.k, t.experts, wk}),
                      "expected positive M and E, N greater than one, equal even K greater than two, "
                      "and even N for non-transposed weight");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DualLevelGroupedMatmulTiling::CheckAddressBounds()
{
    auto& t = tilingData_;
    // Address products and hardware descriptor dimensions must be representable.
    constexpr int64_t maxDim = std::numeric_limits<int32_t>::max() - 1;
    if (std::max({t.m, t.n, t.k, t.experts}) > maxDim) {
        ReportInvalid(context_, "M,N,K,E", DimsString({t.m, t.n, t.k, t.experts}),
                      "each dimension must be at most " + std::to_string(maxDim));
        return ge::GRAPH_FAILED;
    }
    const int64_t level0Size = static_cast<int64_t>(t.level0Size);
    const int64_t level0Groups = CeilDiv(t.k, level0Size);
    const int64_t mxScaleGroups = CeilDiv(t.k, MX_SCALE_K);
    const int64_t weightRows = transposeWeight_ ? t.n : t.k, weightCols = transposeWeight_ ? t.k : t.n;
    const int64_t weightRowBlocks = CeilDiv(weightRows, NZ_ROWS);
    const int64_t weightColBlocks = CeilDiv(weightCols, NZ_COLS);
    if (!ProductFits({static_cast<uint64_t>(t.m), static_cast<uint64_t>(t.k)}) ||
        // The MX kernel multiplies the padded FP4 element stride by expert
        // before converting the offset to packed bytes. Check that product too.
        !ProductFits({static_cast<uint64_t>(t.experts), static_cast<uint64_t>(weightColBlocks),
                      static_cast<uint64_t>(weightRowBlocks), NZ_ROWS, NZ_COLS}) ||
        !ProductFits({static_cast<uint64_t>(t.experts), static_cast<uint64_t>(weightColBlocks),
                      static_cast<uint64_t>(weightRowBlocks), NZ_BLOCK_BYTES}) ||
        !ProductFits({static_cast<uint64_t>(t.experts), static_cast<uint64_t>(t.n),
                      static_cast<uint64_t>(mxScaleGroups), MX_SCALE_PAIR}) ||
        !ProductFits({static_cast<uint64_t>(t.m), static_cast<uint64_t>(t.n), sizeof(uint16_t)}) ||
        !ProductFits({static_cast<uint64_t>(t.m), static_cast<uint64_t>(level0Groups), sizeof(float)}) ||
        !ProductFits({static_cast<uint64_t>(t.experts), static_cast<uint64_t>(level0Groups), static_cast<uint64_t>(t.n),
                      sizeof(float)})) {
        ReportInvalid(context_, "M,N,K,E", DimsString({t.m, t.n, t.k, t.experts}),
                      "tensor offsets and padded weight element counts must fit int64");
        return ge::GRAPH_FAILED;
    }
    t.weightStrideBytes = static_cast<uint64_t>(weightColBlocks) * weightRowBlocks * NZ_BLOCK_BYTES;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DualLevelGroupedMatmulTiling::CheckInputShapes()
{
    auto* ctx = context_;
    auto& t = tilingData_;
    const auto* ws = ctx->GetInputShape(DualLevelInputIndex::WEIGHT);
    const auto* wd = ctx->GetInputDesc(DualLevelInputIndex::WEIGHT);
    const int64_t level0Size = static_cast<int64_t>(t.level0Size);
    const int64_t level0Groups = CeilDiv(t.k, level0Size);
    const int64_t mxScaleGroups = CeilDiv(t.k, MX_SCALE_K);
    const int64_t weightRows = transposeWeight_ ? t.n : t.k;
    const int64_t weightCols = transposeWeight_ ? t.k : t.n;
    const int64_t weightRowBlocks = CeilDiv(weightRows, NZ_ROWS);
    const int64_t weightColBlocks = CeilDiv(weightCols, NZ_COLS);
    // The individual-op runtime can normalize an NZ tensor's context storage
    // shape to its logical shape. Its format remains NZ; derive the physical
    // padded group stride from that logical shape in either representation.
    const bool validWeightStorage =
        Match(ws->GetStorageShape(), {t.experts, weightColBlocks, weightRowBlocks, NZ_ROWS, NZ_COLS}) ||
        Match(ws->GetStorageShape(), {t.experts, weightRows, weightCols});
    if (wd->GetDataType() != ge::DT_FLOAT4_E2M1 || wd->GetStorageFormat() != ge::FORMAT_FRACTAL_NZ ||
        !validWeightStorage) {
        ReportInvalid(ctx, "weight",
                      "dtype=" + std::to_string(wd->GetDataType()) + ", format=" +
                          std::to_string(wd->GetStorageFormat()) + ", storage=" + ShapeString(ws->GetStorageShape()),
                      "expected FP4 E2M1 FRACTAL_NZ with the padded storage shape or its normalized logical shape");
        return ge::GRAPH_FAILED;
    }
    if (!NdInput(ctx, DualLevelInputIndex::X, ge::DT_FLOAT4_E2M1, {t.m, t.k}) ||
        !NdInput(ctx, DualLevelInputIndex::X_LEVEL0_SCALE, ge::DT_FLOAT, {t.m, level0Groups}) ||
        !NdInput(ctx, DualLevelInputIndex::X_LEVEL1_SCALE, ge::DT_FLOAT8_E8M0, {t.m, mxScaleGroups, MX_SCALE_PAIR}) ||
        !NdInput(ctx, DualLevelInputIndex::WEIGHT_LEVEL0_SCALE, ge::DT_FLOAT, {t.experts, level0Groups, t.n})) {
        return ge::GRAPH_FAILED;
    }
    const bool validGroups =
        groupListType_ == SPARSE_GROUP_LIST ?
            NdInput(ctx, DualLevelInputIndex::GROUP_LIST, ge::DT_INT64, {t.experts, GROUP_LIST_PAIR_SIZE}) :
            NdInput(ctx, DualLevelInputIndex::GROUP_LIST, ge::DT_INT64, {t.experts});
    const auto scaleRows = transposeWeight_ ? t.n : mxScaleGroups;
    const auto scaleCols = transposeWeight_ ? mxScaleGroups : t.n;
    if (!validGroups || !NdInput(ctx, DualLevelInputIndex::WEIGHT_LEVEL1_SCALE, ge::DT_FLOAT8_E8M0,
                                 {t.experts, scaleRows, scaleCols, MX_SCALE_PAIR})) {
        return ge::GRAPH_FAILED;
    }
    t.hasBias = ctx->GetOptionalInputShape(DualLevelInputIndex::BIAS) != nullptr;
    if (t.hasBias && !NdInput(ctx, DualLevelInputIndex::BIAS, ge::DT_FLOAT, {t.experts, t.n})) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DualLevelGroupedMatmulTiling::CheckOutputShape()
{
    auto* ctx = context_;
    auto& t = tilingData_;
    const auto* yd = ctx->GetOutputDesc(0);
    const auto* ys = ctx->GetOutputShape(0);
    if (yd == nullptr || ys == nullptr || (outputDtype_ != ge::DT_FLOAT16 && outputDtype_ != ge::DT_BF16) ||
        yd->GetDataType() != outputDtype_ || yd->GetStorageFormat() != ge::FORMAT_ND ||
        !Match(ys->GetOriginShape(), {t.m, t.n}) || !Match(ys->GetStorageShape(), {t.m, t.n})) {
        ReportInvalid(ctx, "out", ys == nullptr ? "missing shape" : ShapeString(ys->GetStorageShape()),
                      "expected FP16/BF16 ND output with shape " + DimsString({t.m, t.n}));
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DualLevelGroupedMatmulTiling::GetPlatformInfo()
{
    auto* ctx = context_;
    auto& t = tilingData_;
    platform_ascendc::PlatformAscendC platform(ctx->GetPlatformInfo());
    if (platform.GetSocVersion() != platform_ascendc::SocVersion::ASCEND950 || platform.GetCoreNumAic() == 0 ||
        platform.GetCoreNumAiv() < platform.GetCoreNumAic() * AIV_COUNT) {
        ReportInvalid(ctx, "platform",
                      "SoC=" + std::to_string(static_cast<int>(platform.GetSocVersion())) +
                          ", AIC=" + std::to_string(platform.GetCoreNumAic()) +
                          ", AIV=" + std::to_string(platform.GetCoreNumAiv()),
                      "expected Ascend 950 with a Cube core and the required Vector cores per Cube core");
        return ge::GRAPH_FAILED;
    }
    uint64_t ub = 0, l1 = 0, l0a = 0, l0b = 0, l0c = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, l0a);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, l0b);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0c);
    // All paths share the original MX L1 partitioning. Select a scheduler tile
    // from aggregate M/N; actual expert lengths are balanced on device.
    if (l1 < MIN_L1_BYTES || l0a < MIN_L0_OPERAND_BYTES || l0b < MIN_L0_OPERAND_BYTES || l0c < MIN_L0C_BYTES) {
        ReportInvalid(ctx, "core memory",
                      "L1=" + std::to_string(l1) + ", L0A=" + std::to_string(l0a) + ", L0B=" + std::to_string(l0b) +
                          ", L0C=" + std::to_string(l0c),
                      "insufficient capacity for the dual-level MX buffers");
        return ge::GRAPH_FAILED;
    }
    t.coreNum = platform.GetCoreNumAic();
    ubSize_ = ub;
    libApiWorkspaceSize_ = platform.GetLibApiWorkSpaceSize();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DualLevelGroupedMatmulTiling::CalBasicBlock()
{
    auto* ctx = context_;
    auto& t = tilingData_;
    // Same selection rule as Ascend 950 GMM MXFP4, owned by this experimental operator.
    constexpr int64_t MAX_BASIC_BLOCK = 256;
    constexpr int64_t CUBE_BLOCK = 16;
    constexpr int64_t FP4_INNER_ALIGN = 64;
    t.baseM = Ops::Base::CeilAlign(std::min(t.m, MAX_BASIC_BLOCK), CUBE_BLOCK);
    t.baseN = Ops::Base::CeilAlign(std::min(t.n, MAX_BASIC_BLOCK), transposeWeight_ ? CUBE_BLOCK : FP4_INNER_ALIGN);
    const auto requiredUb = CalcEpilogueUbBytes(t.baseM, t.baseN);
    if (ubSize_ < requiredUb) {
        ReportInvalid(ctx, "UB", std::to_string(ubSize_),
                      "expected at least " + std::to_string(requiredUb) + " bytes for the basic block");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DualLevelGroupedMatmulTiling::SaveTilingData()
{
    auto* ctx = context_;
    auto& t = tilingData_;
    t.transposeWeight = transposeWeight_;
    t.groupListType = groupListType_;
    t.outputBf16 = outputDtype_ == ge::DT_BF16;
    auto* raw = ctx->GetRawTilingData();
    auto* workspace = ctx->GetWorkspaceSizes(1);
    if (raw == nullptr || raw->GetData() == nullptr || raw->GetCapacity() < sizeof(t) || workspace == nullptr) {
        ReportInvalid(ctx, "tiling buffers", raw == nullptr ? "missing buffer" : std::to_string(raw->GetCapacity()),
                      "expected workspace and a tiling buffer with capacity of at least " + std::to_string(sizeof(t)));
        return ge::GRAPH_FAILED;
    }
    std::memcpy(raw->GetData(), &t, sizeof(t));
    raw->SetDataSize(sizeof(t));
    workspace[0] = libApiWorkspaceSize_;
    ctx->SetBlockDim(t.coreNum);
    const auto key = t.outputBf16 ? (t.transposeWeight ? DLGMM_BF16_TRANSPOSE_TILING_KEY : DLGMM_BF16_TILING_KEY) :
                                    (t.transposeWeight ? DLGMM_FP16_TRANSPOSE_TILING_KEY : DLGMM_FP16_TILING_KEY);
    ctx->SetTilingKey(key);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DualLevelGroupedMatmulTiling::DoTiling()
{
    if (context_ == nullptr || context_->GetAttrs() == nullptr || context_->GetPlatformInfo() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const bool valid = CheckAttributes() == ge::GRAPH_SUCCESS && GetShapeAttrs() == ge::GRAPH_SUCCESS &&
                       CheckAddressBounds() == ge::GRAPH_SUCCESS && CheckInputShapes() == ge::GRAPH_SUCCESS &&
                       CheckOutputShape() == ge::GRAPH_SUCCESS && GetPlatformInfo() == ge::GRAPH_SUCCESS &&
                       CalBasicBlock() == ge::GRAPH_SUCCESS && SaveTilingData() == ge::GRAPH_SUCCESS;
    return valid ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED;
}
namespace {
ge::graphStatus TilingDualLevelGroupedMatmul(gert::TilingContext* context)
{
    return DualLevelGroupedMatmulTiling(context).DoTiling();
}
ge::graphStatus TilingPrepareForDualLevelGroupedMatmul(gert::TilingParseContext*)
{
    return ge::GRAPH_SUCCESS;
}
} // namespace
IMPL_OP_OPTILING(DualLevelGroupedMatmul)
    .Tiling(TilingDualLevelGroupedMatmul)
    .TilingParse<DualLevelGroupedMatmulCompileInfo>(TilingPrepareForDualLevelGroupedMatmul);
} // namespace optiling
