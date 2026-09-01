/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_BASE_TILING_H_
#define QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_BASE_TILING_H_

#include <cstdint>

#include "op_host/tiling_base.h"
#include "../../../op_kernel/arch35/qkv_rms_norm_rope_cache_with_k_scale_mrope_mx_layout.h"
#include "qkv_rms_norm_rope_cache_with_k_scale_tiling.h"

namespace optiling {
namespace QkvRmsNormRopeCacheWithKScale {

enum class RopeMode : uint64_t {
    ROPE = 0,
    MROPE = 1
};
enum class QQuantMode : uint64_t {
    PER_TOKEN_PER_HEAD = 0,
    NO_QUANT = 1,
    MX_QUANT = 2
};
enum class KQuantMode : uint64_t {
    PER_TOKEN_PER_HEAD = 0,
    MX_QUANT = 1
};
enum class Scene : uint8_t {
    ROPE,
    MROPE,
    MROPE_MX
};
struct SceneTraits {
    ge::DataType qOutDtype;
    ge::DataType kCacheDtype;
    ge::DataType kScaleCacheDtype;
    uint32_t vScaleRank;
    bool needsRotation;
    bool needsMropePosition;
    bool usesMxScaleLayout;
    bool usesAivOnlyKernel;
    uint64_t qkvInputRowsPerAiv;
};
struct TensorContractInfo {
    gert::Shape shape;
    gert::Stride stride;
    ge::DataType dtype = ge::DT_UNDEFINED;
    bool descriptorPresent = false;
    bool shapePresent = false;
    bool stridePresent = false;
};

struct ContractInput {
    uint64_t totalTokens = 0;
    uint64_t batch = 0;
    uint64_t numQHeads = 0;
    uint64_t numKHeads = 0;
    uint64_t numVHeads = 0;
    uint64_t headDim = 0;
    uint64_t maxSeqLen = 0;
    uint64_t blockNum = 0;
    uint64_t blockSize = 0;
    uint64_t layoutQkv = 0;
    uint64_t layoutQOut = 0;

    TensorContractInfo qkv;
    TensorContractInfo qGamma;
    TensorContractInfo kGamma;
    TensorContractInfo cosSin;
    TensorContractInfo slotMapping;
    TensorContractInfo kCache;
    TensorContractInfo vCache;
    TensorContractInfo kScaleCache;
    TensorContractInfo queryStartLoc;
    TensorContractInfo seqLens;
    TensorContractInfo rotation;
    TensorContractInfo vScale;
    TensorContractInfo mropePosition;
};

class QkvRmsNormRopeCacheWithKScaleBaseTiling : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    static constexpr uint64_t SUPPORTED_HEAD_DIM = 128;
    static constexpr uint64_t MAX_Q_HEAD_NUM = 64;
    static constexpr uint64_t Q_TO_K_HEAD_RATIO = 8;
    static constexpr uint64_t MAX_TOKEN_TILE = 16;
    static constexpr uint64_t L0A_FULL_DIM_SEGMENT_ROWS = 128;
    static constexpr uint64_t L0C_MAX_ROWS = 256;
    static constexpr uint64_t QKV_INPUT_ROWS_PER_AIV = 160;
    static constexpr uint64_t MROPE_COMPACT_INPUT_ROWS_PER_AIV = 80;
    static constexpr uint64_t QK_OUTPUT_ROWS_PER_AIV = 128;
    static constexpr uint64_t QK_PREPROCESS_UB_BYTES = 64UL * 1024UL;
    static constexpr uint64_t QK_PREPROCESS_BLOCK_BYTES = 32;
    static constexpr uint64_t QK_PREPROCESS_UB_NZ_STRIDE_ALIGN = 16;
    static constexpr uint64_t QK_PREPROCESS_NZ_D_BLOCKS = 8;
    static constexpr uint64_t V_OUTPUT_ROWS_PER_AIV = 80;
    static constexpr uint64_t DIM_TILE = 128;
    static constexpr uint64_t AIV_PER_AIC = 2;
    static constexpr uint64_t MAX_TOKENS = 262144;

    explicit QkvRmsNormRopeCacheWithKScaleBaseTiling(gert::TilingContext *context)
        : TilingBaseClass(context)
    {}
    ~QkvRmsNormRopeCacheWithKScaleBaseTiling() override = default;

    void Reset(gert::TilingContext *context) override;

protected:
    bool IsCapable() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    void DumpTilingInfo() override;

private:
    void Reset();

    ge::graphStatus ValidateHeadNums() const;
    ge::graphStatus ValidateScalarInputs() const;
    ge::graphStatus ValidateDtypes() const;
    ge::graphStatus ValidateMropeSection();
    ge::graphStatus ValidateQOutDtypeAttr() const;
    ge::graphStatus ValidateShapeRank(const TensorContractInfo &tensor, const char *tensorName,
                                      uint64_t expectedRank) const;
    ge::graphStatus ValidateShapeInputsForDerivedFields() const;
    ge::graphStatus ValidateNonCacheShapeRanks() const;
    ge::graphStatus ValidateQkvAndGammaShapes() const;
    ge::graphStatus ValidatePositionInputShapes() const;
    ge::graphStatus ValidateRotationAndScaleShapes() const;
    ge::graphStatus ValidateNonCacheShapes() const;
    ge::graphStatus ValidateCacheShapes() const;
    ge::graphStatus ValidateShapes() const;
    ge::graphStatus ValidateStrides() const;
    bool TrySelectTokenTile(uint64_t tokenTile) const;
    uint64_t SelectTokenTile() const;
    // Checks the device-side field widths and copy strides derived from one
    // M-RoPE MX tile. UB offsets and total size are validated by the layout
    // builder and SelectMropeMxTokenTile(), respectively.
    bool AreMropeMxCopyAndVfFieldsRepresentable(uint64_t tokenTile,
                                                const ::QkvRmsNormRopeCacheWithKScale::MropeMxUbLayout &layout) const;
    // Selects the largest tile that fits the position window, UB capacity,
    // DataCopy parameters, and VF loop/stride fields. Outputs are written only
    // when a feasible candidate is found.
    ge::graphStatus SelectMropeMxTokenTile(uint64_t &tokenTile,
                                           ::QkvRmsNormRopeCacheWithKScale::MropeMxUbLayout &selectedLayout) const;
    void FillTilingData(uint64_t tokenTile);
    void FillMropeMxTilingData(uint64_t tokenTile);
    ge::graphStatus ValidateParsedInput() const;
    ge::graphStatus ComputeTilingData();
    void LogTensorInfo(const char *tensorName, const TensorContractInfo &info) const;
    void LogContractInput() const;
    void LogTilingData() const;
    ge::graphStatus ParseHeadNumsAttr();
    ge::graphStatus ParseLayoutQkvAttr();
    ge::graphStatus ParseLayoutQOutAttr();
    ge::graphStatus ParseQQuantModeAttr();
    ge::graphStatus ParseKQuantModeAttr();
    ge::graphStatus ParseQOutDtypeAttr();
    void CacheMropeSectionAttr();
    ge::graphStatus ResolveScene();
    float ParseEpsilonAttr() const;
    ge::graphStatus FillShapeDerivedFields();
    ge::graphStatus FillRequiredTensorInputs();
    void FillOptionalTensorInputs();
    ge::graphStatus BuildContractInput();
    ge::graphStatus ValidateCompileInfo(const QkvRmsNormRopeCacheWithKScaleCompileInfo &compileInfo) const;
    /**
     * @param scene Validated operator scene produced by ResolveScene().
     */
    static const SceneTraits &GetSceneTraits(Scene scene);
    const char *opName_ = "QkvRmsNormRopeCacheWithKScale";
    const QkvRmsNormRopeCacheWithKScaleCompileInfo *compileInfo_ = nullptr;
    ContractInput input_;
    QkvRmsNormRopeCacheWithKScaleTilingData tilingData_;
    uint64_t aicNum_ = 0;
    uint64_t aivNum_ = 0;
    RopeMode ropeMode_ = RopeMode::ROPE;
    QQuantMode qQuantMode_ = QQuantMode::PER_TOKEN_PER_HEAD;
    KQuantMode kQuantMode_ = KQuantMode::PER_TOKEN_PER_HEAD;
    Scene scene_ = Scene::ROPE;
    ge::DataType qOutDtypeAttr_ = ge::DT_FLOAT8_E4M3FN;
    float epsilon_ = 1e-6f;
    uint64_t tilingDataSize_ = 0;
    uint64_t mropeSectionSize_ = 0;
    bool hasMropeSection_ = false;
    const int64_t *mropeSectionData_ = nullptr;
    uint64_t mropeSectionH_ = 0;
    uint64_t mropeSectionW_ = 0;
};

} // namespace QkvRmsNormRopeCacheWithKScale
} // namespace optiling

#endif // QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_BASE_TILING_H_
