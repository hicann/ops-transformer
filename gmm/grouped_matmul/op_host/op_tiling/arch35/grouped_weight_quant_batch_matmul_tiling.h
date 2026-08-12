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
 * \file grouped_weight_quant_batch_matmul_tiling.h
 * \brief
 */
#ifndef GROUPED_WEIGHT_QUANT_BATCH_MATMUL_TILING_H
#define GROUPED_WEIGHT_QUANT_BATCH_MATMUL_TILING_H

#include <graph/utils/type_utils.h>
#include <sstream>
#include <map>
#include <unordered_set>

#include "../grouped_matmul_tiling.h"
#include "grouped_quant_matmul_tiling.h"
#include "../../../op_kernel/arch35/grouped_matmul_tiling_data_apt.h"
#include "log/log.h"
#include "register/op_impl_registry.h"

namespace optiling {
constexpr uint32_t X_IDX = 0;
constexpr uint32_t WEIGHT_IDX = 1;
constexpr uint32_t BIAS_IDX = 2;
constexpr uint32_t ANTIQUANT_SCALE_IDX = 5;
constexpr uint32_t ANTIQUANT_OFFSET_IDX = 6;
constexpr uint64_t ATTR_SPLIT_ITEM_IDX = 0;
constexpr uint64_t ATTR_TRANS_W_IDX = 2;
constexpr uint64_t ATTR_TRANS_X_IDX = 3;
constexpr uint64_t ATTR_GROUPTYPE_IDX = 4;
constexpr uint32_t ATTR_GROUP_LIST_TYPE_IDX = 5;

constexpr uint32_t MAX_X_DIM = 6UL;
constexpr uint32_t MIN_X_DIM = 2UL;
constexpr size_t ANTIQUANT_SCALE_DIM_NUM = 4;       // MX格式antiquantScale维度数
constexpr size_t PENULTIMATE_DIM = 2;               // 倒数第2维
constexpr size_t ANTEPENULTIMATE_DIM = 3;           // 倒数第3维
constexpr int64_t MX_GROUP_FACTOR = 2;              // MX格式groupNum计算因子
constexpr int64_t A16W4_SINGLE_PER_CHANNEL_DIM = 2; // 单单单: scale/offset [E, N]
constexpr int64_t A16W4_SINGLE_PER_GROUP_DIM = 3;   // 单单单: scale/offset [E, G, N]
constexpr int64_t A16W4_MULTI_PER_CHANNEL_DIM = 1;  // 多多多: scale/offset [N_i]
constexpr int64_t A16W4_MULTI_PER_GROUP_DIM = 2;    // 多多多: scale/offset [G_i, N_i]

constexpr uint32_t BASIC_BLOCK_BASE_M = 256;
constexpr uint32_t BASIC_BLOCK_BASE_M_WITH_BIAS = 240;
constexpr uint32_t BASIC_BLOCK_BASE_N = 256;
constexpr uint32_t BASIC_BLOCK_BASE_K = 64;
constexpr uint32_t BASIC_BLOCK_BASE_N_MIN = 128;
constexpr uint32_t STEP_K_4 = 4;
constexpr uint32_t STEP_K_3 = 3;
constexpr uint32_t DEPTH_8 = 8;
constexpr uint32_t BUFFER_NUM_2 = 2;

constexpr uint32_t SCALE_FACTOR_DEFAULT = 1;
constexpr uint32_t SCALE_FACTOR_MIN = 1;
constexpr uint32_t SCALE_FACTOR_B_BIT = 8;
constexpr uint32_t SCALE_FACTOR_M_BIT = 16;
constexpr uint32_t SCALE_FACTOR_N_BIT = 24;

constexpr int32_t B16_DATA_SIZE = 2;
constexpr int32_t B8_DATA_SIZE = 1;

constexpr uint32_t MX_GROUP_SIZE = 32;

constexpr uint32_t AIC_AIV_CORE_RATIO = 2;

constexpr uint32_t S8S4_BASEK_MULTIPLIER = 2;                       // S8S4场景MAD采用S8类型，baseK需放大2倍
constexpr uint64_t FP4_PER_FP32 = 8;                                // 1个float32/int32表示8个fp4/int4
constexpr uint64_t DEFAULT_WORKSPACE_SIZE = 16UL * 1024UL * 1024UL; // 16MB默认workspace大小

struct TailBlockResplitParam {
    uint32_t mainBlockSize = 0;
    uint64_t mainBlockCount = 0;
    uint16_t firstTailBlockSize = 0;
    uint16_t secondTailBlockSize = 0;
    uint16_t firstTailBlockCount = 0;
    uint16_t secondTailBlockCount = 0;
};

enum class GroupType : int64_t {
    NO_SPLIT = -1,
    SPLIT_M = 0,
    SPLIT_N = 1,
    SPLIT_K = 2,
};

enum class QuantType : uint8_t {
    NONE = 0,
    PER_TENSOR = 1,
    PER_CHANNEL = 2,
    PER_GROUP = 3,
    MX = 4
};

enum class WeightFormat : uint8_t {
    ND = 0,
    FRACTAL_NZ = 1,
};

// 对应0位 平台大类
enum class SocVersionType : uint8_t {
    RESERVERD = 0,
    SUPPORT_L0C_TO_OUT = 1,
    SUPPORT_L1_TO_BT_BF16 = 2,
};

// 对应1位 平台小类
enum class SocVersionSubType : uint8_t {
    RESERVERD = 0,
};

// 对应2-3位 伪量化场景
enum class QuantizationScenario : uint8_t {
    DEFAULT = 0,
};

// 对应4位 算法大类
enum class OptimizationAlgorithmCategory : uint8_t {
    VECTOR_ANTIQUANT = 0,
    MULTI_SCALE_DEQUANT = 1,
    FIXPIPE_ANTIQUANT = 2,
};

// 对应5位 算法小类
enum class OptimizationAlgorithmSubCategory : uint8_t {
    VDEFAULT = 0,
    SPLIT_K = 1,
    N_FIRST_TAIL_RESPLIT = 2,
    N_FIRST_BASIC_BLOCK = 3,
};

// 对应6-9位 fixp模板自定义组合
enum class FixpipeConfiguration : uint16_t {
    A_NORMAL_LOAD = 0,
    A_SINGLE_M_SINGLE_K_FULL_LOAD = 1,
};

enum class CustomSplitKConfiguration : uint8_t {
    A_NORMAL_LOAD = 0,
    A_MK_FULL_LOAD = 1,
};

// 对应14位 表示转置场景，transA/transB
enum class TransposeSituation : uint8_t {
    A_NOT_TRANS_B_NOT_TRANS = 0,
    A_NOT_TRANS_B_TRANS = 1,
    A_TRANS_B_NOT_TRANS = 2,
    A_TRANS_B_TRANS = 3,
};

// 对应17位 可选输入是否存在 hasAntiquantOffset/hasBias
enum class OptionInputSituation : uint8_t {
    ANTIQUANT_OFFSET_NOT_EXIST_BIAS_NOT_EXIST = 0,
    ANTIQUANT_OFFSET_NOT_EXIST_BIAS_EXIST = 1,
    ANTIQUANT_OFFSET_EXIST_BIAS_NOT_EXIST = 2,
    ANTIQUANT_OFFSET_EXIST_BIAS_EXIST = 3,
    ANTIQUANT_OFFSET_NOT_EXIST_BIAS_FP32_EXIST = 4,
    ANTIQUANT_OFFSET_EXIST_BIAS_FP32_EXIST = 6,
};

enum class Mte2Configuration : uint8_t {
    MTE2_INNER_SIZE_512_BUF_NUM_2 = 0,
    MTE2_INNER_SIZE_512_BUF_NUM_4 = 1,
    MTE2_INNER_SIZE_1024_BUF_NUM_2 = 2,
    MTE2_INNER_SIZE_256_BUF_NUM_4 = 3,
    MTE2_INNER_SIZE_512_BUF_NUM_DEFAULT = 4, // w8 w4在非性能场景下复用一组设置
    MTE2_INNER_SIZE_384_BUF_NUM_3 = 5,
    MTE2_INNER_SIZE_256_BUF_NUM_2 = 6,
    MTE2_INNER_SIZE_DYNAMIC_BUF_NUM_4 = 15, // MXA8W4 NZ场景，K轴长度动态调整
};

class TilingKeyConfigure {
public:
    // 对应0-1位 平台大类，平台小类
    uint8_t socVersionType = 0;

    // 对应2-3位 伪量化场景
    uint8_t quantizationScenario = 0;

    // 对应4-5位 算法大类、算法小类
    uint8_t algorithm = 0;

    // 对应14位 表示转置场景，transA/transB
    uint8_t transposeSituation = 0;

    // 对应15位 表示反量化的类型 perchannel/pertensor/perGroup
    uint8_t antiquantType = 0;

    // 对应16位 表示量化的类型 perchannel/perTensor/None
    uint8_t quantType = 0;

    // 对应17位 可选输入是否存在 hasAntiquantOffset/hasBias
    uint8_t optionInputSituation = 0;

    // 对应18位 weight的数据类型 weightNd/weightNz
    uint8_t weightFormat = 0;

    // 对应6-9位 模板自定义组合
    uint16_t templateCustom = 0;

    // 模板参数，表示MxA8W4单多单场景
    bool isSingleMultiSingle = false;

    // 对应10-13位 api常量化保留位
    uint16_t apiConstexpr = 0;

public:
    void PrintTilingKeyLog() const
    {
        std::stringstream ss;
        ss << "socVersionType: " << static_cast<uint32_t>(this->socVersionType)
           << " quantizationScenario: " << static_cast<uint32_t>(this->quantizationScenario)
           << " algorithm: " << static_cast<uint32_t>(this->algorithm)
           << " transposeSituation: " << static_cast<uint32_t>(this->transposeSituation)
           << " antiquantType: " << static_cast<uint32_t>(this->antiquantType)
           << " quantType: " << static_cast<uint32_t>(this->quantType)
           << " optionInputSituation: " << static_cast<uint32_t>(this->optionInputSituation)
           << " weightFormat: " << static_cast<uint32_t>(this->weightFormat)
           << " templateCustom: " << static_cast<uint32_t>(this->templateCustom)
           << " isSingleMultiSingle: " << static_cast<uint32_t>(this->isSingleMultiSingle)
           << " apiConstexpr: " << static_cast<uint32_t>(this->apiConstexpr);
        OP_LOGI("GMMWeightQuantBatchMatmul", "tilingKeyConfigure: %s", ss.str().c_str());
        return;
    }

    uint64_t GenTilingKey() const;
};

class GroupedS8S4BasicApiTiling : public GroupedQmmTiling {
public:
    explicit GroupedS8S4BasicApiTiling(gert::TilingContext *context);
    ~GroupedS8S4BasicApiTiling() override = default;

    void Reset(gert::TilingContext *context) override
    {
        Ops::Transformer::OpTiling::TilingBaseClass::Reset(context);
        Reset();
    }

protected:
    bool IsCapable() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    bool AnalyzeDtype() override;
    bool AnalyzeAttrs() override;
    bool AnalyzeInputs() override;
    uint64_t GetTilingKey() const override;
    void Reset() override;

private:
    enum class DequantMode : uint8_t {
        SYMMETRIC_PER_GROUP = 0,
        ASYMMETRIC_PER_CHANNEL = 1,
    };

    bool CheckTensorLists() const;
    bool CheckCommonShapes(const gert::Shape &xShape, const gert::Shape &wShape) const;
    bool CheckBiasShape() const;
    bool CheckPerTokenScaleShape() const;
    bool CheckAntiquantInputsEmpty() const;
    bool CheckWeightStorageShape(const gert::StorageShape &weightShape);
    bool CheckScaleAndOffsetShapes();
    bool SetLogicalMKN(const gert::Shape &xShape, const gert::Shape &wShape);
    void SetBasicBlock();
    ge::graphStatus ValidateFixedTileResources(uint64_t kL1) const;

    static constexpr uint32_t QUANT_GROUP_SIZE = 256U;
    static constexpr uint32_t S8S4_BASE_M = 256U;
    static constexpr uint32_t S8S4_ASYM_BASE_M = 256U;
    static constexpr uint32_t S8S4_BASE_N = 256U;
    static constexpr uint32_t S8S4_BASE_K = 128U;
    static constexpr uint32_t S8S4_PER_GROUP_K_L1 = 256U;
    static constexpr uint32_t S8S4_PER_CHANNEL_K_L1 = 256U;
    static constexpr uint64_t MAX_K_PER_GROUP = 18432UL;
    static constexpr uint64_t PACKED_INT32_RATIO = 8UL;
    static constexpr uint64_t NZ_BASE_K_ALIGN = 16UL;
    static constexpr uint64_t NZ_BASE_N_ALIGN = 32UL;
    static constexpr uint64_t NZ_SPECIAL_K_ALIGN = 32UL;
    static constexpr uint64_t NZ_SPECIAL_N_ALIGN = 16UL;
    static constexpr size_t TUNING_CONFIG_EXPECTED_TOKEN_INDEX = 0UL;
    static constexpr size_t TUNING_CONFIG_WEIGHT_FORMAT_INDEX = 1UL;
    static constexpr int64_t TUNING_CONFIG_SPECIAL_WEIGHT_FORMAT = 1L;
    static constexpr int64_t GROUP_LIST_COUNT = 1L;
    static constexpr int8_t X_SEPARATED = 2;
    static constexpr int8_t NO_SEPARATED = 3;

    bool hasOffset_ = false;
    bool specialWeightFormat_ = false;
    bool weightPackedInt32_ = false;
    uint32_t expectedTokenNum_ = 0;
    uint64_t userWorkspaceSize_ = 0;
    DequantMode dequantMode_ = DequantMode::SYMMETRIC_PER_GROUP;
    GroupedMatmulTilingData::GMMS8S4BasicApiTilingData tilingData_;
};

class GroupedWeightQuantBatchMatmulTiling {
public:
    bool SetTiling(gert::TilingContext *context);

protected:
    bool CheckCoreNum(const gert::TilingContext *context) const;
    bool SetShapeList(const gert::TilingContext *context);
    bool CheckEmptyTensor(const gert::TilingContext *context) const;
    bool CheckEmptyTensorSingleXMultiWeightSingleY(const gert::TilingContext *context) const;
    bool CheckEmptyTensorDefault(const gert::TilingContext *context) const;
    bool CheckTensorListSize() const;
    bool CheckTensorDtype(const gert::TilingContext *context, uint32_t attrIdx, size_t idx,
                          const ge::DataType &tensorDtype, const std::string &tensorType) const;
    bool CheckTensorDtypeSingleXMultiWeightSingleY(const gert::TilingContext *context, uint32_t attrIdx, size_t idx,
                                                   const ge::DataType &tensorDtype,
                                                   const std::string &tensorType) const;
    bool IsNzFormat(const gert::TilingContext *context, uint32_t attrIdx, size_t idx) const;
    bool CheckXAndWeightFormat(const gert::TilingContext *context, size_t idx) const;
    bool CheckNotNullPtr(const gert::TilingContext *context, uint32_t attrIdx, size_t idx) const;
    bool CheckNotNull(const gert::TilingContext *context, size_t idx) const;
    bool CheckTensorDimEqualTarget(const gert::TilingContext *context, uint32_t attrIdx, size_t idx, uint32_t targetDim,
                                   const std::string &tensorType) const;
    bool CheckTensorDimSingleXSingleWeightSingleY(const gert::TilingContext *context, size_t idx) const;
    bool CheckTensorDimMultiXMultiWeightMultiY(const gert::TilingContext *context, size_t idx) const;
    bool CheckTensorDim(const gert::TilingContext *context, size_t idx) const;
    bool CheckTensorShape(const gert::TilingContext *context, uint32_t attrIdx, size_t idx,
                          const std::string &tensorType) const;
    bool CheckTensorShapeSingleXMultiWeightSingleY(const gert::TilingContext *context, uint32_t attrIdx, size_t idx,
                                                   const std::string &tensorType) const;
    bool CheckDimValue(const gert::TilingContext *context, size_t idx) const;
    bool CheckWeightInnerAxisEven(const gert::TilingContext *context, size_t idx) const;
    bool CheckWeightSingleXMultiWeightSingleY(const gert::TilingContext *context) const;
    bool CheckAntiquantScaleSingleXMultiWeightSingleY(const gert::TilingContext *context) const;
    bool CheckAntiquantOffsetSingleXMultiWeightSingleY(const gert::TilingContext *context) const;
    bool CheckBiasSingleXMultiWeightSingleY(const gert::TilingContext *context) const;
    bool CheckEveryTensor(const gert::TilingContext *context) const;
    bool CheckEveryTensorSingleXMultiWeightSingleY(const gert::TilingContext *context) const;
    bool CheckGroupList(const gert::TilingContext *context) const;
    bool AnalyzeAttr(const gert::TilingContext *context);
    bool AnalyzeInput(const gert::TilingContext *context);
    bool CalcResplitTiling(const gert::TilingContext *context);
    bool SetBaseTiling();
    void SetMatMulTiling();
    void SetTilingKey(gert::TilingContext *context);
    bool SetCustomParam(gert::TilingContext *context);
    bool IsA16W4ND() const;
    bool IsA16W4NDPergroup() const;
    bool IsMxA8W4() const;
    bool CheckPerTokenScale(const gert::TilingContext *context) const;
    bool CheckUnsupportDataFlow() const;
    bool CheckUnsupportedRemainingCases() const;
    bool CheckAntiQuantDtype() const;
    bool CheckBiasDtype() const;
    bool CheckGroupTypeAndSplitItem() const;
    bool CheckTransposeStatus() const;
    bool SetShapeListSplitMSingleXSingleWeightSingleY(const gert::TilingContext *context);
    bool SetShapeListSplitMSingleXMultiWeightSingleY(const gert::TilingContext *context);
    bool SetShapeListMultiXMultiWeightMultiY(const gert::TilingContext *context);
    bool GetDimFromEnd(const gert::Shape &shape, size_t posFromEnd, uint64_t &out) const;
    uint16_t GetTensorListSize(const gert::TilingContext *context, uint32_t attrIdx) const;
    void GetNumOfInputs(const gert::TilingContext *context);
    bool SetAntiquantGroupSize(const gert::TilingContext *context);
    bool DeriveGroupSizeSingle(const gert::TilingContext *context);
    bool DeriveGroupSizeMulti(const gert::TilingContext *context);
    bool GetA16W4MultiScaleShape(const gert::TilingContext *context, uint16_t idx, gert::Shape &scaleShape) const;
    bool DeriveA16W4MultiPergroupSizeFromScale(const gert::Shape &scaleShape, int64_t kSize, uint16_t idx,
                                               uint32_t &groupSize) const;
    bool CheckA16W4MultiPergroupScaleShape(const gert::TilingContext *context, uint16_t idx, int64_t expectedDimNum,
                                           uint32_t expectedGroupSize) const;
    bool CheckMultiA16W4PerChannelShape(const gert::TilingContext *context) const;
    bool CheckAntiquantOffsetMatchScale(const gert::TilingContext *context, uint16_t idx,
                                        const gert::Shape &scaleShape) const;
    bool CheckGroupSize() const;
    bool GetC0Size(const gert::TilingContext *context, ge::DataType dtype, uint64_t &c0Size) const;
    void CalcFullNumBlocksResplitTiling(uint64_t c0Size);
    void CalcNoFullNumBlocksResplitTiling(uint64_t c0Size);
    bool CheckResplitTilingResult(const gert::TilingContext *context) const;
    void PrintInputParam(const gert::TilingContext *context) const;
    void PrintTilingResult(const gert::TilingContext *context);
    bool EnableTailResplit() const;

private:
    int32_t mList_[GroupedMatmul::MAX_TENSOR_CONT] = {0};
    int32_t kList_[GroupedMatmul::MAX_TENSOR_CONT] = {0};
    int32_t nList_[GroupedMatmul::MAX_TENSOR_CONT] = {0};

    bool transA_ = false;
    bool transB_ = false;
    bool isSingleX_ = true;
    bool isSingleWeight_ = true;
    bool isSingleY_ = true;
    bool isSingleMultiSingle_ = false;
    bool hasBias_ = false;
    bool weightNzFlag_ = false;
    bool hasAntiquantOffset_ = false;

    uint64_t mSize_ = 0;
    uint64_t kSize_ = 0;
    uint64_t nSize_ = 0;
    uint64_t nSizeOri_ = 0;
    int64_t groupType_ = static_cast<int64_t>(GroupType::SPLIT_M);
    int64_t splitItem_ = 0;
    uint32_t groupNum_ = 0;
    int64_t groupListType_ = 0;
    uint32_t coreNum_ = 0;
    uint32_t aivNum_ = 0;
    uint32_t groupSize_ = 0;
    uint8_t cubeNumBlocksN_ = 0;

    uint16_t numX_ = 0;
    uint16_t numWeight_ = 0;
    uint16_t numBias_ = 0;
    uint16_t numAntiquantScale_ = 0;
    uint16_t numAntiquantOffset_ = 0;

    ge::DataType xDType_ = ge::DT_UNDEFINED;
    ge::DataType weightDtype_ = ge::DT_UNDEFINED;
    ge::DataType biasDtype_ = ge::DT_UNDEFINED;
    ge::DataType antiquantScaleDtype_ = ge::DT_UNDEFINED;
    ge::DataType antiquantOffsetDtype_ = ge::DT_UNDEFINED;

    TailBlockResplitParam resplitParam_;
    TilingKeyConfigure tilingKeyConfig_;
    GroupedMatmulTilingData::GMMWeightQuantTilingData tilingData_;
};
} // namespace optiling

#endif // GROUPED_WEIGHT_QUANT_BATCH_MATMUL_TILING_H
