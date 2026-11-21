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
 * \file grouped_no_quant_matmul_tiling.h
 * \brief
 */
#ifndef GROUPED_NO_QUANT_MATMUL_TILING_H
#define GROUPED_NO_QUANT_MATMUL_TILING_H

#include "grouped_no_quant_tiling_base.h"
#include "../../../op_kernel/arch35/grouped_matmul_tiling_data_apt.h"
#include "log/log.h"
#include "log/error_code.h"
#include "register/op_impl_registry.h"

namespace optiling {
// Constants exclusive to the GMM operator (common and group-parsing constants live in grouped_no_quant_tiling_base.h).
constexpr uint64_t TILING_KEY_DELIMITER = 9UL;
constexpr uint64_t NZ_DIM_NUM = 4U;
constexpr uint64_t BIAS_TABLE_NUM = 256UL;
constexpr uint64_t DATA_SIZE_FP32 = 4UL;
constexpr uint32_t INDEX_BIAS = 2U;
constexpr uint32_t INDEX_GROUPLIST = 7U;
constexpr uint32_t INDEX_Y = 0U;
constexpr uint32_t FP32_DTYPE_SIZE = 4U;
constexpr uint64_t ATTR_IDX_SPLIT_ITEM = 0UL;
constexpr uint64_t ATTR_IDX_TRANS_W = 2UL;
constexpr uint64_t ATTR_IDX_TRANS_X = 3UL;
constexpr uint64_t ATTR_IDX_GROUPTYPE = 4UL;
constexpr uint64_t ATTR_IDX_GROUP_LIST_TYPE = 5UL;
constexpr uint64_t ATTR_IDX_ACT_TYPE = 6UL;
// Same values as the aclnn actType parameter (GMMActType): 0=NONE, 1=RELU, 2=GELU_TANH.
constexpr int64_t ACT_TYPE_NONE = 0L;
constexpr int64_t ACT_TYPE_RELU = 1L;
constexpr int64_t ACT_TYPE_GELU_TANH = 2L;
constexpr uint64_t BLOCK_SIZE = 32UL;
constexpr uint64_t ALIGN_DOWN_16 = 15UL;
constexpr uint64_t ALIGN_128 = 128UL;
constexpr uint64_t PARTA_L1_SIZE = 256UL * 1024UL;

enum class GMMTrans : std::uint8_t {
    NO_TRANS = 0,
    A_TRANS = 1,
    B_TRANS = 2,
    AB_TRANS = 3
};

enum class Model : std::uint8_t {
    BASIC = 0
};

class TilingKeyBuilder {
public:
    // Bits 0-1: platform major/minor class.
    uint8_t model = 0;

    // Bit 20: transpose flag.
    uint8_t gmmTrans = 0;

    // Activation type: 0-none, 1-relu, 2-gelu tanh.
    uint8_t act = 0;

public:
    uint64_t GenTilingKey() const;
};

class GroupedNoQuantMatmulTiling : public GroupedNoQuantTilingBase {
public:
    GroupedNoQuantMatmulTiling()
        : GroupedNoQuantTilingBase(INDEX_GROUPLIST)
    {}
    bool SetTiling(gert::TilingContext* context);

protected:
    // L1 reservation for the bias table: reserved whenever a bias input is present.
    uint64_t GetBiasReserveSize() const override;
    // Output tensor mode derived from splitItem (2&3 mean a single output).
    bool IsSingleY() const override
    {
        return isSingleY_;
    }

    bool Init(const gert::TilingContext* context);
    bool CheckNoQuantGroupList(const gert::TilingContext* context) const;
    // Validates that the groupList length is consistent with groupNum_ (common to all split scenarios).
    bool CheckGroupListLen(const gert::TilingContext* context) const;
    bool CalBaseMMTiling(const gert::TilingContext* context, const GMMCompileInfo* compileInfoPtr);
    bool CalL1Tiling(const gert::TilingContext* context, const GMMCompileInfo* compileInfoPtr);
    void SetDisableL2Cache(const gert::TilingContext* context, const GMMCompileInfo* compileInfoPtr);
    bool GetAttrs(const gert::TilingContext* context);
    bool CalMatMulTiling(const gert::TilingContext* context, const GMMCompileInfo* compileInfoPtr);
    void SetGMMTiling();
    void SetMatMulTiling();
    void SetTilingKey(gert::TilingContext* context);
    bool GMMGetTensorShapeSplitM(const gert::TilingContext* context, const gert::Shape xShape,
                                 const gert::Shape wShape);
    bool GMMGetTensorShapeSplitK(const gert::TilingContext* context, const gert::Shape xShape,
                                 const gert::Shape wShape);
    void PrintTilingResult(const gert::TilingContext* context);
    bool CheckWeightNZShape(const gert::TilingContext* context, int64_t numInOneBlk) const;
    bool HandleGroupTypeDispatch(const gert::TilingContext* context, const gert::Shape& xShape,
                                 const gert::Shape& wShape, uint32_t wDimNum);

private:
    bool isSingleX_ = true;
    bool isSingleWeight_ = true;
    bool isSingleY_ = true;
    bool hasBias_ = false;
    bool weightNzFlag_ = false;

    int64_t splitItem_ = 0;
    int64_t activeType_ = ACT_TYPE_NONE;
    uint32_t groupSize_ = 0;
    uint32_t splitM_ = 1U;

    TilingKeyBuilder tilingKeyBuilder_;
    GroupedMatmulTilingData::GMMNoQuantTilingData tilingData_;
};
} // namespace optiling

#endif // GROUPED_WEIGHT_QUANT_BATCH_MATMUL_TILING_H
