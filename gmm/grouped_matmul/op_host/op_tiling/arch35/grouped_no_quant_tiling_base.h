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
 * \file grouped_no_quant_tiling_base.h
 * \brief arch35(ascend950) no-quant 场景 GMM/GMM-ADD tiling 公共基类。
 */
#ifndef GROUPED_NO_QUANT_TILING_BASE_H
#define GROUPED_NO_QUANT_TILING_BASE_H

#include <algorithm>

#include "securec.h"

#include "../grouped_matmul_tiling.h"
#include "log/log.h"

namespace optiling {
constexpr uint64_t BASIC_BLOCK_SIZE_16 = 16UL;
constexpr uint64_t STEP_K_DEFAULT = 4UL;
constexpr uint64_t DEPTH_DEFAULT = 8UL;
constexpr uint64_t DB_SIZE = 2UL;
constexpr uint32_t MIN_DIM = 2U;
constexpr uint64_t BASE_M_DEFAULT = 256UL;
constexpr uint64_t BASE_N_DEFAULT = 256UL;
constexpr uint64_t BASE_K_DEFAULT = 64UL;
constexpr uint64_t NUM_TWO = 2UL;
constexpr uint32_t MAX_TENSOR = 1024U;
constexpr uint32_t DIM_ONE = 1U;
constexpr uint32_t DIM_TWO = 2U;
constexpr uint32_t DIM_THREE = 3U;
constexpr uint32_t DIM_FOUR = 4U;
constexpr uint32_t INDEX_X = 0U;
constexpr uint32_t INDEX_WEIGHT = 1U;
constexpr int32_t NO_SPLIT = -1;
constexpr int32_t SPLIT_M = 0;
constexpr int32_t SPLIT_K = 2;

class GroupedNoQuantTilingBase {
public:
    virtual ~GroupedNoQuantTilingBase() = default;

protected:
    explicit GroupedNoQuantTilingBase(uint32_t groupListIndex)
        : groupListIndex_(groupListIndex)
    {}

    // ---- 公共 shape 解析：split-K single-single-single(s-s-s) ----
    bool SplitKSingleXSingleWeightSingleY(const gert::TilingContext* context, const gert::Shape xShape,
                                          const gert::Shape wShape);

    // ---- 分组 shape 解析：x/weight/out 张量组合模式（当前仅 GMM 派生类调用） ----
    bool SplitMSingleXSingleWeightSingleY(const gert::Shape xShape, const gert::Shape wShape);
    bool SplitMSingleXSeparatedWeight(const gert::TilingContext* context, const gert::Shape xShape);
    bool SeparatedXSeparatedWeight(const gert::TilingContext* context);
    bool SeparatedXSingleWeight(const gert::TilingContext* context, const gert::Shape wShape);
    bool SplitKSingleXSeparatedWeight(const gert::TilingContext* context, const gert::Shape xShape);

    // ---- 公共切分算法 ----
    void FormulateBasicBlock(const GMMCompileInfo* compileInfoPtr, uint32_t remainCoreNum);
    void CalcTailBasicBlock(const GMMCompileInfo* compileInfoPtr);
    void CalAswtL1Tiling(const GMMCompileInfo* compileInfoPtr);
    // split-K 公共主流程：按组均分核数切 basic block；单组时追加 tail 块与 L1 depth 调整
    void CalSplitKBasicBlockTiling(const GMMCompileInfo* compileInfoPtr);
    // k分组/单组场景m/n固定，host直接判断l0cdb：整块超L0C半区且m超baseM时使能
    void SetL0cDbTiling(const GMMCompileInfo* compileInfoPtr);
    // m分组/不分组固定使能l0cdb：mL1/nL1记录原始baseM/baseN并减半baseN（L0块），kernel按组真实m生效
    void EnableL0cDbTiling();

    template <typename TMmTilingData>
    void FillMmTilingData(TMmTilingData& mmTilingData) const
    {
        mmTilingData.M = m_;
        mmTilingData.N = n_;
        mmTilingData.Ka = k_;
        mmTilingData.mL1 = static_cast<int32_t>(mL1_);
        mmTilingData.nL1 = static_cast<int32_t>(nL1_);
        mmTilingData.kL1 = static_cast<int32_t>(baseK_ * std::max(std::min(stepKa_, stepKb_), 1UL));
        mmTilingData.baseM = baseM_;
        mmTilingData.baseN = baseN_;
        mmTilingData.baseK = baseK_;
        mmTilingData.stepKa = stepKa_;
        mmTilingData.stepKb = stepKb_;
        mmTilingData.dbL0C = static_cast<int32_t>(dbL0C_);
    }
    template <typename TTilingData>
    bool SaveTilingData(gert::TilingContext* context, const TTilingData& tilingData) const
    {
        size_t* workspaces = context->GetWorkspaceSizes(1); // get second variable
        OP_CHECK_IF(workspaces == nullptr, OP_LOGE(context->GetNodeName(), "workspaces is nullptr."), return false);
        workspaces[0] = 32U; // 32: default workspace size

        context->SetBlockDim(usedCoreNum_);
        OP_CHECK_IF(context->GetRawTilingData() == nullptr,
                    OP_LOGE(context->GetNodeName(), "RawTilingData is nullptr."), return false);
        errno_t ret = memcpy_s(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity(),
                               reinterpret_cast<const void*>(&tilingData), sizeof(tilingData));
        if (ret != EOK) {
            OP_LOGE(context->GetNodeName(), "memcpy_s failed, ret = %d", ret);
            return false;
        }
        context->GetRawTilingData()->SetDataSize(sizeof(tilingData));
        return true;
    }

    // ---- bias 表 L1 预留：GMM 有 bias 输入时预留表空间，GMM_ADD 无 bias（默认 0） ----
    virtual uint64_t GetBiasReserveSize() const
    {
        return 0UL;
    }

    // ---- 输出张量模式：GMM 按 splitItem 解析单/多输出，GMM_ADD 输出恒为单张量 ----
    virtual bool IsSingleY() const
    {
        return true;
    }

    // ---- 公共成员 ----
    bool transposeX_ = false;
    bool transposeWeight_ = false;
    bool kZero_ = false;
    bool weightNoL2Cache_ = false;
    uint64_t m_ = 0;
    uint64_t k_ = 0;
    uint64_t n_ = 0;
    int64_t groupType_ = 0;
    uint32_t groupNum_ = 0;
    int64_t groupListType_ = 0;
    uint32_t xKDim_ = 0;
    uint32_t weightNDim_ = 0;
    uint32_t xDimNum_ = 0;
    int64_t nzFactor_ = 1; // for weight nz format
    uint64_t baseM_ = BASE_M_DEFAULT;
    uint64_t baseN_ = BASE_N_DEFAULT;
    uint64_t baseK_ = BASE_K_DEFAULT;
    // L1 tile大小，使能l0cdb时记录原始baseM/baseN（baseN_随后减半作为L0块下发）
    uint64_t mL1_ = 0UL;
    uint64_t nL1_ = 0UL;
    // L0C乒乓分段数：m分组/不分组固定为2由kernel按组使能，k分组/单组由SetL0cDbTiling静态判定
    uint64_t dbL0C_ = 1UL;
    uint32_t usedCoreNum_ = 0;
    uint64_t stepKa_ = STEP_K_DEFAULT;
    uint64_t stepKb_ = STEP_K_DEFAULT;
    uint64_t depthA1_ = DEPTH_DEFAULT;
    uint64_t depthB1_ = DEPTH_DEFAULT;
    uint64_t mTailCnt_ = 1UL;
    uint64_t nTailCnt_ = 1UL;
    ge::DataType xDType_ = ge::DT_UNDEFINED;
    ge::DataType weightDtype_ = ge::DT_UNDEFINED;

    // groupList 输入下标随算子 IR 不同（GMM: 7, GMM_ADD: 2），由派生类构造时指定
    uint32_t groupListIndex_;
};
} // namespace optiling

#endif // GROUPED_NO_QUANT_TILING_BASE_H
