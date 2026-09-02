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
 * \file matmul_reduce_scatter_tiling_base_v2.h
 * \brief
 */
#ifndef __MATMUL_REDUCE_SCATTER_TILING_BASE_V2_H__
#define __MATMUL_REDUCE_SCATTER_TILING_BASE_V2_H__

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "op_host/tiling_base.h"
#include "op_host/op_tiling/mc2_tiling_utils.h"
#include "op_host/op_tiling/matmul_formulaic_tiling.h"
#include "../../op_kernel/matmul_reduce_scatter_v2_c_tiling.h"

namespace optiling {
// Input
constexpr size_t INPUT_X1 = 0;
constexpr size_t INPUT_X2 = 1;
constexpr size_t BIAS = 2;
constexpr size_t SCALE_INV1 = 3;
constexpr size_t SCALE_INV2 = 4;
constexpr size_t QUANT_SCALE = 5;

// Output
constexpr size_t OUTPUT_Y = 0;
constexpr size_t OUTPUT_AMAX = 2;

// Attr
constexpr size_t GROUPSIZE_IDX = 7;

// Dtype
constexpr std::initializer_list<ge::DataType> MXDTYPE_SUPPORT_LIST = {
    ge::DataType::DT_FLOAT8_E4M3FN, ge::DataType::DT_FLOAT8_E5M2, ge::DataType::DT_FLOAT4_E2M1};

// HcclLimit
constexpr static uint64_t REDUCE_SCATTER_V2_HCCL_NUM_LIMIT = 63;

class MatmulReduceScatterTilingBase : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit MatmulReduceScatterTilingBase(gert::TilingContext *context)
        : TilingBaseClass(context)
    {}
    ~MatmulReduceScatterTilingBase() override = default;

    void Reset(gert::TilingContext *context) override
    {
        TilingBaseClass::Reset(context);
        Reset();
    }

protected:
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoLibApiTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    uint64_t GetTilingKey() const override;

    // tiling
    void DoFormulaticTiling(Mc2Tiling::RCSTiling &rcsCfg);
    void SetStorageAWorkSpaceSize(Mc2Tiling::RCSTiling &rcsCfg);
    void SetRcsTilingData(Mc2Tiling::RCSTiling &rcsCfg);
    uint32_t ReduceScatterSpliteM(mc2tiling::TilingArgs &args, uint32_t maxTileCnt = 64) const;
    ge::graphStatus DoSplitMTiling(Mc2Tiling::RCSTiling &rcsCfg);
    virtual CutResult GetTilingResult();
    uint32_t GetRankSize(const char *group) const;
    void Reset();
    bool ReduceScatterCheckShapeInfo();
    bool CheckInputScale() const;
    bool CheckGroupSize() const;
    bool CheckBias() const;
    bool CheckAttrInfoValid(uint64_t kValue);
    void SetReduceScatterTilingArgsDataType();
    void SetReduceScatterTilingArgsShapeInfo();
    void SetReduceScatterTilingArgsBasicInfo();
    void SetReduceScatterTilingArgs();
    ge::graphStatus SetReduceScatterTilingArgsCommAlgo();
    virtual ge::graphStatus CheckInput()
    {
        return ge::GRAPH_SUCCESS;
    }
    inline mc2tiling::TilingArgs GetMc2tilingArgs()
    {
        return args_;
    };
    void SetTilingResult(Mc2Tiling::RCSTiling &rcsCfg, ::TCubeTiling &mmTiling, ::TCubeTiling &tailTiling,
                         uint32_t &debugMode, uint32_t &dataType);
    void SetMsgDataInfo(const Mc2Tiling::RCSTiling &rcsCfg, const ::TCubeTiling &mmTiling,
                        const ::TCubeTiling &tailTiling, uint32_t debugMode);
    ge::graphStatus CheckHCCLSize();
    ge::graphStatus AdjustHCCLLimit(Mc2Tiling::RCSTiling &rcsCfg, mc2tiling::Mc2QuantMode quantMmMode);

    mc2tiling::TilingArgs args_;
    platform_ascendc::SocVersion socVersion_;
    NpuArch npuArch_;
    bool isA2APath_;
    const char *opName_ = nullptr;
    int64_t rankSize_{0};
    uint8_t commMode_{0};
    uint64_t tileMValue_{0};  // mc2 切块后主块M的大小；
    uint64_t tailMValue_{0};  // mc2 切块后尾块M的大小；
    uint64_t longTileLen_{0}; // mc2 切块后长块的大小；
    uint64_t mmResultLen_{0};
    uint32_t libApiWorkSpaceSize_{0};
};
} // namespace optiling

#endif //__MATMUL_REDUCE_SCATTER_TILING_H__
