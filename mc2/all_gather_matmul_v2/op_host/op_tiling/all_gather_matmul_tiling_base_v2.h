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
 * \file all_gather_matmul_tiling_base_v2.h
 * \brief
 */
#ifndef __ALL_GATHER_MATMUL_TILING_BASE_V2__
#define __ALL_GATHER_MATMUL_TILING_BASE_V2__

#pragma once
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "weight_quant_batch_matmul_v2/op_host/op_tiling/weight_quant_batch_matmul_v2_tiling.h"
#include "quant_batch_matmul_v3/op_host/op_tiling/quant_batch_matmul_v3_tiling.h"
#include "op_host/op_tiling/mc2_tiling_struct.h"
#include "op_host/op_tiling/matmul_formulaic_tiling.h"
#include "mat_mul_v3/op_host/op_tiling/matmul_v3_tiling.h"
#include "op_host/op_tiling/mc2_tiling_utils.h"
#include "../../op_kernel/all_gather_matmul_tiling_data.h"

namespace optiling {
constexpr size_t INPUT_X1 = 0;
constexpr size_t INPUT_X2 = 1;
constexpr size_t BIAS = 2;
constexpr size_t SCALE_INV1 = 3;
constexpr size_t SCALE_INV2 = 4;
constexpr size_t SCALE = 5;
constexpr size_t CONTEXT = 6;

constexpr size_t OUTPUT_Y = 0;
constexpr size_t GATHER_OUT = 1;
constexpr size_t OUTPUT_AMAX = 2;

constexpr size_t GROUP = 0;
constexpr size_t IS_TRANS_A = 1;
constexpr size_t IS_TRANS_B = 2;
constexpr size_t GATHER_IDX = 3;
constexpr size_t COMM_TURN = 4;
constexpr size_t RANK_SIZE = 5;
constexpr size_t GROUPSIZE_INDEX = 7;
constexpr size_t IS_GATHER_OUT = 8;
constexpr size_t IS_AMAX_OUT = 9;
constexpr size_t Y_DTYPE = 10;
constexpr size_t COMM_MODE_INDEX = 11;

constexpr uint64_t PERBLOCK_SCALE_SIZE = 128;
constexpr uint32_t COMM_VERSION3 = 3U;
constexpr uint64_t HCCL_MAX_COMM_TILES = 62; // 总通信次数上限63, tile+tail最大62, 1个保留给x1Scale通信的情况

class AllGatherMatmulTilingBase : public TilingBaseClass {
public:
    explicit AllGatherMatmulTilingBase(gert::TilingContext *context)
        : TilingBaseClass(context)
    {}
    ~AllGatherMatmulTilingBase() override = default;

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

    // L2tiling不是所有的子类都有，所以子类中实现
    ge::graphStatus AnalyzeShapeAttr();
    bool AnalyzeAttrs();
    bool AnalyzeInputs();
    bool SetCommAlgo();
    // tiling
    void DoAllGatherTiling(Mc2Tiling::RCSTiling &rcsCfg, ::TCubeTiling &mmTiling, ::TCubeTiling &tailTiling,
                           uint32_t &debugMode, uint32_t &dataType);
    void SetRcsTilingData(Mc2Tiling::RCSTiling &rcsCfg);
    void DoSplitMTiling(Mc2Tiling::RCSTiling &rcfCfg);
    void PostDoSplitMTiling(Mc2Tiling::RCSTiling &rcfCfg, mc2tiling::Mc2QuantMode quantMmMode);
    virtual CutResult GetTilingResult();
    virtual ge::graphStatus CheckInput()
    {
        return ge::GRAPH_SUCCESS;
    }

    void Reset();

    mc2tiling::HcclDataType GetDataType(ge::DataType type);
    uint32_t AllGatherSplitM(mc2tiling::TilingArgs &args, uint32_t maxTileCnt);
    uint64_t GetStorageA(Mc2Tiling::RCSTiling &rcsCfg);
    uint64_t CalcGatherLen(uint32_t dimA, uint32_t dimB, uint64_t alignAddrLen);
    bool CheckInputParaEmptyPointer();
    bool CheckInputScale();
    bool CheckGroupSize();
    bool CheckInputParaArraySize();
    bool CheckInputAndOutputParaFormat();
    bool CheckGatherOutPara();
    bool CheckOutputParaDim0();
    bool CheckBiasParaDim0();
    bool CheckParaInvaild();
    void SetTilingArgsDim();
    void SetTilingArgsDataType();
    void SetTilingArgsGatherStatus();
    void SetMC2AllGatherDataInfo(Mc2Tiling::RCSTiling &rcsCfg, ::TCubeTiling &mmTiling, ::TCubeTiling &tailTiling,
                                 uint32_t debugMode);
    ge::graphStatus CheckHCCLSize();

    mc2tiling::TilingArgs args_;
    NpuArch npuArch_;
    const char *opName_{nullptr};
    const char *group_{nullptr};
    const char *commMode_{nullptr};
    uint64_t tileMValue_{0};
    uint64_t tailMValue_{0};
    uint64_t drMValue_{0};
    int64_t rankSize_{0};
    uint32_t biasLen_{0};
    uint32_t storageA_{0};
    uint32_t libApiWorkSpaceSize_{0};
    bool outputIsFp8_{false};
    bool inputIsBf16Fp16_{false};
    uint8_t commAlgorithm_{0};
    bool enableNd2Nz_{false};
    bool castBias_{false};
    uint32_t gatherIndex_{0};
};
} // namespace optiling

#endif // __ALL_GATHER_MATMUL_TILING_BASE__
