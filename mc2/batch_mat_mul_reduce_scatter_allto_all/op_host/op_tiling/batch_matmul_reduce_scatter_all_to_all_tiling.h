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
 * \file batch_matmul_reduce_scatter_all_to_all_tiling.h
 * \brief
 */

#ifndef __BATCH_MATMUL_REDUCE_SCATTER_ALL_TO_ALL_TILING_H__
#define __BATCH_MATMUL_REDUCE_SCATTER_ALL_TO_ALL_TILING_H__

#pragma once
#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"
#include "op_host/op_tiling/mc2_tiling_struct.h"
#include "../../op_kernel/batch_mat_mul_reduce_scatter_allto_all_tiling_struct.h"
#include "../../../3rd/batch_mat_mul_v3/op_host/op_tiling/batch_mat_mul_v3_base_tiling.h"

namespace optiling {
struct ReduceScatterAlltoAllBatchInfo {
    uint32_t batchA = 1;
    uint32_t batchA0 = 1;
    uint32_t batchA1 = 1;
    uint32_t batchA2 = 1;
    uint32_t batchA3 = 1;
    uint32_t batchB = 1;
    uint32_t batchB0 = 1;
    uint32_t batchB1 = 1;
    uint32_t batchB2 = 1;
    uint32_t batchB3 = 1;
    uint32_t batchC = 1;
    uint32_t batchC0 = 1;
    uint32_t batchC1 = 1;
    uint32_t batchC2 = 1;
    uint32_t batchC3 = 1;
    bool biasWithBatch = false;
};

struct ReduceScatterAlltoAllMatmulInfo {
    const char *opName = nullptr;
    bool isWeightTrans = false;
    bool isBias = false;
    ge::DataType aType = ge::DT_FLOAT16;
    ge::DataType bType = ge::DT_FLOAT16;
    ge::DataType cType = ge::DT_FLOAT16;
    ge::DataType biasType = ge::DT_FLOAT16;
    uint64_t mValue = 0L;
    uint64_t kValue = 0L;
    uint64_t nValue = 0L;
};

class BatchMatMulReduceScatterAlltoAllTiling : public Mc2batch_mat_mul_v3::Mc2BatchMatmulV3BaseTiling {
public:
    BatchMatMulReduceScatterAlltoAllTiling(gert::TilingContext *context, Mc2BatchMatmulTilingData &bmmTilingData,
                                           ReduceScatterAlltoAllBatchInfo &BMMV3BatchInfo,
                                           ReduceScatterAlltoAllMatmulInfo &MMV3ArgsInfo)
        : Mc2BatchMatmulV3BaseTiling(context, bmmTilingData),
          BMMV3BatchInfo_(BMMV3BatchInfo),
          MMV3ArgsInfo_(MMV3ArgsInfo)
    {}

    ge::graphStatus GetShapeAttrsInfo() override
    {
        args_.opName = MMV3ArgsInfo_.opName;
        args_.isBTrans = MMV3ArgsInfo_.isWeightTrans;
        args_.hasBias = false; // 出于性能考虑，当前bias计算在外部进行，算子内不会有bias场景
        args_.aType = MMV3ArgsInfo_.aType;
        args_.bType = MMV3ArgsInfo_.bType;
        args_.cType = MMV3ArgsInfo_.cType;
        args_.biasType = MMV3ArgsInfo_.biasType;

        args_.aFormat = ge::FORMAT_ND;
        args_.bFormat = ge::FORMAT_ND;
        args_.outFormat = ge::FORMAT_ND;

        args_.mValue = MMV3ArgsInfo_.mValue;
        args_.kValue = MMV3ArgsInfo_.kValue;
        args_.nValue = MMV3ArgsInfo_.nValue;

        OP_LOGD(args_.opName, " args_.mValue %lu", args_.mValue);
        OP_LOGD(args_.opName, " args_.kValue %lu", args_.kValue);
        OP_LOGD(args_.opName, " args_.nValue %lu", args_.nValue);

        OP_LOGD("BatchMatMulReduceScatterAlltoAllTiling", " batchA %u", BMMV3BatchInfo_.batchA);
        OP_LOGD("BatchMatMulReduceScatterAlltoAllTiling", " batchA0 %u", BMMV3BatchInfo_.batchA0);
        OP_LOGD("BatchMatMulReduceScatterAlltoAllTiling", " batchA1 %u", BMMV3BatchInfo_.batchA1);
        OP_LOGD("BatchMatMulReduceScatterAlltoAllTiling", " batchA2 %u", BMMV3BatchInfo_.batchA2);
        OP_LOGD("BatchMatMulReduceScatterAlltoAllTiling", " batchA3 %u", BMMV3BatchInfo_.batchA3);
        OP_LOGD("BatchMatMulReduceScatterAlltoAllTiling", " batchB %u", BMMV3BatchInfo_.batchB);
        OP_LOGD("BatchMatMulReduceScatterAlltoAllTiling", " batchB0 %u", BMMV3BatchInfo_.batchB0);
        OP_LOGD("BatchMatMulReduceScatterAlltoAllTiling", " batchB1 %u", BMMV3BatchInfo_.batchB1);
        OP_LOGD("BatchMatMulReduceScatterAlltoAllTiling", " batchB2 %u", BMMV3BatchInfo_.batchB2);
        OP_LOGD("BatchMatMulReduceScatterAlltoAllTiling", " batchB3 %u", BMMV3BatchInfo_.batchB3);
        OP_LOGD("BatchMatMulReduceScatterAlltoAllTiling", " batchC %u", BMMV3BatchInfo_.batchC);
        OP_LOGD("BatchMatMulReduceScatterAlltoAllTiling", " batchC0 %u", BMMV3BatchInfo_.batchC0);
        OP_LOGD("BatchMatMulReduceScatterAlltoAllTiling", " batchC1 %u", BMMV3BatchInfo_.batchC1);
        OP_LOGD("BatchMatMulReduceScatterAlltoAllTiling", " batchC2 %u", BMMV3BatchInfo_.batchC2);
        OP_LOGD("BatchMatMulReduceScatterAlltoAllTiling", " batchC3 %u", BMMV3BatchInfo_.batchC3);

        batchInfo_.batchA = BMMV3BatchInfo_.batchA;
        batchInfo_.batchA0 = BMMV3BatchInfo_.batchA0;
        batchInfo_.batchA1 = BMMV3BatchInfo_.batchA1;
        batchInfo_.batchA2 = BMMV3BatchInfo_.batchA2;
        batchInfo_.batchA3 = BMMV3BatchInfo_.batchA3;
        batchInfo_.batchB = BMMV3BatchInfo_.batchB;
        batchInfo_.batchB0 = BMMV3BatchInfo_.batchB0;
        batchInfo_.batchB1 = BMMV3BatchInfo_.batchB1;
        batchInfo_.batchB2 = BMMV3BatchInfo_.batchB2;
        batchInfo_.batchB3 = BMMV3BatchInfo_.batchB3;
        batchInfo_.batchC = BMMV3BatchInfo_.batchC;
        batchInfo_.batchC0 = BMMV3BatchInfo_.batchC0;
        batchInfo_.batchC1 = BMMV3BatchInfo_.batchC1;
        batchInfo_.batchC2 = BMMV3BatchInfo_.batchC2;
        batchInfo_.batchC3 = BMMV3BatchInfo_.batchC3;
        batchInfo_.biasWithBatch = false;
        return ge::GRAPH_SUCCESS;
    }

private:
    ReduceScatterAlltoAllBatchInfo BMMV3BatchInfo_;
    ReduceScatterAlltoAllMatmulInfo MMV3ArgsInfo_;
};
} // namespace optiling

#endif //__BATCH_MATMUL_REDUCE_SCATTER_ALL_TO_ALL_TILING_H__
