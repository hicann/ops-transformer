/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file swin_attention_ffn_tiling.cpp
 * \brief
 */
#include "swin_attention_ffn_tiling.h"
#include "register/op_def_registry.h"
#include "platform/platform_info.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "log/log.h"
#include "tiling_base/tiling_templates_registry.h"
#include "tiling_base/tiling_base.h"
#include "util/math_util.h"

#include <cmath>
#include <iostream>

using namespace std;
using namespace matmul_tiling;
using namespace ge;
using namespace AscendC;

namespace optiling
{
#define USE_CORE_THRESHOLD 24 // Use 24 aicore in 910B
#define TRANSPOSE_BLOCK_SIZE (8 * 128)
#define TRANSPOSE_SPACE_WIDTH 8
#define MATMUL_SIZE 512
#define DIM_M 1
#define DIM_K 2
#define DIM_N 1
#define WORKSPACE_SIZE (100 * 1024 * 1024)

const std::string OP_NAME = "SwinAttentionFFN";

class SAFFNTilingComp
{
    public:
        SwinAttentionFFNTilingData tiling;
        ge::graphStatus SAFFNTilingProc(gert::TilingContext* context);
    protected:
        uint32_t m;
        uint32_t n;
        uint32_t k;
        uint32_t bm;
        uint32_t batchSize;
        uint32_t bmmFormerNum;
        uint32_t bmmTailNum;
        uint32_t bmmFormerBatchNum;
        uint32_t bmmTailBatchNum;
        uint32_t dataNumPerBatchA;
        uint32_t dataNumPerBatchD;
        uint32_t dataNumPerLoop;

        uint32_t tpBlockSize;
        uint32_t tpSpaceCnt;
        uint32_t tpSpaceH;
        uint32_t tpSpaceW;
        uint32_t blockInSpace;
        uint32_t tpSpaceSize;
        uint32_t tpSpaceWTransposed;
        uint32_t tpSpaceHTransposed;

        uint32_t shift1;
        uint32_t shift2;
        uint32_t aivNum;
        uint32_t aicNum;

        uint64_t ubSize;
        uint64_t l1Size;
        uint64_t l0cSize;

        ge::graphStatus SAFFNTilingMM();
        ge::graphStatus SAFFNTilingTPROLL(const gert::TilingContext* context);
        ge::graphStatus SAFFNTilingPrint() const;
};

ge::graphStatus SAFFNTilingComp::SAFFNTilingMM() {
    m = MATMUL_SIZE;
    batchSize = bm / MATMUL_SIZE;
    dataNumPerBatchD = m * n;
    dataNumPerBatchA = m * k;
    bmmFormerNum = batchSize % aivNum;
    bmmTailNum = aivNum - bmmFormerNum;
    bmmTailBatchNum = batchSize / aivNum;
    bmmFormerBatchNum = bmmTailBatchNum + 1;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SAFFNTilingComp::SAFFNTilingTPROLL(const gert::TilingContext* context) {
    auto compileInfo = static_cast<const SwinAttentionFFNCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);

    uint32_t tmpAicNum = uint32_t(compileInfo->aicNum);
    aicNum = tmpAicNum > USE_CORE_THRESHOLD ? USE_CORE_THRESHOLD : tmpAicNum;
    aivNum = aicNum * 2; // aiv:aic == 2:1 in 910B
    ubSize = compileInfo->ubSize;
    l1Size = compileInfo->l1Size;
    l0cSize = compileInfo->l0cSize;

    const gert::RuntimeAttrs* runtimeAttrs = context->GetAttrs();
    const gert::TypedContinuousVector<int64_t> *shifts = runtimeAttrs->GetListInt(0);
    shift1 =  uint32_t(*shifts->GetData());
    shift2 =  uint32_t(*(shifts->GetData() + 1));

    const gert::StorageShape* shapeA = context->GetInputShape(0);
    const gert::StorageShape* shapeB = context->GetInputShape(1);
    batchSize = shapeA->GetStorageShape().GetDim(0);

    m = shapeA->GetStorageShape().GetDim(DIM_M);
    k = shapeA->GetStorageShape().GetDim(DIM_K);
    n = shapeB->GetStorageShape().GetDim(DIM_N);
    bm = batchSize * m;

    tpBlockSize = TRANSPOSE_BLOCK_SIZE;
    tpSpaceW = TRANSPOSE_SPACE_WIDTH;
    tpSpaceCnt = tpSpaceH = static_cast<uint32_t>(sqrt((batchSize * m * k) / (tpSpaceW * tpBlockSize)));
    blockInSpace = tpSpaceH * tpSpaceW;
    tpSpaceSize = blockInSpace * tpBlockSize;
    tpSpaceWTransposed = tpSpaceH;
    tpSpaceHTransposed = tpSpaceW;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SAFFNTilingComp::SAFFNTilingPrint() const {
    OP_LOGI(
        OP_NAME.c_str(),
        "tpBlockSize: %d, tpSpaceCnt: %d, tpSpaceH: %d, tpSpaceW: %d, blockInSpace: %d, tpSpaceSize: %d\n\
        tpSpaceWTransposed: %d, tpSpaceHTransposed: %d, GetTiling success!, ubSize: %zu, l1Size: %zu\n\
        l0cSize: %zu, TCubeTiling size: %zu, m: %d, k: %d, n: %d, aicNum: %d, aivNum: %d\n\
        batchSize: %d, bmmFormerNum: %d, bmmTailNum: %d, bmmFormerBatchNum: %d, bmmTailBatchNum: %d\n\
        dataNumPerLoop: %d, dataNumPerBatchA: %d, dataNumPerBatchD: %d\n",
        tpBlockSize, tpSpaceCnt, tpSpaceH, tpSpaceW, blockInSpace, tpSpaceSize, tpSpaceWTransposed,
        tpSpaceHTransposed, ubSize, l1Size, l0cSize, tiling.bmmTilingData.GetDataSize(), m, k, n,
        aicNum, aivNum, batchSize, bmmFormerNum, bmmTailNum, bmmFormerBatchNum, bmmTailBatchNum,
        dataNumPerLoop, dataNumPerBatchA, dataNumPerBatchD
    );
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SAFFNTilingComp::SAFFNTilingProc(gert::TilingContext *context) {
    SAFFNTilingTPROLL(context);
    SAFFNTilingMM();
    tiling.set_batchSize(batchSize);
    tiling.set_bmmFormerNum(bmmFormerNum);
    tiling.set_bmmTailNum(bmmTailNum);
    tiling.set_bmmFormerBatchNum(bmmFormerBatchNum);
    tiling.set_bmmTailBatchNum(bmmTailBatchNum);
    tiling.set_dataNumPerBatchA(dataNumPerBatchA);
    tiling.set_dataNumPerBatchD(dataNumPerBatchD);
    tiling.set_aivNum(aivNum);
    tiling.set_tpBlockSize(tpBlockSize);
    tiling.set_tpSpaceCnt(tpSpaceCnt);
    tiling.set_tpSpaceH(tpSpaceH);
    tiling.set_tpSpaceW(tpSpaceW);
    tiling.set_blockInSpace(blockInSpace);
    tiling.set_tpSpaceSize(tpSpaceSize);
    tiling.set_tpSpaceWTransposed(tpSpaceWTransposed);
    tiling.set_tpSpaceHTransposed(tpSpaceHTransposed);
    tiling.set_shift1(shift1);
    tiling.set_shift2(shift2);

    MatmulApiTiling bmm;
    bmm.SetAType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
    bmm.SetBType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
    bmm.SetCType(TPosition::VECCALC, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
    bmm.SetBiasType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
    bmm.SetShape(m, n, k);
    bmm.SetOrgShape(m, n, k);
    bmm.SetBias(true);
    bmm.SetTraverse(MatrixTraverse::FIRSTN);

    bmm.GetTiling(tiling.bmmTilingData);
    dataNumPerLoop = bmm.GetBaseM() * bmm.GetBaseN();
    tiling.set_dataNumPerLoop(dataNumPerLoop);
    SAFFNTilingPrint();

    context->SetBlockDim(aicNum);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t* workspaces = context->GetWorkspaceSizes(1);
    workspaces[0] = WORKSPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingFuncForSwinAttentionFFN(gert::TilingContext* context)
{
    SAFFNTilingComp tilingCompute;
    auto ret = tilingCompute.SAFFNTilingProc(context);
    return ret;
}

static ge::graphStatus TilingPrepareForSwinAttentionFFN(gert::TilingParseContext* context) {
    // auto compileInfo = GetCompileInfoPtr<SwinAttentionFFNCompileInfo>(context);
    auto compileInfo = context->GetCompiledInfo<SwinAttentionFFNCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto acInfo = platform_ascendc::PlatformAscendC(platformInfo);

    uint64_t aicNum = 0;
    uint64_t ubSize = 0;
    uint64_t l1Size = 0;
    uint64_t l0cSize = 0;

    aicNum = static_cast<uint64_t>(acInfo.GetCoreNumAic());
    acInfo.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    acInfo.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1Size);
    acInfo.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0cSize);
    compileInfo->aicNum = aicNum;
    compileInfo->ubSize = ubSize;
    compileInfo->l1Size = l1Size;
    compileInfo->l0cSize = l0cSize;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(SwinAttentionFFN)
.Tiling(TilingFuncForSwinAttentionFFN)
.TilingParse<SwinAttentionFFNCompileInfo>(TilingPrepareForSwinAttentionFFN); // 向框架注册入口函数

}