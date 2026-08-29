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
 * \file ncai_case.cpp
 * \brief NativeSelectedAttention / NativeSelectedAttentionGrad 测试用例.
 */

#include "ncai_case.h"
#include <utility>
#include <tikicpulib.h>
#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "tests/utils/tests_utils_log.h"
#include "tests/utils/platform.h"
#include "op_host/tiling_base.h"

/**
 * 以下函数声明需要保持与 CMakeList.txt 中调用 OpsTest_Level2_AddOp 函数时 KERNEL_PRIVATE_COMPILE_DEFINITIONS_EXT
 * 参数所控制的 Kernel 入口一致.
 */

#define NCAI_KERNEL_PARAM                                                                                             \
    (__gm__ uint8_t * query, __gm__ uint8_t * key, __gm__ uint8_t * value, __gm__ uint8_t * attention_mask,             \
    __gm__ uint8_t * block_table, __gm__ uint8_t * actualQSeqLen, __gm__ uint8_t * actualCmpKvSeqLen,                   \
    __gm__ uint8_t * actualSelKvSeqLen, __gm__ uint8_t * topkMask, __gm__ uint8_t * output,                    \
    __gm__ uint8_t * topkIndicesOut, __gm__ uint8_t * workspace, __gm__ uint8_t * tiling)

typedef void(*NcaiKernalFunc) NCAI_KERNEL_PARAM;

extern "C" __global__ __aicore__ void nsa_compress_attention_infer NCAI_KERNEL_PARAM;

using namespace ops::adv::tests::ncaInfer;
using TensorIntf = ops::adv::tests::utils::TensorIntf;
using Case = ops::adv::tests::utils::Case;
using Platform = ops::adv::tests::utils::Platform;


bool RunNcai(void *func, uint64_t tilingKey, int64_t blockDim, std::vector<TensorIntf *> &inputs,
                       std::vector<TensorIntf *> &outputs, uint8_t *workspace, uint8_t *tilingData)
{   
    constexpr uint32_t ATTEN_OUTPUT_INDEX = 0;
    constexpr uint32_t SELECT_OUTPUT_INDEX = 1;
    constexpr uint32_t QUERY_INPUT_INDEX = 0;
    constexpr uint32_t KEY_INPUT_INDEX = 1;
    constexpr uint32_t VALUE_INPUT_INDEX = 2;
    constexpr uint32_t ATTEN_MASK_INPUT_INDEX = 3;
    constexpr uint32_t BLOCK_TABLE_INPUT_INDEX = 4;
    constexpr uint32_t ACT_Q_SEQ_LEN_INPUT_INDEX = 5;
    constexpr uint32_t ACT_CMP_KV_SEQ_LEN_INPUT_INDEX = 6;
    constexpr uint32_t ACT_SEL_KV_SEQ_LEN_INPUT_INDEX = 7;
    constexpr uint32_t TOPK_MASK_INPUT_INDEX = 8;

    // Kernel 运行
    auto kernelFunc = (NcaiKernalFunc)func;
    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(kernelFunc, blockDim,
                inputs[QUERY_INPUT_INDEX]->GetDevData(),  // query
                inputs[KEY_INPUT_INDEX]->GetDevData(),  // key
                inputs[VALUE_INPUT_INDEX]->GetDevData(),  // value
                inputs[ATTEN_MASK_INPUT_INDEX]->GetDevData(),  // attentionMaskOptional
                inputs[BLOCK_TABLE_INPUT_INDEX]->GetDevData(),  // blockTableOptional
                inputs[ACT_Q_SEQ_LEN_INPUT_INDEX]->GetDevData(),  // actualQSeqLenOptional
                inputs[ACT_CMP_KV_SEQ_LEN_INPUT_INDEX]->GetDevData(),  // actualCmpKvSeqLenOptional
                inputs[ACT_SEL_KV_SEQ_LEN_INPUT_INDEX]->GetDevData(),  // actualSelKvSeqLenOptional
                inputs[TOPK_MASK_INPUT_INDEX]->GetDevData(),  // topkMaskOptional
                outputs[ATTEN_OUTPUT_INDEX]->GetDevData(), // output
                outputs[SELECT_OUTPUT_INDEX]->GetDevData(),  // topkIndicesOut
                workspace, tilingData);
    return true;
}

extern "C" ge::graphStatus TilingNsaCompressAttentionInferStub(gert::TilingContext *context)
{   
    constexpr uint32_t ACT_Q_SEQ_LEN_INPUT_INDEX = 5;
    constexpr uint32_t ACT_CMP_KV_SEQ_LEN_INPUT_INDEX = 6;
    constexpr uint32_t ACT_SEL_KV_SEQ_LEN_INPUT_INDEX = 7;
    auto *ncaInferCase = static_cast<NcaInferCase *>(Case::GetCurrentCase());
    if (ncaInferCase != nullptr) {
        NcaInferCase::DoTilingParam p;
        p.ctx = context;
        p.ret = ge::GRAPH_SUCCESS;
        p.actQSeqLenTensor = const_cast<gert::Tensor *>(context->GetOptionalInputTensor(ACT_Q_SEQ_LEN_INPUT_INDEX));
        p.actSeqCmpKVLenTensor = const_cast<gert::Tensor *>(context->GetOptionalInputTensor(ACT_CMP_KV_SEQ_LEN_INPUT_INDEX));
        p.actSeqSelKVLenTensor = const_cast<gert::Tensor *>(context->GetOptionalInputTensor(ACT_SEL_KV_SEQ_LEN_INPUT_INDEX));
        if (!ncaInferCase->DoOpTiling(p)) {
            return p.ret;
        }
        return ncaInferCase->NcaiTilingFunc(context);
    }
    return ge::GRAPH_FAILED;
}

bool NcaInferCase::InitParam()
{   
    std::vector<int64_t> qShape;
    std::vector<int64_t> outShape;
    std::vector<int64_t> topkOutShape;
    int64_t headSizeVO = 128;
    int64_t maxBlockPerQuery = 32;
    int64_t defaultBlockNum = 1000;
    mParam.actualQSeqLenOptionalLenList = std::vector(mParam.B, static_cast<int64_t>(mParam.S1));
    mParam.actualCmpKvSeqLenOptionalLenList = std::vector(mParam.B, static_cast<int64_t>(mParam.S2));
    mParam.actualSelKvSeqLenOptionalLenList = std::vector(mParam.B, static_cast<int64_t>((mParam.S2 - 1) * mParam.compressStride + mParam.compressBlockSize));

    int64_t qSeqLenSum = 0;
    int64_t kvSeqLenSum = 0;
    for (long &it : mParam.actualQSeqLenOptionalLenList) {
        qSeqLenSum += it;
    }
    for (long &it : mParam.actualCmpKvSeqLenOptionalLenList) {
        kvSeqLenSum += it;
    }
    int64_t blockNum = (kvSeqLenSum + mParam.pageBlockSize - 1) / mParam.pageBlockSize;
    if (mParam.S2 == -1) {
        blockNum = defaultBlockNum;
        maxBlockPerQuery = (mParam.S2 + mParam.pageBlockSize - 1) / mParam.pageBlockSize;
    }

    if (mParam.layoutOptional == "TND") {
        qShape = {qSeqLenSum, mParam.N1, mParam.D};
        outShape = {mParam.B, mParam.N1, headSizeVO};
        topkOutShape = {mParam.B, mParam.N2, mParam.selectBlockCount};
    } else {
        qShape = {mParam.B, mParam.S1, mParam.N1, mParam.D};
        outShape = {mParam.B, mParam.S1, mParam.N1, headSizeVO};
        topkOutShape = {mParam.B, mParam.S1, mParam.N2, mParam.selectBlockCount};
    }
    query = Tensor("query", qShape, mParam.layoutOptional.c_str(), ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    key = Tensor("key", {blockNum, mParam.pageBlockSize, mParam.N2 * mParam.D}, mParam.layoutOptional.c_str(), ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    value = Tensor("value", {blockNum, mParam.pageBlockSize, mParam.N2 * headSizeVO}, mParam.layoutOptional.c_str(), ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    attentionMaskOptional = Tensor("attentionMaskOptional", {128, 128}, mParam.layoutOptional.c_str(), ge::DataType::DT_BOOL, ge::FORMAT_ND);
    blockTableOptional = Tensor("blockTableOptional", {mParam.B, maxBlockPerQuery}, mParam.layoutOptional.c_str(), ge::DataType::DT_INT32, ge::FORMAT_ND);
    
    actualQSeqLenOptional = Tensor("actualQSeqLenOptional", {mParam.B}, mParam.layoutOptional.c_str(), ge::DataType::DT_INT64, ge::FORMAT_ND);
    actualCmpKvSeqLenOptional = Tensor("actualCmpKvSeqLenOptional", {mParam.B}, mParam.layoutOptional.c_str(), ge::DataType::DT_INT64, ge::FORMAT_ND);
    actualSelKvSeqLenOptional = Tensor("actualSelKvSeqLenOptional", {mParam.B}, mParam.layoutOptional.c_str(), ge::DataType::DT_INT64, ge::FORMAT_ND);
    
    topkMaskOptional = Tensor("topkMaskOptional", {20, 4096}, mParam.layoutOptional.c_str(), ge::DataType::DT_BOOL, ge::FORMAT_ND);

    output = Tensor("output", outShape, mParam.layoutOptional.c_str(), ge::DataType::DT_FLOAT16, ge::FORMAT_ND);
    topkIndicesOut = Tensor("topkIndicesOut", topkOutShape, mParam.layoutOptional.c_str(), ge::DataType::DT_INT32, ge::FORMAT_ND);
    
    if (!InitTensor(actualQSeqLenOptional, mParam.actualQSeqLenOptionalLenList) || 
        !InitTensor(actualCmpKvSeqLenOptional, mParam.actualCmpKvSeqLenOptionalLenList) ||
        !InitTensor(actualSelKvSeqLenOptional, mParam.actualSelKvSeqLenOptionalLenList)) {
        return false;
    }
    return true;
}

bool NcaInferCase::InitOpInfo()
{   
    constexpr uint32_t TILING_DATA_MAX_SIZE = 2280;
    auto *ncaiKernalFunc = (void *)nsa_compress_attention_infer;

    bool rst = mCtx.SetOpName("NsaCompressAttentionInfer");
    rst = rst && mCtx.SetDeterministic(false);
    rst = rst && mCtx.SetInputs({&query, &key, &value, &attentionMaskOptional, &blockTableOptional, &actualQSeqLenOptional, &actualCmpKvSeqLenOptional, &actualSelKvSeqLenOptional, &topkMaskOptional});
    rst = rst && mCtx.SetOutputs({&output, &topkIndicesOut});
    rst = rst && mCtx.SetAttrs({{"numHeads", mParam.numHeads},
                                {"numKeyValueHeads", mParam.numKeyValueHeads},
                                {"selectBlockSize", mParam.selectBlockSize},
                                {"selectBlockCount", mParam.selectBlockCount},
                                {"compressBlockSize", mParam.compressBlockSize},
                                {"compressStride", mParam.compressStride},
                                {"scaleValue", mParam.scaleValue},
                                {"layoutOptional", mParam.layoutOptional},
                                {"pageBlockSize", mParam.pageBlockSize},
                                {"sparseMode", mParam.sparseMode}});
    rst = rst && mCtx.SetKernelRunCbf(RunNcai);
    rst = rst && mCtx.SetTilingDataMaxSize(TILING_DATA_MAX_SIZE); // max tilingDataLen
    rst = rst && mCtx.SetKernelMainFunc(ncaiKernalFunc);
    rst = rst && mOpInfo.SetContext(&mCtx);

    auto *platform = Platform::GetGlobalPlatform();
    if (platform == nullptr) {
        LOG_ERR("Global Platform is null");
        return false;
    }

    NcaiTilingFunc = (gert::OpImplRegisterV2::TilingKernelFunc)platform->LoadOpTilingSoSym("TilingNsaCompressAttentionInfer");
    if (NcaiTilingFunc == nullptr) {
        LOG_ERR("Can't get origin tiling func, ncai(%p)", NcaiTilingFunc);
        return false;
    }
    IMPL_OP(NsaCompressAttentionInfer).Tiling(TilingNsaCompressAttentionInferStub);

    return rst;
}

bool NcaInferCase::InitCurrentCasePtr()
{
    Case::mCurrentCasePtr = this;
    return true;
}

bool NcaInferCase::Run()
{
    if (!mEnable) {
        return true;
    }
    if (!mOpInfo.ProcessTiling(mName)) {
        return false;
    }
    if (!mOpInfo.ProcessKernel(mName)) {
        return false;
    }
    return true;
}

NcaInferCase::NcaInferCase(const char *name, bool enable, const char *dbgInfo, OpInfo incre, ncaInferParam param)
    : Case(name, enable, dbgInfo), mOpInfo(std::move(incre)), mParam(std::move(param))
{
    this->mOpInfo.mName = "NsaCompressAttentionInfer";
}

NcaInferCase::NcaInferCase()
{
}

NcaInferCase::ncaInferParam::ncaInferParam()
{
}

NcaInferCase::ncaInferParam::ncaInferParam(int64_t pB, int64_t pS1, int64_t pS2, int64_t pN1, int64_t pG, int64_t pD, int64_t pN2,
        int64_t pnumHeads, int64_t pnumKeyValueHeads, int64_t pselectBlockSize, 
        int64_t pselectBlockCount,int64_t pcompressBlockSize,int64_t pcompressStride, float pscaleValue, 
        std::string playoutOptional, int64_t ppageBlockSize, int64_t pSparseMode)
    : B(pB), S1(pS1), S2(pS2), N1(pN1), G(pG), D(pD), N2(pN2), numHeads(pnumHeads),
    numKeyValueHeads(pnumKeyValueHeads), selectBlockSize(pselectBlockSize), selectBlockCount(pselectBlockCount), 
    compressBlockSize(pcompressBlockSize), compressStride(pcompressStride), scaleValue(pscaleValue), 
    layoutOptional(playoutOptional), pageBlockSize(ppageBlockSize), sparseMode(pSparseMode)
{
}

bool NcaInferCase::DoOpTiling(DoTilingParam& tilingParam) {
    if (tilingParam.ctx == nullptr) {
        return false;
    }
    if (tilingParam.actQSeqLenTensor != nullptr && mParam.actualQSeqLenOptionalLenList.size() != 0) {
        tilingParam.actQSeqLenTensor->SetData(gert::TensorData{mParam.actualQSeqLenOptionalLenList.data()});
    }
    if (tilingParam.actSeqCmpKVLenTensor != nullptr && mParam.actualCmpKvSeqLenOptionalLenList.size() != 0) {
        tilingParam.actSeqCmpKVLenTensor->SetData(gert::TensorData{mParam.actualCmpKvSeqLenOptionalLenList.data()});
    }
    if (tilingParam.actSeqSelKVLenTensor != nullptr && mParam.actualSelKvSeqLenOptionalLenList.size() != 0) {
        tilingParam.actSeqSelKVLenTensor->SetData(gert::TensorData{mParam.actualSelKvSeqLenOptionalLenList.data()});
    }
    return true;
}