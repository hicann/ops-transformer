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
 * \file pfa_case.cpp
 * \brief PromptFlashAttention 测试用例.
 */
#include "pfa_case.h"
#include <utility>
#include <tikicpulib.h>
#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "tests/utils/log.h"
#include "tests/utils/platform.h"
#include "tiling/pfa/tiling_data.h"
#include "tiling/pfa/tiling_stub.h"
#include "../../op_kernel/prompt_flash_attention.cpp"

/**
 * 以下函数声明需要保持与 CMakeList.txt 中调用 OpsTest_Level2_AddOp 函数时 KERNEL_PRIVATE_COMPILE_DEFINITIONS_EXT
 * 参数所控制的 Kernel 入口一致.
 */

#define PFA_KERNEL_PARAM                                                                                               \
    (__gm__ uint8_t * query, __gm__ uint8_t * key, __gm__ uint8_t * value, __gm__ uint8_t * pseShift,                  \
     __gm__ uint8_t * attenMask, __gm__ uint8_t * actualSeqLengths, __gm__ uint8_t * actualSeqLengthsKV,               \
     __gm__ uint8_t * deq_scale1, __gm__ uint8_t * quant_scale1, __gm__ uint8_t * deq_scale2,                          \
     __gm__ uint8_t * quant_scale2, __gm__ uint8_t * quant_offset2, __gm__ uint8_t * learnableSink,                    \
     __gm__ uint8_t * attentionOut, __gm__ uint8_t * workspace, __gm__ uint8_t * tiling)

typedef void(*PfaKernelFunc) PFA_KERNEL_PARAM;

// extern "C" __global__ __aicore__ void prompt_flash_attention PFA_KERNEL_PARAM;

using namespace ops::adv::tests::pfa;
using Tensor = ops::adv::tests::utils::TensorIntf;
using Case = ops::adv::tests::utils::Case;
using Platform = ops::adv::tests::utils::Platform;

bool RunTemplatePromptFlashAttention(std::function<void(PFA_INPUT_DTYPE)> func,
                           uint64_t tilingKey, int64_t blockDim, std::vector<Tensor *> &inputs,
                           std::vector<Tensor *> &outputs, uint8_t *workspace, uint8_t *tilingData)
{
    // Kernel 运行
    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(func, blockDim, inputs[0]->GetDevData(), inputs[1]->GetDevData(), inputs[2]->GetDevData(),
                inputs[3]->GetDevData(), inputs[4]->GetDevData(), inputs[5]->GetDevData(), inputs[6]->GetDevData(),
                inputs[7]->GetDevData(), inputs[8]->GetDevData(), inputs[9]->GetDevData(), inputs[10]->GetDevData(),
                inputs[11]->GetDevData(), inputs[12]->GetDevData(), outputs[0]->GetDevData(), workspace, tilingData);
    return true;
}

bool RunPromptFlashAttention(void *func, uint64_t tilingKey, int64_t blockDim, std::vector<Tensor *> &inputs,
                             std::vector<Tensor *> &outputs, uint8_t *workspace, uint8_t *tilingData)
{
    // Kernel 运行
    auto kernelFunc = (PfaKernelFunc)func;
    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(kernelFunc, blockDim, inputs[0]->GetDevData(), inputs[1]->GetDevData(), inputs[2]->GetDevData(),
                inputs[3]->GetDevData(), inputs[4]->GetDevData(), inputs[5]->GetDevData(), inputs[6]->GetDevData(),
                inputs[7]->GetDevData(), inputs[8]->GetDevData(), inputs[9]->GetDevData(), inputs[10]->GetDevData(),
                inputs[11]->GetDevData(), inputs[12]->GetDevData(), outputs[0]->GetDevData(), workspace, tilingData);
    return true;
}

extern "C" ge::graphStatus TilingPromptFlashAttentionStub(gert::TilingContext* context)
{
    auto* pfaCase = static_cast<PfaCase*>(Case::GetCurrentCase());
    if (pfaCase != nullptr) {
        PfaCase::DoTilingParam p;
        p.ctx = context;
        p.ret = ge::GRAPH_SUCCESS;
        p.actualSeqLengthsTensor = const_cast<gert::Tensor*>(context->GetOptionalInputTensor(5)); // 5: the index of actual seqlength
        p.actualSeqLengthsKVTensor = const_cast<gert::Tensor*>(context->GetOptionalInputTensor(6)); // 6: the index of actual kvseqlength
        if (!pfaCase->DoOpTiling(p)) {
            return p.ret;
        }
        return pfaCase->pfaTilingFunc(context);
    }
    return ge::GRAPH_FAILED;
}

bool PfaCase::InitParam()
{
    h = mParam.n * mParam.d;
    int64_t kvNum = mParam.n;
    if (mParam.kvNumHeads != 0) {
        kvNum = mParam.kvNumHeads;
    }
    int64_t kvH = kvNum * mParam.d;
    if (mParam.layout == "BSH") {
        query = Tensor("query", {mParam.b, mParam.s, h}, "BSH", mParam.qDataType, ge::FORMAT_ND);
        key = Tensor("key", {mParam.b, mParam.s, kvH}, "BSH", mParam.kvDataType, ge::FORMAT_ND);
        value = Tensor("value", {mParam.b, mParam.s, kvH}, "BSH", mParam.kvDataType, ge::FORMAT_ND);
        attentionOut = Tensor("attentionOut", {mParam.b, mParam.s, h}, "BSH", mParam.outDataType, ge::FORMAT_ND);
    } else if (mParam.layout == "BNSD") {
        query = Tensor("query", {mParam.b, mParam.n, mParam.s, mParam.d}, "BNSD", mParam.qDataType, ge::FORMAT_ND);
        key = Tensor("key", {mParam.b, kvNum, mParam.s, mParam.d}, "BNSD", mParam.kvDataType, ge::FORMAT_ND);
        value = Tensor("value", {mParam.b, kvNum, mParam.s, mParam.d}, "BNSD", mParam.kvDataType, ge::FORMAT_ND);
        attentionOut =
            Tensor("attentionOut", {mParam.b, mParam.n, mParam.s, mParam.d}, "BNSD", mParam.outDataType, ge::FORMAT_ND);
    } else if (mParam.layout == "BSND") {
        query = Tensor("query", {mParam.b, mParam.s, mParam.n, mParam.d}, "BSND", mParam.qDataType, ge::FORMAT_ND);
        key = Tensor("key", {mParam.b, mParam.s, kvNum, mParam.d}, "BSND", mParam.kvDataType, ge::FORMAT_ND);
        value = Tensor("value", {mParam.b, mParam.s, kvNum, mParam.d}, "BSND", mParam.kvDataType, ge::FORMAT_ND);
        attentionOut =
            Tensor("attentionOut", {mParam.b, mParam.s, mParam.n, mParam.d}, "BSND", mParam.outDataType, ge::FORMAT_ND);
    }
    if (mParam.attenMaskType == AttenMaskShapeType::B_N_1_S) {
        attenMask =
            Tensor("attenMask", {mParam.b, mParam.n, 1, mParam.s}, "B_N_1_S", ge::DataType::DT_BOOL, ge::FORMAT_ND);
    } else if (mParam.attenMaskType == AttenMaskShapeType::B_1_S) {
        attenMask = Tensor("attenMask", {mParam.b, 1, mParam.s}, "B_1_S", ge::DataType::DT_BOOL, ge::FORMAT_ND);
    }

    if (mParam.actualSeqLength.size() != 0) {
        actualSeqLengths = Tensor("actualSeqLengths", {mParam.b}, "B", ge::DataType::DT_INT64, ge::FORMAT_ND);
    }
    if (mParam.actualSeqLengthKV.size() != 0) {
        actualSeqLengthsKV = Tensor("actualSeqLengthsKV", {mParam.b}, "B", ge::DataType::DT_INT64, ge::FORMAT_ND);
    }

    if (mParam.quantType == QuantShapeType::ALL_1) {
        deqScale1 = Tensor("deqScale1", {1}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND);
        quantScale1 = Tensor("quantScale1", {1}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND);
        deqScale2 = Tensor("deqScale2", {1}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND);
        quantScale2 = Tensor("quantScale2", {1}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND);
        quantOffset2 = Tensor("quantOffset2", {1}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND);
    } else if (mParam.quantType == QuantShapeType::PER_1) {
        deqScale1 = Tensor("deqScale1", {1}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND);
        quantScale1 = Tensor("quantScale1", {1}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND);
        deqScale2 = Tensor("deqScale2", {1}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND);
    } else if (mParam.quantType == QuantShapeType::POST_1) {
        quantScale2 = Tensor("quantScale2", {1}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND);
        quantOffset2 = Tensor("quantOffset2", {1}, "1", ge::DataType::DT_FLOAT, ge::FORMAT_ND);
    }
    if (mParam.hasLearnableSink) {
        learnableSink = Tensor("learnableSink", {mParam.n}, "N", mParam.qDataType, ge::FORMAT_ND);
    }
    return true;
}

bool PfaCase::InitOpInfo()
{
    bool rst = mCtx.SetOpName("PromptFlashAttention");
    rst = rst && mCtx.SetDeterministic(false);
    rst = rst && mCtx.SetInputs({&query, &key, &value, &pseShift, &attenMask, &actualSeqLengths, &actualSeqLengthsKV,
                                 &deqScale1, &quantScale1, &deqScale2, &quantScale2, &quantOffset2, &learnableSink});
    rst = rst && mCtx.SetOutputs({&attentionOut});
    rst = rst && mCtx.SetTilingDataMaxSize(4096);
    rst = rst && mCtx.SetAttrs({{"num_heads", mParam.numHeads},
                                {"scale_value", mParam.scaleValue},
                                {"pre_tokens", mParam.preTokens},
                                {"next_tokens", mParam.nextTokens},
                                {"input_layout", mParam.layout},
                                {"num_key_value_heads", mParam.kvNumHeads},
                                {"sparse_mode", mParam.sparseMode},
                                {"inner_precise", mParam.innerPrecise},
                                {"block_size", mParam.blockSize}});
    #ifdef SUPPORT_KERNEL
        rst = rst && mCtx.SetKernelRunCbf(RunPromptFlashAttention);
        rst = rst && mCtx.SetKernelMainFunc((void *)nullptr);
        if(this->mPfaKernelTemplateFunc){
            rst = rst && mCtx.SetKernelRunTemplateCbf(RunTemplatePromptFlashAttention);
            rst = rst && mCtx.SetKernelTemplateMainFunc(this->mPfaKernelTemplateFunc);
        }
    #endif
    rst = rst && mOpInfo.SetContext(&mCtx);
    auto* platform = Platform::GetGlobalPlatform();
    if (platform == nullptr) {
        LOG_ERR("Global Platform is null");
        return false;
    }
    pfaTilingFunc = (gert::OpImplRegisterV2::TilingKernelFunc)platform->LoadOpTilingSoSym("TilingPromptFlashAttention");
    if(pfaTilingFunc == nullptr) {
        LOG_ERR("Can't get origin tiling func, pfa(%p)",pfaTilingFunc);
        return false;
    }
    IMPL_OP(PromptFlashAttention).Tiling(TilingPromptFlashAttentionStub);
    return rst;
}

bool PfaCase::InitCurrentCasePtr()
{
    Case::mCurrentCasePtr = this;
    return true;
}

bool PfaCase::Run()
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

PfaCase::PfaCase(const char *name, bool enable, const char *dbgInfo,
           const std::function<void(PFA_INPUT_DTYPE)>& templatekeyKernelFunc,
           OpInfo prompt, Param param)
    : Case(name, enable, dbgInfo), mOpInfo(std::move(prompt)), mParam(std::move(param))
{
    this->mOpInfo.mName = "PromptFlashAttention";
    this->mPfaKernelTemplateFunc = templatekeyKernelFunc;
}

PfaCase::PfaCase(const char *name, bool enable, const char *dbgInfo, OpInfo prompt, Param param)
    : Case(name, enable, dbgInfo), mOpInfo(std::move(prompt)), mParam(std::move(param))
{
    this->mOpInfo.mName = "PromptFlashAttention";
}

PfaCase::PfaCase()
{
}

PfaCase::Param::Param()
{
}

PfaCase::Param::Param(int64_t pB, int64_t pN, int64_t pS, int64_t pD, std::string pLayout, int64_t pNumHeads,
                      int64_t pKvNumHeads, float pScaleValue, int64_t pBlockSize, int64_t pInnerPrecise,
                      int64_t pSparseMode, int64_t pPreTokens, int64_t pNextTokens)
    : b(pB), n(pN), s(pS), d(pD), layout(pLayout), numHeads(pNumHeads), kvNumHeads(pKvNumHeads),
      scaleValue(pScaleValue), blockSize(pBlockSize), innerPrecise(pInnerPrecise), sparseMode(pSparseMode),
      preTokens(pPreTokens), nextTokens(pNextTokens)
{
}

bool PfaCase::DoOpTiling(DoTilingParam &tilingParam) {
    if (tilingParam.ctx == nullptr) {
        return false;
    }
    if (tilingParam.actualSeqLengthsTensor != nullptr && mParam.actualSeqLength.size() != 0) {
        tilingParam.actualSeqLengthsTensor->SetData(gert::TensorData{mParam.actualSeqLength.data()});
    }
    if (tilingParam.actualSeqLengthsKVTensor != nullptr && mParam.actualSeqLengthKV.size() != 0) {
        tilingParam.actualSeqLengthsKVTensor->SetData(gert::TensorData{mParam.actualSeqLengthKV.data()});
    }
    return true;
}