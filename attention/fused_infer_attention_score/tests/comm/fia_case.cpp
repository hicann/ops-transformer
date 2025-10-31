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
 * \file fia_case.cpp
 * \brief FusedInferAttentionScore 测试用例.
 */
#include "fia_case.h"
#include <utility>
#include <tikicpulib.h>
#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "tests/utils/log.h"
#include "tests/utils/platform.h"
#include "tiling/fia/tiling_data.h"
#include "tiling/fia/tiling_stub.h"
#include "tiling_base/tiling_base.h"
#include "../../op_kernel/fused_infer_attention_score.cpp"

/**
 * 以下函数声明需要保持与 CMakeList.txt 中调用 OpsTest_Level2_AddOp 函数时 KERNEL_PRIVATE_COMPILE_DEFINITIONS_EXT
 * 参数所控制的 Kernel 入口一致.
 */
#define FIA_KERNEL_PARAM                                                                                               \
    (__gm__ uint8_t * query, __gm__ uint8_t * key, __gm__ uint8_t * value, __gm__ uint8_t * pse_shift,                 \
     __gm__ uint8_t * attenMask, __gm__ uint8_t * actualSeqLengths, __gm__ uint8_t * actualSeqLengthsKV,               \
     __gm__ uint8_t * deq_scale1, __gm__ uint8_t * quant_scale1, __gm__ uint8_t * deq_scale2,                          \
     __gm__ uint8_t * quant_scale2, __gm__ uint8_t * quant_offset2, __gm__ uint8_t * antiquantScale,                   \
     __gm__ uint8_t * antiquantOffset, __gm__ uint8_t * blocktable, __gm__ uint8_t * queryPaddingSize,                 \
     __gm__ uint8_t * kvPaddingSize, __gm__ uint8_t * keyAntiquantScale, __gm__ uint8_t * keyAntiquantOffset,          \
     __gm__ uint8_t * valueAntiquantScale, __gm__ uint8_t * valueAntiquantOffset, __gm__ uint8_t * keySharedPrefix,    \
     __gm__ uint8_t * valueSharedPrefix, __gm__ uint8_t * actualSharedPrefixLen, __gm__ uint8_t * attentionOut,        \
     __gm__ uint8_t * softmaxLse, __gm__ uint8_t * workspace, __gm__ uint8_t * tiling)

typedef void(*FiaKernelFunc) FIA_KERNEL_PARAM;

// extern "C" __global__ __aicore__ void fused_infer_attention_score FIA_KERNEL_PARAM;

using namespace ops::adv::tests::fia;
using TensorIntf = ops::adv::tests::utils::TensorIntf;
using Case = ops::adv::tests::utils::Case;
using Platform = ops::adv::tests::utils::Platform;
using Tensor = ops::adv::tests::utils::Tensor;
using TensorList = ops::adv::tests::utils::TensorList;

bool RunTemplateFusedInferAttentionScore(std::function<void(FIA_INPUT_DTYPE)> func, uint64_t tilingKey, int64_t blockDim, std::vector<TensorIntf *> &inputs,
                                 std::vector<TensorIntf *> &outputs, uint8_t *workspace, uint8_t *tilingData)
{
    // Kernel 运行
    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(func, blockDim, inputs[0]->GetDevData(), inputs[1]->GetDevData(), inputs[2]->GetDevData(),
                inputs[3]->GetDevData(), inputs[4]->GetDevData(), inputs[5]->GetDevData(), inputs[6]->GetDevData(),
                inputs[7]->GetDevData(), inputs[8]->GetDevData(), inputs[9]->GetDevData(), inputs[10]->GetDevData(),
                inputs[11]->GetDevData(), inputs[12]->GetDevData(), inputs[13]->GetDevData(), inputs[14]->GetDevData(),
                inputs[15]->GetDevData(), inputs[16]->GetDevData(), inputs[17]->GetDevData(), inputs[18]->GetDevData(),
                inputs[19]->GetDevData(), inputs[20]->GetDevData(), inputs[21]->GetDevData(), inputs[22]->GetDevData(),
                inputs[23]->GetDevData(), outputs[0]->GetDevData(), outputs[1]->GetDevData(), workspace, tilingData);
    return true;
}

bool RunFusedInferAttentionScore(void *func, uint64_t tilingKey, int64_t blockDim, std::vector<TensorIntf *> &inputs,
                                 std::vector<TensorIntf *> &outputs, uint8_t *workspace, uint8_t *tilingData)
{
    // Kernel 运行
    auto kernelFunc = (FiaKernelFunc)func;
    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(kernelFunc, blockDim, inputs[0]->GetDevData(), inputs[1]->GetDevData(), inputs[2]->GetDevData(),
                inputs[3]->GetDevData(), inputs[4]->GetDevData(), inputs[5]->GetDevData(), inputs[6]->GetDevData(),
                inputs[7]->GetDevData(), inputs[8]->GetDevData(), inputs[9]->GetDevData(), inputs[10]->GetDevData(),
                inputs[11]->GetDevData(), inputs[12]->GetDevData(), inputs[13]->GetDevData(), inputs[14]->GetDevData(),
                inputs[15]->GetDevData(), inputs[16]->GetDevData(), inputs[17]->GetDevData(), inputs[18]->GetDevData(),
                inputs[19]->GetDevData(), inputs[20]->GetDevData(), inputs[21]->GetDevData(), inputs[22]->GetDevData(),
                inputs[23]->GetDevData(), outputs[0]->GetDevData(), outputs[1]->GetDevData(), workspace, tilingData);
    return true;
}

extern "C" ge::graphStatus TilingFusedInferAttentionStub(gert::TilingContext* context)
{
    auto* fiaCase = static_cast<FiaCase*>(Case::GetCurrentCase());
    if (fiaCase != nullptr) {
        FiaCase::DoTilingParam p;
        p.ctx = context;
        p.ret = ge::GRAPH_SUCCESS;
        p.actualSeqLengthsTensor = const_cast<gert::Tensor*>(context->GetOptionalInputTensor(5)); // 5: the index of actual seqlength
        p.actualSeqLengthsKVTensor = const_cast<gert::Tensor*>(context->GetOptionalInputTensor(6)); // 6: the index of actual kvseqlength
        p.actualSharedPrefixLen = const_cast<gert::Tensor*>(context->GetOptionalInputTensor(23)); // 23: the index of actualSharedPrefixLen
        if (!fiaCase->DoOpTiling(p)) {
            return p.ret;
        }
        return fiaCase->fiaTilingFunc(context);
    }
    return ge::GRAPH_FAILED;
}

const std::string INVALID_LAYOUT = "BSDD";

bool FiaCase::IsMla() const
{
    if (mParam.mode == CaseMode::MLA_NOQUANT ||
        mParam.mode == CaseMode::MLA_ANTIQUANT ||
        mParam.mode == CaseMode::MLA_FULLQUANT) {
        return true;
    }
    return false;
}

Tensor FiaCase::ConstructTensor(std::string name, ShapeParam shapeParam, std::string layout,
    ge::DataType dtype, ge::Format format) const
{
    if (layout == "BSH") {
        return Tensor(name.c_str(), {shapeParam.b, shapeParam.s, shapeParam.n * shapeParam.d}, layout.c_str(), dtype, format);
    } else if (layout == "BSND") {
        return Tensor(name.c_str(), {shapeParam.b, shapeParam.s, shapeParam.n, shapeParam.d}, layout.c_str(), dtype, format);
    } else if (layout == "BNSD") {
        return Tensor(name.c_str(), {shapeParam.b, shapeParam.n, shapeParam.s, shapeParam.d}, layout.c_str(), dtype, format);
    } else if (layout == "TND") {
        return Tensor(name.c_str(), {shapeParam.t, shapeParam.n, shapeParam.d}, layout.c_str(), dtype, format);
    } else if (layout == "NTD") {
        return Tensor(name.c_str(), {shapeParam.n, shapeParam.t, shapeParam.d}, layout.c_str(), dtype, format);
    } else {
        // invalid tensor, layout is BSDD
        return Tensor(name.c_str(), {shapeParam.b, shapeParam.s, shapeParam.d, shapeParam.d}, layout.c_str(), dtype, format);
    }
}

TensorList FiaCase::ConstructTensorList(std::string name, ShapeParam shapeParam, std::string layout,
    ge::DataType dtype, ge::Format format) const
{
    if (layout == "BSH") {
        return TensorList(name.c_str(), {shapeParam.b, shapeParam.s, shapeParam.n * shapeParam.d}, layout.c_str(), dtype, format);
    } else if (layout == "BSND") {
        return TensorList(name.c_str(), {shapeParam.b, shapeParam.s, shapeParam.n, shapeParam.d}, layout.c_str(), dtype, format);
    } else if (layout == "BNSD") {
        return TensorList(name.c_str(), {shapeParam.b, shapeParam.n, shapeParam.s, shapeParam.d}, layout.c_str(), dtype, format);
    } else if (layout == "TND") {
        return TensorList(name.c_str(), {shapeParam.t, shapeParam.n, shapeParam.d}, layout.c_str(), dtype, format);
    } else if (layout == "NTD") {
        return TensorList(name.c_str(), {shapeParam.n, shapeParam.t, shapeParam.d}, layout.c_str(), dtype, format);
    } else {
        // invalid tensor, layout is BSDD
        return TensorList(name.c_str(), {shapeParam.b, shapeParam.s, shapeParam.d, shapeParam.d}, layout.c_str(), dtype, format);
    }
}

std::string FiaCase::GetQueryLayout(std::string layout) const
{
    const std::map<std::string, std::string> layoutMap = {
        {"BSH",       "BSH",},
        {"BSND",      "BSND",},
        {"BNSD",      "BNSD",},
        {"TND",       "TND",},
        {"BSH_NBSD",  "BSH",},
        {"BSND_NBSD", "BSND",},
        {"BNSD_NBSD", "BNSD",},
        {"TND_NTD",   "TND",},
        {"NTD_TND",   "NTD",},
        {"BNSD_BSND", "BNSD",}
    };

    auto it = layoutMap.find(layout);
    return (it != layoutMap.end()) ? it->second : INVALID_LAYOUT;
}

std::string FiaCase::GetOutLayout(std::string layout) const
{
    const std::map<std::string, std::string> layoutMap = {
        {"BSH",       "BSH" },
        {"BSND",      "BSND"},
        {"BNSD",      "BNSD"},
        {"TND",       "TND" },
        {"BSH_NBSD",  "NBSD"},
        {"BSND_NBSD", "NBSD"},
        {"BNSD_NBSD", "NBSD"},
        {"TND_NTD",   "NTD" },
        {"NTD_TND",   "TND" },
        {"BNSD_BSND", "BSND"}
    };

    auto it = layoutMap.find(layout);
    return (it != layoutMap.end()) ? it->second : INVALID_LAYOUT;
}

std::string FiaCase::GetKvLayout(std::string layout) const
{
    std::string qLayout = GetQueryLayout(layout);
    if (mParam.storageMode != CaseKvStorageMode::PAGE_ATTENTION) {
        return qLayout;
    } else {
        if (qLayout == INVALID_LAYOUT) {
            return INVALID_LAYOUT;
        } else if (qLayout == "BSH" || qLayout == "TND" || qLayout == "NTD") {
            return "BSH";
        } else {
            return "BNSD";
        }
    }
}

bool FiaCase::InitInOutParam()
{
    ShapeParam queryParam = {mParam.b, mParam.qs, mParam.numHeads, mParam.qkHeadDim, mParam.t};
    int64_t kvS = (mParam.storageMode == CaseKvStorageMode::PAGE_ATTENTION) ? mParam.blockSize : mParam.s;
    ShapeParam keyParam = {mParam.b, kvS, mParam.kvNumHeads, mParam.qkHeadDim, mParam.t};
    ShapeParam valueParam = {mParam.b, kvS, mParam.kvNumHeads, mParam.vHeadDim, mParam.t};
    ShapeParam attenOutParam = {mParam.b, mParam.qs, mParam.numHeads, mParam.vHeadDim, mParam.t};

    std::string qLayout = GetQueryLayout(mParam.layout);
    std::string kvLayout = GetKvLayout(mParam.layout);
    std::string outLayout = GetOutLayout(mParam.layout);

    query = ConstructTensor("query", queryParam, qLayout, mParam.qDataType, ge::FORMAT_ND);
    key = ConstructTensorList("key", keyParam, kvLayout, mParam.kDataType, ge::FORMAT_ND);
    value = ConstructTensorList("value", valueParam, kvLayout, mParam.vDataType, ge::FORMAT_ND);
    attentionOut = ConstructTensor("attentionOut", attenOutParam, outLayout, mParam.outDataType, ge::FORMAT_ND);

    if (IsMla()) {
        ShapeParam queryRopeParam = {mParam.b, mParam.qs, mParam.numHeads, mParam.ropeHeadDim, mParam.t};
        ShapeParam keyRopeParam = {mParam.b, kvS, mParam.kvNumHeads, mParam.ropeHeadDim, mParam.t};
        queryRope = ConstructTensor("queryRope", queryRopeParam, qLayout, mParam.qDataType, ge::FORMAT_ND);
        keyRope = ConstructTensor("keyRope", keyRopeParam, kvLayout, mParam.kDataType, ge::FORMAT_ND);
    }

    if (mParam.storageMode == CaseKvStorageMode::PAGE_ATTENTION) {
        blocktable = Tensor("blockTable", {mParam.b, mParam.s / mParam.blockSize}, "BSND", ge::DT_INT32, ge::FORMAT_ND);
    }
    return true;
}

bool FiaCase::InitEnhanceParamActualSeqLenQ()
{
    if (mParam.actualSeqLength.empty() && (mParam.layout == "TND" ||
        mParam.layout == "NTD" || mParam.layout == "TND_NTD")) {
        for (int64_t i = 0; i < mParam.b; ++i) {
            mParam.actualSeqLength.push_back((i + 1) * mParam.qs);
        }
    }

    if (mParam.actualSeqLength.size() != 0) {
        int64_t actualSeqLengthSize = mParam.actualSeqLength.size();
        actualSeqLengths = Tensor("actualSeqLengths",
            {actualSeqLengthSize}, "B", ge::DataType::DT_INT64, ge::FORMAT_ND);
    }
    return true;
}

bool FiaCase::InitEnhanceParamActualSeqLenKv()
{
    if (mParam.actualSeqLengthKV.empty()) {
        for (int64_t i = 0; i < mParam.b; ++i) {
            if (mParam.layout == "TND" || mParam.layout == "NTD" ||
                mParam.layout == "TND_NTD") {
                mParam.actualSeqLengthKV.push_back((i + 1) * mParam.s);
            } else {
                mParam.actualSeqLengthKV.push_back(mParam.s);
            }
        }
    }

    if (mParam.actualSeqLengthKV.size() != 0) {
        int64_t actualSeqLengthKVSize = mParam.actualSeqLengthKV.size();
        actualSeqLengthsKV = Tensor("actualSeqLengthsKV",
            {actualSeqLengthKVSize}, "B", ge::DataType::DT_INT64, ge::FORMAT_ND);
    }
    return true;
}

bool FiaCase::InitEnhanceParam()
{
    constexpr int64_t DEFAULT_B = 48;
    constexpr int64_t DEFAULT_S1 = 1;
    constexpr int64_t DEFAULT_S2 = 4096;
    constexpr int64_t DEFAULT_N1 = 64;
    constexpr int64_t DEFAULT_N2 = 1;
    constexpr int64_t DEFAULT_QK_HEAD_DIM = 512;
    constexpr int64_t DEFAULT_V_HEAD_DIM = 512;
    constexpr int64_t DEFAULT_ROPE_HEAD_DIM = 64;
    constexpr int64_t DEFAULT_BLOCK_SIZE = 128;

    mParam.numHeads = (mParam.numHeads == 0) ? DEFAULT_N1 : mParam.numHeads;
    mParam.kvNumHeads = (mParam.kvNumHeads == 0) ? DEFAULT_N2 : mParam.kvNumHeads;
    mParam.b = (mParam.b == 0) ? DEFAULT_B : mParam.b;
    mParam.qs = (mParam.qs == 0) ? DEFAULT_S1 : mParam.qs;
    mParam.s = (mParam.s == 0) ? DEFAULT_S2 : mParam.s;
    mParam.qkHeadDim = (mParam.qkHeadDim == 0) ? DEFAULT_QK_HEAD_DIM : mParam.qkHeadDim;
    mParam.vHeadDim = (mParam.vHeadDim == 0) ? DEFAULT_V_HEAD_DIM : mParam.vHeadDim;
    mParam.ropeHeadDim = (mParam.ropeHeadDim == 0) ? DEFAULT_ROPE_HEAD_DIM : mParam.ropeHeadDim;
    mParam.blockSize = (mParam.blockSize == 0) ? DEFAULT_BLOCK_SIZE : mParam.blockSize;

    if (mParam.layout == "TND" || mParam.layout == "NTD" || mParam.layout == "TND_NTD") {
        mParam.t = (mParam.t == 0) ? mParam.b * mParam.qs : mParam.t;
    }

    InitEnhanceParamActualSeqLenQ();
    InitEnhanceParamActualSeqLenKv();
    InitInOutParam();

    return true;
}

bool FiaCase::InitBasicParam()
{   
    if (mParam.h == 0) {
        h = mParam.n * mParam.d;
    } else {
        h = mParam.h;
    }
    
    int64_t kvNum = mParam.n;
    if (mParam.kvNumHeads != 0) {
        kvNum = mParam.kvNumHeads;
    }
    int64_t kvH = kvNum * mParam.d;

    if (mParam.layout == "BSH") {
        query = Tensor("query", {mParam.b, 1, h}, "BSH", mParam.qDataType, ge::FORMAT_ND);
        key = TensorList("key", {mParam.b, mParam.s, kvH}, "BSH", mParam.kDataType, ge::FORMAT_ND);
        value = TensorList("value", {mParam.b, mParam.s, kvH}, "BSH", mParam.vDataType, ge::FORMAT_ND);
        attentionOut = Tensor("attentionOut", {mParam.b, 1, h}, "BSH", mParam.outDataType, ge::FORMAT_ND);
    } else if (mParam.layout == "BNSD") {
        query = Tensor("query", {mParam.b, mParam.n, mParam.qs, mParam.d}, "BNSD", mParam.qDataType, ge::FORMAT_ND);
        key = TensorList("key", {mParam.b, kvNum, mParam.s, mParam.d}, "BNSD", mParam.kDataType, ge::FORMAT_ND);
        value = TensorList("value", {mParam.b, kvNum, mParam.s, mParam.d}, "BNSD", mParam.vDataType, ge::FORMAT_ND);
        attentionOut =
            Tensor("attentionOut", {mParam.b, mParam.n, mParam.qs, mParam.d}, "BNSD", mParam.outDataType, ge::FORMAT_ND);
    } else if (mParam.layout == "BSND") {
        query = Tensor("query", {mParam.b, 1, mParam.n, mParam.d}, "BSND", mParam.qDataType, ge::FORMAT_ND);
        key = TensorList("key", {mParam.b, mParam.s, kvNum, mParam.d}, "BSND", mParam.kDataType, ge::FORMAT_ND);
        value = TensorList("value", {mParam.b, mParam.s, kvNum, mParam.d}, "BSND", mParam.vDataType, ge::FORMAT_ND);
        attentionOut =
            Tensor("attentionOut", {mParam.b, 1, mParam.n, mParam.d}, "BSND", mParam.outDataType, ge::FORMAT_ND);
    } else if (mParam.layout == "TND") {
        query = Tensor("query", {mParam.t,  mParam.n, mParam.d}, "TND", mParam.qDataType, ge::FORMAT_ND);
        key = TensorList("key", {mParam.b, kvNum, mParam.s, mParam.d}, "BNSD", mParam.kDataType, ge::FORMAT_ND);
        value = TensorList("value", {mParam.b, kvNum, mParam.s, mParam.d}, "BNSD", mParam.vDataType, ge::FORMAT_ND);
        attentionOut =
            Tensor("attentionOut", {mParam.t, mParam.n, mParam.d}, "TND", mParam.outDataType, ge::FORMAT_ND);
    } else if (mParam.layout == "TND_NTD") {
        query = Tensor("query", {mParam.t,  mParam.n, mParam.d}, "TND", mParam.qDataType, ge::FORMAT_ND);
        key = TensorList("key", {mParam.b, kvNum, mParam.s, mParam.d}, "BNSD", mParam.kDataType, ge::FORMAT_ND);
        value = TensorList("value", {mParam.b, kvNum, mParam.s, mParam.d}, "BNSD", mParam.vDataType, ge::FORMAT_ND);
        attentionOut =
            Tensor("attentionOut", {mParam.n, mParam.t, mParam.d}, "NTD", mParam.outDataType, ge::FORMAT_ND);
    }
    // layout error check
    else if (mParam.layout == "BSDD") {
        query = Tensor("query", {mParam.b, 1, mParam.d, mParam.d}, "BSDD", mParam.qDataType, ge::FORMAT_ND);
        key = TensorList("key", {mParam.b, mParam.s, mParam.d, mParam.d}, "BSDD", mParam.kDataType, ge::FORMAT_ND);
        value = TensorList("value", {mParam.b, mParam.s, mParam.d, mParam.d}, "BSDD", mParam.vDataType, ge::FORMAT_ND);
        attentionOut =
            Tensor("attentionOut", {mParam.b, 1, mParam.d, mParam.d}, "BSDD", mParam.outDataType, ge::FORMAT_ND);
    }
    if (mParam.actualSeqLength.size() != 0) {
        int64_t actualSeqLengthSize = mParam.actualSeqLength.size();
        actualSeqLengths = Tensor("actualSeqLengths", {actualSeqLengthSize}, "B", ge::DataType::DT_INT64, ge::FORMAT_ND);
    }
    if (mParam.actualSeqLengthKV.size() != 0) {
        int64_t actualSeqLengthKVSize = mParam.actualSeqLengthKV.size();
        actualSeqLengthsKV = Tensor("actualSeqLengthsKV", {actualSeqLengthKVSize}, "B", ge::DataType::DT_INT64, ge::FORMAT_ND);
    }

    return true;
}

bool FiaCase::InitParam()
{
    if (mParam.mode != CaseMode::DEFAULT) {
        return InitEnhanceParam();
    } else {
        return InitBasicParam();
    }
}

bool FiaCase::InitOpInfo()
{
    bool rst = mCtx.SetOpName("FusedInferAttentionScore");
    rst = rst && mCtx.SetDeterministic(false);
    rst = rst && mCtx.SetInputs({&query,
                                 &key,
                                 &value,
                                 &pseShift,
                                 &attenMask,
                                 &actualSeqLengths,
                                 &actualSeqLengthsKV,
                                 &deqScale1,
                                 &quantScale1,
                                 &deqScale2,
                                 &quantScale2,
                                 &quantOffset2,
                                 &antiquantScale,
                                 &antiquantOffset,
                                 &blocktable,
                                 &queryPaddinSize,
                                 &kvPaddingSize,
                                 &keyAntiquantScale,
                                 &keyAntiquantOffset,
                                 &valueAntiquantScale,
                                 &valueAntiquantOffset,
                                 &keySharedPrefix,
                                 &valueSharedPrefix,
                                 &actualSharedPrefixLen,
                                 &queryRope,
                                 &keyRope,
                                 &keyRopeAntiquantScale,
                                 &dequantScaleQuery,
                                 &qStartIdx,
                                 &kvStartIdx});
    rst = rst && mCtx.SetTilingDataMaxSize(4096);   
    rst = rst && mCtx.SetOutputs({&attentionOut, &softmaxLse});
    
    rst = rst && mCtx.SetAttrs({
                                {"num_head", mParam.numHeads},
                                {"scale_value", mParam.scaleValue},
                                {"pre_tokens", mParam.pre_tokens},
                                {"next_tokens", mParam.next_tokens},
                                {"input_layout", mParam.layout},
                                {"num_key_value_heads", mParam.kvNumHeads},
                                {"sparse_mode", mParam.sparse_mode},
                                {"inner_precise", mParam.innerPrecise},
                                {"block_size", mParam.blockSize},
                                {"antiquant_mode", mParam.antiquant_mode},
                                {"softmax_lse_flag", mParam.softmax_lse_flag},
                                {"key_antiquant_mode", mParam.key_antiquant_mode},
                                {"value_antiquant_mode", mParam.value_antiquant_mode},
                                {"query_quant_mode", mParam.queryQuantMode},
                                {"pse_type", mParam.pseType}});

    #ifdef SUPPORT_KERNEL
        rst = rst && mCtx.SetKernelRunCbf(RunFusedInferAttentionScore);
        rst = rst && mCtx.SetKernelMainFunc((void *)nullptr);
        if (FiaKernelTemplateFunc) {
            rst = rst && mReverseCtx.SetKernelRunTemplateCbf(RunTemplateFusedInferAttentionScore);
            rst = rst && mReverseCtx.SetKernelTemplateMainFunc(FiaKernelTemplateFunc);
        }
    #endif
    rst = rst && mOpInfo.SetContext(&mCtx);
    auto* platform = Platform::GetGlobalPlatform();

    if (platform == nullptr) {
        LOG_ERR("Global Platform is null");
        return false;
    }
    fiaTilingFunc = (gert::OpImplRegisterV2::TilingKernelFunc)platform->LoadOpTilingSoSym("DoOpTilingFusedInferAttentionScore");
    if(fiaTilingFunc == nullptr) {
        LOG_ERR("Can't get origin tiling func");
        return false;
    }
    IMPL_OP(FusedInferAttentionScore).Tiling(TilingFusedInferAttentionStub);
    return rst;
}

bool FiaCase::InitCurrentCasePtr()
{
    Case::mCurrentCasePtr = this;
    return true;
}

bool FiaCase::Run()
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

FiaCase::FiaCase(const char *name, bool enable, const char *dbgInfo, 
                 const std::function<void(FIA_INPUT_DTYPE)>& templatekeyKernelFunc,
                 OpInfo fia, Param param)
    : Case(name, enable, dbgInfo), mOpInfo(std::move(fia)), mParam(std::move(param))
{
    this->mOpInfo.mName = "FusedInferAttentionScore";
    FiaKernelTemplateFunc = templatekeyKernelFunc;
}

FiaCase::FiaCase(const char *name, bool enable, const char *dbgInfo, OpInfo fia, Param param)
    : Case(name, enable, dbgInfo), mOpInfo(std::move(fia)), mParam(std::move(param))
{
    this->mOpInfo.mName = "FusedInferAttentionScore";
}

FiaCase::FiaCase()
{
}
FiaCase::Param::Param()
{
}

bool FiaCase::DoOpTiling(DoTilingParam &tilingParam) {
    if (tilingParam.ctx == nullptr) {
        return false;
    }
    if (tilingParam.actualSeqLengthsTensor != nullptr && mParam.actualSeqLength.size() != 0) {
        tilingParam.actualSeqLengthsTensor->SetData(gert::TensorData{mParam.actualSeqLength.data()});
    }
    if (tilingParam.actualSeqLengthsKVTensor != nullptr && mParam.actualSeqLengthKV.size() != 0) {
        tilingParam.actualSeqLengthsKVTensor->SetData(gert::TensorData{mParam.actualSeqLengthKV.data()});
    }
    if (tilingParam.actualSharedPrefixLen != nullptr && mParam.actualSharedPrefixLens.size() != 0) {
        tilingParam.actualSharedPrefixLen->SetData(gert::TensorData{mParam.actualSharedPrefixLens.data()});
    }
    return true;
}