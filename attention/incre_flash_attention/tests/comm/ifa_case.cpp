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
 * \file ifa_case.cpp
 * \brief IncreFlashAttentionScore 测试用例.
 */
#include "ifa_case.h"
#include <utility>
#include <tikicpulib.h>
#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "tests/utils/log.h"
#include "tests/utils/platform.h"
#include "tiling/ifa/tiling_data.h"
#include "tiling/ifa/tiling_stub.h"
#include "tiling_base/tiling_base.h"
#include "../../op_kernel/incre_flash_attention_tiling.h"

/**
 * 以下函数声明需要保持与 CMakeList.txt 中调用 OpsTest_Level2_AddOp 函数时 KERNEL_PRIVATE_COMPILE_DEFINITIONS_EXT
 * 参数所控制的 Kernel 入口一致.
 */

#define IFA_KERNEL_PARAM                                                                                               \
    (__gm__ uint8_t * query, __gm__ uint8_t * key, __gm__ uint8_t * value, __gm__ uint8_t * pseShift,                  \
     __gm__ uint8_t * attenMask, __gm__ uint8_t * actualSeqLengths, __gm__ uint8_t * deqScale1,                        \
     __gm__ uint8_t * quantScale1, __gm__ uint8_t * deqScale2, __gm__ uint8_t * quantScale2,                           \
     __gm__ uint8_t * quantOffset2, __gm__ uint8_t * antiquantScale, __gm__ uint8_t * antiquantOffset,                 \
     __gm__ uint8_t * blocktable, __gm__ uint8_t * kvPaddingSize, __gm__ uint8_t * attentionOut,                       \
     __gm__ uint8_t * workspace, __gm__ uint8_t * tiling)

typedef void(*IfaKernelFunc) IFA_KERNEL_PARAM;

using namespace ops::adv::tests::ifa;
using TensorIntf = ops::adv::tests::utils::TensorIntf;
using Case = ops::adv::tests::utils::Case;
using Platform = ops::adv::tests::utils::Platform;

bool RunTemplateIncreFlashAttention(std::function<void(IFA_INPUT_DTYPE)> func, uint64_t tilingKey, int64_t blockDim, std::vector<TensorIntf *> &inputs,
                            std::vector<TensorIntf *> &outputs, uint8_t *workspace, uint8_t *tilingData)
{
    // Kernel 运行
    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(func, blockDim, inputs[0]->GetDevData(), inputs[1]->GetDevData(), inputs[2]->GetDevData(),
                inputs[3]->GetDevData(), inputs[4]->GetDevData(), inputs[5]->GetDevData(), inputs[6]->GetDevData(),
                inputs[7]->GetDevData(), inputs[8]->GetDevData(), inputs[9]->GetDevData(), inputs[10]->GetDevData(),
                inputs[11]->GetDevData(), inputs[12]->GetDevData(), inputs[13]->GetDevData(), inputs[14]->GetDevData(),
                outputs[0]->GetDevData(), workspace, tilingData);
    return true;
}

bool RunIncreFlashAttention(void *func, uint64_t tilingKey, int64_t blockDim, std::vector<TensorIntf *> &inputs,
                            std::vector<TensorIntf *> &outputs, uint8_t *workspace, uint8_t *tilingData)
{
    // Kernel 运行
    auto kernelFunc = (IfaKernelFunc)func;
    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(kernelFunc, blockDim, inputs[0]->GetDevData(), inputs[1]->GetDevData(), inputs[2]->GetDevData(),
                inputs[3]->GetDevData(), inputs[4]->GetDevData(), inputs[5]->GetDevData(), inputs[6]->GetDevData(),
                inputs[7]->GetDevData(), inputs[8]->GetDevData(), inputs[9]->GetDevData(), inputs[10]->GetDevData(),
                inputs[11]->GetDevData(), inputs[12]->GetDevData(), inputs[13]->GetDevData(), inputs[14]->GetDevData(),
                outputs[0]->GetDevData(), workspace, tilingData);
    return true;
}

extern "C" ge::graphStatus TilingIncreFlashAttentionStub(gert::TilingContext *context)
{
    auto *ifaCase = static_cast<IfaCase *>(Case::GetCurrentCase());
    if (ifaCase != nullptr) {
        IfaCase::DoTilingParam p;
        p.ctx = context;
        p.ret = ge::GRAPH_SUCCESS;
        p.actualSeqLengthsTensor = const_cast<gert::Tensor *>(context->GetOptionalInputTensor(5)); // 5:act_seq_len idx
        if (!ifaCase->DoOpTiling(p)) {
            return p.ret;
        }
        return ifaCase->ifaTilingFunc(context);
    }
    return ge::GRAPH_FAILED;
}

bool IfaCase::InitParam()
{
    h = mParam.n * mParam.d;
    int64_t kvNum = mParam.n;
    if (mParam.kvNumHeads != 0) {
        kvNum = mParam.kvNumHeads;
    }
    int64_t kvH = kvNum * mParam.d;

    if (mParam.layout == "BSH") {
        query = Tensor("query", {mParam.b, 1, h}, "BSH", mParam.qDataType, ge::FORMAT_ND);
        key = Tensor("key", {mParam.b, mParam.s, kvH}, "BSH", mParam.kvDataType, ge::FORMAT_ND);
        value = Tensor("value", {mParam.b, mParam.s, kvH}, "BSH", mParam.kvDataType, ge::FORMAT_ND);
        attentionOut = Tensor("attentionOut", {mParam.b, 1, h}, "BSH", mParam.outDataType, ge::FORMAT_ND);
    } else if (mParam.layout == "BNSD") {
        query = Tensor("query", {mParam.b, mParam.n, 1, mParam.d}, "BNSD", mParam.qDataType, ge::FORMAT_ND);
        key = Tensor("key", {mParam.b, kvNum, mParam.s, mParam.d}, "BNSD", mParam.kvDataType, ge::FORMAT_ND);
        value = Tensor("value", {mParam.b, kvNum, mParam.s, mParam.d}, "BNSD", mParam.kvDataType, ge::FORMAT_ND);
        attentionOut =
            Tensor("attentionOut", {mParam.b, mParam.n, 1, mParam.d}, "BNSD", mParam.outDataType, ge::FORMAT_ND);
    } else if (mParam.layout == "BSND") {
        query = Tensor("query", {mParam.b, 1, mParam.n, mParam.d}, "BSND", mParam.qDataType, ge::FORMAT_ND);
        key = Tensor("key", {mParam.b, mParam.s, kvNum, mParam.d}, "BSND", mParam.kvDataType, ge::FORMAT_ND);
        value = Tensor("value", {mParam.b, mParam.s, kvNum, mParam.d}, "BSND", mParam.kvDataType, ge::FORMAT_ND);
        attentionOut =
            Tensor("attentionOut", {mParam.b, 1, mParam.n, mParam.d}, "BSND", mParam.outDataType, ge::FORMAT_ND);
    }
    if (mParam.attenMaskType == AttenMaskShapeType::B_N_1_S) {
        attenMask =
            Tensor("attenMask", {mParam.b, mParam.n, 1, mParam.s}, "B_N_1_S", ge::DataType::DT_BOOL, ge::FORMAT_ND);
    } else if (mParam.attenMaskType == AttenMaskShapeType::B_1_S) {
        attenMask = Tensor("attenMask", {mParam.b, 1, mParam.s}, "B_1_S", ge::DataType::DT_BOOL, ge::FORMAT_ND);
    }

    if (mParam.pseShiftType == PseShiftShapeType::B_N_1_S) {
        pseShift = Tensor("pseShift", {mParam.b, mParam.n, 1, mParam.s}, "B_N_1_S", mParam.qDataType, ge::FORMAT_ND);
    } else if (mParam.pseShiftType == PseShiftShapeType::_1_N_1_S) {
        pseShift = Tensor("pseShift", {1, mParam.n, 1, mParam.s}, "_1_N_1_S", mParam.qDataType, ge::FORMAT_ND);
    }

    if (mParam.actualSeqLength.size() == 1) {
        actualSeqLengths = Tensor("actualSeqLengths", {1}, "1", ge::DataType::DT_INT64, ge::FORMAT_ND);
    } else if (mParam.actualSeqLength.size() != 0) {
        actualSeqLengths = Tensor("actualSeqLengths", {mParam.b}, "B", ge::DataType::DT_INT64, ge::FORMAT_ND);
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

    if (mParam.antiQuantType == AntiQuantShapeType::_2_H) {
        antiquantScale = Tensor("antiquantScale", {2, kvH}, "2_H", mParam.qDataType, ge::FORMAT_ND);
        antiquantOffset = Tensor("antiquantOffset", {2, kvH}, "2_H", mParam.qDataType, ge::FORMAT_ND);
    } else if (mParam.antiQuantType == AntiQuantShapeType::_2_N_D) {
        antiquantScale = Tensor("antiquantScale", {2, kvNum, mParam.d}, "2_N_D", mParam.qDataType, ge::FORMAT_ND);
        antiquantOffset = Tensor("antiquantOffset", {2, kvNum, mParam.d}, "2_N_D", mParam.qDataType, ge::FORMAT_ND);
    } else if (mParam.antiQuantType == AntiQuantShapeType::_2_N_1_D) {
        antiquantScale = Tensor("antiquantScale", {2, kvNum, 1, mParam.d}, "2_N_1_D", mParam.qDataType, ge::FORMAT_ND);
        antiquantOffset =
            Tensor("antiquantOffset", {2, kvNum, 1, mParam.d}, "2_N_1_D", mParam.qDataType, ge::FORMAT_ND);
    }

    if (mParam.blocktable.size() == 2) {
        blocktable = Tensor("blocktable", {mParam.blocktable[0], mParam.blocktable[1]}, "A_B", ge::DataType::DT_INT32,
                            ge::FORMAT_ND);
    }

    if (mParam.enbaleKvPaing) {
        kvPaddingSize = Tensor("kvPaddingSize", {mParam.kvPaddingSize}, "1", ge::DataType::DT_INT64, ge::FORMAT_ND);
    }
    return true;
}

bool IfaCase::InitOpInfo()
{
    bool rst = mCtx.SetOpName("IncreFlashAttention");
    rst = rst && mCtx.SetDeterministic(false);
    rst = rst && mCtx.SetInputs({&query, &key, &value, &pseShift, &attenMask, &actualSeqLengths, &deqScale1,
                                 &quantScale1, &deqScale2, &quantScale2, &quantOffset2, &antiquantScale,
                                 &antiquantOffset, &blocktable, &kvPaddingSize});
    rst = rst && mCtx.SetOutputs({&attentionOut});
    rst = rst && mCtx.SetAttrs({{"num_head", mParam.numHeads},
                                {"scale_value", mParam.scaleValue},
                                {"input_layout", mParam.layout},
                                {"num_key_value_heads", mParam.kvNumHeads},
                                {"block_size", mParam.blockSize},
                                {"inner_precise", mParam.innerPrecise}});
    rst = rst && mCtx.SetKernelRunCbf(RunIncreFlashAttention);
    rst = rst && mCtx.SetTilingDataMaxSize(3000); // 3000 : max tilingDataLen
    rst = rst && mCtx.SetKernelMainFunc((void *)nullptr);
    if (IfaKernelTemplateFunc) {
        rst = rst && mCtx.SetKernelRunTemplateCbf(RunTemplateIncreFlashAttention);
        rst = rst && mCtx.SetKernelTemplateMainFunc(IfaKernelTemplateFunc);
    }
    rst = rst && mOpInfo.SetContext(&mCtx);

    auto *platform = Platform::GetGlobalPlatform();
    if (platform == nullptr) {
        LOG_ERR("Global Platform is null");
        return false;
    }

    ifaTilingFunc = (gert::OpImplRegisterV2::TilingKernelFunc)platform->LoadOpTilingSoSym("TilingIncreFlashAttention");
    if (ifaTilingFunc == nullptr) {
        LOG_ERR("Can't get origin tiling func, ifa(%p)", ifaTilingFunc);
        return false;
    }
    IMPL_OP(IncreFlashAttention).Tiling(TilingIncreFlashAttentionStub);
    return rst;
}

bool IfaCase::InitCurrentCasePtr()
{
    Case::mCurrentCasePtr = this;
    return true;
}

bool IfaCase::Run()
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

IfaCase::IfaCase(const char *name, bool enable, const char *dbgInfo, 
            const std::function<void(IFA_INPUT_DTYPE)>& templatekeyKernelFunc,
            OpInfo incre, Param param)
    : Case(name, enable, dbgInfo), mOpInfo(std::move(incre)), mParam(std::move(param))
{
    this->mOpInfo.mName = "IncreFlashAttention";
    IfaKernelTemplateFunc = templatekeyKernelFunc;
}

IfaCase::IfaCase(const char *name, bool enable, const char *dbgInfo, OpInfo incre, Param param)
    : Case(name, enable, dbgInfo), mOpInfo(std::move(incre)), mParam(std::move(param))
{
    this->mOpInfo.mName = "IncreFlashAttention";
}

IfaCase::IfaCase()
{
}
IfaCase::Param::Param()
{
}
IfaCase::Param::Param(int64_t pB, int64_t pN, int64_t pS, int64_t pD, std::string pLayout, int64_t pNumHeads,
                      int64_t pKvNumHeads, float pScaleValue, int64_t pBlockSize, int64_t pInnerPrecise,
                      std::vector<int64_t> pActualSeqLength)
    : b(pB), n(pN), s(pS), d(pD), layout(pLayout), numHeads(pNumHeads), kvNumHeads(pKvNumHeads),
      scaleValue(pScaleValue), blockSize(pBlockSize), innerPrecise(pInnerPrecise), actualSeqLength(pActualSeqLength)
{
}


bool IfaCase::DoOpTiling(DoTilingParam& tilingParam) {
  if (tilingParam.ctx == nullptr) {
    return false;
  }
  if (tilingParam.actualSeqLengthsTensor != nullptr && mParam.actualSeqLength.size() != 0) {
    tilingParam.actualSeqLengthsTensor->SetData(gert::TensorData{mParam.actualSeqLength.data()});
  }
  return true;
}