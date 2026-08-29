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
 * \file ffn_case.cpp
 * \brief FFN 测试用例.
 */

#include "ffn_case.h"
#include <utility>
#include <tikicpulib.h>
#include <register/op_impl_registry.h>
#include <graph/utils/type_utils.h>
#include "tests/utils/tests_utils_log.h"
#include "tests/utils/io.h"
#include "tests/utils/platform.h"
#include "tiling/ffn/tiling_data.h"
#include "op_host/tiling_templates_registry.h"

using Case = ops::adv::tests::utils::Case;
using FFNCase = ops::adv::tests::ffn::FFNCase;
using ops::adv::tests::utils::ReadFile;
using ops::adv::tests::utils::WriteFile;

/**
 * 以下函数声明需要保持与 CMakeList.txt 中调用 OpsTest_Level2_AddOp 函数时 KERNEL_PRIVATE_COMPILE_DEFINITIONS_EXT
 * 参数所控制的 Kernel 入口一致.
 */

#define FFN_KERNEL_PARAM                                                                                               \
    (__gm__ uint8_t * x, __gm__ uint8_t * weight1, __gm__ uint8_t * weight2, __gm__ uint8_t * expertTokens,            \
     __gm__ uint8_t * bias1, __gm__ uint8_t * bias2, __gm__ uint8_t * scale, __gm__ uint8_t * offset,                  \
     __gm__ uint8_t * deqScale1, __gm__ uint8_t * deqScale2, __gm__ uint8_t * antiquant_scale1,                        \
     __gm__ uint8_t * antiquant_scale2, __gm__ uint8_t * antiquant_offset1, __gm__ uint8_t * antiquant_offset2,        \
     __gm__ uint8_t * y, __gm__ uint8_t * workSpace, __gm__ uint8_t * tiling)

typedef void(*FFNKernelFunc) FFN_KERNEL_PARAM;

extern "C" __global__ __aicore__ void ffn_quant_fp16 FFN_KERNEL_PARAM;
extern "C" __global__ __aicore__ void ffn_quant_bf16 FFN_KERNEL_PARAM;
extern "C" __global__ __aicore__ void ffn_fp16 FFN_KERNEL_PARAM;
extern "C" __global__ __aicore__ void ffn_bf16 FFN_KERNEL_PARAM;
extern "C" __global__ __aicore__ void ffn_a16w8_fp16 FFN_KERNEL_PARAM;
extern "C" __global__ __aicore__ void ffn_a16w8_bf16 FFN_KERNEL_PARAM;
extern "C" __global__ __aicore__ void ffn_a16w4_fp16 FFN_KERNEL_PARAM;
extern "C" __global__ __aicore__ void ffn_a16w4_bf16 FFN_KERNEL_PARAM;

using namespace ops::adv::tests::ffn;
using ops::adv::tests::utils::Platform;
using ops::adv::tests::utils::Tensor;
using ops::adv::tests::utils::TensorIntf;

enum KernelParams {
    X = 0,
    WEIGHT1,
    WEIGHT2,
    EXPERT_TOKENS,
    BIAS1,
    BIAS2,
    SCALE,
    OFFSET,
    DEQ_SCALE1,
    DEQ_SCALE2,
    ANTIQUANT_SCALE1,
    ANTIQUANT_SCALE2,
    ANTIQUANT_OFFSET1,
    ANTIQUANT_OFFSET2
};

bool RunFFN(void *func, uint64_t tilingKey, int64_t blockDim, std::vector<TensorIntf *> &inputs,
            std::vector<TensorIntf *> &output, uint8_t *workspace, uint8_t *tilingData)
{
    (void)blockDim;
    // Kernel 运行
    auto kernelFunc = (FFNKernelFunc)func;
    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(kernelFunc, 1, inputs[X]->GetDevData(), inputs[WEIGHT1]->GetDevData(), inputs[WEIGHT2]->GetDevData(),
                inputs[EXPERT_TOKENS]->GetDevData(), inputs[BIAS1]->GetDevData(), inputs[BIAS2]->GetDevData(),
                inputs[SCALE]->GetDevData(), inputs[OFFSET]->GetDevData(), inputs[DEQ_SCALE1]->GetDevData(),
                inputs[DEQ_SCALE2]->GetDevData(), inputs[ANTIQUANT_SCALE1]->GetDevData(),
                inputs[ANTIQUANT_SCALE2]->GetDevData(), inputs[ANTIQUANT_OFFSET1]->GetDevData(),
                inputs[ANTIQUANT_OFFSET2]->GetDevData(), output[0]->GetDevData(), workspace, tilingData);
    return true;
}

FFNCase::FFNCase() : FFNCase("Undefined", true, "", OpInfo(), Param(), 0)
{
}

FFNCase::FFNCase(const char *name, bool enable, const char *dbgInfo, OpInfo opInfo, Param param,
                 int32_t tilingTemplatePriority)
    : Case(name, enable, dbgInfo, tilingTemplatePriority), mOpInfo(std::move(opInfo)), mParam(std::move(param))
{
    this->mOpInfo.mName = "FFN";
}

bool FFNCase::InitParam()
{
    if (mParam.mExpertTokensData.size() > 0) {
        size_t dataSize = mParam.mExpertTokensData.size() * sizeof(int64_t);
        uint8_t *addr = mParam.mTensors["expertTokens"].AllocDevData(0, dataSize);
        if (addr == nullptr) {
            LOG_ERR("Tensor(%s, %zu) AllocDevData Failed.", mParam.mTensors["expertTokens"].Name().c_str(), dataSize);
            return false;
        }
        std::string fileName = this->mName + "_expertToken.bin";
        if (!WriteFile(fileName, mParam.mExpertTokensData.data(), dataSize)) {
            LOG_ERR("Write expertToken data to file[%s] failed", fileName.c_str());
            return false;
        }
        if (!ReadFile(fileName, dataSize, addr, dataSize)) {
            LOG_ERR("Read expertToken data[%s] to tensor failed", fileName.c_str());
            return false;
        }
    }
    return true;
}

bool FFNCase::InitOpInfo()
{
    auto *ffnKernelFunc = (void *)ffn_quant_fp16;

    if (mParam.mTensors["x"].GetDataType() == mParam.mTensors["weight1"].GetDataType()) {
        if (mParam.mTensors["weight1"].GetDataType() == ge::DataType::DT_INT8) { // 量化
            mIsQuant = true;
            if (mParam.mTensors["y"].GetDataType() == ge::DataType::DT_FLOAT16) {
                ffnKernelFunc = (void *)ffn_quant_fp16;
            } else {
                ffnKernelFunc = (void *)ffn_quant_bf16;
            }
        } else { // 非量化
            if (mParam.mTensors["weight1"].GetDataType() == ge::DataType::DT_FLOAT16) {
                ffnKernelFunc = (void *)ffn_fp16;
            } else {
                ffnKernelFunc = (void *)ffn_bf16;
            }
        }
    } else { // 伪量化
        if (mParam.mTensors["weight1"].GetDataType() == ge::DataType::DT_INT8) {
            if (mParam.mTensors["x"].GetDataType() == ge::DataType::DT_FLOAT16) {
                ffnKernelFunc = (void *)ffn_a16w8_fp16;
            } else {
                ffnKernelFunc = (void *)ffn_a16w8_bf16;
            }
        } else {
            if (mParam.mTensors["x"].GetDataType() == ge::DataType::DT_FLOAT16) {
                ffnKernelFunc = (void *)ffn_a16w4_fp16;
            } else {
                ffnKernelFunc = (void *)ffn_a16w4_bf16;
            }
        }
    }

    bool rst = mCtx.SetOpName(mOpInfo.mName.c_str());
    rst = rst && mCtx.SetDeterministic(mOpInfo.mCtr.mDeterministic);
    rst = rst && mCtx.SetInputs({&mParam.mTensors["x"], &mParam.mTensors["weight1"], &mParam.mTensors["weight2"],
                                 &mParam.mTensors["expertTokens"], &mParam.mTensors["bias1"], &mParam.mTensors["bias2"],
                                 &mParam.mTensors["scale"], &mParam.mTensors["offset"], &mParam.mTensors["deqScale1"],
                                 &mParam.mTensors["deqScale2"], &mParam.mTensors["antiquant_scale1"],
                                 &mParam.mTensors["antiquant_scale2"], &mParam.mTensors["antiquant_offset1"],
                                 &mParam.mTensors["antiquant_offset2"]});
    rst = rst && mCtx.SetOutputs({&mParam.mTensors["y"]});
    rst = rst && mCtx.SetAttrs({{"activation", mParam.mActivation},
                                {"inner_precise", mParam.mInnerPrecise},
                                {"output_dtype", mParam.mOutputDtype},
                                {"tokens_index_flag", mParam.mTokensIndexFlag}});
    rst = rst && mCtx.SetKernelRunCbf(RunFFN);
    rst = rst && mCtx.SetKernelMainFunc((void *)ffnKernelFunc);
    rst = rst && mOpInfo.SetContext(&mCtx);
    return rst;
}

bool FFNCase::Run()
{
    if (!mEnable) {
        return true;
    }
    if (!mOpInfo.ProcessTiling(mName)) {
        return false;
    }
    auto *ffnTiling = const_cast<FFNTilingData *>((const FFNTilingData *)(mCtx.GetTilingData()));
    if (ffnTiling == nullptr) {
        LOG_ERR("Tiling failed!");
        return false;
    }
    if (mIsQuant || mCtx.GetTilingKey() == 15) { // 15: 伪量化msd模板
        if (ffnTiling->mm1TilingData.baseN * ffnTiling->mm1TilingData.baseK >
            16384) {                             // 16384: int8场景右矩阵占l0b的1/4
            ffnTiling->mm1TilingData.baseN /= 2; // 2: 将l0b占用大小减半
        }
        if (ffnTiling->mm2TilingData.baseN * ffnTiling->mm2TilingData.baseK >
            16384) {                             // 16384: int8场景右矩阵占l0b的1/4
            ffnTiling->mm2TilingData.baseN /= 2; // 2: 将l0b占用大小减半
        }
    }
    if (ffnTiling->mm1TilingData.M == 0) {
        int mmTilingSize = sizeof(TCubeTiling) / sizeof(int32_t);
        for (int i = 0; i < mmTilingSize; i++) {
            *((int32_t*)(&ffnTiling->mm1TilingData) + i) = 1;
            *((int32_t*)(&ffnTiling->mm2TilingData) + i) = 1;
        }
    }
    if (!mOpInfo.ProcessKernel(mName)) {
        return false;
    }
    return true;
}

bool FFNCase::InitCurrentCasePtr()
{
    Case::mCurrentCasePtr = this;
    return true;
}

Tensor ops::adv::tests::ffn::GenTensor(const char *name, const std::initializer_list<int64_t> &shape,
                                       ge::DataType dType, ge::Format format)
{
    return Tensor(name, shape, "", dType, format);
}