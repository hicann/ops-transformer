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
 * \file grouped_matmul_case.cpp
 * \brief GroupedMatmul 测试用例.
 */

#include "grouped_matmul_case.h"
#include <utility>
#include <tikicpulib.h>
#include <register/op_impl_registry.h>
#include <graph/utils/type_utils.h>
#include "tests/utils/log.h"
#include "tests/utils/io.h"
#include "tests/utils/platform.h"
#include "tiling/gmm/tiling_data.h"
#include "tiling_base/tiling_templates_registry.h"
using namespace Ops::Transformer::OpTiling;
using Case = ops::adv::tests::utils::Case;
using GroupedMatmulCase = ops::adv::tests::grouped_matmul::GroupedMatmulCase;
using ops::adv::tests::utils::ReadFile;
using ops::adv::tests::utils::WriteFile;

/**
 * 以下函数声明需要保持与 CMakeList.txt 中调用 OpsTest_Level2_AddOp 函数时 KERNEL_PRIVATE_COMPILE_DEFINITIONS_EXT
 * 参数所控制的 Kernel 入口一致.
 */

#define GROUPEDMATMUL_KERNEL_PARAM                                                  \
    (GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR scale,                        \
     GM_ADDR offset, GM_ADDR antiquantScale, GM_ADDR antiquantOffset,               \
     GM_ADDR groupList, GM_ADDR perTokenScale, GM_ADDR y,                           \
     GM_ADDR workspace, GM_ADDR tiling)

using GroupedMatmulKernelFunc = void(*) GROUPEDMATMUL_KERNEL_PARAM;

__global__ __aicore__ void grouped_matmul_fp16 GROUPEDMATMUL_KERNEL_PARAM;
__global__ __aicore__ void grouped_matmul_bf16 GROUPEDMATMUL_KERNEL_PARAM;
__global__ __aicore__ void grouped_matmul_fp32 GROUPEDMATMUL_KERNEL_PARAM;
__global__ __aicore__ void grouped_matmul_quant_int8 GROUPEDMATMUL_KERNEL_PARAM;
__global__ __aicore__ void grouped_matmul_quant_bf16 GROUPEDMATMUL_KERNEL_PARAM;
__global__ __aicore__ void grouped_matmul_quant_fp16 GROUPEDMATMUL_KERNEL_PARAM;
__global__ __aicore__ void grouped_matmul_quant_int32 GROUPEDMATMUL_KERNEL_PARAM;
__global__ __aicore__ void grouped_matmul_a16w8_bf16 GROUPEDMATMUL_KERNEL_PARAM;
__global__ __aicore__ void grouped_matmul_a16w8_fp16 GROUPEDMATMUL_KERNEL_PARAM;
__global__ __aicore__ void grouped_matmul_a16w4_bf16 GROUPEDMATMUL_KERNEL_PARAM;
__global__ __aicore__ void grouped_matmul_a16w4_fp16 GROUPEDMATMUL_KERNEL_PARAM;
__global__ __aicore__ void grouped_matmul_a8w4_msd GROUPEDMATMUL_KERNEL_PARAM;
__global__ __aicore__ void grouped_matmul_a4w4 GROUPEDMATMUL_KERNEL_PARAM;


using namespace ops::adv::tests::grouped_matmul;
using ops::adv::tests::utils::Platform;
using ops::adv::tests::utils::TensorList;
using ops::adv::tests::utils::TensorIntf;

enum class KernelParams {
    X = 0,
    WEIGHT,
    BIAS,
    SCALE,
    OFFSET,
    ANTIQUANT_SCALE,
    ANTIQUANT_OFFSET,
    GROUP_LIST,
    PER_TOKEN_SCALE
};

bool RunGroupedMatmul(void *func, uint64_t tilingKey, int64_t blockDim, std::vector<TensorIntf *> &inputs,
                      std::vector<TensorIntf *> &output, uint8_t *workspace, uint8_t *tilingData)
{
    (void)blockDim;
    // Kernel 运行
    auto kernelFunc = (GroupedMatmulKernelFunc)func;
    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(kernelFunc, 1,
                inputs[static_cast<int>(KernelParams::X)]->GetDevData(),
                inputs[static_cast<int>(KernelParams::WEIGHT)]->GetDevData(),
                inputs[static_cast<int>(KernelParams::BIAS)]->GetDevData(),
                inputs[static_cast<int>(KernelParams::SCALE)]->GetDevData(),
                inputs[static_cast<int>(KernelParams::OFFSET)]->GetDevData(),
                inputs[static_cast<int>(KernelParams::ANTIQUANT_SCALE)]->GetDevData(),
                inputs[static_cast<int>(KernelParams::ANTIQUANT_OFFSET)]->GetDevData(),
                inputs[static_cast<int>(KernelParams::GROUP_LIST)]->GetDevData(),
                inputs[static_cast<int>(KernelParams::PER_TOKEN_SCALE)]->GetDevData(),
                output[0]->GetDevData(),
                workspace, tilingData);
    return true;
}

bool RunTemplateGroupedMatmul(std::function<void(GMM_INPUT_DTYPE)> func, uint64_t tilingKey,
                              int64_t blockDim, std::vector<TensorIntf *> &inputs,
                              std::vector<TensorIntf *> &output, uint8_t *workspace, uint8_t *tilingData)
{
    (void)blockDim;
    // Kernel 运行
    ICPU_SET_TILING_KEY(tilingKey);
    // ICPU_RUN_KF(func, blockDim,
    //             inputs[static_cast<int>(KernelParams::X)]->GetDevData(),
    //             inputs[static_cast<int>(KernelParams::WEIGHT)]->GetDevData(),
    //             inputs[static_cast<int>(KernelParams::BIAS)]->GetDevData(),
    //             inputs[static_cast<int>(KernelParams::SCALE)]->GetDevData(),
    //             inputs[static_cast<int>(KernelParams::OFFSET)]->GetDevData(),
    //             inputs[static_cast<int>(KernelParams::ANTIQUANT_SCALE)]->GetDevData(),
    //             inputs[static_cast<int>(KernelParams::ANTIQUANT_OFFSET)]->GetDevData(),
    //             inputs[static_cast<int>(KernelParams::GROUP_LIST)]->GetDevData(),
    //             inputs[static_cast<int>(KernelParams::PER_TOKEN_SCALE)]->GetDevData(),
    //             output[0]->GetDevData(),
    //             workspace, tilingData);
    return true;
}

GroupedMatmulCase::GroupedMatmulCase() : GroupedMatmulCase("Undefined", true, "", OpInfo(), Param(), 0)
{
}

GroupedMatmulCase::GroupedMatmulCase(const char *name, bool enable, const char *dbgInfo, OpInfo opInfo, Param param,
                                     int32_t tilingTemplatePriority)
    : Case(name, enable, dbgInfo, tilingTemplatePriority), mOpInfo(std::move(opInfo)), mParam(std::move(param))
{
    this->mOpInfo.mName = "GroupedMatmul";
}

bool GroupedMatmulCase::Run()
{
    if(!mEnable) {
        return true;
    }
    if (!mOpInfo.ProcessTiling(mName)) {
        return false;
    }
    auto *groupedMatmulTiling = const_cast<GMMTilingData *>((const GMMTilingData *)(mCtx.GetTilingData()));
    if (groupedMatmulTiling == nullptr) {
        LOG_ERR("Tiling failed!");
        return false;
    }
    if (!mOpInfo.ProcessKernel(mName)) {
        return false;
    }
    return true;
}

bool GroupedMatmulCase::InitParam()
{
    if(mParam.mGroupListData.size() > 0) {
        size_t dataSize = mParam.mGroupListData.size() * sizeof(int64_t);
        uint8_t *addr = mParam.mGroupList.AllocDevData(0, dataSize);
        if (addr == nullptr) {
            LOG_ERR("TensorList(%s, %zu) AllocDevData Failed.", mParam.mGroupList.Name().c_str(), dataSize);
            return false;
        }
        std::string fileName = this->mName + "_groupList.bin";
        if (!WriteFile(fileName, mParam.mGroupListData.data(), dataSize)) {
            LOG_ERR("Write groupList data to file[%s] failed", fileName.c_str());
            return false;
        }
        if (!ReadFile(fileName, dataSize, addr, dataSize)) {
            LOG_ERR("Read groupList data[%s] to tensor failed", fileName.c_str());
            return false;
        }
    }
    return true;
}

// void *GetGroupedMatmulKernelFunc(Param& mParam) {
//     auto *groupedMatmulKernelFunc = (void *)grouped_matmul_fp16;
//     if (mParam.mTensorLists["x"].GetDataType() == mParam.mTensorLists["weight"].GetDataType()) {
//         if (mParam.mTensorLists["weight"].GetDataType() == ge::DataType::DT_INT8) { //量化
//             if (mParam.mTensorLists["y"].GetDataType() == ge::DataType::DT_INT8) {
//                 groupedMatmulKernelFunc = (void *)grouped_matmul_quant_int8;
//             } else if (mParam.mTensorLists["y"].GetDataType() == ge::DataType::DT_FLOAT16) {
//                 groupedMatmulKernelFunc = (void *)grouped_matmul_quant_fp16;
//             } else if (mParam.mTensorLists["y"].GetDataType() == ge::DataType::DT_BF16) {
//                 groupedMatmulKernelFunc = (void *)grouped_matmul_quant_bf16;
//             } else {
//                 groupedMatmulKernelFunc = (void *)grouped_matmul_quant_int32;
//             }
//         } else if (mParam.mTensorLists["x"].GetDataType() == ge::DataType::DT_INT4) {
//             groupedMatmulKernelFunc = (void *)grouped_matmul_a4w4;
//         } else {//非量化
//             if (mParam.mTensorLists["weight"].GetDataType() == ge::DataType::DT_FLOAT16) {
//                 groupedMatmulKernelFunc = (void *)grouped_matmul_fp16;
//             } else if (mParam.mTensorLists["weight"].GetDataType() == ge::DataType::DT_BF16) {
//                 groupedMatmulKernelFunc = (void *)grouped_matmul_bf16;
//             } else {
//                 groupedMatmulKernelFunc = (void *)grouped_matmul_fp32;
//             }
//         }
//     } else {//伪量化
//         if (mParam.mTensorLists["weight"].GetDataType() == ge::DataType::DT_INT8) {
//             if (mParam.mTensorLists["x"].GetDataType() == ge::DataType::DT_FLOAT16) {
//                 groupedMatmulKernelFunc = (void *)grouped_matmul_a16w8_fp16;
//             } else {
//                 groupedMatmulKernelFunc = (void *)grouped_matmul_a16w8_bf16;
//             }
//         } else {
//             if (mParam.mTensorLists["x"].GetDataType() == ge::DataType::DT_INT8 && mParam.mTensorLists["weight"].GetDataType() == ge::DataType::DT_INT4) {
//                 groupedMatmulKernelFunc = (void *)grouped_matmul_a8w4_msd;
//             }
//             else if (mParam.mTensorLists["x"].GetDataType() == ge::DataType::DT_FLOAT16) {
//                 groupedMatmulKernelFunc = (void *)grouped_matmul_a16w4_fp16;
//             } else {
//                 groupedMatmulKernelFunc = (void *)grouped_matmul_a16w4_bf16;
//             }
//         }
//     }
//     return groupedMatmulKernelFunc;
// }
std::function<void(GROUPEDMATMUL_KERNEL_PARAM)> groupedMatmulKernelTemplateFunc;

bool GroupedMatmulCase::InitOpInfo()
{
    // auto *groupedMatmulKernelFunc = GetGroupedMatmulKernelFunc(mParam);
    bool rst = mCtx.SetOpName(mOpInfo.mName.c_str());
    rst = rst && mCtx.SetDeterministic(mOpInfo.mCtr.mDeterministic);
    rst = rst && mCtx.SetInputs({&mParam.mTensorLists["x"], &mParam.mTensorLists["weight"], &mParam.mTensorLists["bias"],
                                 &mParam.mTensorLists["scale"], &mParam.mTensorLists["offset"],
                                 &mParam.mTensorLists["antiquant_scale"], &mParam.mTensorLists["antiquant_offset"],
                                 &mParam.mGroupList, &mParam.mPerTokenScale});
    rst = rst && mCtx.SetOutputs({&mParam.mTensorLists["y"]});
    rst = rst && mCtx.SetAttrs({{"split_item", mParam.mSplitItem},
                                {"dtype", mParam.mDtype},
                                {"transpose_weight", mParam.mTransposeWeight},
                                {"transpose_x", mParam.mTransposeX},
                                {"group_type", mParam.mGroupType},
                                {"group_list_type", mParam.mGroupListType},
                                {"act_type", mParam.mActType},
                                {"tuning_config", mParam.mTuningConfig}
                                });
    // rst = rst && mCtx.SetKernelRunCbf(RunGroupedMatmul);
    // rst = rst && mCtx.SetKernelMainFunc((void *)groupedMatmulKernelFunc);
    rst = rst && mCtx.SetKernelRunTemplateCbf(RunTemplateGroupedMatmul);
    rst = rst && mCtx.SetKernelTemplateMainFunc(groupedMatmulKernelTemplateFunc);
    rst = rst && mOpInfo.SetContext(&mCtx);
    return rst;
}

bool GroupedMatmulCase::InitCurrentCasePtr()
{
    Case::mCurrentCasePtr = this;
    return true;
}

