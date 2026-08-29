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
 * \file quant_quant_grouped_matmul_inplace_add_inplace_add_case.cpp
 * \brief QuantGroupedMatmulInplaceAdd 测试用例.
 */

#include "quant_grouped_matmul_inplace_add_case.h"
#include <utility>
#include <tikicpulib.h>
#include <register/op_impl_registry.h>
#include <graph/utils/type_utils.h>
#include "tests/utils/tests_utils_log.h"
#include "tests/utils/io.h"
#include "tests/utils/platform.h"
#include "tiling/qgmmia/tiling_data.h"
#include "op_host/tiling_templates_registry.h"
using namespace Ops::Transformer::OpTiling;
using Case = ops::adv::tests::utils::Case;
using QuantGroupedMatmulInplaceAddCase =
    ops::adv::tests::quant_grouped_matmul_inplace_add::QuantGroupedMatmulInplaceAddCase;
using ops::adv::tests::utils::ReadFile;
using ops::adv::tests::utils::WriteFile;

/**
 * 以下函数声明需要保持与 CMakeList.txt 中调用 OpsTest_Level2_AddOp 函数时 KERNEL_PRIVATE_COMPILE_DEFINITIONS_EXT
 * 参数所控制的 Kernel 入口一致.
 */

#define QUANTGROUPEDMATMULINPLACEADD_KERNEL_PARAM                                                                      \
    (GM_ADDR x1, GM_ADDR x2, GM_ADDR scale2, GM_ADDR groupList, GM_ADDR yIn, GM_ADDR scale1, GM_ADDR y,                \
     GM_ADDR workspace, GM_ADDR tiling)

using QuantGroupedMatmulInplaceAddKernelFunc = void(*) QUANTGROUPEDMATMULINPLACEADD_KERNEL_PARAM;

extern "C" __global__ __aicore__ void quant_grouped_matmul_inplace_add_quant_int8
    QUANTGROUPEDMATMULINPLACEADD_KERNEL_PARAM;

using namespace ops::adv::tests::quant_grouped_matmul_inplace_add;
using ops::adv::tests::utils::Platform;
using ops::adv::tests::utils::TensorIntf;
using ops::adv::tests::utils::TensorList;

enum class KernelParams {
    X1 = 0,
    X2,
    SCALE2,
    GROUP_LIST,
    Y,
    SCALE1
};

bool RunGroupedMatmul(void *func, uint64_t tilingKey, int64_t blockDim, std::vector<TensorIntf *> &inputs,
                      std::vector<TensorIntf *> &output, uint8_t *workspace, uint8_t *tilingData)
{
    (void)blockDim;
    // Kernel 运行
    auto kernelFunc = (QuantGroupedMatmulInplaceAddKernelFunc)func;
    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(kernelFunc, 1, inputs[static_cast<int>(KernelParams::X1)]->GetDevData(),
                inputs[static_cast<int>(KernelParams::X2)]->GetDevData(),
                inputs[static_cast<int>(KernelParams::SCALE2)]->GetDevData(),
                inputs[static_cast<int>(KernelParams::GROUP_LIST)]->GetDevData(),
                inputs[static_cast<int>(KernelParams::Y)]->GetDevData(),
                inputs[static_cast<int>(KernelParams::SCALE1)]->GetDevData(), output[0]->GetDevData(), workspace,
                tilingData);
    return true;
}

QuantGroupedMatmulInplaceAddCase::QuantGroupedMatmulInplaceAddCase()
    : QuantGroupedMatmulInplaceAddCase("Undefined", true, "", OpInfo(), Param(), 0)
{
}

QuantGroupedMatmulInplaceAddCase::QuantGroupedMatmulInplaceAddCase(const char *name, bool enable, const char *dbgInfo,
                                                                   OpInfo opInfo, Param param,
                                                                   int32_t tilingTemplatePriority)
    : Case(name, enable, dbgInfo, tilingTemplatePriority), mOpInfo(std::move(opInfo)), mParam(std::move(param))
{
    this->mOpInfo.mName = "QuantGroupedMatmulInplaceAdd";
}

bool QuantGroupedMatmulInplaceAddCase::Run()
{
    if (!mEnable) {
        return true;
    }
    if (!mOpInfo.ProcessTiling(mName)) {
        return false;
    }
    auto *groupedMatmulTiling =
        const_cast<QGmmInplaceAddTilingDataParams *>((const QGmmInplaceAddTilingDataParams *)(mCtx.GetTilingData()));
    if (groupedMatmulTiling == nullptr) {
        LOG_ERR("Tiling failed!");
        return false;
    }
    if (!mOpInfo.ProcessKernel(mName)) {
        return false;
    }
    return true;
}

bool QuantGroupedMatmulInplaceAddCase::InitParam()
{
    if (mParam.mGroupListData.size() > 0) {
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

void *GetQuantGroupedMatmulInplaceAddKernelFunc()
{
    auto *quantgroupedMatmulInplaceAddKernelFunc = (void *)quant_grouped_matmul_inplace_add_quant_int8;
    return quantgroupedMatmulInplaceAddKernelFunc;
}

bool QuantGroupedMatmulInplaceAddCase::InitOpInfo()
{
    auto *quantgroupedMatmulInplaceAddKernelFunc = GetQuantGroupedMatmulInplaceAddKernelFunc();
    bool rst = mCtx.SetOpName(mOpInfo.mName.c_str());
    rst = rst && mCtx.SetDeterministic(mOpInfo.mCtr.mDeterministic);
    rst = rst && mCtx.SetInputs({&mParam.mTensorLists["x1"], &mParam.mTensorLists["x2"], &mParam.mTensorLists["scale2"],
                                 &mParam.mGroupList, &mParam.mTensorLists["y"], &mParam.mPerTokenScale});
    rst = rst && mCtx.SetOutputs({&mParam.mTensorLists["y"]});
    rst = rst && mCtx.SetAttrs({{"group_list_type", mParam.mGroupListType}, {"group_size", mParam.mGroupSize}});
    rst = rst && mCtx.SetKernelRunCbf(RunGroupedMatmul);
    rst = rst && mCtx.SetKernelMainFunc((void *)quantgroupedMatmulInplaceAddKernelFunc);
    rst = rst && mOpInfo.SetContext(&mCtx);
    return rst;
}

bool QuantGroupedMatmulInplaceAddCase::InitCurrentCasePtr()
{
    Case::mCurrentCasePtr = this;
    return true;
}
