/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file torch_interface.cpp
 * \brief Simplified PyTorch entry for flash_attn_minimal.
 *        Fixed shape: B=1,N1=32,N2=1,S1=8192,S2=8192,D=128.
 *        No tiling, no isCausal, no attnMask, no combine.
 */

#include <Python.h>
#include <torch/all.h>
#include <torch/library.h>
#include "acl/acl.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"

#include "tiling/platform/platform_ascendc.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "kernel_operator_list_tensor_intf.h"
#include "op_kernel/fa_kernel_interface.h"

namespace ascend_ops {
namespace FA {

bool CheckInput(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v)
{
    if (!q.defined() || q.numel() == 0) {
        printf("Error: query undefined or empty.\n");
        return false;
    }
    if (!k.defined() || k.numel() == 0) {
        printf("Error: key undefined or empty.\n");
        return false;
    }
    if (!v.defined() || v.numel() == 0) {
        printf("Error: value undefined or empty.\n");
        return false;
    }
    if (q.scalar_type() != at::kBFloat16 || k.scalar_type() != at::kBFloat16 || v.scalar_type() != at::kBFloat16) {
        printf("Error: Q/K/V dtype only support bfloat16.\n");
        return false;
    }
    if (q.dim() != 4 || k.dim() != 4 || v.dim() != 4) {
        printf("Error: Q/K/V only support 4-dim BSND layout.\n");
        return false;
    }
    if (q.size(0) != k.size(0) || q.size(0) != v.size(0)) {
        printf("Error: batch size mismatch.\n");
        return false;
    }
    if (k.size(1) != v.size(1)) {
        printf("Error: key/value S mismatch.\n");
        return false;
    }
    if (q.size(3) != 128 || k.size(3) != 128 || v.size(3) != 128) {
        printf("Error: D must be 128.\n");
        return false;
    }
    if (q.size(1) != k.size(1)) {
        printf("Error: S1 must be equal to S2.\n");
        return false;
    }
    if (q.size(1) % 128 != 0) {
        printf("Error: S1 must be divisible by 128.\n");
        return false;
    }
    if (q.size(2) < k.size(2)) {
        printf("Error: N1 must be >= N2.\n");
        return false;
    }
    if (q.size(2) % k.size(2) != 0) {
        printf("Error: N1 must be divisible by N2.\n");
        return false;
    }
    return true;
}

at::Tensor FlashAttnNpu(const at::Tensor &q,
                     const at::Tensor &k,
                     const at::Tensor &v,
                     double softmaxScale)
{
    TORCH_CHECK(CheckInput(q, k, v), "FA input validation failed");

    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();

    int32_t devIdx = q.device().index();
    c10_npu::NPUStream stream = c10_npu::getCurrentNPUStream(devIdx);
    void *aclstream = stream.stream(true);

    at::Tensor output = at::empty(q.sizes(), q.options());

    uint32_t B  = static_cast<uint32_t>(q.size(0));
    uint32_t N1 = static_cast<uint32_t>(q.size(2));
    uint32_t N2 = static_cast<uint32_t>(k.size(2));
    uint32_t S1 = static_cast<uint32_t>(q.size(1));
    uint32_t S2 = static_cast<uint32_t>(k.size(1));

    constexpr uint32_t fixedAicNum = 32;
    constexpr uint32_t fixedAivNum = 64;
    uint32_t blockDimToBeSet = ascendcPlatform->CalcTschBlockDim(
        fixedAivNum, ascendcPlatform->GetCoreNumAic(), ascendcPlatform->GetCoreNumAiv());

    auto workspaceTensor = at::empty({0}, at::TensorOptions().dtype(at::kByte).device(q.options().device()));

    auto aclCal = [=]() -> int {
        GM_ADDR gq  = (GM_ADDR)(q.data_ptr());
        GM_ADDR gk  = (GM_ADDR)(k.data_ptr());
        GM_ADDR gv  = (GM_ADDR)(v.data_ptr());
        GM_ADDR go  = (GM_ADDR)(output.data_ptr());
        GM_ADDR gws = (GM_ADDR)(workspaceTensor.data_ptr());
        FaKernel<<<blockDimToBeSet, nullptr, aclstream>>>(gq, gk, gv, go, gws, blockDimToBeSet, (float)softmaxScale,
            B, N1, N2, S1, S2);
        return 0;
    };
    at_npu::native::OpCommand::RunOpApiV2("FA", aclCal);

    auto syncRet = aclrtSynchronizeStream(aclstream);
    TORCH_CHECK(syncRet == ACL_ERROR_NONE, "FA kernel synchronize failed");

    return output;
}

TORCH_LIBRARY_IMPL(ascend_ops, PrivateUse1, m)
{
    m.impl("flash_attn_minimal", FlashAttnNpu);
}

TORCH_LIBRARY(ascend_ops, m)
{
    m.def(R"(flash_attn_minimal(Tensor q,
             Tensor k,
             Tensor v,
             float softmaxScale = 0) -> Tensor)");
}
}  // namespace FA
}  // namespace ascend_ops

extern "C" {
PyObject* PyInit__C(void)
{
    static struct PyModuleDef module_def = {
        PyModuleDef_HEAD_INIT, "_C", NULL, -1, NULL,
    };
    return PyModule_Create(&module_def);
}
}

