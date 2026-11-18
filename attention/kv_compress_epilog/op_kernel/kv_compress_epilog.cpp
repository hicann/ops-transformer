/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file kv_compress_epilog.cpp
 * \brief KV compress epilog kernel 统一入口，按编译目标 arch 预处理器分派（mixed_quant_sparse_flash_mla
 *        同此约定）：
 *        - __CCE_AICORE__ == 220 (DAV_2201: A2 ascend910b / A3 ascend910_93) → arch22/ 下
 *          quant_mode=1 (int8) kernel，KERNEL_TYPE_AIV_ONLY；
 *        - 其余 (arch35: ascend950/ascend960dt/ascend350) → arch35/ 下 mode 0/1/2 kernel。
 *        两个分支互斥编译，arch22 kernel 不引用任何 arch35 头文件，
 *        删除 op_kernel/arch35 不影响 arch22 (quant_mode=1) 的编译与运行。
 */

#if (__CCE_AICORE__ == 220)
#include "arch22/kv_compress_epilog_kernel_arch22.h"
#else
#include "arch35/kv_compress_epilog_kernel.h"
#endif

using namespace AscendC;

extern "C" __global__ __aicore__ void kv_compress_epilog(GM_ADDR cache, GM_ADDR x, GM_ADDR slot_mapping,
                                                         GM_ADDR cache_out, GM_ADDR workspace, GM_ADDR tiling)
{
#if (__CCE_AICORE__ == 220)
    (void)cache_out;
    (void)workspace;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#endif
    TPipe pipe;

    // Get tiling data
    GET_TILING_DATA_WITH_STRUCT(KvCompressEpilogTilingData, tilingDataIn, tiling);
    const KvCompressEpilogTilingData* __restrict__ tilingData = &tilingDataIn;

#if (__CCE_AICORE__ == 220)
    if (TILING_KEY_IS(0)) {
        KvCompressEpilogOps::KvCompressEpilogArch22<DTYPE_X, DTYPE_SLOT_MAPPING, DTYPE_CACHE> op(&pipe);
        op.Init(x, slot_mapping, cache, tilingData);
        op.Process();
    }
#else
    int64_t oriOverflowMode = AscendC::GetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>();
    if (TILING_KEY_IS(0)) {
        KvCompressEpilogOps::KvCompressEpilog<DTYPE_X, DTYPE_SLOT_MAPPING, DTYPE_CACHE> op(&pipe);
        op.Init(x, slot_mapping, cache, tilingData);
        op.Process();
    }
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(oriOverflowMode);
#endif
}
