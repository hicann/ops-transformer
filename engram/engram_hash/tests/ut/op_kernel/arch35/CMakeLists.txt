# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if (UT_TEST_ALL OR OP_KERNEL_UT)
    # engram_hash 是 SIMT (__simt_vf__) 算子: VF 线程数 = validTokens*chunkCols 动态非 32 倍数,
    # 且 kernel 用裸 (__ubuf__*)0 基址+偏移访问 UB; CPU 仿真 (tikicpulib) 对这两者均不支持
    # (线程数非 32 倍数 SIGABRT, 非 0 偏移 MTE2 写 PV_MEM SIGSEGV)，故不注册 kernel 仿真 UT
    # set(engram_hash_tiling_files
    #     ${CMAKE_CURRENT_SOURCE_DIR}/../../../../op_host/engram_hash_tiling_arch35.cpp)
    # param1: 算子名 (kernel 命名) param2: soc (与设备/ tikicpulib 一致) param3: 编译宏 param4: tiling 源码
    # AddOpTestCase(engram_hash "Ascend950PR_9599"
    #     "-DDTYPE_NGRAM_TOKEN_IDS=int32_t -DDTYPE_MULTIPLIERS=int64_t -DDTYPE_VOCAB_SIZES=int32_t -DDTYPE_OFFSETS=int32_t -DDTYPE_HASH_INDICES=int32_t"
    #     "${engram_hash_tiling_files}")
endif()
