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
    # 算子自己的tiling文件路径
    set(kv_compress_epilog_tiling_files
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../../op_host/arch35/kv_compress_epilog_tiling_arch35.cpp)
    # param1: 算子名 (kernel 命名) param2: soc (与设备/ tikicpulib 一致) param3: 编译宏 param4: tiling 源码
    AddOpTestCase(kv_compress_epilog "Ascend950PR_9599" "-DDTYPE_X=bfloat16_t -DDTYPE_SLOT_MAPPING=int32_t -DDTYPE_CACHE=uint8_t" "${kv_compress_epilog_tiling_files}")
endif()
