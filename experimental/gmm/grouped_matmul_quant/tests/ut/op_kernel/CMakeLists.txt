# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------


# The last argument gives AddOpTestCase the kernel path explicitly because this
# operator is nested under experimental/gmm rather than a top-level module.
if((UT_TEST_ALL OR OP_KERNEL_UT) AND NOT UT_DONE)
    AddOpTestCase(grouped_matmul_quant "ascend910B1" ""
        "${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/grouped_matmul_quant_tiling.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../../op_kernel/grouped_matmul_quant.cpp")
endif()
