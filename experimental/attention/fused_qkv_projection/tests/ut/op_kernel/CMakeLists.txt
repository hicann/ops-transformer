# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# op_kernel UT 暂不启用（tiling 文件依赖 lib/tiling_api.h，kernel UT 环境缺少该包含路径）
# if(UT_TEST_ALL OR OP_KERNEL_UT)
#     set(fused_qkv_projection_tiling_files
#         ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/fused_qkv_projection_tiling.cpp)
#     AddOpTestCase(fused_qkv_projection "ascend910B1" "-DDTYPE_X=float" "${fused_qkv_projection_tiling_files}")
# endif()
