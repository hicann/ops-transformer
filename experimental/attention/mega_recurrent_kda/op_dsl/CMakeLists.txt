# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


# Native/Torch DSL operator; no ACLNN adapter is required.
add_op_to_compiled_list()
if(BUILD_OPEN_PROJECT)
    # Keep the baseline framework contract: create its empty Host objects.
    add_modules_sources()
    if(CANNBOTDSL_TARGET STREQUAL "dav-9201")
        enable_cannbotdsl_kernel(mega_recurrent_kda "${CMAKE_CURRENT_LIST_DIR}/arch92")
    else()
        enable_cannbotdsl_kernel(mega_recurrent_kda "${CMAKE_CURRENT_LIST_DIR}/arch35")
    endif()
endif()
