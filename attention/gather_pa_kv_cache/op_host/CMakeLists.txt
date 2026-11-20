# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE
        gather_pa_kv_cache_def.cpp
    )

    get_npu_arch_by_soc_version("${ASCEND_COMPUTE_UNIT}" kv_cache_npu_arch)
    if(kv_cache_npu_arch STREQUAL "dav_3510" OR kv_cache_npu_arch STREQUAL "dav_9201"
       OR kv_cache_npu_arch STREQUAL "dav_9202")
        add_ops_compile_options(
            OP_NAME GatherPaKvCache
            COMPUTE_UNIT "${kv_cache_npu_arch}"
            OPTIONS -mllvm -cce-aicore-dcci-before-kernel-end=false
        )
    else()
        add_ops_compile_options(
            OP_NAME GatherPaKvCache
            OPTIONS --cce-auto-sync=off
                    -Wno-deprecated-declarations
                    -Werror
        )
    endif()
endif()

if(NOT BUILD_OPS_RTY_KERNEL)
    add_op_to_compiled_list()
    add_modules_sources(OPTYPE gather_pa_kv_cache ACLNNTYPE aclnn_exclude)
endif()
