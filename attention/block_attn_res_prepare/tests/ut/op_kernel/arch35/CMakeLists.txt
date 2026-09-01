# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if(UT_TEST_ALL OR OP_KERNEL_UT)
    set(block_attn_res_prepare_tiling_files
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../../op_host/op_tiling/block_attn_res_prepare_base_tiling.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../../op_host/op_tiling/block_attn_res_prepare_vector_tiling.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../../op_host/op_tiling/block_attn_res_prepare_mix_tiling.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/block_attn_res_prepare_tiling_registry_stub.cpp
    )

    set(block_attn_res_prepare_soc_version "Ascend950PR_9599")
    set(block_attn_res_prepare_kernel_file
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../../op_kernel/block_attn_res_prepare_apt.cpp)
    AddOpTestCase(block_attn_res_prepare "${block_attn_res_prepare_soc_version}" ""
                  "${block_attn_res_prepare_tiling_files}" "${block_attn_res_prepare_kernel_file}")

    set(block_attn_res_prepare_tiling_tmp
        "block_attn_res_prepare_${block_attn_res_prepare_soc_version}_tiling_tmp")
    set(block_attn_res_prepare_cases_obj
        "block_attn_res_prepare_${block_attn_res_prepare_soc_version}_cases_obj")
    if(TARGET ${block_attn_res_prepare_cases_obj})
        # The temporary tiling SO is loaded directly while generating the kernel-UT tiling header.
        # Include the common opbase utility objects required by the tiling shape diagnostics.
        target_sources(${block_attn_res_prepare_tiling_tmp} PRIVATE
            $<$<TARGET_EXISTS:opbase_util_objs>:$<TARGET_OBJECTS:opbase_util_objs>>)
        target_link_libraries(${block_attn_res_prepare_tiling_tmp} PRIVATE
            -Wl,--whole-archive rt2_registry_static -Wl,--no-whole-archive register)

        set(ops_tensor_kernel_ut_dir "${OPTENSOR_SOURCE_PATH}/tests/ut/op_kernel")
        if(NOT EXISTS "${ops_tensor_kernel_ut_dir}/blaze_kernel_stub.h" OR
           NOT EXISTS "${ops_tensor_kernel_ut_dir}/kernel_ut_runner.h" OR
           NOT EXISTS "${ops_tensor_kernel_ut_dir}/k3_pvwrap.cpp")
            message(FATAL_ERROR "ops-tensor kernel UT support files are unavailable: ${ops_tensor_kernel_ut_dir}")
        endif()

        target_include_directories(${block_attn_res_prepare_cases_obj} PRIVATE
            "${ops_tensor_kernel_ut_dir}" "${OPTENSOR_SOURCE_PATH}/include" ${TENSOR_API} ${TENSOR_API}/include)
        target_sources(${OP_KERNEL_MODULE_NAME}_${block_attn_res_prepare_soc_version}_cases_obj
                       PRIVATE "${ops_tensor_kernel_ut_dir}/k3_pvwrap.cpp")
    endif()
endif()
