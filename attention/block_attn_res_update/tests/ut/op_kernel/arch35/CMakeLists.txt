# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

function(add_block_attn_res_update_kernel_ut block_attn_res_update_soc_version)
    string(REPLACE "ascend" "Ascend" block_attn_res_update_target_soc "${block_attn_res_update_soc_version}")
    if(NOT TARGET tikicpulib::${block_attn_res_update_target_soc})
        get_target_property(sdk_target tikicpulib::${block_attn_res_update_soc_version} ALIASED_TARGET)
        if(NOT sdk_target)
            set(sdk_target tikicpulib::${block_attn_res_update_soc_version})
        endif()
        add_library(tikicpulib::${block_attn_res_update_target_soc} ALIAS ${sdk_target})
    endif()
    # AddOpTestCase generates a tiling header and SO in this directory; isolate them per SoC.
    set(CMAKE_CURRENT_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/${block_attn_res_update_soc_version}")
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
    set(block_attn_res_update_tiling_files
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../../op_host/op_tiling/arch35/block_attn_res_update_tiling.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../../op_host/op_tiling/arch35/block_attn_res_update_tiling_base.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../../op_host/op_tiling/arch35/block_attn_res_update_full_d_tiling.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/block_attn_res_update_tiling_registry_stub.cpp
    )

    AddOpTestCase(block_attn_res_update "${block_attn_res_update_soc_version}" ""
                  "${block_attn_res_update_tiling_files}")

    # AddOpTestCase loads this temporary tiling SO through CANN's V2 registry
    # while generating the tiling header. Export the same registry entry points
    # as the final op-kernel tiling SO so CANN does not fall back to legacy loading.
    set(block_attn_res_update_tiling_tmp
        "block_attn_res_update_${block_attn_res_update_target_soc}_tiling_tmp")
    if(NOT TARGET ${block_attn_res_update_tiling_tmp})
        return()
    endif()
    set_target_properties(${block_attn_res_update_tiling_tmp} PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
    target_link_libraries(${block_attn_res_update_tiling_tmp} PRIVATE
        -Wl,--whole-archive rt2_registry_static -Wl,--no-whole-archive register)

    # Reuse the ops-tensor kernel-UT runner and PV wrapper instead of putting
    # CPU-simulator compatibility branches in the production kernel.
    set(ops_tensor_kernel_ut_dir "${OPTENSOR_SOURCE_PATH}/tests/ut/op_kernel")
    if(NOT EXISTS "${ops_tensor_kernel_ut_dir}/blaze_kernel_stub.h" OR
       NOT EXISTS "${ops_tensor_kernel_ut_dir}/kernel_ut_runner.h" OR
       NOT EXISTS "${ops_tensor_kernel_ut_dir}/k3_pvwrap.cpp")
        message(FATAL_ERROR "ops-tensor kernel UT support files are unavailable: ${ops_tensor_kernel_ut_dir}")
    endif()

    set(block_attn_res_update_cases_obj
        "block_attn_res_update_${block_attn_res_update_target_soc}_cases_obj")
    target_include_directories(${block_attn_res_update_cases_obj} PRIVATE "${ops_tensor_kernel_ut_dir}")
    target_sources(${OP_KERNEL_MODULE_NAME}_${block_attn_res_update_soc_version}_cases_obj
                   PRIVATE "${ops_tensor_kernel_ut_dir}/k3_pvwrap.cpp")
endfunction()

if(UT_TEST_ALL OR OP_KERNEL_UT)
    if(NOT "${ASCEND_OP_NAME}" STREQUAL "ALL" AND NOT "block_attn_res_update" IN_LIST ASCEND_OP_NAME)
        return()
    endif()
    if(NOT "arch35" IN_LIST ARCH_DIRECTORY)
        return()
    endif()
    set(block_attn_res_update_soc_versions "")
    if("ascend950" IN_LIST ASCEND_COMPUTE_UNIT OR "ascend350" IN_LIST ASCEND_COMPUTE_UNIT)
        list(APPEND block_attn_res_update_soc_versions "Ascend950PR_9599")
    endif()
    if("ascend960dt" IN_LIST ASCEND_COMPUTE_UNIT)
        # Select an actual SDK product alias for the requested platform.
        set(block_attn_res_update_960dt_soc "")
        foreach(soc IN ITEMS Ascend960DT_968B7 ascend960DT_968B7 Ascend960DT_96887 ascend960DT_96887)
            if(TARGET tikicpulib::${soc})
                set(block_attn_res_update_960dt_soc "${soc}")
                break()
            endif()
        endforeach()
        if(NOT block_attn_res_update_960dt_soc)
            message(FATAL_ERROR "BlockAttnResUpdate kernel UT requires a CANN SDK with a 960DT tikicpulib target")
        endif()
        list(APPEND block_attn_res_update_soc_versions "${block_attn_res_update_960dt_soc}")
    endif()
    foreach(soc IN LISTS block_attn_res_update_soc_versions)
        add_block_attn_res_update_kernel_ut("${soc}")
    endforeach()
endif()
