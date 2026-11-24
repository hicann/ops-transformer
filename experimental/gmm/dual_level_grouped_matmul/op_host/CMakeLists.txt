# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under CANN Open Software License Agreement Version 2.0.
add_op_to_compiled_list()
if(BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE dual_level_grouped_matmul_def.cpp)
endif()
if(UT_TEST_ALL OR OP_HOST_UT OR OP_API_UT)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../tests ${CMAKE_CURRENT_BINARY_DIR}/tests)
endif()
add_ops_compile_options(OP_NAME DualLevelGroupedMatmul OPTIONS --cce-auto-sync=on)
if(NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(
        OP_API_INDEPENDENT ON
        OP_API_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../op_api
        OPTYPE dual_level_grouped_matmul
        ACLNNTYPE aclnn_exclude
    )
    target_include_directories(${OPHOST_NAME}_tiling_obj PRIVATE "${OPTENSOR_SOURCE_PATH}/include")
endif()
