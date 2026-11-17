# ---------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ---------------------------------------------------------------------------------------------------------
get_filename_component(PARENT_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)
get_filename_component(OP_NAME ${PARENT_DIR} NAME)

if (BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG)
  if (NOT (UT_TEST_ALL OR OP_KERNEL_AICPU_UT))
    add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)
    set(CMAKE_CXX_COMPILER ${ASCEND_DIR}/toolkit/toolchain/hcc/bin/aarch64-target-linux-gnu-g++)
  endif()
  get_filename_component(_AICPU_PARENT_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)
  get_filename_component(_AICPU_OP_NAME ${_AICPU_PARENT_DIR} NAME)
  check_aicpu_builtin_pkg_admission(${_AICPU_OP_NAME} _AICPU_BUILTIN_PKG_ALLOWED)

  file(GLOB_RECURSE JSON_FILE ${CMAKE_CURRENT_SOURCE_DIR}/*.json)
  if(JSON_FILE AND _AICPU_BUILTIN_PKG_ALLOWED)
    set_property(GLOBAL APPEND PROPERTY AICPU_JSON_FILES ${JSON_FILE})
  endif()

  file(GLOB AICPU_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*_aicpu*.cpp)
  message(STATUS "[ ${OP_NAME}] Found aicpu sources: ${AICPU_SRC}, ascend dir: ${ASCEND_DIR}, ophsot name: ${OPHOST_NAME}")

  if(AICPU_SRC AND _AICPU_BUILTIN_PKG_ALLOWED)
    set(OBJ_NAME  ${OP_NAME}_cust_obj)
    add_aicpu_cust_kernel_modules(${OBJ_NAME})
    target_sources(${OBJ_NAME} PRIVATE ${AICPU_SRC})
  endif()
endif()

if(UT_TEST_ALL OR OP_KERNEL_AICPU_UT)
    AddAicpuOpTestCase( ${OP_NAME})
endif()
