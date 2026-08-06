# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.18.4)

if(UT_TEST_ALL OR OP_GRAPH_UT)
  set(OP_GRAPH_UT_EXE ${PKG_NAME}_op_graph_ut)

  add_opgraph_ut_modules(${OP_GRAPH_MODULE_NAME})

  add_executable(${OP_GRAPH_UT_EXE}
      test_op_graph_main.cpp
      )
  foreach(OPBASE_OBJ_TARGET opbase_util_objs opbase_infer_objs)
    if(TARGET ${OPBASE_OBJ_TARGET})
      add_dependencies(${OP_GRAPH_UT_EXE} ${OPBASE_OBJ_TARGET})
    endif()
  endforeach()
  target_compile_definitions(${OP_GRAPH_UT_EXE} PRIVATE LOG_CPP)
  target_compile_options(${OP_GRAPH_UT_EXE} PUBLIC -fPIE -fno-access-control)
  target_include_directories(${OP_GRAPH_UT_EXE} PRIVATE
      ${ASCEND_DIR}/pkg_inc
      )
  target_link_directories(${OP_GRAPH_UT_EXE} PRIVATE
      ${ASCEND_DIR}/${SYSTEM_PREFIX}/lib64
      ${CMAKE_BINARY_DIR}/es_packages/lib64
      )
  target_link_libraries(${OP_GRAPH_UT_EXE} PRIVATE
      $<BUILD_INTERFACE:intf_llt_pub_asan_cxx17>
      -Wl,--no-as-needed
      -Wl,--whole-archive
          ${OP_GRAPH_MODULE_NAME}_cases
          $<$<TARGET_EXISTS:${GRAPH_PLUGIN_NAME}_obj>:${GRAPH_PLUGIN_NAME}_obj>
      -Wl,--no-whole-archive
      metadef
      graph
      graph_base
      exe_graph
      ge_common
      register
      platform
      gtest
      c_sec
      ascendalog
      unified_dlog
      dl
      -Wl,--as-needed
      )

  if(ENABLE_UT_EXEC)
    add_custom_command(
      TARGET ${OP_GRAPH_UT_EXE}
      POST_BUILD
      COMMAND
        ${CMAKE_COMMAND} -E env LD_LIBRARY_PATH=$ENV{LD_LIBRARY_PATH}:${CMAKE_BINARY_DIR}/es_packages/lib64
        ./${OP_GRAPH_UT_EXE}
      COMMENT "Run ops op_graph utest"
      )
  endif()
endif()
