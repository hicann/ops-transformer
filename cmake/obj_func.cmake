# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

# useage: add_modules_sources(DIR OPTYPE ACLNNTYPE)
# ACLNNTYPE 支持类型aclnn/aclnn_inner/aclnn_exclude
# OPTYPE 和 ACLNNTYPE 需一一对应

# 用于custom自定算子包host侧obj生成
macro(add_modules_sources)
  set(multiValueArgs OPTYPE ACLNNTYPE)

  cmake_parse_arguments(MODULE "" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

  # opapi 默认全部编译
  file(GLOB OPAPI_SRCS ${SOURCE_DIR}/op_api/*.cpp)
  if (OPAPI_SRCS)
    # aclnn
    add_opapi_modules()
    target_sources(${OPHOST_NAME}_opapi_obj PRIVATE ${OPAPI_SRCS})
  else()
    if (NOT TARGET ${OPHOST_NAME}_opapi_obj)
      add_opapi_modules()
      add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/opapi_stub.cpp
          COMMAND touch ${CMAKE_CURRENT_BINARY_DIR}/opapi_stub.cpp
      )
      target_sources(${OPHOST_NAME}_opapi_obj PRIVATE
            ${CMAKE_CURRENT_BINARY_DIR}/opapi_stub.cpp
      )
    endif()
  endif()
  file(GLOB OPAPI_HEADERS ${SOURCE_DIR}/op_api/aclnn_*.h)
  if (OPAPI_HEADERS)
    target_sources(${OPHOST_NAME}_aclnn_exclude_headers INTERFACE ${OPAPI_HEADERS})
  endif()

  # 是否编译该算子已经由op_add_subdirectory和每个二级目录判断完毕，默认走到这里全编

  file(GLOB OPINFER_SRCS ${SOURCE_DIR}/*_infershape*.cpp)
  if (OPINFER_SRCS)
    # proto
    add_infer_modules()
    target_sources(${OPHOST_NAME}_infer_obj PRIVATE ${OPINFER_SRCS})
  else()
    if (NOT TARGET ${OPHOST_NAME}_infer_obj)
      add_library(${OPHOST_NAME}_infer_obj OBJECT)
      add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/proto_stub.cpp
          COMMAND touch ${CMAKE_CURRENT_BINARY_DIR}/proto_stub.cpp
      )
      target_sources(${OPHOST_NAME}_infer_obj PRIVATE
            ${CMAKE_CURRENT_BINARY_DIR}/proto_stub.cpp
      )
    endif()
  endif()

  file(GLOB_RECURSE SUB_OPTILING_SRC ${SOURCE_DIR}/op_tiling/*.cpp)
  file(GLOB OPTILING_SRCS 
      ${SOURCE_DIR}/*fallback*.cpp
      ${SOURCE_DIR}/*_tiling*.cpp
      ${SOURCE_DIR}/op_tiling/arch35/*.cpp
      ${SOURCE_DIR}/../op_graph/fallback_*.cpp
      ${SOURCE_DIR}/../graph_plugin/fallback_*.cpp)
  if (OPTILING_SRCS OR SUB_OPTILING_SRC)
    # tiling
    add_tiling_modules()
    target_sources(${OPHOST_NAME}_tiling_obj PRIVATE ${OPTILING_SRCS} ${SUB_OPTILING_SRC})
    # target_include_directories(${OPHOST_NAME}_tiling_obj PRIVATE ${SOURCE_DIR}/../../ ${SOURCE_DIR})
  endif()

  file(GLOB AICPU_SRCS ${MODULE_DIR}/*_aicpu*.cpp)
  if (AICPU_SRCS)
    add_aicpu_kernel_modules()
    target_sources(${OPHOST_NAME}_aicpu_obj PRIVATE ${AICPU_SRCS})
  endif()
  if (MODULE_OPTYPE)
    list(LENGTH MODULE_OPTYPE OpTypeLen)
    list(LENGTH MODULE_ACLNNTYPE AclnnTypeLen)
    if(NOT ${OpTypeLen} EQUAL ${AclnnTypeLen})
      message(FATAL_ERROR "OPTYPE AND ACLNNTYPE Should be One-to-One")
    endif()
    math(EXPR index "${OpTypeLen} - 1")
    foreach(i RANGE ${index})
      list(GET MODULE_OPTYPE ${i} OpType)
      list(GET MODULE_ACLNNTYPE ${i} AclnnType)
      if (${AclnnType} STREQUAL "aclnn" OR ${AclnnType} STREQUAL "aclnn_inner" OR ${AclnnType} STREQUAL "aclnn_exclude")
        file(GLOB OPDEF_SRCS ${SOURCE_DIR}/${OpType}_def*.cpp)

        if (OPDEF_SRCS)
          target_sources(${OPHOST_NAME}_opdef_${AclnnType}_obj INTERFACE ${OPDEF_SRCS})
        endif()
      elseif(${AclnnType} STREQUAL "no_need_aclnn")
        message(STATUS "aicpu or host aicpu no need aclnn.")
      else()
        message(FATAL_ERROR "ACLNN TYPE UNSPPORTED, ONLY SUPPORT aclnn/aclnn_inner/aclnn_exclude")
      endif()
    endforeach()
  else()
    file(GLOB OPDEF_SRCS ${SOURCE_DIR}/*_def*.cpp)
    if(OPDEF_SRCS)
      message(FATAL_ERROR
      "Should Manually specify aclnn/aclnn_inner/aclnn_exclude\n"
      "usage: add_modules_sources(OPTYPE optypes ACLNNTYPE aclnntypes)\n"
      "example: add_modules_sources(OPTYPE add ACLNNTYPE aclnn_exclude)"
      )
    endif()
  endif()
endmacro()

macro(add_mc2_modules_sources)
  set(multiValueArgs OPTYPE ACLNNTYPE)

  cmake_parse_arguments(MODULE "" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

  #opapi 默认全部编译
  file(GLOB OPAPI_SRCS ${SOURCE_DIR}/op_api/*.cpp)
  if (OPAPI_SRCS)
    # aclnn
    add_opapi_modules()
    target_sources(${OPHOST_NAME}_opapi_obj PRIVATE ${OPAPI_SRCS})
  endif()

  # 获取算子层级目录名称，判断是否编译该算子
  get_filename_component(PARENT_DIR ${SOURCE_DIR} DIRECTORY)
  get_filename_component(OP_NAME ${PARENT_DIR} NAME)
  list(FIND ASCEND_OP_NAME ${OP_NAME} INDEX)
  # 记录全局的COMPILED_OPS和COMPILED_OP_DIRS，其中COMPILED_OP_DIRS只记录到算子名，例如moe/moe_token_permute_with_routing_map_grad
  set(COMPILED_OPS ${COMPILED_OPS} ${OP_NAME} CACHE STRING "Compiled Ops" FORCE)
  set(COMPILED_OP_DIRS ${COMPILED_OP_DIRS} ${PARENT_DIR} CACHE STRING "Compiled Ops Dirs" FORCE)

  file(GLOB OPINFER_SRCS ${SOURCE_DIR}/*_infershape*.cpp)
  if (OPINFER_SRCS)
    # proto
    add_infer_modules()
    target_sources(${OPHOST_NAME}_infer_obj PRIVATE ${OPINFER_SRCS})
  endif()

  file(GLOB_RECURSE OPTILING_SRCS
      ${SOURCE_DIR}/op_tiling/*.cpp
      ${SOURCE_DIR}/op_tiling/arch35/*.cpp
      ${SOURCE_DIR}/../op_graph/fallback*.cpp
  )
  if (OPTILING_SRCS)
    # tiling
    add_tiling_modules()
    target_sources(${OPHOST_NAME}_tiling_obj PRIVATE 
      ${OPTILING_SRCS}
      ${OPS_TRANSFORMER_DIR}/mc2/common/src/matmul_formulaic_tiling.cpp
      ${OPS_TRANSFORMER_DIR}/mc2/common/src/mc2_tiling_utils.cpp
      ${OPS_TRANSFORMER_DIR}/mc2/common/src/mc2_log.cpp
      ${OPS_TRANSFORMER_DIR}/mc2/3rd/ops_legacy/op_tiling/op_cache_tiling.cpp
      ${OPS_TRANSFORMER_DIR}/mc2/3rd/ops_legacy/op_tiling/runtime_kb_api.cpp
      ${OPS_TRANSFORMER_DIR}/mc2/3rd/ops_legacy/op_api/op_legacy_api.cpp
    )
  endif()

  file(GLOB GENTASK_SRCS
      ${SOURCE_DIR}/../op_graph/*_gen_task*.cpp #各个算子的gen task 文件
  )
  if(GENTASK_SRCS)
    add_opmaster_ct_gentask_modules()
    target_sources(${OPHOST_NAME}_opmaster_ct_gentask_obj PRIVATE ${GENTASK_SRCS})
  endif()

  file(GLOB AICPU_SRCS ${MODULE_DIR}/*_aicpu*.cpp)
  if (AICPU_SRCS)
    add_aicpu_kernel_modules()
    target_sources(${OPHOST_NAME}_aicpu_obj PRIVATE ${AICPU_SRCS})
  endif()
  if (MODULE_OPTYPE)
    list(LENGTH MODULE_OPTYPE OpTypeLen)
    list(LENGTH MODULE_ACLNNTYPE AclnnTypeLen)
    if(NOT ${OpTypeLen} EQUAL ${AclnnTypeLen})
      message(FATAL_ERROR "OPTYPE AND ACLNNTYPE Should be One-to-One")
    endif()
    math(EXPR index "${OpTypeLen} - 1")
    foreach(i RANGE ${index})
      list(GET MODULE_OPTYPE ${i} OpType)
      list(GET MODULE_ACLNNTYPE ${i} AclnnType)
      if (${AclnnType} STREQUAL "aclnn" OR ${AclnnType} STREQUAL "aclnn_inner" OR ${AclnnType} STREQUAL "aclnn_exclude")
        file(GLOB OPDEF_SRCS ${SOURCE_DIR}/${OpType}_def*.cpp)

        if (OPDEF_SRCS)
          target_sources(${OPHOST_NAME}_opdef_${AclnnType}_obj INTERFACE ${OPDEF_SRCS})
        endif()
      elseif(${AclnnType} STREQUAL "no_need_aclnn")
        message(STATUS "aicpu or host aicpu no need aclnn.")
      else()
        message(FATAL_ERROR "ACLNN TYPE UNSPPORTED, ONLY SUPPORT aclnn/aclnn_inner/aclnn_exclude")
      endif()
    endforeach()
  else()
    file(GLOB OPDEF_SRCS ${SOURCE_DIR}/*_def*.cpp)
    if(OPDEF_SRCS)
      message(FATAL_ERROR
      "Should Manually specify aclnn/aclnn_inner/aclnn_exclude\n"
      "usage: add_modules_sources(OPTYPE optypes ACLNNTYPE aclnntypes)\n"
      "example: add_modules_sources(OPTYPE add ACLNNTYPE aclnn_exclude)"
      )
    endif()
  endif()
endmacro()

# 添加opapi object
function(add_opapi_modules)
  if (NOT TARGET ${OPHOST_NAME}_opapi_obj)
    add_library(${OPHOST_NAME}_opapi_obj OBJECT)
    set(BUILD_UT OFF CACHE BOOL "No UT Compilation" FORCE)
    if(UT_TEST_ALL OR OP_API_UT)
      set(BUILD_UT ON CACHE BOOL "Build OpApi UT Compilation" FORCE)
    endif()
    target_include_directories(${OPHOST_NAME}_opapi_obj
      PRIVATE
      ${OPAPI_INCLUDE}
      $<BUILD_INTERFACE:${ASCEND_CANN_PACKAGE_PATH}/include>
      $<BUILD_INTERFACE:${ASCEND_CANN_PACKAGE_PATH}/include/aclnn>
      $<$<BOOL:${BUILD_OPEN_PROJECT}>:$<BUILD_INTERFACE:${ASCEND_CANN_PACKAGE_PATH}/include/experiment/metadef/common/util>>
      $<$<BOOL:${BUILD_OPEN_PROJECT}>:$<BUILD_INTERFACE:${ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_PREFIX}/include/op_common>>
      $<$<BOOL:${BUILD_OPEN_PROJECT}>:$<BUILD_INTERFACE:${ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_PREFIX}/include/op_common/op_host>>
      $<$<BOOL:${BUILD_OPEN_PROJECT}>:$<BUILD_INTERFACE:${ASCEND_CANN_PACKAGE_PATH}/include/experiment/hccl/external>>
      ${OPS_TRANSFORMER_DIR}/mc2/common/inc
      ${OPS_TRANSFORMER_DIR}/mc2/3rd
    )
    target_compile_definitions(${OPHOST_NAME}_opapi_obj PRIVATE
      _GLIBCXX_USE_CXX11_ABI=0
    )
    target_compile_options(${OPHOST_NAME}_opapi_obj
      PRIVATE
      -Dgoogle=ascend_private
      -DACLNN_LOG_FMT_CHECK
    )
    target_link_libraries(${OPHOST_NAME}_opapi_obj
      PUBLIC
      $<BUILD_INTERFACE:intf_pub>
      -Wl,--whole-archive
      ops_aclnn
      -Wl,--no-whole-archive
      $<$<BOOL:${dlog_FOUND}>:$<BUILD_INTERFACE:dlog_headers>>
      nnopbase
      profapi
      ge_common_base
      ascend_dump
      ascendalog
      dl
      )
  endif()
endfunction()

# 添加infer object
function(add_infer_modules)
  if (NOT TARGET ${OPHOST_NAME}_infer_obj)
    add_library(${OPHOST_NAME}_infer_obj OBJECT)
    set(BUILD_UT OFF CACHE BOOL "No UT Compilation" FORCE)
    if(UT_TEST_ALL OR PROTO_UT OR PASS_UT OR PLUGIN_UT OR ONNX_PLUGIN_UT)
      set(BUILD_UT ON CACHE BOOL "Build InferShape UT Compilation" FORCE)
    endif()
    target_include_directories(${OPHOST_NAME}_infer_obj
      PRIVATE ${OP_PROTO_INCLUDE}
      $<BUILD_INTERFACE:${ASCEND_CANN_PACKAGE_PATH}/include>
      $<BUILD_INTERFACE:${OPS_TRANSFORMER_DIR}/common/include>
      $<$<BOOL:${BUILD_OPEN_PROJECT}>:$<BUILD_INTERFACE:${ASCEND_CANN_PACKAGE_PATH}/include/experiment>>
      $<$<BOOL:${BUILD_OPEN_PROJECT}>:$<BUILD_INTERFACE:${ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_PREFIX}/include/op_common>>
      $<$<BOOL:${BUILD_OPEN_PROJECT}>:$<BUILD_INTERFACE:${ASCEND_CANN_PACKAGE_PATH}/include/experiment/hccl/external>>
      $<$<BOOL:${BUILD_OPEN_PROJECT}>:$<BUILD_INTERFACE:${ASCEND_CANN_PACKAGE_PATH}/include/experiment/metadef/common/util>>
      $<$<BOOL:${BUILD_OPEN_PROJECT}>:$<BUILD_INTERFACE:${ASCEND_CANN_PACKAGE_PATH}/include/external>>
      ${OPS_TRANSFORMER_DIR}/mc2/common/inc
      ${OPS_TRANSFORMER_DIR}/mc2/3rd
    )
    target_compile_definitions(${OPHOST_NAME}_infer_obj
      PRIVATE
      LOG_CPP
      OPS_UTILS_LOG_SUB_MOD_NAME="OP_PROTO"
      $<$<BOOL:${BUILD_UT}>:ASCEND_OPSPROTO_UT>
    )
    target_compile_options(${OPHOST_NAME}_infer_obj
      PRIVATE
      $<$<NOT:$<BOOL:${BUILD_UT}>>:-DDISABLE_COMPILE_V1>
      -Dgoogle=ascend_private
      -fvisibility=hidden
    )
    target_link_libraries(${OPHOST_NAME}_infer_obj
      PRIVATE
      $<BUILD_INTERFACE:intf_pub>
      $<BUILD_INTERFACE:ops_transformer_utils_proto_headers>
      $<$<BOOL:${alog_FOUND}>:$<BUILD_INTERFACE:alog_headers>>
      $<$<BOOL:${dlog_FOUND}>:$<BUILD_INTERFACE:dlog_headers>>
      -Wl,--whole-archive
      rt2_registry_static
      -Wl,--no-whole-archive
      -Wl,--no-as-needed
      exe_graph
      graph
      graph_base
      register
      ascendalog
      error_manager
      platform
      -Wl,--as-needed
      c_sec
    )
  endif()
endfunction()

# 添加tiling object
function(add_tiling_modules)
  if (NOT TARGET ${OPHOST_NAME}_tiling_obj)
    add_library(${OPHOST_NAME}_tiling_obj OBJECT)
    add_dependencies(${OPHOST_NAME}_tiling_obj json)
    set(BUILD_UT OFF CACHE BOOL "No UT Compilation" FORCE)
    if(UT_TEST_ALL OR TILING_UT)
      set(BUILD_UT ON CACHE BOOL "Build OpTiling UT Compilation" FORCE)
    endif()
    target_include_directories(${OPHOST_NAME}_tiling_obj
      PRIVATE ${OP_TILING_INCLUDE}
      $<BUILD_INTERFACE:${ASCEND_CANN_PACKAGE_PATH}/include>
      $<BUILD_INTERFACE:${OPS_TRANSFORMER_DIR}/common/include>
      
      $<$<BOOL:${BUILD_OPEN_PROJECT}>:$<BUILD_INTERFACE:${ASCEND_CANN_PACKAGE_PATH}/include/experiment>>
      $<$<BOOL:${BUILD_OPEN_PROJECT}>:$<BUILD_INTERFACE:${ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_PREFIX}/include/op_common/op_host>>
      $<$<BOOL:${BUILD_OPEN_PROJECT}>:$<BUILD_INTERFACE:${ASCEND_CANN_PACKAGE_PATH}/include/experiment/metadef/common/util>>
      $<$<BOOL:${BUILD_OPEN_PROJECT}>:$<BUILD_INTERFACE:${ASCEND_CANN_PACKAGE_PATH}/include/experiment>>
      ${OPS_TRANSFORMER_DIR}/mc2/common/inc
      ${OPS_TRANSFORMER_DIR}/mc2/3rd
    )
    target_compile_definitions(${OPHOST_NAME}_tiling_obj
      PRIVATE
      LOG_CPP
      OPS_UTILS_LOG_SUB_MOD_NAME="OP_TILING"
      $<$<BOOL:${BUILD_UT}>:ASCEND_OPTILING_UT>
    )
    target_compile_options(${OPHOST_NAME}_tiling_obj
      PRIVATE
      $<$<NOT:$<BOOL:${BUILD_UT}>>:-DDISABLE_COMPILE_V1>
      -Dgoogle=ascend_private
      -fvisibility=hidden
      -fno-strict-aliasing
    )
    target_link_libraries(${OPHOST_NAME}_tiling_obj
      PRIVATE
      # # $<BUILD_INTERFACE:$<IF:$<BOOL:${BUILD_UT}>, intf_llt_pub_asan_cxx17, intf_pub_cxx17>>
      $<BUILD_INTERFACE:intf_pub>
      $<BUILD_INTERFACE:ops_transformer_utils_tiling_headers>
      $<$<BOOL:${alog_FOUND}>:$<BUILD_INTERFACE:alog_headers>>
      $<$<BOOL:${dlog_FOUND}>:$<BUILD_INTERFACE:dlog_headers>>
      -Wl,--whole-archive
      rt2_registry_static
      -Wl,--no-whole-archive
      -Wl,--no-as-needed
      graph
      graph_base
      exe_graph
      platform
      register
      # ascendalog
      error_manager
      -Wl,--as-needed
      -Wl,--whole-archive
      tiling_api
      -Wl,--no-whole-archive
      # mmpa
      c_sec
    )
  endif()
endfunction()

function(add_graph_plugin_modules)
  if(NOT TARGET ${GRAPH_PLUGIN_NAME}_obj)
    add_library(${GRAPH_PLUGIN_NAME}_obj OBJECT)
    target_include_directories(${GRAPH_PLUGIN_NAME}_obj PRIVATE 
      ${OP_PROTO_INCLUDE}
    )
    target_compile_definitions(${GRAPH_PLUGIN_NAME}_obj PRIVATE OPS_UTILS_LOG_SUB_MOD_NAME="GRAPH_PLUGIN" LOG_CPP)
    target_compile_options(
      ${GRAPH_PLUGIN_NAME}_obj PRIVATE $<$<NOT:$<BOOL:${ENABLE_TEST}>>:-DDISABLE_COMPILE_V1> -Dgoogle=ascend_private
                                       -fvisibility=hidden
      )
    target_link_libraries(
      ${GRAPH_PLUGIN_NAME}_obj
      PRIVATE $<BUILD_INTERFACE:$<IF:$<BOOL:${ENABLE_TEST}>,intf_llt_pub_asan_cxx17,intf_pub_cxx17>>
              $<BUILD_INTERFACE:dlog_headers>
              $<$<TARGET_EXISTS:ops_base_util_objs>:$<TARGET_OBJECTS:ops_base_util_objs>>
              $<$<TARGET_EXISTS:ops_base_infer_objs>:$<TARGET_OBJECTS:ops_base_infer_objs>>
      )
  endif()
endfunction()

# 添加gentask object
function(add_opmaster_ct_gentask_modules)
  message(STATUS "add_opmaster_ct_gentask_modules start")
  if (NOT TARGET ${OPHOST_NAME}_opmaster_ct_gentask_obj)
    add_library(${OPHOST_NAME}_opmaster_ct_gentask_obj OBJECT)
    add_dependencies(${OPHOST_NAME}_opmaster_ct_gentask_obj json)
    set(BUILD_UT OFF CACHE BOOL "No UT Compilation" FORCE)

    target_include_directories(${OPHOST_NAME}_opmaster_ct_gentask_obj
      PRIVATE ${OP_TILING_INCLUDE}
      $<$<BOOL:${BUILD_OPEN_PROJECT}>:$<BUILD_INTERFACE:${ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_PREFIX}/include>>
      $<$<BOOL:${BUILD_OPEN_PROJECT}>:$<BUILD_INTERFACE:${ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_PREFIX}/include/experiment/metadef/common/util>>
    )
    target_compile_definitions(${OPHOST_NAME}_opmaster_ct_gentask_obj
      PRIVATE
      LOG_CPP
    )
    target_compile_options(${OPHOST_NAME}_opmaster_ct_gentask_obj
      PRIVATE
      $<$<NOT:$<BOOL:${BUILD_UT}>>:-DDISABLE_COMPILE_V1>
      -Dgoogle=ascend_private
      -fvisibility=hidden
      -fno-strict-aliasing
    )
    message(STATUS "xxxx compile add_opmaster_ct_gentask_modules")
    target_link_libraries(${OPHOST_NAME}_opmaster_ct_gentask_obj
      PRIVATE
      $<BUILD_INTERFACE:intf_pub_cxx17>
      $<$<BOOL:${alog_FOUND}>:$<BUILD_INTERFACE:alog_headers>>
      $<$<BOOL:${dlog_FOUND}>:$<BUILD_INTERFACE:dlog_headers>>
    )
  endif()
endfunction()


# useage: add_graph_plugin_sources()
macro(add_graph_plugin_sources)
  set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

  # 获取算子层级目录名称，判断是否编译该算子
  get_filename_component(PARENT_DIR ${SOURCE_DIR} DIRECTORY)
  get_filename_component(OP_NAME ${PARENT_DIR} NAME)
  if(DEFINED ASCEND_OP_NAME
     AND NOT "${ASCEND_OP_NAME}" STREQUAL ""
     AND NOT "${ASCEND_OP_NAME}" STREQUAL "all"
     AND NOT "${ASCEND_OP_NAME}" STREQUAL "ALL"
    )
    if(NOT ${OP_NAME} IN_LIST ASCEND_OP_NAME)
      return()
    endif()
  endif()

  file(GLOB GRAPH_PLUGIN_SRCS 
      ${SOURCE_DIR}/*_graph_plugin*.cpp
  )
  if(GRAPH_PLUGIN_SRCS)
    add_graph_plugin_modules()
    target_sources(${GRAPH_PLUGIN_NAME}_obj PRIVATE ${GRAPH_PLUGIN_SRCS})
  endif()

  file(GLOB GRAPH_PLUGIN_PROTO_HEADERS ${SOURCE_DIR}/*_proto*.h)
  if(GRAPH_PLUGIN_PROTO_HEADERS)
    target_sources(${GRAPH_PLUGIN_NAME}_proto_headers INTERFACE ${GRAPH_PLUGIN_PROTO_HEADERS})
  endif()
endmacro()