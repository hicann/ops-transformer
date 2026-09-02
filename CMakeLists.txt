# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.18.4)
include(cmake/fetch_cann_cmake.cmake)
if(DEFINED RULE_LAUNCH AND RULE_LAUNCH)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ${RULE_LAUNCH})
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ${RULE_LAUNCH})
endif()

project(cann_ops-transformer)

set(CMAKE_C_FLAGS_RELEASE "")
set(CMAKE_CXX_FLAGS_RELEASE "")

init_cann_project()
if(NOT ASCEND_INSTALL_PATH AND DEFINED ENV{ASCEND_HOME_PATH})
    set(ASCEND_INSTALL_PATH $ENV{ASCEND_HOME_PATH})
endif()

option(BUILD_OPEN_PROJECT         "Build open ascend ops project."  ON)
option(BUILD_OPS_RTY_KERNEL       "Build return yellow kernel."     OFF)
option(ENABLE_CCACHE              "Enable ccache capability"        ON)
option(ENABLE_BUILT_IN            "Enable built-in package"         OFF)
option(ENABLE_STATIC              "Enable Static"                   OFF)
option(ENABLE_TILING_SINK         "Enable tiling sink"              ON)
option(ENABLE_AICPU               "Enable aicpu"                    ON)
option(ENABLE_EXPERIMENTAL        "Enable experimental module"      OFF)
option(ENABLE_TEST                "Enable test"                     OFF)
option(ENABLE_UT_EXEC             "Enable exec ut"                  OFF)
option(ENABLE_ASAN                "Enable asan"                     OFF)
option(ENABLE_VALGRIND            "Enable valgrind"                 OFF)
option(OP_HOST_UT                 "Enable ophost ut"                OFF)
option(OP_API_UT                  "Enable opapi ut"                 OFF)
option(OP_GRAPH_UT                "Enable graph ut"                 OFF)
option(OP_KERNEL_UT               "Enable kernel ut"                OFF)
option(OP_KERNEL_AICPU_UT         "Enable aicpu kernel ut"          OFF)
option(UT_TEST_ALL                "Enable all ut"                   OFF)
option(ENABLE_OOM                 "Enable kernel oom"               OFF)
option(ENABLE_MSSANITIZER         "Enable mssanitizer mode"         OFF)
option(ENABLE_DUMP_CCE            "Enable dump_cce mode"            OFF)
option(UT_INFERSHAPE_FLAG         "Enable infershape ut"            OFF)

set(ASCEND_COMPUTE_UNIT           "ascend910b"           CACHE   STRING   "soc that need to be compiled")
set(ASCEND_OP_NAME                "ALL"                           CACHE   STRING   "operators that need to be compiled")
set(ASCEND_MODULE_NAME            "ALL"                           CACHE   STRING   "modules that need to be compiled")
set(ARCH_DIRECTORY                ""                           CACHE   STRING   "arch directory that need to be compiled")
set(VENDOR_NAME                   "custom"                     CACHE   STRING   "vendor name")
set(ASCEND_ALL_COMPUTE_UNIT       "ascend910b;ascend910_93;ascend950;ascend350;mc62;kirinx90;kirin9030" CACHE STRING "all soc list")
set(ASCEND_ALL_MODULE_NAME        "mc2;attention;moe;ffn;mhc;posembedding;gmm;mamba" CACHE STRING "all module list")

#SOC和ARCH目录对应关系:ascend310p:arch20,ascend910b:arch22,ascend910_93:arch22,ascend950:arch35,ascend350:arch35,mc62:arch38,kirinx90:arch22,kirin9030:arch22
set(SOC_VERSION_LIST ascend310p ascend910b ascend910_93 ascend950 ascend350 mc62 kirinx90 kirin9030)
set(ARCH_DIRECTORY_LIST arch20 arch22 arch22 arch35 arch35 arch38 arch22 arch22)
set(BISHENG_FLAGS "" CACHE STRING "bisheng compiler flags")
set(ENABLE_AICPU_KERNEL FALSE CACHE BOOL "Enable AICPU Kernel")
set(PACKAGE_TYPE "run" CACHE STRING "package type: run/rpm/deb/all")
set_property(CACHE PACKAGE_TYPE PROPERTY STRINGS run rpm deb all)
set(SUPPORTED_PACKAGE_TYPES run rpm deb all)
if(NOT PACKAGE_TYPE IN_LIST SUPPORTED_PACKAGE_TYPES)
  message(FATAL_ERROR "PACKAGE_TYPE only supports run/rpm/deb/all, got: ${PACKAGE_TYPE}")
endif()

# 获取基础 CANN 路径
if (CUSTOM_ASCEND_CANN_PACKAGE_PATH)
    set(ASCEND_CANN_PACKAGE_PATH  ${CUSTOM_ASCEND_CANN_PACKAGE_PATH})
elseif (DEFINED ENV{ASCEND_HOME_PATH})
    set(ASCEND_CANN_PACKAGE_PATH  $ENV{ASCEND_HOME_PATH})
elseif (DEFINED ENV{ASCEND_OPP_PATH})
    get_filename_component(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_OPP_PATH}/.." ABSOLUTE)
else()
    set(ASCEND_CANN_PACKAGE_PATH  "/usr/local/Ascend/latest")
endif ()
message(STATUS "ASCEND_CANN_PACKAGE_PATH=${ASCEND_CANN_PACKAGE_PATH}")

get_filename_component(OPS_ADV_DIR                  "${CMAKE_CURRENT_SOURCE_DIR}"           REALPATH)
get_filename_component(OPS_ADV_CMAKE_DIR            "${OPS_ADV_DIR}/cmake"                  REALPATH)
include(cmake/func.cmake)

function(should_add_module MODULE_NAME RESULT)
    if(DEFINED ASCEND_MODULE_NAME AND NOT "${ASCEND_MODULE_NAME}" STREQUAL "" AND NOT "${ASCEND_MODULE_NAME}" STREQUAL "all" AND NOT "${ASCEND_MODULE_NAME}" STREQUAL "ALL")
        string(REPLACE "," ";" MODULE_LIST "${ASCEND_MODULE_NAME}")
        if(${MODULE_NAME} IN_LIST MODULE_LIST)
            set(${RESULT} TRUE PARENT_SCOPE)
        else()
            set(${RESULT} FALSE PARENT_SCOPE)
        endif()
    else()
        set(${RESULT} TRUE PARENT_SCOPE)
    endif()
endfunction()

include(version.cmake)
if (BUILD_OPEN_PROJECT)
    check_cann_pkg_build_deps(${CANN_VERSION_PACKAGES})
endif()
add_cann_version_info_targets()

foreach(SOC_VERSION ${ASCEND_COMPUTE_UNIT})
  list(FIND SOC_VERSION_LIST ${SOC_VERSION} INDEX)
  if(NOT INDEX EQUAL -1)
    list(GET ARCH_DIRECTORY_LIST ${INDEX} VAL)
    list(APPEND ARCH_DIRECTORY ${VAL})
  else()
    message(STATUS "unsupported chip type")
    if ((NOT BUILD_OPS_RTY_KERNEL) AND (BUILD_OPEN_PROJECT))
      include(cmake/build_empty_package.cmake)
      cpack_empty_package()
      return()
    endif()
  endif()
endforeach()

if(PROJECT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
  message(STATUS "compile project with library")
  option(BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG "Build ops-transformer with cann pkg" ON)
else()
  message(STATUS "compile project with src")
  option(BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG "Build ops-transformer with cann source" OFF)
endif()

if(UNIX)
  set(SYSTEM_PREFIX ${CMAKE_SYSTEM_PROCESSOR}-linux)
endif()

#外部传参
if(NOT ${CMAKE_BUILD_MODE} STREQUAL "FALSE")
  if(ENABLE_DEBUG)
    set(CMAKE_BUILD_MODE "${CMAKE_BUILD_MODE} -g")
  endif()
  set(COMPILE_OP_MODE ${CMAKE_BUILD_MODE})
else()
  if(ENABLE_TEST)
    set(COMPILE_OP_MODE "-O0 -g")
  endif()
  if(ENABLE_DEBUG)
    set(CMAKE_BUILD_MODE "-g")
  endif()
endif()

set(PKG_NAME transformer)
set(OPS_TRANSFORMER_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(CMAKE_CXX_STANDARD 17 CACHE STRING "c++17 is needed for this project")
set_directory_properties(PROPERTIES
    ADDITIONAL_MAKE_CLEAN_FILES "${CMAKE_BINARY_DIR}/_CPack_Packages"
)

include(cmake/third_party/opbase.cmake)
if(DOWNLOAD_OPS_TEST_KIT)
  include(cmake/third_party/ops_test_kit.cmake)
endif()
include(cmake/config.cmake)
add_cann_third_party(json)

if (ENABLE_TEST)
    set(CMAKE_CXX_COMPILE_OBJECT "<CMAKE_CXX_COMPILER> <DEFINES> <INCLUDES> <FLAGS> -o <OBJECT> -c <SOURCE>")
    set(CMAKE_C_COMPILE_OBJECT "<CMAKE_C_COMPILER> <DEFINES> <INCLUDES> <FLAGS> -o <OBJECT> -c <SOURCE>")
    add_cann_third_party(gtest)
    if (TARGET GTest::gtest AND NOT TARGET gtest)
        add_library(gtest ALIAS GTest::gtest)
    endif()

endif()
include(${OPS_ADV_CMAKE_DIR}/ut.cmake)

if (BUILD_OPEN_PROJECT)
    include(cmake/intf.cmake)
    add_definitions(-DBUILD_OPEN_PROJECT)
    if (BUILD_OPS_RTY_KERNEL)
        message(STATUS "Build return yellow kernel.")
        include(cmake/rty_obj_func.cmake)
    else()
        message(STATUS "Start building custom package.")
        include(ExternalProject)
        include(cmake/dependencies.cmake)
        include(cmake/third_party/ops-tensor.cmake)
        include(cmake/third_party/catlass.cmake)
        include(cmake/variables.cmake)
        include(cmake/obj_func.cmake)
        add_cann_third_party(eigen)
        add_cann_third_party(abseil-cpp)
        add_cann_third_party(protobuf)


        include(cmake/custom_build.cmake)
        message(STATUS "End building custom package.")
        include(cmake/opbuild.cmake)
        if (ENABLE_OPS_HOST)
            gen_aclnn_with_opdef()
        endif()
        if (ENABLE_STATIC)
            include(cmake/static.cmake)
        endif()
        if (ENABLE_TILING_SINK)
            pack_tiling_sink()
        endif()
        if (ENABLE_BUILT_IN)
            message(STATUS "Start building built-in package.")
            include(cmake/symbol.cmake)
            gen_norm_symbol()
            include(cmake/package.cmake)
            pack_built_in()
        else()
            message(STATUS "Start building custom package.")
            include(cmake/symbol.cmake)
            if (ENABLE_AICPU)
                gen_cust_aicpu_symbol()
            endif()
        endif()
        return()
    endif()
else()
    include(cmake/dependencies.cmake)
    include(cmake/third_party/ops-tensor.cmake)
    include(cmake/third_party/catlass.cmake)
    include(cmake/variables.cmake)
    include(cmake/opbuild.cmake)
    include(cmake/rty_obj_func.cmake)
    include(cmake/intf_pub_linux.cmake)
endif()

if (BUILD_OPS_RTY_KERNEL)
    set(CMAKE_MODULE_PATH
        ${CMAKE_MODULE_PATH}
        ${CMAKE_CURRENT_LIST_DIR}/cmake/modules
    )

    set(CMAKE_PREFIX_PATH
        ${CMAKE_PREFIX_PATH}
        ${ASCEND_CANN_PACKAGE_PATH}
    )

    set(_op_host_aclnn_link
            $<BUILD_INTERFACE:intf_pub>
            exe_graph
            register
            c_sec
    )

    find_package(alog MODULE)

    if(NOT ${alog_FOUND})
        add_definitions(-DALOG_NOT_FOUND)
    endif()

    add_library(op_host_aclnn SHARED EXCLUDE_FROM_ALL)
    target_link_libraries(op_host_aclnn PRIVATE
            ${_op_host_aclnn_link}
    )
    target_compile_options(op_host_aclnn PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-std=gnu++1z>
    )

    add_library(op_host_aclnnInner SHARED EXCLUDE_FROM_ALL)
    target_link_libraries(op_host_aclnnInner PRIVATE
            ${_op_host_aclnn_link}
    )
    target_compile_options(op_host_aclnnInner PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-std=gnu++1z>
    )

    add_library(op_host_aclnnExc SHARED EXCLUDE_FROM_ALL)
    target_link_libraries(op_host_aclnnExc PRIVATE
            ${_op_host_aclnn_link}
    )
    target_compile_options(op_host_aclnnExc PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-std=gnu++1z>
    )

    # op proto
    add_library(opsproto SHARED)
    target_compile_options(opsproto PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-std=c++11>
            -fvisibility=hidden
    )
    target_compile_definitions(opsproto PRIVATE
            LOG_CPP
            PROCESS_LOG
    )
    target_link_libraries(opsproto PRIVATE
            $<BUILD_INTERFACE:intf_pub>
            $<BUILD_INTERFACE:ops_transformer_utils_proto_headers>
            $<$<BOOL:${alog_FOUND}>:$<BUILD_INTERFACE:alog_headers>>
            -Wl,--whole-archive
            rt2_registry
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
    set_target_properties(opsproto PROPERTIES OUTPUT_NAME
            cust_opsproto_rt2.0
    )
    install(TARGETS opsproto
            LIBRARY DESTINATION packages/vendors/${VENDOR_NAME}_transformer/op_proto/lib/linux/${CMAKE_SYSTEM_PROCESSOR}
    )

    add_ops_tiling_keys(
            OP_NAME "ALL"
            TILING_KEYS ${TILING_KEY}
    )

    add_opc_config(
            OP_NAME "ALL"
            CONFIG ${OP_DEBUG_CONFIG}
    )

    if(ADD_OPS_COMPILE_OPTION_V2)
        add_ops_compile_options(
                OP_NAME "ALL"
                OPTIONS ${OPS_COMPILE_OPTIONS}
        )
    endif()
endif ()

add_subdirectory(common)

if (NOT BUILD_OPS_RTY_KERNEL)
    should_add_module("mc2" ADD_MC2)
    if(ADD_MC2)
        add_subdirectory(mc2)
    endif()
    should_add_module("posembedding" ADD_POSEMBEDDING)
    if(ADD_POSEMBEDDING)
        add_subdirectory(posembedding)
    endif()
endif()

if(NOT DEFINED COMPILED_OPS)
    set(COMPILED_OPS ${COMPILED_OPS} CACHE STRING "Comp")
    set(COMPILED_OPS CACHE STRING "Compiled Ops" FORCE)
set(COMPILED_OP_DIRS CACHE STRING "Compiled Ops Dirs" FORCE)
endif()

if(NOT DEFINED COMPILED_OP_DIRS)
    set(COMPILED_OP_DIRS CACHE STRING "Compiled Ops Dirs" FORCE)
endif()

set(OP_LIST)
set(OP_DIR_LIST)
op_add_subdirectory(OP_LIST OP_DIR_LIST)

set_property(GLOBAL PROPERTY _OP_HOST_COMPILE_PHASE "DIRECT")
foreach (OP_DIR ${OP_DIR_LIST})
    if (EXISTS "${OP_DIR}/op_host")
        add_subdirectory(${OP_DIR}/op_host)
    else()
        add_subdirectory(${OP_DIR})
    endif()
endforeach ()

should_add_module("moe" ADD_MOE)
if(ADD_MOE)
    add_subdirectory(moe)
    list(APPEND OP_LIST "moe_init_routing_v2")
    list(APPEND OP_LIST "moe_token_unpermute_with_ep_grad")
    list(APPEND OP_DIR_LIST ${CMAKE_CURRENT_SOURCE_DIR}/moe/moe_init_routing_v2)
    list(APPEND OP_DIR_LIST ${CMAKE_CURRENT_SOURCE_DIR}/moe/moe_token_unpermute_with_ep_grad)
endif()
should_add_module("ffn" ADD_FFN)
if(ADD_FFN)
    add_subdirectory(ffn)
    list(APPEND OP_LIST "ffn")
    list(APPEND OP_DIR_LIST ${CMAKE_CURRENT_SOURCE_DIR}/ffn/ffn)
endif()
should_add_module("attention" ADD_ATTENTION)
if(ADD_ATTENTION)
    add_subdirectory(attention)
endif()
should_add_module("mhc" ADD_MHC)
if(ADD_MHC)
    add_subdirectory(mhc)
endif()
should_add_module("mamba" ADD_MAMBA)
if(ADD_MAMBA)
    add_subdirectory(mamba)
endif()
list(APPEND OP_LIST ${COMPILED_OPS})
list(REMOVE_DUPLICATES OP_LIST)
list(APPEND OP_DIR_LIST ${COMPILED_OP_DIRS})
list(REMOVE_DUPLICATES OP_DIR_LIST)

set(OP_DEPEND_DIR_LIST)
op_add_depend_directory(
        OP_LIST ${OP_LIST}
        OP_DIR_LIST OP_DEPEND_DIR_LIST
)

set_property(GLOBAL PROPERTY _OP_HOST_COMPILE_PHASE "DEPENDENCY")
foreach (OP_DEPEND_DIR ${OP_DEPEND_DIR_LIST})
    if (EXISTS "${OP_DEPEND_DIR}/op_host")
        add_subdirectory(${OP_DEPEND_DIR}/op_host)
    else()
        add_subdirectory(${OP_DEPEND_DIR})
    endif()
endforeach ()
set_property(GLOBAL PROPERTY _OP_HOST_COMPILE_PHASE "DIRECT")

if (BUILD_OPS_RTY_KERNEL)
    get_target_property(base_aclnn_srcs op_host_aclnn SOURCES)
    get_target_property(base_aclnn_inner_srcs op_host_aclnnInner SOURCES)
    get_target_property(base_aclnn_exclude_srcs op_host_aclnnExc SOURCES)
    set(base_aclnn_binary_dir ${ASCEND_AUTOGEN_DIR})

    set(generate_aclnn_srcs)
    set(generate_aclnn_inner_srcs)
    set(generate_aclnn_headers)
    set(generate_proto_dir ${base_aclnn_binary_dir})
    set(generate_exclude_proto_srcs)
    set(generate_proto_srcs)
    set(generate_proto_headers)

    if (base_aclnn_srcs)
        foreach (_src ${base_aclnn_srcs})
            string(REGEX MATCH "^${CMAKE_CURRENT_SOURCE_DIR}" is_match "${_src}")
            if (is_match)
                get_filename_component(name_without_ext ${_src} NAME_WE)

                string(REGEX REPLACE "_def$" "" _op_name ${name_without_ext})
                list(APPEND generate_aclnn_srcs ${base_aclnn_binary_dir}/aclnn_${_op_name}.cpp)
                list(APPEND generate_aclnn_headers ${base_aclnn_binary_dir}/aclnn_${_op_name}.h)
                list(APPEND generate_proto_srcs    ${generate_proto_dir}/${_op_name}_proto.cpp)
                list(APPEND generate_proto_headers ${generate_proto_dir}/${_op_name}_proto.h)
            endif ()
        endforeach ()
    else ()
        add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/op_host_aclnn_stub.cpp
                COMMAND touch ${CMAKE_CURRENT_BINARY_DIR}/op_host_aclnn_stub.cpp
        )

        target_sources(op_host_aclnn PRIVATE
                ${CMAKE_CURRENT_BINARY_DIR}/op_host_aclnn_stub.cpp
        )
    endif ()

    if (base_aclnn_inner_srcs)
        foreach (_src ${base_aclnn_inner_srcs})
            string(REGEX MATCH "^${CMAKE_CURRENT_SOURCE_DIR}" is_match "${_src}")
            if (is_match)
                get_filename_component(name_without_ext ${_src} NAME_WE)
                string(REGEX REPLACE "_def$" "" _op_name ${name_without_ext})
                list(APPEND generate_aclnn_inner_srcs ${base_aclnn_binary_dir}/inner/aclnnInner_${_op_name}.cpp)
                list(APPEND generate_proto_srcs    ${generate_proto_dir}/inner/${_op_name}_proto.cpp)
                list(APPEND generate_proto_headers ${generate_proto_dir}/inner/${_op_name}_proto.h)
            endif ()
        endforeach ()
    else ()
        add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/op_host_aclnn_inner_stub.cpp
                COMMAND touch ${CMAKE_CURRENT_BINARY_DIR}/op_host_aclnn_inner_stub.cpp
        )

        target_sources(op_host_aclnnInner PRIVATE
                ${CMAKE_CURRENT_BINARY_DIR}/op_host_aclnn_inner_stub.cpp
        )
    endif ()

    if (base_aclnn_exclude_srcs)
        foreach (_src ${base_aclnn_exclude_srcs})
            string(REGEX MATCH "^${CMAKE_CURRENT_SOURCE_DIR}" is_match "${_src}")
            if (is_match)
                get_filename_component(name_without_ext ${_src} NAME_WE)
                string(REGEX REPLACE "_def$" "" _op_name ${name_without_ext})
                list(APPEND generate_exclude_proto_srcs    ${generate_proto_dir}/exc/${_op_name}_proto.cpp)
                list(APPEND generate_proto_srcs            ${generate_proto_dir}/exc/${_op_name}_proto.cpp)
                list(APPEND generate_proto_headers         ${generate_proto_dir}/exc/${_op_name}_proto.h)
            endif ()
        endforeach ()
    else()
        add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/op_host_aclnn_exc_stub.cpp
                COMMAND touch ${CMAKE_CURRENT_BINARY_DIR}/op_host_aclnn_exc_stub.cpp
        )

        target_sources(op_host_aclnnExc PRIVATE
                ${CMAKE_CURRENT_BINARY_DIR}/op_host_aclnn_exc_stub.cpp
        )
    endif ()

    # 零算子（所有请求算子均被 CANN 版本检查跳过）：打印被跳过算子，写标志文件供 build.sh 跳过编译
    if (NOT generate_aclnn_srcs AND NOT generate_aclnn_inner_srcs AND NOT generate_exclude_proto_srcs)
        get_property(_cann_skipped_ops GLOBAL PROPERTY CANN_VERSION_SKIPPED_OPS)
        if (_cann_skipped_ops)
            list(REMOVE_DUPLICATES _cann_skipped_ops)
            get_cann_package_version(_cann_cur_ver)
            message(STATUS "当前 CANN 版本为 ${_cann_cur_ver}（soc=${ASCEND_COMPUTE_UNIT}），以下算子要求的最低版本不满足，不符合编译要求，已跳过编译：")
            foreach(_cann_skipped_op IN LISTS _cann_skipped_ops)
                message(STATUS "  - ${_cann_skipped_op}")
            endforeach()
        endif()
        file(WRITE "${CMAKE_BINARY_DIR}/no_ops.flag" "all requested ops skipped by CANN version check\n")
    else()
        file(REMOVE "${CMAKE_BINARY_DIR}/no_ops.flag")
    endif()


    if (generate_aclnn_srcs OR generate_aclnn_inner_srcs)
        set(ops_aclnn_src ${generate_aclnn_srcs} ${generate_aclnn_inner_srcs})
    else ()
        set(ops_aclnn_src ${CMAKE_CURRENT_BINARY_DIR}/ops_aclnn_src_stub.cpp)

        add_custom_command(OUTPUT ${ops_aclnn_src}
                COMMAND touch ${ops_aclnn_src}
        )
    endif ()

    set_source_files_properties(${ops_aclnn_src}
            PROPERTIES GENERATED TRUE
    )
    add_library(ops_aclnn STATIC
            ${ops_aclnn_src}
    )
    target_compile_options(ops_aclnn PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-std=gnu++1z>
    )
    target_link_libraries(ops_aclnn PRIVATE
            $<BUILD_INTERFACE:intf_pub>
    )
    add_dependencies(ops_aclnn opbuild_gen_default opbuild_gen_inner)
    add_dependencies(ops-transformer version_ops-transformer_info)

    set_source_files_properties(${generate_proto_srcs}
            PROPERTIES GENERATED TRUE
    )
    target_sources(opsproto PRIVATE
            ${generate_proto_srcs}
    )
    add_dependencies(opsproto ops_transformer_proto_headers)

    install(FILES ${generate_proto_headers}
            DESTINATION packages/vendors/${VENDOR_NAME}_transformer/op_proto/inc OPTIONAL
    )

    add_library(ops_transformer_proto_headers INTERFACE)

    target_include_directories(ops_transformer_proto_headers INTERFACE
            $<BUILD_INTERFACE:${generate_proto_dir}>
            $<BUILD_INTERFACE:${generate_proto_dir}/inner>
            $<BUILD_INTERFACE:${generate_proto_dir}/exc>
            $<INSTALL_INTERFACE:include/ops_adv/proto>
    )

    add_dependencies(ops_transformer_proto_headers opbuild_gen_default opbuild_gen_inner opbuild_gen_exc)

    if (NOT BUILD_OPEN_PROJECT)
        if (generate_proto_srcs)
            install_package(
                    PACKAGE ops_adv
                    TARGETS ops_proto_headers
                    FILES ${generate_proto_headers}
                    DESTINATION include/ops_adv/proto
            )
        endif ()
    endif ()

    if (generate_aclnn_srcs)
        add_custom_command(OUTPUT ${generate_aclnn_srcs} ${generate_aclnn_headers}
                COMMAND mkdir -p ${base_aclnn_binary_dir}
                COMMAND OPS_PROTO_SEPARATE=1
                OPS_ACLNN_GEN=1
                OPS_PROJECT_NAME=aclnn
                ${OP_BUILD_TOOL}
                $<TARGET_FILE:op_host_aclnn>
                ${base_aclnn_binary_dir}
        )
    endif ()

    add_custom_target(opbuild_gen_default
            DEPENDS ${generate_aclnn_srcs} ${generate_aclnn_headers} op_host_aclnn
    )

    if (generate_aclnn_inner_srcs)
        add_custom_command(OUTPUT ${generate_aclnn_inner_srcs}
                COMMAND mkdir -p ${base_aclnn_binary_dir}/inner
                COMMAND OPS_PROTO_SEPARATE=1
                OPS_ACLNN_GEN=1
                OPS_PROJECT_NAME=aclnnInner
                ${OP_BUILD_TOOL}
                $<TARGET_FILE:op_host_aclnnInner>
                ${base_aclnn_binary_dir}/inner
        )
    endif ()

    add_custom_target(opbuild_gen_inner
            DEPENDS ${generate_aclnn_inner_srcs} op_host_aclnnInner
    )

    if (generate_exclude_proto_srcs)
        add_custom_command(OUTPUT ${generate_exclude_proto_srcs}
                COMMAND mkdir -p ${base_aclnn_binary_dir}/exc
                COMMAND OPS_PROTO_SEPARATE=1
                OPS_ACLNN_GEN=0
                OPS_PROJECT_NAME=aclnnExc
                ${OP_BUILD_TOOL}
                $<TARGET_FILE:op_host_aclnnExc>
                ${base_aclnn_binary_dir}/exc
        )
    endif ()

    add_custom_target(opbuild_gen_exc
            DEPENDS ${generate_exclude_proto_srcs} op_host_aclnnExc
    )

    add_custom_target(generate_transformer_adapt_py
            COMMAND ${HI_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/scripts/util/ascendc_impl_build.py
            \"\"
            \"\"
            \"\"
            \"\"
            ${ASCEND_IMPL_OUT_DIR}
            ${ASCEND_AUTOGEN_DIR}
            --opsinfo-dir ${base_aclnn_binary_dir} ${base_aclnn_binary_dir}/inner ${base_aclnn_binary_dir}/exc
            --compute-unit ${ASCEND_COMPUTE_UNIT}
            --enable-experimental ${ENABLE_EXPERIMENTAL}
    )

    add_dependencies(generate_transformer_adapt_py opbuild_gen_default opbuild_gen_inner opbuild_gen_exc)

    foreach (_op_name ${OP_LIST})
        get_property(_op_version_skipped GLOBAL PROPERTY ${_op_name}_CANN_VERSION_SKIPPED)
        if(_op_version_skipped)
            message(STATUS "Skip install adapt py for ${_op_name}: CANN version check failed")
            continue()
        endif()
        install(FILES ${ASCEND_IMPL_OUT_DIR}/dynamic/${_op_name}.py
                DESTINATION ${IMPL_DYNAMIC_INSTALL_DIR}
                OPTIONAL
        )

        foreach (_suffix a2 a3 apt)
            if(EXISTS ${ASCEND_IMPL_OUT_DIR}/dynamic/${_op_name}_${_suffix}.py)
                install(FILES ${ASCEND_IMPL_OUT_DIR}/dynamic/${_op_name}_${_suffix}.py
                    DESTINATION ${IMPL_DYNAMIC_INSTALL_DIR}
                    OPTIONAL
                )
            endif()
        endforeach()
    endforeach ()

    install(DIRECTORY ${OPS_ADV_UTILS_KERNEL_INC}/
            DESTINATION ${IMPL_INSTALL_DIR}/ascendc/common
    )

    foreach (op_dir ${OP_DIR_LIST})
        get_filename_component(_op_name "${op_dir}" NAME)

        get_property(_op_version_skipped GLOBAL PROPERTY ${_op_name}_CANN_VERSION_SKIPPED)
        if(_op_version_skipped)
            message(STATUS "Skip install kernel files for ${_op_name}: CANN version check failed")
            continue()
        endif()

        if (EXISTS "${op_dir}/op_kernel")
            file(GLOB KERNEL_FILES
                ${op_dir}/op_kernel/*.cpp
                ${op_dir}/op_kernel/*.h
        )
        else()
            file(GLOB KERNEL_FILES
                ${op_dir}/*.cpp
                ${op_dir}/*.h
        )
        endif()

        install(FILES ${KERNEL_FILES}
                DESTINATION ${IMPL_INSTALL_DIR}/ascendc/${_op_name}
                OPTIONAL
        )

        install(DIRECTORY ${op_dir}/arch22
            DESTINATION ${IMPL_INSTALL_DIR}/ascendc/${_op_name}
            OPTIONAL
        )

        install(DIRECTORY ${op_dir}/arch35
                DESTINATION ${IMPL_INSTALL_DIR}/ascendc/${_op_name}
                OPTIONAL
        )

        install(DIRECTORY ${op_dir}/arch38
                DESTINATION ${IMPL_INSTALL_DIR}/ascendc/${_op_name}
                OPTIONAL
        )

        install(DIRECTORY ${op_dir}/regbase/opkernel
                DESTINATION ${IMPL_INSTALL_DIR}/ascendc/${_op_name}/regbase
                OPTIONAL
        )
    endforeach ()

    add_custom_target(prepare_build ALL)
    add_custom_target(generate_compile_cmd ALL)
    add_custom_target(generate_ops_info ALL)
    add_dependencies(prepare_build generate_transformer_adapt_py generate_compile_cmd)

    foreach (compute_unit ${ASCEND_COMPUTE_UNIT})
        add_compile_cmd_target(
                COMPUTE_UNIT ${compute_unit}
        )

        add_ops_info_target(
                COMPUTE_UNIT ${compute_unit}
        )
    endforeach ()

    add_custom_target(ops_transformer_kernel ALL)
    add_custom_target(ops_transformer_config ALL)
    add_dependencies(ops_transformer_kernel ops_transformer_config)

    foreach (compute_unit ${ASCEND_COMPUTE_UNIT})
        add_bin_compile_target(
                COMPUTE_UNIT
                ${compute_unit}
                OP_INFO
                ${OP_DIR_LIST}
        )
    endforeach ()
endif ()
