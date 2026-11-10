# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ######################################################################################################################
# 调用opbuild工具，生成aclnn/aclnnInner/.ini的算子信息库 等文件
# generate outpath: ${ASCEND_AUTOGEN_PATH}/${sub_dir}
# ######################################################################################################################
function(gen_opbuild_target)
  set(oneValueArgs TARGET PREFIX GENACLNN OUT_DIR OUT_SUB_DIR)
  set(multiValueArgs IN_SRCS OUT_SRCS OUT_HEADERS)
  cmake_parse_arguments(OPBUILD "" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  if(NOT OPBUILD_IN_SRCS)
    message(STATUS "No ${OPBUILD_PREFIX} srcs, skip ${OPBUILD_TARGET}")
    return()
  endif()

  add_library(gen_op_host_${OPBUILD_PREFIX} SHARED ${OPBUILD_IN_SRCS})
  target_link_libraries(gen_op_host_${OPBUILD_PREFIX} PRIVATE
                        $<BUILD_INTERFACE:intf_pub_cxx17>
                        exe_graph
                        register
                        c_sec
  )
  target_compile_options(gen_op_host_${OPBUILD_PREFIX} PRIVATE
    -fno-common
  )

  add_custom_command(OUTPUT ${OPBUILD_OUT_SRCS} ${OPBUILD_OUT_HEADERS}
                     COMMAND OPS_PROTO_SEPARATE=1
                             OPS_PROJECT_NAME=${OPBUILD_PREFIX}
                             OPS_ACLNN_GEN=${OPBUILD_GENACLNN}
                             OPS_PRODUCT_NAME=\"${ASCEND_COMPUTE_UNIT}\"
                             ${OP_BUILD_TOOL}
                             $<TARGET_FILE:gen_op_host_${OPBUILD_PREFIX}>
                             ${OPBUILD_OUT_DIR}/${OPBUILD_OUT_SUB_DIR}
  )

  add_custom_target(${OPBUILD_TARGET}
                    DEPENDS ${OPBUILD_OUT_SRCS} ${OPBUILD_OUT_HEADERS}
  )
  add_dependencies(${OPBUILD_TARGET} gen_op_host_${OPBUILD_PREFIX})
  if(TARGET op_build)
    add_dependencies(${OPBUILD_TARGET} op_build)
  endif()
endfunction()

function(gen_aclnn_classify host_obj prefix ori_out_srcs ori_out_headers opbuild_out_srcs opbuild_out_headers)
  get_target_property(module_sources ${host_obj} INTERFACE_SOURCES)
  set(sub_dir)
  # aclnn\aclnnExc以aclnn开头，aclnnInner以aclnnInner开头
  if("${prefix}" STREQUAL "aclnn")
    set(file_prefix "aclnn")
    set(need_gen_aclnn 1)
  elseif("${prefix}" STREQUAL "aclnnInner")
    set(sub_dir inner)
    set(file_prefix "aclnnInner")
    set(need_gen_aclnn 1)
  elseif("${prefix}" STREQUAL "aclnnExc")
    set(sub_dir exc)
    set(file_prefix "aclnn")
    set(need_gen_aclnn 0)
  else()
    message(FATAL_ERROR "UnSupported aclnn prefix type, must be in aclnn/aclnnInner/aclnnExc")
  endif()

  set(out_src_path ${ASCEND_AUTOGEN_PATH}/${sub_dir})
  file(MAKE_DIRECTORY ${out_src_path})
  get_filename_component(out_src_path ${out_src_path} REALPATH)
  set(in_srcs)
  set(out_srcs)
  set(out_headers)
  if(module_sources)
    foreach(file ${module_sources})
      get_filename_component(name_without_ext ${file} NAME_WE)
      string(REGEX REPLACE "_def$" "" _op_name ${name_without_ext})
      list(APPEND in_srcs ${file})
      list(APPEND out_srcs ${out_src_path}/${file_prefix}_${_op_name}.cpp)
      list(APPEND out_headers ${out_src_path}/${file_prefix}_${_op_name}.h)
    endforeach()
  endif()
  # opbuild_gen_aclnn/opbuild_gen_aclnnInner/opbuild_gen_aclnnExc
  if("${prefix}" STREQUAL "aclnnExc")
    get_target_property(exclude_headers ${OPHOST_NAME}_aclnn_exclude_headers INTERFACE_SOURCES)
    if(exclude_headers)
      set(${opbuild_out_headers} ${ori_out_headers} ${exclude_headers} PARENT_SCOPE)
    endif()
  else()
    set(${opbuild_out_srcs} ${ori_out_srcs} ${out_srcs} PARENT_SCOPE)
    if ("${prefix}" STREQUAL "aclnn")
      set(${opbuild_out_headers} ${ori_out_headers} ${out_headers} PARENT_SCOPE)
    endif()
  endif()
endfunction()

function(gen_aclnn_master_header aclnn_master_header_name aclnn_master_header opbuild_out_headers)
  # 规范化，防止生成的代码编译失败
  string(REGEX REPLACE "[^a-zA-Z0-9_]" "_" aclnn_master_header_name "${aclnn_master_header_name}")
  string(TOUPPER ${aclnn_master_header_name} aclnn_master_header_name)

  # 生成include内容
  set(aclnn_all_header_include_content "")
  foreach(header_file ${opbuild_out_headers})
    get_filename_component(header_name ${header_file} NAME)
    set(aclnn_all_header_include_content "${aclnn_all_header_include_content}#include \"${header_name}\"\n")
  endforeach()

  # 根据模板生成头文件
  message(STATUS "create aclnn master header file: ${aclnn_master_header}")
  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/aclnn_ops_transformer.h.in"
    "${aclnn_master_header}"
    @ONLY
  )
endfunction()

# The generated runtime uses cache defaults. Scope those defaults and restore the caller's cache.
function(_cannbotdsl_add_runtime source)
  set(_temporary_cache)
  set(CANN_ROOT "${ASCEND_CANN_PACKAGE_PATH}")
  set(OPBASE_INCLUDE_DIR "${NNOPBASE_OPDEV_INC_DIR}")
  set(OPBASE_LIBRARY nnopbase)
  foreach(_variable CANN_ROOT OPBASE_INCLUDE_DIR OPBASE_LIBRARY)
    if(DEFINED CACHE{${_variable}})
      get_property(_${_variable}_type CACHE "${_variable}" PROPERTY TYPE)
      set_property(CACHE "${_variable}" PROPERTY TYPE STRING)
    else()
      set(${_variable} "${${_variable}}" CACHE STRING "Scoped OpKit dependency")
      list(APPEND _temporary_cache "${_variable}")
    endif()
  endforeach()
  add_subdirectory("${source}" "${CMAKE_BINARY_DIR}/cannbot_dsl/runtime")
  foreach(_variable CANN_ROOT OPBASE_INCLUDE_DIR OPBASE_LIBRARY)
    if(_variable IN_LIST _temporary_cache)
      unset(${_variable} CACHE)
    else()
      set_property(CACHE "${_variable}" PROPERTY TYPE "${_${_variable}_type}")
    endif()
  endforeach()
endfunction()

# Attach ACLNN sources declared by the Host build rules.
function(cannbotdsl_add_modules out_headers)
  get_filename_component(_dsl_root "${ASCEND_BINARY_OUT_DIR}/../cannbot_dsl" ABSOLUTE)
  set(_headers)
  set(_map "{ local:\n")
  set(_obj ${OPHOST_NAME}_cannbotdsl_opapi_obj)
  add_library(${_obj} OBJECT)
  set_target_properties(${_obj} PROPERTIES POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN YES)
  target_compile_options(${_obj} PRIVATE -fno-fast-math -ffp-contract=off)
  get_property(_packages GLOBAL PROPERTY CANNBOTDSL_ACLNN_PACKAGES)
  foreach(_stage IN LISTS _packages)
    get_property(_layout GLOBAL PROPERTY "CANNBOTDSL_LAYOUT_${_stage}")
    file(READ "${_layout}" _package)
    string(JSON _aclnn_type GET "${_package}" aclnn_type)
    string(JSON _runtime GET "${_package}" runtime)
    get_property(_job GLOBAL PROPERTY "CANNBOTDSL_BUILD_${_stage}")
    add_dependencies(${_obj} ${_job})
    if(TARGET ${OPHOST_NAME}_opapi_obj)
      add_dependencies(${OPHOST_NAME}_opapi_obj ${_job})
    endif()
    if(NOT TARGET opkit_runtime)
      _cannbotdsl_add_runtime("${_runtime}")
    endif()
    target_link_libraries(${_obj} PUBLIC opkit::runtime register)
    if(TARGET ${OPHOST_NAME}_opapi_obj)
      target_link_libraries(${OPHOST_NAME}_opapi_obj PRIVATE opkit::runtime)
    endif()
    string(JSON _last LENGTH "${_package}" operators)
    math(EXPR _last "${_last} - 1")
    foreach(_index RANGE ${_last})
      string(JSON _operator GET "${_package}" operators ${_index})
      string(JSON _module GET "${_package}" modules "${_operator}")
      get_filename_component(_stem "${_module}" NAME)
      set(_header "aclnn_${_stem}.h")
      set(_module_dir "${_stage}/${_module}")
      set(_sources)
      string(JSON _source_count LENGTH "${_package}" sources "${_operator}")
      math(EXPR _last_source "${_source_count} - 1")
      foreach(_source_index RANGE ${_last_source})
        string(JSON _source GET "${_package}" sources "${_operator}" ${_source_index})
        list(APPEND _sources "${_stage}/${_source}")
      endforeach()
      set_source_files_properties(${_sources} PROPERTIES GENERATED TRUE)
      if(_aclnn_type STREQUAL "aclnn")
        target_link_options(${_obj} INTERFACE
          "LINKER:--require-defined=aclnn${_operator}"
          "LINKER:--require-defined=aclnn${_operator}GetWorkspaceSize")
      endif()
      target_sources(${_obj} PRIVATE ${_sources})
      target_include_directories(${_obj} PRIVATE "${_module_dir}" "${_module_dir}/op_host" "${_module_dir}/op_api")
      if(TARGET ${OPHOST_NAME}_opapi_obj)
        target_include_directories(${OPHOST_NAME}_opapi_obj PRIVATE "${_module_dir}/op_api")
      endif()
      string(APPEND _map "  Prepare${_operator};\n  ${_operator}L0Contract;\n  OpKitPrepareDirectInvoke${_operator};\n")
      if(_aclnn_type STREQUAL "aclnn")
        set(_public_contract "${_dsl_root}/${_operator}/public_api.h")
        file(CONFIGURE OUTPUT "${_public_contract}" CONTENT "#pragma GCC visibility push(default)\n#include \"${_module_dir}/op_api/${_header}\"\n#pragma GCC visibility pop\n")
        set_property(SOURCE ${_sources} APPEND PROPERTY COMPILE_OPTIONS "-include" "${_public_contract}")
        if(_header IN_LIST _headers)
          message(FATAL_ERROR "Duplicate DSL public header: ${_header}")
        endif()
        list(APPEND _headers "${_header}")
        list(APPEND ${out_headers} "${_module_dir}/op_api/${_header}")
      endif()
    endforeach()
  endforeach()
  string(APPEND _map "};\n")
  set(_map_file "${CMAKE_CURRENT_BINARY_DIR}/cannbot_dsl/internal.map")
  file(CONFIGURE OUTPUT "${_map_file}" CONTENT "${_map}" @ONLY)
  target_link_options(${_obj} INTERFACE "LINKER:-z,defs"
    "LINKER:--exclude-libs,libopkit_runtime.a" "LINKER:--version-script=${_map_file}")
  set_property(TARGET ${_obj} PROPERTY INTERFACE_LINK_DEPENDS "${_map_file}")
  set(${out_headers} "${${out_headers}}" PARENT_SCOPE)
endfunction()

function(gen_aclnn_with_opdef)
  set(opbuild_out_srcs)
  set(opbuild_out_headers)
  get_property(_dsl_packages GLOBAL PROPERTY CANNBOTDSL_ACLNN_PACKAGES)
  if(_dsl_packages)
    cannbotdsl_add_modules(opbuild_out_headers)
  endif()
  gen_aclnn_classify(${OPHOST_NAME}_opdef_aclnn_obj aclnn "${opbuild_out_srcs}" "${opbuild_out_headers}"
    opbuild_out_srcs opbuild_out_headers)
  gen_aclnn_classify(${OPHOST_NAME}_opdef_aclnn_inner_obj aclnnInner "${opbuild_out_srcs}" "${opbuild_out_headers}"
    opbuild_out_srcs opbuild_out_headers)
  gen_aclnn_classify(${OPHOST_NAME}_opdef_aclnn_exclude_obj aclnnExc "${opbuild_out_srcs}" "${opbuild_out_headers}"
    opbuild_out_srcs opbuild_out_headers)

  set(filtered_opbuild_out_headers)
  foreach(header_file ${opbuild_out_headers})
    set(skip_this_header FALSE)

    get_filename_component(header_name ${header_file} NAME)
    if(header_name MATCHES "aclnn_([^.]+)\\.h")
      set(op_name ${CMAKE_MATCH_1})
      string(TOUPPER ${op_name} op_name_upper)
      string(REPLACE "-" "_" op_name_upper ${op_name_upper})

      set(skip_var_name "${op_name_upper}_SKIP_HEADER")
      if(DEFINED ${skip_var_name} AND ${skip_var_name})
        message(STATUS "Skipping header packaging for operator: ${op_name}")
        set(skip_this_header TRUE)
      endif()
    endif()

    if(NOT skip_this_header)
      list(APPEND filtered_opbuild_out_headers ${header_file})
    endif()
  endforeach()

  set(opbuild_out_headers ${filtered_opbuild_out_headers})

  # 创建汇总头文件
  if(NOT ENABLE_BUILT_IN)
    set(aclnn_master_header_name "aclnn_ops_transformer_${VENDOR_NAME}")
  else()
    set(aclnn_master_header_name "aclnn_ops_transformer")
  endif()
  set(aclnn_master_header "${CMAKE_CURRENT_BINARY_DIR}/${aclnn_master_header_name}.h")
  gen_aclnn_master_header(${aclnn_master_header_name} "${aclnn_master_header}" "${opbuild_out_headers}")

  set(mc2_aclnn_master_headers "")
  if (NOT ENABLE_BUILT_IN AND NOT ("${ASCEND_OP_NAME}" STREQUAL "ALL"))
    foreach(op_name IN LISTS ASCEND_OP_NAME)
      file(GLOB matching_file "${OPS_TRANSFORMER_DIR}/mc2/${op_name}/op_api/aclnn_*.h")
      list(APPEND mc2_aclnn_master_headers ${matching_file})
    endforeach()
  else()
    file(GLOB matching_file "${OPS_TRANSFORMER_DIR}/mc2/*/op_api/aclnn_*.h")
    list(APPEND mc2_aclnn_master_headers ${matching_file})
  endif()

  # 读取 level2_ops_list 文件
  set(level2_ops_list_file "${CMAKE_CURRENT_LIST_DIR}/cmake/level2_ops_list")
  if(EXISTS ${level2_ops_list_file})
    file(STRINGS ${level2_ops_list_file} level2_ops_list)
  else()
    set(level2_ops_list "")
  endif()

  # 过滤 mc2 头文件：只在 level2_ops_list 中的才安装到 level2
  set(mc2_level2_headers "")
  foreach(header_file ${mc2_aclnn_master_headers})
    get_filename_component(header_name ${header_file} NAME)
    list(FIND level2_ops_list "${header_name}" idx)
    if(NOT idx EQUAL -1)
      list(APPEND mc2_level2_headers ${header_file})
    endif()
  endforeach()

  # 将头文件安装到packages/vendors/vendor_name/op_api/include
  if (NOT ENABLE_BUILT_IN)
    install(FILES ${opbuild_out_headers} DESTINATION ${ACLNN_INC_INSTALL_DIR} OPTIONAL)
    install(FILES ${aclnn_master_header} DESTINATION ${ACLNN_INC_INSTALL_DIR} OPTIONAL)
    if (BUILD_OPEN_PROJECT AND mc2_aclnn_master_headers)
      install(FILES ${mc2_aclnn_master_headers} DESTINATION ${ACLNN_INC_INSTALL_DIR} OPTIONAL)
    endif()
  else()
    install(FILES ${opbuild_out_headers} DESTINATION ${ACLNN_INC_INSTALL_DIR} OPTIONAL)
    install(FILES ${aclnn_master_header} DESTINATION ${ACLNN_INC_INSTALL_DIR} OPTIONAL)
    install(FILES ${opbuild_out_headers} DESTINATION ${ACLNN_INC_LEVEL2_INSTALL_DIR} OPTIONAL)
    install(FILES ${aclnn_master_header} DESTINATION ${ACLNN_INC_LEVEL2_INSTALL_DIR} OPTIONAL)
    install(CODE "
      set(level2_dest \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${ACLNN_INC_LEVEL2_INSTALL_DIR}\")
      foreach(h ${opbuild_out_headers} ${aclnn_master_header})
        get_filename_component(h_name \"\${h}\" NAME)
        set(dest_file \"\${level2_dest}/\${h_name}\")
        if(EXISTS \"\${dest_file}\")
          execute_process(
            COMMAND ${ASCEND_PYTHON_EXECUTABLE} \"${PROJECT_SOURCE_DIR}/scripts/util/add_deprecation_warning.py\" \"\${dest_file}\"
          )
        endif()
      endforeach()
    ")
    if (BUILD_OPEN_PROJECT)
      install(FILES ${mc2_aclnn_master_headers} DESTINATION ${ACLNN_INC_INSTALL_DIR} OPTIONAL)
      install(FILES ${mc2_level2_headers} DESTINATION ${ACLNN_INC_LEVEL2_INSTALL_DIR} OPTIONAL)
      install(CODE "
        set(level2_dest \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${ACLNN_INC_LEVEL2_INSTALL_DIR}\")
        foreach(h ${mc2_aclnn_master_headers} ${mc2_level2_headers})
          get_filename_component(h_name \"\${h}\" NAME)
          set(dest_file \"\${level2_dest}/\${h_name}\")
          if(EXISTS \"\${dest_file}\")
            execute_process(
              COMMAND ${ASCEND_PYTHON_EXECUTABLE} \"${PROJECT_SOURCE_DIR}/scripts/util/add_deprecation_warning.py\" \"\${dest_file}\"
            )
          endif()
        endforeach()
      ")
    endif()
  endif()

  if (ENABLE_STATIC)
    install(FILES ${opbuild_out_headers} DESTINATION ${CMAKE_BINARY_DIR}/static_library_files/include/aclnnop OPTIONAL)
    install(FILES ${aclnn_master_header} DESTINATION ${CMAKE_BINARY_DIR}/static_library_files/include/aclnnop OPTIONAL)
    install(FILES ${opbuild_out_headers} DESTINATION ${CMAKE_BINARY_DIR}/static_library_files/include/aclnnop/level2 OPTIONAL)
    install(FILES ${aclnn_master_header} DESTINATION ${CMAKE_BINARY_DIR}/static_library_files/include/aclnnop/level2 OPTIONAL)
    if (BUILD_OPEN_PROJECT)
      install(FILES ${mc2_aclnn_master_headers} DESTINATION ${CMAKE_BINARY_DIR}/static_library_files/include/aclnnop OPTIONAL)
      install(FILES ${mc2_level2_headers} DESTINATION ${CMAKE_BINARY_DIR}/static_library_files/include/aclnnop/level2 OPTIONAL)
    endif()
  endif()

  # ascendc_impl_gen depends opbuild_custom_gen_aclnn_all, for opbuild will generate .ini
  set(dependency_list)
  if(TARGET opbuild_gen_aclnn)
    list(APPEND dependency_list opbuild_gen_aclnn)
  endif()
  if(TARGET opbuild_gen_aclnnInner)
    list(APPEND dependency_list opbuild_gen_aclnnInner)
  endif()
  if(TARGET opbuild_gen_aclnnExc)
    list(APPEND dependency_list opbuild_gen_aclnnExc)
  endif()
  if(NOT dependency_list)
    message(STATUS "no operator info to generate")
    return()
  endif()
  add_custom_target(opbuild_custom_gen_aclnn_all)
  add_dependencies(opbuild_custom_gen_aclnn_all ${dependency_list})
  if(opbuild_out_srcs)
    set_source_files_properties(${opbuild_out_srcs} PROPERTIES GENERATED TRUE)
    add_library(opbuild_gen_aclnn_all OBJECT ${opbuild_out_srcs})
    add_dependencies(
      opbuild_gen_aclnn_all
      opbuild_custom_gen_aclnn_all
    )
    target_include_directories(opbuild_gen_aclnn_all
      PRIVATE
      ${OPAPI_INCLUDE}
    )
  endif()
endfunction()

function(gen_aicpu_ini_from_opdef)
  message(STATUS "Opbuild generating AICPU ini/json from OpDef")
  cmake_parse_arguments(OPBUILD "" "OUT_DIR;PROJECT_NAME;ACCESS_PREFIX;ENABLE_SOURCE" "OPS_SRC" ${ARGN})
  if(NOT OPBUILD_OPS_SRC)
    message(STATUS "No AICPU OpDef sources, skip.")
    return()
  endif()

  file(MAKE_DIRECTORY ${OPBUILD_OUT_DIR})

  set(AICPU_OPDEF_SO ${OPBUILD_OUT_DIR}/libaicpu_ops.so)
  set(AICPU_INI ${OPBUILD_OUT_DIR}/aicpu_kernel.ini)
  set(opbuild_env "")
  if(NOT "${OPBUILD_PROJECT_NAME}x" STREQUAL "x")
    list(APPEND opbuild_env "OPS_PROJECT_NAME=${OPBUILD_PROJECT_NAME}")
  endif()
  if(NOT "${OPBUILD_ACCESS_PREFIX}x" STREQUAL "x")
    list(APPEND opbuild_env "OPS_DIRECT_ACCESS_PREFIX=${OPBUILD_ACCESS_PREFIX}")
  endif()

  add_custom_command(
    OUTPUT ${AICPU_INI}
    COMMAND ${CMAKE_CXX_COMPILER} -g -fPIC -shared -std=c++11 ${OPBUILD_OPS_SRC} -D_GLIBCXX_USE_CXX11_ABI=0
            -I ${ASCEND_CANN_PACKAGE_PATH}/include
            -I ${CMAKE_SOURCE_DIR}/common/include
            -L ${ASCEND_CANN_PACKAGE_PATH}/lib64
            -lexe_graph -lregister -ltiling_api
            -o ${AICPU_OPDEF_SO}
    COMMAND ${CMAKE_COMMAND} -E env ENABLE_SOURCE_PACKAGE=${OPBUILD_ENABLE_SOURCE} ${opbuild_env}
            ${OP_BUILD_TOOL} ${AICPU_OPDEF_SO} ${OPBUILD_OUT_DIR} --aicpu
    DEPENDS ${OPBUILD_OPS_SRC}
    COMMENT "Generating aicpu_kernel.ini from *_aicpu_def.cpp"
    VERBATIM
  )

  message(STATUS "Opbuild generating AICPU ini/json from OpDef - done")
endfunction()

function(merge_graph_headers)
  set(oneValueArgs TARGET OUT_DIR)
  cmake_parse_arguments(MGPROTO "" "${oneValueArgs}" "" ${ARGN})
  get_target_property(proto_headers ${GRAPH_PLUGIN_NAME}_proto_headers INTERFACE_SOURCES)
  add_custom_command(OUTPUT ${MGPROTO_OUT_DIR}/ops_proto_transformer.h
    COMMAND ${ASCEND_PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/util/merge_proto.py
    ${proto_headers}
    --output-file ${MGPROTO_OUT_DIR}/ops_proto_transformer.h
  )
  add_custom_command(
    OUTPUT ${MGPROTO_OUT_DIR}/ops_proto_transformer.cpp
    COMMAND ${CMAKE_COMMAND} -E copy
      ${MGPROTO_OUT_DIR}/ops_proto_transformer.h
      ${MGPROTO_OUT_DIR}/ops_proto_transformer.cpp
    DEPENDS ${MGPROTO_OUT_DIR}/ops_proto_transformer.h
  )
  add_custom_target(${MGPROTO_TARGET} ALL
    DEPENDS
    ${MGPROTO_OUT_DIR}/ops_proto_transformer.h
    ${MGPROTO_OUT_DIR}/ops_proto_transformer.cpp
  )
endfunction()
