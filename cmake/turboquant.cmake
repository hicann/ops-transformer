# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

function(gen_turboquant_aicpu_symbol)
  set(metadata_target mixed_quant_sparse_flash_mla_metadata_cust_obj)
  if(NOT ENABLE_BUILT_IN OR NOT TARGET ${metadata_target})
    return()
  endif()

  # Package only TurboQuant's metadata; retain the baseline AICPU generators.
  set(metadata_json ${OPS_TRANSFORMER_DIR}/attention/mixed_quant_sparse_flash_mla_metadata/op_kernel_aicpu/mixed_quant_sparse_flash_mla_metadata_aicpu.json)
  set(json_generator ${OPS_TRANSFORMER_DIR}/scripts/util/gen_turboquant_aicpu_info.py)
  set(turboquant_json ${CMAKE_BINARY_DIR}/aicpu_transformer_turboquant.json)
  set(kernel_name libtransformer_turboquant_aicpu.so)
  set(kernel_output ${CMAKE_BINARY_DIR}/${kernel_name})
  set(compat_vendor opp/vendors/ops_transformer_turboquant)
  set(arm_compiler ${ASCEND_DIR}/toolkit/toolchain/hcc/bin/aarch64-target-linux-gnu-g++)

  add_custom_command(
    OUTPUT ${turboquant_json}
    COMMAND ${ASCEND_PYTHON_EXECUTABLE} ${json_generator}
            ${metadata_json} ${turboquant_json} ${kernel_name}
    DEPENDS ${metadata_json} ${json_generator}
    VERBATIM
  )
  add_custom_target(turboquant_aicpu_json ALL DEPENDS ${turboquant_json})

  set(kernel_libraries)
  foreach(library libaicpu_context.a libbase_ascend_protobuf.a)
    if(EXISTS ${ASCEND_DIR}/ops_base/lib64/${library})
      list(APPEND kernel_libraries ${ASCEND_DIR}/ops_base/lib64/${library})
    else()
      list(APPEND kernel_libraries ${ASCEND_DIR}/lib64/${library})
    endif()
  endforeach()

  add_custom_command(
    OUTPUT ${kernel_output}
    COMMAND ${arm_compiler} -shared $<TARGET_OBJECTS:${metadata_target}>
      -Wl,--whole-archive ${kernel_libraries} -Wl,--no-whole-archive
      -Wl,-Bsymbolic -Wl,--exclude-libs=libbase_ascend_protobuf.a -s
      -o ${kernel_output}
    DEPENDS ${metadata_target} $<TARGET_OBJECTS:${metadata_target}> ${kernel_libraries}
    COMMENT "Linking the A2/A3 TurboQuant metadata kernel"
    COMMAND_EXPAND_LISTS
    VERBATIM
  )
  add_custom_target(turboquant_aicpu_kernels ALL DEPENDS ${kernel_output})

  install(FILES ${turboquant_json} DESTINATION opp/built-in/op_impl/aicpu/config)
  install(FILES ${kernel_output} DESTINATION opp/built-in/op_impl/aicpu/kernel)
  # Some nnopbase releases discover CUSTAICPUKernel only through vendor paths.
  install(FILES ${turboquant_json}
    DESTINATION ${compat_vendor}/op_impl/cpu/config
    RENAME cust_aicpu_kernel.json
  )
  install(FILES ${kernel_output} DESTINATION ${compat_vendor}/op_impl/cpu/aicpu_kernel/impl)
endfunction()
