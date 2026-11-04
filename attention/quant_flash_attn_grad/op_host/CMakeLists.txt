# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
set(_qfag_excluded_versions "9.2.0-beta.1;9.2.0-beta.2")
set(_qfag_cann_version "")
set(_qfag_ver_src "")
if(DEFINED CANN_VERSION AND NOT "${CANN_VERSION}" STREQUAL "")
    set(_qfag_cann_version "${CANN_VERSION}")
    set(_qfag_ver_src "cmake variable CANN_VERSION")
else()
    foreach(_qfag_ver_file
            "${ASCEND_CANN_PACKAGE_PATH}/toolkit/version.info"
            "${ASCEND_CANN_PACKAGE_PATH}/version.info"
            "${ASCEND_CANN_PACKAGE_PATH}/opp/version.info"
            "${ASCEND_CANN_PACKAGE_PATH}/compiler/version.info")
        if(EXISTS "${_qfag_ver_file}")
            file(READ "${_qfag_ver_file}" _qfag_ver_content)
            string(REPLACE "\r" "" _qfag_ver_content "${_qfag_ver_content}")
            string(REPLACE "\n" ";" _qfag_ver_lines "${_qfag_ver_content}")
            foreach(_qfag_line IN LISTS _qfag_ver_lines)
                if(_qfag_line MATCHES "^Version=")
                    string(SUBSTRING "${_qfag_line}" 8 -1 _qfag_cann_version)
                    set(_qfag_ver_src "${_qfag_ver_file}")
                    break()
                endif()
            endforeach()
            if(NOT "${_qfag_cann_version}" STREQUAL "")
                break()
            endif()
        endif()
    endforeach()
endif()

string(TOLOWER "${_qfag_cann_version}" _qfag_cann_version)
if("${_qfag_cann_version}" STREQUAL "")
    message(WARNING "[quant_flash_attn_grad] Cannot determine CANN version, exclusion check skipped, operator will be compiled")
else()
    message(STATUS "[quant_flash_attn_grad] CANN version ${_qfag_cann_version} (from ${_qfag_ver_src}), excluded: ${_qfag_excluded_versions}")
endif()
if(_qfag_cann_version IN_LIST _qfag_excluded_versions)
    message(STATUS "[quant_flash_attn_grad] Skipped: operator not supported on CANN version ${_qfag_cann_version}")
    set_property(GLOBAL PROPERTY quant_flash_attn_grad_CANN_VERSION_SKIPPED TRUE)
    set_property(GLOBAL APPEND PROPERTY CANN_VERSION_SKIPPED_OPS "quant_flash_attn_grad (cann=${_qfag_cann_version} not supported)")
    return()
endif()

require_pypto_pro(quant_flash_attn_grad)
add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnInner PRIVATE
        quant_flash_attn_grad_def.cpp
    )
    add_modules_sources(OPTYPE quant_flash_attn_grad ACLNNTYPE aclnn_inner)
    enable_pypto_kernel(quant_flash_attn_grad)
endif()

set(CONDITION_UNIT ${ASCEND_COMPUTE_UNIT})
if("${CONDITION_UNIT}" STREQUAL "ascend950")
    # PyPTO kernel: run codegen at configure time and exposes PYPTO_GEN_DIR for the host tiling -I below.
    # Must be called BEFORE add_modules_sources(...) (which reads PYPTO_GEN_DIR).
    add_ops_compile_options(
        OP_NAME QuantFlashAttnGrad
        COMPUTE_UNIT Ascend950PR_9599
        OPTIONS -DENABLE_CV_COMM_VIA_SSBUF=true
    )
endif()

if (BUILD_OPS_RTY_KERNEL)
    add_ops_compile_options(
        OP_NAME QuantFlashAttnGrad
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -Werror
    )
else()
    add_ops_compile_options(
        OP_NAME QuantFlashAttnGrad
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
    )
endif()
if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources_with_soc(OPTYPE quant_flash_attn_grad ACLNNTYPE aclnn_inner)
endif()

if (NOT BUILD_OPS_RTY_KERNEL)
    add_tiling_modules()
endif()
