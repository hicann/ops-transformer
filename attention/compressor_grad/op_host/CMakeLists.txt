# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# 双栈：arch35(ascend950)=PyPTO，arch92(ascend960dt)=PyPTO，arch22(ascend910b/ascend910_93)=原生 AscendC
set(_COMPRESSOR_GRAD_WITH_PYPTO OFF)
if ("ascend950" IN_LIST ASCEND_COMPUTE_UNIT OR "ascend960dt" IN_LIST ASCEND_COMPUTE_UNIT)
    set(_COMPRESSOR_GRAD_WITH_PYPTO ON)
endif()

if (_COMPRESSOR_GRAD_WITH_PYPTO)
    require_pypto_pro(compressor_grad)
endif()
add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE
    compressor_grad_def.cpp
    )
    if (_COMPRESSOR_GRAD_WITH_PYPTO)
        # 配置期 codegen，导出 PYPTO_GEN_DIR 供 tiling -I；须在 add_modules_sources 之前调用
        enable_pypto_kernel(compressor_grad)
    endif()
endif()

set(CONDITION_UNIT ${ASCEND_COMPUTE_UNIT})
if("${CONDITION_UNIT}" STREQUAL "ascend950")
    add_ops_compile_options(
        OP_NAME CompressorGrad
        COMPUTE_UNIT Ascend950PR_9599
        OPTIONS -DENABLE_CV_COMM_VIA_SSBUF=true
    )
elseif("${CONDITION_UNIT}" STREQUAL "ascend960dt")
    add_ops_compile_options(
        OP_NAME CompressorGrad
        COMPUTE_UNIT Ascend960DT_968B7
        OPTIONS -DENABLE_CV_COMM_VIA_SSBUF=true
    )
endif()

# --cce-auto-sync：arch22(910b/910_93) 显式 on（scatter 数值正确性依赖）；
# PyPTO(950/960dt) 显式 off（kernel 自管理跨核同步，与 quant_compressor 约定一致）
if (BUILD_OPS_RTY_KERNEL)
    add_ops_compile_options(
        OP_NAME CompressorGrad
        OPTIONS -Wno-deprecated-declarations
                -Werror
    )
else()
    add_ops_compile_options(
        OP_NAME CompressorGrad
        OPTIONS -Wno-deprecated-declarations
    )
endif()
if(NOT _COMPRESSOR_GRAD_WITH_PYPTO)
    add_ops_compile_options(
        OP_NAME CompressorGrad
        OPTIONS --cce-auto-sync=on
    )
endif()
add_ops_compile_options(
    OP_NAME CompressorGrad
    COMPUTE_UNIT Ascend950PR_9599
    OPTIONS --cce-auto-sync=off
)
add_ops_compile_options(
    OP_NAME CompressorGrad
    COMPUTE_UNIT Ascend960DT_968B7
    OPTIONS --cce-auto-sync=off
)

if(NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE compressor_grad ACLNNTYPE aclnn)
    add_tiling_modules()
    # arch22 tiling 位于 op_host/arch22/（不在 add_modules_sources glob 范围），显式列出
    target_sources(${OPHOST_NAME}_tiling_obj PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/arch22/compressor_grad_tiling.cpp
    )
    if (_COMPRESSOR_GRAD_WITH_PYPTO)
        # PyPTO tiling 与 register 中的 arch35 分支仅在 950/960dt(PyPTO) 构建时编译
        set_source_files_properties(
            ${CMAKE_CURRENT_SOURCE_DIR}/compressor_grad_tiling.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/compressor_grad_tiling_register.cpp
            TARGET_DIRECTORY ${OPHOST_NAME}_tiling_obj
            PROPERTIES COMPILE_DEFINITIONS COMPRESSOR_GRAD_HAS_ARCH35)
    endif()
endif()
