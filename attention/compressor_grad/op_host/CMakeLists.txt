# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================
require_pypto_pro(compressor_grad)
add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE
    compressor_grad_def.cpp
    )
    enable_pypto_kernel(compressor_grad)
endif()

set(CONDITION_UNIT ${ASCEND_COMPUTE_UNIT})
if("${CONDITION_UNIT}" STREQUAL "ascend950")
    # PyPTO kernel: runs codegen at configure time and exposes PYPTO_GEN_DIR for the host tiling -I below.
    # Must be called BEFORE add_modules_sources(...) (which reads PYPTO_GEN_DIR).
    add_ops_compile_options(
        OP_NAME CompressorGrad
        COMPUTE_UNIT Ascend950PR_9599
        OPTIONS -DENABLE_CV_COMM_VIA_SSBUF=true
    )
endif()

if (BUILD_OPS_RTY_KERNEL)
    add_ops_compile_options(
        OP_NAME CompressorGrad
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -Werror
    )
else()
    add_ops_compile_options(
        OP_NAME CompressorGrad
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
    )
endif()

if(NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE compressor_grad ACLNNTYPE aclnn)
endif()

if(NOT BUILD_OPS_RTY_KERNEL)
    add_tiling_modules()
endif()

