# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE
        generic_block_sparse_attention_def.cpp
    )
endif()

if (BUILD_OPS_RTY_KERNEL)
    add_ops_compile_options(
            OP_NAME GenericBlockSparseAttention
            OPTIONS --cce-auto-sync=off
                    -DASCENDC_DUMP=1
                    -Wno-deprecated-declarations
                    -Werror
                    -fpermissive
    )
else()
    # GBSA's arch22 infrastructure reuses the Catlass coordinate/layout
    # primitives as a third-party dependency instead of copying them into the
    # operator tree.  asc_opc does not add the Catlass include root
    # automatically.  Prefer the third-party catlass resolved by
    # cmake/third_party/catlass.cmake (v2.0.0) and only fall back to the
    # headers shipped inside the CANN installation when it is unavailable.
    set(GBSA_CATLASS_INCLUDE "")
    if (DEFINED CATLASS AND EXISTS "${CATLASS}/include/catlass/catlass.hpp")
        set(GBSA_CATLASS_INCLUDE "${CATLASS}/include")
    else()
        set(GBSA_CATLASS_FALLBACK
            "${ASCEND_DIR}/opp/built-in/op_impl/ai_core/tbe/impl/ops_legacy/ascendc/common/catlass/include")
        if (EXISTS "${GBSA_CATLASS_FALLBACK}/catlass/catlass.hpp")
            set(GBSA_CATLASS_INCLUDE "${GBSA_CATLASS_FALLBACK}")
        endif()
    endif()
    if (NOT "${GBSA_CATLASS_INCLUDE}" STREQUAL "")
        message(STATUS "GBSA kernel catlass include: ${GBSA_CATLASS_INCLUDE}")
        add_ops_compile_options(
                OP_NAME GenericBlockSparseAttention
                OPTIONS "-I${GBSA_CATLASS_INCLUDE}"
                        "-DCATLASS_ARCH=2201")
    endif()
    add_modules_sources(OPTYPE generic_block_sparse_attention ACLNNTYPE aclnn_exclude)
endif()
