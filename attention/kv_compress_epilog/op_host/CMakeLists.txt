# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE
        kv_compress_epilog_def.cpp
    )

    if("${ASCEND_COMPUTE_UNIT}" STREQUAL "ascend950")
        add_ops_compile_options(
            OP_NAME KvCompressEpilog
            COMPUTE_UNIT Ascend950PR_9599
            OPTIONS -mllvm -cce-aicore-dcci-before-kernel-end=false
                -mllvm -cce-aicore-hoist-movemask=false
                --op_relocatable_kernel_binary=true
        )
    elseif("${ASCEND_COMPUTE_UNIT}" STREQUAL "ascend350")
        add_ops_compile_options(
            OP_NAME KvCompressEpilog
            COMPUTE_UNIT Ascend350_355e
            OPTIONS -mllvm -cce-aicore-dcci-before-kernel-end=false
                -mllvm -cce-aicore-hoist-movemask=false
                --op_relocatable_kernel_binary=true
        )
    elseif("${ASCEND_COMPUTE_UNIT}" STREQUAL "ascend960dt")
        add_ops_compile_options(
            OP_NAME KvCompressEpilog
            COMPUTE_UNIT ascend960dt_968B7
            OPTIONS -mllvm -cce-aicore-dcci-before-kernel-end=false
                -mllvm -cce-aicore-hoist-movemask=false
                --op_relocatable_kernel_binary=true
        )
    elseif("${ASCEND_COMPUTE_UNIT}" STREQUAL "ascend960pr")
        add_ops_compile_options(
            OP_NAME KvCompressEpilog
            COMPUTE_UNIT Ascend960PR_969A1
            OPTIONS -mllvm -cce-aicore-dcci-before-kernel-end=false
                -mllvm -cce-aicore-hoist-movemask=false
                --op_relocatable_kernel_binary=true
        )
    else()
        add_ops_compile_options(
            OP_NAME KvCompressEpilog
            OPTIONS --cce-auto-sync=off
                    -Wno-deprecated-declarations
                    -Werror
                    -mllvm -cce-aicore-hoist-movemask=false
                    --op_relocatable_kernel_binary=true
        )
    endif()
endif()

if(NOT BUILD_OPS_RTY_KERNEL)
    add_op_to_compiled_list()
    add_modules_sources(OPTYPE kv_compress_epilog ACLNNTYPE aclnn_exclude)

    # arch 专属 tiling 源位于 arch22/ arch35/ 子目录，add_modules_sources 仅收集顶层 *_tiling*.cpp，
    # 在此显式追加；arch35 分派编译单元 kv_compress_epilog_tiling.cpp 仅在 arch35 源码存在时
    # 定义 KV_CE_ENABLE_ARCH35_TILING（删除 arch35 目录后 arch22 quant_mode=1 仍可独立编译运行）。
    target_sources(${OPHOST_NAME}_tiling_obj PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/arch22/kv_compress_epilog_tiling_arch22.cpp)
    set(KV_CE_TILING_ARCH35_SRC ${CMAKE_CURRENT_SOURCE_DIR}/arch35/kv_compress_epilog_tiling_arch35.cpp)
    if(EXISTS ${KV_CE_TILING_ARCH35_SRC})
        target_sources(${OPHOST_NAME}_tiling_obj PRIVATE ${KV_CE_TILING_ARCH35_SRC})
        set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/kv_compress_epilog_tiling.cpp
            TARGET_DIRECTORY ${OPHOST_NAME}_tiling_obj
            PROPERTIES COMPILE_DEFINITIONS KV_CE_ENABLE_ARCH35_TILING)
    endif()

    # kernel 源安装时 op_kernel/arch* 目录由 filter_copy_files 依据 ARCH_DIRECTORY 与本算子级
    # 覆盖变量（func.cmake: ${_op_name}_${_COMPUTE_UNIT}_arch_directory）放行
    set(kv_compress_epilog_ascend910b_arch_directory arch22 CACHE INTERNAL
        "kv_compress_epilog kernel arch dir for ascend910b")
    set(kv_compress_epilog_ascend910_93_arch_directory arch22 CACHE INTERNAL
        "kv_compress_epilog kernel arch dir for ascend910_93")
    set(kv_compress_epilog_ascend950_arch_directory arch35 CACHE INTERNAL
        "kv_compress_epilog kernel arch dir for ascend950")
    set(kv_compress_epilog_ascend960dt_arch_directory arch35 CACHE INTERNAL
        "kv_compress_epilog kernel arch dir for ascend960dt")
    set(kv_compress_epilog_ascend350_arch_directory arch35 CACHE INTERNAL
        "kv_compress_epilog kernel arch dir for ascend350")
endif()
