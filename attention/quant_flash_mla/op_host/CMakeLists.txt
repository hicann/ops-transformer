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

set(QUANT_FLASH_MLA_SKIP_HEADER TRUE CACHE INTERNAL "Skip packaging header for quant_flash_mla")
set(QUANT_FLASH_MLA_INNER_SKIP_HEADER TRUE CACHE INTERNAL "Skip packaging inner header for quant_flash_mla")

# kernel 入口 quant_flash_mla.cpp 顶部无条件 include ../../common/op_kernel（arch35 公共头），
# arch22 收编入口另跨算子 include quant_flash_mla_metadata 的布局头（qfmla_metadata_arch22_layout.h）
# 与 incre_flash_attention 的公共头/service 头（fork 副本复用 IFA 原文件），
# 所有 compute unit（含 ascend910b/ascend910_93 收编分支）构建均需拷贝这三个依赖目录。
# 另补 prompt_flash_attention（传递依赖）：IFA 的 op_host tiling 源(incre_flash_attention_tiling_arch38.cpp)
# 引用 PFA arch38 tiling 类，需其随 liboptiling 一并编译，否则该库带未定义 vtable 符号
# （--ops 过滤下 PFA 自身不被构建；构建期无 tiling 库加载不报错，运行期 dlopen 会失败）
set(quant_flash_mla_depends attention/common attention/quant_flash_mla_metadata attention/incre_flash_attention attention/prompt_flash_attention CACHE INTERNAL "Dependencies for quant_flash_mla")

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnInner PRIVATE
        quant_flash_mla_def.cpp
    )
endif()

set(CONDITION_UNIT ${ASCEND_COMPUTE_UNIT})
if("${CONDITION_UNIT}" STREQUAL "ascend950")
    set(quant_flash_mla_depends attention/common CACHE INTERNAL "Dependencies for quant_flash_mla")
    add_ops_compile_options(
        OP_NAME QuantFlashMla
        COMPUTE_UNIT Ascend950PR_9599
        OPTIONS -DENABLE_CV_COMM_VIA_SSBUF=true
    )
elseif("${CONDITION_UNIT}" STREQUAL "ascend960dt")
    set(quant_flash_mla_depends attention/common CACHE INTERNAL "Dependencies for quant_flash_mla")
endif()

file(GLOB_RECURSE ARCH35_CHECKER_SRC_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/checkers/*checker*.cpp
)

if (BUILD_OPS_RTY_KERNEL)
    add_ops_compile_options(
            OP_NAME QuantFlashMla
            OPTIONS --cce-auto-sync=off
                    -Wno-deprecated-declarations
                    -Werror
    )
else()
    add_ops_compile_options(
            OP_NAME QuantFlashMla
            OPTIONS --cce-auto-sync=off
                    -Wno-deprecated-declarations
                    -Werror
    )
    add_tiling_modules()
    # arch 模板源（arch35/arch92/arch22）不在本文件显式列举：add_modules_sources_with_soc 按
    # SOC->arch 目录映射（根 CMakeLists: ascend910b/ascend910_93->arch22）自动 glob
    # ${SOURCE_DIR}/${ARCH}/*_tiling*.cpp 加入 tiling_obj，
    # 故 arch22/quant_flash_mla_tiling_arch22.cpp 在 910B/910_93 构建下自动参与编译，与
    # arch35/arch92 的接入方式一致，无需（也不应）加入下面的通用源列表。
    target_sources(${OPHOST_NAME}_tiling_obj PRIVATE
    ${ARCH35_CHECKER_SRC_FILES}
    qfmla_tiling_info_parser.cpp
    )
    add_modules_sources_with_soc(
        OP_API_INDEPENDENT ON
        OP_API_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../op_api
        OPTYPE quant_flash_mla
        ACLNNTYPE aclnn_inner
    )
endif()
