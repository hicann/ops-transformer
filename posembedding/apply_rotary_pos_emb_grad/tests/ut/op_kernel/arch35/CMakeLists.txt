# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if (UT_TEST_ALL OR OP_KERNEL_UT)
# 算子自己的tiling文件路径
    set(apply_rotary_pos_emb_grad_tiling_files
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../../op_host/apply_rotary_pos_emb_grad_tiling.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../../op_host/apply_rotary_pos_emb_grad_tiling_a.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../../op_host/apply_rotary_pos_emb_grad_tiling_ab.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../../op_host/apply_rotary_pos_emb_grad_tiling_bab.cpp
        )
    # 使用AddOpTestCase
    # param1：算子名称，以kernel方式命名
    # param2：soc版本。须使用 CANN cpudebug 模拟库提供的标签(950 系为 Ascend950PR_9599, 见
    #         $ASCEND_HOME_PATH/tools/cpudebug/cmake/targets-tikicpulib.cmake), 与真机型号无关
    # param3：自定义编译选项，一般填写测试的一种典型数据类型组合
    # param4：该算子依赖的所有tiling源码文件
    AddOpTestCase(apply_rotary_pos_emb_grad "Ascend950PR_9599" "-DDTYPE_X=float" "${apply_rotary_pos_emb_grad_tiling_files}")

    # tiling cpp 依赖 atvoss/reduce(本仓唯一), 其 ReduceOpTiling 虚函数实现在 CANN libops_base.so,
    # UT 框架(ut.cmake)的 tiling_tmp 目标未链接该库, 会因 vtable 符号缺失导致 gen_tiling_head
    # 脚本 dlopen 失败, 此处补链
    target_link_libraries(apply_rotary_pos_emb_grad_Ascend950PR_9599_tiling_tmp PRIVATE
        ${ASCEND_DIR}/aarch64-linux/lib64/libops_base.so)

    # kernel/apt 源码 include 的 atvoss 等公共头位于仓内 third_party/opbase 的
    # pkg_inc/op_common 与 include/op_common 之下(如 atvoss/reduce/*.h、op_kernel/platform_util.h),
    # 且 grad kernel 头以 op_kernel 目录为基准相对引用前向算子 apply_rotary_pos_emb 的公共头,
    # UT 框架给 cases_obj 的 include 目录均未包含这些路径, 此处补充
    target_include_directories(apply_rotary_pos_emb_grad_Ascend950PR_9599_cases_obj PRIVATE
        ${PROJECT_SOURCE_DIR}/third_party/opbase/pkg_inc/op_common
        ${PROJECT_SOURCE_DIR}/third_party/opbase/include/op_common
        ${PROJECT_SOURCE_DIR}/third_party/opbase/include
        ${PROJECT_SOURCE_DIR}/posembedding/apply_rotary_pos_emb_grad/op_kernel
        ${PROJECT_SOURCE_DIR}/posembedding/apply_rotary_pos_emb/op_kernel)
endif()
