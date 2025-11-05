# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

if (BUILD_OPEN_PROJECT)  # custom
    target_sources(op_host_aclnnInner PRIVATE
        elastic_receivable_test_def.cpp
    )
    add_mc2_modules_sources(OPTYPE elastic_receivable_test ACLNNTYPE aclnn_inner)
    set(MC2_OPT ON PARENT_SCOPE)
    set(elastic_receivable_test_depends mc2/common mc2/3rd mc2/moe_distribute_dispatch PARENT_SCOPE)
else()
    add_mc2_modules_sources(OPTYPE elastic_receivable_test ACLNNTYPE aclnn_inner) 
endif()
