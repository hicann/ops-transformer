# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if (UT_TEST_ALL OR OP_KERNEL_UT)
    AddOpTestCase(distribute_barrier_extend "Ascend950PR_9599"
                  "-DDTYPE_X_REF=half -include${CMAKE_CURRENT_SOURCE_DIR}/../distribute_barrier_extend_tiling_def.h" ""
                  "${CMAKE_CURRENT_SOURCE_DIR}/../../../../op_kernel/arch35/distribute_barrier_extend_apt.cpp")
    if (TARGET distribute_barrier_extend_Ascend950PR_9599_cases_obj)
        target_compile_definitions(distribute_barrier_extend_Ascend950PR_9599_cases_obj PRIVATE
            Hcomm=DistributeBarrierHcommStub trap=DistributeBarrierTrap
            Trap=DistributeBarrierTrapApi TrapImpl=DistributeBarrierTrapImpl)
        target_sources(distribute_barrier_extend_Ascend950PR_9599_cases_obj PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/../test_distribute_barrier_extend.cpp)
    endif()
endif()
