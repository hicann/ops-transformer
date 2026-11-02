# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if(UT_TEST_ALL OR OP_API_UT)
    add_modules_ut_sources(UT_NAME ${OP_API_MODULE_NAME} MODE PRIVATE DIR ${CMAKE_CURRENT_SOURCE_DIR})

    # The test overrides aclOpExecutor::ConvertToTensor and resolves the original
    # implementation with dlopen/dlsym. The case object is added directly to the
    # final UT executables, so link dependencies on the object library do not
    # propagate. Link libdl after the shared UT framework creates those targets.
    if(CMAKE_DL_LIBS)
        if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.19)
            function(chunk_kda_fwd_link_dl dl_lib)
                foreach(ut_target IN LISTS ARGN)
                    if(TARGET ${ut_target})
                        get_target_property(ut_link_libraries ${ut_target} LINK_LIBRARIES)
                        if(ut_link_libraries)
                            list(FIND ut_link_libraries ${dl_lib} dl_index)
                        else()
                            set(dl_index -1)
                        endif()
                        if(dl_index EQUAL -1)
                            target_link_libraries(${ut_target} PRIVATE ${dl_lib})
                        endif()
                    endif()
                endforeach()
            endfunction()
            cmake_language(
                DEFER DIRECTORY ${PROJECT_SOURCE_DIR}
                CALL chunk_kda_fwd_link_dl ${CMAKE_DL_LIBS} ${PKG_NAME}_op_host_ut ${PKG_NAME}_op_api_ut)
        else()
            set_property(DIRECTORY ${PROJECT_SOURCE_DIR} APPEND PROPERTY LINK_LIBRARIES ${CMAKE_DL_LIBS})
        endif()
    endif()
endif()
