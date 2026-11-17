# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Registration libraries are loaded separately by opbuild. Each nonempty
# library needs the helper implementation and headers from the same opbase.
function(add_registration_arch_helper target)
    set(_source "${OPBASE_SOURCE_PATH}/src/op_common/op_host/util/op_arch_helper.cpp")
    if(NOT EXISTS "${_source}")
        message(FATAL_ERROR "Missing opbase architecture helper: ${_source}")
    endif()
    get_target_property(_sources ${target} SOURCES)
    if(NOT _source IN_LIST _sources)
        target_sources(${target} PRIVATE "${_source}")
    endif()
    target_include_directories(${target} BEFORE PRIVATE "${OPBASE_SOURCE_PATH}/include")
endfunction()
