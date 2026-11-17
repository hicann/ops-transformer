# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Keep NPU architecture distinct from the source directory (arch35 is shared
# by several different NPU architectures). Unknown SoCs return an empty string.
function(get_npu_arch_by_soc_version soc_version out_var)
    string(TOLOWER "${soc_version}" _soc)
    if(_soc MATCHES "^ascend910b[1-4]$" OR _soc STREQUAL "ascend910b"
       OR _soc STREQUAL "ascend910b2c" OR _soc STREQUAL "ascend910b4-1"
       OR _soc STREQUAL "ascend910_93" OR _soc MATCHES "^ascend910_93(91|81|72|92|82|62)$")
        set(_arch dav_2201)
    elseif(_soc STREQUAL "ascend950" OR _soc STREQUAL "ascend950pr_9599"
           OR _soc STREQUAL "ascend350" OR _soc STREQUAL "ascend350_355e")
        set(_arch dav_3510)
    elseif(_soc STREQUAL "ascend960dt" OR _soc STREQUAL "ascend960dt_968b7")
        set(_arch dav_9201)
    elseif(_soc STREQUAL "ascend960pr" OR _soc STREQUAL "ascend960pr_969a1")
        set(_arch dav_9202)
    else()
        set(_arch "")
    endif()
    set(${out_var} "${_arch}" PARENT_SCOPE)
endfunction()

# Expand an architecture into the long SoC identifiers consumed by OPC and
# opdesc_parser. Existing SoC entries pass through unchanged.
function(expand_compile_compute_unit unit out_var)
    string(TOLOWER "${unit}" _unit)
    if(_unit STREQUAL "dav_2201")
        set(_units ascend910b1 ascend910b2 ascend910b2c ascend910b3 ascend910b4 ascend910b4-1
            ascend910_9391 ascend910_9381 ascend910_9372 ascend910_9392 ascend910_9382 ascend910_9362)
    elseif(_unit STREQUAL "dav_3510")
        set(_units ascend950pr_9599 ascend350_355e)
    elseif(_unit STREQUAL "dav_9201")
        set(_units ascend960dt_968b7)
    elseif(_unit STREQUAL "dav_9202")
        set(_units ascend960pr_969a1)
    elseif(_unit MATCHES "^dav_")
        message(FATAL_ERROR "Unsupported NPU architecture in COMPUTE_UNIT: ${unit}")
    else()
        set(_units "${unit}")
    endif()
    set(${out_var} "${_units}" PARENT_SCOPE)
endfunction()
