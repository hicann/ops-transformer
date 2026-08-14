# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

function(cpack_empty_package)
    add_cann_third_party(makeself-fetch)

  if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
      message(STATUS "Detected architecture: x86_64")
      set(ARCH x86_64)
  elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|arm")
      message(STATUS "Detected architecture: ARM64")
      set(ARCH aarch64)
  else ()
      message(WARNING "Unknown architecture: ${CMAKE_SYSTEM_PROCESSOR}")
  endif ()

  # 空包早于 variables.cmake return，需自行设置默认 component（deb/rpm staging 依赖）
  set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME ops-transformer)

  # CPack config
  install(FILES ${CMAKE_BINARY_DIR}/version.ops-transformer.info
      DESTINATION share/info/ops_transformer
      RENAME version.info
  )
  install(FILES ${CMAKE_SOURCE_DIR}/scripts/package/ops_transformer/scripts/help.info
      DESTINATION share/info/ops_transformer/script
  )
  install(FILES ${CMAKE_SOURCE_DIR}/scripts/package/ops_transformer/scripts/empty_package_scripts/install.sh
      DESTINATION share/info/ops_transformer/script
      PERMISSIONS OWNER_EXECUTE OWNER_READ OWNER_WRITE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
  )
  install(FILES ${CMAKE_SOURCE_DIR}/scripts/package/ops_transformer/scripts/empty_package_scripts/cleanup.sh
      DESTINATION share/info/ops_transformer/script
      PERMISSIONS OWNER_EXECUTE OWNER_READ OWNER_WRITE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
  )
  string(FIND "${ASCEND_COMPUTE_UNIT}" ";" SEMICOLON_INDEX)
  if (SEMICOLON_INDEX GREATER -1)
      # 截取分号前的字串
      math(EXPR SUBSTRING_LENGTH "${SEMICOLON_INDEX}")
      string(SUBSTRING "${ASCEND_COMPUTE_UNIT}" 0 "${SUBSTRING_LENGTH}" compute_unit)
  else()
      # 没有分号取全部内容
      set(compute_unit "${ASCEND_COMPUTE_UNIT}")
  endif()
  set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/build_out)
  set_cann_cpack_config(ops-transformer COMPUTE_UNIT ${compute_unit} SHARE_INFO_NAME ops_transformer
      OUTPUT ${CMAKE_INSTALL_PREFIX} PACKAGE_TYPE "${PACKAGE_TYPE}")
endfunction()
