# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
set(OPS_TEST_KIT_TAG_ID master)
set(OPS_TEST_KIT_PKG ops-test-kit.tar.gz)

if(EXISTS "${PROJECT_SOURCE_DIR}/build/third_party/ops-test-kit")
  get_filename_component(OPS_TEST_KIT_SOURCE_PATH
                         ${PROJECT_SOURCE_DIR}/build/third_party/ops-test-kit REALPATH)
  message(STATUS "Find ops-test-kit source dir: ${OPS_TEST_KIT_SOURCE_PATH}")
elseif(EXISTS "${CANN_3RD_LIB_PATH}/ops-test-kit")
  get_filename_component(OPS_TEST_KIT_SOURCE_PATH
                         ${CANN_3RD_LIB_PATH}/ops-test-kit REALPATH)
  message(STATUS "Find ops-test-kit source dir: ${OPS_TEST_KIT_SOURCE_PATH}")
else()
  if(EXISTS "${CANN_3RD_LIB_PATH}/pkg/${OPS_TEST_KIT_PKG}")
    set(OPS_TEST_KIT_URL "file://${CANN_3RD_LIB_PATH}/pkg/${OPS_TEST_KIT_PKG}")
    message(STATUS "[ThirdPartyLib][ops-test-kit] found in ${OPS_TEST_KIT_URL}.")
    include(FetchContent)
    FetchContent_Declare(
      ops_test_kit
      URL ${OPS_TEST_KIT_URL}
      SOURCE_DIR ${PROJECT_SOURCE_DIR}/build/third_party/ops-test-kit
    )
  else()
    execute_process(
      COMMAND git remote get-url origin
      WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
      OUTPUT_VARIABLE GIT_REMOTE_URL
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET
    )
    if(GIT_REMOTE_URL MATCHES "^git@|^ssh://")
      set(OPS_TEST_KIT_GIT_URL "git@atomgit.com:cann/ops-test-kit.git")
      message(STATUS "[ThirdPartyLib][ops-test-kit] using SSH protocol: ${OPS_TEST_KIT_GIT_URL}")
    else()
      set(OPS_TEST_KIT_GIT_URL "https://atomgit.com/cann/ops-test-kit.git")
      message(STATUS "[ThirdPartyLib][ops-test-kit] using HTTPS protocol: ${OPS_TEST_KIT_GIT_URL}")
    endif()
    include(FetchContent)
    FetchContent_Declare(
      ops_test_kit
      GIT_REPOSITORY ${OPS_TEST_KIT_GIT_URL}
      GIT_TAG ${OPS_TEST_KIT_TAG_ID}
      GIT_PROGRESS TRUE
      SOURCE_DIR ${PROJECT_SOURCE_DIR}/build/third_party/ops-test-kit
    )
  endif()
  FetchContent_Populate(ops_test_kit)
  set(OPS_TEST_KIT_SOURCE_PATH ${PROJECT_SOURCE_DIR}/build/third_party/ops-test-kit)
endif()
