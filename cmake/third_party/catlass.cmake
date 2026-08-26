# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
set(CATLASS_TAG_ID 769cd40a8716b28650b6bebb08db4834eea4462f) # v2.0.0

if(EXISTS ${TOP_DIR}/open_source/catlass)
  get_filename_component(CATLASS_SOURCE_PATH
                         ${TOP_DIR}/open_source/catlass REALPATH)
  message(STATUS "Find catlass source dir: ${CATLASS_SOURCE_PATH}")
elseif(EXISTS "${CANN_3RD_LIB_PATH}/catlass")
  get_filename_component(CATLASS_SOURCE_PATH
                         ${CANN_3RD_LIB_PATH}/catlass REALPATH)
  message(STATUS "Find catlass source dir: ${CATLASS_SOURCE_PATH}")
else()
  if(EXISTS "${PROJECT_SOURCE_DIR}/build/_deps/catlass-subbuild")
    file(REMOVE_RECURSE ${PROJECT_SOURCE_DIR}/build/_deps/catlass-subbuild)
  endif()
  include(FetchContent)

  FetchContent_Declare(
    catlass
    GIT_REPOSITORY https://gitcode.com/cann/catlass.git
    GIT_TAG ${CATLASS_TAG_ID}
    GIT_PROGRESS TRUE
    GIT_SUBMODULES ""
    SOURCE_DIR ${CANN_3RD_LIB_PATH}/catlass)

  FetchContent_Populate(catlass)

  set(CATLASS_SOURCE_PATH ${CANN_3RD_LIB_PATH}/catlass)
endif()

set(CATLASS ${CATLASS_SOURCE_PATH})
set(CATLASS_DIR ${CATLASS_SOURCE_PATH})
