# ----------------------------------------------------------------------------
 # Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 # This file is a part of the CANN Open Software.
 # Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 # Please refer to the License for details. You may not use this file except in compliance with the License.
 # THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 # INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 # See LICENSE in the root of the software repository for the full text of the License.
 # ----------------------------------------------------------------------------
 if(BUILD_OPEN_PROJECT)
   file(GLOB OPTILING_SRCS src/*.cpp)
   if(OPTILING_SRCS)
     add_tiling_modules()
     target_sources(${OPHOST_NAME}_tiling_obj PRIVATE ${OPTILING_SRCS})
   endif()
   if(NOT ENABLE_BUILT_IN)
    file(GLOB OPPROTO_SRCS
     src/mc2_moe_utils.cpp
     src/mc2_common_infershape.cpp
     src/mc2_hcom_topo_info.cpp
     )
     if(OPPROTO_SRCS)
       add_infer_modules()
       target_sources(${OPHOST_NAME}_infer_obj PRIVATE ${OPPROTO_SRCS})
     endif()
     file(GLOB OPAPI_SRC
       src/mc2_hcom_topo_info.cpp
       src/mc2_common_infershape.cpp
     )
     if(OPAPI_SRC)
       add_opapi_modules()
       target_sources(${OPHOST_NAME}_opapi_obj PRIVATE ${OPAPI_SRC})
    endif()
   endif()
 endif()