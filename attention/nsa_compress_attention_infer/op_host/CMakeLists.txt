# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE
        nsa_compress_attention_infer_def.cpp
    )
    add_ops_compile_options(
            OP_NAME NsaCompressAttentionInfer
            OPTIONS --cce-auto-sync=off
                    -Wno-deprecated-declarations
                    -Werror
                    -fpermissive
    )
endif()

if (NOT BUILD_OPS_RTY_KERNEL) 
    add_modules_sources(OPTYPE nsa_compress_attention_infer ACLNNTYPE aclnn_exclude)
endif()

