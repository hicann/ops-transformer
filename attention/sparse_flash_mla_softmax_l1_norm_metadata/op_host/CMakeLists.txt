# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
add_op_to_compiled_list()

add_modules_sources(OPTYPE sparse_flash_mla_softmax_l1_norm_metadata ACLNNTYPE aclnn)

# This operator intentionally keeps a hand-written public API header because
# its argument list contains the metadata-specific optional tensors and scalar
# attributes.  There is no *_def.cpp for opbuild to regenerate that header,
# so install it explicitly with the generated aclnn headers.
install(FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/op_api/aclnn_sparse_flash_mla_softmax_l1_norm_metadata.h
    DESTINATION ${ACLNN_INC_INSTALL_DIR}
    OPTIONAL
)
