# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under CANN Open Software License Agreement Version 2.0.
# ----------------------------------------------------------------------------

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE
        scaled_cosine_attention_score_def.cpp
    )
endif()

add_ops_compile_options(
    OP_NAME ScaledCosineAttentionScore
    OPTIONS --cce-auto-sync=off
            -Wno-deprecated-declarations
            -Werror
)

add_modules_sources(
    HOSTNAME ${OPHOST_NAME}
    MODE PRIVATE
    DIR ${CMAKE_CURRENT_SOURCE_DIR}
    OPTYPE scaled_cosine_attention_score
    ACLNNTYPE aclnn)
