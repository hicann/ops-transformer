
#!/bin/bash
# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
# This script only compiles the AscendC operator and creates a shared library file in lib directory
# `bisheng` CLI: https://www.hiascend.com/document/detail/zh/cannCommunity/800/developmentguide/opdevg/BishengCompiler/atlas_bisheng_10_0003.html
# Adapted from: https://gitee.com/ascend/mstt/tree/master/sample/pytorch_adapter

# validate cann environment
source ../../scripts/num_cores_map.sh
source ../../scripts/check_cann.sh
if ! check_cann_environment false; then
    exit 1
fi

# to place the .so library there
mkdir -p lib

rm -f lib/libquest_prefill_metadata.so
bisheng -fPIC -shared -xcce -O2 -std=c++17 \
    --cce-soc-version=$SOC_VERSION --cce-soc-core-type=VecCore \
    -I${ASCEND_TOOLKIT_HOME}/compiler/tikcpp/tikcfw \
    -I${ASCEND_TOOLKIT_HOME}/compiler/tikcpp/tikcfw/impl \
    -I${ASCEND_TOOLKIT_HOME}/compiler/tikcpp/tikcfw/interface \
    -I${ASCEND_TOOLKIT_HOME}/include \
    -o lib/libquest_prefill_metadata.so quest_prefill_metadata.cpp

rm -f lib/libquest_block_select_paged.so
bisheng -fPIC -shared -xcce -O2 -std=c++17 \
    --cce-soc-version=$SOC_VERSION --cce-soc-core-type=VecCore \
    -I${ASCEND_TOOLKIT_HOME}/compiler/tikcpp/tikcfw \
    -I${ASCEND_TOOLKIT_HOME}/compiler/tikcpp/tikcfw/impl \
    -I${ASCEND_TOOLKIT_HOME}/compiler/tikcpp/tikcfw/interface \
    -I${ASCEND_TOOLKIT_HOME}/include \
    -o lib/libquest_block_select_paged.so quest_block_select_paged.cpp
