#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set -eo pipefail
li_tools_dir=$(cd "$(dirname "$0")" && pwd)
li_build_dir=${1:?Usage: bash tools/build_standalone.sh BUILD_DIR [ascend910b|ascend910_93]}
li_soc=${2:-ascend910b}
source "${ASCEND_HOME_PATH:-/usr/local/Ascend/cann}/set_env.sh"
mkdir -p "$li_build_dir"
li_build_dir=$(cd "$li_build_dir" && pwd)
cmake -S "$li_tools_dir/standalone" -B "$li_build_dir" \
  -DCMAKE_BUILD_TYPE=Release -DASCEND_COMPUTE_UNIT="$li_soc"
cmake --build "$li_build_dir" --target all binary -j8
cmake --install "$li_build_dir" --prefix "$li_build_dir/stage"
cmake --build "$li_build_dir" --target package -j8
