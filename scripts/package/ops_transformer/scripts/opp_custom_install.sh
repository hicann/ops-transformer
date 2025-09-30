#!/bin/bash
# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

curpath=$(dirname $(readlink -f "$0"))
SCENE_FILE="${curpath}""/../scene.info"
OPP_COMMON="${curpath}""/opp_common.sh"
common_func_path="${curpath}/common_func.inc"
. "${OPP_COMMON}"
. "${common_func_path}"
# init arch 
architecture=$(uname -m)
architectureDir="${architecture}-linux"

while true; do
    case "$1" in
    --install-path=*)
        install_path=$(echo "$1" | cut -d"=" -f2-)
        shift
        ;;
    --version-dir=*)
        version_dir=$(echo "$1" | cut -d"=" -f2)
        shift
        ;;
    --latest-dir=*)
        latest_dir=$(echo "$1" | cut -d"=" -f2)
        shift
        ;;
    -*)
        shift
        ;;
    *)
        break
        ;;
    esac
done
get_version_dir "opp_kernel_version_dir" "$install_path/$version_dir/opp_kernel/version.info"

if [ -z "$opp_kernel_version_dir" ]; then
    # create op_api soft link
    logandprint "[INFO]: Start create opapi softlinks."
    createOpapiSoftlink "${install_path}/${version_dir}"
    return_code=$?
    if [ ${return_code} -eq 0 ]; then
        logandprint "[INFO]: Create opapi softlinks successfully!"
    elif [ ${return_code} -eq 3 ]; then
        logandprint "[WARNING]: opapi source file does not exist!"
    else
        logandprint "[ERROR]: Create opapi softlinks failed!"
    fi
fi
