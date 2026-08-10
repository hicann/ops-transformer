#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
#
# soc_validator.sh - 昇腾 SoC 名称校验库
#
# 该文件是独立可被 source 的 bash 库, 提供如下接口:
#   soc_is_supported <soc>   : 判断单个 SoC 名称是否受支持
#   validate_soc_list <list> : 校验逗号/分号分隔的 SoC 列表
#   get_supported_soc_text   : 输出受支持 SoC 列表的提示文本
#
# 使用示例(build.sh 中):
#   source "${CURRENT_DIR}/scripts/util/soc_validator.sh"
#   if ! soc_is_supported "${ascend_soc}"; then
#       echo "[ERROR] ${ascend_soc} 不受支持, 请选择: $(get_supported_soc_text)"
#       exit 1
#   fi
#
# 注意:
#   - 本库被 source 时不执行任何逻辑, 也不会调用 exit(), 不会影响调用方的 shell 选项;
#   - 本库故意不设置 set -euo pipefail, 以避免改变 source 方(如 build.sh)的 shell 行为,
#     但库内实现已兼容 set -e / set -u / set -o pipefail 的调用方;
#   - 除 INVALID_SOC_ITEM 报告变量外, 所有函数均为纯函数, 无其他副作用。

# -----------------------------------------------------------------------------------------------------------
# 受支持的昇腾 SoC 名称列表
# -----------------------------------------------------------------------------------------------------------
# 默认包含受支持的 SoC(与 build.sh 的 SUPPORT_COMPUTE_UNIT_SHORT 保持一致, 含出包遍历的空包 soc)。
# 调用方(如 build.sh)可在 source 本库后按需覆盖此列表, 也可在本库被 source 之前预先定义
# SUPPORTED_SOC_LIST 以覆盖默认值。
if [[ -z "${SUPPORTED_SOC_LIST+x}" ]]; then
    SUPPORTED_SOC_LIST=("ascend910b" "ascend910_93" "ascend950" "ascend310p" "ascend310b" "ascend910" "ascend610lite" "kirinx90" "kirin9030" "mc62")
fi

# 报告变量: 当 validate_soc_list 校验失败时, 该变量被置为第一个非法 SoC 项, 供调用方输出错误信息
# shellcheck disable=SC2034 # 该变量仅在本库中被赋值, 由外部调用方(source 本库的脚本)读取, 故此处不读取属正常
INVALID_SOC_ITEM=""

# -----------------------------------------------------------------------------------------------------------
# soc_is_supported - 判断单个 SoC 名称是否为受支持的 SoC
# -----------------------------------------------------------------------------------------------------------
# 入参:   $1 - 待校验的 SoC 名称(前后空白会被裁剪, 大小写不敏感)
# 返回值: 0 - 该 SoC 受支持; 1 - 该 SoC 不受支持(含空字符串与纯空白输入)
# 说明:   先裁剪前后空白, 再转为小写, 最后与 SUPPORTED_SOC_LIST 逐项精确匹配
soc_is_supported() {
    local soc="${1:-}"
    local item=""

    # 去除前导空白
    soc="${soc#"${soc%%[![:space:]]*}"}"
    # 去除尾部空白
    soc="${soc%"${soc##*[![:space:]]}"}"
    # 转为小写, 实现大小写不敏感的匹配
    soc="${soc,,}"

    # 空字符串(含纯空白输入)直接判定为不受支持
    if [[ -z "${soc}" ]]; then
        return 1
    fi

    # 与受支持列表逐项精确匹配
    for item in "${SUPPORTED_SOC_LIST[@]}"; do
        if [[ "${item}" == "${soc}" ]]; then
            return 0
        fi
    done
    return 1
}

# -----------------------------------------------------------------------------------------------------------
# validate_soc_list - 校验逗号/分号分隔的 SoC 列表
# -----------------------------------------------------------------------------------------------------------
# 入参:   $1 - 以逗号和/或分号分隔的 SoC 列表字符串
# 返回值: 0 - 列表中所有项均受支持; 1 - 列表为空(含纯空白)或存在非法项
# 副作用: 校验失败时, 将 INVALID_SOC_ITEM 置为第一个非法项(已裁剪前后空白), 供调用方报告错误
validate_soc_list() {
    local raw="${1:-}"
    local trimmed=""
    local item=""
    local normalized=""

    # 每次调用先清空报告变量, 避免残留上一次的非法项
    INVALID_SOC_ITEM=""

    # 去除列表整体前导与尾部空白
    trimmed="${raw#"${raw%%[![:space:]]*}"}"
    trimmed="${trimmed%"${trimmed##*[![:space:]]}"}"

    # 空字符串(含纯空白输入)直接判定为非法
    if [[ -z "${trimmed}" ]]; then
        return 1
    fi

    # 将逗号与分号统一替换为换行符, 便于逐项读取校验
    normalized="${trimmed//[;,]/$'\n'}"

    # 逐项裁剪空白并校验, 遇到第一个非法项立即返回 1
    while IFS= read -r item; do
        # 去除单项前导与尾部空白
        item="${item#"${item%%[![:space:]]*}"}"
        item="${item%"${item##*[![:space:]]}"}"
        if ! soc_is_supported "${item}"; then
            # shellcheck disable=SC2034 # 报告变量由外部调用方读取, 本库内仅赋值不读取
            INVALID_SOC_ITEM="${item}"
            return 1
        fi
    done <<<"${normalized}"
    return 0
}

# -----------------------------------------------------------------------------------------------------------
# get_supported_soc_text - 输出受支持 SoC 列表的提示文本
# -----------------------------------------------------------------------------------------------------------
# 入参:   无
# 返回值: 0
# 输出:   受支持 SoC 名称以", "分隔的单行文本, 例如:
#         "ascend910b, ascend910_93, ascend950, ascend310p, kirinx90, kirin9030, mc62"
# 说明:   用于在错误信息中提示用户可选的 SoC 列表
get_supported_soc_text() {
    local item=""
    local result=""
    local first=1

    for item in "${SUPPORTED_SOC_LIST[@]}"; do
        if [[ "${first}" -eq 1 ]]; then
            result="${item}"
            first=0
        else
            result="${result}, ${item}"
        fi
    done
    echo "${result}"
    return 0
}
