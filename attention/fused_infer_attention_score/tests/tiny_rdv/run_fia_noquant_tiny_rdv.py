#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import json
import os
import subprocess
from fused_infer_attention_test import print_log


def get_all_json_files(folder_path):
    """
    读取文件夹下所有 .json 结尾的文件，返回路径列表
    :param folder_path: 文件夹路径
    :return: 所有 .json 文件的路径 list
    """
    json_files = []

    # 遍历文件夹
    for file_name in os.listdir(folder_path):
        # 判断是否以 .json 结尾（忽略大小写：.JSON / .Json 也能识别）
        if file_name.lower().endswith(".json"):
            # 拼接完整文件路径
            full_path = os.path.join(folder_path, file_name)
            # 只添加文件，排除文件夹
            if os.path.isfile(full_path):
                json_files.append(full_path)

    return json_files


def parse_input(cmd_list, key, value):
    if key == "act_seq_lens_q" or key == "act_seq_lens_kv":
        if len(value) > 0:
            cmd_list.append(f"--{key}")
            cmd_list.extend(map(str, value))  # 自动展开列表
    elif key == "mask_shape":
        if len(value) > 0:
            cmd_list.append(f"--{key}")
            cmd_list.append(str(value))
    elif isinstance(value, bool):
        if value:
            cmd_list.append(f"--{key}")
    else:
        cmd_list.append(f"--{key}")
        cmd_list.append(str(value))
    return


if __name__ == "__main__":
    TARGET_FOLDER = "./case_json"
    json_file_list = get_all_json_files(TARGET_FOLDER)

    # json_file_list = ["./case_json/aclnnFusedInferAttentionScoreV2_PFA_default_CON_FP16_2_2_128_128_259_4.json"]  # 单跑验证时，放开这条注释，内部填写要执行用例的json路径

    for json_path in json_file_list:
        json_name = os.path.basename(json_path)
        name_without_ext, ext = os.path.splitext(json_name)

        log_base_path = "./res_log"
        if not os.path.exists(log_base_path):
            os.makedirs(log_base_path)
        log_name = "./res_log/" + name_without_ext + ".log"

        with open(json_path, "r", encoding="utf-8") as f:
            data_dict = json.load(f)

        cmd_list = ["python", "fused_infer_attention_test.py"]
        for key, value in data_dict.items():
            parse_input(cmd_list, key, value)

        print_log("start exec cmd:")
        print_log(" ".join(cmd_list))
        with open(log_name, "w") as f:
            res = subprocess.run(cmd_list, stdout=f, stderr=f)
