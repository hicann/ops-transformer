# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import os
import re
import logging

black_list = [
    "moe_gather_v2",
    "moe_inplace_index_add",
    "moe_inplace_index_add_with_sorted",
    "moe_masked_scatter",
]
op_level_list = [
    "moe_token_permute_with_routing_map",
    "moe_token_permute_with_routing_map_grad",
    "moe_token_unpermute_with_routing_map",
]

op_level_list_moe_distribute = [
    "moe_distribute_combine_v2",
    "moe_distribute_combine_v3",
    "moe_distribute_dispatch_v2",
    "moe_distribute_dispatch_v3",
]

op_level_list_barrier = ["distribute_barrier", "distribute_barrier_extend"]

# obj分组字典,需要分组的算子在字典中对应soc xxxx: obj分组数
group_op_dict = {
    "ascend910b": {
        "fused_infer_attention_score": 10,
        "incre_flash_attention": 5,
        "prompt_flash_attention": 5,
        "flash_attention_score": 3,
        "flash_attention_score_grad": 3,
    },
    "ascend910_93": {
        "fused_infer_attention_score": 10,
        "incre_flash_attention": 5,
        "prompt_flash_attention": 5,
        "flash_attention_score": 3,
        "flash_attention_score_grad": 3,
    },
    "ascend950": {
        "fused_infer_attention_score": 20,
        "flash_attention_score": 5,
        "flash_attention_score_grad": 5,
        "mega_moe": 20,
    },
    "ascend350": {
        "flash_attention_score_grad": 5,
    },
}


def get_sh_files(gen_dir):
    """获取目录中所有 .sh 文件名（不包含路径）"""
    sh_files = []
    for item in os.listdir(gen_dir):
        item_path = os.path.join(gen_dir, item)
        if os.path.isfile(item_path) and item.lower().endswith(".sh"):
            sh_files.append(item)
    return sh_files


def parse_opname_from_filename(filename):
    """
    从文件名解析 op_name。
    要求格式：xxx-<opname>-<digits>.sh
    成功返回 op_name，失败返回 None
    """
    parts = filename.split("-")
    if len(parts) < 3:
        return None

    return parts[1]


def count_opnames(sh_filenames):
    """统计每个 op_name 出现的次数"""
    opname_to_count = {}
    for filename in sh_filenames:
        op_name = parse_opname_from_filename(filename)
        if op_name is not None:
            opname_to_count[op_name] = opname_to_count.get(op_name, 0) + 1
    opname_to_count_sorted = dict(sorted(opname_to_count.items()))
    return opname_to_count_sorted


def grouped(gen_path, soc, group_size):
    result: list[list[str]] = [[] for _ in range(group_size)]
    if not os.path.isdir(gen_path):
        return result
    sh_files = get_sh_files(gen_path)
    op_counts = count_opnames(sh_files)

    all_rows = []
    added_op_levels = set()
    special_task = ""
    special_task_moe_distribute = ""
    special_task_barrier = ""
    for op_name, count in op_counts.items():
        op_name_real = op_name
        if op_name.endswith(("_a2", "_a3", "_apt")):
            op_name_real = op_name.rsplit("_", 1)[0]
        if op_name_real in black_list:
            continue
        for i in range(count):
            if op_name_real in op_level_list:
                if op_name_real in added_op_levels:
                    continue
                else:
                    added_op_levels.add(op_name_real)
                    special_task = special_task + str(op_name_real) + ","
            elif op_name_real in op_level_list_moe_distribute:
                if op_name_real in added_op_levels:
                    continue
                else:
                    added_op_levels.add(op_name_real)
                    special_task_moe_distribute = (
                        special_task_moe_distribute + str(op_name_real) + ","
                    )
            elif op_name_real in op_level_list_barrier:
                if op_name_real in added_op_levels:
                    continue
                else:
                    added_op_levels.add(op_name_real)
                    special_task_barrier = (
                        special_task_barrier + str(op_name_real) + ","
                    )
            else:
                row_string = f"{op_name_real},{count}-{i}"
                all_rows.append(row_string)
    if len(special_task) != 0:
        special_task = special_task[:-1]
        all_rows.append(special_task)

    if len(special_task_moe_distribute) != 0:
        special_task_moe_distribute = special_task_moe_distribute[:-1]
        all_rows.append(special_task_moe_distribute)

    if len(special_task_barrier) != 0:
        special_task_barrier = special_task_barrier[:-1]
        all_rows.append(special_task_barrier)

    for idx, row in enumerate(all_rows):
        result[idx % group_size].append(row)

    return result


def grouped_back(gen_path, soc, group_size):
    result: list[list[str]] = [[] for _ in range(group_size)]
    if not os.path.isdir(gen_path):
        return result
    sh_files = get_sh_files(gen_path)
    op_counts = count_opnames(sh_files)

    added_op_levels = set()
    special_task_parts = []
    special_task_parts_moe_distribute = []
    special_task_parts_barrier = []
    current_group_index = 0

    for op_name, count in op_counts.items():
        op_name_real = op_name
        if op_name.endswith(("_a2", "_a3", "_apt")):
            op_name_real = op_name.rsplit("_", 1)[0]

        if op_name_real in black_list:
            continue
        if op_name_real in op_level_list:
            if op_name_real not in added_op_levels:
                added_op_levels.add(op_name_real)
                special_task_parts.append(str(op_name_real))
            continue
        if op_name_real in op_level_list_moe_distribute:
            if op_name_real not in added_op_levels:
                added_op_levels.add(op_name_real)
                special_task_parts_moe_distribute.append(str(op_name_real))
            continue
        if op_name_real in op_level_list_barrier:
            if op_name_real not in added_op_levels:
                added_op_levels.add(op_name_real)
                op_level_list_barrier.append(str(op_name_real))
            continue
        if count >= group_size:
            for i in range(group_size):
                row_string = f"{op_name_real},{group_size}-{i}"
                result[current_group_index].append(row_string)
                current_group_index = (current_group_index + 1) % group_size
        else:
            for i in range(count):
                row_string = f"{op_name_real},{count}-{i}"
                result[current_group_index].append(row_string)
                current_group_index = (current_group_index + 1) % group_size

    if special_task_parts:
        special_task = ",".join(special_task_parts)
        result[current_group_index].append(special_task)

    if special_task_parts_moe_distribute:
        special_task_moe_distribute = ",".join(special_task_parts_moe_distribute)
        result[current_group_index].append(special_task_moe_distribute)

    if special_task_parts_barrier:
        special_task_barrier = ",".join(special_task_parts_barrier)
        result[current_group_index].append(special_task_barrier)

    return result


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def should_skip_directory(dir_name):
    """
    判断是否应该跳过该目录
    """
    skip_dirs = {
        "build",
        "cmake",
        "common",
        "docs",
        "examples",
        "experimental",
        "scripts",
        "tests",
        "third_party",
        "3rdparty",
        "torch_extension",
        "3rd",
    }
    return dir_name in skip_dirs


def parse_foreach_config(config_str):
    """
    解析 FOREACH_OPDEF 中的配置字符串
    """
    config_mapping = {
        "A2": "ascend910b",
        "910_93": "ascend910_93",
        "A5": "ascend950",
        "910B": "ascend910b",
        "910B_93": "ascend910_93",
        "910B_95": "ascend950",
        "950": "ascend950",
        "910": "ascend910",
        "910_55": "ascend910_55",
    }

    found_configs = []
    config_str_upper = config_str.upper()

    priority_checks = [
        ("A2", "ascend910b"),
        ("910_93", "ascend910_93"),
        ("A5", "ascend950"),
        ("910_55", "ascend910_55"),
        ("910B", "ascend910b"),
        ("910B_93", "ascend910_93"),
        ("910B_95", "ascend950"),
        ("950", "ascend950"),
        ("910", "ascend910"),
    ]

    for key, value in priority_checks:
        if key in config_str_upper and value not in found_configs:
            found_configs.append(value)

    return found_configs


def extract_static_map_configs(content):
    """
    从静态map中提取配置名称
    """
    configs = []

    map_patterns = [
        r"static\s+const\s+std::map<std::string[^>]*>\s+\w+\s*=\s*\{([^}]+)\}",
        r'\{"([a-zA-Z0-9_]+)"[^}]*\}',
    ]

    for pattern in map_patterns:
        matches = re.findall(pattern, content, re.DOTALL)
        for match in matches:
            config_matches = re.findall(r'"([a-zA-Z0-9_]+)"', match)
            configs.extend(config_matches)  # 直接追加，不判断重复
    return list(set(configs))  # 最终一次去重


def extract_set_ascend_config_calls(content):
    """
    提取 SetAscendConfig 调用中的配置名称
    """
    configs = []

    pattern1 = r'SetAscendConfig\([^,]+,\s*"([^"]+)"\)'
    pattern2 = r'SetAscendConfig\([^,]+,\s*"([^"]+)",\s*"([^"]+)"\)'

    matches1 = re.findall(pattern1, content)
    for match in matches1:
        if match not in configs:
            configs.append(match)

    matches2 = re.findall(pattern2, content)
    for match in matches2:
        version, dst_version = match
        if version not in configs:
            configs.append(version)
        if dst_version not in configs:
            configs.append(dst_version)

    return list(set(configs))


def extract_foreach_opdef_configs(content):
    """
    提取 FOREACH_OPDEF 相关格式的配置
    """
    configs = []

    pattern1 = r"FOREACH_OPDEF\(([^,]+),"
    matches1 = re.findall(pattern1, content)
    for match in matches1:
        config_str = match.strip()
        configs.extend(parse_foreach_config(config_str))

    pattern2 = r"FOREACH_OPDEF_END_([^(]+)\("
    matches2 = re.findall(pattern2, content)
    for match in matches2:
        config_str = match.strip()
        configs.extend(parse_foreach_config(config_str))

    return list(set(configs))


def extract_traditional_aicore_configs(content):
    """
    提取传统的 AICore 配置名称
    """
    configs = []

    traditional_patterns = [
        r'this->AICore\(\)\.AddConfig\("([a-zA-Z0-9_]+)"',
        r'\.AddConfig\("([a-zA-Z0-9_]+)"',
        r'AddConfig\("([a-zA-Z0-9_]+)"',
    ]

    for pattern in traditional_patterns:
        matches = re.findall(pattern, content)
        for match in matches:
            if match not in configs:
                configs.append(match)

    return configs


def extract_ai_core_configs(file_path):
    """
    从 _def.cpp 文件中提取 AICore 配置名称
    """
    configs = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 方法1：匹配传统的 AICore 配置
        traditional_configs = extract_traditional_aicore_configs(content)
        if traditional_configs:
            configs.extend(traditional_configs)

        # 方法2：匹配 FOREACH_OPDEF 相关格式
        foreach_configs = extract_foreach_opdef_configs(content)
        if foreach_configs:
            configs.extend(foreach_configs)

        # 方法3：匹配静态map + SetAscendConfig 格式
        static_map_configs = extract_static_map_configs(content)
        set_ascend_configs = extract_set_ascend_config_calls(content)

        all_other_configs = list(set(static_map_configs + set_ascend_configs))
        if all_other_configs:
            configs.extend(all_other_configs)

        # 去重并返回
        return list(set(configs))

    except Exception as e:
        logger.error(f"读取文件 {file_path} 时出错: {e}")
        return []


def grouped_def(repository_path, soc, group_size):
    result: list[list[str]] = [[] for _ in range(group_size)]
    init_oplist = []
    op_list = []

    for root, dirs, files in os.walk(repository_path):
        # 过滤掉不需要的目录
        dirs[:] = [d for d in dirs if not should_skip_directory(d)]

        for file in files:
            if file.endswith("_def.cpp"):
                file_path = os.path.join(root, file)
                op_name = file.replace("_def.cpp", "")

                # 提取 AICore 配置
                ai_core_configs = extract_ai_core_configs(file_path)

                if soc in ai_core_configs:
                    init_oplist.append(op_name)
    init_oplist.sort()
    special_task = ""
    special_task_moe_distribute = ""
    special_task_barrier = ""
    for op_name in init_oplist:
        if op_name in black_list:
            continue
        if op_name in op_level_list:
            special_task = special_task + str(op_name) + ","
            continue
        if op_name in op_level_list_moe_distribute:
            special_task_moe_distribute = (
                special_task_moe_distribute + str(op_name) + ","
            )
            continue
        if op_name in op_level_list_barrier:
            special_task_barrier = special_task_barrier + str(op_name) + ","
            continue
        if op_name in group_op_dict[soc]:
            op_group_size = group_op_dict[soc][op_name]
            if group_size > 1 and op_group_size > group_size:
                op_group_size = group_size
            for i in range(op_group_size):
                row_string = f"{op_name},{op_group_size}-{i}"
                op_list.append(row_string)
        else:
            op_list.append(op_name)

    if len(special_task) != 0:
        special_task = special_task[:-1]
        op_list.append(special_task)

    if len(special_task_moe_distribute) != 0:
        special_task_moe_distribute = special_task_moe_distribute[:-1]
        op_list.append(special_task_moe_distribute)

    if len(special_task_barrier) != 0:
        special_task_barrier = special_task_barrier[:-1]
        op_list.append(special_task_barrier)

    op_list.sort()
    for idx, row in enumerate(op_list):
        result[idx % group_size].append(row)

    return result


def main(repository_path, soc, group_size=1):
    project_path = os.path.abspath(repository_path)
    gen_path = os.path.abspath(
        os.path.join(project_path, "build", "binary", soc, "gen")
    )
    if group_size > 1:
        op_data = grouped_back(gen_path, soc, group_size)
    else:
        op_data = grouped_def(project_path, soc, group_size)
    return op_data


if __name__ == "__main__":
    file_path = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    temp = main(file_path, "ascend910b", 1)
    size = sum(len(sublist) for sublist in temp)
    logger.info(f"Group dicts: {temp}")
    logger.info(f"Total task size: {size}")
