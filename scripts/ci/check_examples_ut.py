#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""检测算子的 examples 与 ut 是否缺失或为空。

算子判定：目录下存在 op_host 或 op_kernel 子目录，op_host 下存在def文件，
且目录名不属于公共库（common、*_common 等）；无def文件为接口类算子，不参与统计。
example 判定：examples 目录不存在，目录内（含子目录）无任何源码文件，
或仅有 arch35 一个子目录且def文件声明支持ascend910b/ascend910_93（纯950等非910芯片算子不统计）。
ut 判定：tests/ut 目录不存在，或目录内除 CMakeLists.txt 外无任何文件。
检测结果按类别汇总写入 build/check_examples_ut.csv，
列为: 算子路径, 算子名, OPTYPE(def文件中OP_ADD声明的值), 类别。
指定pr_filelist时（-p参数或首个位置参数为文件），仅检查pr涉及的算子，
并基于build/tmp/ops_config.txt的依赖图（同build.sh的parse_op_dependencies）做正向+反向依赖展开。
"""

import argparse
import csv
import glob
import os
import re
import sys
from functools import lru_cache
from typing import Iterator, List, NamedTuple, Optional, Set, Tuple

# 非算子类目的顶级目录
NON_CATEGORY_DIRS = {
    ".git",
    "build",
    "build_out",
    "cmake",
    "common",
    "docs",
    "examples",
    "oat_reports",
    "scripts",
    "st",
    "tests",
    "third_party",
    "torch_extension",
}

# 公共库目录名（不算子）
COMMON_LIB_NAMES = {"common", "sort_lib", "index_check", "foreach_utils", "pool_utils"}

# example 内视为有效内容的文件后缀
EXAMPLE_EXTS = (".cpp", ".cc", ".py")

# 问题类别
ISSUE_EXAMPLES_MISSING = "examples缺失/为空"
ISSUE_ARCH35_ONLY = "examples仅含arch35目录"
ISSUE_UT_MISSING = "ut缺失/为空"


class OpIssue(NamedTuple):
    """算子缺失项。"""

    op_path: str
    missing_examples: bool
    arch35_only_examples: bool
    missing_ut: bool


def is_common_lib(name: str) -> bool:
    """判断是否为公共库目录。"""
    return (
        name in COMMON_LIB_NAMES
        or name.endswith("_common")
        or name.endswith("_utils")
        or name.endswith("_util")
    )


def issue_categories(issue: OpIssue) -> List[str]:
    """返回算子命中的问题类别列表。"""
    categories = []
    if issue.missing_examples:
        categories.append(ISSUE_EXAMPLES_MISSING)
    if issue.arch35_only_examples:
        categories.append(ISSUE_ARCH35_ONLY)
    if issue.missing_ut:
        categories.append(ISSUE_UT_MISSING)
    return categories


def has_file_with_ext(root: str, exts: Tuple[str, ...]) -> bool:
    """判断目录内（递归）是否存在指定后缀的文件。"""
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if filename.endswith(exts):
                return True
    return False


def has_ut_content(root: str) -> bool:
    """判断 ut 目录内除 CMakeLists.txt 外是否有其他文件。"""
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if filename != "CMakeLists.txt":
                return True
    return False


def is_arch35_only_examples(examples_dir: str) -> bool:
    """判断 examples 目录下是否仅有 arch35 一个子目录（无顶层示例文件及其他子目录，CMakeLists.txt除外）。"""
    entries = [entry for entry in os.listdir(examples_dir) if entry != "CMakeLists.txt"]
    return (
        len(entries) == 1
        and entries[0] == "arch35"
        and os.path.isdir(os.path.join(examples_dir, "arch35"))
    )


def iter_category_dirs(repo_root: str) -> Iterator[str]:
    """遍历所有算子类目目录（含 experimental 下的二级类目）。"""
    for entry in sorted(os.listdir(repo_root)):
        path = os.path.join(repo_root, entry)
        if (
            entry in NON_CATEGORY_DIRS
            or entry.startswith(".")
            or not os.path.isdir(path)
        ):
            continue
        if entry == "experimental":
            for sub in sorted(os.listdir(path)):
                sub_path = os.path.join(path, sub)
                if os.path.isdir(sub_path):
                    yield sub_path
        else:
            yield path


def has_def_file(op_path: str) -> bool:
    """判断算子op_host下是否存在def文件，无def文件为接口类算子，不参与统计。"""
    op_host_dir = os.path.join(op_path, "op_host")
    if not os.path.isdir(op_host_dir):
        return False
    return bool(
        glob.glob(os.path.join(op_host_dir, f"{os.path.basename(op_path)}_def*.cpp"))
        or glob.glob(os.path.join(op_host_dir, "*_def*.cpp"))
    )


def iter_op_dirs(repo_root: str) -> Iterator[str]:
    """遍历仓库内所有算子目录（不含无def文件的接口类算子）。"""
    for category in iter_category_dirs(repo_root):
        for entry in sorted(os.listdir(category)):
            op_path = os.path.join(category, entry)
            if not os.path.isdir(op_path) or is_common_lib(entry):
                continue
            if not has_def_file(op_path):
                continue
            if os.path.isdir(os.path.join(op_path, "op_host")) or os.path.isdir(
                os.path.join(op_path, "op_kernel")
            ):
                yield op_path


def check_op(op_path: str) -> Optional[OpIssue]:
    """检查单个算子，无问题返回 None。"""
    missing_examples = True
    arch35_only_examples = False
    examples_dir = os.path.join(op_path, "examples")
    if os.path.isdir(examples_dir):
        missing_examples = not has_file_with_ext(examples_dir, EXAMPLE_EXTS)
        # def文件仅声明950等非910芯片的算子，仅有arch35 examples属正常，不视为缺失
        arch35_only_examples = (
            not missing_examples
            and is_arch35_only_examples(examples_dir)
            and def_supports_910(op_path)
        )

    missing_ut = True
    ut_dir = os.path.join(op_path, "tests", "ut")
    if os.path.isdir(ut_dir):
        missing_ut = not has_ut_content(ut_dir)

    if missing_examples or arch35_only_examples or missing_ut:
        return OpIssue(op_path, missing_examples, arch35_only_examples, missing_ut)
    return None


def check_all(
    repo_root: str, op_dirs: Optional[List[str]] = None
) -> Tuple[List[OpIssue], int]:
    """检查算子，op_dirs为None时全仓扫描，否则仅检查指定算子目录，返回问题列表与算子总数。"""
    issues: List[OpIssue] = []
    total = 0
    op_paths = op_dirs if op_dirs is not None else iter_op_dirs(repo_root)
    for op_path in op_paths:
        total += 1
        issue = check_op(op_path)
        if issue is not None:
            issues.append(issue)
    return issues, total


def get_op_categories(repo_root: str) -> List[str]:
    """从cmake/variables.cmake的OP_CATEGORY_LIST读取算子类目列表。"""
    variables_cmake = os.path.join(repo_root, "cmake", "variables.cmake")
    if os.path.isfile(variables_cmake):
        with open(variables_cmake, encoding="utf-8") as file:
            for line in file:
                match = re.search(r"set\(OP_CATEGORY_LIST\s+(.*)\)", line)
                if match:
                    return match.group(1).replace('"', "").split()
    # ops-transformer keeps its category list in the CI parser instead of
    # cmake/variables.cmake. Keep this list aligned with that parser.
    return ["attention", "ffn", "gmm", "mamba", "mc2", "mhc", "moe", "posembedding"]


def parse_ops_from_filelist(
    repo_root: str, pr_filelist: str
) -> Tuple[Set[str], Set[str]]:
    """解析pr_filelist，返回(涉及的算子目录, 涉及的算子名)。

    算子名取二级目录名，类目公共目录记为'<类目>.common'（与ops_config.txt中的键一致），
    其他公共库（*_common、*_utils等）不在依赖图中，跳过。
    """
    categories = set(get_op_categories(repo_root))
    involved_dirs: Set[str] = set()
    involved_names: Set[str] = set()
    with open(pr_filelist, encoding="utf-8") as file:
        for line in file:
            file_path = re.sub(r"^[MADRC]\t", "", line.strip())
            if not file_path or file_path.startswith("#"):
                continue
            if os.path.isabs(file_path):
                file_path = os.path.relpath(file_path, repo_root)
            parts = file_path.split("/")
            if len(parts) < 2:
                continue
            category, second = parts[0], parts[1]
            prefix = ""
            if category in categories:
                prefix = category
            elif (
                category == "experimental"
                and len(parts) >= 3
                and parts[1] in categories
            ):
                category, second, prefix = (
                    parts[1],
                    parts[2],
                    f"experimental/{parts[1]}",
                )
            if not prefix:
                continue
            op_dir = os.path.join(repo_root, prefix, second)
            if not os.path.isdir(op_dir):
                continue
            if second == "common":
                involved_names.add(f"{category}.common")
            elif is_common_lib(second):
                continue
            else:
                involved_dirs.add(op_dir)
                involved_names.add(second)
    return involved_dirs, involved_names


def expand_involved_ops(
    repo_root: str, involved_dirs: Set[str], involved_names: Set[str]
) -> List[str]:
    """展开涉及算子；无仓库依赖解析器时退化为直接涉及算子。"""
    involved = {path for path in involved_dirs if has_def_file(path)}
    if not involved_names:
        return sorted(involved)
    sys.path.insert(
        0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "util")
    )
    try:
        from dependency_parser import OpDependenciesParser

        parser = OpDependenciesParser(os.path.join(repo_root, "build"))
    except (ImportError, ModuleNotFoundError, RuntimeError, OSError) as err:
        print(
            f"提示: 未启用依赖展开({err})，仅检查pr_filelist直接涉及的算子",
            file=sys.stderr,
        )
        return sorted(involved)
    known_ops = set(parser.all_ops)
    known_names = [name for name in involved_names if name in known_ops]
    for name in involved_names - set(known_names):
        print(
            f"警告: {name} 不在ops_config.txt依赖图中（如PR新增算子），仅直接检查",
            file=sys.stderr,
        )
    forward, reverse = (
        parser.get_dependencies_by_ops(known_names) if known_names else ([], [])
    )
    expanded_names = {
        name for name in set(forward) | set(reverse) if not name.endswith(".common")
    }
    dirs_by_name: dict = {}
    for op_dir in iter_op_dirs(repo_root):
        dirs_by_name.setdefault(os.path.basename(op_dir), []).append(op_dir)
    for name in expanded_names:
        involved.update(dirs_by_name.get(name, []))
    return sorted(involved)


def print_report(issues: List[OpIssue], total: int, repo_root: str) -> None:
    """打印检测报告。"""
    no_examples = [i for i in issues if i.missing_examples]
    arch35_only = [
        i for i in issues if i.arch35_only_examples and not i.missing_examples
    ]
    no_ut = [i for i in issues if i.missing_ut]

    print(f"算子总数: {total}, 正常: {total - len(issues)}, 异常: {len(issues)}")
    print(f"  examples 缺失/为空: {len(no_examples)}")
    print(f"  examples 仅含arch35目录: {len(arch35_only)}")
    print(f"  ut 缺失/为空: {len(no_ut)}")
    print("")

    for title, items in (
        ("缺 examples:", no_examples),
        ("examples 仅含arch35目录:", arch35_only),
        ("缺 ut:", no_ut),
    ):
        print(title)
        for issue in items:
            print(f"  {os.path.relpath(issue.op_path, repo_root)}")
        print("")


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="检测算子 examples 与 ut 是否缺失或为空。"
    )
    parser.add_argument(
        "repo_root",
        nargs="?",
        default=None,
        help="仓库根目录，默认为脚本所在仓库根目录；若传入的是文件则视为pr_filelist",
    )
    parser.add_argument(
        "-p",
        "--pr-filelist",
        default=None,
        dest="pr_filelist",
        help="PR变更文件列表（每行一个路径），指定后仅检查涉及的算子（含依赖展开）",
    )
    return parser.parse_args()


def find_def_files(op_path: str) -> List[str]:
    """查找算子op_host下的def文件，优先<算子名>_def*.cpp，无则退化为任意*_def*.cpp。"""
    op_host_dir = os.path.join(op_path, "op_host")
    if not os.path.isdir(op_host_dir):
        return []
    def_files = sorted(
        glob.glob(os.path.join(op_host_dir, f"{os.path.basename(op_path)}_def*.cpp"))
    )
    if not def_files:
        def_files = sorted(glob.glob(os.path.join(op_host_dir, "*_def*.cpp")))
    return def_files


# 910系列芯片标识，def文件中声明支持这些芯片的算子，仅有arch35 examples时才视为缺失
SOC_910_PATTERNS = ('AddConfig("ascend910b"', 'AddConfig("ascend910_93"')


@lru_cache(maxsize=None)
def def_supports_910(op_path: str) -> bool:
    """判断算子def文件是否声明支持ascend910b或ascend910_93芯片。"""
    for def_file in find_def_files(op_path):
        try:
            with open(def_file, encoding="utf-8", errors="ignore") as file:
                content = file.read()
        except OSError:
            continue
        if any(pattern in content for pattern in SOC_910_PATTERNS):
            return True
    return False


@lru_cache(maxsize=None)
def get_op_type(op_path: str) -> str:
    """从算子def文件解析OP_ADD声明的OpType（大写驼峰），与func.cmake中get_op_type_from_def_files规则一致，解析失败返回空串。"""
    for def_file in find_def_files(op_path):
        try:
            with open(def_file, encoding="utf-8", errors="ignore") as file:
                content = file.read()
        except OSError:
            continue
        match = re.search(r"OP_ADD\(\s*([A-Za-z0-9_]+)", content)
        if match:
            return match.group(1)
    return ""


def write_csv(issues: List[OpIssue], csv_path: str, repo_root: str) -> int:
    """将检测到的问题按类别汇总写入csv文件，每行一条问题记录，返回行数。

    列格式: 算子(相对仓库根目录路径), 算子名, OPTYPE(def文件中OP_ADD声明的值), 类别。
    """
    rows = []
    for category in (ISSUE_EXAMPLES_MISSING, ISSUE_ARCH35_ONLY, ISSUE_UT_MISSING):
        for issue in issues:
            if category in issue_categories(issue):
                rows.append(
                    (
                        os.path.relpath(issue.op_path, repo_root),
                        os.path.basename(issue.op_path),
                        get_op_type(issue.op_path),
                        category,
                    )
                )
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["算子", "算子名", "OPTYPE", "类别"])
        writer.writerows(rows)
    return len(rows)


def main() -> int:
    """主函数。"""
    args = parse_args()
    repo_root, pr_filelist = args.repo_root, args.pr_filelist
    if repo_root and os.path.isfile(repo_root) and pr_filelist is None:
        repo_root, pr_filelist = None, repo_root
    if repo_root is None:
        repo_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    if not os.path.isdir(repo_root):
        print(f"仓库根目录不存在: {repo_root}", file=sys.stderr)
        return 2

    op_dirs = None
    if pr_filelist:
        if not os.path.isfile(pr_filelist):
            print(f"pr_filelist不存在: {pr_filelist}", file=sys.stderr)
            return 2
        involved_dirs, involved_names = parse_ops_from_filelist(repo_root, pr_filelist)
        if not involved_dirs and not involved_names:
            print(f"pr_filelist未解析到算子: {pr_filelist}")
            return 0
        op_dirs = expand_involved_ops(repo_root, involved_dirs, involved_names)
        print(f"pr_filelist涉及算子(依赖展开可用时包含依赖): {len(op_dirs)}个")

    issues, total = check_all(repo_root, op_dirs)
    print_report(issues, total, repo_root)
    build_dir = os.path.join(repo_root, "build")
    os.makedirs(build_dir, exist_ok=True)
    csv_path = os.path.join(build_dir, "check_examples_ut.csv")
    count = write_csv(issues, csv_path, repo_root)
    print(f"检测结果已写入: {csv_path} ({count}条)")
    if issues:
        # 供CI流水线消费的机器可读输出，与.gitcode/scripts/compile.sh的Check_Example约定一致
        op_names = ",".join(
            sorted({os.path.basename(issue.op_path) for issue in issues})
        )
        print(f"check_example_failed={op_names}")
    return 1 if issues else 0


if __name__ == "__main__":
    sys.exit(main())
