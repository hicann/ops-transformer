#!/usr/bin/env python3
# coding: utf-8
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
获取修改文件应触发的soc版本.
"""

import argparse
import logging
from pathlib import Path
from typing import List
from parse_changed_files import Module
from parse_changed_files import Parser


class ParserSoc(Parser):
    """
    规则文件、修改文件列表文件解析.
    """

    _Modules: List[
        Module
    ] = []  # 保存规则文件(tests/test_soc_config.yaml)内设置的模块列表

    @staticmethod
    def main() -> str:
        # 脚本运行参数注册，包括配置文件和修改文件列表
        ps = argparse.ArgumentParser(
            description="Get socs according to changed files", epilog="End!"
        )
        ps.add_argument(
            "-c",
            "--classify",
            required=True,
            nargs=1,
            type=Path,
            help="tests/test_soc_config.yaml",
        )
        ps.add_argument(
            "-f",
            "--file",
            required=True,
            nargs=1,
            type=Path,
            help="changed files desc file.",
        )
        # 子命令行，绑定一下获取soc的函数
        sub_ps = ps.add_subparsers(help="Sub-Command")
        p_soc = sub_ps.add_parser("get_related_soc", help="Get related soc.")
        p_soc.set_defaults(func=ParserSoc.get_related_soc)
        # 处理，先获取配置文件的信息，然后是修改文件列表
        args = ps.parse_args()
        logging.debug(args)
        if not ParserSoc.parse_classify_file(file=Path(args.classify[0])):
            return ""
        if not ParserSoc.parse_changed_file(file=Path(args.file[0])):
            return ""
        ParserSoc.print_details()
        rst_soc = args.func()
        return rst_soc

    @classmethod
    def get_related_soc(cls):
        def soc_test_list_append(soc, soc_test_option_lst):
            if soc not in soc_test_option_lst:
                soc_test_option_lst.append(soc)

        soc_test_option_lst: List[str] = []
        for p in cls._ChangedPaths:
            for m in cls._Modules:
                new_options = m.get_test_options(f=p)
                for soc in new_options:
                    soc_test_list_append(soc, soc_test_option_lst)
        if len(soc_test_option_lst) == 0:
            logging.info("lst is empty, trigger soc ascend910b")
            return "ascend910b"
        soc_test_ut_str: str = ""
        for soc in soc_test_option_lst:
            if soc not in cls._UTExcludes:
                soc_test_ut_str += f"{soc},"
        soc_test_ut_str = f"{soc_test_ut_str}"
        soc_test_ut_str = soc_test_ut_str[:-1]
        logging.info(f"Trigger UTs of soc: {soc_test_ut_str}")
        return soc_test_ut_str

    @classmethod
    def get_ops_test_option_lst(cls) -> List[str]:
        def ops_test_list_append(opt, ops_test_option_lst):
            if opt not in ops_test_option_lst:
                ops_test_option_lst.append(opt)

        ops_test_option_lst: List[str] = []
        for p in cls._ChangedPaths:
            for m in cls._Modules:
                new_options = m.get_test_example_ops_test_options(f=p)
                for opt in new_options:
                    ops_test_list_append(opt, ops_test_option_lst)
        return ops_test_option_lst


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s][%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    print(ParserSoc.main())
