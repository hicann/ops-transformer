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
"""Select CI test targets from module rules and repository-relative changes."""

import argparse
import logging
from pathlib import Path

import yaml


class Module:
    """One terminal rule, independent of its depth in the configuration tree."""

    def __init__(self, name):
        self.name = name
        self.src_files = []
        self.src_exclude_files = []
        self.options = []
        self.test_excludes = []

    def update_classify_cfg(self, desc):
        fields = {}
        for key, attribute in (
            ("src", "src_files"),
            ("exclude", "src_exclude_files"),
            ("options", "options"),
        ):
            values = desc.get(key, [])
            if isinstance(values, (str, Path)):
                values = [values]
            values = list(dict.fromkeys(values))
            if key != "options":
                values = [Path(value) for value in values]
                if any(value.is_absolute() for value in values):
                    logging.error("Rule %s requires relative %s paths", self.name, key)
                    return False
            fields[attribute] = values
        for attribute, values in fields.items():
            setattr(self, attribute, values)
        tests = desc.get("test", {})
        self.test_excludes = [
            kind
            for kind in ("examples", "ut")
            if isinstance(tests, dict) and not tests.get(kind, True)
        ]
        return True

    def get_test_options(self, f):
        ancestors = {f, *f.parents}
        included = ancestors.intersection(self.src_files)
        excluded = ancestors.intersection(self.src_exclude_files)
        return self.options.copy() if included and not excluded else []

    def get_test_example_ops_test_options(self, f):
        return self.get_test_options(f)

    def print_details(self):
        logging.debug(
            "Module %s: src=%s exclude=%s options=%s",
            self.name,
            self.src_files,
            self.src_exclude_files,
            self.options,
        )


class Parser:
    # Keep the interface used by get_soc_version.ParserSoc. Parsing replaces
    # state on the receiving class rather than mutating an inherited list.
    _Modules = []
    _ChangedPaths = []
    _UTExcludes = []
    _ExamplesExcludes = []

    @staticmethod
    def main():
        commands = (
            "get_related_ut",
            "get_related_ut_mc2",
            "get_related_ut_exclude_mc2",
            "get_related_examples",
        )
        cli = argparse.ArgumentParser(description=__doc__)
        cli.add_argument("-c", "--classify", required=True, type=Path)
        cli.add_argument("-f", "--file", required=True, type=Path)
        cli.add_argument("command", choices=commands)
        args = cli.parse_args()
        if Parser.parse_classify_file(args.classify) and Parser.parse_changed_file(
            args.file
        ):
            Parser.print_details()
            return getattr(Parser, args.command)()
        return ""

    @classmethod
    def file_filter(cls, file_path):
        return file_path.suffix not in (".md", ".json", ".ini") and "docs/" not in str(
            file_path
        )

    @classmethod
    def parse_classify_file(cls, file):
        try:
            with Path(file).open(encoding="utf-8") as stream:
                configuration = yaml.safe_load(stream)
        except (OSError, yaml.YAMLError) as error:
            logging.error("Cannot read module rules: %s", error)
            return False

        # Visit each mapping once. A module is a leaf: src/test/options are
        # properties, never additional nodes to expand through the root.
        pending = [("root", configuration)]
        modules = []
        while pending:
            name, node = pending.pop()
            if not isinstance(node, dict):
                continue
            if "module" not in node:
                pending.extend(
                    (f"{name}/{key}", value)
                    for key, value in reversed(list(node.items()))
                )
                continue
            if node["module"]:
                rule = Module(name)
                if not rule.update_classify_cfg(node):
                    return False
                modules.append(rule)

        cls._Modules = modules
        cls._UTExcludes = [
            rule.name.rsplit("/", 1)[-1]
            for rule in modules
            if "ut" in rule.test_excludes
        ]
        cls._ExamplesExcludes = [
            rule.name.rsplit("/", 1)[-1]
            for rule in modules
            if "examples" in rule.test_excludes
        ]
        return True

    @classmethod
    def parse_changed_file(cls, file):
        try:
            lines = Path(file).read_text(encoding="utf-8").splitlines()
        except OSError as error:
            logging.error("Cannot read changed paths: %s", error)
            return False
        paths = list(
            dict.fromkeys(Path(line.strip()) for line in lines if line.strip())
        )
        if any(path.is_absolute() for path in paths):
            logging.error("Changed paths must be repository-relative")
            return False
        cls._ChangedPaths = [path for path in paths if cls.file_filter(path)]
        return True

    @classmethod
    def print_details(cls):
        for rule in cls._Modules:
            rule.print_details()
        logging.debug("Changed sources: %s", cls._ChangedPaths)

    @classmethod
    def _collect_options(cls, mc2=None, examples=False):
        selected = {}
        for path in cls._ChangedPaths:
            if mc2 is not None and ("mc2" in path.parts) != mc2:
                continue
            for rule in cls._Modules:
                method = (
                    rule.get_test_example_ops_test_options
                    if examples
                    else rule.get_test_options
                )
                selected.update(dict.fromkeys(method(path)))
        return list(selected)

    @staticmethod
    def _format_options(options, excluded):
        if "all" in options:
            return "all"
        return "".join(f"{option};" for option in options if option not in excluded)

    @classmethod
    def get_ops_test_ut_str(cls, ops_test_option_lst):
        return cls._format_options(ops_test_option_lst, cls._UTExcludes)

    @classmethod
    def get_ops_test_option_lst(cls):
        return cls._collect_options(examples=True)

    @classmethod
    def get_related_ut(cls):
        return cls.get_ops_test_ut_str(cls._collect_options())

    @classmethod
    def get_related_ut_mc2(cls):
        return cls.get_ops_test_ut_str(cls._collect_options(mc2=True))

    @classmethod
    def get_related_ut_exclude_mc2(cls):
        return cls.get_ops_test_ut_str(cls._collect_options(mc2=False))

    @classmethod
    def get_related_examples(cls):
        return cls._format_options(cls.get_ops_test_option_lst(), cls._ExamplesExcludes)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(Parser.main())
