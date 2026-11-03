# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "parse_changed_files.py"
SPEC = importlib.util.spec_from_file_location("parse_changed_files", SCRIPT_PATH)
PARSE_CHANGED_FILES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PARSE_CHANGED_FILES)


class ParseClassifyFileTest(unittest.TestCase):
    def setUp(self):
        class IsolatedParser(PARSE_CHANGED_FILES.Parser):
            _Modules = []
            _ChangedPaths = []
            _UTExcludes = []
            _ExamplesExcludes = []

        self.parser = IsolatedParser
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        self.temp_path = Path(temp_dir.name)

    def parse_config(self, content):
        config_file = self.temp_path / "test_config.yaml"
        config_file.write_text(textwrap.dedent(content), encoding="utf-8")
        self.assertTrue(self.parser.parse_classify_file(config_file))

    def test_two_level_modules_are_parsed_once_with_full_names(self):
        self.parse_config(
            """
            attention:
              shared:
                module: true
                src: [attention/shared]
                options: [attention_shared]
            moe:
              shared:
                module: true
                src: [moe/shared]
                options: [moe_shared]
        """
        )

        self.assertEqual(len(self.parser._Modules), 2)
        self.assertEqual(
            [module.name for module in self.parser._Modules],
            ["root/attention/shared", "root/moe/shared"],
        )
        self.assertEqual(self.parser._Modules[0].options, ["attention_shared"])
        self.assertEqual(self.parser._Modules[1].options, ["moe_shared"])

    def test_three_level_modules_are_parsed_once_with_full_names(self):
        self.parse_config(
            """
            mc2:
              matrix:
                shared:
                  module: true
                  src: [mc2/matrix/shared]
                  options: [matrix_shared]
            examples:
              mc2:
                shared:
                  module: true
                  src: [examples/mc2/shared]
                  options: [example_shared]
        """
        )

        self.assertEqual(len(self.parser._Modules), 2)
        self.assertEqual(
            [module.name for module in self.parser._Modules],
            ["root/mc2/matrix/shared", "root/examples/mc2/shared"],
        )

    def test_test_selection_and_source_excludes_are_preserved(self):
        self.parse_config(
            """
            attention:
              enabled:
                module: true
                src: [attention/enabled]
                exclude: [attention/enabled/skipped]
                options: [enabled, no_ut, no_examples]
              no_ut:
                module: true
                src: [attention/no_ut]
                test:
                  ut: false
                options: [no_ut]
            mc2:
              matrix:
                no_examples:
                  module: true
                  src: [mc2/matrix/no_examples]
                  test:
                    examples: false
                  options: [no_examples]
        """
        )

        self.assertEqual(len(self.parser._Modules), 3)
        self.assertEqual(self.parser._UTExcludes, ["no_ut"])
        self.assertEqual(self.parser._ExamplesExcludes, ["no_examples"])
        enabled = self.parser._Modules[0]
        self.assertEqual(
            enabled.get_test_options(Path("attention/enabled/skipped/op.cpp")), []
        )
        self.assertEqual(enabled.get_test_options(Path("unrelated/op.cpp")), [])

        changed_file = self.temp_path / "changed_files"
        changed_file.write_text(
            "attention/enabled/op.cpp\nmc2/matrix/no_examples/op.cpp\n",
            encoding="utf-8",
        )
        self.assertTrue(self.parser.parse_changed_file(changed_file))
        self.assertEqual(
            set(self.parser.get_related_ut().split(";")) - {""},
            {"enabled", "no_examples"},
        )
        self.assertEqual(
            set(self.parser.get_related_examples().split(";")) - {""},
            {"enabled", "no_ut"},
        )
        self.assertEqual(self.parser.get_related_ut_mc2(), "no_examples;")
        self.assertEqual(
            set(self.parser.get_related_ut_exclude_mc2().split(";")) - {""},
            {"enabled", "no_examples"},
        )

    def test_paths_use_directory_boundaries_and_support_single_strings(self):
        self.parse_config(
            """
            op:
              module: true
              src: ops/foo
              exclude: ops/foo/private
              options: foo
        """
        )
        rule = self.parser._Modules[0]
        for path in ("ops/foo", "ops/foo/kernel.cpp", "ops/foo/private_other/k.cpp"):
            with self.subTest(path=path):
                self.assertEqual(rule.get_test_options(Path(path)), ["foo"])
        for path in ("ops/foobar/k.cpp", "ops/foo/private", "ops/foo/private/k.cpp"):
            with self.subTest(path=path):
                self.assertEqual(rule.get_test_options(Path(path)), [])

    def test_disabled_modules_and_metadata_are_not_expanded(self):
        self.parse_config(
            """
            metadata: [unused]
            disabled:
              module: false
              src: [disabled]
              options: [disabled]
            enabled:
              module: true
              src: [src, src]
              options: [first, second, first]
        """
        )
        self.assertEqual([rule.name for rule in self.parser._Modules], ["root/enabled"])
        self.assertEqual(self.parser._Modules[0].src_files, [Path("src")])
        self.assertEqual(self.parser._Modules[0].options, ["first", "second"])

    def test_reloading_rules_replaces_modules_and_exclusions(self):
        self.parse_config("old: {module: true, options: old, test: {ut: false}}")
        self.parse_config("new: {module: true, src: new, options: new}")
        self.assertEqual([rule.name for rule in self.parser._Modules], ["root/new"])
        self.assertEqual(self.parser._UTExcludes, [])
        self.assertEqual(self.parser._ExamplesExcludes, [])

    def test_changed_paths_filter_non_sources_and_replace_previous_input(self):
        changed_file = self.temp_path / "changed"
        changed_file.write_text(
            "src/k.cpp\nsrc/k.cpp\n\ndocs/a.h\nsrc/README.md\na.json\na.ini\n",
            encoding="utf-8",
        )
        self.assertTrue(self.parser.parse_changed_file(changed_file))
        self.assertEqual(self.parser._ChangedPaths, [Path("src/k.cpp")])
        changed_file.write_text("src/new.h\n", encoding="utf-8")
        self.assertTrue(self.parser.parse_changed_file(changed_file))
        self.assertEqual(self.parser._ChangedPaths, [Path("src/new.h")])

    def test_invalid_rules_do_not_publish_partial_state(self):
        self.parse_config("old: {module: true, src: old, options: old}")
        config = self.temp_path / "invalid.yaml"
        for content in (
            "first: {module: true, src: first}\nbad: {module: true, src: /absolute}",
            "bad: {module: true, exclude: /absolute}",
            "bad: [unterminated",
        ):
            with self.subTest(content=content), self.assertLogs(level="ERROR"):
                config.write_text(content, encoding="utf-8")
                self.assertFalse(self.parser.parse_classify_file(config))
                self.assertEqual(
                    [rule.name for rule in self.parser._Modules], ["root/old"]
                )

    def test_invalid_changes_do_not_publish_partial_state(self):
        self.parser._ChangedPaths = [Path("old.cpp")]
        changed_file = self.temp_path / "changed"
        changed_file.write_text("new.cpp\n/absolute.cpp\n", encoding="utf-8")
        with self.assertLogs(level="ERROR"):
            self.assertFalse(self.parser.parse_changed_file(changed_file))
        self.assertEqual(self.parser._ChangedPaths, [Path("old.cpp")])

    def test_missing_files_return_false(self):
        missing = self.temp_path / "missing"
        with self.assertLogs(level="ERROR"):
            self.assertFalse(self.parser.parse_classify_file(missing))
            self.assertFalse(self.parser.parse_changed_file(missing))

    def test_all_target_is_preserved_for_each_selector(self):
        self.parse_config(
            "all: {module: true, src: ., options: all, test: {ut: false, examples: false}}"
        )
        self.parser._ChangedPaths = [Path("mc2/a.cpp"), Path("src/b.cpp")]
        for command in (
            "get_related_ut",
            "get_related_ut_mc2",
            "get_related_ut_exclude_mc2",
            "get_related_examples",
        ):
            with self.subTest(command=command):
                self.assertEqual(getattr(self.parser, command)(), "all")

    def test_empty_configuration_and_changes_select_nothing(self):
        self.parse_config("")
        changed_file = self.temp_path / "changed"
        changed_file.write_text("\n", encoding="utf-8")
        self.assertTrue(self.parser.parse_changed_file(changed_file))
        self.assertEqual(self.parser.get_related_ut(), "")
        self.assertEqual(self.parser.get_related_examples(), "")

    def test_cli_selectors_and_soc_script_keep_their_output_contract(self):
        self.parse_config(
            """
            first: {module: true, src: src, options: first}
            second: {module: true, src: mc2, options: second, test: {examples: false}}
        """
        )
        changed_file = self.temp_path / "changed"
        changed_file.write_text("src/a.cpp\nmc2/b.cpp\n", encoding="utf-8")
        for command, expected in (
            ("get_related_ut", "first;second;"),
            ("get_related_ut_mc2", "second;"),
            ("get_related_ut_exclude_mc2", "first;"),
            ("get_related_examples", "first;"),
        ):
            with self.subTest(command=command):
                result = subprocess.run(
                    [
                        sys.executable,
                        str(SCRIPT_PATH),
                        "-c",
                        str(self.temp_path / "test_config.yaml"),
                        "-f",
                        str(changed_file),
                        command,
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                self.assertEqual(result.stdout, expected + "\n")
        soc_config = self.temp_path / "soc.yaml"
        soc_config.write_text(
            "soc: {module: true, src: ., options: [ascend910b, ascend950]}",
            encoding="utf-8",
        )
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT_PATH.with_name("get_soc_version.py")),
                "-c",
                str(soc_config),
                "-f",
                str(changed_file),
                "get_related_soc",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertEqual(
            set(result.stdout.strip().split(",")), {"ascend910b", "ascend950"}
        )


if __name__ == "__main__":
    unittest.main()
