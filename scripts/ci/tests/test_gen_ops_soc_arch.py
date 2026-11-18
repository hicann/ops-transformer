# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import importlib.util
import tempfile
import unittest
from pathlib import Path

script = Path(__file__).resolve().parents[1] / "gen_ops_soc.py"
spec = importlib.util.spec_from_file_location("gen_ops_soc", script)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ArchRegistrationTest(unittest.TestCase):
    def test_multiline_and_overloads(self):
        content = """
        opbase::ArchConfigHelper::AddConfigByArch(
            this->AICore(), Ops::Base::DAV_3510, cfg);
        opbase::ArchConfigHelper::AddConfigByArch(this->AICore(), NpuArch::DAV_2201);
        """
        self.assertEqual(
            set(module.extract_arch_aicore_configs(content)),
            {"ascend950", "ascend350", "ascend910b", "ascend910_93"},
        )

    def test_960_and_comments(self):
        content = """
        // AddConfigByArch(this->AICore(), Ops::Base::DAV_3510, cfg);
        /* AddConfigByArch(this->AICore(), Ops::Base::DAV_2201, cfg); */
        AddConfigByArch(this->AICore(), Ops::Base::DAV_9201, cfg);
        AddConfigByArch(this->AICore(), Ops::Base::DAV_9202, cfg);
        """
        self.assertEqual(
            set(module.extract_arch_aicore_configs(content)),
            {"ascend960dt", "ascend960pr"},
        )

    def test_unknown_and_dynamic_arch_fail(self):
        for value in ("Ops::Base::DAV_9999", "dynamic_arch", '"DAV_3510"'):
            with (
                self.subTest(value=value),
                self.assertRaises(module.ArchitectureConfigurationError),
            ):
                module.extract_arch_aicore_configs(
                    f"AddConfigByArch(this->AICore(), {value}, cfg);"
                )

    def test_string_literals_are_not_registrations(self):
        content = 'LOG("AddConfigByArch(this->AICore(), DAV_3510, cfg)");'
        self.assertEqual(module.extract_arch_aicore_configs(content), [])

    def test_cpp_digit_separators_do_not_hide_registration(self):
        for separator in ("\n", " "):
            content = (
                "auto a=1'000;"
                + separator
                + "AddConfigByArch(this->AICore(), Ops::Base::DAV_3510);"
                + separator
                + "auto b=2'000;"
            )
            with self.subTest(separator=separator):
                self.assertEqual(
                    set(module.extract_arch_aicore_configs(content)),
                    {"ascend950", "ascend350"},
                )

    def test_grouped_def_preserves_mixed_registration(self):
        with tempfile.TemporaryDirectory() as root:
            op_dir = Path(root) / "posembedding" / "example" / "op_host"
            op_dir.mkdir(parents=True)
            (op_dir / "example_def.cpp").write_text(
                'this->AICore().AddConfig("ascend910b");\n'
                "opbase::ArchConfigHelper::AddConfigByArch(\n"
                "    this->AICore(), Ops::Base::DAV_3510, cfg);\n",
                encoding="utf-8",
            )
            for soc in ("ascend950", "ascend350", "ascend910b"):
                with self.subTest(soc=soc):
                    self.assertIn("example", module.grouped_def(root, soc, 1)[0])
            self.assertNotIn("example", module.grouped_def(root, "ascend960dt", 1)[0])


if __name__ == "__main__":
    unittest.main()
