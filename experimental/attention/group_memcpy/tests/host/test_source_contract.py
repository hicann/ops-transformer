"""CPU-only packaging/registration checks; these do not validate NPU execution."""

import ast
import json
from pathlib import Path
import re
import unittest

OP = Path(__file__).resolve().parents[2]
ROOT = OP.parents[2]
TORCH = OP / "torch_extension"
DTYPES = [
    "INT8",
    "UINT8",
    "INT16",
    "INT32",
    "INT64",
    "FLOAT16",
    "BF16",
    "FLOAT",
    "BOOL",
]
SEMANTIC = [
    "int8",
    "uint8",
    "int16",
    "int32",
    "int64",
    "float16",
    "bfloat16",
    "float",
    "bool",
]


class SourceContract(unittest.TestCase):
    def test_opdef_and_json_agree(self):
        definitions = json.loads((OP / "group_memcpy.json").read_text(encoding="utf-8"))
        self.assertEqual(len(definitions), 1)
        definition = definitions[0]
        self.assertEqual(definition["op"], "GroupMemcpy")
        self.assertEqual(definition["attr"], [])
        source = (OP / "op_host/group_memcpy_def.cpp").read_text(encoding="utf-8")
        for name, kind in [("src", "input_desc"), ("dst", "output_desc")]:
            port = definition[kind][0]
            self.assertEqual(port["name"], name)
            self.assertEqual(port["param_type"], "dynamic")
            self.assertEqual(port["type"], SEMANTIC)
            self.assertEqual(port["format"], ["ND"] * len(DTYPES))
        self.assertEqual(re.findall(r"ge::DT_(\w+)", source), DTYPES + DTYPES)
        self.assertIn('AddConfig("ascend910b")', source)

    def test_binary_config_covers_dtypes(self):
        config = json.loads(
            (OP / "op_host/config/ascend910b/group_memcpy_binary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(config["op_type"], "GroupMemcpy")
        self.assertEqual(len(config["op_list"]), len(DTYPES))
        expected = ["float32" if dtype == "float" else dtype for dtype in SEMANTIC]
        for entry, dtype in zip(config["op_list"], expected):
            for direction, name in [("inputs", "src"), ("outputs", "dst")]:
                port = entry[direction][0][0]
                self.assertEqual(
                    (port["name"], port["dtype"], port["format"], port["paramType"]),
                    (name, dtype, "ND", "dynamic"),
                )

    def test_all_build_sources_exist(self):
        required = [
            "torch_extension/__init__.py",
            "torch_extension/group_memcpy.py",
            "torch_extension/csrc/group_memcpy.cpp",
            "op_kernel/group_memcpy.cpp",
            "op_kernel/group_memcpy.h",
            "op_host/group_memcpy_def.cpp",
            "op_host/group_memcpy_infershape.cpp",
            "op_host/group_memcpy_tiling.cpp",
            "op_host/group_memcpy_tiling.h",
            "op_host/op_api/aclnn_group_memcpy.cpp",
            "op_host/op_api/aclnn_group_memcpy.h",
            "tests/aclnn/main.cpp",
            "tests/host/test_tiling_policy.cpp",
        ]
        for relative in required:
            self.assertTrue((OP / relative).is_file(), relative)
        cmake = (OP / "CMakeLists.txt").read_text(encoding="utf-8")
        self.assertIn("add_subdirectory(op_host)", cmake)
        host = (OP / "op_host/CMakeLists.txt").read_text(encoding="utf-8")
        self.assertIn("ACLNNTYPE aclnn_exclude", host)
        self.assertIn("op_host_aclnnExc", host)

    def test_python_sources_parse(self):
        files = [
            OP / "tests/test_group_memcpy.py",
            TORCH / "group_memcpy.py",
            TORCH / "__init__.py",
        ]
        for path in files:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        wrapper = ast.parse(files[1].read_text(encoding="utf-8"))
        self.assertTrue(
            any(
                isinstance(node, ast.FunctionDef) and node.name == "group_memcpy"
                for node in wrapper.body
            )
        )

    def test_torch_registration_is_integrated(self):
        wrapper = (TORCH / "group_memcpy.py").read_text(encoding="utf-8")
        self.assertIn('"group_memcpy(Tensor[] src, Tensor(a!)[] dst) -> ()"', wrapper)
        self.assertIn("OpBuilder", wrapper)
        self.assertIn('"Meta"', wrapper)
        self.assertIn('"PrivateUse1"', wrapper)
        self.assertIn("torch.ops.cann_ops_transformer.group_memcpy", wrapper)
        binding = (TORCH / "csrc/group_memcpy.cpp").read_text(encoding="utf-8")
        self.assertIn("PYBIND11_MODULE", binding)
        self.assertIn("RunOpApiV2", binding)
        for path in (
            ROOT
            / "experimental/npu_ops_transformer_ext/npu_ops_transformer_ext/npu_ops_def.cpp",
            ROOT
            / "experimental/npu_ops_transformer_ext/npu_ops_transformer_ext/ops.py",
            ROOT
            / "experimental/npu_ops_transformer_ext/npu_ops_transformer_ext/__init__.py",
            ROOT / "experimental/npu_ops_transformer_ext/CMakeLists.txt",
        ):
            self.assertNotIn("group_memcpy", path.read_text(encoding="utf-8"))

    def test_public_symbols_and_kernel_name(self):
        header = (OP / "op_host/op_api/aclnn_group_memcpy.h").read_text(
            encoding="utf-8"
        )
        source = (OP / "op_host/op_api/aclnn_group_memcpy.cpp").read_text(
            encoding="utf-8"
        )
        for symbol in ["aclnnGroupMemcpyGetWorkspaceSize", "aclnnGroupMemcpy"]:
            self.assertIn(symbol + "(", header)
            self.assertIn(symbol + "(", source)
        kernel = (OP / "op_kernel/group_memcpy.cpp").read_text(encoding="utf-8")
        self.assertIn("void group_memcpy(", kernel)
        self.assertIn("GET_TILING_DATA", kernel)
        self.assertIn("TILING_KEY_IS(0)", kernel)
        self.assertNotIn("<<<", kernel)

    def test_document_links_resolve(self):
        for path in (
            OP / "README.md",
            OP / "docs/aclnnGroupMemcpy.md",
            OP / "docs/migration.md",
            OP / "docs/torchapi_group_memcpy.md",
            OP / "tests/README.md",
        ):
            text = path.read_text(encoding="utf-8")
            for target in re.findall(r"\]\(([^)\s]+)\)", text):
                if "://" in target or target.startswith("#"):
                    continue
                self.assertTrue(
                    (path.parent / target.split("#")[0]).exists(), (path.name, target)
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
