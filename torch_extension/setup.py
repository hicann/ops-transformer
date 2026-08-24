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
import shutil
import glob
import logging
from setuptools import setup, find_packages
from setuptools import Command
from setuptools.command.build_py import build_py as _build_py
from setuptools.command.sdist import sdist as _sdist

try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel
except ImportError:
    _bdist_wheel = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DESCRIPTION = "CannOpsTransformer"
VERSION = "1.0.0"
BASE_PACKAGE_NAME = "cann_ops_transformer"

HERE = os.path.dirname(os.path.abspath(__file__))
OPS_TRANSFORMER_ROOT = os.path.normpath(os.path.join(HERE, ".."))
_src_path = os.path.join(HERE, BASE_PACKAGE_NAME)

_ops_env = os.environ.get("TORCH_EXTENSION_OPS", "").strip()
_selected_ops = (
    frozenset(op.strip() for op in _ops_env.split(",") if op.strip())
    if _ops_env
    else None
)

_vendor_env = os.environ.get("TORCH_EXTENSION_VENDOR", "").strip()

if _selected_ops:
    PACKAGE_NAME = "%s_%s" % (BASE_PACKAGE_NAME, _vendor_env or "custom")
else:
    PACKAGE_NAME = BASE_PACKAGE_NAME

if PACKAGE_NAME != BASE_PACKAGE_NAME and not os.path.isdir(
    os.path.join(HERE, PACKAGE_NAME)
):
    os.symlink(BASE_PACKAGE_NAME, os.path.join(HERE, PACKAGE_NAME))


_selected_op_categories = []

if _selected_ops is not None:
    _ops_dir = os.path.join(_src_path, "ops")
    for cat in os.listdir(_ops_dir):
        cat_path = os.path.join(_ops_dir, cat)
        if not os.path.isdir(cat_path):
            continue
        for name in os.listdir(cat_path):
            if name in _selected_ops:
                _selected_op_categories.append((cat, name))


def _non_python_files(directory):
    paths = []
    for path, dirs, filenames in os.walk(directory):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        if _selected_ops is not None:
            rel_path = os.path.relpath(path, directory)
            parts = rel_path.split(os.sep)
            if (
                len(parts) >= 2
                and parts[0] in ("ops", "csrc")
                and parts[1] not in _selected_ops
            ):
                continue
        for filename in filenames:
            if filename.endswith((".h", ".cpp")):
                paths.append(os.path.join(path, filename))
    return paths


_extra_packages = []
for _subdir in ("csrc", "common"):
    _root = os.path.join(_src_path, _subdir)
    if os.path.isdir(_root):
        for root, _, _ in os.walk(_root):
            rel = os.path.relpath(root, HERE).replace(os.sep, ".")
            _extra_packages.append(rel)


# --- 收集 ops-transformer/ 下所有 torch_extension/ 中的算子文件 ---
_op_category_inits = {}
_op_py_files = []
_op_cpp_files = []


def _collect_op(op_cat, op_dir, te_dir):
    _selected_op_categories.append((op_cat, op_dir))

    for f in sorted(os.listdir(te_dir)):
        if f.endswith(".py"):
            f_src = os.path.join(te_dir, f)
            _op_py_files.append((os.path.join("ops", op_cat, op_dir, f), f_src))

    csrc_dir = os.path.join(te_dir, "csrc")
    if os.path.isdir(csrc_dir):
        for cpp in sorted(os.listdir(csrc_dir)):
            if cpp.endswith(".cpp"):
                _op_cpp_files.append(
                    (os.path.join("csrc", op_cat, cpp), os.path.join(csrc_dir, cpp))
                )


for cat in sorted(os.listdir(OPS_TRANSFORMER_ROOT)):
    cat_path = os.path.join(OPS_TRANSFORMER_ROOT, cat)
    if not os.path.isdir(cat_path) or cat.startswith((".", "_")):
        continue

    cat_init_src = os.path.join(cat_path, "__init__.py")
    if os.path.isfile(cat_init_src):
        _op_category_inits[cat] = cat_init_src

    for name in sorted(os.listdir(cat_path)):
        sub_path = os.path.join(cat_path, name)

        # 2-level: <cat>/<name>/torch_extension  (e.g. attention/flash_attn)
        torch_extension = os.path.join(sub_path, "torch_extension")
        if os.path.isdir(torch_extension):
            if _selected_ops is not None and name not in _selected_ops:
                continue
            _collect_op(cat, name, torch_extension)
            continue

        # 3-level: <cat>/<subcat>/<op>/torch_extension  (e.g. experimental/attention/<op>)
        if not os.path.isdir(sub_path):
            continue
        for op_name in sorted(os.listdir(sub_path)):
            if _selected_ops is not None and op_name not in _selected_ops:
                continue
            torch_extension = os.path.join(sub_path, op_name, "torch_extension")
            if os.path.isdir(torch_extension):
                _collect_op(name, op_name, torch_extension)


_sorted_selected_op_categories = sorted(set(_selected_op_categories))

_entry_points = []
if _selected_ops is not None:
    for _cat, _op_name in _sorted_selected_op_categories:
        _entry_points.append(
            "%s = %s.ops.%s.%s:%s" % (_op_name, PACKAGE_NAME, _cat, _op_name, _op_name)
        )


class Clean(Command):
    description = "clean build artifacts"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        for pattern in ("build", "dist", "*.egg-info"):
            for path in glob.glob(os.path.join(HERE, pattern)):
                if os.path.isdir(path):
                    shutil.rmtree(path)
                    logger.info("removing %s", path)
        if PACKAGE_NAME != BASE_PACKAGE_NAME:
            link = os.path.join(HERE, PACKAGE_NAME)
            if os.path.islink(link):
                os.remove(link)


class BuildPyWithOps(_build_py):
    def run(self):
        _clean_build_artifacts()
        super().run()
        build_pkg = os.path.join(self.build_lib, PACKAGE_NAME)
        if _selected_ops is not None:
            for subdir in ("ops", "csrc"):
                base_dir = os.path.join(build_pkg, subdir)
                if not os.path.isdir(base_dir):
                    continue
                for cat_name in os.listdir(base_dir):
                    cat_dir = os.path.join(base_dir, cat_name)
                    if not os.path.isdir(cat_dir):
                        continue
                    for sub_op in os.listdir(cat_dir):
                        if sub_op not in _selected_ops and sub_op != "__pycache__":
                            target = os.path.join(cat_dir, sub_op)
                            if os.path.isdir(target):
                                shutil.rmtree(target)
                                logger.info(
                                    "removing centralized %s/%s/%s",
                                    subdir,
                                    cat_name,
                                    sub_op,
                                )
            ops_cat_inits = {}
            for sel_cat, sel_op in _sorted_selected_op_categories:
                if sel_cat not in ops_cat_inits:
                    ops_cat_inits[sel_cat] = []
                ops_cat_inits[sel_cat].append(sel_op)
            for sel_cat, sel_ops in ops_cat_inits.items():
                dst = os.path.join(build_pkg, "ops", sel_cat, "__init__.py")
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                with open(dst, "w") as f:
                    for sel_op in sel_ops:
                        f.write("from .%s import %s\n" % (sel_op, sel_op))
        else:
            for category, init_src in _op_category_inits.items():
                dst = os.path.join(build_pkg, "ops", category, "__init__.py")
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(init_src, dst)
        for rel_path, src_abs in _op_py_files:
            dst = os.path.join(build_pkg, rel_path)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if PACKAGE_NAME != BASE_PACKAGE_NAME:
                with open(src_abs, "r") as _f:
                    _content = _f.read()
                _content = _content.replace(
                    "from cann_ops_transformer.op_builder",
                    "from %s.op_builder" % PACKAGE_NAME,
                )
                _content = _content.replace(
                    "import cann_ops_transformer.op_builder",
                    "import %s.op_builder" % PACKAGE_NAME,
                )
                with open(dst, "w") as _f:
                    _f.write(_content)
            else:
                shutil.copy2(src_abs, dst)
        for rel_path, src_abs in _op_cpp_files:
            dst = os.path.join(build_pkg, rel_path)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src_abs, dst)
        common_src_dir = os.path.join(_src_path, "common")
        if os.path.isdir(common_src_dir):
            for root_dir, _, filenames in os.walk(common_src_dir):
                for filename in filenames:
                    src_abs = os.path.join(root_dir, filename)
                    rel_path = os.path.relpath(src_abs, _src_path)
                    dst = os.path.join(build_pkg, rel_path)
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    shutil.copy2(src_abs, dst)


_all_packages = find_packages() + _extra_packages
if _selected_ops is not None:
    _prefix = BASE_PACKAGE_NAME + "."
    _all_packages = [
        pkg.replace(_prefix, PACKAGE_NAME + ".", 1) if pkg.startswith(_prefix) else pkg
        for pkg in _all_packages
        if pkg == PACKAGE_NAME or pkg.startswith(PACKAGE_NAME + ".")
    ]


def _clean_symlink():
    if PACKAGE_NAME != BASE_PACKAGE_NAME:
        link = os.path.join(HERE, PACKAGE_NAME)
        if os.path.islink(link):
            os.remove(link)


class SdistWithClean(_sdist):
    def run(self):
        _clean_build_artifacts()
        super().run()
        _clean_symlink()


def _clean_build_artifacts():
    for pattern in ("build", "dist", "*.egg-info"):
        for path in glob.glob(os.path.join(HERE, pattern)):
            if os.path.isdir(path):
                shutil.rmtree(path)
                logger.info("removing %s", path)


if _bdist_wheel is not None:

    class BdistWheelWithClean(_bdist_wheel):
        def run(self):
            _clean_build_artifacts()
            super().run()
            _clean_symlink()
else:
    BdistWheelWithClean = None


_cmdclass = {
    "build_py": BuildPyWithOps,
    "sdist": SdistWithClean,
    "clean": Clean,
}
if BdistWheelWithClean is not None:
    _cmdclass["bdist_wheel"] = BdistWheelWithClean


setup(
    name=PACKAGE_NAME,
    version=VERSION,
    description=DESCRIPTION,
    author="CANN",
    license="CANN Open Software License Agreement Version 2.0",
    url="https://gitcode.com/cann/ops-transformer/tree/master/torch_extension",
    install_requires=["torch>=2.6.0", "torch_npu"],
    packages=_all_packages,
    package_data={PACKAGE_NAME: _non_python_files(_src_path)},
    entry_points={"cann_ops_transformer.ops": _entry_points} if _entry_points else {},
    cmdclass=_cmdclass,
    zip_safe=False,
)
