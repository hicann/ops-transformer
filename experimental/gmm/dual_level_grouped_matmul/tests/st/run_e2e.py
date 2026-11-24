# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under CANN Open Software License Agreement Version 2.0.
"""Load the selected experimental Torch package before invoking unmodified TTK.

Usage (from ops-test-kit): python3 /path/to/run_e2e.py e2e -i cases.csv ...
Set CANN_TRANSFORMER_PACKAGE for a vendor-suffixed single-op build.
"""

import importlib
import os
import runpy
import sys

_package = importlib.import_module(
    os.environ.get("CANN_TRANSFORMER_PACKAGE", "cann_ops_transformer")
)
# Vendor builds keep the canonical torch.ops namespace. Expose the selected
# package under that namespace so discovery cannot load an unrelated install.
sys.modules.setdefault("cann_ops_transformer", _package)

if __name__ == "__main__":
    runpy.run_module("ttk", run_name="__main__")
