# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Load this operator's sibling modules without relying on the old repository."""

import hashlib
import importlib.util
from importlib.machinery import ModuleSpec
import os
from pathlib import Path
import sys

OPERATOR_ROOT = Path(__file__).resolve().parents[1]


def load_sample(relative_path):
    relative = Path(relative_path)
    if len(relative.parts) != 2 or relative.suffix != ".py":
        raise ValueError(f"expected operator/module.py, got {relative_path!r}")
    operator, filename = relative.parts
    if operator == "flash_kda_grad":
        root = OPERATOR_ROOT / "op_dsl"
    elif operator == "flash_kda_fwd":
        external = os.environ.get("KDA_DSL_ROOT")
        if not external:
            raise FileNotFoundError(
                "The optional forward operator is not bundled. Set KDA_DSL_ROOT "
                "to a compatible CANNBotDSL checkout for forward integration tests."
            )
        root = Path(external).expanduser().resolve() / "samples" / operator
    else:
        raise ValueError(f"unsupported operator {operator!r}")
    path = root / filename
    if not path.is_file():
        raise FileNotFoundError(path)
    # Isolate checkouts as well as sibling imports; never fall back to an installed wheel.
    key = hashlib.sha256(str(root).encode()).hexdigest()[:16]
    prefix = f"_cannbot_kdag_{key}"
    if prefix not in sys.modules:
        package_spec = ModuleSpec(prefix, loader=None, is_package=True)
        package_spec.submodule_search_locations = [str(root)]
        sys.modules[prefix] = importlib.util.module_from_spec(package_spec)
    name = f"{prefix}.{relative.stem}"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load operator {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return module
