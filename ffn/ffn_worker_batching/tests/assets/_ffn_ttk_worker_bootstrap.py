# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Keep the forkserver NPU-free; import the installed wheel only after fork.

This module must use only the standard library at import time. Importing Torch
or cann_ops_transformer in the forkserver can initialize NPU state and poison
all subsequently forked workers.
"""

import os

_loaded_in_child = False


def _load_after_fork():
    global _loaded_in_child
    if _loaded_in_child:
        return
    _loaded_in_child = True
    try:
        import importlib
        import pathlib
        import sys
        import sysconfig

        site = pathlib.Path(sysconfig.get_path("purelib")).resolve()
        sys.path.insert(0, str(site))
        package = importlib.import_module("cann_ops_transformer")
        expected = (site / "cann_ops_transformer" / "__init__.py").resolve()
        if pathlib.Path(package.__file__).resolve() != expected:
            raise RuntimeError(
                f"FFN worker imported wrong package: {package.__file__}; expected {expected}"
            )
        print(f"FFN TTK worker installed package: {package.__file__}", flush=True)
    except BaseException:
        import traceback

        traceback.print_exc()
        # At-fork callback exceptions would otherwise be ignored by Python.
        os._exit(70)


os.register_at_fork(after_in_child=_load_after_fork)
