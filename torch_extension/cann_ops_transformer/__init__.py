# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)

try:
    import torch  # noqa: F401
    import torch_npu  # noqa: F401
    from . import ops  # noqa: F401
except ImportError as e:
    logger.error(f"导入cann_ops_transformer失败: {e}，请检查torch/torch_npu是否已安装")
    raise

_op_namespace = torch.ops.cann_ops_transformer

for _name in dir(ops):
    if _name.startswith("_") or hasattr(_op_namespace, _name):
        continue
    globals()[_name] = getattr(ops, _name)
del _name


def __getattr__(name):
    if hasattr(_op_namespace, name):
        handle = getattr(_op_namespace, name)
        globals()[name] = handle
        return handle
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    _seen = set(globals()) | set(dir(_op_namespace)) | set(dir(ops))
    return sorted(_seen)
