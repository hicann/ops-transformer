# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""arch92 uses one AIV lane per AIC and the shared PA numerical stages."""

if __package__:
    from . import _mega_kda_common as _common
    from .mega_recurrent_kda_frontend import export_precision_kernels
else:
    import _mega_kda_common as _common
    from mega_recurrent_kda_frontend import export_precision_kernels


class _HeadImplementation:
    def __init__(self, heads):
        self.heads = heads

    def execute_variable(self, arguments, prefix, cores):
        output = _common.execute_sequence(arguments, prefix, cores, self.heads)
        return output, _common._WORKSPACE_KEEP

    def export_kernels(self):
        _common.export_sequence(self.heads)


def for_heads(heads):
    return _HeadImplementation(heads)
