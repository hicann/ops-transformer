# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import torch

# 参数设置
total_tokens = 256
H_q = 32
D = 128
batch = 2
stemBlockSize = 128
stemStride = 16
max_seqlen_pad_stemBlockSize = (
    stemBlockSize  # = ceil(max_seqlen/stemBlockSize)*stemBlockSize
)

# 创建输入 tensor
q = torch.randn(total_tokens, H_q, D, dtype=torch.float8_e4m3fn).npu()
qScale = torch.randn(total_tokens, H_q, dtype=torch.float32).npu()
qSeqLens = torch.tensor([128, 128], dtype=torch.int32).npu()
cuSeqLensQ = torch.tensor([0, 128, 256], dtype=torch.int32).npu()

# 调用算子
qFlat = torch.ops.cann_ops_transformer.npu_stem_oam_prep_varlen_q(
    q=q,
    qSeqLens=qSeqLens,
    cuSeqLensQ=cuSeqLensQ,
    qScale=qScale,
    kScale=None,
    stemBlockSize=stemBlockSize,
    stemStride=stemStride,
)

print(f"qFlat shape: {qFlat.shape}")
print(f"qFlat dtype: {qFlat.dtype}")
print(f"qFlat[0, 0, 0, :10]: {qFlat[0, 0, 0, :10]}")
