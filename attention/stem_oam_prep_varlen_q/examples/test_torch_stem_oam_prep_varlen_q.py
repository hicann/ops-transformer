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
import torch_npu

# 参数设置
total_tokens = 256
H_q = 32
D = 128
batch = 2
stem_block_size = 128
stem_stride = 16

# 创建输入 tensor
q = (torch.randn(total_tokens, H_q, D) * 0.1).to(torch.float8_e4m3fn).npu()
q_scale = torch.ones(total_tokens, H_q, dtype=torch.float32).npu()
q_seq_lens = [128, 128]
cu_seq_lens_q = [0, 128, 256]

# 调用算子
qFlat = torch.ops.cann_ops_transformer.stem_oam_prep_varlen_q(
    q,
    q_seq_lens,
    cu_seq_lens_q,
    q_scale=q_scale,
    stem_block_size=stem_block_size,
    stem_stride=stem_stride,
)
torch.npu.synchronize()

print(f"qFlat shape: {qFlat.shape}")
print(f"qFlat dtype: {qFlat.dtype}")
print(f"qFlat[0, 0, 0, :10]: {qFlat[0, 0, 0, :10]}")
