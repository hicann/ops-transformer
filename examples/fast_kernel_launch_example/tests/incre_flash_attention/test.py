# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import sys
import logging
import torch
import torch.nn as nn
import ascend_ops

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# =============================================
# 🔧 CONFIGURATION CENTER (可配置化)
# =============================================


class AttentionConfig:
    def __init__(self):
        # Attention Dimensions
        self.batch_size = 18
        self.q_head_num = 64
        self.kv_head_num = 1
        self.q_seq = 1
        self.block_size = 128
        self.head_dim = 128

        # KV Cache Dimensions
        self.kv_seq_length = 8192
        self.max_block_num_per_batch = (self.kv_seq_length // self.block_size) + 1
        self.block_num = self.batch_size * self.max_block_num_per_batch

        # Attention Scale
        self.softmax_scale = 1.0 / (self.head_dim**0.5)


# =============================================
# 🧠 MODEL DEFINITION (模块化：模型清晰分离)
# =============================================


class FusedAttentionNetwork(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, param: dict):
        # Step 1: Compute metadata
        metadata = torch.ops.ascend_ops.npu_fused_infer_attention_score_metadata(
            **param["metaParam"]
        )
        param["faParam"]["metadata"] = metadata

        # Step 2: Call fused attention
        return torch.ops.ascend_ops.npu_fused_infer_attention_score(**param["faParam"])


# =============================================
# 📥 INPUT GENERATION (输入构造独立函数)
# =============================================


def generate_inputs(config: AttentionConfig):
    # Query
    q = (
        torch.randn(config.batch_size, config.q_head_num, config.q_seq, config.head_dim)
        .to(dtype=torch.bfloat16)
        .npu()
    )

    # KV Cache (INT8 quantized)
    key_cache = (
        torch.randint(
            0,
            100,
            (
                config.block_num,
                config.kv_head_num,
                config.head_dim // 32,
                config.block_size,
                32,
            ),
        )
        .to(dtype=torch.int8)
        .npu()
    )

    value_cache = (
        torch.randint(
            0,
            100,
            (
                config.block_num,
                config.kv_head_num,
                config.head_dim // 32,
                config.block_size,
                32,
            ),
        )
        .to(dtype=torch.int8)
        .npu()
    )

    # Block Table
    block_table = (
        torch.arange(
            config.batch_size * config.max_block_num_per_batch, dtype=torch.int32
        )
        .view(config.batch_size, config.max_block_num_per_batch)
        .npu()
    )

    # Sequence lengths
    actual_seq_kvlen = torch.tensor(
        [config.kv_seq_length] * config.batch_size, dtype=torch.int64
    ).npu()

    # Dequantization scales
    dequant_scale_key = (
        torch.randn(config.kv_head_num, 1, config.head_dim)
        .to(dtype=torch.bfloat16)
        .npu()
    )

    dequant_scale_value = (
        torch.randn(config.kv_head_num, 1, config.head_dim)
        .to(dtype=torch.bfloat16)
        .npu()
    )

    # FA Param
    fa_param = {
        "query": q,
        "key": key_cache,
        "value": value_cache,
        "actual_seq_kvlen": actual_seq_kvlen,
        "block_table": block_table,
        "dequant_scale_key": dequant_scale_key,
        "dequant_scale_value": dequant_scale_value,
        "num_query_heads": config.q_head_num,
        "num_key_value_heads": config.kv_head_num,
        "softmax_scale": config.softmax_scale,
        "block_size": config.block_size,
        "input_layout": "BNSD",
        "sparse_mode": 0,
        "inner_precise": 1,
        "key_quant_mode": 0,
        "value_quant_mode": 0,
    }
    # Meta Param
    meta_param = {
        "batch_size": config.batch_size,
        "query_seq_size": config.q_seq,
        "query_head_num": config.q_head_num,
        "key_head_num": config.kv_head_num,
        "head_dim": config.head_dim,
        "block_size": config.block_size,
        "max_block_num_per_batch": config.max_block_num_per_batch,
        "actual_seq_lengths_kv": actual_seq_kvlen.to(dtype=torch.int64),
        "layout_query": "BNSD",
    }

    return {"metaParam": meta_param, "faParam": fa_param}


# =============================================
# 🏁 MAIN EXECUTION (主入口函数)
# =============================================


def main():
    # 1. Load config
    config = AttentionConfig()

    # 2. Reset Dynamo (for clean compile)
    logger.info("🔄 Resetting TorchDynamo...")
    torch._dynamo.reset()

    # 3. Build model
    logger.info("🧠 Building FusedAttentionNetwork...")
    model = FusedAttentionNetwork().npu()

    # 4. Generate inputs
    logger.info("📥 Generating input tensors...")
    param = generate_inputs(config)

    # 5. Run inference
    logger.info("🚀 Running inference...")
    try:
        output, softmaxlse = model(param)
        torch.npu.synchronize()

        # 9. logger.info results
        logger.info(f"📊 Output shape: {output.shape}")
        logger.info(f"📉 Softmax output shape: {softmaxlse.shape}")

        # 10. Pretty logger.info output (first few values)
        logger.info("\n📌 First 5 values of output (BNSD):")
        logger.info(output[:5, :5, :5, :5].cpu().float().numpy())

    except Exception as e:
        logger.info(f"❌ Inference failed: {e}")
        raise


# =============================================
# 🚀 ENTRY POINT
# =============================================
if __name__ == "__main__":
    main()
