# sparse_lightning_indexer 算子测试框架

基于 pytest 测试框架，实现 sparse_lightning_indexer（candidate consumer / candidate_mode==2）算子的功能验证：

- **CPU 侧**：复现算子功能用以生成 golden 数据（NEG_HUGE 降级 leak 语义 → 原 topk 管线）
- **NPU 侧**：通过 TorchNPU 进行算子直调获取实际数据
- **精度对比**：复用 LIV2 的 `result_compare_method.check_result`（多重集合门 →
  `compare_topk_valid` 官方规则 → -1 槽硬校验）

golden 基础设施（GeneralizedLIV2 / 用例数据生成）复用同级 lightning_indexer_v2 算子的
tests/pytest 目录（共享 golden 模块），本目录仅实现 consumer 语义扩展。

## 文件结构

- `sparse_lightning_indexer_golden.py`      # 标杆：consumer golden（降级 + topk）+ NPU driver + 候选构造
- `test_sparse_lightning_indexer_model.py`  # 模型场景测试套
- `pytest.ini`                              # 测试标记
- （复用 ../../lightning_indexer_v2/tests/pytest/：lightning_indexer_v2_golden.py、
   result_compare_method.py、liv2_candidate_golden.py —— 标杆与精度比较器）

## 模型场景（BSND，与 LIV2 模型套一一配对）

| 参数 | 取值 |
|---|---|
| layout | BSND×BSND（泛化：BSND×PA_BBND） |
| N1 (q_head_num) | 32 / 16 / 8 / 4 |
| D (head_dim) | 128 |
| topK | 512 |
| candidate_topk_blocks (candBlocks) | 2048 |
| candidate_block_size | 8 |
| S1 / S2 | 4096 / 4096（S2 为压缩后 K 长度；泛化：16K 满覆盖边界） |
| cmp_ratio | 2 |
| sparse_mode (mask_mode) | 0 / 3 |
| 候选来源 | cand_spec="self"（同 score 过参考 source 行为构造，真实 source→consumer 链路形态） |
| 其余泛化 | dtype FP16/BF16、B>1 变长 seqused_k、PA_BBND key |

用例清单（默认 10 条，`SLI_MODEL_LARGE=1` 追加 16K 边界用例）：

| 用例 | 覆盖点 |
|---|---|
| SLI_MODEL_N1_32_MODE3 | 主规格：B=1，mask_mode=3 causal，cmp_residual_k=[1] |
| SLI_MODEL_N1_16_MODE3 | N1=16，B=2 |
| SLI_MODEL_N1_8_MODE3 | N1=8，B=2 |
| SLI_MODEL_N1_4_MODE3 | N1=4，B=4 |
| SLI_MODEL_N1_32_MODE0 | sparse_mode=0（no mask），cmp_residual_k=None |
| SLI_MODEL_N1_4_MODE0 | sparse_mode=0，N1=4，B=2 |
| SLI_MODEL_N1_32_MODE3_BF16 | BF16 dtype |
| SLI_MODEL_N1_8_MODE0_BF16 | BF16 + sparse_mode=0 |
| SLI_MODEL_VARLEN_MODE3 | B=2 seqused_k=[4096, 2049]（S2 非对齐） |
| SLI_MODEL_PA_MODE3 | BSND q × PA_BBND k（block_table） |
| SLI_MODEL_S2_16K_N1_4_MODE3 | k_seq=16384（numBlocks==candBlocks 满覆盖边界），需 `SLI_MODEL_LARGE=1` |

## 参数限制

- **candidate_topk_indices**：REQUIRED，INT32；BSND `[B,S1,N2,candBlocks]`；candBlocks ∈ (0, 2048] 且 64 的倍数
- **candidate_block_size**：[2,64] 且为 2 的幂（本套 8），与 source 侧必须一致
- **topk**：(0, 2048]（无 off 回退）；**return_value**：恒 0
- **平台**：仅 ascend910b / ascend910_93

## 环境配置

1. 确认 TorchNPU 为最新版本
2. 参考 Attention 融合算子 Experimental 使用说明激活 CANN 包和自定义算子包

## 运行

```bash
# 全套（Ascend910B / 910_93）
python3 -m pytest -rA -s test_sparse_lightning_indexer_model.py -v -m ci

# 指定用例
SLI_CASE_NAMES=SLI_MODEL_N1_32_MODE3 python3 -m pytest -s test_sparse_lightning_indexer_model.py -m ci

# 追加 16K 满覆盖边界用例
SLI_MODEL_LARGE=1 python3 -m pytest -s test_sparse_lightning_indexer_model.py -m ci
```

算子未注册时自动 skip。
