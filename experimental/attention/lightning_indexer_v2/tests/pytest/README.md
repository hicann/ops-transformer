# lightning_indexer_v2 算子测试框架

基于 pytest 测试框架，实现 lightning_indexer_v2（两级 TopK candidate source / candidate_mode==1）算子的功能验证：

- **CPU 侧**：复现算子功能用以生成 golden 数据
- **NPU 侧**：通过 TorchNPU 进行算子直调获取实际数据
- **精度对比**：CPU golden 与 NPU 结果精度对比验证算子功能

## 文件结构

- `lightning_indexer_v2_golden.py`       # 标杆：CPU 侧算子 golden 实现（GeneralizedLIV2 + candidate source 路径）
- `liv2_candidate_golden.py`             # 标杆：candidate golden（select_candidate_blocks_ref numpy/torch 双实现，纯 CPU 可导入）
- `liv2_parameter_normalization.py`      # 标杆配套：参数归一化（golden 依赖）
- `result_compare_method.py`             # 精度比较器：check_result / check_result_candidate 等
- `test_lightning_indexer_v2_model.py`   # 模型场景测试套
- `pytest.ini`                           # 测试标记

## 模型场景（BSND）

| 参数 | 取值 |
|---|---|
| layout | BSND×BSND（泛化：BSND×PA_BBND） |
| N1 (q_head_num) | 32 / 16 / 8 / 4 |
| D (head_dim) | 128 |
| topK | 512 |
| candidate_topk_blocks | 2048 |
| candidate_block_size | 8 |
| S1 / S2 | 4096 / 4096（S2 为压缩后 K 长度；泛化：16K 满覆盖边界） |
| cmp_ratio | 2 |
| sparse_mode (mask_mode) | 0 / 3 |
| 其余泛化 | dtype FP16/BF16、B>1 变长 seqused_k、PA_BBND key |

用例清单（默认 10 条，`LIV2_MODEL_LARGE=1` 追加 16K 边界用例）：

| 用例 | 覆盖点 |
|---|---|
| LIV2_MODEL_N1_32_MODE3 | 主规格：B=1，mask_mode=3 causal，cmp_residual_k=[1] |
| LIV2_MODEL_N1_16_MODE3 | N1=16，B=2 |
| LIV2_MODEL_N1_8_MODE3 | N1=8，B=2 |
| LIV2_MODEL_N1_4_MODE3 | N1=4，B=4 |
| LIV2_MODEL_N1_32_MODE0 | sparse_mode=0（no mask），cmp_residual_k=None |
| LIV2_MODEL_N1_4_MODE0 | sparse_mode=0，N1=4，B=2 |
| LIV2_MODEL_N1_32_MODE3_BF16 | BF16 dtype |
| LIV2_MODEL_N1_8_MODE0_BF16 | BF16 + sparse_mode=0 |
| LIV2_MODEL_VARLEN_MODE3 | B=2 seqused_k=[4096, 2049]（S2 非对齐） |
| LIV2_MODEL_PA_MODE3 | BSND q × PA_BBND k（block_table） |
| LIV2_MODEL_S2_16K_N1_4_MODE3 | k_seq=16384（numBlocks==candidate_topk_blocks 满覆盖边界），需 `LIV2_MODEL_LARGE=1` |

Per case 校验：
1. `sparse_indices` vs golden（source 不得改变位置级 topk）；
2. `candidate_block_length` 恒空 (0,)；
3. `candidate_topk_indices` vs golden（行级块号集合 + -1 槽数硬校验）。

## 环境配置

1. 确认 TorchNPU 为最新版本
2. 参考 Attention 融合算子 Experimental 使用说明激活 CANN 包和自定义算子包

## 运行

```bash
# 全套（Ascend910B / 910_93，需已编译 candidate 算子）
python3 -m pytest -rA -s test_lightning_indexer_v2_model.py -v -m ci

# 指定用例
LIV2_CASE_NAMES=LIV2_MODEL_N1_32_MODE3 python3 -m pytest -s test_lightning_indexer_v2_model.py -m ci

# 追加 16K 满覆盖边界用例
LIV2_MODEL_LARGE=1 python3 -m pytest -s test_lightning_indexer_v2_model.py -m ci
```

> 说明：candidate source 仅支持 arch22（Ascend910B / Ascend910_93）；arch35（Ascend950）上
> candidate-on 被 host 拒绝，本套件直接 skip。算子未注册时自动 skip。
