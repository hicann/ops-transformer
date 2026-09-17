# KdaInputProj算子测试框架

## 文件结构

```text
pytest/
├── README.md                          # 测试框架说明
├── test_kda_input_proj_accuracy.py    # 四路输出端到端精度用例
├── kda_input_proj_mx_quant_golden.py  # Stage1 DynamicMxQuant CPU golden
├── kda_input_proj_qkv_golden.py       # Stage2 QuantMatmul(qkv) CPU golden
└── kda_input_proj_bgg_golden.py       # beta / gate / g CPU golden
```

## 功能说明

基于pytest验证KdaInputProj四路输出：

- **qkv**：CPU 上按 kernel 的 OCP MX 量化语义生成 `quant_x`/`x_scale`，再做 MX 矩阵乘得到 BF16 golden。
- **beta**：FP32 `sigmoid(x @ w_beta)`。
- **gate / g**：BF16 矩阵乘。

仅支持 Ascend 950PR/950DT。非该器件时用例会 skip。

## 使用方法

在仓库根目录执行：

```bash
python3 -m pytest -rA -s \
  attention/kda_input_proj/tests/pytest/test_kda_input_proj_accuracy.py \
  -v -m ci
```
