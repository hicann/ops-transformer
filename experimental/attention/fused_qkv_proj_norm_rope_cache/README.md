# FusedQkvProjNormRopeCache

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| Ascend 950PR&950DT系列产品 | √ |
| Atlas A3 训练/推理系列产品 | × |
| Atlas A2 训练/推理系列产品 | × |
| Atlas 200I/500 A2推理产品 | × |
| Atlas推理系列产品 | × |
| Atlas训练系列产品 | × |

## 功能说明

基于 CANNBot-DSL 的 WeLMv4 QKV 投影全融合算子。单个 `MIX_AIC_1_2`
复合核完成 QKV 投影、K RMSNorm、Q/K RoPE、BF16 转换与 KV cache 散写，
QKV 中间张量保留在片上，不回写 GM。

```text
qkv(M, 2560) = hidden(M, 2048) @ weight(2560, 2048)^T      BF16 输入，FP32 累加
├── [0:1536]     6 个 Q head：前 192 维直通，尾 64 维 RoPE    -> q(M, 1536)
├── [1536:1792]  K head：RMSNorm -> 尾 64 维 RoPE             -> k(M, 256) / k_cache[slot]
├── [1792:2048]  V head：FP32 -> BF16                         -> v(M, 256) / v_cache[slot]
├── [2048:2304]  mirror_k：直通                               -> mirror_k(M, 256)
└── [2304:2560]  mirror_v：直通                               -> mirror_v(M, 256)
```

RoPE 为 rotate-half：每个 256 维 head 的尾部 64 维拆成 `x1 = tail[0:32]`、
`x2 = tail[32:64]`，按 `cos_sin_cache` 当前行计算
`x1*cos - x2*sin`、`x1*sin + x2*cos`。`cos_sin_cache` 每行前 32 列为 cos、后 32 列为 sin。

`weight` 的 N 可以取 2560（kv-mirror 层，10 个 head 单元）或 2048
（普通 QKV 层，8 个 head 单元）。8 单元层不产生 mirror 输出。

## 接口

```python
from fused_qkv_proj_norm_rope_cache import fused_qkv_proj_norm_rope_cache

q, k, v, mirror_k, mirror_v = fused_qkv_proj_norm_rope_cache(
    hidden, weight, k_gamma, positions, cos_sin_cache, slot_mapping,
    k_cache, v_cache, eps=1e-6, return_v=True, positions_contiguous=True,
)
```

| 参数 | dtype / shape | 说明 |
| :--- | :--- | :--- |
| `hidden` | BF16 `[M, 2048]` | 投影输入，连续 |
| `weight` | BF16 `[N, 2048]` | `nn.Linear` 的 `(out, in)` 布局；N 为 2560 或 2048 |
| `k_gamma` | BF16 `[256]` | K RMSNorm 权重 |
| `positions` | INT32/INT64 `[M]` | 当前 token 位置；prefill 要求连续，decode 可任意 |
| `cos_sin_cache` | FP32 `[P, 64]` | 前 32 列为 cos，后 32 列为 sin |
| `slot_mapping` | INT32/INT64 `[M]` | cache 行号，`-1` 表示跳过 |
| `k_cache` / `v_cache` | BF16 `[slots, 256]` | 分页 KV cache，原地写入 |
| `eps` | float | RMSNorm epsilon，默认 `1e-6` |
| `return_v` | bool | 是否额外物化 `v`，默认 `True` |
| `positions_contiguous` | bool | `True` 走 prefill 连续位置路径，`False` 走 decode 逐 token 路径 |

返回值默认为 `(q, k, v, mirror_k, mirror_v)`；8 单元层为
`(q, k, v)`，`return_v=False` 时不返回 `v`。

## 约束

- 仅支持 Ascend 950PR/950DT（ARCH 3510）。
- 依赖 CANNBot-DSL 0.8.0（`RingBuffer`、`Constexpr`、`Tensor.partition`）。
- `hidden`、`weight`、`k_gamma`、`cos_sin_cache` 必须连续；cache 需可展平为 `(slots, 256)`。
- 当前只覆盖 BF16 权重路径，不含 modelslim-mxfp8 权重与 E8M0 scale 路径。

## 目录结构

```text
fused_qkv_proj_norm_rope_cache/
├── CMakeLists.txt                 # 算子登记 + cannbotdsl kernel 构建
├── op_dsl/
│   ├── fused_qkv_proj_norm_rope_cache.py   # DSL 实现与 AOT export
│   └── aclnn_fused_qkv_proj_norm_rope_cache.py
├── torch_extension/               # torch 算子边界（Meta + PrivateUse1）
└── tests/pytest/
    └── test_fused_qkv_proj_norm_rope_cache_dsl.py
```

## 精度验证

`tests/pytest/test_fused_qkv_proj_norm_rope_cache_dsl.py` 使用 CPU FP32 golden，
共 50 组用例，覆盖：
- prefill 连续位置 M=1/2/7/8/16/31/33/63/64/65/72/100/127/128/200/255/256/257/300/511/512/513/1000
- decode 任意位置、非零位置起点、自动位置模式
- 8/10 单元层（N=2048/2560）
- 2/3/5 步长跳过散写、`eps=1e-1/1e-3/1e-5`、`return_v=False`
- 小 cache、分页 3D cache、`int32` 索引、小/大 cos/sin 表
- 当前流 AIC/AIV 配额变化（32→8）触发独立编译与正确重编译
- 正式 Torch 注册入口的 8 单元层 Meta/实际输出 rank/shape/dtype 一致性

```bash
source ${install_path}/ascend-toolkit/set_env.sh
cd experimental/attention/fused_qkv_proj_norm_rope_cache/tests/pytest
python -m pytest -q test_fused_qkv_proj_norm_rope_cache_dsl.py
```
