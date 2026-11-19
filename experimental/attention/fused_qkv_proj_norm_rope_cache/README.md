# FusedQkvProjNormRopeCache

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :------: |
| <term>Ascend 950PR&950DT系列产品</term> | √ |
| <term>Atlas A3系列产品</term> | × |
| <term>Atlas A2系列产品</term> | × |
| <term>Atlas 200I/500 A2推理产品</term> | × |
| <term>Atlas推理系列产品</term> | × |
| <term>Atlas训练系列产品</term> | × |

## 功能说明

- 算子功能：基于 CANNBot-DSL 的 WeLMv4 QKV 投影全融合算子。单个 `MIX_AIC_1_2` 复合核完成 QKV 投影、K RMSNorm、Q/K RoPE、BF16 转换与 KV cache 散写，QKV 中间张量保留在片上，不回写 GM。

- 计算公式：

    记输入激活为 $X\in\mathbb{R}^{M\times2048}$，权重为 $W\in\mathbb{R}^{N\times2048}$，$N$ 取 2048 或 2560，则：

    $$
    Y = XW^{T}\in\mathbb{R}^{M\times N}
    $$

    当 $N=2560$（kv-mirror 层）时，按 head 单元切分：

    $$
    q = \operatorname{RoPE}\left(Y[:,0:1536]\right),\quad
    k = \operatorname{RoPE}\left(\operatorname{RMSNorm}\left(Y[:,1536:1792]\right)\right)
    $$

    $$
    v = \operatorname{BF16}\left(Y[:,1792:2048]\right),\quad
    mirror\_k = \operatorname{BF16}\left(Y[:,2048:2304]\right),\quad
    mirror\_v = \operatorname{BF16}\left(Y[:,2304:2560]\right)
    $$

    当 $N=2048$（普通 QKV 层）时只产生 $q$、$k$、$v$，不产生 mirror 输出。

    K RMSNorm 为：

    $$
    k_{norm} = \frac{Y_k}{\sqrt{\operatorname{mean}(Y_k^{2})+\epsilon}}\odot\gamma
    $$

    RoPE 为 rotate-half：每个 256 维 head 的尾部 64 维拆成 $x_1=tail[0:32]$、$x_2=tail[32:64]$，按 `cos_sin_cache` 当前行计算：

    $$
    x_1' = x_1\cos - x_2\sin,\qquad x_2' = x_1\sin + x_2\cos
    $$

    `cos_sin_cache` 每行前 32 列为 cos、后 32 列为 sin。`slot_mapping` 非负时将 $k$、$v$ 原地写入对应 cache 行，值为 -1 的 token 跳过散写。

- 实现说明：

    权重按 head 单元切成 $N/256$ 个 N tile；Cube 以 `PAIR_M=256` 行成对处理，一对 128 行共享一次 A 载入和一次 B 切片，降低 L1 流量。L0C 的 FP32 结果经 Fixpipe 按行折半直达两个 AIV 的 UB，向量段就地完成 RoPE、RMSNorm、BF16 转换与散写。启动核数按 `min(ceil(M/256)*N/256, 当前流可用 AIC 组数)` 推导，流配额变化会触发独立编译与重编译。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| :--- | :--- | :--- | :--- | :--- |
| hidden | 输入 | 投影输入，shape 为 `[M,2048]`，要求连续。 | BFLOAT16 | ND |
| weight | 输入 | `nn.Linear` 的 `(out,in)` 布局，shape 为 `[N,2048]`，`N` 仅支持 2048 或 2560。 | BFLOAT16 | ND |
| k_gamma | 输入 | K RMSNorm 权重，shape 为 `[256]`。 | BFLOAT16 | ND |
| positions | 输入 | 当前 token 位置，shape 为 `[M]`；prefill 要求连续，decode 可任意。 | INT32、INT64 | ND |
| cos_sin_cache | 输入 | RoPE 表，shape 为 `[P,64]`；前 32 列为 cos，后 32 列为 sin。 | FLOAT32 | ND |
| slot_mapping | 输入 | cache 行号，shape 为 `[M]`；-1 表示跳过该 token。 | INT32、INT64 | ND |
| k_cache | 输入 | 分页 K cache 输入；最后一维为 256，可展平为 `[slots,256]`。 | BFLOAT16 | ND |
| k_cache | 输出 | 原地更新后的分页 K cache，与输入为同一张量。 | BFLOAT16 | ND |
| v_cache | 输入 | 分页 V cache 输入；shape 与 `k_cache` 相同。 | BFLOAT16 | ND |
| v_cache | 输出 | 原地更新后的分页 V cache，与输入为同一张量。 | BFLOAT16 | ND |
| eps | 属性 | K RMSNorm 的数值稳定项，默认值为 `1e-6`。 | FLOAT | - |
| positions_contiguous | 属性 | 是否走 prefill 连续位置路径，默认值为 `True`；`False` 时走 decode 逐 token 路径。 | BOOL | - |

## 约束说明

- 图模式（ACL Graph）捕获期间跳过依赖设备值回读的参数校验（`positions` 连续性/范围、`slot_mapping` 范围与唯一性）；调用方必须保证捕获与回放时索引合法。
- 仅支持 Ascend 950PR/950DT（ARCH 3510）。
- `M >= 1`；`hidden`、`weight`、`k_gamma`、`cos_sin_cache` 必须连续，且位于同一 NPU。
- `weight` 的 N 仅支持 2048（普通 QKV 层）或 2560（kv-mirror 层）。
- `k_cache` / `v_cache` 必须连续，最后一维为 256，可展平为 `[slots,256]`；两者 shape 必须一致。
- `positions_contiguous=True` 且 `M > 1` 时，要求 `positions` 严格连续；`positions` 取值范围为 `[0, P-1]`。
- `slot_mapping` 取值范围为 `[-1, slots-1]`；-1 表示跳过散写，非负槽位必须唯一，重复槽位会被拒绝。
- 当前只覆盖 BF16 权重路径，不包含 modelslim-mxfp8 权重与 E8M0 scale 路径。
- 当前提供前向实现，不提供反向。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| :--- | :--- | :--- |
| PyTorch API | - | 通过[torchapi_fused_qkv_proj_norm_rope_cache.md](docs/torchapi_fused_qkv_proj_norm_rope_cache.md)接口调用，支持单算子与`torch.compile`（`npugraph_ex`）图模式，固定返回 `(q,k,v,mirror_k,mirror_v)` 五个输出。 |
| DSL 直接调用 | [fused_qkv_proj_norm_rope_cache.py](op_dsl/fused_qkv_proj_norm_rope_cache.py) | 将 `experimental/attention/fused_qkv_proj_norm_rope_cache/op_dsl` 加入 `PYTHONPATH` 后调用，支持 `return_v`、`validate` 和 `positions_contiguous=None` 自动探测。 |

### PyTorch API 调用示例

```python
import torch
import torch_npu
from cann_ops_transformer.ops import fused_qkv_proj_norm_rope_cache

torch.npu.set_device(0)
device = torch.device("npu:0")
tokens, units, max_pos, slots = 72, 10, 4096, 1024

hidden = (torch.randn(tokens, 2048, device=device) * 0.1).bfloat16()
weight = (torch.randn(units * 256, 2048, device=device) * 0.05).bfloat16()
k_gamma = (torch.randn(256, device=device) * 0.1 + 1.0).bfloat16()
positions = torch.arange(tokens, dtype=torch.int64, device=device)
cos_sin_cache = torch.randn(max_pos, 64, dtype=torch.float32, device=device)
slot_mapping = torch.arange(tokens, dtype=torch.int64, device=device)
k_cache = torch.zeros(slots, 256, dtype=torch.bfloat16, device=device)
v_cache = torch.zeros(slots, 256, dtype=torch.bfloat16, device=device)

q, k, v, mirror_k, mirror_v = fused_qkv_proj_norm_rope_cache(
    hidden, weight, k_gamma, positions, cos_sin_cache, slot_mapping,
    k_cache, v_cache, eps=1e-6, positions_contiguous=True,
)
print(q.shape, k.shape, v.shape, mirror_k.shape, mirror_v.shape)
```

### 图模式调用

```python
class QkvModel(torch.nn.Module):
    def forward(self, hidden, weight, k_gamma, positions, cos_sin_cache,
                slot_mapping, k_cache, v_cache):
        return fused_qkv_proj_norm_rope_cache(
            hidden, weight, k_gamma, positions, cos_sin_cache, slot_mapping,
            k_cache, v_cache, eps=1e-6, positions_contiguous=True,
        )

model = torch.compile(
    QkvModel().eval(), backend="npugraph_ex", fullgraph=True, dynamic=False,
)
out = model(hidden, weight, k_gamma, positions, cos_sin_cache,
            slot_mapping, k_cache, v_cache)
```

### 精度验证

`tests/pytest/test_fused_qkv_proj_norm_rope_cache_dsl.py` 使用 CPU FP32 golden，共 51 组用例，覆盖：

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

## 参考资源

- [PyTorch API 文档（中文）](docs/torchapi_fused_qkv_proj_norm_rope_cache.md)
- [PyTorch API Document (English)](docs/torchapi_fused_qkv_proj_norm_rope_cache_en.md)
- [算子单元测试](tests/pytest/test_fused_qkv_proj_norm_rope_cache_dsl.py)
