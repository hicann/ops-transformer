# fused_qkv_proj_norm_rope_cache

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3系列产品</term>：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2系列产品</term>：不支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2推理产品</term>：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas推理系列产品</term>：不支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

- **接口功能**：`cann_ops_transformer.fused_qkv_proj_norm_rope_cache` 将 QKV 投影、K RMSNorm、Q/K RoPE、BF16 转换与 KV cache 散写融合为一次 Torch 自定义算子调用。该接口直接调用 DSL 实现，QKV 中间张量保留在片上，不写回 GM。完整的计算语义、调度与精度说明见 [FusedQkvProjNormRopeCache](../README.md)。

- **计算语义**：

    记输入激活为 $X\in\mathbb{R}^{M\times2048}$，权重为 $W\in\mathbb{R}^{N\times2048}$，$N$ 取 2048 或 2560，则：

    $$
    Y = XW^{T}\in\mathbb{R}^{M\times N}
    $$

    当 $N=2560$（kv-mirror 层）时：

    $$
    q = \operatorname{RoPE}\left(Y[:,0:1536]\right),\quad
    k = \operatorname{RoPE}\left(\operatorname{RMSNorm}\left(Y[:,1536:1792]\right)\right)
    $$

    $$
    v = \operatorname{BF16}\left(Y[:,1792:2048]\right),\quad
    mirror\_k = \operatorname{BF16}\left(Y[:,2048:2304]\right),\quad
    mirror\_v = \operatorname{BF16}\left(Y[:,2304:2560]\right)
    $$

    当 $N=2048$（普通 QKV 层）时只产生 $q$、$k$、$v$，Torch 接口的 `mirror_k`、`mirror_v` 返回 shape 为 `(0,256)` 的空张量。

    K RMSNorm 为：

    $$
    k_{norm} = \frac{Y_k}{\sqrt{\operatorname{mean}(Y_k^{2})+\epsilon}}\odot\gamma
    $$

    RoPE 为 rotate-half：每个 256 维 head 的尾部 64 维拆成 $x_1=tail[0:32]$、$x_2=tail[32:64]$，按 `cos_sin_cache` 当前行计算：

    $$
    x_1' = x_1\cos - x_2\sin,\qquad x_2' = x_1\sin + x_2\cos
    $$

    `cos_sin_cache` 每行前 32 列为 cos、后 32 列为 sin。`slot_mapping` 非负时将 $k$、$v$ 原地写入对应 cache 行，值为 -1 的 token 跳过散写。

## 函数原型

```python
import cann_ops_transformer

cann_ops_transformer.fused_qkv_proj_norm_rope_cache(
    hidden: Tensor,
    weight: Tensor,
    k_gamma: Tensor,
    positions: Tensor,
    cos_sin_cache: Tensor,
    slot_mapping: Tensor,
    k_cache: Tensor,
    v_cache: Tensor,
    *,
    eps: float = 1e-6,
    positions_contiguous: bool = True,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
```

也可以使用 `from cann_ops_transformer.ops import fused_qkv_proj_norm_rope_cache` 导入该接口。接口注册名为 `torch.ops.cann_ops_transformer.fused_qkv_proj_norm_rope_cache`。Torch 接口固定返回五个输出；如需 `return_v=False` 或显式关闭参数校验，请直接调用 [op_dsl/fused_qkv_proj_norm_rope_cache.py](../op_dsl/fused_qkv_proj_norm_rope_cache.py) 中的 DSL 入口。

## 参数说明

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| hidden | Tensor | 必选 | 投影输入，要求连续。数据格式为 ND。 | bfloat16 | (M, 2048) |
| weight | Tensor | 必选 | `nn.Linear` 的 `(out,in)` 布局权重，要求连续；N 仅支持 2048 或 2560。数据格式为 ND。 | bfloat16 | (N, 2048) |
| k_gamma | Tensor | 必选 | K RMSNorm 权重。数据格式为 ND。 | bfloat16 | (256) |
| positions | Tensor | 必选 | 当前 token 位置；prefill 要求连续，decode 可任意。数据格式为 ND。 | int32、int64 | (M) |
| cos_sin_cache | Tensor | 必选 | RoPE 表，前 32 列为 cos，后 32 列为 sin。数据格式为 ND。 | float32 | (P, 64) |
| slot_mapping | Tensor | 必选 | cache 行号，-1 表示跳过该 token。数据格式为 ND。 | int32、int64 | (M) |
| k_cache | Tensor | 必选 | 分页 K cache，原地写入；最后一维为 256，可展平为 (slots, 256)。数据格式为 ND。 | bfloat16 | (slots, 256) |
| v_cache | Tensor | 必选 | 分页 V cache，原地写入；shape 与 `k_cache` 相同。数据格式为 ND。 | bfloat16 | (slots, 256) |
| eps | float | 可选 | K RMSNorm 的数值稳定项，默认值为 `1e-6`。 | float | - |
| positions_contiguous | bool | 可选 | 是否走 prefill 连续位置路径，默认值为 `True`；设为 `False` 时走 decode 逐 token 路径。 | bool | - |

## 返回值说明

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| q | Tensor | 必选 | Q 投影与 RoPE 结果。数据格式为 ND。 | bfloat16 | (M, 1536) |
| k | Tensor | 必选 | K 投影、RMSNorm 与 RoPE 结果。数据格式为 ND。 | bfloat16 | (M, 256) |
| v | Tensor | 必选 | V 投影结果。Torch 接口固定返回该输出。数据格式为 ND。 | bfloat16 | (M, 256) |
| mirror_k | Tensor | 必选 | mirror K 输出；N=2560 时为 `(M,256)`，N=2048 时为空张量 `(0,256)`。数据格式为 ND。 | bfloat16 | (M, 256) 或 (0, 256) |
| mirror_v | Tensor | 必选 | mirror V 输出；N=2560 时为 `(M,256)`，N=2048 时为空张量 `(0,256)`。数据格式为 ND。 | bfloat16 | (M, 256) 或 (0, 256) |

`k_cache` 和 `v_cache` 为原地更新，不包含在返回值中。

## 约束说明

- 该接口支持训练、推理场景下使用。
- 该接口支持单算子模式和 torch.compile（npugraph_ex）图模式调用。
- 图模式（ACL Graph）捕获期间不执行依赖设备值回读的参数校验（`positions` 连续性/范围、`slot_mapping` 范围与唯一性），以避免同步正在捕获的 stream；调用方必须保证捕获与回放时索引始终合法。单算子模式仍执行全部校验。
- 仅支持 Ascend 950PR/950DT（ARCH 3510）。
- `M >= 1`；`hidden`、`weight`、`k_gamma`、`cos_sin_cache` 必须连续，且位于同一 NPU。
- `weight` 的 N 仅支持 2048（普通 QKV 层）或 2560（kv-mirror 层）。
- `k_cache` / `v_cache` 必须连续，最后一维为 256，可展平为 `(slots,256)`；两者 shape 必须一致。
- `positions_contiguous=True` 且 `M > 1` 时，要求 `positions` 严格连续；`positions` 取值范围为 `[0, P-1]`。
- `slot_mapping` 取值范围为 `[-1, slots-1]`；-1 表示跳过散写，非负槽位必须唯一，重复槽位会被拒绝。
- 当前只覆盖 BF16 权重路径，不包含 modelslim-mxfp8 权重与 E8M0 scale 路径。
- 当前提供前向实现，不提供反向。

## 确定性/Batch一致性

- 确定性说明：默认支持确定性计算。
- Batch一致性说明：默认支持Batch一致性。

## 调用示例

- 单算子模式调用

    ```python
    import torch
    import torch_npu
    from cann_ops_transformer.ops import fused_qkv_proj_norm_rope_cache

    torch.npu.set_device(0)
    device = torch.device("npu:0")
    tokens, units, max_pos, slots = 72, 10, 4096, 1024

    # 构造投影输入、权重、RoPE 表与分页 cache
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

- 图模式调用

    ```python
    import torch
    import torch_npu
    from cann_ops_transformer.ops import fused_qkv_proj_norm_rope_cache

    class QkvModel(torch.nn.Module):
        def forward(self, hidden, weight, k_gamma, positions, cos_sin_cache,
                    slot_mapping, k_cache, v_cache):
            return fused_qkv_proj_norm_rope_cache(
                hidden, weight, k_gamma, positions, cos_sin_cache, slot_mapping,
                k_cache, v_cache, eps=1e-6, positions_contiguous=True,
            )

    # 图模式调用，cache 为原地更新
    model = torch.compile(
        QkvModel().eval(), backend="npugraph_ex", fullgraph=True, dynamic=False,
    )
    out = model(hidden, weight, k_gamma, positions, cos_sin_cache,
                slot_mapping, k_cache, v_cache)
    ```
