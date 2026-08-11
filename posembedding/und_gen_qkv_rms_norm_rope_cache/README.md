# UndGenQkvRmsNormRopeCache

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                 |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term> |      ×     |
| <term>Atlas 推理系列产品</term> |      ×     |
| <term>Atlas 训练系列产品</term> |      ×     |

## 功能说明

- 算子功能：多模态大模型推理prefill/decode阶段的Q/K/V主干融合算子。把undecoded段（und_qkv）与
  generated段（gen_qkv）按`cat_indices`间接寻址拼成输出序列，逐token完成：

  1. **Split**：按N维把每个token拆成Q[Hq, D] / K[Hk, D] / V[Hv, D]；
  2. **RMSNorm**：Q/K分别用und/gen两套权重（按`src_t < und_len`选择）做归一化；
  3. **MRoPE**：按`positions[3, T]`取三轴（时间/高度/宽度）cos_sin，依`mrope_section`合并成一份后
     对Q/K做标准RoPE；V不参与norm/rope；
  4. **输出**：Q作为独立输出；K/V经Cast float32→bf16后按`slot_mapping`写入分页KV Cache。

- 输入输出支持以下数据场景：

    ```text
    und_qkv:       [und_len, N, D]           N = Hq + Hk + Hv
    gen_qkv:       [gen_len, N, D]           当前必填，gen_len > 0
    und_weights_q: [D]   und_weights_k: [D]
    gen_weights_q: [D]   gen_weights_k: [D]  可选，与 gen_qkv 同进同出
    cos_sin_cache: [max_pos, D]              前半 cos、后半 sin
    slot_mapping:  [T]                       slot = block_idx * Bs + row_idx
    positions:     [3, T]
    cat_indices:   [T]                       当前必填（不传的恒等映射路径暂不支持）
    k_cache:       [Bn, Bs, Hk, D]           调用方预分配，原地写入
    v_cache:       [Bn, Bs, Hv, D]           调用方预分配，原地写入
    q:             [T, Hq, D]                T = und_len + gen_len
    ```

- **KV Cache布局固定为连续BBND**（`[Bn, Bs, N, D]`），不支持非连续/BNBD物理布局。
- 索引类张量（`slot_mapping` / `positions` / `cat_indices`）统一int64。
- `mrope_section`为空列表时退化为`[D/2, 0, 0]`，此时等价标准RoPE。

## 参数说明

| 参数名 | 输入/输出 | 数据类型 | 数据格式 | 说明 |
| :----- | :-------- | :------- | :------- | :--- |
| und_qkv | 输入 | BFLOAT16 | ND | undecoded段QKV，[und_len, N, D] |
| und_weights_q | 输入 | BFLOAT16 | ND | und段Q的RMSNorm权重，[D] |
| und_weights_k | 输入 | BFLOAT16 | ND | und段K的RMSNorm权重，[D] |
| cos_sin_cache | 输入 | FLOAT32 | ND | cos/sin缓存，[max_pos, D] |
| k_cache | 输入/输出 | BFLOAT16 | ND | K Cache，[Bn, Bs, Hk, D]，原地写入 |
| v_cache | 输入/输出 | BFLOAT16 | ND | V Cache，[Bn, Bs, Hv, D]，原地写入 |
| slot_mapping | 输入 | INT64 | ND | 每个token的cache行偏移，[T] |
| positions | 输入 | INT64 | ND | MRoPE三轴位置，[3, T] |
| gen_qkv | 可选输入（当前必填） | BFLOAT16 | ND | generated段QKV，[gen_len, N, D]，gen_len > 0 |
| gen_weights_q | 可选输入（当前必填） | BFLOAT16 | ND | gen段Q的RMSNorm权重，[D] |
| gen_weights_k | 可选输入（当前必填） | BFLOAT16 | ND | gen段K的RMSNorm权重，[D] |
| cat_indices | 可选输入（当前必填） | INT64 | ND | out_t → src_t映射，[T] |
| q | 输出 | BFLOAT16 | ND | 处理后的Q，[T, Hq, D] |
| num_heads_q | 属性 | INT | - | Q头数Hq |
| num_heads_k | 属性 | INT | - | K头数Hk |
| num_heads_v | 属性 | INT | - | V头数Hv |
| norm_eps | 属性 | FLOAT | - | RMSNorm epsilon，默认1e-6 |
| mrope_section | 属性 | LIST_INT | - | MRoPE三轴分段，默认[] |

## 约束说明

- 仅支持<term>Ascend 950PR/Ascend 950DT</term>（DAV_3510, arch35）。
- 计算全程使用float32，KV Cache以BF16存储，不支持FP8/INT8量化。
- headDim固定为128。
- (Hq, Hk, Hv)支持(8, 1, 1)与(16, 2, 2)两种组合。
- cache block_size不设限：Bs不参与任何寻址与切分（cache强制连续BBND，slot直接当扁平行号用），
  唯一约束是容量。上板已验证Bs ∈ {16, 64, 100, 128, 256, 512}，含非2的幂。
- T = und_len + gen_len不设上限，只要求为正，真实上限是KV Cache容量（要求`Bn * Bs >= T`）。
  上板已验证到T = 64K。
- `slot_mapping`的取值必须唯一且落在[0, Bn*Bs)，重复slot在多核下结果不确定。
- `positions`取值须落在[0, max_pos)、`cat_indices`取值须落在[0, T)。这三类取值属运行期数据，
  Host侧看不到tensor内容、无法校验，由调用方保证。
- 入参校验统一收敛在tiling一层（aclnn L2只查空指针、InferShape只查推导前提），
  因此非法入参不在`aclnnUndGenQkvRmsNormRopeCacheGetWorkspaceSize`阶段报错，
  而是在执行阶段由tiling返回失败。
- **当前实现要求`gen_qkv`、`gen_weights_q`、`gen_weights_k`、`cat_indices`四个可选输入全部提供，
  且`gen_len > 0`**；缺省的退化路径（纯prefill、单序列恒等映射）暂不支持，tiling会直接拦截。
  IR/OpDef保留OPTIONAL声明，后续放开时删掉`CheckSupportRange`中对应校验即可。
- `k_cache`/`v_cache`必须内存连续。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| :------- | :------- | :--- |
| PyTorch API | [test_torch_und_gen_qkv_rms_norm_rope_cache.py](./examples/test_torch_und_gen_qkv_rms_norm_rope_cache.py) | 通过[und_gen_qkv_rms_norm_rope_cache](../../torch_extension/cann_ops_transformer/docs/zh/und_gen_qkv_rms_norm_rope_cache.md)接口方式调用UndGenQkvRmsNormRopeCache算子。 |
| aclnn调用 | [test_aclnn_und_gen_qkv_rms_norm_rope_cache.cpp](./examples/test_aclnn_und_gen_qkv_rms_norm_rope_cache.cpp) | 通过[aclnnUndGenQkvRmsNormRopeCache](./docs/aclnnUndGenQkvRmsNormRopeCache.md)接口方式调用UndGenQkvRmsNormRopeCache算子。两段式调用，样例含slot_mapping预计算。 |
| 图模式调用 | [test_geir_und_gen_qkv_rms_norm_rope_cache.cpp](./examples/test_geir_und_gen_qkv_rms_norm_rope_cache.cpp) | 通过[算子IR](./op_graph/und_gen_qkv_rms_norm_rope_cache_proto.h)构图方式调用UndGenQkvRmsNormRopeCache算子。 |
