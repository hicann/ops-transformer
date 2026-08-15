# Sparse Flash MLA 系列算子拦截说明

## 1. 文档范围

本文档说明以下三个算子新增的 Host 侧参数拦截代码及当前实际拦截链路：

- `sparse_flash_mla`
- `mixed_quant_sparse_flash_mla`
- `quant_sparse_flash_mla`

拦截规则参考：

- `torch_extension/cann_ops_transformer/docs/zh/sparse_flash_mla.md`
- `torch_extension/cann_ops_transformer/docs/zh/mixed_quant_sparse_flash_mla.md`
- `torch_extension/cann_ops_transformer/docs/zh/quant_sparse_flash_mla.md`

代码采用与 `flash_attn/op_host/checkers` 相同的分层形式，将检查过程划分为：

1. 单参数检查（`CheckSinglePara`）
2. 参数存在性检查（`CheckParaExistence`）
3. 特性交叉检查（`CheckFeature`）
4. 多参数一致性检查（`CheckMultiPara`）

三个算子的公共检查代码统一放在本目录；混合量化和全量化算子的目录只保存各自的适配入口和差异规则。
当前新增 Checker 代码均予以保留并参与编译；`mixed_quant_sparse_flash_mla`和
`quant_sparse_flash_mla`已使用新增 Checker；`sparse_flash_mla`按架构分流，DAV_2201
（Atlas A2/A3）使用原有旧 Checker，DAV_3510（Atlas A5）使用新增 Checker。

## 2. 代码结构

```text
attention/
├── sparse_flash_mla/op_host/checkers/
│   ├── base_checker.{h,cpp}                 # Checker 基类及公共检查工具
│   ├── checker_context.h                    # 三算子统一检查上下文
│   ├── checker_adapter.h                    # TilingInfo 到统一上下文的适配
│   ├── checker_runner.{h,cpp}               # 四阶段检查编排
│   ├── common_checker.{h,cpp}               # Q/KV/输出、布局和公共属性
│   ├── seq_len_checker.{h,cpp}              # 序列长度类 Tensor
│   ├── sparse_compression_checker.{h,cpp}   # 稀疏索引、TopK 和压缩参数
│   ├── mask_checker.{h,cpp}                 # Mask 与窗口联动
│   ├── paged_attention_checker.{h,cpp}      # Paged Attention
│   ├── sinks_checker.{h,cpp}                # Sinks
│   ├── metadata_checker.{h,cpp}             # Metadata
│   ├── softmax_lse_checker.{h,cpp}          # Softmax LSE 输出
│   ├── sparse_flash_mla_checker.{h,cpp}     # sparse_flash_mla 入口
│   └── checker_sources.cmake                # 公共源码一次性注册
├── mixed_quant_sparse_flash_mla/op_host/checkers/
│   ├── mixed_quant_variant_checker.{h,cpp}  # 混合量化特有规则
│   └── mixed_quant_sparse_flash_mla_checker.{h,cpp}
└── quant_sparse_flash_mla/op_host/checkers/
    ├── quant_variant_checker.{h,cpp}        # 全量化及 descale 规则
    └── quant_sparse_flash_mla_checker.{h,cpp}
```

`RegisterCommonCheckers` 按以下顺序注册公共 Checker：

```text
Common
  → SeqLen
  → SparseCompression
  → Mask
  → PagedAttention
  → Sinks
  → Metadata
  → SoftmaxLse
```

混合量化和全量化入口会在公共 Checker 后追加各自的差异 Checker；当前 M/Q 均已启用该执行链。
`sparse_flash_mla`在 DAV_3510 上执行上述公共 Checker，DAV_2201 继续执行原有旧 Checker。

任一阶段、任一 Checker 返回失败后立即停止，Tiling 返回 `GRAPH_FAILED`。

## 3. 新增 Checker 公共拦截

### 3.1 Tensor 通用要求

| 检查项 | 拦截规则 |
| --- | --- |
| 输入/输出存在性 | `q`、`ori_kv`、`attention_out`、`sinks`、`metadata`必须存在；`cmp_kv`和`softmax_lse`可选 |
| 数据格式 | 所有被检查的 Tensor 仅支持 `ND` |
| 空 Tensor | Q、KV、输出、索引、长度、Block Table、Sinks 等任一维度小于等于0时拦截 |
| Q 布局 | 仅支持 `BSND`、`TND` |
| KV 布局 | 仅支持 `BSND`、`TND`、`PA_BBND` |
| 布局组合 | 非 `PA_BBND` 时，`layout_q`与`layout_kv`必须相同；PA 时 Q 可为 `BSND`或`TND` |
| Q/输出一致性 | `attention_out`的 rank 和 shape 必须与`q`完全相同 |
| Q 轴范围 | `q_n`在 `[1, 128]`内，Q head dim 固定为512，batch/sequence/token维均大于0 |
| KV 轴范围 | `kv_n`固定为1，序列、token、block 数和 block size 均必须有效 |
| Softmax scale | `softmax_scale`必须为有限值，拒绝 NaN 和 Inf |

### 3.2 三算子数据类型和 Head Dim 差异

| 算子 | q | ori_kv/cmp_kv | attention_out | KV head dim |
| --- | --- | --- | --- | --- |
| `sparse_flash_mla` | FP16/BF16 | FP16/BF16 | FP16/BF16 | 512 |
| `mixed_quant_sparse_flash_mla` | BF16 | FP8_E4M3FN | BF16 | `quant_mode=1`时608；`quant_mode=2`时584 |
| `quant_sparse_flash_mla` | HIFLOAT8 | HIFLOAT8 | BF16 | 512 |

`sparse_flash_mla`还要求`q`、所有非空 KV 和`attention_out`的数据类型完全一致。

在 Atlas A2/A3 上，`sparse_flash_mla`的`q_n`仅支持：

```text
1, 2, 4, 8, 16, 32, 64, 128
```

Ascend 950 上只检查`1 <= q_n <= 128`。

### 3.3 序列长度 Tensor

以下 Tensor 均要求 `int32`、`ND`、一维且非空：

- `cu_seqlens_q`
- `cu_seqlens_ori_kv`
- `cu_seqlens_cmp_kv`
- `seqused_q`
- `seqused_ori_kv`
- `seqused_cmp_kv`
- `cmp_residual_kv`

存在性和 shape 规则如下：

| 参数 | 存在性 | shape |
| --- | --- | --- |
| `cu_seqlens_q` | Q 为 TND 时必传；其他布局禁止传入 | `(b+1,)` |
| `cu_seqlens_ori_kv` | KV 为 TND 时必传；其他布局禁止传入 | `(b+1,)` |
| `cu_seqlens_cmp_kv` | KV 为 TND且`cmp_kv`存在时必传；无`cmp_kv`时禁止传入 | `(b+1,)` |
| `seqused_q` | 可选 | `(b,)` |
| `seqused_ori_kv` | PA 场景通常必传；`ORI_SPARSE`中ori侧`mask_mode=0`，或`ORI_CMP_SPARSE`中两侧`mask_mode=0`，且传入`ori_topk_length`时可不传 | `(b,)` |
| `seqused_cmp_kv` | PA 且`cmp_kv`存在时通常必传；仅`ORI_CMP_SPARSE`中两侧`mask_mode=0`且传入`cmp_topk_length`时可不传 | `(b,)` |
| `cmp_residual_kv` | 由压缩模式和 Mask 联动决定 | `(b,)` |

Tensor 内部数值（例如累积长度单调性、首尾值及 residual 范围）在 Tiling 阶段不可读取，由调用方保证。

### 3.4 稀疏索引、TopK Length 与压缩属性

公共规则：

- `cmp_ratio > 0`。
- SWA（未传`cmp_kv`）要求`cmp_ratio=1`。
- `topk_value_mode`当前只支持1。
- `cmp_sparse_indices`、`cmp_topk_length`、`cmp_residual_kv`以及 cmp 侧长度/Block Table 均依赖`cmp_kv`。
- 稀疏索引仅支持 `int32`、`ND`：
  - BSND：`(b, q_s, kv_n, topk)`
  - TND：`(q_t, kv_n, topk)`
- TopK Length 仅支持 `int32`、`ND`：
  - BSND：`(b, q_s, kv_n)`
  - TND：`(q_t, kv_n)`

三个算子统一采用以下成对规则：

- `ori_mask_mode=0`且传入`ori_sparse_indices`时，必须传`ori_topk_length`；其他情况禁止传`ori_topk_length`。
- `cmp_mask_mode=0`且传入`cmp_sparse_indices`时，必须传`cmp_topk_length`；其他情况禁止传`cmp_topk_length`。

全稀疏分为两种模式：

- `ORI_SPARSE`：存在`ori_sparse_indices`且不存在`cmp_kv`，只检查ori侧稀疏参数；ori侧`mask_mode=0`时，
  `ori_topk_length`必须传入，并可替代`seqused_ori_kv`。
- `ORI_CMP_SPARSE`：`ori_sparse_indices`、`cmp_kv`和`cmp_sparse_indices`同时存在，检查ori、cmp两侧
  稀疏参数；两侧`mask_mode=0`时，`ori_topk_length`和`cmp_topk_length`必须传入，并可分别替代
  `seqused_ori_kv`和`seqused_cmp_kv`。

因此，`sparse_flash_mla`与另外两个算子一样，允许在`ori_mask_mode=0`时传入`ori_sparse_indices`及配套的`ori_topk_length`。

### 3.5 Mask 和窗口

| 参数 | 支持范围 |
| --- | --- |
| `ori_mask_mode` | 0、3、4 |
| `cmp_mask_mode` | 0、3 |
| `ori_win_left`、`ori_win_right` | -1或非负数 |

交叉规则：

- 非 Sliding Window（`ori_mask_mode != 4`）要求`ori_win_left=ori_win_right=-1`。
- `ori_mask_mode=4`时才允许使用非负窗口值。
- SWA 要求`cmp_mask_mode=0`。
- `sparse_flash_mla`的 HCA/CSA 要求`cmp_mask_mode=3`。
- 混合量化算子的 Causal/Sliding Window 模式（`ori_mask_mode=3/4`）在存在`cmp_kv`时要求`cmp_mask_mode=3`。
- 三个算子传入`ori_sparse_indices`时，均要求`ori_mask_mode=cmp_mask_mode=0`。
- Atlas A2/A3 上的`sparse_flash_mla`固定要求`ori_win_left=127`、`ori_win_right=0`。

### 3.6 Paged Attention

| 检查项 | 拦截规则 |
| --- | --- |
| Block Table dtype/format | `int32`、`ND` |
| Block Table rank | 2 |
| Block Table shape | 第一维必须等于 batch，第二维必须大于0 |
| 非 PA 布局 | 禁止传入`ori_block_table`和`cmp_block_table` |
| PA 布局 | 必须传`ori_block_table`；`cmp_block_table`与`cmp_kv`同时存在或同时不存在 |
| Seqused 联动 | Block Table 通常要求对应`seqused_*_kv`；`ORI_SPARSE`可用`ori_topk_length`替代ori侧，`ORI_CMP_SPARSE`可用对应`topk_length`分别替代两侧，且需满足各模式的`mask_mode=0`约束 |
| KV block size | 范围 `[1, 1024]` |
| A2/A3 sparse block size | 额外要求16对齐 |

Block Table 内的具体 block id 值由调用方保证。

### 3.7 Sinks、Metadata 和 Softmax LSE

| 参数 | 拦截规则 |
| --- | --- |
| `sinks` | 当前版本必传；`float32`、`ND`、shape为`(q_n,)` |
| `metadata` | 当前版本必传；`int32`、`ND`、shape为`(1024,)` |
| `softmax_lse` | 可存在或不存在；仅在`return_softmax_lse=true`且存在时要求`float32`、`ND` |

`return_softmax_lse=false`或`softmax_lse`不存在时，跳过该参数的所有检查。仅在
`return_softmax_lse=true`且`softmax_lse`存在时检查以下 shape：

| 条件 | shape |
| --- | --- |
| `return_softmax_lse=true`且 Q 为 BSND | `(b, kv_n, q_s, q_n/kv_n)` |
| `return_softmax_lse=true`且 Q 为 TND | `(kv_n, q_t, q_n/kv_n)` |

启用并传入`softmax_lse`时，同时检查`q_n`可被`kv_n`整除。

## 4. 逐参数校验明细

下表中的适用范围使用以下缩写：

- S：`sparse_flash_mla`
- M：`mixed_quant_sparse_flash_mla`
- Q：`quant_sparse_flash_mla`

“检查内容”描述新增 Checker 已实现的 Host 侧检查规则；当前 M/Q 和 DAV_3510 上的 S 已实际执行
这些规则，DAV_2201 上的 S 仍执行旧 Checker。Tensor 内部元素值无法在 Tiling 阶段读取的，会
单独标明“值由用户保证”。

### 4.1 核心输入和量化参数

| 参数 | 适用 | 存在性 | dtype、format、rank/shape | 一致性及交叉检查 |
| --- | --- | --- | --- | --- |
| `q` | S/M/Q | 必传；desc 和 shape 均不可为空 | S：FP16/BF16；M：BF16；Q：HIFLOAT8。仅支持ND；BSND为4维，TND为3维；任一维度必须大于0 | `q_d=512`；`1 <= q_n <= 128`；S在A2/A3上`q_n`仅支持1、2、4、8、16、32、64、128；shape必须与`attn_out`相同；布局与`layout_q`一致 |
| `ori_kv` | S/M/Q | 当前版本必传 | S：FP16/BF16；M：FP8_E4M3FN；Q：HIFLOAT8。仅支持ND；TND为3维，BSND/PA_BBND为4维；任一维度必须大于0 | `kv_n=1`；S/Q的head dim为512；M在`quant_mode=1/2`时分别为608/584；BSND的batch必须等于Q的batch；PA block size在`[1,1024]`内，S在A2/A3上还要求16对齐 |
| `cmp_kv` | S/M/Q | 可选 | 存在时执行与`ori_kv`相同的dtype、ND、rank、非空和轴检查 | 不存在时禁止传入所有cmp侧从属Tensor，并要求`cmp_ratio=1`、`cmp_mask_mode=0`；存在时BSND batch必须与Q一致；S要求其dtype与`q`、`ori_kv`一致 |
| `q_descale` | Q | 当前版本必传 | `float32`、ND、shape为`(1,)` | 仅在`quant_mode=1`下支持；当前Q算子只支持该模式 |
| `ori_kv_descale` | Q | 当前版本必传 | `float32`、ND、shape为`(1,)` | 与`ori_kv`配套使用 |
| `cmp_kv_descale` | Q | `cmp_kv`存在时必传；`cmp_kv`不存在时禁止传入 | `float32`、ND、shape为`(1,)` | 存在状态必须与`cmp_kv`完全一致 |
| `sinks` | S/M/Q | 当前版本必传 | `float32`、ND、1维、非空 | shape必须为`(q_n,)`，长度等于Q头数 |
| `metadata` | S/M/Q | 当前版本必传 | `int32`、ND、shape为`(1024,)` | Checker只能检查描述信息，无法验证其内容是否由本次调用的同一组参数生成 |

### 4.2 稀疏压缩参数

| 参数 | 适用 | 单参数及shape检查 | 存在性和交叉检查 | 无法检查的内容 |
| --- | --- | --- | --- | --- |
| `ori_sparse_indices` | S/M/Q | `int32`、ND、非空；BSND为`(b,q_s,kv_n,topk)`，TND为`(q_t,kv_n,topk)` | 三个算子均可选；传入时要求`ori_mask_mode=cmp_mask_mode=0`，且必须按规则传入`ori_topk_length` | 每个元素是否为-1或合法ori token索引 |
| `cmp_sparse_indices` | S/M/Q | `int32`、ND、非空；BSND为`(b,q_s,kv_n,topk)`，TND为`(q_t,kv_n,topk)`；最后一维必须大于0 | 依赖`cmp_kv`，无`cmp_kv`时禁止传入；S中传入后识别为CSA，A2/A3上TopK仅支持512或1024；M/Q在`cmp_mask_mode=0`时要求`cmp_topk_length` | 每个元素是否为-1或合法cmp token索引 |
| `ori_topk_length` | S/M/Q | `int32`、ND、非空；BSND为`(b,q_s,kv_n)`，TND为`(q_t,kv_n)` | 三个算子规则相同：仅当`ori_sparse_indices`存在且`ori_mask_mode=0`时必传，其他情况禁止传入 | 每个位置的TopK长度是否小于等于索引最后一维 |
| `cmp_topk_length` | S/M/Q | `int32`、ND、非空；BSND为`(b,q_s,kv_n)`，TND为`(q_t,kv_n)` | 三个算子规则相同：仅当`cmp_kv`、`cmp_sparse_indices`存在且`cmp_mask_mode=0`时必传，其他情况禁止传入 | 每个位置的TopK长度是否有效 |
| `cmp_residual_kv` | S/M/Q | `int32`、ND、1维、非空，shape为`(b,)` | 无`cmp_kv`时禁止传入；S的HCA/CSA必传；M/Q在`cmp_mask_mode=3`且`cmp_ratio!=1`时必传 | 每个元素是否位于`[0,cmp_ratio)`，以及压缩前后长度恢复关系 |

### 4.3 序列长度参数

| 参数 | 适用 | 单参数及shape检查 | 存在性和布局联动 | 无法检查的内容 |
| --- | --- | --- | --- | --- |
| `cu_seqlens_q` | S/M/Q | `int32`、ND、1维、非空，shape为`(b+1,)` | `layout_q=TND`时必传；其他Q布局禁止传入 | 首元素是否为0、是否单调非递减、末元素是否等于`q_t` |
| `cu_seqlens_ori_kv` | S/M/Q | `int32`、ND、1维、非空，shape为`(b+1,)` | `layout_kv=TND`时必传；其他KV布局禁止传入 | 首元素、单调性、末元素和ori KV实际长度 |
| `cu_seqlens_cmp_kv` | S/M/Q | `int32`、ND、1维、非空，shape为`(b+1,)` | `layout_kv=TND`且`cmp_kv`存在时必传；其他情况禁止传入 | 首元素、单调性、末元素和cmp KV实际长度 |
| `seqused_q` | S/M/Q | 存在时要求`int32`、ND、1维、非空，shape为`(b,)` | 可选，不作为PA必选参数 | 每个元素是否非负且不超过对应Q长度 |
| `seqused_ori_kv` | S/M/Q | 存在时要求`int32`、ND、1维、非空，shape为`(b,)` | BSND等非PA场景可选；PA场景通常必传；`ORI_SPARSE`或`ORI_CMP_SPARSE`满足对应`mask_mode=0`约束并传入`ori_topk_length`时可不传 | 每个元素是否非负且不超过对应ori KV长度 |
| `seqused_cmp_kv` | S/M/Q | 存在时要求`int32`、ND、1维、非空，shape为`(b,)` | 无`cmp_kv`时禁止传入；PA且`cmp_kv`存在时通常必传；仅`ORI_CMP_SPARSE`中两侧`mask_mode=0`且传入`cmp_topk_length`时可不传 | 每个元素是否非负且不超过对应cmp KV长度 |

### 4.4 Paged Attention 参数

| 参数 | 适用 | 单参数及shape检查 | 存在性和交叉检查 | 无法检查的内容 |
| --- | --- | --- | --- | --- |
| `ori_block_table` | S/M/Q | `int32`、ND、2维、非空；第一维必须等于batch | `layout_kv=PA_BBND`时必传，非PA禁止传入；通常要求`seqused_ori_kv`，`ORI_SPARSE`或`ORI_CMP_SPARSE`满足对应`mask_mode=0`约束时可用`ori_topk_length`替代 | block id是否为正整数、是否越界，以及第二维是否覆盖全部有效KV |
| `cmp_block_table` | S/M/Q | `int32`、ND、2维、非空；第一维必须等于batch | 仅在`layout_kv=PA_BBND`且`cmp_kv`存在时必传；其存在状态必须与`cmp_kv`一致；通常要求`seqused_cmp_kv`，仅`ORI_CMP_SPARSE`中两侧`mask_mode=0`时可用`cmp_topk_length`替代 | block id是否合法，以及第二维是否覆盖全部有效cmp KV |

`ori_kv`和`cmp_kv`自身的PA block size检查包含：范围`[1,1024]`；S在Atlas A2/A3上额外要求16对齐。

### 4.5 属性参数

| 参数 | 适用 | 校验内容 |
| --- | --- | --- |
| `softmax_scale` | S/M/Q | 必须是有限浮点数；NaN、正负Inf均拦截 |
| `cmp_ratio` | S/M/Q | 必须大于0；无`cmp_kv`的SWA固定为1；S在A2/A3上的CSA固定为4、HCA固定为128；与`cmp_residual_kv`的逐元素数值关系由用户保证 |
| `ori_mask_mode` | S/M/Q | 仅支持0、3、4；决定窗口和稀疏索引联动；M中存在`cmp_kv`且取3/4时要求`cmp_mask_mode=3`；存在`ori_sparse_indices`时必须为0 |
| `cmp_mask_mode` | S/M/Q | 仅支持0、3；SWA固定为0；S不含`ori_sparse_indices`的HCA/CSA固定为3；三个算子存在`ori_sparse_indices`时必须为0；影响TopK Length和Residual的存在性 |
| `ori_win_left` | S/M/Q | 只能为-1或非负数；非Sliding Window要求为-1；S在A2/A3上固定为127 |
| `ori_win_right` | S/M/Q | 只能为-1或非负数；非Sliding Window要求为-1；S在A2/A3上固定为0 |
| `layout_q` | S/M/Q | 仅支持`BSND`、`TND`；决定Q、稀疏索引、TopK Length和LSE的rank/shape，以及`cu_seqlens_q`存在性 |
| `layout_kv` | S/M/Q | 仅支持`BSND`、`TND`、`PA_BBND`；非PA时必须与`layout_q`相同；决定KV rank、KV长度Tensor和Block Table存在性 |
| `topk_value_mode` | S/M/Q | 当前仅支持1，其他值直接拦截 |
| `return_softmax_lse` | S/M/Q | bool类型由算子Schema保证；为`false`时完全跳过`softmax_lse`检查；为`true`且输出存在时检查有效LSE dtype、format和shape；输出不存在时不拦截 |
| `quant_mode` | M/Q | M必须存在且仅支持1、2，并决定KV head dim为608或584；Q仅支持1 |
| `rope_head_dim` | M | 仅支持64 |

### 4.6 输出参数

| 参数 | 适用 | 存在性 | dtype、format、rank/shape | 一致性检查 |
| --- | --- | --- | --- | --- |
| `attn_out`（文档中也称`attention_out`） | S/M/Q | 必须存在 | S：FP16/BF16；M/Q：BF16；仅支持ND；rank随`layout_q`为4或3；任一维度必须大于0 | shape必须与`q`完全一致；S还要求dtype与`q`一致 |
| `softmax_lse` | S/M/Q | 可选；存在或不存在均合法 | `return_softmax_lse=false`时不检查；开启且存在时要求`float32`、ND，BSND shape为`(b,kv_n,q_s,q_n/kv_n)`，TND shape为`(kv_n,q_t,q_n/kv_n)` | 关闭或不存在时跳过全部检查；存在且开启时要求`q_n`能被`kv_n`整除，所有相关轴与Q/KV一致；输出内容不在Host侧检查 |

### 4.7 前置 Metadata 接口参数说明

本文新增 Checker 设计为运行在三个主算子的 Tiling 入口，启用后仅接收前置Metadata接口生成的
`metadata` Tensor；当前 M/Q 和 DAV_3510 上的 S 已接入实际入口，DAV_2201 上的 S 仍走旧
Checker。因此：

- `*_sparse_flash_mla_metadata`接口中的`num_heads_q`、`num_heads_kv`、`head_dim`、`batch_size`、`max_seqlen_*`、`has_ori_kv`和`has_cmp_kv`等参数，不会在主算子 Checker 中再次逐项读取。
- 主算子 Checker只验证`metadata`为`int32`、ND、shape `(1024,)`。
- Metadata是否由与主算子完全一致的布局、序列长度、压缩率、Mask、窗口和KV存在状态生成，只能由调用方保证。

## 5. 算子特有拦截

### 5.1 sparse_flash_mla

DAV_2201（arch22/Atlas A2/A3）由原有`SMLATilingCheck`执行；DAV_3510（Atlas A5）由新增
`SparseFlashMlaChecker`执行。

按输入组合识别计算模式：

| 模式 | 输入组合 | 公共约束 | Atlas A2/A3附加约束 |
| --- | --- | --- | --- |
| SWA | 仅`ori_kv` | `cmp_ratio=1`、`cmp_mask_mode=0` | `cmp_ratio=1` |
| ORI_SPARSE | `ori_kv + ori_sparse_indices + ori_topk_length` | `ori_mask_mode=cmp_mask_mode=0`、`cmp_ratio=1` | `cmp_ratio=1` |
| HCA | `ori_kv + cmp_kv`，无`cmp_sparse_indices` | `cmp_mask_mode=3`、必须有`cmp_residual_kv` | `cmp_ratio=128` |
| CSA | `ori_kv + cmp_kv + cmp_sparse_indices` | `cmp_mask_mode=3`、必须有`cmp_residual_kv` | `cmp_ratio=4`，TopK仅支持512或1024 |
| ORI_CMP_SPARSE | `ori_kv + ori_sparse_indices + ori_topk_length + cmp_kv + cmp_sparse_indices + cmp_topk_length` | `ori_mask_mode=cmp_mask_mode=0` | `cmp_ratio=4`，cmp TopK仅支持512或1024 |

Ascend 950 的 HCA/CSA 仅要求`cmp_ratio > 0`，CSA TopK 要求大于0。

### 5.2 mixed_quant_sparse_flash_mla

当前由新增`MixedQuantSparseFlashMlaChecker`执行。

特有属性：

- `quant_mode`仅支持1或2。
- `rope_head_dim`仅支持64。
- Q/输出仅支持 BF16。
- KV 仅支持 FP8_E4M3FN。
- `quant_mode=1`时 KV head dim 为608。
- `quant_mode=2`时 KV head dim 为584。
- 当`cmp_mask_mode=3`且`cmp_ratio != 1`时，必须传入`cmp_residual_kv`。

### 5.3 quant_sparse_flash_mla

当前由新增`QuantSparseFlashMlaChecker`执行。

特有属性和 Tensor：

- `quant_mode`仅支持1。
- Q、ori KV、cmp KV 仅支持 HIFLOAT8。
- 输出仅支持 BF16。
- `q_descale`当前版本必传。
- `ori_kv_descale`当前版本必传。
- `cmp_kv_descale`与`cmp_kv`同时存在或同时不存在。
- 三个 descale Tensor 均要求`float32`、`ND`、shape为`(1,)`。
- 当`cmp_mask_mode=3`且`cmp_ratio != 1`时，必须传入`cmp_residual_kv`。

## 6. 新旧拦截链路关系

当前拦截入口如下：

| 算子/架构 | 实际拦截类 |
| --- | --- |
| `sparse_flash_mla` / DAV_2201（Atlas A2/A3） | 原有`SMLATilingCheck` |
| `sparse_flash_mla` / DAV_3510（Atlas A5） | 新增`SparseFlashMlaChecker` |
| `mixed_quant_sparse_flash_mla` | 新增`MixedQuantSparseFlashMlaChecker` |
| `quant_sparse_flash_mla` | 新增`QuantSparseFlashMlaChecker` |

SMLA 解析完成后根据`npuArch`实例化 Checker：DAV_2201 使用旧 Checker，DAV_3510 使用新增
Checker；MQSMLA 和 QSMLA 继续实例化新增 Checker。CMake 不额外添加 M/Q 的旧
`*_check*.cpp`，新增 Checker 中不再保留 arch22/A2-A3 专用规则，原`sparse_variant_checker`
也不再编译。

解析器仍负责从 Tiling Context 获取 Tensor、属性、布局、shape 和硬件信息；解析完成后执行对应 Checker。

## 7. 检查边界

Host Tiling 阶段只检查可获得的描述信息和属性值，包括：

- Tensor 是否存在
- dtype、format、rank、shape
- 布局和硬件平台
- 标量属性范围
- 参数之间的存在性和形状联动

以下内容无法在当前阶段完整读取或验证，由调用方保证：

- `cu_seqlens_*`的首元素、末元素及单调性
- `seqused_*`的逐元素范围
- `cmp_residual_kv`每个元素是否位于`[0, cmp_ratio)`
- 稀疏索引是否为-1或合法 token 索引
- TopK Length 的逐元素范围
- Block Table 内的 block id 是否有效
- `metadata`是否由完全一致的前置参数生成

## 8. 拦截日志规范

三个算子的新增 Checker 禁止使用通用的 `OP_LOGE`，错误日志按照失败原因选择结构化宏：

| 错误类别 | 使用的日志宏 |
| --- | --- |
| 参数缺失、参数不应存在或参数存在性联动错误 | `OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON` |
| 单个/多个属性值或取值范围错误 | `OP_LOGE_FOR_INVALID_VALUE*`、`OP_LOGE_FOR_INVALID_VALUES_WITH_REASON` |
| dtype 错误或多个 Tensor dtype 不一致 | `OP_LOGE_FOR_INVALID_DTYPE*`、`OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON` |
| Tensor format 错误 | `OP_LOGE_FOR_INVALID_FORMAT` |
| dim num（rank）错误 | `OP_LOGE_FOR_INVALID_SHAPEDIM` |
| shape、轴长度或 shape size 错误 | `OP_LOGE_FOR_INVALID_SHAPE*`、`OP_LOGE_FOR_INVALID_SHAPES*`、`OP_LOGE_FOR_INVALID_SHAPESIZE*` |

日志同时提供参数名、实际值/形状以及正确值或失败原因，便于直接定位拦截条件；所有失败原因和说明文本均以大写字母开头。
