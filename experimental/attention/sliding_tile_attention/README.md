# SlidingTileAttention

## 产品支持情况

| 产品 | 是否支持 |
| :-- | :--: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | 未验证 |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> | √ |

## 功能说明

SlidingTileAttention 面向视频生成模型中的长序列注意力。输入视频 token 按固定的
`(6, 8, 8)` 三维 tile 划分，每个图像 query 仅访问本 head 滑动窗口内的图像 KV；
存在文本时，图像 query 还访问全部有效文本 KV，文本区 query 则访问全部图像 KV
和有效文本 KV。接口语义与 FastVideo 的 `sliding_tile_attention` 对齐。

对每个 batch、head 和 query 位置 $i$：

$$
s_{i,j}=\frac{Q_iK_j^T}{\sqrt{D}},\qquad
O_i=\sum_{j\in\mathcal{V}(i)}\operatorname{Softmax}(s_i)_jV_j
$$

其中 $\mathcal{V}(i)$ 是 query 的可见 KV 集合。算子采用分块 online softmax，
score、softmax 状态和输出累加均使用 FLOAT32，不生成完整的 $S\times S$ 矩阵。

## 参数说明

以下为 aclnn 接口 `aclnnSlidingTileAttentionGetWorkspaceSize` 的参数，完整说明见
[docs/aclnnSlidingTileAttention.md](docs/aclnnSlidingTileAttention.md)。

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 |
| -- | -- | -- | -- | -- |
| query | 输入 | Query，shape 为 `[B, N, S, D]`。 | FLOAT16、BFLOAT16 | ND（BNSD） |
| key | 输入 | Key，shape、dtype 与 query 一致。 | FLOAT16、BFLOAT16 | ND（BNSD） |
| value | 输入 | Value，shape、dtype 与 query 一致。 | FLOAT16、BFLOAT16 | ND（BNSD） |
| output | 输出 | Attention 结果，shape、dtype 与 query 一致。 | FLOAT16、BFLOAT16 | ND（BNSD） |
| windowSize / windowSizeLen | 输入 | 三维窗口 `(Kt, Kh, Kw)` 数组；`windowSizeLen` 为 1 时广播到所有 head，否则等于 N。 | INT64 | - |
| textLength | 输入 | 有效文本 KV 数；`hasText=false` 时按 0 处理。 | INT64 | - |
| hasText | 输入 | 是否包含文本区。 | BOOL | - |
| seqShape | 输入 | 图像 token 画布，取值见约束说明。 | 字符串 | - |

内部实现：按头维 D 选择 kernel，头维不等于 kernel 头维时先在头维尾部补零、算完再切回原头维
（补零列不改变任何 `Q·K` 的值），score 缩放系数固定为 `1/sqrt(D)`（D 为原始头维）。

| D | 执行方式 |
| -- | -- |
| 1～32 | 本算子的 MIX kernel，补零到 32（D=16 的 cube 加载比 D=32 慢数倍） |
| 33～48 | 本算子的 MIX kernel，补零到 48 |
| 49～64 | 补零到 64；序列不超过 892 个 384-token 块（block mask 不超过 760 KB）时调用 ops-transformer 的 BlockSparseAttention，否则用本算子的 MIX kernel |
| 65～128 | 补零到 128，调用 BlockSparseAttention |

本算子的 MIX kernel 数据流与 BlockSparseAttention 的 A2 kernel 相同，但不使用 block mask：host 只下发每个 head
的窗口（单位为 tile）和 tile 网格，向量核按窗口直接写出每个 `(head, query tile)` 的 KV tile 列表，所有 batch
共用，内存和耗时随选中的 tile 数增长，而不是随 tile 数的平方增长。补零时 kernel 会改写这份补零副本
（Q 预乘 score 缩放系数，V 的补零列填 1 以得到行概率和），不修改调用者的输入。

## 约束说明

- `query/key/value/output` 均为四维 BNSD Tensor，shape 和 dtype 完全一致，且
  `B/N/S/D` 均大于 0。
- `1 <= D <= 128`（16 的倍数以外的头维同样支持）；`B`、`N` 没有额外上限，受 device 内存限制。
- `seqShape` 为 `"TxHxW"` 形式的图像 token 画布（`x` 也可写作 `X` 或 `×`），`T/H/W` 均为 `[1, 200]` 内的
  十进制整数，且 `T` 能被 6 整除、`H/W` 能被 8 整除（按 `6×8×8` 划分 tile），否则返回参数错误。
  图像 token 数为 `T*H*W`，tile 网格为 `T/6 × H/8 × W/8`。ATK 用例中的 `"1"`、`"2"`、`"3"` 分别表示
  `"30x48x80"`、`"36x48x48"`、`"18x48x80"`。
- `hasText=false` 时，`S` 必须等于图像 token 数。
- `hasText=true` 时，`0 <= textLength <= 384`，且
  `imageSeqLen + textLength <= S <= imageSeqLen + 384`（文本区为一个 384 token 的块，图像 token 数本身总是
  384 的倍数，因此 FastVideo 把序列补齐到 384 倍数的用法也在范围内）。图像区后的前
  `textLength` 个 token 是有效文本 KV，其余预留 token 仅作为 text query 输出，
  不参与 KV softmax。
- `windowSizeLen` 只能为 1 或 N。每个窗口包含 3 个正整数（单位为 tile）；与 FastVideo 一致，窗口以 query
  tile 为中心、每侧 `k/2` 个 tile，因此偶数 `k` 实际覆盖 `k+1` 个 tile；超过对应 tile grid 的窗口维度按
  grid 截断，边界处窗口整体平移以保持有效范围。
- `query/key/value` 支持非连续 Tensor，L2 接口会转换为连续存储。
- 算子为确定性实现：相同输入和属性在相同软硬件环境下多次执行得到相同输出。
- 当前真机测试平台为 Atlas A2（Ascend 910B3）、CANN 9.0.0。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| -- | -- | -- |
| aclnn API | [examples/test_aclnn_sliding_tile_attention.cpp](examples/test_aclnn_sliding_tile_attention.cpp) | 展示初始化、Tensor 创建、两段式接口调用、同步和资源释放。 |

host 侧 InferShape 单元测试见 [tests/README.md](tests/README.md)。完整的精度、泛化、确定性与性能自验证
矩阵随社区任务验收交付件提交，不复制进本目录。

构建自定义算子包：

```bash
bash build.sh --make_clean --experimental --pkg \
  --soc=ascend910b --ops=sliding_tile_attention \
  --vendor_name=custom_transformer
```

安装生成的 `.run` 包并加载其 `set_env.bash` 后，可编译运行示例。详细接口说明见
[docs/aclnnSlidingTileAttention.md](docs/aclnnSlidingTileAttention.md)。

## 参考资源

- [FastVideo Python 接口](https://github.com/hao-ai-lab/FastVideo/blob/4ddcdf541f32b63b5c684016c903658e2e2b6f67/fastvideo-kernel/python/fastvideo_kernel/ops.py)
- [FastVideo Triton 实现](https://github.com/hao-ai-lab/FastVideo/blob/4ddcdf541f32b63b5c684016c903658e2e2b6f67/fastvideo-kernel/python/fastvideo_kernel/triton_kernels/st_attn_triton.py)
