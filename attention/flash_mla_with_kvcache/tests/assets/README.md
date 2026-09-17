# FlashMlaWithKvcache TTK assets

在安装好 TTK、torch_npu 和当前分支 cann_ops_transformer 的 Ascend 环境运行：

```bash
cd attention/flash_mla_with_kvcache/tests/assets
python gen_cases.py
python -m ttk e2e -i testcase/flash_mla_with_kvcache_e2e_coverage.csv --plugin . --aclgraph --dev 1 --pc 1
```

默认生成 coverage CSV，共 50 条。`python gen_cases.py --suite smoke` 仍可生成原来的 16 条 smoke。

| 组别 | 条数 | 代码分支/边界 |
| --- | ---: | --- |
| 完整组合 | 16 | FP16/BF16 × PA_BBND/PA_NZ × mask 0/3 × LSE 开关 |
| KV 边界 | 20 | N1=64、两种 dtype，长度 1、111/112/113、127/128/129、223/224/225；tile=112、page=128 |
| 可选参数 | 4 | seqused_q 不传；max_seqlen_q/kv 均不传或仅 q 不传 |
| KV stride | 4 | BBND 轴 0；NZ 轴 0、1、0+1；非零 storage offset |
| Q/调度 | 6 | Q=3/17/33、长 KV 8193/4096、四 batch 不等长；最后一条仅 kv max 不传 |

长 KV 仅配 Q=1/2；每条输入 storage 总计小于 16 MiB（不包含运行时 workspace、
编译进程和 golden 临时内存）。顺序/倒序物理页映射、非整页和 block_table 的 -1 padding 均覆盖。
N1=64时Q=3位于 metadata 的 `g*maxQ <= 2*M` 边界，Q=17/33进入另一侧；长 KV 用于探测 FD 切分，
实际 FD/section 数由设备核数和代价模型决定，不宣称覆盖所有调度分支。
NZ 轴 1 大小为 1，其 stride 用例仅覆盖退化轴元数据，不等价于多 KV 头访存。
未纳入空序列/未使用 Q padding、超出当前拦截范围的布局、以及需要大内存才能触发的 L2 section 场景。

CSV 覆盖 FP16/BF16、PA_BBND/PA_NZ、mask_mode 0/3、
不同 batch 序列长度、非整页 KV 和 LSE 开关。Query 固定 TND、N1支持64/96、D=576，
输出 NTD、head_dim_v=512，block_size=128，max_seqlen_q/kv 均为 -1。
可选参数专项用例会省略 max_seqlen 属性，使用接口的 -1 默认值。
张量顺序与算子一致：q、k_cache、block_table、cache_seqlens、cu_seqlens_q、
seqused_q、attn_mask、metadata。小整数张量由 attributes 中的 `*_values` 填充。

`npu_preprocess` 在 H2D 后生成 metadata 并填入预分配槽；图模式通过 companion
metadata 算子生成。默认按 AIC+AIV 不超过 256 核预留容量，可使用
`python gen_cases.py --core-count <核数上界>` 调整，容量不足时会明确报错。

`impl/flash_mla_with_kvcache_golden.py` 和 `impl/attention_math.py` 来自用户指定的
ops-transformer-testkit golden，前者仅调整为包内导入。计算逻辑保持一致，TTK
适配层负责转 CPU，关闭 LSE 时跳过该输出比较。Attention 输出采用 FlashAttn
assets 的精度策略；LSE 使用 testkit provider 的绝对误差 0.1。

在线 kernel 编译由 TTK 的 `_compile_model_aclgraph` 配置控制：
`fullgraph=True, options={"static_kernel_compile": True}`，不是 CSV 字段。
这些用例是覆盖输入集，不代表全部已通过精度验证；此前 BF16 smoke 存在精度失败，
本次不放宽比较阈值。
