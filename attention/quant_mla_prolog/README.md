# QuantMlaProlog

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| <term>Ascend 950PR&950DT系列产品</term> | √ |
| <term>Atlas A3系列产品</term> | × |
| <term>Atlas A2系列产品</term> | × |
| <term>Atlas 200I/500 A2推理产品</term> | × |
| <term>Atlas推理系列产品</term> | × |
| <term>Atlas训练系列产品</term> | × |

## 功能说明

多头潜在注意力（Multi-Head Latent Attention，MLA）前处理使用均方根归一化（Root Mean Square Normalization，RMSNorm）和旋转位置编码（Rotary Positional Embedding，RoPE）。Q表示Query，KV表示共享Key/Value低秩向量，KR表示Key的旋转子段；QA/QB/KVA对应三路投影，Wkb将Q的非位置编码（NoPE）部分吸收到512维。ND为线性密排存储，NZ为FRACTAL_NZ分块格式；BF16/FP32分别表示bfloat16/float32，HiF8使用uint8原始编码。

完成多头潜在注意力（Multi-Head Latent Attention，MLA）推理前处理，包含投影、均方根归一化（Root Mean Square Normalization，RMSNorm）、旋转位置编码（Rotary Positional Embedding，RoPE）、可选Q/Cache量化及Cache原地更新。

以下公式描述数学依赖，实际计算保留下述类型转换点。`T`为token数，`N`为head数，`Dn`为Wkb归约宽度，`Dr`为旋转维度，KV低秩维度固定为512，`B,S`为Cache页数和页大小。

公式中的`wqa`、`wqb`、`wkva`按ND的`(输出维度,输入维度)`方向表示，NZ存储对应相同的数学权重。

$$
\operatorname{RMSNorm}(v,\gamma)_j=\frac{v_j\gamma_j}{\sqrt{\frac{1}{d}\sum_{i=0}^{d-1}v_i^2+\epsilon}}.
$$

$$
R=\operatorname{RMSNorm}(P(x,wqa),\gamma_{qa})\cdot\texttt{qr\_scale}.
$$

$$
(QR,s_{QR})=\operatorname{DynamicQuant}(R).
$$

$$
U=\operatorname{reshape}_{T,N,Dn+Dr}(P(QR,wqb)),\quad
Q^N_{t,n}=U_{t,n,:Dn} wkb_n,\quad Q^R=\operatorname{RoPE}(U_{:,:,Dn:}).
$$

$$
V=P(x,wkva),\quad KV=\operatorname{RMSNorm}(V_{:,:512},\gamma_{kva})\cdot\texttt{kvr\_scale},\quad KR=\operatorname{RoPE}(V_{:,512:}).
$$

输出编码前的Q为Q-NoPE与Q-RoPE拼接。两路归一化共用epsilon；KR不参与归一化或kvr_scale缩放；Wkb只作用于Q-NoPE。

MXFP8解码E4M3FN数据，每个32元素K分组点积乘对应激活与权重E8M0 descale：

$$
P(A,W)_{t,o}=\sum_g s_A[t,g]s_W[o,g]\sum_{k=32g}^{32g+31}\operatorname{Decode}(A_{t,k})\operatorname{Decode}(W_{o,k}).
$$

HiF8先对解码值做FP32累加，再依次乘权重per-channel descale和激活全局descale。MXFP8 QR按行、每32值分组动态量化为E4M3FN/E8M0；HiF8 QR对完整`(T,q_lora)`使用一个HiF8/FLOAT全局scale。有限正常分组的MX scale为：

$$
s_g=2^{\lfloor\log_2(\max |R_g|)\rfloor-8},\qquad QR_g=\operatorname{E4M3FN}(R_g/s_g).
$$

MX QR保留局部BF16预舍入/饱和及归一化后的BF16舍入。全零、次正规及特殊值分组按实现显式处理指数，全零分组不直接求log。MX投影结果中的有限非零FP32次正规数置为正零。

矩阵投影使用FP32累加；BF16输入实现保留投影及归一化结果的BF16中间转换。量化实现的Q-NoPE在Wkb前后转换为BF16，RoPE结果转换为BF16。不能用不含这些转换的一条实数矩阵链替代数值路径。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| --- | --- | --- | --- | --- |
| `x` | 输入 | 输入隐状态。Shape为`(T,dim)`。 | MXFP8:FLOAT8_E4M3FN; HiF8:UINT8 | ND |
| `wqa` | 输入 | Query低秩投影权重，将dim维输入投影到q_lora维。逻辑Shape为`ND:(q_lora,dim); NZ:(dim,q_lora)`。 | MXFP8:FLOAT8_E4M3FN; HiF8:UINT8 | ND/FRACTAL_NZ |
| `wqb` | 输入 | Q上投影权重。逻辑Shape为`ND:(N*(Dn+Dr),q_lora); NZ:(q_lora,N*(Dn+Dr))`。 | MXFP8:FLOAT8_E4M3FN; HiF8:UINT8 | ND/FRACTAL_NZ |
| `wkva` | 输入 | KVA投影权重。逻辑Shape为`ND:(512+Dr,dim); NZ:(dim,512+Dr)`。 | MXFP8:FLOAT8_E4M3FN; HiF8:UINT8 | ND/FRACTAL_NZ |
| `wkb` | 输入 | 每头Q-NoPE吸收权重，不参与Cache生成。Shape为`(N,Dn,512)`。 | BFLOAT16 | ND |
| `descale_x` | 输入 | 激活反量化乘子。Shape为`MXFP8:(T,dim/32); HiF8:(1,)`。 | MXFP8:FLOAT8_E8M0; HiF8:FLOAT | ND |
| `descale_wqa` | 输入 | Wqa反量化乘子；O=q_lora，K=dim。Shape为`ND:(O,K/64,2); NZ:(K/64,O,2); HiF8:(1,O)`。 | MXFP8:FLOAT8_E8M0/UINT8(NZ); HiF8:FLOAT | ND/FRACTAL_NZ |
| `descale_wqb` | 输入 | Wqb反量化乘子；O=N*(Dn+Dr)，K=q_lora。Shape为`ND:(O,K/64,2); NZ:(K/64,O,2); HiF8:(1,O)`。 | MXFP8:FLOAT8_E8M0/UINT8(NZ); HiF8:FLOAT | ND/FRACTAL_NZ |
| `descale_wkva` | 输入 | Wkva反量化乘子；O=512+Dr，K=dim。Shape为`ND:(O,K/64,2); NZ:(K/64,O,2); HiF8:(1,O)`。 | MXFP8:FLOAT8_E8M0/UINT8(NZ); HiF8:FLOAT | ND/FRACTAL_NZ |
| `norm_weight_qa` | 输入 | QA均方根归一化的gamma。Shape为`(q_lora,)`。 | FLOAT | ND |
| `norm_weight_kva` | 输入 | KV均方根归一化的gamma。Shape为`(512,)`。 | FLOAT | ND |
| `kv_cache` | 输入/输出 | 原地更新的KV/KR合并Cache；仅写入有效索引对应的槽位。Shape为`(B,S,1,Dcache)`。 | mode 0:BFLOAT16; mode 1/2:FLOAT8_E4M3FN; mode 3:UINT8 | ND carrier |
| `cache_index` | 输入 | token对应的线性Cache槽位；负值跳过Cache写入。Shape为`(T,)`。 | INT64 | ND |
| `w_qr_quant_mode` | 属性 | 输入/权重/QR量化模式，使用WQrQuantMode；必填。 | 枚举 | - |
| `qkv_quant_mode` | 属性 | Q/Cache输出模式，取本算子QkvQuantMode枚举，见模式表。 | 枚举 | - |
| `quant_scale_kv` | 可选输入 | 默认None；模式2/3必填，须为有限值，允许0和负数；模式0/1必须为None。Cache按value*quant_scale_kv量化。Shape为`(1,)`。 | FLOAT | ND |
| `rope_sin` | 可选输入 | 默认None；启用RoPE时必填，与rope_cos成对传入。Shape为`(T,Dr)`。 | BFLOAT16 | ND |
| `rope_cos` | 可选输入 | 默认None；RoPE余弦参数，与rope_sin成对传入。Shape为`(T,Dr)`。 | BFLOAT16 | ND |
| `norm_eps` | 可选属性 | 默认1e-5；两路RMSNorm共用，转FP32后有限且大于0，拒绝bool。 | FLOAT | - |
| `cache_mode` | 可选属性 | CacheMode.PA_BSND或CacheMode.PA_NZ；模式1仅支持PA_BSND。可选，默认PA_BSND。 | 枚举 | - |
| `qr_scale` | 可选属性 | 默认1.0；QA归一化后的乘子，FP32有限，允许0和负数，拒绝bool。 | FLOAT | - |
| `kvr_scale` | 可选属性 | 默认1.0；KV归一化后的乘子，不缩放KR；数值要求与qr_scale一致。 | FLOAT | - |
| `rope_mode` | 可选属性 | 默认RopeMode.ROPE_INTERLEAVE_HALF；支持NO_ROPE、ROPE_INTERLEAVE_HALF和ROPE_HALF。 | 枚举 | - |
| `return_qr` | 可选属性 | 默认False；为True时填充qr输出槽位。 | BOOL | - |
| `q` | 输出 | Q-NoPE与Q-RoPE拼接，每头512+Dr个值。Shape为`(T,N,512+Dr)`。 | mode 0/1:BFLOAT16; mode 2:FLOAT8_E4M3FN; mode 3:UINT8 | ND |
| `descale_q` | 输出 | Q反量化乘子；模式0/1及空输入模式3返回空Tensor。Shape为`mode 2:(T,N,1); mode 3,T>0:(1,); otherwise:(0,)`。 | FLOAT | ND |
| `qr` | 输出 | 仅return_qr=True时填充，否则Shape为(0,)。Shape为`return_qr=True:(T,q_lora); otherwise:(0,)`。 | MXFP8:FLOAT8_E4M3FN; HiF8:UINT8 | ND |
| `descale_qr` | 输出 | QR反量化乘子；未输出QR或HiF8空输入时为空。Shape为`QR+MXFP8:(T,q_lora/32); QR+HiF8,T>0:(1,); otherwise:(0,)`。 | MXFP8:FLOAT8_E8M0; HiF8:FLOAT | ND |

表中`wqa`、`wqb`、`wkva`的Shape均指Python可见的逻辑形状（`Tensor.shape`），三者须全部使用ND或全部使用FRACTAL_NZ。ND按`(输出维度,输入维度)`传入；NZ按`(输入维度,输出维度)`传入转置后的同一组权重，并采用实际的FRACTAL_NZ存储。仅转置或reshape不能完成NZ格式转换。

Cache为单一合并Buffer，Q-NoPE和Q-RoPE拼为一个q输出。

## 约束说明

- `T >= 0`；`dim`为1024～8192内的256倍数；`q_lora`为640、1024、1536或2048；`N`为1～128；`(Dn,Dr)`为`(128,64)`、`(192,64)`或`(256,0)`；KV低秩维度固定为512。
- `B > 0`；`S`为16～1024内的16倍数。模式1仅支持PA_BSND，其余模式支持PA_BSND/PA_NZ。输入只接受合轴的`(T,dim)`，不接受`(B,S,dim)`或PA_BLK布局。
- 所有输入位于同一NPU。除Cache外均须连续；Cache内侧stride为`(Dcache,Dcache,1)`，且`stride(0) >= S*Dcache`，支持首轴间隙及非零storage offset。未命中槽位和页间空隙的字节保持不变。
- 调用方保证非负Cache索引小于`B*S`且唯一；算子不回读或检查索引值。负索引仅跳过Cache写入，对应token仍参与Q/QR及全局量化统计。
- `Dr=0`必须指定`NO_ROPE`；执行RoPE时sin/cos必须成对传入。禁用时可同时为None，若传入仍检查Shape及类型。
- `norm_eps`转FP32后须有限且大于0；`qr_scale/kvr_scale`须为有限FP32，允许0和负数，均拒绝bool。模式2/3的`quant_scale_kv`须为有限值，允许0和负数，模式0/1必须为None。
- `T=0`仍校验元信息，按接口返回空输出，不启动计算Kernel，Cache不变。模式0、2、3的Cache宽度为`512+Dr`，模式1为`512+2*Dr+16`。
- 支持Decode/Prefill推理，不提供自动求导或功能化Cache更新。输出和按阶段复用的临时空间由内部申请，无外部out/workspace参数；非空调用要求当前Stream具有有效MIX核资源。
- 非空输入使用quant_scale_kv时，在执行前及修改scale后调用公开入口的precheck；该操作在图外回读scale，执行入口不重复检查其值。ND激活/权重descale必须为E8M0，UINT8仅用于合法分配的NZ权重descale载体。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| --- | --- | --- |
| PyTorch API | [Eager/ACLGraph/NPUGraph-EX](docs/torchapi_quant_mla_prolog.md#调用示例) | [quant_mla_prolog](docs/torchapi_quant_mla_prolog.md) |

通过`cann_ops_transformer.ops.quant_mla_prolog`调用，显式指定`w_qr_quant_mode`和`qkv_quant_mode`。输入与Cache须位于同一NPU，`kv_cache`原地更新。

## 参考资源

- [PyTorch API](docs/torchapi_quant_mla_prolog.md)
