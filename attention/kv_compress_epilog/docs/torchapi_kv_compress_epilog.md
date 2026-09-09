# kv_compress_epilog

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：不支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 推理产品</term>：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas 推理系列产品</term>：不支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas 训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

- **接口功能**：

    在KV Cache的Epilog阶段对cache进行原地量化压缩更新，封装aclnnKvCompressEpilog。将bfloat16激活值x量化压缩后，按slot_mapping散写到cache，slot_mapping中值为-1的token跳过不处理。x尾轴d的后64列为rope段、前d-64列为nope段。支持三种量化模式：
  - `"fp8_bf16"`/`"fp8_e8m0"`（内部0/1）：逐组动态量化压缩为FP8（uint8打包），rope段保留bfloat16；两者区别仅为每组scale的表示（bf16 / e8m0）；
  - `"hifloat8_fp4"`（内部2）：rope段hifloat8**静态**量化（乘x_scale后取hifloat8），nope段per-group **float4_E2M1 动态**量化（scale=组内amax/6以bfloat16写出）。

- 语义说明：

    底层aclnn接口对cacheRef为原地实现；本Python接口同为原地语义：直接在输入cache上更新，未被slot_mapping命中的行保留原值。schema中cache标注为`Tensor(a!)`，**本接口无返回值**，调用后直接使用入参cache。

## 函数原型

```python
cann_ops_transformer.kv_compress_epilog(cache, x, slot_mapping, *, quant_group_size=64, quant_mode="fp8_e8m0", round_scale=True, x_scale=1.0) -> None
```

## 参数说明

<table style="undefined;table-layout: fixed; width:1625px"><colgroup>
<col style="width: 147px">
<col style="width: 132px">
<col style="width: 132px">
<col style="width: 480px">
<col style="width: 330px">
<col style="width: 280px">
</colgroup>
<thead>
<tr>
    <th>参数名</th>
    <th>参数类型</th>
    <th>可选/必选</th>
    <th>描述</th>
    <th>数据类型</th>
    <th>维度(shape)</th>
</tr>
</thead>
<tbody>
    <tr>
        <td>cache</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>当前层的分页KV Cache，<b>仅支持四维shape</b>（num_slots = blockNum × blockSize），<b>headDim须 ≥ kvCacheCol</b>（kvCacheCol计算见约束说明）。在<b>blockNum维支持非连续</b>（分页）。数据格式为$ND$。</td>
        <td>uint8</td>
        <td>(blockNum, blockSize, 1, headDim)</td>
    </tr>
    <tr>
        <td>x</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>待量化的激活输入，d需满足d % 64 == 0且64 &lt; d ≤ 8192。数据格式为$ND$。</td>
        <td>bfloat16</td>
        <td>(bs, d)</td>
    </tr>
    <tr>
        <td>slot_mapping</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>token到cache slot的索引映射，元素取值范围[-1, num_slots - 1]（num_slots = blockNum × blockSize），-1表示跳过该token；维度需等于x的维度减1。数据格式为$ND$。</td>
        <td>int32、int64</td>
        <td>(bs)</td>
    </tr>
    <tr>
        <td>quant_group_size</td>
        <td>int</td>
        <td>可选</td>
        <td>量化分组大小，默认值64。quant_mode=0/1时内核固定按64处理；quant_mode=2时仅支持16/32/64，且要求(d-64) % quant_group_size == 0。</td>
        <td>INT</td>
        <td>-</td>
    </tr>
    <tr>
        <td>quant_mode</td>
        <td>str</td>
        <td>可选</td>
        <td>量化模式（字符串，大小写不敏感；TorchNPU内部经枚举映射为算子侧int）。默认值<code>"fp8_e8m0"</code>。三种模式中 <code>"fp8_bf16"</code> 与 <code>"fp8_e8m0"</code> 为同一种per-group FP8 量化方法（nope/rope整体逐组动态量化为FP8(e4m3) 并打包进uint8），仅scale表示不同。可选值与含义：
            <ul>
                <li><code>"fp8_bf16"</code>(内部0)：per-group FP8 量化，组大小由quant_group_size决定（内核固定按64），每组一个 <b>bf16</b> scale。</li>
                <li><code>"fp8_e8m0"</code>(内部1)：per-group FP8 量化，每组一个 <b>e8m0</b>(2 的幂指数) scale，即MX-FP8 微缩放。</li>
                <li><code>"hifloat8_fp4"</code>(内部2)：rope段（尾轴后64 列）HiFloat8 <b>静态</b>量化（乘x_scale后取HiFloat8）；nope段（前d-64 列）per-group <b>float4_E2M1 动态</b>量化（scale=组内amax/6，以bf16 写出）。</li>
            </ul>
            下文约束说明中出现的quant_mode=0/1/2 指上表对应的内部int取值。</td>
        <td>STR</td>
        <td>-</td>
    </tr>
    <tr>
        <td>round_scale</td>
        <td>bool</td>
        <td>可选</td>
        <td>group模式下是否对每组scale向上取到2的幂。默认值True。quant_mode=2下不生效。</td>
        <td>BOOL</td>
        <td>-</td>
    </tr>
    <tr>
        <td>x_scale</td>
        <td>float</td>
        <td>可选</td>
        <td>quant_mode=2时为rope段hifloat8静态量化的缩放系数；quant_mode=0/1下预留未使用。默认值1.0。</td>
        <td>float</td>
        <td>-</td>
    </tr>
</tbody>
</table>

## 返回值说明

该接口无返回值。计算结果直接原地写回输入张量`cache`，`cache`在计算后shape和dtype保持不变；未被`slot_mapping`命中的行保留原值。

## 约束说明

- 该接口支持推理场景下使用。
- 该接口支持单算子模式和TorchAir图模式调用（经graph_convert注册的GE converter下沉为KvCompressEpilog算子，eager与图模式结果一致）。
- cache仅支持四维shape `[blockNum, blockSize, 1, headDim]`，倒数第二维固定为1（每token一个压缩向量），不支持其他维数。
- cache仅在blockNum维支持非连续：各block可不紧密排布，但block内（blockSize、headDim维）须连续。
- headDim约束：cache末维headDim必须 ≥ 每行写出字节数kvCacheCol = 对齐(concatCol)，concatCol按量化模式计算：
  - quant_mode=0/1：concatCol = (d−64) + 128 + ⌈(d−64)/64⌉ × scaleBytes，scaleBytes为mode0=2、mode1=1；
  - quant_mode=2：concatCol = 64 + (d−64)/2 + (d−64)/quant_group_size × 2；
  - 对齐规则：quant_mode=1不补齐，kvCacheCol = concatCol；quant_mode=0/2按32B对齐，kvCacheCol = ⌈concatCol / 32⌉ × 32。
  - 示例：d=512、quant_mode=1 → kvCacheCol=583 → headDim ≥ 583（示例中取640）。
- slot_mapping的维度应等于x的维度减1，即slot_mapping为x除最后一维外的所有维度展平。
- x的最后一维（d轴）需满足d % 64 == 0且64 < d ≤ 8192，按每64个元素一组进行逐组量化。
- quant_mode=2时，quant_group_size仅支持16/32/64，且nope段长度(d-64)需能被quant_group_size整除。
- slot_mapping中值为-1的token会被跳过不处理；其余有效元素取值范围为[0, num_slots - 1]，且元素值应保证不重复，重复时不保证结果正确性。

## 确定性计算

默认支持确定性计算。

## 调用说明

- 单算子模式调用

    ```python
    import torch
    import torch_npu
    from cann_ops_transformer.ops import kv_compress_epilog

    # cache为四维 [blockNum, blockSize, 1, headDim]，num_slots = blockNum*blockSize
    # headDim须 >= kvCacheCol（d=256/quant_mode=1 时kvCacheCol=323，取head_dim=384 满足）
    block_num, block_size, head_dim = 128, 16, 384
    bs, d = 1024, 256

    cache = torch.zeros(block_num, block_size, 1, head_dim, dtype=torch.uint8).npu()
    x = torch.randn(bs, d, dtype=torch.bfloat16).npu()
    slot_mapping = torch.randint(0, block_num * block_size, (bs,), dtype=torch.int32).npu()

    kv_compress_epilog(
        cache, x, slot_mapping, quant_group_size=64, quant_mode="fp8_e8m0", round_scale=True, x_scale=1.0)
    print(cache.shape, cache.dtype)
    ```

- 图模式（torchair）调用

    复用上面单算子示例构造的cache/x/slot_mapping，用torchair后端编译后调用即可：

    ```python
    import torchair

    class KvCompressEpilogModel(torch.nn.Module):
        def forward(self, cache, x, slot_mapping):
            kv_compress_epilog(
                cache, x, slot_mapping,
                quant_group_size=64, quant_mode="fp8_e8m0", round_scale=True, x_scale=1.0)
            return cache

    model = KvCompressEpilogModel().npu()
    npu_backend = torchair.get_npu_backend()
    model = torch.compile(model, backend=npu_backend, dynamic=False)
    model(cache, x, slot_mapping)
    print(cache.shape, cache.dtype)
    ```
