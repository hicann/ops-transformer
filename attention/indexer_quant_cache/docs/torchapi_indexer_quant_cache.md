# indexer_quant_cache

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

    在Indexer注意力机制的Epilog阶段对KV Cache进行原地动态量化压缩更新，封装aclnnIndexerQuantCache。将float16/bfloat16激活值x按量化粒度（mxfp8/mxfp4 模式每32个元素一组，fp8/hifloat8 模式整行一组）动态量化为FP8（E4M3/E5M2）或MX-FP4，并按slot_mapping将量化结果与对应scale散写到cache/cache_scale，slot_mapping中值为-1的token跳过不处理。支持mxfp8、fp8、hifloat8、mxfp4 四种量化模式（各模式含义见参数说明quant_mode）。

- 语义说明：

    底层aclnn接口对cacheRef、cacheScaleRef为原地实现；本Python接口同为原地语义：直接在输入cache、cache_scale上更新，未被slot_mapping命中的行保留原值。schema中cache、cache_scale分别标注为`Tensor(a!)`、`Tensor(b!)`，**本接口无返回值**，调用后直接使用入参cache、cache_scale。

## 函数原型

```python
cann_ops_transformer.indexer_quant_cache(cache, cache_scale, x, slot_mapping, *, quant_mode="fp8", round_scale=True, x_scale=1.0) -> None
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
        <td>当前层的分页KV Cache，<b>仅支持四维shape</b>（num_slots = blockNum × blockSize），<b>headDim须 ≥ d</b>（MX-FP4以fp4元素计）。在<b>blockNum维支持非连续</b>（分页）。MX-FP4模式下cache为FP4（每字节打包2个fp4值）。数据格式为$ND$。</td>
        <td>float8_e4m3fn、float8_e5m2、uint8、float4_E2M1、float4_e1m2</td>
        <td>(blockNum, blockSize, 1, headDim)</td>
    </tr>
    <tr>
        <td>cache_scale</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>每块量化的scale因子，<b>仅支持四维shape</b>，<b>scaleHeadDim须 ≥ scaleCol</b>；含HiFloat8/quant_mode=2在内的所有量化模式均校验（scaleCol取值与HiFloat8规则详见约束说明）。数据格式为$ND$。</td>
        <td>float、float8_e8m0</td>
        <td>(blockNum, blockSize, 1, scaleHeadDim)</td>
    </tr>
    <tr>
        <td>x</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>待量化的激活输入，d需满足d % 32 == 0且d ≤ 8192。数据格式为$ND$。</td>
        <td>float16、bfloat16</td>
        <td>(bs, d)</td>
    </tr>
    <tr>
        <td>slot_mapping</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>token到cache slot的索引映射，元素取值范围[-1, num_slots - 1]，-1表示跳过该token；维度需等于x的维度减1。数据格式为$ND$。</td>
        <td>int32</td>
        <td>(bs)</td>
    </tr>
    <tr>
        <td>quant_mode</td>
        <td>str</td>
        <td>可选</td>
        <td>量化模式（字符串，大小写不敏感；TorchNPU内部经枚举映射为算子侧int）。默认值<code>"fp8"</code>。可选值与含义：
            <ul>
                <li><code>"mxfp8"</code>(内部0)：MX-FP8 微缩放量化，x每32 个元素一组，输出FP8(e4m3/e5m2)，每组一个e8m0(2 的幂指数) scale，cache_scale为float8_e8m0。</li>
                <li><code>"fp8"</code>(内部1)：逐token动态FP8 量化，整行（headDim）一组，输出FP8(e4m3/e5m2)，每行一个float32 scale（scaleCol=1），cache_scale为float。</li>
                <li><code>"hifloat8"</code>(内部2)：HiFloat8 静态量化（按x_scale缩放后取HiFloat8），整行一组，cache_scale为float。</li>
                <li><code>"mxfp4"</code>(内部3)：MX-FP4 微缩放量化，x每32 个元素一组，输出FP4(每字节打包2 个值)，每组一个e8m0 scale，cache_scale为float8_e8m0。</li>
            </ul>
            下文约束说明中出现的quant_mode=0/1/2/3 指上表对应的内部int取值。</td>
        <td>STR</td>
        <td>-</td>
    </tr>
    <tr>
        <td>round_scale</td>
        <td>bool</td>
        <td>可选</td>
        <td>MX-FP8模式（quant_mode=0）下是否对scale进行舍入。默认值True。</td>
        <td>BOOL</td>
        <td>-</td>
    </tr>
    <tr>
        <td>x_scale</td>
        <td>float</td>
        <td>可选</td>
        <td>HiFloat8模式（quant_mode=2）下的全局scale乘数。默认值1.0。</td>
        <td>float</td>
        <td>-</td>
    </tr>
</tbody>
</table>

## 返回值说明

该接口无返回值。计算结果直接原地写回输入张量`cache`与`cache_scale`，二者在计算后shape和dtype保持不变；未被`slot_mapping`命中的行保留原值。

## 约束说明

- 该接口支持推理场景下使用。
- 该接口支持单算子模式调用。
- cache与cache_scale的数据类型组合需与quant_mode匹配：fp8/hifloat8 模式cache_scale为float；mxfp8/mxfp4 模式cache_scale为float8_e8m0；mxfp4 模式cache为FP4（uint8打包）。
- cache与cache_scale（所有量化模式，含HiFloat8）均仅支持四维shape `[blockNum, blockSize, 1, headDim]`，倒数第二维固定为1（每token一个量化向量）；num_slots = blockNum × blockSize。
- cache/cache_scale仅在blockNum维支持非连续：各block可不紧密排布，但block内须连续。
- headDim长度约束：
  - cache.headDim ≥ d（MX-FP4以fp4元素计，d个fp4值占⌈d/2⌉字节）；
  - cache_scale.headDim ≥ scaleCol，scaleCol = mxfp8/mxfp4：⌈d/32⌉；fp8/hifloat8：1；
  - 示例：d=128、quant_mode=0 → scaleCol=4 → cache.headDim ≥ 128 且cache_scale.headDim ≥ 4。
- slot_mapping的维度应等于x的维度减1，即slot_mapping为x除最后一维外的所有维度展平。
- slot_mapping中值为-1的token会被跳过不处理；其余有效元素取值范围为[0, num_slots - 1]，且元素值应保证不重复，重复时不保证结果正确性。

## 确定性计算

默认支持确定性计算。

## 调用说明

- 单算子模式调用

    ```python
    import torch
    import torch_npu
    from cann_ops_transformer.ops import indexer_quant_cache

    # cache/cache_scale为四维 [blockNum, blockSize, 1, headDim]，num_slots = blockNum*blockSize
    block_num, block_size, d = 128, 16, 128   # num_slots = 2048; mode1 headDim=d=128 >= d
    bs = 1024

    cache = torch.zeros(block_num, block_size, 1, d, dtype=torch.float8_e4m3fn).npu()
    cache_scale = torch.zeros(block_num, block_size, 1, 1, dtype=torch.float32).npu()  # mode1 scaleCol=1
    x = torch.randn(bs, d, dtype=torch.float16).npu()
    slot_mapping = torch.randint(0, block_num * block_size, (bs,), dtype=torch.int32).npu()

    indexer_quant_cache(
        cache, cache_scale, x, slot_mapping, quant_mode="fp8", round_scale=True, x_scale=1.0)
    print(cache.shape, cache.dtype, cache_scale.shape, cache_scale.dtype)
    ```

- 图模式（torchair）调用

    复用上面单算子示例构造的cache/cache_scale/x/slot_mapping，用torchair后端编译后调用即可：

    ```python
    import torchair

    class IndexerQuantCacheModel(torch.nn.Module):
        def forward(self, cache, cache_scale, x, slot_mapping):
            indexer_quant_cache(
                cache, cache_scale, x, slot_mapping,
                quant_mode="fp8", round_scale=True, x_scale=1.0)
            return cache, cache_scale

    model = IndexerQuantCacheModel().npu()
    npu_backend = torchair.get_npu_backend()
    model = torch.compile(model, backend=npu_backend, dynamic=False)
    model(cache, cache_scale, x, slot_mapping)
    print(cache.shape, cache.dtype, cache_scale.shape, cache_scale.dtype)
    ```
