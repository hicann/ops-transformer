## NsaCompressWithCache

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      √     |
|<term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>|      √     |

## 功能说明

+ 算子功能：用于Native-Sparse-Attention推理阶段的KV压缩，每次推理每个batch会产生一个新的token，每当某个batch的token数量凑满一个compress\_block时，该算子会将该batch的后compress\_block个token压缩成一个compress\_token，算法流程如下：
  
  1. 检查act\_seq\_lens是否有满足$s \ge compressBlockSize$ 且 $(s - compressBlockSize) \% stride ==0$的序列长度；
  2. 找到满足序列长度的batchIdx，根据block\_table找到该batch的后compress\_block\_size个token压缩；
  3. 执行压缩算法；
  4. 根据slot\_mapping写回到output\_cache中。
+ 计算公式：

$$
compressIdx=(s-compressBlockSize)/stride\\ 
outputCacheRef[slotMapping[i]] = input[compressIdx*stride : compressIdx*stride+compressBlockSize]*weight[:]
$$

## 参数说明

<table class="tg"><thead>
  <tr>
    <th class="tg-0pky">参数名</th>
    <th class="tg-0pky">输入/输出/属性</th>
    <th class="tg-0pky">描述</th>
    <th class="tg-0pky">数据类型</th>
    <th class="tg-0pky">数据格式</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-0pky">s</td>
    <td class="tg-0pky">属性</td>
    <td class="tg-0pky">当前batch的token长度。</td>
    <td class="tg-0pky">INT64</td>
    <td class="tg-0pky">-</td>
  </tr>
  <tr>
    <td class="tg-0pky">compressBlockSize</td>
    <td class="tg-0pky">属性</td>
    <td class="tg-0pky">压缩滑窗大小。</td>
    <td class="tg-0pky">INT64</td>
    <td class="tg-0pky">-</td>
  </tr>
  <tr>
    <td class="tg-0pky">stride</td>
    <td class="tg-0pky">属性</td>
    <td class="tg-0pky">两次压缩滑窗间隔大小。</td>
    <td class="tg-0pky">INT64</td>
    <td class="tg-0pky">-</td>
  </tr>
  <tr>
    <td class="tg-0lax">weight</td>
    <td class="tg-0lax">输入</td>
    <td class="tg-0lax">k/v值的压缩weight。</td>
    <td class="tg-0lax">BFLOAT16、FLOAT16</td>
    <td class="tg-0lax">ND</td>
  </tr>
  <tr>
    <td class="tg-0pky">input</td>
    <td class="tg-0pky">输入</td>
    <td class="tg-0pky">k/v值的cache。</td>
    <td class="tg-0pky">BFLOAT16、FLOAT16</td>
    <td class="tg-0pky">ND</td>
  </tr>
  <tr>
    <td class="tg-0pky">slotMapping</td>
    <td class="tg-0pky">输入</td>
    <td class="tg-0pky">每个batch尾部压缩数据存储的位置的索引。</td>
    <td class="tg-0pky">INT32</td>
    <td class="tg-0pky">ND</td>
  </tr>
  <tr>
    <td class="tg-0pky">outputCacheRef</td>
    <td class="tg-0pky">输入/输出</td>
    <td class="tg-0pky">输出的cache。</td>
    <td class="tg-0pky">BFLOAT16、FLOAT16</td>
    <td class="tg-0pky">ND</td>
  </tr>
</tbody></table>

## 约束说明

* input和weight满足broadcast关系，input的第三维大小与weight的第二维大小相等。
* compressBlockSize、stride必须是16的整数倍，且compressBlockSize>=stride，compressBlockSize<=64。
* actSeqLenType目前仅支持取值1。
* layoutOptional取值可以是BSH、SBH、BSND、BNSD、TND，但是不会生效。
* pageBlockSize只能是64或者128。
* headDim是16的整数倍，且headDim<=256。
* 不支持input/weight/outputCache为空输入。
* slotMapping的值无重复，否则会导致计算结果不稳定。
* blockTableOptional的值不超过blockNum，否则会发生越界。
* actSeqLenOptional的值不应该超过序列最大长度。
* headNum<=64，且headNum>50时headNum%2=0。

## 调用说明

| 调用方式  | 样例代码                                                                | 说明                                                                                          |
| ----------- | ------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| aclnn接口 | [test_aclnn_nsa_compress_with_cache](./examples/test_aclnn_nsa_compress_with_cache.cpp) | 通过[`aclnnNsaCompressWithCache`](./docs/aclnnNsaCompressWithCache.md)接口方式调用NsaCompressWithCache算子。 |