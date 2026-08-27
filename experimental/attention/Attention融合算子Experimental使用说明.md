# Attention融合算子Experimental使用说明

## 前提条件

1. 环境部署：调用算子之前，请先参考[环境部署](../../docs/zh/install/quick_install.md)完成基础环境搭建，其中**安装依赖**章节的版本约束以如下内容为准：
    - python >= 3.10.0
    - torch >= 2.7.0
    - gcc >= 9.0.0
    - torch\_npu >=2.7.0

2. 安装torch与TorchNPU包

   根据实际环境，下载对应torch包并安装：`torch-${torch_version}+cpu-${python_version}-linux_${arch}.whl`下载链接为：[官网地址](http://download.pytorch.org/whl/torch)

   安装命令如下：

    ```sh
    pip install torch-${torch_version}+cpu-${python_version}-linux_${arch}.whl
    ```

   根据实际环境，安装对应TorchNPU包：`torch_npu-${torch_version}-${python_version}-linux_${arch}.whl`下载链接为：[官网地址](https://gitcode.com/Ascend/pytorch/releases)

   安装命令如下：

    ```sh
    pip install torch_npu-${torch_version}-${python_version}-linux_${arch}.whl
    ```

    - \$\{torch\_version\}：表示torch包版本号。
    - \$\{python\_version\}：表示python版本号。
    - \$\{arch\}：表示CPU架构，如aarch64、x86_64。

## 算子列表

项目提供的所有算子列表如下：

<table><thead>
  <tr>
    <th rowspan="2">算子名</th>
    <th rowspan="2">接口文档</th>
    <th colspan="2">算子实现</th>
    <th rowspan="2">torch调用</th>
    <th colspan="2">图模式调用</th>
    <th rowspan="2">算子执行硬件单元</th>
    <th rowspan="2">说明</th>
  </tr>
  <tr>
    <th>kernel</th>
    <th>host</th>
    <th>Ascend IR</th>
    <th>aclgraph</th>
  </tr></thead>
<tbody>
  <tr>
    <td>compressor</td>
    <td><a href="./compressor/README.md">文档</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>将每4或128个token的KV cache压缩成一个，然后每个token与这些压缩的KV cache进行DSA计算。<br/>算子torch接口调用依赖torch_ops_extension，具体安装方法见<a href="https://gitcode.com/cann/cann-recipes-infer/tree/master/ops/ascendc#torch_ops_extension%E7%AE%97%E5%AD%90%E5%8C%85%E7%BC%96%E8%AF%91%E4%B8%8E%E5%AE%89%E8%A3%85">安装指导</a>。</td>
  </tr>
  <tr>
    <td>kv_quant_sparse_attn_sharedkv</td>
    <td><a href="./kv_quant_sparse_attn_sharedkv/README.md">文档</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>支持量化模式的Sliding Window Attention、Compressed Attention以及Sparse Compressed Attention计算。<br/>算子torch接口调用依赖torch_ops_extension，具体安装方法见<a href="https://gitcode.com/cann/cann-recipes-infer/tree/master/ops/ascendc#torch_ops_extension%E7%AE%97%E5%AD%90%E5%8C%85%E7%BC%96%E8%AF%91%E4%B8%8E%E5%AE%89%E8%A3%85">安装指导</a>。</td>
  </tr>
  <tr>
    <td>kv_quant_sparse_attn_sharedkv_metadata</td>
    <td><a href="./kv_quant_sparse_attn_sharedkv_metadata/README.md">文档</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Cpu</td>
    <td>该算子为kv_quant_sparse_attn_sharedkv算子提供分核结果。<br/>算子torch接口调用依赖torch_ops_extension，具体安装方法见<a href="https://gitcode.com/cann/cann-recipes-infer/tree/master/ops/ascendc#torch_ops_extension%E7%AE%97%E5%AD%90%E5%8C%85%E7%BC%96%E8%AF%91%E4%B8%8E%E5%AE%89%E8%A3%85">安装指导</a>。</td>
  </tr>
  <tr>
    <td>quant_lightning_indexer</td>
    <td><a href="./quant_lightning_indexer/README.md">文档</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>该算子是推理场景下，稀疏Attention前处理的计算，选出关键的稀疏token，并对输入query和key进行量化实现存8算8，获取最大收益。<br/>算子torch接口调用依赖torch_ops_extension，具体安装方法见<a href="https://gitcode.com/cann/cann-recipes-infer/tree/master/ops/ascendc#torch_ops_extension%E7%AE%97%E5%AD%90%E5%8C%85%E7%BC%96%E8%AF%91%E4%B8%8E%E5%AE%89%E8%A3%85">安装指导</a>。</td>
  </tr>
  <tr>
    <td>quant_lightning_indexer_metadata</td>
    <td><a href="./quant_lightning_indexer_metadata/README.md">文档</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Cpu</td>
    <td>该算子为quant_lightning_indexer算子提供分核结果。<br/>算子torch接口调用依赖torch_ops_extension，具体安装方法见<a href="https://gitcode.com/cann/cann-recipes-infer/tree/master/ops/ascendc#torch_ops_extension%E7%AE%97%E5%AD%90%E5%8C%85%E7%BC%96%E8%AF%91%E4%B8%8E%E5%AE%89%E8%A3%85">安装指导</a>。</td>
  </tr>
  <tr>
    <td>sparse_attn_sharedkv</td>
    <td><a href="./sparse_attn_sharedkv/README.md">文档</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>该算子支持非量化的Sliding Window Attention、Compressed Attention以及Sparse Compressed Attention计算。<br/>算子torch接口调用依赖torch_ops_extension，具体安装方法见<a href="https://gitcode.com/cann/cann-recipes-infer/tree/master/ops/ascendc#torch_ops_extension%E7%AE%97%E5%AD%90%E5%8C%85%E7%BC%96%E8%AF%91%E4%B8%8E%E5%AE%89%E8%A3%85">安装指导</a>。</td>
  </tr>
  <tr>
    <td>sparse_attn_sharedkv_metadata</td>
    <td><a href="./sparse_attn_sharedkv_metadata/README.md">文档</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Cpu</td>
    <td>该算子为sparse_attn_sharedkv算子提供分核结果。<br/>算子torch接口调用依赖torch_ops_extension，具体安装方法见<a href="https://gitcode.com/cann/cann-recipes-infer/tree/master/ops/ascendc#torch_ops_extension%E7%AE%97%E5%AD%90%E5%8C%85%E7%BC%96%E8%AF%91%E4%B8%8E%E5%AE%89%E8%A3%85">安装指导</a>。</td>
  </tr>
  <tr>
    <td>blitz_sparse_attention</td>
    <td>/</td>
    <td>/</td>
    <td>/</td>
    <td>/</td>
    <td>/</td>
    <td>/</td>
    <td>/</td>
    <td>算子具体使用方法见<a href="https://gitcode.com/cann/ops-transformer/blob/master/experimental/attention/blitz_sparse_attention/README.md">README文档</a>。</td>
  </tr>
  <tr>
    <td>fused_infer_attention_score</td>
    <td>/</td>
    <td>/</td>
    <td>/</td>
    <td>/</td>
    <td>/</td>
    <td>/</td>
    <td>/</td>
    <td>算子具体使用方法见<a href="https://gitcode.com/cann/ops-transformer/blob/master/experimental/attention/fused_infer_attention_score/README.md">README文档</a>。</td>
  </tr>
  <tr>
    <td>typhoon_mla</td>
    <td>/</td>
    <td>/</td>
    <td>/</td>
    <td>/</td>
    <td>/</td>
    <td>/</td>
    <td>/</td>
    <td>算子具体使用方法见<a href="https://gitcode.com/cann/ops-transformer/blob/master/experimental/attention/typhoon_mla/README.md">README文档</a>。</td>
  </tr>
  <tr>
    <td>fused_gdn_decode</td>
    <td><a href="./fused_gdn_decode/README.md">文档</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>该算子融合GDN单token解码中的QKV拆分、Q/K归一化、门控计算、循环状态更新和输出投影。<br/>算子torch接口调用依赖torch_ops_extension，具体安装方法见<a href="https://gitcode.com/cann/cann-recipes-infer/tree/master/ops/ascendc#torch_ops_extension%E7%AE%97%E5%AD%90%E5%8C%85%E7%BC%96%E8%AF%91%E4%B8%8E%E5%AE%89%E8%A3%85">安装指导</a>。</td>
  </tr>
  <tr>
    <td>stem_indexer</td>
    <td><a href="./stem_indexer/docs/StemIndexer.md">文档</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>该算子是推理场景下，稀疏Attention前处理的计算，选出关键的稀疏block块，获取最大收益。<br/>算子torch接口调用依赖torch_ops_extension，具体安装方法见<a href="https://gitcode.com/cann/cann-recipes-infer/tree/master/ops/ascendc#torch_ops_extension%E7%AE%97%E5%AD%90%E5%8C%85%E7%BC%96%E8%AF%91%E4%B8%8E%E5%AE%89%E8%A3%85">安装指导</a>。</td>
  </tr>
  <tr>
    <td>stem_indexer_metadata</td>
    <td><a href="./stem_indexer_metadata/README.md">文档</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Cpu</td>
    <td>该算子为stem_indexer算子提供分核结果。<br/>算子torch接口调用依赖torch_ops_extension，具体安装方法见<a href="https://gitcode.com/cann/cann-recipes-infer/tree/master/ops/ascendc#torch_ops_extension%E7%AE%97%E5%AD%90%E5%8C%85%E7%BC%96%E8%AF%91%E4%B8%8E%E5%AE%89%E8%A3%85">安装指导</a>。</td>
  </tr>
</tbody>
</table>

## 自定义算子编译

1. 编译自定义算子包

    进入项目根目录，执行如下编译命令：

    ```bash
    bash build.sh --pkg --experimental --soc=${soc_version} --ops=${op_list}
    # 如要使用DeepSeek-V4，910c环境编译命令示例如下：
    # bash build.sh --pkg --experimental --soc=ascend910_93 --ops=compressor,quant_lightning_indexer,quant_lightning_indexer_metadata,sparse_attn_sharedkv,sparse_attn_sharedkv_metadata

    # 如要使用DeepSeek-V4，ascend950环境编译命令示例如下：
    # bash build.sh --pkg --experimental --soc=ascend950 --ops=compressor,quant_lightning_indexer,quant_lightning_indexer_metadata,kv_quant_sparse_attn_sharedkv,kv_quant_sparse_attn_sharedkv_metadata
    ```
    - --soc：\$\{soc\_version\}表示NPU型号。Atlas A2 训练系列产品/Atlas A2 推理系列产品使用"ascend910b"，Atlas A3 训练系列产品/Atlas A3 推理系列产品使用"ascend910_93"。
    - --ops：自定义算子名称，多个自定义算子通过`,`分割。

    若提示如下信息，说明编译成功。
    ```bash
    Self-extractable archive "cann-ops-transformer-custom_linux.${arch}.run" successfully created.
    ```
    编译成功后，run包存放于项目根目录的build_out目录下。

2. 安装自定义算子包

    ```bash
    ./cann-ops-transformer-${vendor_name}_linux-${arch}.run --install-path=${install_path}
    ```

    自定义单算子包安装时若不指定参数`--install-path`，将默认安装于`${ASCEND_HOME_PATH}/opp/vendors`目录下。\$\{ASCEND\_HOME\_PATH\}已通过环境变量配置，表示CANN ops包安装路径。如指定`--install-path=${install_path}`，则自定义算子包将安装于`${install_path}/vendors`目录下。注意自定义算子包不支持卸载。安装完成后，需要激活自定义算子包环境变量或导出环境变量，具体方式如下：

    ```bash
    # experimental自定义单算子包自定义安装路径
    source ${install_path}/vendors/custom_transformer/bin/set_env.bash
    # experimental自定义单算子包默认安装路径${ASCEND_HOME_PATH}/opp/vendors
    # export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/opp/vendors/custom_transformer/op_api/lib/:$LD_LIBRARY_PATH
    ```

## torch_ops_extension包编译安装（可选）

根据[算子列表](#算子列表)章节介绍，如需使用`torch`接口进行算子调用，则需前置安装torch_ops_extension包。

torch_ops_extension包为自定义算子提供了torch.ops的拓展接口，具体编译安装方法请见[torch_ops_extension安装方法](https://gitcode.com/cann/cann-recipes-infer/tree/master/ops/ascendc#torch_ops_extension%E7%AE%97%E5%AD%90%E5%8C%85%E7%BC%96%E8%AF%91%E4%B8%8E%E5%AE%89%E8%A3%85)。

## 自定义算子执行
项目中各算子通过pytest验证各算子的功能是否正常，各算子的pytest调用方法如下表。
<table><thead>
  <tr>
    <th rowspan="2">算子名</th>
    <th rowspan="2">pytest示例</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>compressor</td>
    <td><a href="./compressor/tests/pytest/README.md">示例</a></td>
  </tr>
  <tr>
    <td>kv_quant_sparse_attn_sharedkv</td>
    <td><a href="./kv_quant_sparse_attn_sharedkv/tests/pytest/README.md">示例</a></td>
  </tr>
  <tr>
    <td>quant_lightning_indexer</td>
    <td><a href="./quant_lightning_indexer/tests/pytest/README.md">示例</a></td>
  </tr>
  <tr>
    <td>sparse_attn_sharedkv</td>
    <td><a href="./sparse_attn_sharedkv/tests/pytest/README.md">示例</a></td>
  </tr>
  <tr>
    <td>stem_indexer</td>
    <td><a href="./stem_indexer/tests/pytest/README.md">示例</a></td>
  </tr>
</tbody>
</table>
