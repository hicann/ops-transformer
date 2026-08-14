# MhcPreSinkhornBackward

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- **算子功能**：`MhcPreSinkhornBackward`是`MhcPreSinkhorn`的反向算子，用于计算mHC（Manifold-Constrained Hyper-Connections）结构中Sinkhorn变换的反向梯度传播。
- **主要输出**：`gradX`、`gradPhi`、`gradAlpha`、`gradBias`。
- **前向缓存依赖**：`hPre`、`hcBeforeNorm`、`invRms`、`sumOut`、`normOut`。
- **计算公式**：

  $$
  \begin{aligned}
  gradX &= \nabla_{x}(\text{MhcPreSinkhorn}(x, \phi, \alpha, \text{bias})) \\
  gradPhi &= \nabla_{\phi}(\text{MhcPreSinkhorn}(x, \phi, \alpha, \text{bias})) \\
  gradAlpha &= \nabla_{\alpha}(\text{MhcPreSinkhorn}(x, \phi, \alpha, \text{bias})) \\
  gradBias &= \nabla_{\text{bias}}(\text{MhcPreSinkhorn}(x, \phi, \alpha, \text{bias}))
  \end{aligned}
  $$

## 参数说明

  <table style="table-layout: auto; width: 100%">
    <thead>
      <tr>
        <th style="white-space: nowrap">参数名</th>
        <th style="white-space: nowrap">输入/输出/属性</th>
        <th style="white-space: nowrap">描述</th>
        <th style="white-space: nowrap">数据类型</th>
        <th style="white-space: nowrap">数据格式</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>gradHin</td>
        <td>输入</td>
        <td>输出h_in的梯度。</td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>gradHPost</td>
        <td>输入</td>
        <td>输出h_post的梯度。</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>gradHRes</td>
        <td>输入</td>
        <td>输出h_res的梯度。</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>x</td>
        <td>输入</td>
        <td>前向输入x。</td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>phi</td>
        <td>输入</td>
        <td>前向参数phi。</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>alpha</td>
        <td>输入</td>
        <td>前向参数alpha。</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>bias</td>
        <td>输入</td>
        <td>前向参数bias。</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>hPre</td>
        <td>输入</td>
        <td>前向保存的中间结果h_pre。</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>hcBeforeNorm</td>
        <td>输入</td>
        <td>前向保存的中间结果hc_before_norm。</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>invRms</td>
        <td>输入</td>
        <td>前向保存的中间结果inv_rms。</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>sumOut</td>
        <td>输入</td>
        <td>Sinkhorn变换正向计算保存的中间sum结果。</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>normOut</td>
        <td>输入</td>
        <td>Sinkhorn变换正向计算保存的中间norm结果。</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>hcEps</td>
        <td>属性</td>
        <td>数值稳定性参数，默认值1e-6。</td>
        <td>DOUBLE</td>
        <td>-</td>
      </tr>
      <tr>
        <td>gradX</td>
        <td>输出</td>
        <td>输入x的梯度。</td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>gradPhi</td>
        <td>输出</td>
        <td>参数phi的梯度。</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>gradAlpha</td>
        <td>输出</td>
        <td>参数alpha的梯度。</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>gradBias</td>
        <td>输出</td>
        <td>参数bias的梯度。</td>
        <td>FLOAT32</td>
        <td>ND</td>
      </tr>
    </tbody>
  </table>

## 约束说明

输入支持两种维度格式：BSND（4维）和TND（3维）。其中T = B × S，表示所有Batch序列长度的累加和。各输入/输出在两种格式下的shape对应关系如下：

| 参数 | BSND（4维） | TND（3维） |
| :--- | :--- | :--- |
| gradHin | (B,S,C) | (T,C) |
| gradHPost | (B,S,N) | (T,N) |
| gradHRes | (B,S,N*N) | (T,N*N) |
| x | (B,S,N,C) | (T,N,C) |
| hPre | (B,S,N) | (T,N) |
| hcBeforeNorm | (B,S,N*N+2*N) | (T,N*N+2*N) |
| invRms | (B,S,1) | (T,1) |
| sumOut | (2*sk_iter_count,B,S,N) | (2*sk_iter_count,T,N) |
| normOut | (2*sk_iter_count,B,S,N,N) | (2*sk_iter_count,T,N,N) |
| gradX | (B,S,N,C) | (T,N,C) |

phi和gradPhi的shape记为（2N+N^2，N*C），不受维度格式影响。

- `sk_iter_count`当前仅支持`20`。
- `N`当前支持大于0且不超过`8`。
- `C`大于0、小于100000且可以被128整除。
- 所有带B/S维度的输入需保持维度格式一致（同为4维BSND或同为3维TND）。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| :--- | :--- | :--- |
| aclnn调用 | [test_aclnn_mhc_pre.cpp](examples/test_aclnn_mhc_pre_sinkhorn_backward.cpp) | 通过[aclnnMhcPreSinkhornBackward](docs/aclnnMhcPreSinkhornBackward.md)接口方式调用MhcPreSinkhornBackward算子。 |
