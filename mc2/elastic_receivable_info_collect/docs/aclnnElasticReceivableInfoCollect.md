# aclnnElasticReceivableInfoCollect

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>昇腾910_95 AI处理器</term>                             |    ×     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品 </term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |
| <term>Atlas 200/300/500 推理产品</term>                      |    ×     |

## 功能说明

算子功能：收集并整理检测通信域内的状态位，并将结果输出，供下一步检测流程判断全局联通情况。

## 函数原型

每个算子分为[两段式接口](../../../docs/context/两段式接口.md)，必须先调用 “aclnnElasticReceivableInfoCollectGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnElasticReceivableInfoCollect”接口执行计算。

```cpp
aclnnStatus aclnnElasticReceivableInfoCollectGetWorkspaceSize(
    const char         *group,
    int64_t             worldSize,
    const aclTensor    *y,
    uint64_t           *workspaceSize,
    aclOpExecutor     **executor);
```

```cpp
aclnnStatus aclnnElasticReceivableInfoCollect(
    void           *workspace,
    uint64_t        workspaceSize,
    aclOpExecutor  *executor,
    aclrtStream     stream)
```

## aclnnElasticReceivableInfoCollectGetWorkspaceSize

- **参数说明**
    <table style="undefined;table-layout: fixed; width: 1576px">
     <colgroup>
      <col style="width: 150px">
      <col style="width: 100px">
      <col style="width: 250px">
      <col style="width: 200px">
      <col style="width: 180px">
      <col style="width: 80px">
      <col style="width: 100px">
      <col style="width: 100px">
     </colgroup>
     <thead>
      <tr>
       <th>参数名</th>
       <th>输入/输出</th>
       <th>描述</th>
       <th>使用说明</th>
       <th>数据类型</th>
       <th>数据格式</th>
       <th>维度(shape)</th>
       <th>非连续Tensor</th>
      </tr>
     </thead>
     <tbody>
      <tr>
       <td>group</td>
       <td>输入</td>
       <td>ep通信域名称，专家并行的通信域。</td>
       <td>字符串长度范围为(0, 128)</td>
       <td>STRING</td>
       <td>-</td>
       <td>-</td>
       <td>-</td>
      </tr>
      <tr>
       <td>worldSize</td>
       <td>输入</td>
       <td>通信域大小。</td>
       <td>取值支持[16, 128]内16整数倍的数值</td>
       <td>INT64</td>
       <td>-</td>
       <td>-</td>
       <td>-</td>
      </tr>
      <tr>
       <td>y</td>
       <td>输出</td>
       <td>表示获取到的设备互联信息</td>
       <td>shape为(worldSize, worldSize)</td>
       <td>INT32</td>
       <td>ND</td>
       <td>2</td>
       <td>√</td>
      </tr>
      <tr>
       <td>workspaceSize</td>
       <td>输出</td>
       <td>返回需要在Device侧申请的workspace大小。</td>
       <td>-</td>
       <td>-</td>
       <td>-</td>
       <td>-</td>
       <td>-</td>
      </tr>
      <tr>
       <td>executor</td>
       <td>输出</td>
       <td>返回op执行器，包含了算子的计算流程。</td>
       <td>-</td>
       <td>-</td>
       <td>-</td>
       <td>-</td>
       <td>-</td>
      </tr>
     </tbody>
    </table>

- **返回值**

    返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。

    第一段接口完成入参校验，出现以下场景时报错：

    <table style="undefined;table-layout: fixed; width: 1576px"> <colgroup>
    <col style="width: 170px">
    <col style="width: 170px">
    <col style="width: 400px">
    </colgroup>
    <thead>
    <tr>
    <th>返回值</th>
    <th>错误码</th>
    <th>描述</th>
    </tr></thead>
    <tbody>
    <tr>
    <td>ACLNN_ERR_PARAM_NULLPTR</td>
    <td>161001</td>
    <td>传入的group或y是空指针。</td>
    </tr>
    <tr>
    <td rowspan="3" align="left">ACLNN_ERR_PARAM_INVALID</td>
    <td rowspan="3" align="left">161002</td>
    <td align="left">传入的y的数据类型不在支持的范围内。</td>
    </tr>
    <tr><td align="left">传入的y的数据格式不在支持的范围内。</td></tr>
    <tr><td align="left">传入的y的shape不匹配。</td></tr>
    </tbody></table>


## aclnnElasticReceivableInfoCollect

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1576px">
   <colgroup>
    <col style="width: 170px">
    <col style="width: 170px">
    <col style="width: 800px">
   </colgroup>
   <thead>
    <tr>
     <th>参数名</th>
     <th>输入/输出</th>
     <th>描述</th>
    </tr>
   </thead>
   <tbody>
    <tr>
     <td>workspace</td>
     <td>输入</td>
     <td>在Device侧申请的workspace内存地址。</td>
    </tr>
    <tr>
     <td>workspaceSize</td>
     <td>输入</td>
     <td>在Device侧申请的workspace大小，由第一段接口`aclnnElasticReceivableInfoCollectGetWorkspaceSize`获取。</td>
    </tr>
    <tr>
     <td>executor</td>
     <td>输入</td>
     <td>op执行器，包含了算子计算流程。</td>
    </tr>
    <tr>
     <td>stream</td>
     <td>输入</td>
     <td>指定执行任务的Stream。</td>
    </tr>
   </tbody>
  </table>

- **返回值**

    返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。

## 约束说明

无

