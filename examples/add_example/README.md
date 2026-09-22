# AddExample

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
|Atlas A3系列产品|√|
|Atlas A2系列产品|√|

## 功能说明

- 算子功能：完成加法计算。

- 计算公式：

$$
y = x1 + x2
$$

## 参数说明

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 330px">
  <col style="width: 120px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出/属性</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x1</td>
      <td>输入</td>
      <td>待进行add_example计算的入参，公式中的x1。</td>
      <td>FLOAT、INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>输入</td>
      <td>待进行add_example计算的入参，公式中的x2。</td>
      <td>FLOAT、INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>待进行add_example计算的出参，公式中的y。</td>
      <td>FLOAT、INT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

<table><thead>
  <tr>
    <th>调用方式</th>
    <th>调用样例</th>
    <th>说明</th>
  </tr></thead>
<tbody>
  <tr>
    <td>aclnn调用</td>
    <td><a href="./examples/test_aclnn_add_example.cpp">test_aclnn_add_example</a></td>
    <td rowspan="2">快速调用示例如下文所示，仅供参考，详细算子编译运行方法参见<a href="../../docs/zh/invocation/quick_op_invocation.md">算子调用</a>。</td>
  </tr>
</tbody>
</table>

```bash
## 快速启动
cd example/add_example
# 假设已准备好环境变量
bash build.sh --soc=${soc_version} --ops=add_example
bash build.sh --run_example add_example eager cus
```
