# MoeDistributeDispatch和MoeDistributeCombine算子设计介绍

**本篇算子设计介绍基于<term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>**

## 1. 背景与总体设计方案

### 1.1 MoE架构的通信瓶颈与挑战

#### 1.1.1 MoE架构概述
在大规模模型训练与推理领域，混合专家（Mixture of Experts，MoE）架构凭借其动态专家激活机制所带来的计算稀疏性优势，已成为支撑千亿参数级模型的核心技术方案。该架构通过**分发（Dispatch）** 与**组合（Combine）** 两大关键操作，实现了输入数据的动态分配与多专家输出的高效整合，在维持海量参数规模的同时保障了计算效率。

然而，随着专家并行（Expert Parallelism，EP）规模的持续扩展，专家节点间频繁的数据交互所引发的高额通信开销，已逐渐成为制约大模型推理性能的关键瓶颈。

#### 1.1.2 传统通信方案的局限性

**AllToAllV 通信的效率缺陷**
在动态专家选择机制下，每个Token被分发的目标专家呈现离散分布特征，导致：
- **数据分发不均**：不同专家接收的Token数量存在显著差异，不得不依赖低效的AllToAllV通信；
- **元数据同步开销**：获取收发信息需调用前置AllGather算子收集路由表，并在Host侧完成同步，引入额外通信开销与Stream同步延迟。

**小数据包与Host Bound问题**
在推理场景中，Token数据量通常较小，引发双重挑战：
- **算子下发延迟**：传统Host驱动通信需构造子图并进行调度，其下发时延随EP规模线性增长；
- **RDMA同步开销**：RDMA通信的前后同步过程引入额外的RTT时延。

### 1.2 创新解决方案设计

#### 1.2.1 通算融合算子架构
基于上述瓶颈分析，开发了**MoeDistributeDispatch**与**MoeDistributeCombine**两个通算融合算子。

在DeepSeekV3模型的MoE架构中，采用动态路由机制，每个Token动态选择topK个专家进行处理。其中：
- **Dispatch操作**承担核心调度功能，基于Token与专家的路由对应关系表，采用分布式计算策略：首先将各专家节点需处理的Token数量计算任务下沉至对应设备执行，随后通过AllToAllV通信完成Token的跨设备传输，同时预计算Combine阶段所需参数；
- **Combine操作**负责整合各专家输出的计算结果，执行加权求和，并通过逆向的AllToAllV通信将处理后的Token数据恢复至原始位置，完成整个分布式专家计算的协同与整合。

#### 1.2.2 技术优势
Dispatch/Combine操作本质上是计算与通信的紧密结合。通算融合算子相较于传统的AllToAllV通信实现了以下突破：
- 将路由计算等Host侧逻辑下沉至Device侧，彻底消除Host与Device间的同步开销；
- 实现Combine操作中部分计算与AllToAllV通信的流水并行，有效掩盖计算与通信耗时。

### 1.3 基于AIV+AICPU融合架构的RDMA全互联方案

#### 1.3.1 架构概述
我们基于AIV+AICPU融合架构构建了RDMA全互联（Fullmesh）方案，充分发挥了昇腾硬件NPU的计算与通信能力。

#### 1.3.2 处理流程

**预处理阶段（AIV）**
- 获取每个Token的路由信息。
- 依照专家索引对Token进行重排，将发往同一目标rank的数据汇聚。
- 实现单次通信完成目标rank上所有专家的数据发送，显著减少RDMA下发时延。

**通信驱动（AICPU）**
- AIV将数据在共享内存中的地址、长度信息通过GM中的消息区传递给同处Device侧的AICPU。
- AICPU直接驱动RDMA通信，彻底摒弃传统需要Host侧构造子图和调度RDMA任务的繁琐流程。
- 解决Host侧处理耗时长的问题，消除传统调度方式带来的额外时延。

**通信等待与后处理（AIV）**
- 在通信环节，AIV轮询数据接收Flag，确保所有rank的Token数据全部接收完成，消除RDMA同步带来的通信时延。
- AIV将共享内存中的数据按照专家汇总搬出，为后续FFN层的计算提供数据准备。

#### 1.3.3 技术价值
这一系列优化形成了完整的低延迟处理闭环，实现了：
- **通信计算融合**：将通信准备与计算任务深度融合。
- **设备侧自治**：减少Host侧干预，提升处理效率。
- **全流程优化**：从数据预处理到后处理的端到端性能提升。

## 2. MoeDistributeDispatch实现方案

### 2.1 概述

MoeDistributeDispatch算子的实现方案构建了一个完整的数据分发处理流水线。该方案通过三个核心阶段的紧密协作，实现了从Token路由计算到跨设备分发的全流程优化：

1. **索引计算与Token重排阶段**：基于动态专家分配结果，计算精确的数据分发索引，并通过内存重排优化通信数据布局
2. **数据发送与接收同步阶段**：利用高效的批量通信接口，实现多目标节点的并行数据分发，并内置设备侧同步机制
3. **发送后处理阶段**：对接收数据进行结构化重组，生成下游计算所需的元数据信息

### 2.2 索引计算与Token重排

#### 2.2.1 设计背景

索引计算围绕expertIds输入矩阵展开，该矩阵维度为BS×K，其中expertIds(i,j)表示第i个Token被分配给第j个专家的索引。Combine算子需要Dispatch算子提供expandIdx输出，该矩阵同样为BS×K维度，expandIdx(i,j)表示在全局视角下，第i个Token是发送给专家expertIds(i,j)的第几个Token。

Token重排操作旨在优化通信效率。由于每次通信下发均存在固定时间开销，为最小化下发次数，需要将发送至同一rank的数据在GM内存中连续存放。该操作本质上是实现MoePermute功能。

#### 2.2.2 实现方案

##### 发送状态矩阵设计

构建sendStatus矩阵，维度为worldSize×STATUS_ENTRY_COUNT，其中STATUS_ENTRY_COUNT定义为常量32。矩阵布局如下：

- **计数区**：前FLAG_OFFSET个元素（FLAG_OFFSET=24）记录发送Token数量，sendStatus(i,j)表示本地卡向第i张卡第j个专家发送的Token数量
- **标志区**：第FLAG_OFFSET个元素（从0计数）为同步标志位，用于后续接收同步

约束条件：localExpertNum ≤ FLAG_OFFSET

##### 索引计算流程

遍历expertIds矩阵，对于每个元素expertIds(i,j)：

1. 目标专家索引 = expertIds(i,j)
2. 目标rank索引 = ceil(expertIds(i,j) / localExpertNum)
3. 专家在目标rank上的局部索引 = expertIds(i,j) % localExpertNum
4. expandIdx(i,j) = sendStatus(目标rank索引, 专家局部索引)
5. 更新sendStatus(目标rank索引, 专家局部索引)++

##### Token重排机制

构建专家Token数量前缀和数组expertCumsum，确定Token重排位置需要三个参数：

1. 目标rank索引 = ceil(expertIds(i,j) / localExpertNum)
2. 前置专家Token总数 = expertCumsum(expertIds(i,j)) - expertCumsum(ceil(expertIds(i,j)/localExpertNum) × localExpertNum)
3. Token在目标专家中的序号 = expandIdx(i,j)

通过上述参数计算得到重排后的确切位置。

### 2.3 数据发送与接收同步

#### 2.3.1 设计背景

采用BatchWrite接口进行数据发送，该接口输入为GM指针，指向结构体数组。每个结构体包含以下字段：

- HCCLBUFFER：由HCCL管理的GM内存区域，包括WindowsIn（接收缓冲区）和WindowsOut（发送缓冲区）

BatchWrite接口特性：
- 无内置同步机制
- 每次下发存在固定时间开销（1-2us）
- 设计要求：最小化下发次数，在算子侧实现接收同步

#### 2.3.2 实现方案

##### 窗口分配策略

将WindowsIn和WindowsOut平均划分为worldSize个窗口，每个窗口对应一个rank，存放发送至该rank的连续数据。窗口内数据结构：

- Token数量数组 + FLAG1：对应sendStatus矩阵的相应行
- Tokens数据：重排后的Token数据
- FLAG2：位置与Tokens数量相关，动态确定  

##### 接收同步机制

采用分核双循环等待策略：

1. **Rank分配**：将所有rank平均分配给每个核
2. **第一层循环**：轮询FLAG1值，直到刷新为特定值，确认Token数量数组接收完成
3. **数据处理**：对Token数量数组求和，确定FLAG2位置
4. **第二层循环**：轮询FLAG2值，直到刷新为特定值，确认全部数据接收完成

### 2.4 发送后处理

#### 2.4.1 设计背景

数据接收完成后，需要进行以下处理：

1. **数据重排**：依据元数据将实际数据内容重新排列，确保同一专家的Token在GM内存中保持顺序连续
2. **元数据计算**：生成epRecvCount和expertTokenNum两个关键输出

假设系统包含w张卡，每张卡部署e个专家，接收到的元数据矩阵维度为w×e。epRecvCount输出需要对该矩阵执行转置操作，然后按行主序进行前缀和计算。

expertTokenNum输出为本地每个专家的Token数量前缀和数组，即epRecvCount的最后一列。

#### 2.4.2 实现方案

采用Add+GatherMask+Adds接口组合，高效实现转置+累加和操作，避免标量操作带来的性能损失。

处理流程框架：

1. **行向累加**：使用Add接口，从第0行开始，按顺序将上一行数据对应加到当前行，完成列方向的前缀和预处理
2. **矩阵转置**：使用GatherMask接口，一次性转置整列数据
3. **行前缀和**：使用Adds接口，从第0行开始，将上一行的最后一个元素加到当前行，完成最终的前缀和计算

#### 2.4.3 算子间数据同步机制

##### 设计背景

在分布式专家并行架构中，不同计算节点的数据处理速度存在差异，可能导致数据竞争和一致性问题。具体而言：

- **计算负载不均衡**：不同rank处理的数据量可能存在显著差异
- **异步执行时序**：快rank可能在慢rank完成数据处理前进入下一阶段
- **数据完整性风险**：combine操作可能覆盖尚未完成处理的dispatch数据

@startuml
title 单缓冲区数据竞争时序图

participant "Rank A" as RankA
participant "Rank A WindowsIn" as WinA
participant "Rank B" as RankB

== 初始状态 ==
note over RankA, RankB: 所有节点使用单缓冲区

== Rank A Dispatch阶段 ==
RankA -> WinA: 读取数据（进行中）
RankA -> WinA: 后处理操作

== Rank B Combine阶段 ==
RankB -> WinA: 准备写入数据

== 时序冲突点 ==

group 数据竞争：同时读写同一内存区域
  RankB -> WinA: 开始写入Combine结果
  RankA -> WinA: 仍在读取Dispatch数据
end

group 数据完整性问题
  WinA --> RankA: 读取到部分旧数据+部分新数据
  note right WinA: 计算精度异常
end

group 标志位冲突
  WinA --> RankA: 标志位被数据覆盖
  note right WinA: 死锁风险
end

@enduml

如上图所示，快rank（rank b）在执行combine操作时，慢rank（rank a）的dispatch后处理可能尚未完成。由于缺乏同步接口来阻塞快rank的combine数据发送，可能导致以下问题：

- 写脏尚未完成处理的慢rank数据，引发计算精度异常
- 标志位和Token数据踩踏，引发标志位的丢失更新问题，使其进入了一个无效状态，进而产生死锁风险。



##### 实现方案

采用双缓冲机制实现数据同步，核心设计如下：

**双缓冲架构**
- 将WindowIn（接收缓冲区）和WindowOut（发送缓冲区）均划分为两个等大的存储块
- 在WindowIn第一块的末端（偏移1MB处）设置bufferChosen标志位

**缓冲区选择逻辑**
- Dispatch和Combine算子初始化时读取bufferChosen标志位
- 标志位为0：使用第一组缓冲区（WindowIn/Out Block 0）
- 标志位为1：使用第二组缓冲区（WindowIn/Out Block 1）
- 算子执行完成前翻转标志位：bufferChosen = bufferChosen ^ 1

**同步保证机制**
双缓冲设计的正确性基于以下时序特性：
- 每张卡都需要收到其他所有卡发来的通信Flag，才可以结束。
- 将Dispatch和Combine统称为EP算子，则某张卡的第N个EP算子还未结束时，其他卡的第N+1个EP算子必定还未结束，因为其他卡还无法收到当前卡第N+1个EP算子的通信Flag。
- 其他卡的第N+2个EP算子开始时，当前卡第N个EP算子必定已经结束。
- 第N+2个EP算子和第N个EP算子使用的数据缓冲区可安全重用。

该方案通过缓冲区轮转实现了同步，在保证数据正确性的同时避免了在每次算子结束时进行全卡同步机制的性能开销。


# 三、MoeDistributeCombine实现方案

## 3.1 概述

MoeDistributeCombine算子负责将分布式专家计算的结果进行整合与还原，实现从多专家输出到原始输入格式的转换。

## 3.2 Token重排（ReorderToken）

### 3.2.1 设计背景

Combine算子在Fullmesh架构中采用HCCL高阶API的BatchWrite接口进行卡间通信。该接口每次下发通信任务 均存在固定时间开销，为最小化通信调度开销，需要通过Token重排优化数据布局：

- **通信效率优化**：将发送至同一目标卡的Token在GM内存中连续存放，实现单次批量发送
- **缓冲区管理**：基于Dispatch传递的sendCounts元数据，确定各目标卡的Token分发数量
- **内存布局优化**：按照目标卡分区将Token数据有序追加至WindowsOut缓冲区的相应区域

WindowsIn和WindowsOut是由HCCL管理的GM内存区域，统称为HCCLBUFFER，分别用于数据接收和发送。

### 3.2.2 实现方案

在前缀和形式的sendCounts矩阵中，每个元素(i, j)表示专家i发送到rank j的累计Token数量（从rank 0到rank j）。以下以卡0发送至卡1的数据流程为例，详细说明处理过程。

#### 前缀和矩阵示例
假设卡0的sendCounts矩阵（前缀和形式）如下表所示：

| 专家索引 | rank 0 | rank 1 | rank 2 | rank 3 |
|----------|--------|--------|--------|--------|
| 0        | 1      | 1      | 2      | 3      |
| 1        | 3      | 8      | 10     | 12     |

- **专家0**：发送至rank 1的Token数量 = sendCounts(0,1) - sendCounts(0,0) = 1 - 1 = 0
- **专家1**：发送至rank 1的Token数量 = sendCounts(1,1) - sendCounts(1,0) = 8 - 3 = 5

#### 数据起始位置计算
在expandX缓冲区中，Token按顺序存储，起始索引为0。
- **专家0**：起始位置索引 = sendCounts(0,0) = 1，对应expandX缓冲区中索引为1的Token（即第2个Token）。由于Token数量为0，无需读取数据。
- **专家1**：起始位置索引 = sendCounts(1,0) = 3，对应expandX缓冲区中索引为3的Token（即第4个Token）。从此位置开始读取5个连续Token。

#### 变量定义说明
在实现过程中涉及的关键变量及其含义：

- `expandX`: 输入数据缓冲区，存储待发送的Token数据
- `startTokenAddr`: Token在expandX缓冲区中的起始地址偏移量
- `axisH`: 单个Token的数据个数
- `windowOutGM`: WindowsOut缓冲区的全局内存首地址
- `dstRankId`: 目标rank的卡号
- `rankSizeOnWin`: 单个rank在WindowsOut缓冲区中分配的空间大小
- `localOutWindow`: 当前目标rank在WindowsOut中的本地窗口地址
- `rankTokenNum`: 当前目标rank已处理的Token数量计数器
- `expertId`: 专家编号
- `localMoeExpertNum`: 本卡MoE专家数量
- `worldSize`: EP通信域rank数量
- `tokenNum`: 当前专家发送至目标rank的Token数量

#### 实现流程
1. **目标Rank窗口初始化**：
   - 为每个目标rank分配独立的输出窗口，基地址计算为：`windowOutGM + dstRankId × rankSizeOnWin`
   - 初始化目标rank的Token计数器（`rankTokenNum`），用于跟踪当前rank窗口内的数据写入位置

2. **专家级数据处理**：
   - 遍历每个本地专家（从0到`localMoeExpertNum-1`），计算发送至目标rank的Token数量范围
   - 使用前缀和差值确定Token数量：`tokenNum = sendCounts[expertId, dstRankId] - sendCounts(expertId, dstRankId-1)`（对于`dstRankId=0`，前驱值设为0）
   - 根据前缀和值定位expandX缓冲区中的Token起始地址：`startTokenAddr = sendCounts(expertId, dstRankId-1) × axisH`

3. **数据搬运执行**：
   - 数据从输入GM（expandX）加载到本地UB缓冲区，再写入目标rank的WindowsOut窗口
   - 对于每个Token，数据搬运操作如下：
     - GM-to-UB：从`expandX + startTokenAddr`读取数据到UB
     - UB-to-WindowOut：将UB数据追加到目标rank窗口的当前尾部（`localOutWindow + rankTokenNum × axisH`）
   - 动态更新`startTokenAddr`和`rankTokenNum`，确保数据连续性和完整性

## 3.3 数据发送与接收同步

### 3.3.1 设计背景

Token重排完成后进入数据发送阶段，采用BatchWrite接口进行通信调度：

**BatchWrite接口特性**：
- 输入为GM指针，指向通信任务结构体数组
- 每个结构体对应一个独立的通信任务
- AICPU下发单个通信任务到RoCE的时间开销约1~2us

**通信任务结构体**：
| 字段类型 | 字段名称 | 描述 |
|---------|---------|------|
| UINT64 | localBuf | 本端发送数据的window地址 |
| UINT64 | remoteBuf | 对端接收数据的window地址 |
| UINT64 | count | 该通信任务发送的数据个数 |
| UINT32 | dataType | 该通信任务发送的数据类型 |
| UINT32 | remoteRankId | 该通信任务发送数据的目的卡卡号 |

**使用原则**：
- BatchWrite缺乏内置同步机制，需要开发者实现同步机制确保数据接收完整性
- 通信任务下发次数直接影响性能

### 3.3.2 实现方案

**窗口分配策略**：
- 将WindowsIn和WindowsOut均等划分为worldSize个窗口
- 每个窗口对应一个rank，存放连续的通信数据
- 单rank单次下发实现批量通信

**同步机制设计**：
- 在发送数据尾部添加特殊Flag标记
- 接收端采用分核循环等待策略：
  - 将worldSize个rank平均分配给各计算核
  - 每个核负责若干rank的接收状态查询
  - 轮询Flag值直至刷新为特定标识，确认数据接收完成

**WindowsOut 发送缓冲区**

| Rank 0 窗口 | Rank 1 窗口 | Rank 2 窗口 | ... | Rank (n-1) 窗口 |
|-------------|-------------|-------------|-----|----------------|
| Token数据区  | Token数据区  | Token数据区  | ... | Token数据区     |
| FLAG 标记    | FLAG 标记    | FLAG 标记    | ... | FLAG 标记       |
| ⬇️           | ⬇️           | ⬇️           | ... | ⬆️              |
| 发送至 Rank 0 | 发送至 Rank 1 | 发送至 Rank 2 | ... | 发送至 Rank (n-1) |

**WindowsIn 接收缓冲区**

| Rank 0 窗口 | Rank 1 窗口 | Rank 2 窗口 | ... | Rank (n-1) 窗口 |
|-------------|-------------|-------------|-----|----------------|
| Token数据区  | Token数据区  | Token数据区  | ... | Token数据区     |
| FLAG 标记    | FLAG 标记    | FLAG 标记    | ... | FLAG 标记       |
| ⬆️           | ⬆️           | ⬆️           | ... | ⬆️              |
| 来自 Rank 0  | 来自 Rank 1  | 来自 Rank 2  | ... | 来自 Rank (n-1) |

**布局说明**

- **窗口数量**：`worldSize` 
- **窗口大小**：每个窗口占用 `rankSizeOnWin` 大小的内存空间
- **内存排列**：从低地址到高地址连续排列各个rank的窗口
- **数据结构**：
  - Token数据区：连续存储待发送/已接收的Token数据
  - FLAG标记区：位于窗口尾部，用于通信同步确认

## 3.4 加权求和（Sum）

### 3.4.1 设计背景

在MoE架构中，每个Token被分发至K个专家进行处理，Combine阶段需要：

- **数据整合**：将K倍于原始输入的专家输出Token整合还原
- **加权计算**：基于各专家权重系数进行加权求和
- **格式还原**：恢复至Dispatch输入的原始数据格式

关键挑战在于Token在WinIn中的分布不连续，需要基于路由元数据精确定位。

### 3.4.2 实现方案

**Token定位机制**：

基于expertIds专家索引表和expandIdx输入，计算Token在缓冲区中的地址：

1. **专家Token统计**：根据expertIds统计各专家接收的Token数量
2. **偏移量计算**：计算各专家在其归属卡WinIn窗口内的Token偏移量
   - 专家偏移量 = 前序专家累计Token数量
   - 存储于expertWindowOffset数组
3. **字节偏移转换**：Token数量偏移 × Token字节数H

**地址计算公式**：

TokenAddr(i, j) = windowInGM + rankSizeOnWin × rank + expertWindowOffset(expertId) × H + expandIdx(i, j) × H

其中：
- TokenAddr(i, j): Dispatch第i个Token的第j个专家副本在WinIn中的地址
- windowInGM: WinIn缓冲区首地址
- rankSizeOnWin: 单卡在WinIn中的分配大小
- rank: 专家expertId归属的卡编号
- expertWindowOffset(expertId): 专家在归属卡内的Token偏移量
- expandIdx(i, j): Token在目标专家中的序号
- H: 单个Token的字节数

**加权求和流程**：
1. 遍历专家索引表中的所有Token
2. 基于地址公式定位各专家输出数据
3. 根据权重系数执行加权求和计算
4. 输出还原后的原始格式数据