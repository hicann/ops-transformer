# MoeDistributeDispatch and MoeDistributeCombine Operator Design Introduction

**This operator design introduction is based on <term>Atlas A2 products</term>**

## 1. Background and Overall Design Solution

### 1.1 Communication Bottlenecks and Challenges of the MoE Architecture

#### 1.1.1 Overview of the MoE Architecture

In the field of large-scale model training and inference, the Mixture of Experts (MoE) architecture has become the core technical solution supporting models with hundreds of billions of parameters, due to the computational sparsity advantage brought by its dynamic expert activation mechanism. Through two key operations, **Dispatch** and **Combine**, this architecture dynamically allocates input data and efficiently integrates the outputs of multiple experts, maintaining a massive parameter scale while ensuring computational efficiency.

However, as the scale of Expert Parallelism (EP) continues to expand, the high communication overhead caused by frequent data exchange between expert nodes has gradually become a key bottleneck restricting the inference performance of large models.

#### 1.1.2 Limitations of Traditional Communication Solutions

**Efficiency Defects of AllToAllV Communication**

Under the dynamic expert selection mechanism, the target experts to which each token is distributed show a discrete distribution, resulting in:

- **Uneven data distribution**: The numbers of tokens received by different experts differ significantly, forcing reliance on inefficient AllToAllV communication;
- **Metadata synchronization overhead**: Obtaining the send and receive information requires calling a preceding AllGather operator to collect the routing table and completing synchronization on the host side, which introduces additional communication overhead and stream synchronization latency.

**Small Data Packets and the Host-Bound Problem**

In inference scenarios, the token data volume is usually small, which causes two challenges:

- **Operator launch latency**: Traditional host-driven communication requires constructing a subgraph and performing scheduling, and its launch latency grows linearly with the EP scale;
- **RDMA synchronization overhead**: The pre- and post-synchronization processes of RDMA communication introduce additional RTT latency.

### 1.2 Innovative Solution Design

#### 1.2.1 Communication-Computation Fusion Operator Architecture

Based on the preceding bottleneck analysis, the two communication-computation fusion operators **MoeDistributeDispatch** and **MoeDistributeCombine** were developed.

The MoE architecture of the DeepSeekV3 model adopts a dynamic routing mechanism, where each token dynamically selects topK experts for processing. Specifically:

- The **Dispatch operation** performs the core scheduling function. Based on the routing mapping table between tokens and experts, it adopts a distributed computing strategy: first, the computing task of the number of tokens to be processed by each expert node is offloaded to the corresponding device for execution; then, AllToAllV communication completes the cross-device transfer of tokens; at the same time, the parameters required by the Combine phase are pre-computed;
- The **Combine operation** integrates the computation results output by each expert, performs the weighted summation, and restores the processed token data to the original positions through the reverse AllToAllV communication, completing the collaboration and integration of the entire distributed expert computation.

#### 1.2.2 Technical Advantages

The Dispatch/Combine operations are essentially a tight combination of computation and communication. Compared with traditional AllToAllV communication, the communication-computation fusion operators achieve the following breakthroughs:

- Host-side logic such as routing computation is offloaded to the device side, completely eliminating the synchronization overhead between the host and the device;
- Pipeline parallelism between part of the computation in the Combine operation and the AllToAllV communication is achieved, effectively hiding the computation and communication time consumption.

### 1.3 RDMA Full-Mesh Solution Based on the AIV+AICPU Fusion Architecture

#### 1.3.1 Architecture Overview

The RDMA full-mesh (Fullmesh) solution is built based on the AIV+AICPU fusion architecture, which fully exploits the computation and communication capabilities of Ascend hardware NPUs.

#### 1.3.2 Processing Flow

**Pre-processing Phase (AIV)**

- Obtain the routing information of each token.
- Reorder the tokens by expert index and aggregate the data sent to the same target rank.
- Complete the data sending to all experts on a target rank in a single communication, significantly reducing the RDMA launch latency.

**Communication Driving (AICPU)**

- The AIV transfers the address and length information of the data in the shared memory to the AICPU, which is also on the device side, through the message area in the GM.
- The AICPU directly drives RDMA communication, completely abandoning the cumbersome traditional process that requires the host side to construct a subgraph and schedule RDMA tasks.
- This solves the problem of long host-side processing time and eliminates the additional latency caused by the traditional scheduling method.

**Communication Waiting and Post-processing (AIV)**

- During communication, the AIV polls the data reception flag to ensure that the token data of all ranks is fully received, eliminating the communication latency caused by RDMA synchronization.
- The AIV moves the data in the shared memory out and aggregates it by expert, preparing data for the subsequent FFN layer computation.

#### 1.3.3 Technical Value

This series of optimizations forms a complete low-latency processing loop, achieving:

- **Communication-computation fusion**: Deep integration of communication preparation and computing tasks.
- **Device-side autonomy**: Reduced host-side intervention and improved processing efficiency.
- **Full-process optimization**: End-to-end performance improvement from data pre-processing to post-processing.

## 2. MoeDistributeDispatch Implementation Solution

### 2.1 Overview

The implementation solution of the MoeDistributeDispatch operator builds a complete data distribution processing pipeline. Through the close collaboration of three core phases, this solution achieves full-process optimization from token routing computation to cross-device distribution:

1. **Index computation and token reordering phase**: Computes accurate data distribution indexes based on the dynamic expert allocation results, and optimizes the communication data layout through memory reordering
2. **Data sending and receiving synchronization phase**: Uses efficient batch communication interfaces to implement parallel data distribution to multiple target nodes, with a built-in device-side synchronization mechanism
3. **Post-send processing phase**: Structurally reorganizes the received data and generates the metadata information required by downstream computation

### 2.2 Index Computation and Token Reordering

#### 2.2.1 Design Background

Index computation is performed around the expertIds input matrix, whose dimensions are BS×K, where expertIds(i,j) indicates the index of the expert to which the i-th token is assigned. The Combine operator requires the expandIdx output provided by the Dispatch operator. This matrix also has the BS×K dimensions, and expandIdx(i,j) indicates, from a global perspective, the sequence number of the i-th token among the tokens sent to the expert expertIds(i,j).

The token reordering operation aims to optimize communication efficiency. Since each communication launch has a fixed time overhead, the data sent to the same rank needs to be stored contiguously in the GM memory to minimize the number of launches. This operation is essentially the implementation of the MoePermute function.

#### 2.2.2 Implementation Solution

##### Send Status Matrix Design

Build the sendStatus matrix with the dimensions worldSize×STATUS_ENTRY_COUNT, where STATUS_ENTRY_COUNT is defined as the constant 32. The matrix layout is as follows:

- **Counting area**: The first FLAG_OFFSET elements (FLAG_OFFSET=24) record the numbers of sent tokens. sendStatus(i,j) indicates the number of tokens sent by the local card to the j-th expert on the i-th card
- **Flag area**: The FLAG_OFFSET-th element (counting from 0) is the synchronization flag, used for subsequent reception synchronization

Constraint: localExpertNum ≤ FLAG_OFFSET

##### Index Computation Flow

Traverse the expertIds matrix. For each element expertIds(i,j):

1. Target expert index = expertIds(i,j)
2. Target rank index = ceil(expertIds(i,j) / localExpertNum)
3. Local index of the expert on the target rank = expertIds(i,j) % localExpertNum
4. expandIdx(i,j) = sendStatus(target rank index, local expert index)
5. Update sendStatus(target rank index, local expert index)++

##### Token Reordering Mechanism

Build the expert token count prefix sum array expertCumsum. Three parameters are required to determine the token reordering position:

1. Target rank index = ceil(expertIds(i,j) / localExpertNum)
2. Total number of tokens of preceding experts = expertCumsum(expertIds(i,j)) - expertCumsum(ceil(expertIds(i,j)/localExpertNum) × localExpertNum)
3. Sequence number of the token in the target expert = expandIdx(i,j)

The exact position after reordering is calculated using the preceding parameters.

### 2.3 Data Sending and Receiving Synchronization

#### 2.3.1 Design Background

The BatchWrite interface is used for data sending. The input of this interface is a GM pointer to a structure array. Each structure contains the following fields:

- HCCLBUFFER: The GM memory area managed by HCCL, including WindowsIn (receiving buffer) and WindowsOut (sending buffer)

Features of the BatchWrite interface:

- No built-in synchronization mechanism
- Each launch has a fixed time overhead (1-2us)
- Design requirements: Minimize the number of launches and implement reception synchronization on the operator side

#### 2.3.2 Implementation Solution

##### Window Allocation Strategy

Divide WindowsIn and WindowsOut evenly into worldSize windows. Each window corresponds to a rank and stores the contiguous data sent to that rank. The data structure in a window:

- Token count array + FLAG1: Corresponds to the corresponding row of the sendStatus matrix
- Tokens data: The reordered token data
- FLAG2: Its position is related to the number of tokens and is determined dynamically

##### Reception Synchronization Mechanism

The dual-loop waiting strategy by core is adopted:

1. **Rank assignment**: All ranks are evenly assigned to each core
2. **First loop**: Poll the FLAG1 value until it is refreshed to a specific value, confirming that the token count array has been received
3. **Data processing**: Sum the token count array to determine the FLAG2 position
4. **Second loop**: Poll the FLAG2 value until it is refreshed to a specific value, confirming that all data has been received

### 2.4 Post-send Processing

#### 2.4.1 Design Background

After the data reception is complete, the following processing is required:

1. **Data reordering**: Reorder the actual data content according to the metadata to ensure that the tokens of the same expert remain sequentially contiguous in the GM memory
2. **Metadata computation**: Generate the two key outputs epRecvCount and expertTokenNum

Assume that the system contains w cards and each card deploys e experts. The received metadata matrix has the dimensions w×e. The epRecvCount output requires a transpose operation on this matrix, followed by a prefix sum computation in row-major order.

The expertTokenNum output is the prefix sum array of the token counts of each local expert, that is, the last column of epRecvCount.

#### 2.4.2 Implementation Solution

The combination of the Add+GatherMask+Adds interfaces is used to efficiently implement the transpose + cumulative sum operations, avoiding the performance loss caused by scalar operations.

Processing flow framework:

1. **Row-wise accumulation**: Use the Add interface. Starting from row 0, add the data of the previous row to the current row in order, completing the column-direction prefix sum pre-processing
2. **Matrix transpose**: Use the GatherMask interface to transpose the entire column of data at one time
3. **Row prefix sum**: Use the Adds interface. Starting from row 0, add the last element of the previous row to the current row, completing the final prefix sum computation

#### 2.4.3 Inter-Operator Data Synchronization Mechanism

##### Design Background

In the distributed expert parallelism architecture, the data processing speeds of different computing nodes differ, which may cause data races and consistency problems. Specifically:

- **Unbalanced computing load**: The amounts of data processed by different ranks may differ significantly
- **Asynchronous execution timing**: A fast rank may enter the next phase before a slow rank completes data processing
- **Data integrity risk**: The combine operation may overwrite the dispatch data that has not been processed

@startuml
title Single-Buffer Data Race Sequence Diagram

participant "Rank A" as RankA
participant "Rank A WindowsIn" as WinA
participant "Rank B" as RankB

== Initial state ==
note over RankA, RankB: All nodes use a single buffer

== Rank A Dispatch phase ==
RankA -> WinA: Read data (in progress)
RankA -> WinA: Post-processing operation

== Rank B Combine phase ==
RankB -> WinA: Prepare to write data

== Timing conflict points ==

group Data race: simultaneous read and write on the same memory region
  RankB -> WinA: Start writing the Combine result
  RankA -> WinA: Still reading the Dispatch data
end

group Data integrity problem
  WinA --> RankA: Read part of the old data + part of the new data
  note right WinA: Computation accuracy exception
end

group Flag conflict
  WinA --> RankA: The flag is overwritten by data
  note right WinA: Deadlock risk
end

@enduml

As shown in the preceding figure, when the fast rank (rank b) performs the combine operation, the dispatch post-processing of the slow rank (rank a) may not be completed. Since no synchronization interface blocks the combine data sending of the fast rank, the following problems may occur:

- Dirty writing of the data of the slow rank that has not been processed, causing computation accuracy exceptions
- Trampling of the flag and token data, causing a lost update problem on the flag, which puts it into an invalid state and further creates a deadlock risk.

##### Implementation Solution

The dual-buffer mechanism is used to implement data synchronization. The core design is as follows:

**Dual-Buffer Architecture**

- Divide both WindowIn (receiving buffer) and WindowOut (sending buffer) into two storage blocks of equal size
- Set the bufferChosen flag at the end of the first block of WindowIn (at the offset of 1MB)

**Buffer Selection Logic**

- The Dispatch and Combine operators read the bufferChosen flag during initialization
- If the flag is 0: use the first group of buffers (WindowIn/Out Block 0)
- If the flag is 1: use the second group of buffers (WindowIn/Out Block 1)
- Flip the flag before the operator execution completes: bufferChosen = bufferChosen ^ 1

**Synchronization Guarantee Mechanism**

The correctness of the dual-buffer design is based on the following timing characteristics:

- Each card must receive the communication flags sent by all other cards before it can end.
- If the Dispatch and Combine are collectively referred to as EP operators, when the N-th EP operator of a card has not ended, the (N+1)-th EP operator of other cards must not have ended, because the other cards cannot receive the communication flag of the (N+1)-th EP operator of the current card.
- When the (N+2)-th EP operator of other cards starts, the N-th EP operator of the current card must have ended.
- The data buffers used by the (N+2)-th EP operator and the N-th EP operator can be safely reused.

This solution implements synchronization through buffer rotation, ensuring data correctness while avoiding the performance overhead of the whole-card synchronization mechanism at the end of each operator.

# 3. MoeDistributeCombine Implementation Solution

## 3.1 Overview

The MoeDistributeCombine operator integrates and restores the results of the distributed expert computation, converting the outputs of multiple experts to the original input format.

## 3.2 Token Reordering (ReorderToken)

### 3.2.1 Design Background

In the Fullmesh architecture, the Combine operator uses the BatchWrite interface of the HCCL high-level API for inter-card communication. Each launch of a communication task on this interface has a fixed time overhead. To minimize the communication scheduling overhead, the data layout needs to be optimized through token reordering:

- **Communication efficiency optimization**: Store the tokens sent to the same target card contiguously in the GM memory to implement single batch sending
- **Buffer management**: Determine the number of tokens distributed to each target card based on the sendCounts metadata passed by Dispatch
- **Memory layout optimization**: Append the token data in order to the corresponding area of the WindowsOut buffer by target card partition

WindowsIn and WindowsOut are GM memory areas managed by HCCL, collectively referred to as HCCLBUFFER, and are used for data reception and sending respectively.

### 3.2.2 Implementation Solution

In the sendCounts matrix in prefix sum form, each element (i, j) indicates the cumulative number of tokens sent by expert i to rank j (from rank 0 to rank j). The following uses the data flow of card 0 sending to card 1 as an example to describe the processing in detail.

#### Prefix Sum Matrix Example

Assume that the sendCounts matrix of card 0 (in prefix sum form) is as shown in the following table:

| Expert index | rank 0 | rank 1 | rank 2 | rank 3 |
|----------|--------|--------|--------|--------|
| 0        | 1      | 1      | 2      | 3      |
| 1        | 3      | 8      | 10     | 12     |

- **Expert 0**: The number of tokens sent to rank 1 = sendCounts(0,1) - sendCounts(0,0) = 1 - 1 = 0
- **Expert 1**: The number of tokens sent to rank 1 = sendCounts(1,1) - sendCounts(1,0) = 8 - 3 = 5

#### Data Start Position Computation

In the expandX buffer, tokens are stored in order, starting from index 0.

- **Expert 0**: The start position index = sendCounts(0,0) = 1, corresponding to the token with index 1 in the expandX buffer (that is, the second token). Since the number of tokens is 0, no data needs to be read.
- **Expert 1**: The start position index = sendCounts(1,0) = 3, corresponding to the token with index 3 in the expandX buffer (that is, the fourth token). Read 5 contiguous tokens starting from this position.

#### Variable Definitions

Key variables involved in the implementation and their meanings:

- `expandX`: The input data buffer, which stores the token data to be sent
- `startTokenAddr`: The start address offset of the token in the expandX buffer
- `axisH`: The number of data elements in a single token
- `windowOutGM`: The first address of the global memory of the WindowsOut buffer
- `dstRankId`: The card ID of the target rank
- `rankSizeOnWin`: The size of the space allocated to a single rank in the WindowsOut buffer
- `localOutWindow`: The local window address of the current target rank in WindowsOut
- `rankTokenNum`: The counter of the number of tokens processed for the current target rank
- `expertId`: The expert ID
- `localMoeExpertNum`: The number of MoE experts on the local card
- `worldSize`: The number of ranks in the EP communicator
- `tokenNum`: The number of tokens sent by the current expert to the target rank

#### Implementation Flow

1. **Target rank window initialization**:

   - Allocate an independent output window for each target rank. The base address is calculated as: `windowOutGM + dstRankId × rankSizeOnWin`
   - Initialize the token counter of the target rank (`rankTokenNum`), which is used to track the data writing position in the current rank window

2. **Expert-level data processing**:

   - Traverse each local expert (from 0 to `localMoeExpertNum-1`) and calculate the range of the number of tokens sent to the target rank
   - Determine the number of tokens using the prefix sum difference: `tokenNum = sendCounts[expertId, dstRankId] - sendCounts(expertId, dstRankId-1)` (for `dstRankId=0`, the predecessor value is set to 0)
   - Locate the start address of the tokens in the expandX buffer based on the prefix sum value: `startTokenAddr = sendCounts(expertId, dstRankId-1) × axisH`

3. **Data copy execution**:

   - The data is loaded from the input GM (expandX) to the local UB buffer and then written to the WindowsOut window of the target rank
   - For each token, the data copy operations are as follows:
     - GM-to-UB: Read data from `expandX + startTokenAddr` to the UB
     - UB-to-WindowOut: Append the UB data to the current tail of the target rank window (`localOutWindow + rankTokenNum × axisH`)
   - Dynamically update `startTokenAddr` and `rankTokenNum` to ensure data continuity and integrity

## 3.3 Data Sending and Receiving Synchronization

### 3.3.1 Design Background

After token reordering is complete, the data sending phase begins. The BatchWrite interface is used for communication scheduling:

**Features of the BatchWrite interface**:

- The input is a GM pointer to a structure array of communication tasks
- Each structure corresponds to an independent communication task
- The time overhead for the AICPU to launch a single communication task to RoCE is about 1~2us

**Communication task structure**:

| Field Type | Field Name | Description |
|---------|---------|------|
| UINT64 | localBuf | Window address of the data sent by the local end |
| UINT64 | remoteBuf | Window address for the peer to receive data |
| UINT64 | count | Number of data elements sent by this communication task |
| UINT32 | dataType | Data type sent by this communication task |
| UINT32 | remoteRankId | ID of the destination card to which this communication task sends data |

**Usage principles**:

- BatchWrite lacks a built-in synchronization mechanism. Developers need to implement a synchronization mechanism to ensure data reception integrity
- The number of communication task launches directly affects performance

### 3.3.2 Implementation Solution

**Window allocation strategy**:

- Divide WindowsIn and WindowsOut evenly into worldSize windows
- Each window corresponds to a rank and stores contiguous communication data
- Batch communication is implemented by a single launch for a single rank

**Synchronization mechanism design**:

- Add a special flag marker at the tail of the sent data
- The receiving end adopts the loop waiting strategy by core:
  - Distribute the worldSize ranks evenly to the computing cores
  - Each core is responsible for querying the reception status of several ranks
  - Poll the flag value until it is refreshed to a specific identifier, confirming that the data reception is complete

**WindowsOut sending buffer**

| Rank 0 window | Rank 1 window | Rank 2 window | ... | Rank (n-1) window |
|-------------|-------------|-------------|-----|----------------|
| Token data area  | Token data area  | Token data area  | ... | Token data area     |
| FLAG marker    | FLAG marker    | FLAG marker    | ... | FLAG marker       |
| ⬇️           | ⬇️           | ⬇️           | ... | ⬆️              |
| Sent to Rank 0 | Sent to Rank 1 | Sent to Rank 2 | ... | Sent to Rank (n-1) |

**WindowsIn receiving buffer**

| Rank 0 window | Rank 1 window | Rank 2 window | ... | Rank (n-1) window |
|-------------|-------------|-------------|-----|----------------|
| Token data area  | Token data area  | Token data area  | ... | Token data area     |
| FLAG marker    | FLAG marker    | FLAG marker    | ... | FLAG marker       |
| ⬆️           | ⬆️           | ⬆️           | ... | ⬆️              |
| From Rank 0  | From Rank 1  | From Rank 2  | ... | From Rank (n-1) |

**Layout description**

- **Number of windows**: `worldSize`
- **Window size**: Each window occupies a memory space of `rankSizeOnWin`
- **Memory arrangement**: The windows of all ranks are arranged contiguously from low addresses to high addresses
- **Data structure**:
  - Token data area: Contiguously stores the tokens to be sent or received
  - FLAG marker area: Located at the tail of the window, used for communication synchronization confirmation

## 3.4 Weighted Sum (Sum)

### 3.4.1 Design Background

In the MoE architecture, each token is distributed to K experts for processing. The Combine phase needs to:

- **Data integration**: Integrate and restore the expert output tokens, which are K times the original input
- **Weighted computation**: Perform the weighted summation based on the weight coefficient of each expert
- **Format restoration**: Restore the original data format of the Dispatch input

The key challenge is that tokens are distributed non-contiguously in WinIn, requiring precise positioning based on routing metadata.

### 3.4.2 Implementation Solution

**Token positioning mechanism**:

Based on the expertIds expert index table and the expandIdx input, calculate the address of the token in the buffer:

1. **Expert token statistics**: Count the number of tokens received by each expert based on expertIds
2. **Offset computation**: Calculate the token offset of each expert within the WinIn window of its owning card
   - Expert offset = cumulative number of tokens of the preceding experts
   - Stored in the expertWindowOffset array
3. **Byte offset conversion**: Token count offset × token byte count H

**Address calculation formula**:

TokenAddr(i, j) = windowInGM + rankSizeOnWin × rank + expertWindowOffset(expertId) × H + expandIdx(i, j) × H

Where:

- TokenAddr(i, j): The address in WinIn of the j-th expert copy of the i-th token of Dispatch
- windowInGM: The first address of the WinIn buffer
- rankSizeOnWin: The size allocated to a single card in WinIn
- rank: The ID of the card to which the expert expertId belongs
- expertWindowOffset(expertId): The token offset of the expert within its owning card
- expandIdx(i, j): The sequence number of the token in the target expert
- H: The number of bytes of a single token

**Weighted summation flow**:

1. Traverse all tokens in the expert index table
2. Locate the output data of each expert based on the address formula
3. Perform the weighted summation computation based on the weight coefficients
4. Output the restored data in the original format
