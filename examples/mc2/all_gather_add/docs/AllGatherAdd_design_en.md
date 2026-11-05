# AllGatherAdd Operator Design and Implementation Details

**This operator design and implementation introduction is based on <term>Atlas A2 products</term>**

## 1. Operator Analysis

### 1.1 Operator Logic

The AllGatherAdd operator fuses [AllGather](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/API/ascendcopapi/atlasascendc_api_07_0873.html) communication with the [Add](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/API/ascendcopapi/atlasascendc_api_07_0035.html) addition.

The operator logic is as follows: perform AllGather communication on the operand a of all cards in the communicator to obtain the communication result gather_out, which is the first operand of the Add computation, and then perform the Add operation on gather_out and the other input b to obtain output c.

- The corresponding mathematical expression is:

    $$
    gatherOut=AllGather(a0, a1)
    $$

    $$
    c[i]=gatherOut[i] + b[i]
    $$

### 1.2 Inputs, Outputs, and Attributes

- a and b are the source operands. a is the input operand of the communication, and its shape is fixed to [240, 256] in this sample. b is the addend of Add, and its shape is fixed to [480, 256] in this sample.
- gather_out is the destination operand that stores the AllGather communication result. Its shape is [240 * rank_size, 256], where rank_size is the number of cards in the communicator, which is fixed to 2 in this sample.
- c is the destination operand that stores the Add computation result. Its shape is [240 * rank_size, 256].
- The data type of the operator inputs and outputs is float16, and the [format](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/API/ascendcopapi/atlasascendc_api_07_0961.html) is ND.
- group is an operator attribute that indicates the communicator name, specifying the communicator in which the operator runs.


### 1.3 Kernel Function Name and Parameters

- In this sample, the kernel function is named all_gather_add.
- Based on the analysis of the operator inputs and outputs, the kernel function parameters are determined as aGM, bGM, cGM, and gatherOutGM. aGM and bGM are the memory addresses of the inputs in Global Memory, and cGM and gatherOutGM are the memory addresses of the outputs in Global Memory.

### 1.4 Interfaces Required for Operator Implementation

| Interface | Description | Reference Link |
|--|--------------------------------------------------------------------------|--|
| AllGather | The operator involves AllGather communication, so the Hccl high-level API is required to implement AllGather communication. |https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/commlib/hcclug/hcclug_000001.html  |
| DataCopy | The operator involves data movement of the Add augend and addend between external storage and internal storage, so DataCopy is required to implement the data movement. | https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/API/ascendcopapi/atlasascendc_api_07_0103.html  |
| Add | The computation process involves Add vector computation operations, so the Add basic arithmetic API is required to implement the addition. | https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/API/ascendcopapi/atlasascendc_api_07_0035.html |

### 1.5 Operator Specification

<table border="1" style="table-layout: fixed; width: 100%">
    <colgroup>
        <col style="width: 25%">
        <col style="width: 15%">
        <col style="width: 15%">
        <col style="width: 20%">
        <col style="width: 25%">
    </colgroup>
    <tr>
        <th>Operator Type (OpType)</th>
        <td colspan="4">AllGatherAdd</td>
    </tr>
    <tr>
        <th rowspan="5">Operator Inputs and Outputs</th>
        <th>name</th>
        <th>dataType</th>
        <th>shape</th>
        <th>format</th>
    </tr>
    <tr>
        <td>a</td>
        <td>float16</td>
        <td>(240,256)</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>b</td>
        <td>float16</td>
        <td>(480,256)</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>c</td>
        <td>float16</td>
        <td>(480,256)</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>gather_out</td>
        <td>float16</td>
        <td>(480,256)</td>
        <td>ND</td>
    </tr>
    <tr>
        <th rowspan="2">Operator Attributes</th>
        <td>rank_size</td>
        <td>int32</td>
        <td>2</td>
        <td>\</td>
    </tr>
    <tr>
        <td>group</td>
        <td>string</td>
        <td colspan="2">Communicator. It cannot be constructed manually and must be dynamically requested by calling the hccl interface before the operator is invoked.</td>
    </tr>
    <tr>
        <th>Kernel Function Name</th>
        <td colspan="4">all_gather_add</td>
    </tr>
    <tr>
        <th rowspan="5">Key Interfaces</th>
        <td colspan="4">DataCopy: data movement interface</td>
    </tr>
    <tr>
        <td colspan="4">Add: vector binary instruction interface</td>
    </tr>
    <tr>
        <td colspan="4">AllocTensor, FreeTensor, InitBuffer, Get: memory management interfaces</td>
    </tr>
    <tr>
        <td colspan="4">TBuf: memory management data structure</td>
    </tr>
    <tr>
        <td colspan="4">GetHcclContext: obtains the communication Context</td>
    </tr>
</table>


## Data Flow Analysis

The AllGather operation reorders the inputs of all cards in the communicator by card ID, concatenates them, and finally sends the result to all cards. The Add operation adds the AllGather result received by each card to the other input data of the local card.

The number of cards rank_size in the communicator of this sample is fixed to 2. If the communication is not split into turns, the communication and computation proceed serially. The operator semantics diagram is as follows:

![all_gather_add_compute_semantics_diagram.png](figures/all_gather_add_compute_semantics_diagram.png)

The data of the AllGatherAdd operator performs AllGather communication between cards and Add computation within a card, and the communication and computation parts can proceed in parallel without affecting each other.
Therefore, the communication data can be split into multiple blocks, each communication transmits only one block of data, and each computation only operates on the communication result of the previous turn, with the pipelining covering each other. The communication-computation overlap diagram is as follows:

![all_gather_add_comm_compute_overlap_diagram.png](figures/all_gather_add_comm_compute_overlap_diagram.png)

**Schematic diagram of the AllGatherAdd operator computation process**: (when the communication is split into 2 turns)

1. The AI Core writes the communication information to be executed into the message area in Global Memory, implementing task dispatch. The message area is Global Memory at a specific address. The AI Core and AI CPU pass messages between them by writing to it and polling it, and these operations are uniformly encapsulated in the [Hccl](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/API/ascendcopapi/atlasascendc_api_07_0869.html) high-level API.
2. The AI CPU reads the information of all communication tasks from the message area and starts to execute the first round of AllGather collective communication tasks over the HCCS (Huawei Cache Coherence System, a high-speed interconnection between CPUs and NPUs) link. The following figure shows the schematic diagram of the first round of AllGather communication.

![all_gather_add_first_round_comm_diagram.png](figures/all_gather_add_first_round_comm_diagram.png)

3. After the AI CPU completes the first round of communication tasks, it writes the status that the first round of communication tasks has completed into the message area and starts to execute the second round of communication tasks. Meanwhile, the AI Vector starts to perform the Add computation on the first-round AllGather communication result. The following figure shows the schematic diagram of the second round of communication and the first round of Add computation on rank0.

![all_gather_add_rank0_first_round_add_diagram.png](figures/all_gather_add_rank0_first_round_add_diagram.png)

4. Complete the communication and computation of all data blocks turn by turn according to the communication splitting turns.

## Creating the Operator Project

Create the operator project according to the following directory structure:

```
├── all_gather_add             # AI Core operator name
│   ├── CMakeLists.txt         # Operator compilation configuration file, keep the original file
│   ├── examples               # Operator execution samples
│   ├── op_host                # Operator host directory, containing the operator prototype definition, Tiling, aclnn interfaces, and other related implementations
│   ├── op_kernel              # Operator kernel directory
│   └── README.md              # Operator description document
```

## Operator Prototype Definition

Compared with ordinary operators, communication-computation fusion operators have the following constraints when implementing the [operator prototype definition](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/opdevg/Ascendcopdevg/atlas_ascendc_10_0062.html):

- You must define at least one attribute that indicates the communicator name of the operator, and the number of such attributes must match the number of communicators in which the operator runs. A communicator is the context in which collective communication executes, and it manages the corresponding communication entities (for example, one NPU is one communication entity) and the resources required for communication.
- You must register the operator as a communication-computation fusion operator through the [MC2](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/API/ascendcopapi/atlasascendc_api_07_0954.html) interface in the prototype registration, and configure the communicator name of the operator through the [HcclGroup](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/API/ascendcopapi/atlasascendc_api_07_1002.html) interface.

The prototype definition of the AllGatherAdd operator is as follows:

```cpp
namespace ops {
class AllGatherAdd : public OpDef {
public:
explicit AllGatherAdd(const char *name) : OpDef(name) {
    this->Input("a")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16})
        .Format({ge::FORMAT_ND});
    this->Input("b")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16})
        .Format({ge::FORMAT_ND});

    this->Output("c")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16})
        .Format({ge::FORMAT_ND});
    this->Output("gather_out")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16})
        .Format({ge::FORMAT_ND});

    this->Attr("group").AttrType(REQUIRED).String(); // Attribute of the communication-computation fusion operator, indicating the communicator name
    this->Attr("rank_size").AttrType(REQUIRED).Int(0);

    this->AICore().AddConfig("ascend910b");
    this->AICore().AddConfig("ascend910_93");
    this->MC2().HcclGroup("group"); // The group attribute is configured as the communicator name of this operator
}
};

OP_ADD(AllGatherAdd);
}  // namespace ops

```

## Tiling Implementation

The design of the Tiling strategy for a communication-computation fusion operator mainly includes the communication splitting strategy and the Add multi-core splitting strategy.

- Communication splitting strategy: the size of the communication data block in each turn has a great impact on the performance of the communication-computation fusion operator. To clearly demonstrate the communication-computation overlap flow to first-time users of communication-computation fusion operators, this sample splits the communication into two turns, with the second round of communication overlapping with the Add computation of the previous round's communication result. For how to split communication to achieve the best performance in specific scenarios, refer to the best practice in [Ascend C Best Practices](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/opdevg/Ascendcopdevg/atlas_ascendc_map_10_0002.html): [MC² operator performance tuning case](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/opdevg/Ascendcopdevg/atlas_ascendc_best_practices_10_0043.html).
- Add multi-core splitting: based on the current number of cores (rank_size = 2), split the Y axis of the input shape across cores to obtain the shape size within a single core.

In this sample, the amount of computed data is the data volume of x2, that is, 480 * 256 * sizeof (float16) = 480 * 256 * (2 bytes/1024) KB = 240KB. When the number of AIV cores is 48, the computation volume of a single core is 5KB; when the number of AIV cores is 40, the computation volume of a single core is 6KB. Both cases are smaller than the maximum allocatable Unified Buffer (the [unified buffer](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/opdevg/Ascendcopdevg/atlas_ascendc_10_0008.html) of the AIV, holding the inputs and outputs of vector and scalar computations) size of 192KB. During the Add computation, the space needs to be evenly allocated to the three operands, so each operand has a maximum usable on-chip UB of 64KB (a maximum of 32KB usable when [Double Buffer](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/opdevg/Ascendcopdevg/atlas_ascendc_10_0090.html) is enabled). Therefore, intra-core splitting is not required.

The key steps of the Tiling implementation of this sample are as follows:

- **Define the Tiling struct of the AllGatherAdd operator**:

    The Tiling struct of the communication-computation fusion operator obtained by fusing communication and the Add computation consists of the following two parts:
    - [Tiling struct of the Hccl high-level API](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/API/ascendcopapi/atlasascendc_api_07_10048.html), which defines the Mc2InitTiling and Mc2CcTiling parameters. The Mc2InitTiling parameter is used to initialize the communication task configuration and must be defined as the first parameter of the operator Tiling struct. Mc2CcTiling is the parameter configuration of each specific communication task. Since the AllGatherAdd operator has only one communication task, AllGather, only one Mc2CcTiling parameter needs to be defined.
    - Custom information additionally required by the AllGatherAdd operator.
    The complete Tiling struct definition of the AllGatherAdd operator is as follows:

    ```cpp
    struct AllGatherAddTilingData {
        Mc2InitTiling mc2InitTiling;
        Mc2CcTiling mc2CcTiling;
        uint32_t commTurn; // Communication turn

        uint32_t totalElemNum;
        uint32_t blockElemNum;
        uint32_t tileNum;
        uint32_t addTileElemNum;
        uint32_t gatherTileElemNum;
        uint32_t addCoresPerRank;
    };
    ```

    - commTurn: communication splitting turn. In actual service scenarios, it is an important factor affecting the degree of communication-computation overlap. Refer to [communication-computation fusion basics](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/opdevg/Ascendcopdevg/atlas_ascendc_10_10033.html).
    - tileNum: computation splitting turn. As shown in the Tiling implementation analysis in this section, the UB of a single computation turn has no overflow risk, so this value is 1.
    - totalElemNum: total number of computations, whose value is the ShapeSize of input b, that is, 480.
    - blockElemNum: amount of data that each core needs to process, whose value is totalElemNum / GetBlockDim().
    - addTileElemNum: number of data elements in a single add computation, whose value is blockElemNum/ tileNum.
    - gatherTileElemNum: number of data elements in a single communication, whose value is totalElemNum / rankSize / commTurn.
    - addCoresPerRank: number of cores allocated to each rank in a single add computation, whose value is GetBlockDim() / rankSize.

- **Obtain the Tiling struct object pointer of the AllGatherAdd operator**:

    ```cpp
    AllGatherAddTilingData* tilingData = context->GetTilingData<AllGatherAddTilingData>();
    ```

    context is the object pointer of [TilingContext](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/API/basicdataapi/atlasopapi_07_00555.html). This pointer is passed in by the Tiling entry function AllGatherAddTilingFunc in the [all_gather_add_tiling.cpp](../op_host/op_tiling/all_gather_add_tiling.cpp) file, and it is used to store the context of the operator Tiling computation. In the Tiling implementation of the AllGatherAdd operator, the input/output shapes, input attributes, and other parameters required for the Tiling computation are obtained through this context, and then the Tiling results (for example, TilingKey and TilingData) are saved to the context for subsequent operator execution.

- **Set the custom Tiling struct parameters of the operator**:

    ```cpp
    tilingData->commTurn = COMM_TURN; // When the communication turn is 1, communication and computation are serial; when it is greater than 1, communication-computation overlap is enabled
    tilingData->tileNum = TILE_NUM;
    tilingData->totalElemNum = context->GetInputTensor(1)->GetShapeSize();
    tilingData->blockElemNum = tilingData->totalElemNum / tilingData->commTurn / context->GetBlockDim(); // Each Add computation only processes the data of the length of the previous communication result
    tilingData->addTileElemNum = tilingData->blockElemNum / tilingData->tileNum;
    uint32_t rankSize = *context->GetAttrs()->GetAttrPointer<uint32_t>(static_cast<int>(1));
    tilingData->addCoresPerRank = context->GetBlockDim() / rankSize; // Before the Add computation, the computation address offset of the current core needs to be determined based on the number of cores allocated to each rank
    tilingData->gatherTileElemNum = tilingData->totalElemNum / rankSize / tilingData->commTurn; // Data length of each communication turn
    ```

- **Set the Tiling struct of the Hccl high-level API**:

    According to the communication task type and algorithm configuration, create an Mc2CcTilingConfig class object, and pass the references of the mc2InitTiling and mc2CcTiling members in the operator Tiling struct to the [GetTiling](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/API/ascendcopapi/atlasascendc_api_07_10037.html) method to obtain the Mc2InitTiling and Mc2CcTiling parameters that need to be passed to the kernel side. For details about how to use the Tiling struct of the Hccl high-level API, see the [Hccl Tiling](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/API/ascendcopapi/atlasascendc_api_07_0887.html) usage instructions.

    ```cpp
    // Set the Tiling struct of the Hccl high-level API
    static void InitHcclParam(AllGatherAddTilingData* tilingData, const char* group)
    {
        std::string algConfig = "AllGather=level0:fullmesh";
        Mc2CcTilingConfig mc2CcTilingConfig(group, HCCL_CMD_ALLGATHER, algConfig);
        mc2CcTilingConfig.GetTiling(tilingData->mc2InitTiling);
        mc2CcTilingConfig.GetTiling(tilingData->mc2CcTiling);
    }
    ```

## Kernel Implementation

In the kernel implementation of the AllGatherAdd operator, AllGather communication must be performed on the data of all cards according to the communication splitting turns, and the Add operation must be performed on the communication data of each turn. In the multi-turn communication scenario, when the data of multiple cards is concatenated onto one card, the addresses of adjacent data blocks are not contiguous, while each round of Add computation only operates on the communication result of the previous turn. Therefore, the basic information required by the Add computation needs to be considered:

- Addresses of input operands a and b and output c.
- Splitting strategy of the gather_out and b operands across cores.

**The main flow of the AllGatherAdd operator kernel function is described first**:

The kernel function of the AllGatherAdd operator is defined as follows. The meanings of the aGM, bGM, cGM, and gatherOutGM parameters are described in the operator analysis, and workspaceGM and tilingGM indicate the addresses of the workspace space and the tiling data in Global Memory, respectively.

```cpp
extern "C" __global__ __aicore__ void all_gather_add(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM,
                                                     GM_ADDR gatherGM, GM_ADDR workspaceGM, GM_ADDR tilingGM);
```

- The Add computation depends on AIV cores, so the operator logic runs only on AIV cores.
    Set the kernel type through the KERNEL_TYPE_MIX_AIV_1_0 macro [setting the kernel type](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/API/ascendcopapi/atlasascendc_api_07_0218.html). In the mixed AIC/AIV scenario, it controls the operator execution to launch only the Vector cores on the AI Core.

    ```cpp
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    ```

- Register the operator Tiling struct, obtain the Tiling, and initialize TPipe.

    ```cpp
    // Register the operator Tiling struct
    REGISTER_TILING_DEFAULT(AllGatherAddTilingData);
    GET_TILING_DATA(tilingData, tilingGM);
    TPipe pipe;
    ```

- Initialize the AllGatherAdd class and perform the AllGatherAdd computation flow.

    ```cpp
    AllGatherAdd allGatherAdd;
    allGatherAdd.Init(aGM, bGM, cGM, gatherGM, &tilingData, &pipe);
    allGatherAdd.Process();
    ```

**The specific implementation steps of the computation on the kernel side of the AllGatherAdd operator are described below**:

- The **Init function** defines and assigns the variables required for subsequent computation, initializes the hccl object to prepare for AllGather communication, and initializes the TQue memory to prepare for the Add computation:

    ```cpp
    __aicore__ inline void AllGatherAdd::Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR gatherGM,
                                              AllGatherAddTilingData *tilingData, TPipe *tPipe)
    {
        aGM_ = aGM;
        bGM_ = bGM;
        cGM_ = cGM;
        gatherGM_ = gatherGM;

        tilingData_ = tilingData;
        blockElemNum_ = tilingData->blockElemNum;
        addTileElemNum_ = tilingData->addTileElemNum / ADD_BUFFER_NUM;
        tileNum_ = tilingData->tileNum;
        blockIdx_ = AscendC::GetBlockIdx();

        // Initialize the hccl object
        GM_ADDR contextGM = GetHcclContext<HCCL_GROUP_ID_0>();
        hccl_.InitV2(contextGM, tilingData);
        hccl_.SetCcTilingV2(offsetof(AllGatherAddTilingData, mc2CcTiling));

        tPipe->InitBuffer(inputQueueGather, ADD_BUFFER_NUM, addTileElemNum_ * sizeof(half));
        tPipe->InitBuffer(inputQueueB, ADD_BUFFER_NUM, addTileElemNum_ * sizeof(half));
        tPipe->InitBuffer(outputQueueC, ADD_BUFFER_NUM, addTileElemNum_ * sizeof(half));
    }
    ```

- The **Process function** waits for the completion of each communication block turn by turn in the order of communication first and computation second, according to the communication splitting turns, and performs the Add computation on the communication result gather_out and input b:

    ```cpp
    __aicore__ inline void AllGatherAdd::Process()
    {
        HcclPrepare();
        int addLoop = tileNum_ * ADD_BUFFER_NUM;
        for (int i = 0; i < tilingData_->commTurn; i++) {
            hccl_.Wait(handleId_);
            // Compute the start address of the data that this core needs to process, based on the communication turn and rankSize
            CalcAddGmAddr(i);
            // Perform the Add computation on the previous round's communication result
            for (int j = 0; j < addLoop; j++) {
                CopyIn(j);
                Compute();
                CopyOut(j);
            }
        }
        HcclFinalize();
    }
    ```

    The Process function operates the communication and computation process of the operator.
    First, the HcclPrepare() function calls the AllGather interface to dispatch the communication task. Note that when the communication is split into multiple turns, the strideCount and repeat parameters need to be used together:

    ```cpp
    __aicore__ inline void AllGatherAdd::HcclPrepare()
    {
        elemNumPerRank_ = tilingData_->gatherTileElemNum * tilingData_->commTurn; // With multi-turn communication splitting, when the data of multiple cards is concatenated onto gatherOutGM, this is the start address offset of adjacent data blocks
        // Dispatch the communication task
        handleId_ = hccl_.AllGather<true>(aGM_, gatherGM_, tilingData_->gatherTileElemNum,
                                        HcclDataType::HCCL_DATA_TYPE_FP16, elemNumPerRank_, tilingData_->commTurn);
    }
    ```

    After the first round of communication tasks completes, the operator starts the first round of Add computation. Before starting the computation, the addresses of the data participating in the computation need to be confirmed. The CalcAddGmAddr function computes the start address of the data that this core needs to process based on the communication turn and the number of cards. To achieve communication-computation overlap and multi-core parallelism and improve computation efficiency, the operands of the Add computation are split, and the split data is allocated to different cores for processing. This sample only splits the Y axis of the operands, as shown in the following diagram. In this scenario, each core needs to compute the offsets of the two addends relative to the original data, and the offset data block is passed into the Unified Buffer of the AIV core as an input parameter for the Add computation.

    ![all_gather_add_core_partition_diagram.png](figures/all_gather_add_core_partition_diagram.png)

    As shown in the preceding figure, assume that the first Add computation is currently performed on core AIV-26. According to the logic of the Process() function, when i = 0, the CalcAddGmAddr function is triggered for the first time. According to the description of the [hccl.wait()](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/API/ascendcopapi/atlasascendc_api_07_0878.html) interface, the first round of communication has completed and the second round has started at this point. Therefore, when the CalcAddGmAddr function computes the start address offset of the data that this core needs to process, it first offsets by i * the data length of a single communication, as shown by commOffset in the figure. Each Add operation computes on the result of the previous round of communication. Since the communication is split into multiple turns, the start addresses of the adjacent data blocks of each card in gatherOutGM have an offset (that is, strideCount). Therefore, the rank whose communication data the current core is assigned to process needs to be determined based on the number of cards and the index of the current core. As shown in the figure, 40 AIV cores in total are evenly allocated to two ranks, so AIV-26 is assigned to process the data from rank1, as shown by blockOffset in the figure. After the rank processed by the current core is determined, the final offset additionally needs to include the offset within the data of that rank, that is, 6 * the number of data elements processed by each core.
    The implementation of the CalcAddGmAddr function is as follows:

    ```cpp
    __aicore__ inline void AllGatherAdd::CalcAddGmAddr(int32_t commTurn)
    {
        uint32_t commOffset = commTurn * tilingData_->gatherTileElemNum; // 1. Offset by the size of a single communication data block based on the communication turn
        uint32_t blockOffset = blockIdx_ / tilingData_->addCoresPerRank * elemNumPerRank_; // 2. Determine the rank whose communication data the current core is assigned to process, based on the number of ranks and aivId
        uint32_t totalOffset = commOffset + blockOffset + (blockIdx_ % tilingData_->addCoresPerRank) * blockElemNum_; // 3. The final offset additionally needs to include the offset of the current core within the data of the processed rank
        gatherOutGM.SetGlobalBuffer((__gm__ half*)gatherGM_ + totalOffset, blockElemNum_);
        inputBGM.SetGlobalBuffer((__gm__ half*)bGM_ + totalOffset, blockElemNum_);
        outputCGM.SetGlobalBuffer((__gm__ half*)cGM_ + totalOffset, blockElemNum_);
    }
    ```

    After each core completes the computation of the data addresses, the entire computation of the AllGatherAdd operator can be completed following the [programming paradigm of typical operators](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/opdevg/Ascendcopdevg/atlas_ascendc_10_00033.html): CopyIn (moves data from GM to on-chip Local Memo) -> Compute (calls the Add arithmetic API to complete the computation) -> CopyOut (moves the computation result from Local Memo out to GM). The memory diagram of the Add computation within each core:

    ![all_gather_add_memory_transfer_diagram.png](figures/all_gather_add_memory_transfer_diagram.png)

    - Poll and wait for the completion of the communication and computation of each block, and finally release the resources.

    Combining the preceding code, visit the open-source repository for the complete kernel code.

## Compilation and Running

For operator compilation and deployment, refer to the operator README in the open-source repository:
https://atomgit.com/cann/ops-transformer/blob/master/examples/mc2/all_gather_add/README.md

## Operator Execution Sample

This example operator only supports the aclnn single-operator invocation. For the internal invocation flow, refer to [single-operator API invocation](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/opdevg/Ascendcopdevg/atlas_ascendc_10_0070.html).
After the operator is compiled by the project compilation script, the single-operator API is generated automatically and can be invoked directly in applications.

This section describes how to wrap the two-phase aclnn interfaces generated by the API in the [op_host](../op_host/op_api/) directory, and how to complete the operator invocation and execution in the [test_aclnn_all_gather_add.cpp](../examples/test_aclnn_all_gather_add.cpp) file in the examples directory.

First, wrap the aclnnInnerAllGatherAddGetWorkspaceSize and aclnnInnerAllGatherAdd functions automatically generated by the framework:

- Before calling the aclnnInnerAllGatherAddGetWorkspaceSize function to compute the amount of NPU memory required for this API invocation, validate the input parameters first (including non-emptiness, data type, shape, and so on) and record logs:

  ```cpp
  aclnnStatus aclnnAllGatherAddGetWorkspaceSize(const aclTensor *a, const aclTensor *b, char *group,
                                              int64_t rankSize, const aclTensor *cOut,
                                              const aclTensor *gatherOutOut, uint64_t *workspaceSize,
                                              aclOpExecutor **executor)
  {
    auto retParam = CheckParams(a, b, gatherOutOut, cOut, rankSize);
    CHECK_RET(retParam == ACLNN_SUCCESS, retParam);

    OP_LOGD("A is %s, B is %s.", a->ToString().GetString(), b->ToString().GetString());
    OP_LOGD("Output is %s, gatherOut is %s.", cOut->ToString().GetString(), gatherOutOut->ToString().GetString());

    aclnnStatus ret = aclnnInnerAllGatherAddGetWorkspaceSize(a, b, group, rankSize,
                                                            cOut, gatherOutOut, workspaceSize, executor);
    OP_LOGD("AllGatherAdd, aclnnInnerGetWorkspaceSize ret = %d.", ret);

    return ret;
  }

  ```

- When calling the aclnnInnerAllGatherAdd function to execute the operator computation, the MC2 operator needs to set [HcclServerType](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/API/ascendcopapi/atlasascendc_api_07_00149.html), and record logs for the interface return value:

    ```cpp
    aclnnStatus aclnnAllGatherAdd(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                              aclrtStream stream)
    {
        if (NnopbaseSetHcclServerType) {
            NnopbaseSetHcclServerType(executor, NNOPBASE_HCCL_SERVER_TYPE_AICPU);
        }
        auto ret = aclnnInnerAllGatherAdd(workspace, workspaceSize, executor, stream);
        if (ret != 0) {
            OP_LOGE(ACLNN_ERR_INNER, "This is an error in launch aicore,ret = %d.", ret);
            return ret;
        }
        return ret;
    }

    ```

The following uses pseudocode to describe the code logic of the AllGatherAdd single-operator invocation:

- Initialize the environment and test data:

    ```cpp
    // 1. Generate the test data and CPU golden data based on the fixed shapes
    int ret = GenerateTestData(testData);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] GenerateTestData failed. ret = %d \n", ret);  return ret);

    // 2. AscendCL initialization
    ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclInit failed. ret = %d \n", ret); return ret);
    aclrtStream stream[RANK_DIM];
    for (uint32_t rankId = 0; rankId < RANK_DIM; rankId++) {
        ret = aclrtSetDevice(rankId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetDevice failed. ret = %d \n", ret); return ret);
        ret = aclrtCreateStream(&stream[rankId]);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed. ret = %d \n", ret); return ret);
    }
    int32_t devices[RANK_DIM];
    for (int i = 0; i < RANK_DIM; i++) {
        devices[i] = i;
    }
    // 3. Initialize the collective communicator
    HcclComm comms[RANK_DIM];
    ret = HcclCommInitAll(RANK_DIM, devices, comms);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclCommInitAll failed. ret = %d \n", ret); return ret);

    ```

- Start multiple threads to call the two-phase aclnn interfaces on the two ranks to execute the operator:

    ```cpp
    // 4. Start multiple threads
    std::vector<std::unique_ptrstd::thread> threads(RANK_DIM);
    for (uint32_t rankId = 0; rankId < RANK_DIM; rankId++) {
    args[rankId].rankId = rankId;
    args[rankId].hcclComm = comms[rankId];
    args[rankId].stream = stream[rankId];
    threads[rankId].reset(new(std::nothrow) std::thread(&LaunchOneThreadAllGatherAdd, std::ref(args[rankId]), std::ref(testData)));
    }
    // 5. Copy the host-side data to the device side
    ret = CreateAclTensor(aHostData, aShape, &aDeviceAddr, aclDataType::ACL_FLOAT16, &a);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(bHostData, bShape, &bDeviceAddr, aclDataType::ACL_FLOAT16, &b);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // ......

    // 6. Call the two-phase interfaces
    // Call the first-phase interface
    ret = aclnnAllGatherAddGetWorkspaceSize(
    a, b, hcomName, RANK_DIM, out, gatherOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
    LOG_PRINT("[ERROR] aclnnAllGatherAddGetWorkspaceSize failed. ret = %d \n", ret); return ret);
    // Request device memory based on the workspaceSize computed by the first-phase interface
    if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d \n", ret); return ret);
    }
    // Call the second-phase interface
    ret = aclnnAllGatherAdd(workspaceAddr, workspaceSize, executor, args.stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnAllGatherAdd failed. ret = %d \n", ret); return ret);
    // Synchronize and wait for the task execution to finish
    ret = aclrtSynchronizeStreamWithTimeout(args.stream, 10000);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret = %d \n", ret);
    return ret);
    LOG_PRINT("[INFO] device_%d aclnnAllGatherAdd execute successfully.\n", args.rankId);

    ```

- Perform an accuracy comparison between the operator computation result and the CPU golden result:

    ```cpp
    // 7. Compare the operator computation result with the golden data
    std::vectorop::fp16_t gatherOutData(bShapeSize, 0);
    // Copy the computation result from the device side to the host side
    ret = aclrtMemcpy(gatherOutData.data(), bShapeSize * sizeof(gatherOutData[0]), gatherOutDeviceAddr,
    bShapeSize * sizeof(gatherOutData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    // Compare the AllGather result
    ret = CompareVector(gatherOutData, testData.gatherOut);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] gatherOut compare failed. ret = %d \n", ret); return ret);

    std::vectorop::fp16_t outputData(bShapeSize, 0);
    ret = aclrtMemcpy(outputData.data(), bShapeSize * sizeof(outputData[0]), outDeviceAddr,
    bShapeSize * sizeof(outputData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    // Compare the AllGatherAdd result, with the error within one thousandth
    if (args.rankId == 0) {
    ret = CompareVector(outputData, testData.rank0_c);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] output compare failed. ret = %d \n", ret); return ret);
    } else {
    ret = CompareVector(outputData, testData.rank1_c);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] output compare failed. ret = %d \n", ret); return ret);
    }

    ```

- Release resources:

    ```cpp
    // 8. Release resources
    // Release hccl resources
    auto hcclRet = HcclCommDestroy(args.hcclComm);
    // Release device resources
    if (a != nullptr) {
    aclDestroyTensor(a);
    }
    if (b != nullptr) {
    aclDestroyTensor(b);
    }
    // ......
    if (aDeviceAddr != nullptr) {
    aclrtFree(aDeviceAddr);
    }
    if (bDeviceAddr != nullptr) {
    aclrtFree(bDeviceAddr);
    }
    // ......
    ret = aclrtDestroyStream(args.stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtDestroyStream failed. ret = %d \n", ret); return ret);
    ret = aclrtResetDevice(args.rankId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtResetDevice failed. ret = %d \n", ret); return ret);
    ```
