# sparseModeOptional Parameter Description

This section describes **sparseModeOptional** parameters in different scenarios.

| sparseModeOptional | Description                           | Remarks          |
|--------------------|-------------------------------|--------------|
| 0                  | defaultMask mode.               | -            |
| 1                  | allMask mode.                   | -            |
| 2                  | leftUpCausal mode.              | -            |
| 3                  | rightDownCausal mode.           | -            |
| 4                  | band mode.                      | -            |
| 5                  | prefix non-compression mode.                 | Not supported in the varlen scenario.|
| 6                  | prefix compression mode.                  | -            |
| 7                  | rightDownCausal mode in the varlen outer slicing scenario.| Supported only in the varlen scenario.|
| 8                  | leftUpCausal mode in the varlen outer slicing scenario.   | Supported only in the varlen scenario.|

The working principle of attenMask is to mask the values of the product between query (Q) and the transpose matrix of key (K) at the positions where the **Mask** is **True**, as shown in the following figure:
![Schematic Diagram](../figures/QK_transpose_diagram.png)

The $QK^T$ matrix is masked at the position where **attenMask** is **True**. The effect is as follows:
![Schematic Diagram](../figures/masked_QK_diagram.png)

- If **sparseModeOptional** is set to **0**, the defaultMask mode is used.

  - No mask: The mask operation is unavailable if **attenMaskOptional** is set to **None**, and the values of **preTokensOptional** and **nextTokensOptional** are ignored. The masked $QK^T$ matrix is as follows.
    ![Schematic Diagram](../figures/sparsemode_0_masked_matrix.png)
    
  - If **nextTokensOptional** is set to **0** and **preTokensOptional** is greater than or equal to **Sq**, sparse computation in the causal scenario is performed. **attenMaskOptional** takes the lower triangular matrix. The part between **preTokensOptional** and **nextTokensOptional** needs to be computed. The masked $QK^T$ matrix is as follows.
    ![Schematic Diagram](../figures/sparsemode_0_masked_matrix_1.png)

    **attenMaskOptional** should take the lower triangular matrix, as shown in the following figure.
    ![Schematic Diagram](../figures/attenmask_lower_triangle.png)
    
  - If **preTokensOptional** is less than **Sq** and **nextTokensOptional** is less than **Skv**, and both are greater than or equal to 0, the band scenario is used. In this case, the part between **preTokensOptional** and **nextTokensOptional** needs to be computed. The masked $QK^T$ matrix is as follows.
     ![Schematic Diagram](../figures/sparsemode_0_masked_matrix_2.png)    
     **attenMaskOptional** should take a band matrix, as shown in the following figure.
     ![Schematic Diagram](../figures/attenmask_band_matrix.png)
     
  - **nextTokensOptional** is a negative number. For example, if **preTokensOptional** is **9** and **nextTokensOptional** is **-3**, the part between **preTokensOptional** and **nextTokensOptional** needs to be computed. The masked $QK^T$ is as follows.
     **Note: When nextTokensOptional is a negative number, the value of preTokensOptional must be greater than or equal to the absolute value of nextTokensOptional, and the absolute value of nextTokensOptional must be less than Skv.**    
     ![Schematic Diagram](../figures/sparsemode_0_masked_matrix_3.png)
     
  - **preTokensOptional** is a negative number. For example, if **nextTokensOptional** is **7** and **preTokensOptional** is **-3**, the part between **preTokensOptional** and **nextTokensOptional** needs to be computed. The masked $QK^T$ is as follows.
    **Note: When preTokensOptional is a negative number, the value of nextTokensOptional must be greater than or equal to the absolute value of preTokensOptional, and the absolute value of preTokensOptional must be less than Sq.**
    ![Schematic Diagram](../figures/sparsemode_0_masked_matrix_4.png)
  
- If **sparseModeOptional** is set to **1**, the allMask mode is used, which means that a complete **attenMaskOptional** matrix needs to be passed. In this scenario, the values of **nextTokensOptional** and **preTokensOptional** are ignored. The masked $QK^T$ matrix is as follows.
  ![Schematic Diagram](../figures/sparsemode_1_masked_matrix.png)
  
- If **sparseModeOptional** is set to **2**, the mask in leftUpCausal mode is used, corresponding to the lower triangle with the upper left vertex as the start point. In this scenario, the values of **preTokensOptional** and **nextTokensOptional** are ignored. The masked $QK^T$ matrix is as follows.
  ![Schematic Diagram](../figures/sparsemode_2_masked_matrix.png)

  The passed **attenMaskOptional** is an optimized compressed lower triangular matrix (2048 × 2048). The compressed lower triangular matrix (same below) is as follows.
  ![Schematic Diagram](../figures/attenmask_compressed_lower_triangle.png)
  
- If **sparseModeOptional** is set to **3**, the mask in rightDownCausal mode is used, corresponding to the lower triangle with the lower right vertex as the start point. In this scenario, the values of **preTokensOptional** and **nextTokensOptional** are ignored. **attenMaskOptional** is an optimized compressed lower triangular matrix (2048 × 2048). The masked $QK^T$ matrix is as follows.
  ![Schematic Diagram](../figures/sparsemode_3_masked_matrix.png)
  
- If **sparseModeOptional** is set to **4**, the band mode is used, that is, the part between **preTokensOptional** and **nextTokensOptional** is computed. The start point is the lower right vertex, and **preTokensOptional** and **nextTokensOptional** must have an intersection. **attenMaskOptional** is an optimized compressed lower triangular matrix (2048 × 2048). The masked $QK^T$ matrix is as follows.
  ![Schematic Diagram](../figures/sparsemode_4_masked_matrix.png)
  
- If **sparseModeOptional** is set to **5**, the prefix non-compression mode is used. That is, a matrix with the length of **Sq** and width of **N** is added to the left of **rightDownCausal**. **N** can be obtained from the optional input **prefix**. For example, in the following figure, the prefix is passed as [4,5] in the **batch=2** scenario, and the value of **N** for each batch axis can be different. The parameter start point is the upper left vertex.
  In this scenario, the values of **preTokensOptional** and **nextTokensOptional** are ignored. The data format of the **attenMaskOptional** matrix must be BNSS or B1SS. The masked $QK^T$ matrix is as follows.
  ![Schematic Diagram](../figures/sparsemode_5_masked_matrix.png)
  The matrix to be passed to **attenMaskOptional** is as follows.
  ![Schematic Diagram](../figures/attenmask_matrix.png)
  
- If **sparseModeOptional** is set to **6**, the prefix compression mode is used. That is, in the prefix scenario, **attenMask** is the optimized compressed lower triangular matrix plus the rectangular matrix (3072 × 2048). The upper part is the lower triangular matrix of [2048, 2048], and the lower part is the rectangular matrix of [1024, 2048]. The left half of the rectangular matrix is all 0s, and the right half is all 1s. The matrix to be passed to **attenMaskOptional** is as follows. In this scenario, the values of **preTokensOptional** and **nextTokensOptional** are ignored.
  ![Schematic Diagram](../figures/sparsemode_6_masked_matrix.png)
  
- If **sparseModeOptional** is set to **7**, the varlen scenario is used with outer slicing of long sequences (multi-device query slicing of long sequences in the model script). You need to ensure that the scenario where **sparseModeOptional** is **3** is used before the slicing. In the current mode, you need to set **preTokensOptional** and **nextTokensOptional** (the start point is the lower right vertex) and ensure that the parameters are correct. Otherwise, accuracy drop may occur.
  The following figure shows the masked $QK^T$ matrix. The query is sliced in the second batch, and the key and value are not sliced. The 4 × 6 mask matrix is sliced into 2 × 6 and 2 × 6 masks, which are computed on device 1 and device 2, respectively.

  - The last mask of device 1 is a band mask. Set **preTokensOptional** to **6** (ensure that the value is greater than or equal to the last **Skv**), **nextTokensOptional** to **-2**, **actual_seq_qlen** to **{3,5}**, and **actual_seq_kvlen** to **{3,9}**.
  - The mask type of device 2 remains unchanged after being sliced. Set **sparseModeOptional** to **3**, **actual_seq_qlen** to **{2,7,11}**, and **actual_seq_kvlen** to **{6,11,15}**.
  ![Schematic Diagram](../figures/sparsemode_7_masked_matrix.png)

  Note:
    - When **sparseModeOptional** is set to **7**, band indicates the sparse type of the batch of the last non-empty tensor. If there is only one batch, you need to set the parameters according to the band mode. When **sparseModeOptional** is set to **7**, you need to input a 2048 × 2048 lower triangular mask as the input of the fused operator.
    - The sparse parameters of the band mode generated based on **sparseModeOptional=3** must meet the following conditions:
       - preTokensOptional >= last_Skv
       - last_Sq-last_Skv <= nextTokensOptional <= 0
       - In the current mode, the optional input pse is not supported.
    - Batch in non-band mode must meet the following condition: Sq <= Skv.

- If **sparseModeOptional** is set to **8**, the varlen scenario is used with outer slicing of long sequences. You need to ensure that the scenario where **sparseModeOptional** is set to **2** is used before the outer slicing. In current mode, you need to set **preTokensOptional** and **nextTokensOptional** (the start point is the lower right vertex) and ensure that the parameters are correct. Otherwise, accuracy drop may occur.
  The following figure shows the masked $QK^T$ matrix. The query is sliced in the second batch, the key and value are not sliced, and the 5 × 4 mask matrix is sliced into 2 × 4 and 3 × 4 masks, which are computed on device 1 and device 2, respectively.

  - The mask type of device 1 remains unchanged after being sliced. Set **sparseModeOptional** to **2**, **actual_seq_qlen** to **{3,5}**, and **actual_seq_kvlen** to **{3,7}**.
  - The first mask of device 2 is a band mask. Set **preTokensOptional** to **4** (ensure that the value is greater than or equal to the first **Skv**), **nextTokensOptional** to **1**, **actual_seq_qlen** to **{3,8,12}**, and **actual_seq_kvlen** to **{4,9,13}**.

    ![Schematic Diagram](../figures/sparsemode_8_masked_matrix.png)

  Note:
    - When **sparseModeOptional** is set to **8**, band indicates the sparse type of the batch of the first non-empty tensor. If there is only one batch, you need to set the parameters according to the band mode. When **sparseModeOptional** is set to **8**, you need to input a 2048 × 2048 lower triangular mask as the input of the fused operator.
    - The sparse parameters of the band mode generated based on **sparseModeOptional=2** must meet the following conditions:
       - preTokensOptional >= first_Skv
       - nextTokensOptional >= first_Sq - first_Skv (Set this parameter based on the actual situation.)
       - In the current mode, the optional input pse is not supported.
