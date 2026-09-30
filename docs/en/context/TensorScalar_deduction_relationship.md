# TensorScalar deduction relationship

## Deduction Rules

If the input tensor data type of an API (such as **aclnnAdds** or **aclnnMuls**) is different from that of the input scalar data type, the API internally deduces a data type and converts the input data into the deduced data type for computation.
Note: This deduction rule applies only to the <term>Ascend 910_95 AI Processor</term>.

The rules for type deduction are as follows.

> Note:
>
>- To facilitate description, the data types in the table take abbreviated forms. The meanings of the data types are ACL\_FLOAT\(f32\), ACL\_FLOAT16\(f16\), ACL\_DOUBLE\(f64\), ACL\_BF16\(bf16\), ACL\_INT8\(s8\), ACL\_UINT8\(u8\), ACL\_INT16\(s16\), ACL\_UINT16\(u16\), ACL\_INT32\(s32\), ACL\_UINT32\(u32\), ACL\_INT64\(s64\), ACL\_UINT64\(u64\), ACL\_BOOL\(bool\), ACL\_COMPLEX32\(c32\), ACL\_COMPLEX64\(c64\), and ACL\_COMPLEX128\(c128\).
>- The table header indicates the data type of the input tensor to be deduced, and the leftmost column indicates the data type of the input scalar to be deduced. The corresponding cell in the table indicates the deduced data type.
>- The cross sign (×) in the table indicates that the two types cannot be deduced.

**Table 1** Data type deduction relationship

| Data Type | f32  | f16  | f64  | bf16 |  s8  |  u8  | s16  | u16  | s32  | u32  | s64  | u64  | bool | c32  | c64  | c128 |
| :------: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| **f32**  | f32  | f16  | f64  | bf16 | f32  | f32  | f32  |  ×   | f32  |  ×   | f32  |  ×   | f32  | c32  | c64  | c128 |
| **f16**  | f32  | f16  | f64  | bf16 | f32  | f32  | f32  |  ×   | f32  |  ×   | f32  |  ×   | f32  | c32  | c64  | c128 |
| **f64**  | f32  | f16  | f64  | bf16 | f32  | f32  | f32  |  ×   | f32  |  ×   | f32  |  ×   | f32  | c128 | c128 | c128 |
| **bf16** | f32  | f16  | f64  | bf16 | f32  | f32  | f32  |  ×   | f32  |  ×   | f32  |  ×   | f32  | c32  | c64  | c128 |
|  **s8**  | f32  | f16  | f64  | bf16 |  s8  |  u8  | s16  | u16  | s32  | u32  | s64  | u64  |  s8  | c32  | c64  | c128 |
|  **u8**  | f32  | f16  | f64  | bf16 |  s8  |  u8  | s16  | u16  | s32  | u32  | s64  | u64  |  u8  | c32  | c64  | c128 |
| **s16**  | f32  | f16  | f64  | bf16 |  s8  |  u8  | s16  | u16  | s32  | u32  | s64  | u64  | s16  | c32  | c64  | c128 |
| **u16**  | f32  | f16  | f64  | bf16 |  s8  |  u8  | s16  | u16  | s32  | u32  | s64  | u64  |  ×   | c32  | c64  | c128 |
| **s32**  | f32  | f16  | f64  | bf16 |  s8  |  u8  | s16  | u16  | s32  | u32  | s64  | u64  | s32  | c32  | c64  | c128 |
| **u32**  | f32  | f16  | f64  | bf16 |  s8  |  u8  | s16  | u16  | s32  | u32  | s64  | u64  |  ×   | c32  | c64  | c128 |
| **s64**  | f32  | f16  | f64  | bf16 |  s8  |  u8  | s16  | u16  | s32  | u32  | s64  | u64  | s64  | c32  | c64  | c128 |
| **u64**  | f32  | f16  | f64  | bf16 |  s8  |  u8  | s16  | u16  | s32  | u32  | s64  | u64  |  ×   | c32  | c64  | c128 |
| **bool** | f32  | f16  | f64  | bf16 |  s8  |  u8  | s16  | u16  | s32  | u32  | s64  | u64  | bool | c32  | c64  | c128 |
| **c32**  | c64  | c32  | c128 | c64  | c64  | c64  | c64  | c64  | c64  | c64  | c64  | c64  | c64  | c32  | c64  | c128 |
| **c64**  | c64  | c32  | c128 | c64  | c64  | c64  | c64  | c64  | c64  | c64  | c64  | c64  | c64  | c32  | c64  | c128 |
| **c128** | c64  | c32  | c128 | c64  | c64  | c64  | c64  | c64  | c64  | c64  | c64  | c64  | c64  | c32  | c64  | c128 |

## Deduction Example

- If the data type of the input tensor is float16 and that of the input scalar is float32, the API will convert the float32 data type of the input scalar to float16 and then perform computation.
- If the data type of the input tensor is bool and that of the input scalar is float32, the API will convert the bool data type of the input tensor to float32 and then perform computation.
