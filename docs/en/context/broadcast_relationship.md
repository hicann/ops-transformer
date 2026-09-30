# Broadcast Relationship

## Concept of Broadcasting

Broadcasting describes how an operator treats tensors (or arrays) with different shapes during arithmetic operations. In most cases, the smaller tensor (or array) can be "broadcast" across the larger tensor (or array) so that they have compatible shapes.

The shape parameters of many CANN operator APIs support broadcasting, which can improve the computing efficiency and reduce the memory usage (especially when the data size is large). For more details about the broadcasting technique, see [NumPy](https://numpy.org/doc/stable/user/basics.broadcasting.html).

## Broadcasting Rules

Generally, you need to understand the following broadcasting rules during computation:

- Rule 1: If two arrays have different dimensions, it pads ones on the left side of the shape of the array that has fewer dimensions.
  
  > Note:
  > - Example 1: The number of dimensions reflects the shape of a tensor (or array). For example, if x.shape = (1, 1, 2, 4), the number of dimensions is 4. 
  > - Example 2: To calculate a+b, where a.shape=\(2, 2, 3\) and b.shape=\(2, 3\), array b will be broadcast to b.shape=\(1, 2, 3\).
  
- Rule 2: If the number of dimensions of two arrays is the same and a dimension of an array is 1, the array with the dimension 1 is stretched to match the dimension shape of the other array.

  > Note:
  > In this scenario, you only need to ensure that broadcasting is performed in a dimension. For example, to calculate a+b, where a.shape=\(1, 3\) and b.shape=\(3, 1\), the two arrays will be broadcast to a.shape=\(3, 3\) and b.shape=\(3, 3\).

- Rule 3: If any dimension of two arrays is not equal and neither is equal to one, an error is reported.

Based on the preceding rules, the broadcasting process first expands dimensions according to Rule 1 and then stretches the shape according to Rule 2. The following is an example:

```text
Assume that a.shape = (2, 2, 3). The values are as follows:
[[[1 2 3],[4 5 6]],
 [[1 2 3],[4 5 6]]]
Assume that b.shape = (2, 3). The values are as follows:
[[1 2 3],
 [-1 -2 -3]]
Expand dimensions based on Rule 1 and b.shape = (1, 2, 3). The values are as follows:
[[[1 2 3],
  [-1 -2 -3]]]
Stretch the shape based on Rule 2 and b.shape = (2, 2, 3). The values are as follows:
[[[1 2 3],[-1 -2 -3]],
 [[1 2 3],[-1 -2 -3]]]
Compute a+b. The actual result is as follows:
 [[[2 4 6],[3 3 3]],
  [[2 4 6],[3 3 3]]]
```

## Constraints

If the data types of the two inputs a and b meet the broadcast relationship and their deduced data type is COMPLEX64, COMPLEX128, DOUBLE, INT16, UINT16, or UINT64, the following condition must be met in addition to the preceding broadcast rules. Otherwise, the broadcasting fails and an error is reported during operator execution.
Condition: After combination of the contiguous axes that need to be broadcast and the contiguous axes that do not need to be broadcast, the number of dimensions must be fewer than 6.
Example:

- If a.shape=\(5, 1, 5, 1, 5, 1\) and b.shape=\(5, 5, 5, 5, 5, 5\) do not have axes to be combined, the final shape is 6D. In this case, an error is reported.
- If a.shape=\(5, 1, 5, 5, 1, 1\) and b.shape=\(5, 5, 5, 5, 5, 5\) have the second and third dimensions not to be broadcast but the fourth and fifth dimensions to be broadcast contiguously, the final shape is 4D. In this case, the broadcasting is successful.
