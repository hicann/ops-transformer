import os
import re
import struct
import torch

def hex32_to_float(hex_str):
    """
    将8位16进制字符串（32位）转换为 IEEE 754 float32
    :param hex_str: 如 '41320000'
    :return: float 数值
    """
    # 转成32位无符号整数
    int_value = int(hex_str, 16)
    # struct 解析为 float (大端模式，与内存数据一致)
    float_value = struct.unpack('!f', struct.pack('!I', int_value))[0]
    return float_value


def parse_log_data(log_text):
    """
    解析日志文本，提取所有32位16进制数据并转float
    """
    # 正则匹配 8位16进制数（在[]中间）
    pattern = r'\[([0-9a-fA-F]{8})\]'
    hex_list = re.findall(pattern, log_text)
    
    # 转换为float
    float_results = []
    for idx, h in enumerate(hex_list):
        f_val = hex32_to_float(h)
        float_results.append(f_val)
       #  print(f"地址 0x{idx*4:08x} | 16进制: {h:>8} | float: {f_val:.10f}")
    
#     print("\n✅ 解析完成，共 {} 个float数据".format(len(float_results)))
    return float_results

def list_to_2d_tensor_custom(your_list):
    """
    严格按照你的规则：
    每次搬16个 → 第1行前16 → 第2行前16 → ... → 第8行前16 → 拼成1行128
    最终输出 2D tensor: (总批数, 128)
    """
    # 转成tensor方便操作
    data = torch.tensor(your_list, dtype=torch.float32)
    
    # 计算维度：每8行拼成1行128，每行取前16
    num_rows = data.shape[0]       # 输入总行数
    num_batches = num_rows // 8    # 最终输出行数 = 输入/8
    feature_dim = 128              # 固定最后一维
    chunk_size = 16                # 每次搬16个
    
    # 存储结果
    result = []
    
    # 按8行一组处理
    for i in range(num_batches):
        # 取出当前组的8行
        group = data[i*8 : (i+1)*8]
        # 每行取前16个，拼接成 1×128
        row_128 = torch.cat([row[:chunk_size] for row in group])
        result.append(row_128)
    
    # 拼成最终二维 tensor
    final_tensor = torch.stack(result)
    
    return final_tensor

import re
import struct

def hex_bf16_to_float(hex_str):
    """
    4位 bfloat16 hex → float32（纯Python实现）
    """
    h = int(hex_str, 16)
    sign = (h >> 15) & 0x1
    exp = (h >> 7) & 0xFF
    mantissa = h & 0x7F
    f32_bits = (sign << 31) | (exp << 23) | (mantissa << 16)
    return struct.unpack('!f', struct.pack('!I', f32_bits))[0]

def parse_log_data_bf16(log_text):
    """
    解析 [3f8c3dbb] 格式：
    8位十六进制 = 前4位(高16位) + 后4位(低16位)
    每个4位 = 1个 bfloat16 → 转 float32
    低位在后4位 → 先放后4位的值，再放前4位的值
    """
    # 正则匹配 [] 中的 8 位 hex
    pattern = r'\[([0-9a-fA-F]{8})\]'
    hex_list = re.findall(pattern, log_text)
    
    float_results = []
    for full_hex in hex_list:
        # 拆分：前4位(高16位)、后4位(低16位)
        hex_high = full_hex[:4]  # 前4位：高16位
        hex_low = full_hex[4:]   # 后4位：低16位（低位）
        
        # 低位在后4位 → 先加低位值，再加高位值
        f_low = hex_bf16_to_float(hex_low)
        f_high = hex_bf16_to_float(hex_high)
        
        float_results.append(f_low)
        float_results.append(f_high)
    
    return float_results

import re

import re

def fp8_e4m3_to_float(byte_val):
    """
    FP8 E4M3 格式解码函数
    格式：1位符号位 + 4位指数位 + 3位尾数位
    指数偏移量：7 (标准E4M3偏移)
    """
    # 提取符号位
    sign = (byte_val >> 7) & 0x01
    sign = -1 if sign else 1

    # 提取4位指数
    exp = (byte_val >> 3) & 0x0F

    # 提取3位尾数
    mantissa = byte_val & 0x07

    # E4M3 标准指数偏移量 = 2^(4-1)-1 = 7
    bias = 7

    if exp == 0:
        # 非规格化数/零值
        if mantissa == 0:
            return 0.0  # 零值
        # 非规格化数: 值 = mantissa * 2^(1-bias) * 2^-3
        val = mantissa * (2 ** (1 - bias)) / 8
    elif exp == 0b1111:
        # 指数全1：无穷大或NaN
        if mantissa == 0:
            return float('inf') if sign == 1 else float('-inf')  # 无穷大
        else:
            return float('nan')  # NaN
    else:
        # 规格化数: 值 = (1 + mantissa/8) * 2^(exp - bias)
        val = (1 + mantissa / 8) * (2 ** (exp - bias))

    return sign * val

def parse_fp8e4m3_log(log_data):
    """解析日志数据，提取地址并转换为FP8 E4M3"""
    # 正则匹配：Address XXXXXXXX = [XXXXXXXX]
    pattern = r'Address\s+([0-9a-fA-F]{8})\s*=\s*\[([0-9a-fA-F]{8})\]'
    matches = re.findall(pattern, log_data)

    results = []
    for addr_str, hex32_str in matches:
        # 转换32位十六进制为字节数组（4个字节，小端/大端可切换）
        byte_data = bytes.fromhex(hex32_str)
        # 按顺序拆分4个字节，对应地址连续+0、+1、+2、+3
        base_addr = int(addr_str, 16)
        for i, byte in enumerate(reversed(byte_data)):
            fp_val = fp8_e4m3_to_float(byte)
            results.append(fp_val)
    return results

def fp8_e8m0_to_float(byte_val):
    """
    FP8 E8M0 格式解码函数 (仅限无符号、纯指数格式)
    格式：8位 指数位 (无符号、无尾数、无符号位)
    偏移量：127
    值 = 2 ^ (exp - 127)
    """
    exp = byte_val & 0xFF
    if exp == 0:
        # 0 表示 0
        return 0.0
    # E8M0 公式：value = 2^(exp - 127)
    return 2.0 ** (exp - 127.0)


def parse_fp8_log(log_data):
    """解析日志数据，提取地址并转换为FP8 E8M0"""
    # 正则匹配：Address XXXXXXXX = [XXXXXXXX]
    pattern = r'Address\s+([0-9a-fA-F]{8})\s*=\s*\[([0-9a-fA-F]{8})\]'
    matches = re.findall(pattern, log_data)

    results = []
    for addr_str, hex32_str in matches:
        # 转换32位十六进制为字节数组
        byte_data = bytes.fromhex(hex32_str)
        base_addr = int(addr_str, 16)
        # 保持和你原来逻辑一致：反转字节顺序（小端）
        for i, byte in enumerate(reversed(byte_data)):
            fp_val = fp8_e8m0_to_float(byte)
            results.append(fp_val)
    return results

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "data.txt")
    with open(data_path, "r") as f:
        LOG_DATA = f.read()

    # 解析数据
#     parsed_results = parse_fp8e4m3_log(LOG_DATA)
    parsed_results = parse_log_data(LOG_DATA)
#     tensor = list_to_2d_tensor_custom(parsed_results)
#     print(tensor)

    # 打印结果（表格形式）
    # print(f"{'地址':<10} {'原始字节':<8} {'FP8 E4M3 数值'}")
    # print("-" * 40)
    
#     for res in parsed_results:
#         # print(f"{res['address']:<10} {res['byte_hex']:<8} {res['fp8_e4m3']:.5f}")
#         print(f"{res:.4f}")


       # 初始化计数器
    count = 0
    for res in parsed_results:
    # 打印数值，保留4位小数
       print(f"{res:.4f}", end=" ")
       count += 1
       # 每32个换一行
       if count % 32 == 0:
           print()

#     # 可选：保存到CSV文件
#     import csv
#     with open("float_result.csv", "w", newline="") as f:
#         writer = csv.DictWriter(f, fieldnames=["address", "byte_hex", "float"])
#         writer.writeheader()
#         writer.writerows(parsed_results)

if __name__ == "__main__":
    main()