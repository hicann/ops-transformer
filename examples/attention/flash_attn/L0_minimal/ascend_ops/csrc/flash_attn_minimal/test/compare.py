import datetime
import os
import sys

import numpy as np


def cal_relative_diff_mix(real_data, expect_data, diff_thd):
    if 'nan' in str(expect_data) or 'inf' in str(expect_data) or 'inf' in str(real_data):
        result = 0 if str(real_data) == str(expect_data) else 1
    else:
        diff = abs(float(real_data) - float(expect_data))
        if abs(float(real_data) - float(expect_data)) < diff_thd:
            result = diff
        else:
            result = diff / \
                     (float(max(abs(real_data), abs(expect_data))) + 10e-10)
    return result


def cal_relative_diff(real_data, expect_data, diff_thd, type_str='fp16'):
    if 'nan' in str(expect_data) or 'inf' in str(expect_data):
        if type_str.lower() == 'fp16':
            expect_data = 65504
        else:
            expect_data = 3.4028e38
    diff = abs(float(real_data) - float(expect_data))
    if abs(float(real_data) - float(expect_data)) < diff_thd:
        result = diff
    else:
        result = diff / (float(max(abs(real_data), abs(expect_data))) + 10e-10)
    return result

def display_output(real_data, expect_data, start, end, diff_thd, expect_fp32_data=None, if_mix=False):
    def display_inner(idx):
        j = idx + start
        if if_mix:
            diff_rate = cal_relative_diff_mix(
                expect_data[j], real_data[j], diff_thd)
        else:

            diff_rate = cal_relative_diff(
                expect_data[j], real_data[j], diff_thd)

        if "inf" in str(expect_data[j]) or "nan" in str(expect_data[j]):
            diff_abs = "inf" if "inf" in str(expect_data[j]) else "nan"
            if expect_fp32_data is not None:
                print_log('%08d \t %-7s \t %-7s \t %-7s \t %-7s \t %-7s' % (
                    start + idx + 1, expect_fp32_data[j], expect_data[j], real_data[j], diff_abs, diff_rate))
            else:
                print_log('%08d \t %-7s \t %-7s \t %-7s \t %-7s' % (
                    start + idx + 1, expect_data[j], real_data[j], diff_abs, diff_rate))
        else:
            diff_abs = abs(np.float64(
                expect_data[j]) - np.float64(real_data[j]))
            if expect_fp32_data is not None:
                print_log('%08d \t %0.7f \t %0.7f \t %0.7f \t %0.7f \t %0.7f' % (
                    start + idx + 1, expect_fp32_data[j], expect_data[j], real_data[j], diff_abs, diff_rate))
            else:
                print_log('%08d \t %0.7f \t %0.7f \t %0.7f \t %0.7f' % (
                    start + idx + 1, expect_data[j], real_data[j], diff_abs, diff_rate))

    print_log(
        '---------------------------------------------------------------------------------------')
    if expect_fp32_data is not None:
        print_log(
            'Loop \t ExpFP32Out \t ExpFP16Out \t NPUOut \tFpDiff(min) \t RateDiff')
    else:
        print_log('Loop \t ExpectOut \t RealOut \t FpDiff \t RateDiff')
    print_log(
        '---------------------------------------------------------------------------------------')
    split_count = int(end - start)
    if split_count <= 20:
        for i in range(split_count + 1):
            display_inner(i)
    else:
        for i in range(10):
            display_inner(i)
        print_log('...   \t   ...   \t   ...   \t   ...    \t   ...')
        for i in range(split_count - 10 + 1, split_count + 1):
            display_inner(i)




def print_log(data=None, level='INFO'):
    print("[%s] [%s]-%s:%s - %s" % (datetime.datetime.now().strftime(
        "%Y/%m/%d %H:%M:%S"), level, os.path.basename(sys._getframe().f_back.f_code.co_filename),
                                    str(sys._getframe().f_back.f_lineno).zfill(4), data))


def cal_relative_diff_np(real_data, expect_data, diff_thd):
    a = np.abs(np.subtract(real_data, expect_data))
    b1 = np.maximum(np.abs(real_data), (np.abs(expect_data)))
    #  修改人：d30034183
    if diff_thd == 0.0001:
        temp1 = 126
        temp2 = 0.000001
    # 非float维持之前的对比方式
    else:
        temp1 = 14
        temp2 = diff_thd
    b2 = float((1.0 / (1 << temp1)) / temp2)
    b = np.add(np.maximum(b1, b2), 10e-10)
    result = np.where(a < temp2, a, a / b)
    return result


def cal_relative_diff_np_high(real_data, expect_data, diff_thd):
    a = np.abs(np.subtract(real_data, expect_data))
    b1 = np.maximum(np.abs(real_data), (np.abs(expect_data)))
    #  修改人：d30034183
    if diff_thd == 0.0001:
        temp1 = 126
        temp2 = 0.000001
        temp3 = 0.000000001
    else:
        temp1 = 14
        temp2 = diff_thd
        temp3 = 0.00001
    b2 = float((1.0 / (1 << temp1)) / temp2)
    b = np.add(np.maximum(b1, b2), 10e-10)
    result = np.where(a < temp3, a, a / b)
    return result


def display_error_output(real_data, expect_data, err_idx, relative_diff):
    print_log(
        'Error Line-----------------------------------------------------------------------------')
    print_log('Loop \t ExpectOut \t RealOut \t FpDiff \t RateDiff')
    print_log(
        '---------------------------------------------------------------------------------------')
    count = 0
    len_err = len(err_idx)
    for i in err_idx:
        count += 1
        if count < 10 or (90 < count < 100):
            print_log('%08d \t %.7f \t %.7f \t %.7f \t %.7f' % (
                i, expect_data[i], real_data[i], abs(np.float64(
                    expect_data[i]) - np.float64(real_data[i])),
                relative_diff[count - 1]))
        elif count == 10 or (count == 100 and len_err > 100):
            dot_3 = '...'
            print_log('%08s \t %07s \t %07s \t %07s \t %07s' %
                      (dot_3, dot_3, dot_3, dot_3, dot_3))
        elif count > 100:
            break

    print_log(
        'Max-RE line:---------------------------------------------------------------------------')
    max_error = max(relative_diff)
    m_idx_list = err_idx[np.where(relative_diff == max_error)]
    m_count = 0
    for m_idx in m_idx_list:
        m_count += 1
        if m_count < 4:
            print_log('%08d \t %.7f \t %.7f \t %.7f \t %.7f' % (
                m_idx, expect_data[m_idx], real_data[m_idx],
                abs(np.float64(expect_data[m_idx]) -
                    np.float64(real_data[m_idx])),
                max_error))
        else:
            break
    print_log(
        '---------------------------------------------------------------------------------------')

def compare(expect_data, real_data):
    diff_thd = 0.0001
    pct_thd=0.05
    start = 0
    end = expect_data.size - 1
    max_diff_hd = 10
    max_error_idx = 10000000
    split_count = int(end - start + 1) if end != start else 1
    print_log("[data_compare_np]diff_abs start!")
    diff_abs = np.abs(np.subtract(real_data.astype(
        np.float32), expect_data.astype(np.float32)))
    diff_index = np.where(diff_abs > 0)
    print_log("[data_compare_np]diff_abs end!")
    rdiff = cal_relative_diff_np(real_data[diff_index].astype(np.float32),
                                 expect_data[diff_index].astype(np.float32),
                                 diff_thd)
    err_diff = rdiff[rdiff > diff_thd]
    diff_idx_list = diff_index[0]
    err_idx = diff_idx_list[np.where(rdiff > diff_thd)]

    fulfill_percent = float(split_count - err_diff.size) / \
                      float(split_count) * 100.0

    display_output(real_data, expect_data, start, end, diff_thd)
    # 数据量不上不下的时候加入1的余量
    if len(expect_data) > 100:
        pct_thd = float((len(expect_data) * pct_thd + 1) / len(expect_data))
    pct_thd = (1 - pct_thd) * 100.0
    result = "Pass" if (fulfill_percent >= pct_thd) else "Failed"
    if len(err_diff) > 0:
        max_error = max(err_diff[0:max_error_idx])
        if max_error >= max_diff_hd:
            result = "Failed"
    print_log(
        '---------------------------------------------------------------------------------------')
    print_log('DiffThd  \t PctThd   \t PctRlt   \t Result')
    print_log(
        '---------------------------------------------------------------------------------------')
    print_log('%.4f     \t %.2f%%   \t %.6f%%   \t %s' %
              (diff_thd, pct_thd, fulfill_percent, result))
    if len(err_diff) > 0:
        print_log('Max-RelativeError is: %s. Threshold is: %s.' %
                  (max_error, max_diff_hd))
    if result == "Failed":
        display_error_output(real_data, expect_data,
                             err_idx, err_diff[0:max_error_idx])
    return result == "Pass"

