/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include "graph/graph.h"
#include "graph/operator.h"
#include "graph/tensor.h"
#include "graph/types.h"
#include "ge/ge_api.h"
#include "ge/ge_ir_build.h"
#include "array_ops.h"
#include "../op_graph/dequant_rope_quant_kvcache_proto.h"
using namespace ge;
using std::string;
using std::vector;

struct Cfg {
    int64_t D = 128, Nq = 8, Nkv = 1, B = 2, C1 = 2, C2 = 8;
    bool page = false, bf16 = false, int32x = false, idx2d = false;
    bool useOk = true, useOv = true, useWeight = false, useAct = false, useBias = false;
    float okv = 2.0f, ovv = 2.0f;
    int32_t pageIdx = 0, contigIdx = 0;
    bool expectFail = false;
    bool determinism = false;
    vector<int64_t> splitsOverride;
    vector<int> kAllowed; // allowed cache values (plus 6 for untouched)
    vector<int> vAllowed;
    int64_t expectWritten = -1;
    const char* name = "";
    const char* quantMode = "static";
    const char* layout = "BSND";
    const char* cacheMode = nullptr;
};

static TensorDesc Desc(const vector<int64_t>& s, DataType dt)
{
    TensorDesc d(Shape(s), FORMAT_ND, dt);
    d.SetPlacement(kPlacementHost);
    d.SetFormat(FORMAT_ND);
    d.SetRealDimCnt(s.size());
    return d;
}
static size_t Num(const vector<int64_t>& s)
{
    size_t n = 1;
    for (auto v : s)
        n *= (v > 0 ? v : 1);
    return n;
}

template <typename T>
static void Put(vector<uint8_t>& b, const vector<int64_t>& s, T v)
{
    b.resize(Num(s) * sizeof(T));
    T* p = reinterpret_cast<T*>(b.data());
    for (size_t i = 0; i < Num(s); ++i)
        p[i] = v;
}
static void Put16(vector<uint8_t>& b, const vector<int64_t>& s, uint16_t v)
{
    b.resize(Num(s) * 2);
    uint16_t* p = reinterpret_cast<uint16_t*>(b.data());
    for (size_t i = 0; i < Num(s); ++i)
        p[i] = v;
}
static void PutIdx(vector<uint8_t>& b, const vector<int64_t>& s, int32_t v, DataType dt)
{
    size_t n = 1;
    for (auto x : s)
        n *= x;
    if (dt == DT_INT64) {
        b.resize(n * 8);
        int64_t* p = reinterpret_cast<int64_t*>(b.data());
        for (size_t i = 0; i < n; ++i)
            p[i] = v;
    } else {
        b.resize(n * 4);
        int32_t* p = reinterpret_cast<int32_t*>(b.data());
        for (size_t i = 0; i < n; ++i)
            p[i] = v;
    }
}

struct Out {
    Status st = FAILED;
    vector<vector<uint8_t>> bytes;
    string err;
};

static Out Run(const Cfg& c, int dev)
{
    const int64_t D = c.D, Nq = c.Nq, Nkv = c.Nkv, B = c.B, C1 = c.C1, C2 = c.C2;
    const int64_t H = (Nq + 2 * Nkv) * D;
    vector<int64_t> sx{B, H}, sc{B, D}, skc{C1, C2, Nkv, D};
    vector<int64_t> si = c.idx2d ? vector<int64_t>{B, 1} : vector<int64_t>{B};
    vector<int64_t> sq{B, Nq, D}, skv{B, Nkv, D}, sscale{Nkv * D};
    Out out;
    {
        Graph g("drqk");
        vector<Operator> inputs;
        vector<Tensor> its;
        auto add = [&](const char* nm, const vector<int64_t>& sh, DataType dt, const vector<uint8_t>& bytes) {
            auto n = op::Data(nm).set_attr_index(0);
            auto d = Desc(sh, dt);
            n.update_input_desc_x(d);
            n.update_output_desc_y(d);
            g.AddOp(n);
            its.push_back(Tensor(d, bytes.data(), bytes.size()));
            inputs.push_back(n);
            return n;
        };
        vector<uint8_t> bx, bcos, bsl, bidx, bf, bkc;
        if (c.int32x)
            Put<int32_t>(bx, sx, 1);
        else
            Put16(bx, sx, c.bf16 ? 0x3F80 : 0x3C00);
        Put16(bcos, sc, c.bf16 ? 0x3F80 : 0x3C00);
        Put<int8_t>(bkc, skc, 9);
        if (c.page)
            PutIdx(bidx, si, c.pageIdx, DT_INT32);
        else
            PutIdx(bidx, si, c.contigIdx, DT_INT32);
        Put<float>(bf, sscale, 2.0f);
        auto x = add("x", sx, c.int32x ? DT_INT32 : (c.bf16 ? DT_BF16 : DT_FLOAT16), bx);
        auto cos = add("cos", sc, c.bf16 ? DT_BF16 : DT_FLOAT16, bcos);
        auto sin = add("sin", sc, c.bf16 ? DT_BF16 : DT_FLOAT16, bcos);
        auto kc = add("k_cache", skc, DT_INT8, bkc);
        auto vc = add("v_cache", skc, DT_INT8, bkc);
        auto idx = add("indices", si, (c.name == string("int64_indices")) ? DT_INT64 : DT_INT32, bidx);
        auto sk = add("scale_k", sscale, DT_FLOAT, bf);
        auto sv = add("scale_v", sscale, DT_FLOAT, bf);
        vector<uint8_t> bok, bov;
        if (c.useOk)
            Put<float>(bok, sscale, c.okv);
        if (c.useOv)
            Put<float>(bov, sscale, c.ovv);
        auto op = op::DequantRopeQuantKvcache("drqk_node");
        op.set_input_x(x);
        op.set_input_cos(cos);
        op.set_input_sin(sin);
        op.set_input_k_cache(kc);
        op.set_input_v_cache(vc);
        op.set_input_indices(idx);
        op.set_input_scale_k(sk);
        op.set_input_scale_v(sv);
        if (c.useOk) {
            auto t = add("offset_k", sscale, DT_FLOAT, bok);
            op.set_input_offset_k(t);
        }
        if (c.useOv) {
            auto t = add("offset_v", sscale, DT_FLOAT, bov);
            op.set_input_offset_v(t);
        }
        if (c.useWeight) {
            vector<int64_t> sw{H};
            vector<uint8_t> bw;
            Put<float>(bw, sw, 1.0f);
            auto t = add("weight_scale", sw, DT_FLOAT, bw);
            op.set_input_weight_scale(t);
        }
        if (c.useAct) {
            vector<int64_t> sa{B};
            vector<uint8_t> ba;
            Put<float>(ba, sa, 1.0f);
            auto t = add("activation_scale", sa, DT_FLOAT, ba);
            op.set_input_activation_scale(t);
        }
        if (c.useBias) {
            vector<int64_t> sb{H};
            vector<uint8_t> bb;
            Put<int32_t>(bb, sb, 0);
            auto t = add("bias", sb, DT_INT32, bb);
            op.set_input_bias(t);
        }
        vector<int64_t> splits =
            c.splitsOverride.empty() ? vector<int64_t>{Nq * D, Nkv * D, Nkv * D} : c.splitsOverride;
        op.set_attr_size_splits(splits);
        op.set_attr_quant_mode(c.quantMode);
        op.set_attr_layout(c.layout);
        op.set_attr_kv_output(true);
        op.set_attr_cache_mode(c.cacheMode ? c.cacheMode : (c.page ? "page" : "contiguous"));
        op.update_output_desc_q(Desc(sq, c.int32x ? DT_FLOAT16 : (c.bf16 ? DT_BF16 : DT_FLOAT16)));
        op.update_output_desc_k(Desc(skv, c.int32x ? DT_FLOAT16 : (c.bf16 ? DT_BF16 : DT_FLOAT16)));
        op.update_output_desc_v(Desc(skv, c.int32x ? DT_FLOAT16 : (c.bf16 ? DT_BF16 : DT_FLOAT16)));
        op.update_output_desc_k_cache(Desc(skc, DT_INT8));
        op.update_output_desc_v_cache(Desc(skc, DT_INT8));
        g.AddOp(op);
        g.SetInputs(inputs).SetOutputs({op});
        std::map<AscendString, AscendString> bo;
        Session* sess = new Session(bo);
        uint32_t gid = 0;
        out.st = sess->AddGraph(gid, g);
        vector<Tensor> outs;
        if (out.st == SUCCESS)
            out.st = sess->RunGraph(gid, its, outs);
        if (out.st != SUCCESS)
            out.err = GEGetErrorMsgV2().GetString();
        for (auto& t : outs) {
            size_t sz = (size_t)t.GetTensorDesc().GetShape().GetShapeSize();
            DataType dt = t.GetTensorDesc().GetDataType();
            size_t es = (dt == DT_INT8) ? 1 : (dt == DT_INT32 || dt == DT_FLOAT) ? 4 : 2;
            out.bytes.emplace_back((uint8_t*)t.GetData(), (uint8_t*)t.GetData() + sz * es);
        }
        delete sess;
    }
    return out;
}

static bool In(const vector<int>& v, int x)
{
    return std::find(v.begin(), v.end(), x) != v.end();
}
static bool CheckHist(const vector<uint8_t>& t, const vector<int>& allowed, int64_t expectWritten, string& msg)
{
    int64_t n = (int64_t)t.size();
    const int8_t* p = (const int8_t*)t.data();
    int64_t written = 0;
    int bad = 0;
    for (int64_t i = 0; i < n; ++i) {
        if (p[i] == 9)
            continue;
        ++written;
        if (!In(allowed, (int)p[i]))
            ++bad;
    }
    char buf[400];
    int off = snprintf(buf, sizeof(buf), "written=%ld/%ld bad=%d hist:", (long)written, (long)expectWritten, bad);
    int hist[256] = {0};
    for (int64_t i = 0; i < n; ++i)
        hist[(uint8_t)p[i]]++;
    for (int i = 0; i < 256 && off < 380; ++i)
        if (hist[i])
            off += snprintf(buf + off, sizeof(buf) - off, " %d:%d", (int8_t)i, hist[i]);
    msg = buf;
    return bad == 0 && (expectWritten < 0 || written == expectWritten);
}

static void Case(const Cfg& c, int dev)
{
    Out r = Run(c, dev);
    string tag = c.name;
    if (c.expectFail) {
        printf("[%s] %s ret=%d err=%s\n", (r.st != SUCCESS) ? "PASS" : "FAIL", tag.c_str(), (int)r.st,
               r.err.substr(0, 160).c_str());
        return;
    }
    if (r.st != SUCCESS || r.bytes.size() < 5) {
        printf("[FAIL] %s ret=%d err=%s\n", tag.c_str(), (int)r.st, r.err.substr(0, 200).c_str());
        return;
    }
    int64_t nkvD = c.B * c.Nkv * c.D;
    int64_t expectWritten = c.expectWritten >= 0 ? c.expectWritten : ((c.pageIdx > 0 || c.contigIdx > 0) ? 0 : nkvD);
    string km, vm;
    bool kok = CheckHist(r.bytes[3], c.kAllowed, expectWritten, km);
    bool vok = CheckHist(r.bytes[4], c.vAllowed, expectWritten, vm);
    bool det = true;
    if (c.determinism) {
        for (int round = 0; round < 10 && det; ++round) {
            Out r2 = Run(c, dev);
            det = (r2.st == SUCCESS && r2.bytes.size() >= 5 && r.bytes[3] == r2.bytes[3] && r.bytes[4] == r2.bytes[4]);
        }
    }
    printf("[%s] %s k{%s}%s v{%s}%s det=%d\n", (kok && vok && det) ? "PASS" : "FAIL", tag.c_str(), km.c_str(),
           kok ? "ok" : "BAD", vm.c_str(), vok ? "ok" : "BAD", (int)det);
}

int main(int argc, char** argv)
{
    int dev = argc > 1 ? atoi(argv[1]) : 0;
    std::map<AscendString, AscendString> opts = {{"ge.exec.deviceId", std::to_string(dev).c_str()},
                                                 {"ge.graphRunMode", "1"}};
    if (GEInitialize(opts) != SUCCESS) {
        printf("GEInitialize fail\n");
        return 1;
    }
    Cfg base;
    base.name = "base_f16";
    base.kAllowed = {2, 3};
    base.vAllowed = {2};
    Case(base, dev);

    Cfg bf = base;
    bf.name = "base_bf16";
    bf.bf16 = true;
    Case(bf, dev);

    Cfg d16;
    d16.name = "crash_d16";
    d16.D = 16;
    d16.Nq = 6;
    d16.Nkv = 1;
    d16.kAllowed = {2, 3};
    d16.vAllowed = {2};
    Case(d16, dev);

    Cfg d32;
    d32.name = "small_nkvd_32";
    d32.D = 32;
    d32.Nq = 6;
    d32.Nkv = 1;
    d32.kAllowed = {2, 3};
    d32.vAllowed = {2};
    Case(d32, dev);

    Cfg d48;
    d48.name = "small_nkvd_48";
    d48.D = 48;
    d48.Nq = 6;
    d48.Nkv = 1;
    d48.kAllowed = {2, 3};
    d48.vAllowed = {2};
    Case(d48, dev);

    Cfg r2048;
    r2048.name = "repeat2048";
    r2048.D = 128;
    r2048.Nq = 0;
    r2048.Nkv = 16;
    r2048.B = 2;
    r2048.kAllowed = {2, 3};
    r2048.vAllowed = {2};
    Case(r2048, dev);

    Cfg okOnly;
    okOnly.name = "offset_only_k";
    okOnly.useOv = false;
    okOnly.kAllowed = {2, 3};
    okOnly.vAllowed = {0};
    Case(okOnly, dev);

    Cfg both2;
    both2.name = "offset_k0_v5";
    both2.okv = 0.0f;
    both2.ovv = 5.0f;
    both2.kAllowed = {0, 1};
    both2.vAllowed = {5, 6};
    Case(both2, dev);

    Cfg ovOnly;
    ovOnly.name = "offset_only_v";
    ovOnly.useOk = false;
    ovOnly.ovv = 5.0f;
    ovOnly.kAllowed = {0, 1};
    ovOnly.vAllowed = {5, 6};
    Case(ovOnly, dev);

    Cfg pgdup;
    pgdup.name = "page_dup_det";
    pgdup.page = true;
    pgdup.B = 4;
    pgdup.C1 = 2;
    pgdup.C2 = 8;
    pgdup.pageIdx = 0;
    pgdup.determinism = true;
    pgdup.kAllowed = {2, 3};
    pgdup.vAllowed = {2};
    pgdup.expectWritten = 128;
    Case(pgdup, dev);

    Cfg pgoob;
    pgoob.name = "page_oob";
    pgoob.page = true;
    pgoob.C1 = 2;
    pgoob.C2 = 8;
    pgoob.pageIdx = 16;
    pgoob.kAllowed = {};
    pgoob.vAllowed = {};
    Case(pgoob, dev);

    Cfg ctoob;
    ctoob.name = "contig_oob";
    ctoob.contigIdx = 8;
    ctoob.kAllowed = {};
    ctoob.vAllowed = {};
    Case(ctoob, dev);

    Cfg empty;
    empty.name = "empty_b0";
    empty.B = 0;
    empty.kAllowed = {};
    empty.vAllowed = {};
    Case(empty, dev);

    Cfg i64;
    i64.name = "int64_indices";
    i64.expectFail = true;
    Case(i64, dev);

    Cfg sa;
    sa.name = "string_attr_bad";
    sa.quantMode = "bad";
    sa.kAllowed = {2, 3};
    sa.vAllowed = {2};
    Case(sa, dev);

    Cfg i2d;
    i2d.name = "indices_2d";
    i2d.idx2d = true;
    i2d.kAllowed = {2, 3};
    i2d.vAllowed = {2};
    Case(i2d, dev);

    Cfg xt;
    xt.name = "x_tail_48";
    xt.D = 16;
    xt.Nq = 1;
    xt.Nkv = 1;
    xt.expectFail = true;
    Case(xt, dev);

    Cfg d24;
    d24.name = "d_not16";
    d24.D = 24;
    d24.Nq = 6;
    d24.Nkv = 1;
    d24.expectFail = true;
    Case(d24, dev);

    Cfg sp;
    sp.name = "splits_bad";
    sp.expectFail = true;
    sp.splitsOverride = {1024};
    Case(sp, dev);

    Cfg dq;
    dq.name = "int32_dequant";
    dq.int32x = true;
    dq.useWeight = true;
    dq.useAct = true;
    dq.useBias = true;
    dq.kAllowed = {2, 3};
    dq.vAllowed = {2};
    Case(dq, dev);
    return 0;
}
