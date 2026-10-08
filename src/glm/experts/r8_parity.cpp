#include "r8.hpp"
#include "ggml.h"
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace strata::glm;
namespace {
int checks = 0;
void check(bool pass, const std::string & name) {
    ++checks;
    if (!pass) throw std::runtime_error(name);
}
template<class F> void rejects(F f, const std::string & name) {
    bool rejected = false;
    try { f(); } catch (const std::invalid_argument &) { rejected = true; }
    check(rejected, "not rejected: " + name);
}
template<class T> T read(const void * p) {
    T v;
    std::memcpy(&v, p, sizeof(v));
    return v;
}
// Independent IEEE binary16 decoding, without GGML's F32 dequantizer.
double half(uint16_t h) {
    const int exponent = (h >> 10) & 31, fraction = h & 1023;
    const double v = exponent == 0 ? std::ldexp(double(fraction), -24)
        : exponent == 31 ? (fraction ? std::numeric_limits<double>::quiet_NaN()
                                     : std::numeric_limits<double>::infinity())
                         : std::ldexp(double(1024 + fraction), exponent - 25);
    return h & 0x8000 ? -v : v;
}
const int lut[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
using Convert = void (*)(const void *, void *, int);
using UnpackRows = void (*)(const void *, std::size_t, int, int, int, int, void *, std::size_t);
using Rows = void (*)(const void *, int, const R8Act *, int, int, float *, IQ4VNNIPath);
Convert packer(bool xs) { return xs ? r8_repack_xs : r8_repack_nl; }
Convert unpacker(bool xs) { return xs ? r8_unpack_xs : r8_unpack_nl; }
UnpackRows bounded(bool xs) { return xs ? r8_unpack_xs_rows : r8_unpack_nl_rows; }
Rows rows_fn(bool xs) { return xs ? r8_rows_xs : r8_rows_nl; }
std::size_t stride(bool xs, int width) { return std::size_t(width / (xs ? 256 : 32)) * (xs ? 136 : 18); }
std::vector<unsigned char> weights(bool xs, int width, int rows, std::mt19937 & rng) {
    std::vector<unsigned char> result(stride(xs, width) * rows);
    const uint16_t scales[] = {0, 0x8000, 0x1400, 0x9400, 0x1800, 0x9800, 0x2400, 0xa400};
    const int bs = xs ? 136 : 18;
    for (std::size_t i = 0; i < result.size(); i += bs) {
        for (int j = 0; j < bs; ++j) result[i + j] = static_cast<unsigned char>(rng());
        const uint16_t d = scales[(i / bs) % 8];
        std::memcpy(result.data() + i, &d, 2);
        if (xs) { // Every six-bit signed subscale, including -32, 0 and +31.
            uint16_t high = 0;
            for (int g = 0; g < 8; ++g) {
                const int s = int((i / bs * 8 + g) % 64);
                if (!(g & 1)) result[i + 4 + g / 2] = 0;
                result[i + 4 + g / 2] |= (s & 15) << (4 * (g & 1));
                high |= (s >> 4) << (2 * g);
            }
            std::memcpy(result.data() + i + 2, &high, 2);
        }
    }
    return result;
}
std::vector<unsigned char> repack(bool xs, int width, const unsigned char * src, int rows) {
    const auto group = 8 * stride(xs, width);
    std::vector<unsigned char> out(group * (rows / 8) + 2, 0xa5);
    for (int r = 0; r < rows; r += 8)
        packer(xs)(src + r * stride(xs, width), out.data() + 1 + r * stride(xs, width), width);
    check(out.front() == 0xa5 && out.back() == 0xa5, "repack guard");
    return out;
}
void check_layout() {
    // Assign each logical code and six-bit subscale independently, then inspect
    // every physical field. This does not use unpack or implementation helpers.
    for (bool xs : {false, true}) {
        const int width = xs ? 512 : 96, block = xs ? 256 : 32;
        const int bs = xs ? 136 : 18, tiles = width / block;
        std::vector<unsigned char> src(8 * stride(xs, width), 0);
        for (int r = 0; r < 8; ++r) for (int b = 0; b < tiles; ++b) {
            auto * w = src.data() + r * stride(xs, width) + b * bs;
            const uint16_t d = uint16_t(0x8000u | (r * 257 + b * 13));
            std::memcpy(w, &d, 2);
            for (int g = 0; g < (xs ? 8 : 1); ++g) {
                if (xs) {
                    const int s = (g * 8 + r + 11 * b) % 64;
                    w[4 + g / 2] |= (s & 15) << (4 * (g % 2));
                    auto hi = read<uint16_t>(w + 2);
                    hi |= (s >> 4) << (2 * g);
                    std::memcpy(w + 2, &hi, 2);
                }
                for (int k = 0; k < 32; ++k) {
                    const int code = (r * 3 + b * 7 + g * 5 + k) % 16;
                    w[(xs ? 8 : 2) + g * 16 + k % 16] |= code << (4 * (k / 16));
                }
            }
        }
        const auto packed = repack(xs, width, src.data(), 8);
        for (int b = 0; b < tiles; ++b) {
            const auto * tile = packed.data() + 1 + b * (xs ? 1088 : 144);
            for (int r = 0; r < 8; ++r) {
                check(read<uint16_t>(tile + 2 * r) == uint16_t(0x8000u | (r * 257 + b * 13)), "known tile FP16 bits");
                for (int g = 0; g < (xs ? 8 : 1); ++g) {
                    if (xs) {
                        const int j = g * 8 + r, s = (g * 8 + r + 11 * b) % 64;
                        const int low = (tile[16 + j % 32] >> (4 * (j / 32))) & 15;
                        const int high = (tile[48 + j % 16] >> (2 * (j / 16))) & 3;
                        check((low | (high << 4)) == s, "known tile six-bit subscale");
                    }
                    for (int k = 0; k < 32; ++k) {
                        const int lane = r * 2 + (k % 8) / 4;
                        const int offset = (xs ? 64 + 128 * g : 16) + 64 * (k / 16) + lane * 4 + k % 4;
                        const int code = (tile[offset] >> (4 * ((k % 16) / 8))) & 15;
                        check(code == (r * 3 + b * 7 + g * 5 + k) % 16, "known tile nibble lane");
                    }
                }
            }
        }
    }
}
void check_conversion(bool xs, int width, int groups, std::mt19937 & rng) {
    const int rows = 8 * groups;
    const auto rb = stride(xs, width), bytes = rb * rows;
    auto src = weights(xs, width, rows, rng);
    // Conversion is bit-preserving even for scales unsuitable for arithmetic.
    const uint16_t special[] = {0x0000, 0x8000, 0x7c00, 0xfc00, 0x7e01, 0xfe55, 0x7c01, 0xfc01};
    const int block_bytes = xs ? 136 : 18;
    for (std::size_t offset = 0; offset < src.size(); offset += block_bytes) {
        const auto index = offset / block_bytes;
        const uint16_t bits = index % 16 < 8 ? special[index % 8] : static_cast<uint16_t>(rng());
        std::memcpy(src.data() + offset, &bits, sizeof(bits));
    }
    src.insert(src.begin(), 0xa5);
    src.push_back(0xa5);
    check(r8_row_bytes(xs, width) == rb && r8_group_bytes(xs, width) == rb * 8, "byte size formula");
    auto packed = repack(xs, width, src.data() + 1, rows);
    const auto immutable = packed;
    std::vector<unsigned char> result(bytes + 2, 0xa5);
    for (int r = 0; r < rows; r += 8)
        unpacker(xs)(packed.data() + 1 + r * rb, result.data() + 1 + r * rb, width);
    check(result == src, "byte-unaligned whole-plane roundtrip");
    for (int first = 0; first < rows; ++first) {
        const int count = std::min(rows - first, 1 + (first * 7) % 17);
        std::vector<unsigned char> slice(rb * count + 2, 0xa5);
        bounded(xs)(packed.data() + 1, bytes, width, rows, first, count, slice.data() + 1, rb * count);
        check(slice.front() == 0xa5 && slice.back() == 0xa5, "bounded slice guards");
        check(!std::memcmp(slice.data() + 1, src.data() + 1 + rb * first, rb * count), "arbitrary bounded rows");
    }
    auto bad = [&](int plane, int first, int count, std::size_t sb, std::size_t db) {
        rejects([&] { bounded(xs)(packed.data() + 1, sb, width, plane, first, count, result.data() + 1, db); }, "bounded sizes/range");
    };
    bad(rows - 1, 0, 1, bytes, bytes);
    bad(rows, -1, 1, bytes, bytes);
    bad(rows, rows, 1, bytes, bytes);
    bad(rows, rows - 1, 2, bytes, bytes);
    bad(rows, 0, -1, bytes, bytes);
    bad(rows, 0, 1, bytes - 1, bytes);
    bad(rows, 0, 1, bytes, rb - 1);
    rejects([&] { bounded(xs)(nullptr, bytes, width, rows, 0, 1, result.data(), bytes); }, "null bounded source");
    rejects([&] { bounded(xs)(packed.data() + 1, bytes, width, rows, 0, 1, nullptr, bytes); }, "null bounded destination");
    rejects([&] { bounded(xs)(packed.data() + 1, bytes, width, rows, 0, 1, packed.data() + 2, rb); }, "bounded overlap");
    rejects([&] { packer(xs)(src.data() + 1, src.data() + 2, width); }, "repack overlap");
    rejects([&] { unpacker(xs)(packed.data() + 1, packed.data() + 2, width); }, "unpack overlap");
    rejects([&] { packer(xs)(nullptr, result.data(), width); }, "null repack");
    rejects([&] { unpacker(xs)(packed.data(), nullptr, width); }, "null unpack");
    check(immutable == packed, "conversion source immutable");
}
struct Activation {
    std::vector<unsigned char> native, x64;
    std::vector<int64_t> correction;
    std::vector<float> scale;
    R8Act act;
};
Activation activation(bool q0, int width, int trial) {
    Activation a;
    const int bs = q0 ? 34 : 292, block = q0 ? 32 : 256;
    a.native.resize(std::size_t(width / block) * bs + 2, 0xa5);
    a.correction.resize(width / 32);
    a.scale.resize(width / block);
    for (int b = 0; b < width / block; ++b) {
        if (q0) {
            block_q8_0 q{};
            const uint16_t bits[] = {0, 0x8000, 0x2800, 0xa800, 0x3555, 0xb555};
            q.d = bits[(b + trial) % 6];
            for (int j = 0; j < 32; ++j) q.qs[j] = trial == 0 ? 0 : static_cast<int8_t>((b * 32 + j + trial * 19) % 256 - 128);
            q.qs[0] = trial == 0 ? 0 : -128;
            std::memcpy(a.native.data() + 1 + b * bs, &q, bs);
        } else {
            block_q8_K q{};
            q.d = trial == 0 ? 0.0f : std::ldexp(float((b + trial) % 2 ? -3 : 5), -9 - b % 3);
            for (int j = 0; j < 256; ++j) q.qs[j] = trial == 0 ? 0 : static_cast<int8_t>((j + b + trial * 19) % 256 - 128);
            q.qs[0] = trial == 0 ? 0 : -128;
            for (int j = 0; j < 256; ++j) q.bsums[j / 16] += q.qs[j];
            std::memcpy(a.native.data() + 1 + b * bs, &q, bs);
        }
    }
    a.act = q0 ? r8_prep_q8_0(a.native.data() + 1, width, a.correction.data(), a.scale.data())
               : r8_prep_q8k(a.native.data() + 1, width, a.correction.data(), a.scale.data());
    return a;
}
struct Reference { double sum = 0, absolute = 0; };
Reference reference(bool xs, int width, const unsigned char * row, bool q0, const unsigned char * native) {
    Reference result;
    for (int i = 0; i < width; ++i) {
        const int wb = xs ? 256 : 32, group = (i % wb) / 32, lane = i % 32;
        const auto * w = row + (i / wb) * (xs ? 136 : 18);
        const unsigned char nib = w[(xs ? 8 : 2) + group * 16 + lane % 16];
        const int code = (nib >> (lane < 16 ? 0 : 4)) & 15;
        int s = 1;
        if (xs) s = ((w[4 + group / 2] >> (4 * (group & 1))) & 15)
                    + (((read<uint16_t>(w + 2) >> (2 * group)) & 3) << 4) - 32;
        const int ab = q0 ? 32 : 256, offset = q0 ? 2 : 4;
        const auto * a = native + (i / ab) * (q0 ? 34 : 292);
        const double d = q0 ? half(read<uint16_t>(a)) : double(read<float>(a));
        const double term = half(read<uint16_t>(w)) * s * lut[code] * d * read<int8_t>(a + offset + i % ab);
        result.sum += term;
        result.absolute += std::abs(term);
    }
    return result;
}
double error_bound(const Reference & ref, int width) {
    // Integer dots are exact. The longest legacy F32 chain accumulates one
    // term per 32 weights; twelve extra operations cover conversions, scale
    // products and pair/horizontal reductions. R8 XS's eight group FMAs plus
    // width/256 block FMAs have a shorter chain. No empirical tolerance floor.
    const double u = std::ldexp(1.0, -24), n = width / 32 + 12;
    return (n * u / (1 - n * u)) * ref.absolute
         + n * double(std::numeric_limits<float>::denorm_min());
}
void within(float got, const Reference & ref, int width, const std::string & label) {
    const double bound = error_bound(ref, width);
    check(std::isfinite(got) && std::abs(double(got) - ref.sum) <= bound,
          label + " err=" + std::to_string(std::abs(double(got) - ref.sum)) + " bound=" + std::to_string(bound));
}
void math_case(bool xs, bool q0, int width, const unsigned char * raw, int rows,
               std::vector<Activation> & activations, int max_act) {
    auto packed = repack(xs, width, raw, rows);
    const auto frozen = packed;
    std::vector<R8Act> acts;
    for (auto & a : activations) acts.push_back(a.act);
    const int type = xs ? GGML_TYPE_IQ4_XS : GGML_TYPE_IQ4_NL;
    std::vector<IQ4VNNIPath> paths{IQ4VNNIPath::scalar, IQ4VNNIPath::automatic};
    if (r8_available()) paths.push_back(IQ4VNNIPath::vnni);
    for (int count = 1; count <= max_act; ++count) {
        std::vector<float> got(std::size_t(count) * rows + 2, 53), old(rows);
        for (auto path : paths) {
            rows_fn(xs)(packed.data() + 1, width, acts.data(), count, rows, got.data() + 1, path);
            check(got.front() == 53 && got.back() == 53, "row output guards");
            for (int t = 0; t < count; ++t) {
                const auto * native = activations[t].native.data() + 1;
                // Legacy APIs require block alignment; R8 and the independent
                // decoder retain the byte-unaligned, byte-identical source.
                if (q0) {
                    std::vector<block_q8_0> aligned(width / 32);
                    std::memcpy(aligned.data(), native, aligned.size() * sizeof(block_q8_0));
                    iq4_vnni_nl_q8_0_rows(raw, width, aligned.data(), rows, old.data(), path);
                } else {
                    std::vector<block_q8_K> aligned(width / 256);
                    std::memcpy(aligned.data(), native, aligned.size() * sizeof(block_q8_K));
                    iq4_vnni_rows(type, raw, width, aligned.data(), rows, old.data(), path);
                }
                for (int r = 0; r < rows; ++r) {
                    const auto ref = reference(xs, width, raw + stride(xs, width) * r, q0, native);
                    within(got[1 + t * rows + r], ref, width, "R8 decoded-double parity");
                    within(old[r], ref, width, "legacy identical-native decoded-double parity");
                    const double bound = 2 * error_bound(ref, width);
                    check(std::abs(double(got[1 + t * rows + r]) - old[r]) <= bound, "R8 versus legacy triangle bound");
                }
            }
        }
    }
    if (q0) {
        for (auto & a : activations) {
            a.x64.resize(std::size_t(width / 32) * 64 + 2, 0xa5);
            for (int b = 0; b < width / 32; ++b) {
                r8_x64_store_q8_0(a.native.data() + 1 + b * 34, a.x64.data() + 1 + b * 64);
                check(read<uint16_t>(a.x64.data() + 1 + b * 64 + 44)
                      == read<uint16_t>(a.native.data() + 1 + b * 34), "x64 native FP16 bits at offset44");
                check(!std::memcmp(a.native.data() + 1 + b * 34 + 2, a.x64.data() + 1 + b * 64, 32), "x64 native int8 bytes");
                check(read<float>(a.x64.data() + 1 + b * 64 + 40) == half(read<uint16_t>(a.native.data() + 1 + b * 34)), "x64 native exact scale");
            }
            check(a.x64.front() == 0xa5 && a.x64.back() == 0xa5, "x64 guards");
            a.act = r8_x64_act(a.x64.data() + 1, width);
        }
        acts.clear();
        for (auto & a : activations) acts.push_back(a.act);
        std::vector<float> out(std::size_t(max_act) * rows);
        for (auto path : paths) {
            rows_fn(xs)(packed.data() + 1, width, acts.data(), max_act, rows, out.data(), path);
            for (int t = 0; t < max_act; ++t) for (int r = 0; r < rows; ++r)
                within(out[t * rows + r], reference(xs, width, raw + r * stride(xs, width), q0,
                       activations[t].native.data() + 1), width, "x64 native decoded-double parity");
        }
    }
    check(packed == frozen, "math weights immutable");
}
void invalid_math() {
    auto a = activation(false, 256, 1);
    std::vector<unsigned char> w(1088), destination(1088);
    std::vector<float> out(66, 53);
    const auto old = out;
    auto reject = [&](const void * weight, int width, const R8Act * act, int count, int rows, float * output, IQ4VNNIPath path) {
        rejects([&] { r8_rows_xs(weight, width, act, count, rows, output, path); }, "rows invalid contract");
        check(out == old, "invalid rows write output");
    };
    reject(nullptr, 256, &a.act, 1, 8, out.data(), IQ4VNNIPath::scalar);
    reject(w.data(), 256, nullptr, 1, 8, out.data(), IQ4VNNIPath::scalar);
    reject(w.data(), 256, &a.act, 1, 8, nullptr, IQ4VNNIPath::scalar);
    for (int count : {0, 9}) reject(w.data(), 256, &a.act, count, 8, out.data(), IQ4VNNIPath::scalar);
    for (int rows : {0, 1, 7, 9}) reject(w.data(), 256, &a.act, 1, rows, out.data(), IQ4VNNIPath::scalar);
    for (int width : {0, -256, 32, 255}) reject(w.data(), width, &a.act, 1, 8, out.data(), IQ4VNNIPath::scalar);
    for (int field = 0; field < 9; ++field) {
        auto bad = a.act;
        if (field == 0) bad.qs = nullptr;
        if (field == 1) bad.corr = nullptr;
        if (field == 2) bad.d = nullptr;
        if (field == 3) bad.gpb = 1;
        if (field == 4) bad.gpb = 0;
        if (field == 5) bad.bstride = 255;
        if (field == 6) bad.cstride = 0;
        if (field == 7) bad.dstride = 0;
        if (field == 8) bad.bstride = -1;
        reject(w.data(), 256, &bad, 1, 8, out.data(), IQ4VNNIPath::scalar);
    }
    rejects([&] { r8_rows_xs(w.data(), 256, &a.act, 1, 8, reinterpret_cast<float *>(w.data()), IQ4VNNIPath::scalar); }, "weights/output overlap");
    rejects([&] { r8_rows_xs(w.data(), 256, &a.act, 1, 8, a.scale.data(), IQ4VNNIPath::scalar); }, "activation/output overlap");
    auto nl = activation(true, 256, 1);
    auto bad_nl = nl.act;
    bad_nl.gpb = 2;
    rejects([&] { r8_rows_nl(w.data(), 256, &bad_nl, 1, 8, out.data(), IQ4VNNIPath::scalar); }, "NL invalid GPB");
    R8Act mixed[] = {nl.act, a.act};
    rejects([&] { r8_rows_nl(w.data(), 256, mixed, 2, 8, out.data(), IQ4VNNIPath::scalar); }, "NL heterogeneous GPB");
    for (int field = 0; field < 3; ++field) {
        bad_nl = nl.act;
        if (field == 0) bad_nl.bstride = 31;
        if (field == 1) bad_nl.cstride = -1;
        if (field == 2) bad_nl.dstride = -1;
        rejects([&] { r8_rows_nl(w.data(), 256, &bad_nl, 1, 8, out.data(), IQ4VNNIPath::scalar); }, "NL invalid stride");
    }
    check(out == old, "NL invalid cases leave output untouched");
    rejects([&] { r8_prep_q8k(nullptr, 256, a.correction.data(), a.scale.data()); }, "null prep Q8K");
    rejects([&] { r8_prep_q8k(a.native.data(), 256, nullptr, a.scale.data()); }, "null correction");
    rejects([&] { r8_prep_q8_0(a.native.data(), 256, a.correction.data(), nullptr); }, "null scale");
    rejects([&] { r8_x64_store_q8_0(nullptr, destination.data()); }, "null native x64");
    rejects([&] { r8_x64_act(nullptr, 256); }, "null x64 activation");
    if (!r8_available()) reject(w.data(), 256, &a.act, 1, 8, out.data(), IQ4VNNIPath::vnni);
    for (bool xs : {false, true}) for (int width : {0, -32, 31, 33}) {
        rejects([&] { r8_row_bytes(xs, width); }, "invalid row width");
        rejects([&] { r8_group_bytes(xs, width); }, "invalid group width");
    }
    ExpertRef def;
    ExpertRef four{w.data(), w.data(), GGML_TYPE_IQ4_XS, GGML_TYPE_IQ4_NL};
    ExpertRef six{w.data(), w.data(), GGML_TYPE_IQ4_XS, GGML_TYPE_IQ4_NL, w.data(), -1};
    check(def.layout == 0 && four.layout == 0 && six.layout == 0, "ExpertRef default and legacy aggregate layout");
    check(sizeof(void *) != 8 || (sizeof(ExpertRef) == 40 && offsetof(ExpertRef, layout) == 36), "ExpertRef ABI");
}
void legacy_layout_rejection() {
    Geometry g;
    g.n_embd = 256;
    g.ffn_expert = 256;
    const auto gu_bytes = 2 * stride(true, g.n_embd) * g.ffn_expert;
    const auto down_bytes = stride(false, g.ffn_expert) * g.n_embd;
    std::vector<unsigned char> weights(gu_bytes + down_bytes, 0);
    ExpertRef e{weights.data(), weights.data() + gu_bytes, GGML_TYPE_IQ4_XS, GGML_TYPE_IQ4_NL};
    e.layout = 1;
    std::vector<float> output(g.n_embd, 53);
    const auto unchanged = output;
    std::vector<float> cpu_act((cpu_expert_activation_bytes(g, 1) + 3) / 4, 0);
    std::vector<float> cpu_scratch((cpu_expert_scratch_bytes(g, 1) + 3) / 4, 0);
    rejects([&] { cpu_expert_eval(g, e, cpu_act.data(), 1, 1, nullptr, output.data(), cpu_scratch.data()); },
            "legacy CPU eval rejects host R8 layout");
    check(output == unchanged, "legacy CPU layout rejection precedes output writes");
    std::vector<float> vnni_act((vnni_expert_activation_bytes(g, 1) + 3) / 4, 0);
    std::vector<float> vnni_scratch((vnni_expert_scratch_bytes(g, 1) + 3) / 4, 0);
    rejects([&] { vnni_expert_eval(g, e, vnni_act.data(), 1, 1, nullptr, output.data(), vnni_scratch.data(), IQ4VNNIPath::scalar); },
            "legacy VNNI eval rejects host R8 layout");
    check(output == unchanged, "legacy VNNI layout rejection precedes output writes");
}
std::vector<unsigned char> load(const std::string & path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f || f.tellg() < 0) throw std::runtime_error("cannot load " + path);
    if (f.tellg() > std::streamoff(8 * 13 * 1024 * 1024))
        throw std::runtime_error("R8 fixture exceeds the bounded eight-expert sample: " + path);
    std::vector<unsigned char> bytes(static_cast<std::size_t>(f.tellg()));
    f.seekg(0);
    if (!f.read(reinterpret_cast<char *>(bytes.data()), bytes.size())) throw std::runtime_error("read failed " + path);
    return bytes;
}
void fixture(const std::string & dir) {
    constexpr int width = 4096, mid_width = 2048, T = 4, slice = 32;
    const auto gu = stride(true, width) * mid_width, down = stride(false, mid_width) * width;
    const auto blob = load(dir + "/experts.bin"), xb = load(dir + "/x.f32");
    check(blob.size() == 8 * (2 * gu + down), "blk3-final eight calibrated expert sizes");
    check(xb.size() == std::size_t(T) * width * sizeof(float), "blk3-final captured T4 input size");
    std::vector<float> x(T * width), gate(mid_width), up(mid_width), mid(mid_width);
    std::memcpy(x.data(), xb.data(), xb.size());
    const auto path = r8_available() ? IQ4VNNIPath::vnni : IQ4VNNIPath::scalar;
    for (int e = 0; e < 8; ++e) {
        const auto * base = blob.data() + e * (2 * gu + down);
        std::vector<Activation> input;
        input.reserve(T);
        for (int t = 0; t < T; ++t) {
            input.push_back(activation(false, width, 1));
            auto & a = input.back();
            std::vector<block_q8_K> native(width / 256);
            iq4_vnni_quantize(x.data() + t * width, width, native.data());
            std::memcpy(a.native.data() + 1, native.data(), native.size() * sizeof(block_q8_K));
            a.act = r8_prep_q8k(a.native.data() + 1, width, a.correction.data(), a.scale.data());
        }
        math_case(true, false, width, base, slice, input, T);
        math_case(true, false, width, base + gu, slice, input, T);
        std::vector<Activation> qk, q0;
        qk.reserve(T); q0.reserve(T);
        for (int t = 0; t < T; ++t) {
            std::vector<block_q8_K> aligned(width / 256);
            std::memcpy(aligned.data(), input[t].native.data() + 1, aligned.size() * sizeof(block_q8_K));
            iq4_vnni_rows(GGML_TYPE_IQ4_XS, base, width, aligned.data(), mid_width, gate.data(), path);
            iq4_vnni_rows(GGML_TYPE_IQ4_XS, base + gu, width, aligned.data(), mid_width, up.data(), path);
            for (int r = 0; r < mid_width; ++r) {
                const float gc = std::min(10.0f, gate[r]);
                mid[r] = gc / (1 + std::exp(-gc)) * std::max(-10.0f, std::min(10.0f, up[r]));
            }
            qk.push_back(activation(false, mid_width, 1));
            q0.push_back(activation(true, mid_width, 1));
            auto & k = qk.back(); auto & n = q0.back();
            std::vector<block_q8_K> native_k(mid_width / 256);
            std::vector<block_q8_0> native_0(mid_width / 32);
            iq4_vnni_quantize(mid.data(), mid_width, native_k.data());
            iq4_vnni_quantize_q8_0(mid.data(), mid_width, native_0.data());
            std::memcpy(k.native.data() + 1, native_k.data(), native_k.size() * sizeof(block_q8_K));
            std::memcpy(n.native.data() + 1, native_0.data(), native_0.size() * sizeof(block_q8_0));
            k.act = r8_prep_q8k(k.native.data() + 1, mid_width, k.correction.data(), k.scale.data());
            n.act = r8_prep_q8_0(n.native.data() + 1, mid_width, n.correction.data(), n.scale.data());
        }
        math_case(false, false, mid_width, base + 2 * gu, slice, qk, T);
        math_case(false, true, mid_width, base + 2 * gu, slice, q0, T);
        std::cout << "fixture expert=" << e << " T4 gate/up/down same-native 32-row slice PASS\n";
    }
}
} // namespace
int main(int argc, char ** argv) {
    try {
        bool quick = false, require = false;
        std::string dir;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--quick") quick = true;
            else if (arg == "--require-vnni") require = true;
            else if (arg == "--fixture" && i + 1 < argc) dir = argv[++i];
            else throw std::runtime_error("usage: glm_r8_parity [--quick] [--require-vnni] [--fixture DIR]");
        }
        if (require && !r8_available()) throw std::runtime_error("AVX512 VNNI required but unavailable");
        std::cout << "R8 path=" << (r8_available() ? "forced-vnni+scalar" : "scalar")
                  << " analytical F32 error bounds; not bitwise dot parity\n";
        check_layout();
        invalid_math();
        legacy_layout_rejection();
        std::mt19937 rng(0x52385041);
        for (bool xs : {true, false}) {
            for (int width : {256, 4096, 2048, 32, 96}) {
                if (xs && width % 256) continue;
                for (int groups : {1, 8, 16, 32}) check_conversion(xs, width, groups, rng);
                for (bool q0 : {false, true}) {
                    if ((xs && q0) || (!q0 && width % 256)) continue;
                    for (int trial = 0; trial < (quick ? 2 : 6); ++trial) {
                        auto raw = weights(xs, width, 32, rng);
                        std::vector<Activation> acts;
                        acts.reserve(8);
                        for (int t = 0; t < 8; ++t) acts.push_back(activation(q0, width, trial == 0 ? 0 : t + trial));
                        math_case(xs, q0, width, raw.data(), 32, acts, 8);
                    }
                }
            }
        }
        if (!dir.empty()) fixture(dir);
        std::cout << "R8 parity PASS checks=" << checks << '\n';
        return 0;
    } catch (const std::exception & e) { std::cerr << e.what() << '\n'; return 1; }
}
