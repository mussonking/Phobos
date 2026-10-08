#pragma once
// The CPU instruction set the GLM host kernels run on, chosen once per process: the best the CPU has, or lower with
// GLM_CPU_ISA=vnni|avx2|scalar (auto = unset). avx2 on an AVX512 VNNI machine runs the community path (AVX2 + FMA +
// F16C: Intel 12th-14th gen, Core Ultra, Ryzen <= 5000) end to end; asking for more than the CPU has fails.
//   vnni:   AVX512 F/BW/VL/VNNI + F16C: R8 arena, VNNI miss and prefill kernels (the measured prod path)
//   avx2:   AVX2 + FMA + F16C: GGUF arena, AVX2 int8 miss kernels, no prefill CPU share (every miss uploaded)
//   scalar: neither: exact F32 miss rows (reference), no prefill CPU share
#include <cstdlib>
#include <cstring>
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#include <cpuid.h>
#endif
#include <stdexcept>
#include <string>

namespace strata::glm {

enum class CpuIsa { scalar = 0, avx2 = 1, vnni = 2 };

inline const char * cpu_isa_name(CpuIsa i) {
    return i == CpuIsa::vnni ? "vnni" : i == CpuIsa::avx2 ? "avx2" : "scalar";
}

inline CpuIsa cpu_isa_hardware() {
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
    __builtin_cpu_init();
    const bool f16c = __builtin_cpu_supports("f16c");
    if (f16c && __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
        __builtin_cpu_supports("avx512vl") && __builtin_cpu_supports("avx512vnni"))
        return CpuIsa::vnni;
    if (f16c && __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) return CpuIsa::avx2;
#endif
    return CpuIsa::scalar;
}

// AVX-VNNI (VEX ymm vpdpbusd, CPUID 7.1 EAX bit 4: Intel 12th gen+ / Core Ultra, Zen 5): reported only, no kernel
// uses it yet (an AVX2 machine with it would take a ymm form of the VNNI kernels)
inline bool cpu_has_avx_vnni() {
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
    __builtin_cpu_init();
    unsigned a = 0, b = 0, c = 0, d = 0;
    return __builtin_cpu_supports("avx2") && __get_cpuid_count(7, 1, &a, &b, &c, &d) && (a & (1u << 4));
#else
    return false;
#endif
}

inline CpuIsa cpu_isa() {
    static const CpuIsa isa = [] {
        const CpuIsa hw = cpu_isa_hardware();
        const char * e = std::getenv("GLM_CPU_ISA");
        if (!e || !*e || !std::strcmp(e, "auto")) return hw;
        CpuIsa want;
        if (!std::strcmp(e, "vnni")) want = CpuIsa::vnni;
        else if (!std::strcmp(e, "avx2")) want = CpuIsa::avx2;
        else if (!std::strcmp(e, "scalar")) want = CpuIsa::scalar;
        else throw std::runtime_error(std::string("GLM_CPU_ISA=") + e + ": expected auto, vnni, avx2 or scalar");
        if (want > hw)
            throw std::runtime_error(std::string("GLM_CPU_ISA=") + e + ": this CPU only has " + cpu_isa_name(hw));
        return want;
    }();
    return isa;
}

}  // namespace strata::glm
