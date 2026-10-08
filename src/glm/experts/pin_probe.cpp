// Targeted CUDA-host allocation probe; run only under mrun pack --gpu.
// glm_pin_probe --sizes GB[,GB..]: which page-locking the arena can get at each size (WSL2 refuses cudaHostRegister):
// cudaHostRegister on a touched anonymous mapping, then cudaHostAlloc, each freed before the next; MemAvailable must
// exceed the size + 2 GB.
// glm_pin_probe --blocks MB[,MB..] [--max-gb G]: total page-locked reached with MB-sized blocks (total vs
// per-allocation cap). glm_pin_probe --bw MB: H2D pinned / pageable and host memcpy rates.
#include <cuda_runtime_api.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <unistd.h>

#include <chrono>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::uint64_t mem_available() {
    std::ifstream info("/proc/meminfo");
    std::string key, unit;
    std::uint64_t value;
    while (info >> key >> value >> unit)
        if (key == "MemAvailable:") return value * 1024;
    return 0;
}

double secs_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

int sizes(const std::string & list) {
    std::stringstream ss(list);
    std::string tok;
    int bad = 0;
    while (std::getline(ss, tok, ',')) {
        const double gb = std::stod(tok);
        const std::size_t bytes = (std::size_t) (gb * (1ULL << 30));
        if (mem_available() < bytes + (2ULL << 30)) {
            std::printf("%.0f GiB: skipped (MemAvailable %.1f GiB)\n", gb, mem_available() / double(1ULL << 30));
            continue;
        }
        void * m = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
        if (m == MAP_FAILED) throw std::runtime_error("mmap failed");
        std::memset(m, 0x53, bytes);
        auto t0 = std::chrono::steady_clock::now();
        const cudaError_t er = cudaHostRegister(m, bytes, cudaHostRegisterDefault);
        const double tr = secs_since(t0);
        if (er == cudaSuccess) cudaHostUnregister(m);
        else cudaGetLastError();
        munmap(m, bytes);
        void * h = nullptr;
        t0 = std::chrono::steady_clock::now();
        const cudaError_t ea = cudaHostAlloc(&h, bytes, cudaHostAllocDefault);
        const double ta = secs_since(t0);
        if (ea == cudaSuccess) {
            std::memset(h, 0x53, bytes);
            cudaFreeHost(h);
        } else {
            cudaGetLastError();
        }
        std::printf("%.0f GiB: cudaHostRegister %s (%.2f s), cudaHostAlloc %s (%.2f s)\n", gb,
                    er == cudaSuccess ? "OK" : cudaGetErrorName(er), tr, ea == cudaSuccess ? "OK" : cudaGetErrorName(ea), ta);
        bad += ea != cudaSuccess;
    }
    return bad ? 1 : 0;
}

// --blocks MB[,MB..] [--max-gb G]: per-allocation cap or total cap? For each block size, page-lock MB blocks until
// a call fails or the total reaches G GiB (default MemAvailable - 4 GiB, at most 64): cudaHostRegister on
// consecutive ranges of one touched mapping (as an arena would be split), then separate cudaHostAlloc blocks. Each
// mode frees everything before the next. Prints the total reached and the first error.
int blocks(const std::string & list, double max_gb) {
    const std::size_t avail = mem_available();
    std::size_t cap = avail > (4ULL << 30) ? avail - (4ULL << 30) : 0;
    if (max_gb > 0) cap = std::min(cap, (std::size_t) (max_gb * (1ULL << 30)));
    else cap = std::min(cap, (std::size_t) 64 << 30);
    std::printf("blocks: MemAvailable %.1f GiB, cap %.1f GiB\n", avail / double(1ULL << 30), cap / double(1ULL << 30));
    std::stringstream ss(list);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        const std::size_t blk = (std::size_t) std::stoull(tok) << 20;
        const std::size_t n_max = cap / blk;
        if (!n_max) continue;
        // register: one mapping of n_max blocks, registered range by range
        const std::size_t bytes = n_max * blk;
        auto * m = (uint8_t *) mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
        if (m == (uint8_t *) MAP_FAILED) throw std::runtime_error("mmap failed");
        std::memset(m, 0x53, bytes);
        std::size_t n_reg = 0;
        cudaError_t er = cudaSuccess;
        auto t0 = std::chrono::steady_clock::now();
        for (; n_reg < n_max; ++n_reg)
            if ((er = cudaHostRegister(m + n_reg * blk, blk, cudaHostRegisterDefault)) != cudaSuccess) break;
        const double tr = secs_since(t0);
        cudaGetLastError();
        for (std::size_t i = 0; i < n_reg; ++i) cudaHostUnregister(m + i * blk);
        munmap(m, bytes);
        // hostalloc: separate blocks
        std::vector<void *> hs;
        cudaError_t ea = cudaSuccess;
        t0 = std::chrono::steady_clock::now();
        while (hs.size() < n_max) {
            void * h = nullptr;
            if ((ea = cudaHostAlloc(&h, blk, cudaHostAllocDefault)) != cudaSuccess) break;
            std::memset(h, 0x53, blk);
            hs.push_back(h);
        }
        const double ta = secs_since(t0);
        cudaGetLastError();
        for (void * h : hs) cudaFreeHost(h);
        const double g = blk / double(1ULL << 30);
        std::printf("blocks %zu MB: cudaHostRegister %zu blocks = %.2f GiB%s%s (%.2f s); cudaHostAlloc %zu blocks = %.2f GiB%s%s "
                    "(%.2f s)\n", blk >> 20, n_reg, n_reg * g, n_reg == n_max ? " (cap reached)" : ", then ",
                    n_reg == n_max ? "" : cudaGetErrorName(er), tr, hs.size(), hs.size() * g,
                    hs.size() == n_max ? " (cap reached)" : ", then ", hs.size() == n_max ? "" : cudaGetErrorName(ea), ta);
    }
    return 0;
}

// --bw MB: H2D copy rates for a staged-upload estimate: from a pinned block (cudaMemcpyAsync), from pageable memory
// (cudaMemcpy), and host memcpy pageable -> pinned (the bounce buffer's fill). Best of 5.
int bandwidth(std::size_t mb) {
    const std::size_t bytes = mb << 20;
    void * d = nullptr, * h = nullptr;
    if (cudaMalloc(&d, bytes) != cudaSuccess) throw std::runtime_error("cudaMalloc failed");
    if (cudaHostAlloc(&h, bytes, cudaHostAllocDefault) != cudaSuccess) throw std::runtime_error("cudaHostAlloc failed");
    std::vector<uint8_t> p(bytes, 0x53);
    std::memset(h, 0x53, bytes);
    cudaStream_t st;
    cudaStreamCreate(&st);
    auto best = [&](auto && f) {
        double b = 1e30;
        for (int i = 0; i < 5; ++i) {
            const auto t0 = std::chrono::steady_clock::now();
            f();
            b = std::min(b, secs_since(t0));
        }
        return bytes / b / 1e9;
    };
    const double pin = best([&] { cudaMemcpyAsync(d, h, bytes, cudaMemcpyHostToDevice, st); cudaStreamSynchronize(st); });
    const double pag = best([&] { cudaMemcpy(d, p.data(), bytes, cudaMemcpyHostToDevice); });
    const double cpy = best([&] { std::memcpy(h, p.data(), bytes); });
    std::printf("bw %zu MB: H2D pinned %.1f GB/s, H2D pageable %.1f GB/s, host memcpy into pinned %.1f GB/s\n", mb, pin,
                pag, cpy);
    cudaStreamDestroy(st);
    cudaFreeHost(h);
    cudaFree(d);
    return 0;
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        if (argc == 3 && !std::strcmp(argv[1], "--sizes")) return sizes(argv[2]);
        if ((argc == 3 || (argc == 5 && !std::strcmp(argv[3], "--max-gb"))) && !std::strcmp(argv[1], "--blocks"))
            return blocks(argv[2], argc == 5 ? std::stod(argv[4]) : 0);
        if (argc == 3 && !std::strcmp(argv[1], "--bw")) return bandwidth(std::stoull(argv[2]));
        if (argc != 2) throw std::runtime_error("usage: glm_pin_probe BYTES");
        const std::size_t bytes = std::stoull(argv[1]);
        if (!bytes || bytes > 100000000000ULL)
            throw std::runtime_error("probe allocation must be in (0,100 GB]");
        std::ifstream info("/proc/meminfo");
        std::string key, unit;
        std::uint64_t value, available = 0;
        while (info >> key >> value >> unit) {
            if (key == "MemAvailable:") { available = value * 1024; break; }
        }
        constexpr std::uint64_t reserve = 80ULL * 1024 * 1024 * 1024;
        if (available < bytes + reserve)
            throw std::runtime_error("insufficient MemAvailable; keep 80 GiB reserve for shared services");
        rlimit limit{};
        if (getrlimit(RLIMIT_MEMLOCK, &limit)) throw std::runtime_error("getrlimit failed");
        std::printf("mem_available=%llu bytes=%zu memlock_soft=%llu memlock_hard=%llu\n",
                    (unsigned long long)available, bytes,
                    (unsigned long long)limit.rlim_cur, (unsigned long long)limit.rlim_max);
        void * p = nullptr;
        const auto start = std::chrono::steady_clock::now();
        const auto error = cudaHostAlloc(&p, bytes, cudaHostAllocDefault);
        if (error != cudaSuccess) {
            std::printf("cudaHostAlloc FAILED code=%d name=%s error=%s\n",
                        int(error), cudaGetErrorName(error), cudaGetErrorString(error));
            return 1;
        }
        const auto allocated = std::chrono::steady_clock::now();
        // Touch every page without a second allocation or keeping a file mapping.
        auto * data = static_cast<volatile unsigned char *>(p);
        const std::size_t page = std::size_t(sysconf(_SC_PAGESIZE));
        for (std::size_t i = 0; i < bytes; i += page) data[i] = 0x53;
        if (data[0] != 0x53 || data[(bytes - 1) / page * page] != 0x53) {
            cudaFreeHost(p);
            throw std::runtime_error("pinned page touch verification failed");
        }
        const auto release = cudaFreeHost(p);
        if (release != cudaSuccess) throw std::runtime_error(cudaGetErrorString(release));
        std::printf("cudaHostAlloc PASS bytes=%zu above_memlock=%d all_pages_touched=1 freed=1 alloc_s=%.3f (not a throughput benchmark)\n",
                    bytes, int(bytes > limit.rlim_cur),
                    std::chrono::duration<double>(allocated - start).count());
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
