// ctx_gather_probe: cost of a host-resident MLA latent cache at decode. The DSA indexer selects 512 pools of 4
// consecutive cells per token (2048 cells); with the latent in mapped pinned host memory the attention would read
// those rows over PCIe (zero-copy). This times that gather (random pools out of --ctx cells) from host memory against
// the same gather from device memory, for row sizes of the F16 latent (1024 B) and an int8 one (512 B + scale).
//
//   glm_ctx_gather_probe [--ctx 131072] [--pools 512] [--tokens 1,2,4] [--reps 50]
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void ck(cudaError_t e, const char * what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

std::string arg(int argc, char ** argv, const char * key, const std::string & def) {
    for (int i = 1; i + 1 < argc; ++i)
        if (!std::strcmp(argv[i], key)) return argv[i + 1];
    return def;
}

// one block per selected pool: reads its 4 rows (4 * row_bytes contiguous), 16 B per thread per step
__global__ void gather_pools(const uint4 * __restrict__ src, const int * __restrict__ pools, int row16,
                             unsigned * __restrict__ out) {
    const uint4 * p = src + (size_t) pools[blockIdx.x] * 4 * row16;
    unsigned acc = 0;
    for (int i = threadIdx.x; i < 4 * row16; i += blockDim.x) {
        const uint4 v = p[i];
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0x12345678u) out[blockIdx.x] = acc;   // keeps the loads
}

}  // namespace

int main(int argc, char ** argv) {
    const int64_t ctx = std::atoll(arg(argc, argv, "--ctx", "131072").c_str());
    const int pools = std::atoi(arg(argc, argv, "--pools", "512").c_str());
    const int reps = std::atoi(arg(argc, argv, "--reps", "50").c_str());
    std::vector<int> tokens;
    {
        const std::string s = arg(argc, argv, "--tokens", "1,2,4");
        for (size_t a = 0; a < s.size();) {
            size_t b = s.find(',', a);
            if (b == std::string::npos) b = s.size();
            tokens.push_back(std::atoi(s.substr(a, b - a).c_str()));
            a = b + 1;
        }
    }
    const int64_t n_pool = ctx / 4;
    std::mt19937 rng(42);
    unsigned * d_out = nullptr;
    ck(cudaMalloc(&d_out, 65536 * sizeof(unsigned)), "out");
    cudaEvent_t e0, e1;
    ck(cudaEventCreate(&e0), "event");
    ck(cudaEventCreate(&e1), "event");
    std::printf("ctx %lld cells (%lld pools), %d pools selected per token (DSA top-k), %d reps\n", (long long) ctx,
                (long long) n_pool, pools, reps);
    for (const int row : {1024, 528}) {   // F16 latent row; int8 latent row (512 B) + scale, padded to 16 B
        const size_t bytes = (size_t) ctx * row;
        void * host = nullptr, * dev = nullptr, * mapped = nullptr;
        ck(cudaHostAlloc(&host, bytes, cudaHostAllocMapped), "host alloc");
        std::memset(host, 1, bytes);
        ck(cudaHostGetDevicePointer(&mapped, host, 0), "mapped");
        ck(cudaMalloc(&dev, bytes), "dev alloc");
        ck(cudaMemcpy(dev, host, bytes, cudaMemcpyHostToDevice), "fill");
        for (const int T : tokens) {
            // T tokens: the union of their selections (independent draws: the worst case, no overlap)
            const int n = pools * T;
            int * d_pools = nullptr;
            ck(cudaMalloc(&d_pools, (size_t) n * reps * sizeof(int)), "pools");
            std::vector<int> h((size_t) n * reps);
            for (auto & v : h) v = (int) (rng() % n_pool);
            ck(cudaMemcpy(d_pools, h.data(), h.size() * sizeof(int), cudaMemcpyHostToDevice), "pools up");
            for (const bool from_host : {false, true}) {
                const uint4 * src = static_cast<const uint4 *>(from_host ? mapped : dev);
                gather_pools<<<n, 128>>>(src, d_pools, row / 16, d_out);   // warm-up
                ck(cudaEventRecord(e0), "record");
                for (int r = 0; r < reps; ++r) gather_pools<<<n, 128>>>(src, d_pools + (size_t) r * n, row / 16, d_out);
                ck(cudaEventRecord(e1), "record");
                ck(cudaEventSynchronize(e1), "sync");
                float ms = 0;
                ck(cudaEventElapsedTime(&ms, e0, e1), "elapsed");
                ms /= reps;
                const double gb = (double) n * 4 * row / 1e9;
                std::printf("row %4d B, T %d (%5d pools, %5.2f MB): %s %7.3f ms per layer = %6.2f GB/s; x 11 MLA "
                            "layers = %6.2f ms per step\n",
                            row, T, n, gb * 1e3, from_host ? "host" : "dev ", ms, gb / (ms / 1e3), 11 * ms);
            }
            cudaFree(d_pools);
        }
        cudaFree(dev);
        cudaFreeHost(host);
    }
    return 0;
}
