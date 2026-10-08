// pcie_probe: host -> V100 copy bandwidth for expert streaming (prefill). Pinned buffers placed on each NUMA node
// (mmap + mbind + cudaHostRegister), copied in expert-blob sized pieces (13.6 MB) on 1 or 2 streams; pageable and
// cudaHostAlloc buffers for reference.
//
//   glm_pcie_probe [--mb 2048] [--blob 13631488]
#include <cuda_runtime.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void ck(cudaError_t e, const char * what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

// Anonymous memory bound to one NUMA node (MPOL_BIND = 2), touched so the pages exist before registration.
void * alloc_on_node(size_t bytes, int node) {
    void * p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) throw std::runtime_error("mmap");
    unsigned long mask = 1ul << node;
    if (syscall(SYS_mbind, p, bytes, 2, &mask, sizeof(mask) * 8, 0) != 0) throw std::runtime_error("mbind");
    std::memset(p, 1, bytes);
    return p;
}

double copy_gbs(const char * src, void * dst, size_t bytes, size_t blob, int n_streams, int reps) {
    std::vector<cudaStream_t> st(n_streams);
    for (auto & s : st) ck(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream");
    const size_t n_blobs = bytes / blob;
    auto run = [&] {
        for (size_t b = 0; b < n_blobs; ++b)
            ck(cudaMemcpyAsync((char *) dst + b * blob, src + b * blob, blob, cudaMemcpyHostToDevice,
                               st[b % n_streams]), "copy");
        for (auto & s : st) ck(cudaStreamSynchronize(s), "sync");
    };
    run();   // warm-up
    const auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; ++r) run();
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    for (auto & x : st) cudaStreamDestroy(x);
    return (double) n_blobs * blob * reps / s / 1e9;
}

}  // namespace

int main(int argc, char ** argv) {
    size_t mb = 2048, blob = 13631488;
    for (int i = 1; i + 1 < argc; i += 2) {
        if (!std::strcmp(argv[i], "--mb")) mb = std::strtoull(argv[i + 1], nullptr, 10);
        else if (!std::strcmp(argv[i], "--blob")) blob = std::strtoull(argv[i + 1], nullptr, 10);
    }
    try {
        const size_t bytes = mb << 20;
        cudaDeviceProp prop{};
        ck(cudaGetDeviceProperties(&prop, 0), "props");
        std::printf("device %s, PCI %02x:%02x, async engines %d; %zu MB in blobs of %zu bytes\n", prop.name,
                    prop.pciBusID, prop.pciDeviceID, prop.asyncEngineCount, mb, blob);
        void * dst;
        ck(cudaMalloc(&dst, bytes), "device buffer");
        for (int node = 0; node < 2; ++node) {
            void * h = alloc_on_node(bytes, node);
            ck(cudaHostRegister(h, bytes, cudaHostRegisterDefault), "register");
            for (int ns : {1, 2})
                std::printf("  pinned (registered) on node %d, %d stream(s): %.2f GB/s\n", node, ns,
                            copy_gbs((const char *) h, dst, bytes, blob, ns, 3));
            ck(cudaHostUnregister(h), "unregister");
            std::printf("  pageable on node %d, 1 stream: %.2f GB/s\n", node,
                        copy_gbs((const char *) h, dst, bytes, blob, 1, 1));
            munmap(h, bytes);
        }
        void * ha;
        ck(cudaHostAlloc(&ha, bytes, cudaHostAllocDefault), "hostalloc");
        std::memset(ha, 1, bytes);
        std::printf("  cudaHostAlloc (default policy), 1 stream: %.2f GB/s\n",
                    copy_gbs((const char *) ha, dst, bytes, blob, 1, 3));
        cudaFreeHost(ha);
        cudaFree(dst);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "ERROR: %s\n", e.what());
        return 1;
    }
    return 0;
}
