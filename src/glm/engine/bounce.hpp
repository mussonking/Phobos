#pragma once
// Pinned bounce ring for host -> device copies out of pageable memory (an arena the OS will not page-lock: WSL2 caps
// a process's page-locked memory at 1 GiB in total). Each copy waits for the next ring slot's previous H2D, fills the
// slot with a parallel memcpy (the caller plus a small thread team), then queues the slot's H2D on the caller's
// stream: stream order is that of a plain cudaMemcpyAsync at the call, and the host fills slot k + 1 while slot k's
// DMA runs. One caller at a time (the uploads are queued from one host thread); the mutex only guards misuse.
#include <cuda_runtime.h>

#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace strata::glm {

class BounceRing {
public:
    struct Seg {
        const void * src;
        size_t bytes;
    };

    // n_slots slots of slot_bytes, page-locked as one block; threads = memcpy threads, the caller included.
    // Throws when the block cannot be page-locked.
    BounceRing(int n_slots, size_t slot_bytes, int threads) : n_(n_slots), slot_(slot_bytes) {
        if (n_slots < 2 || !slot_bytes) throw std::runtime_error("bounce ring: needs >= 2 slots");
        void * p = nullptr;
        if (cudaHostAlloc(&p, (size_t) n_slots * slot_bytes, cudaHostAllocDefault) != cudaSuccess) {
            cudaGetLastError();
            throw std::runtime_error("bounce ring: cudaHostAlloc of " + std::to_string(n_slots * slot_bytes >> 20) +
                                     " MB refused");
        }
        buf_ = static_cast<uint8_t *>(p);
        ev_.resize(n_slots);
        for (auto & e : ev_) cudaEventCreateWithFlags(&e, cudaEventDisableTiming);
        threads_ = threads < 1 ? 1 : threads;
        for (int t = 1; t < threads_; ++t) team_.emplace_back([this, t] { work(t); });
    }
    ~BounceRing() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto & t : team_) t.join();
        for (auto e : ev_) {
            cudaEventSynchronize(e);
            cudaEventDestroy(e);
        }
        cudaFreeHost(buf_);
    }
    BounceRing(const BounceRing &) = delete;
    BounceRing & operator=(const BounceRing &) = delete;

    size_t bytes() const { return (size_t) n_ * slot_; }
    int slots() const { return n_; }
    int threads() const { return threads_; }

    // dst (device) <- the segments, back to back; their total must fit a slot
    void copy(void * dst, const Seg * segs, int n_segs, cudaStream_t s) {
        std::lock_guard<std::mutex> call(call_mu_);
        size_t total = 0;
        for (int i = 0; i < n_segs; ++i) total += segs[i].bytes;
        if (total > slot_) throw std::runtime_error("bounce ring: copy larger than a slot");
        const int k = next_;
        next_ = (next_ + 1) % n_;
        if (cudaEventSynchronize(ev_[k]) != cudaSuccess) throw std::runtime_error("bounce ring: slot wait");
        uint8_t * b = buf_ + (size_t) k * slot_;
        {
            std::lock_guard<std::mutex> lk(mu_);
            job_ = {b, segs, n_segs, total};
            done_ = 0;
            ++gen_;
        }
        cv_.notify_all();
        part(0);
        {
            std::unique_lock<std::mutex> lk(mu_);
            done_cv_.wait(lk, [this] { return done_ == threads_ - 1; });
        }
        if (cudaMemcpyAsync(dst, b, total, cudaMemcpyHostToDevice, s) != cudaSuccess ||
            cudaEventRecord(ev_[k], s) != cudaSuccess)
            throw std::runtime_error("bounce ring: H2D");
    }

private:
    struct Job {
        uint8_t * dst = nullptr;
        const Seg * segs = nullptr;
        int n = 0;
        size_t total = 0;
    };
    // thread t's share: bytes [total * t / T, total * (t + 1) / T) of the concatenated segments, 4 KiB aligned
    void part(int t) {
        const Job j = job_;
        const auto edge = [&](int i) { return i >= threads_ ? j.total : (j.total * i / threads_) & ~(size_t) 4095; };
        size_t a = edge(t);
        const size_t e = edge(t + 1);
        size_t off = 0;
        for (int i = 0; i < j.n && a < e; ++i) {
            const size_t s0 = off, s1 = off + j.segs[i].bytes;
            off = s1;
            if (a >= s1) continue;
            const size_t c = std::min(e, s1) - a;
            std::memcpy(j.dst + a, static_cast<const uint8_t *>(j.segs[i].src) + (a - s0), c);
            a += c;
        }
    }
    void work(int t) {
        uint64_t seen = 0;
        for (;;) {
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [&] { return stop_ || gen_ != seen; });
                if (stop_) return;
                seen = gen_;
            }
            part(t);
            {
                std::lock_guard<std::mutex> lk(mu_);
                ++done_;
            }
            done_cv_.notify_one();
        }
    }

    int n_ = 0, next_ = 0, threads_ = 1;
    size_t slot_ = 0;
    uint8_t * buf_ = nullptr;
    std::vector<cudaEvent_t> ev_;
    std::vector<std::thread> team_;
    std::mutex mu_, call_mu_;
    std::condition_variable cv_, done_cv_;
    Job job_;
    uint64_t gen_ = 0;
    int done_ = 0;
    bool stop_ = false;
};

}  // namespace strata::glm
