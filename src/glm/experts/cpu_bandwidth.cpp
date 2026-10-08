#include "iq4_cpu.hpp"
#include "iq4_vnni.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <sched.h>
#include <sys/mman.h>
#include <unistd.h>

using namespace strata::glm;
namespace {
using Clock = std::chrono::steady_clock;
constexpr int H = 4096, F = 2048, FixtureT = 4, E = 8, Chunk = 32;
constexpr int XS = 23, NL = 20;

void require(bool ok, const std::string & message) {
    if (!ok) throw std::runtime_error(message);
}
std::string text(const std::string & path) {
    std::ifstream f(path);
    require(bool(f), "cannot read " + path);
    std::ostringstream s; s << f.rdbuf();
    return s.str();
}
void pin(int cpu) {
    cpu_set_t set; CPU_ZERO(&set); CPU_SET(cpu, &set);
    require(sched_setaffinity(0, sizeof(set), &set) == 0, "CPU pin failed");
    cpu_set_t actual; CPU_ZERO(&actual);
    require(sched_getaffinity(0, sizeof(actual), &actual) == 0
            && CPU_COUNT(&actual) == 1 && CPU_ISSET(cpu, &actual), "CPU pin verification failed");
}
std::vector<int> cpu_list(const std::string & list) {
    std::vector<int> cpus;
    std::istringstream s(list); std::string part;
    while (std::getline(s, part, ',')) {
        const auto dash = part.find('-');
        std::size_t used = 0;
        const auto first = part.substr(0, dash);
        const int lo = std::stoi(first, &used);
        require(used == first.size(), "invalid CPU list");
        int hi = lo;
        if (dash != std::string::npos) {
            const auto last = part.substr(dash + 1);
            hi = std::stoi(last, &used);
            require(used == last.size(), "invalid CPU range");
        }
        require(lo >= 0 && hi >= lo && hi < CPU_SETSIZE, "invalid CPU range");
        for (int c = lo; c <= hi; ++c) {
            require(std::find(cpus.begin(), cpus.end(), c) == cpus.end(), "duplicate CPU");
            cpus.push_back(c);
        }
    }
    require(!cpus.empty(), "--cpus is required");
    return cpus;
}
void load_into(const std::string & path, void * dst, std::size_t bytes) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    require(bool(f) && f.tellg() == static_cast<std::streamoff>(bytes), "fixture size/read error: " + path);
    f.seekg(0);
    require(bool(f.read(static_cast<char *>(dst), bytes)), "fixture read failed: " + path);
}

// Guard VMAs prevent merging with unrelated anonymous allocations. The sample
// is populated directly into private anonymous pages, never file-mapped/copy-loaded.
struct Arena {
    void * reservation = MAP_FAILED;
    unsigned char * data = nullptr;
    std::size_t size = 0, reserved = 0, page = 0;
    explicit Arena(std::size_t bytes) {
        page = static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
        size = (bytes + page - 1) / page * page;
        reserved = size + 2 * page;
        reservation = mmap(nullptr, reserved, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        require(reservation != MAP_FAILED, "weight mmap failed");
        data = static_cast<unsigned char *>(reservation) + page;
        if (mprotect(data, size, PROT_READ | PROT_WRITE) || madvise(data, size, MADV_NOHUGEPAGE)) {
            munmap(reservation, reserved); reservation = MAP_FAILED;
            throw std::runtime_error("weight arena protection/NOHUGEPAGE failed");
        }
    }
    ~Arena() { if (reservation != MAP_FAILED) munmap(reservation, reserved); }
    void proof(int node, const char * stage) const {
        const auto start = reinterpret_cast<std::uintptr_t>(data);
        std::istringstream maps(text("/proc/self/maps"));
        std::string line, map_line, numa_line;
        while (std::getline(maps, line)) {
            std::istringstream fields(line); std::string range, perms, offset, dev, inode, name;
            fields >> range >> perms >> offset >> dev >> inode;
            std::getline(fields, name);
            const auto dash = range.find('-');
            const auto lo = std::stoull(range.substr(0, dash), nullptr, 16);
            const auto hi = std::stoull(range.substr(dash + 1), nullptr, 16);
            if (lo == start && hi == start + size) {
                require(perms == "rw-p" && inode == "0" && name.find_first_not_of(' ') == std::string::npos,
                        "weight arena is not anonymous private memory");
                map_line = line;
            }
        }
        require(!map_line.empty(), "exact arena VMA absent from maps");
        std::istringstream numa(text("/proc/self/numa_maps"));
        std::size_t total = 0, local = 0, anon = 0;
        while (std::getline(numa, line)) {
            std::istringstream fields(line); std::string addr, item;
            fields >> addr;
            if (std::stoull(addr, nullptr, 16) != start) continue;
            numa_line = line;
            while (fields >> item) {
                if (item.rfind("anon=", 0) == 0) anon = std::stoull(item.substr(5));
                if (item.size() > 2 && item[0] == 'N' && item.find('=') != std::string::npos) {
                    const auto eq = item.find('=');
                    const auto pages = std::stoull(item.substr(eq + 1));
                    total += pages;
                    if (std::stoi(item.substr(1, eq - 1)) == node) local += pages;
                }
            }
        }
        std::cout << "{\"event\":\"numa_proof\",\"stage\":\"" << stage
                  << "\",\"node\":" << node << ",\"expected_pages\":" << size / page
                  << ",\"allocated_pages\":" << total << ",\"anon_pages\":" << anon
                  << ",\"local_pages\":" << local << ",\"maps\":\"" << map_line
                  << "\",\"numa_maps\":\"" << numa_line << "\"}" << std::endl;
        require(total == size / page && anon == total && local * 100 >= total * 99,
                "NUMA admission rejected: arena must be fully allocated and >=99% local");
    }
};

// Workers survive all iterations; only a chunk counter and phase barrier change.
class Pool {
    std::vector<std::thread> workers;
    std::mutex mutex;
    std::condition_variable wake, done;
    std::function<void(int)> job;
    std::atomic<int> next{0};
    int count = 0, pending = 0, ready = 0;
    unsigned generation = 0;
    bool stopping = false;
    std::exception_ptr error;
    void stop() {
        { std::lock_guard<std::mutex> lock(mutex); stopping = true; }
        wake.notify_all();
        for (auto & w : workers) if (w.joinable()) w.join();
    }
public:
    explicit Pool(const std::vector<int> & cpus) {
        try {
            for (int cpu : cpus) workers.emplace_back([this, cpu] {
                std::unique_lock<std::mutex> lock(mutex);
                try { pin(cpu); } catch (...) { error = std::current_exception(); }
                ++ready; done.notify_one();
                unsigned seen = 0;
                while (true) {
                    wake.wait(lock, [&] { return stopping || generation != seen; });
                    if (stopping) return;
                    seen = generation; lock.unlock();
                    try {
                        for (int i; (i = next.fetch_add(1, std::memory_order_relaxed)) < count;) job(i);
                    } catch (...) {
                        std::lock_guard<std::mutex> guard(mutex); error = std::current_exception();
                    }
                    lock.lock();
                    if (--pending == 0) done.notify_one();
                }
            });
            std::unique_lock<std::mutex> lock(mutex);
            done.wait(lock, [&] { return ready == static_cast<int>(cpus.size()); });
            if (error) std::rethrow_exception(error);
        } catch (...) { stop(); throw; }
    }
    ~Pool() { stop(); }
    void run(int n, std::function<void(int)> fn) {
        std::unique_lock<std::mutex> lock(mutex);
        job = std::move(fn); count = n; next = 0; pending = static_cast<int>(workers.size());
        ++generation; wake.notify_all();
        done.wait(lock, [&] { return pending == 0; });
        if (error) std::rethrow_exception(error);
    }
};

// Match the comm field only, conservatively including prefixed/suffixed worker
// variants. Runner.Worker0 was an ambiguous label for a zero-worker requirement.
bool runner_comm(const std::string & comm) {
    return comm.find("Runner.Worker") != std::string::npos;
}
struct Proc { std::uint64_t start = 0, ticks = 0; };
struct Snapshot { std::map<int, Proc> processes; bool runner = false; };
Snapshot snapshot() {
    Snapshot s;
    for (const auto & entry : std::filesystem::directory_iterator("/proc")) {
        const auto name = entry.path().filename().string();
        if (name.empty() || name.find_first_not_of("0123456789") != std::string::npos) continue;
        const int pid = std::stoi(name);
        if (pid == getpid()) continue;
        const auto root = entry.path().string();
        std::ifstream stat(root + "/stat"); std::string line;
        if (!std::getline(stat, line)) {
            require(!std::filesystem::exists(root), "foreign process stat inaccessible; admission unqualified");
            continue;
        }
        const auto close = line.rfind(')');
        const auto open = line.find('(');
        require(open != std::string::npos && close > open, "invalid process stat");
        if (runner_comm(line.substr(open + 1, close - open - 1))) s.runner = true;
        std::istringstream fields(line.substr(close + 2));
        std::string item; Proc p;
        for (int field = 3; field <= 22; ++field) {
            require(bool(fields >> item), "incomplete process stat");
            if (field == 14 || field == 15) p.ticks += std::stoull(item);
            if (field == 22) p.start = std::stoull(item);
        }
        s.processes.emplace(pid, p);
        // Check thread comm, never a command-line substring (including ours).
        std::error_code ec;
        std::filesystem::directory_iterator tasks(root + "/task", ec);
        require(!ec || !std::filesystem::exists(root), "foreign task list inaccessible; admission unqualified");
        if (!ec) for (const auto & task : tasks) {
            std::ifstream comm(task.path().string() + "/comm");
            if (std::getline(comm, item)) {
                if (runner_comm(item)) s.runner = true;
            } else require(!std::filesystem::exists(task.path()), "foreign thread comm inaccessible; admission unqualified");
        }
    }
    return s;
}
double foreign_seconds(const Snapshot & before, const Snapshot & after) {
    std::uint64_t ticks = 0;
    for (const auto & entry : after.processes) {
        const auto old = before.processes.find(entry.first);
        if (old != before.processes.end() && old->second.start == entry.second.start) {
            if (entry.second.ticks >= old->second.ticks) ticks += entry.second.ticks - old->second.ticks;
        } else {
            // New processes are counted in full; vanished processes are not
            // recoverable from /proc. This limitation is explicit in the JSON.
            ticks += entry.second.ticks;
        }
    }
    return double(ticks) / sysconf(_SC_CLK_TCK);
}
void admission_self_test() {
    for (const std::string comm : {"Runner.Worker", "Runner.Worker0", "Runner.Worker1", "Runner.Worker-2", "xRunner.Worker"})
        require(runner_comm(comm), "runner comm false negative: " + comm);
    for (const std::string comm : {"glm_cpu_bandwid", "bash", "Runner.Workflow", "Worker", ""})
        require(!runner_comm(comm), "runner comm false positive: " + comm);
    Snapshot a, b;
    a.processes.emplace(10, Proc{100, 1000});
    a.processes.emplace(11, Proc{101, 400});
    a.processes.emplace(12, Proc{102, 900});
    b.processes.emplace(10, Proc{100, 1040}); // Existing: +40 ticks.
    b.processes.emplace(11, Proc{201, 20});   // Reused PID: count new process.
    b.processes.emplace(13, Proc{103, 30});   // New process: +30 ticks.
    const double tick = 1.0 / sysconf(_SC_CLK_TCK);
    require(std::abs(foreign_seconds(a, b) - 90 * tick) < 1e-12, "foreign CPU delta test failed");
    b.processes.at(10).ticks = 999;           // Counter regression: never underflow.
    require(std::abs(foreign_seconds(a, b) - 50 * tick) < 1e-12, "foreign CPU regression test failed");
    std::cout << "{\"event\":\"admission_self_test\",\"passed\":true}" << std::endl;
}

class Admission {
    std::thread worker;
    std::atomic<bool> stopping{false};
    std::atomic<bool> rejected{false};
    std::mutex mutex;
    std::string reason;
    Snapshot previous;
    Clock::time_point previous_time, deadline;
    double cpu_seconds = 0, max_cores = 0, elapsed = 0;
    int samples = 0;
    bool final_reported = false;
    bool validation_only = false;
    void reject(const std::string & why) {
        { std::lock_guard<std::mutex> lock(mutex); reason = why; }
        rejected = true;
    }
    void sample() {
        const auto current = snapshot(); const auto now = Clock::now();
        const auto dt = std::chrono::duration<double>(now - previous_time).count();
        const double sec = foreign_seconds(previous, current), cores = sec / dt;
        cpu_seconds += sec; elapsed += dt; max_cores = std::max(max_cores, cores); ++samples;
        if (current.runner || previous.runner) reject("Runner.Worker comm variant appeared");
        if (cores > 0.5) reject("foreign CPU exceeds 0.5 core");
        if (now >= deadline) reject("50 second run deadline exceeded");
        previous = current; previous_time = now;
    }
public:
    explicit Admission(bool validate_only = false) : validation_only(validate_only) {
        deadline = Clock::now() + std::chrono::seconds(50);
        if (validation_only) { report("before"); return; }
        previous = snapshot(); previous_time = Clock::now();
        require(!previous.runner, "admission rejected: Runner.Worker comm variant present");
        std::this_thread::sleep_for(std::chrono::milliseconds(600));
        sample(); report("before"); check();
        worker = std::thread([this] {
            try {
                while (!stopping) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(250));
                    sample();
                }
            } catch (const std::exception & e) { reject(e.what()); }
        });
    }
    ~Admission() { finish(); if (!final_reported) report("after"); }
    void finish() { stopping = true; if (worker.joinable()) worker.join(); }
    void check() {
        if (validation_only) {
            require(Clock::now() < deadline, "50 second validation-only deadline exceeded");
            return;
        }
        if (rejected) {
            std::lock_guard<std::mutex> lock(mutex);
            throw std::runtime_error("load admission rejected: " + reason);
        }
    }
    void report(const char * stage) {
        if (std::string(stage) == "after") final_reported = true;
        if (validation_only) {
            std::cout << "{\"event\":\"load_admission_skipped\",\"stage\":\"" << stage
                      << "\",\"reason\":\"explicit_validate_only\",\"performance_measurement\":false"
                      << ",\"externally_qualified\":false}" << std::endl;
            return;
        }
        std::cout << "{\"event\":\"load_proof\",\"stage\":\"" << stage
                  << "\",\"admitted\":" << (rejected ? "false" : "true")
                  << ",\"foreign_cpu_seconds\":" << cpu_seconds << ",\"wall_seconds\":" << elapsed
                  << ",\"max_foreign_cores\":" << max_cores << ",\"samples\":" << samples
                  << ",\"threshold_cores\":0.5,\"sampling_ms\":250,\"self_excluded\":true"
                  << ",\"limitations\":\"sampled proc stat: exited-between-samples CPU and transient runner may escape; no external monitor validation\"}"
                  << std::endl;
    }
};

struct Model {
    const unsigned char * weights;
    const int T; // Runtime token count; fixtures always contain FixtureT rows.
    std::size_t gs = iq4_cpu_row_bytes(XS, H), ds = iq4_cpu_row_bytes(NL, F);
    std::size_t blob = 2 * F * gs + H * ds;
    std::vector<float> x, mid, out, qx, qm;
    bool native;
    Model(const unsigned char * w, const std::string & dir, const std::string & path, int tokens)
        : weights(w), T(tokens), x(T * H), mid(E * T * F), out(E * T * H), native(path == "vnni-native") {
        std::vector<float> captured_x(FixtureT * H);
        load_into(dir + "/x.f32", captured_x.data(), captured_x.size() * sizeof(float));
        std::copy_n(captured_x.data(), x.size(), x.data());
        for (float v : x) require(std::isfinite(v), "nonfinite fixture input");
        if (path != "f32") {
            qx.resize((T * iq4_vnni_q8_bytes(H) + 3) / 4);
            qm.resize((E * T * mid_stride() + 3) / 4);
            for (int t = 0; t < T; ++t)
                iq4_vnni_quantize(x.data() + t * H, H, reinterpret_cast<unsigned char *>(qx.data()) + t * iq4_vnni_q8_bytes(H));
        }
    }
    std::size_t mid_stride() const { return native ? iq4_vnni_q8_0_bytes(F) : iq4_vnni_q8_bytes(F); }
    void iteration(Pool & pool, bool f32, Admission & admission) {
        pool.run(E * F / Chunk, [&](int job) {
            admission.check();
            const int e = job / (F / Chunk), row = (job % (F / Chunk)) * Chunk;
            const auto * gate = weights + e * blob + row * gs;
            const auto * up = gate + F * gs;
            if (f32) {
                const float * input[FixtureT]; for (int t = 0; t < T; ++t) input[t] = x.data() + t * H;
                float a[FixtureT], b[FixtureT];
                for (int r = 0; r < Chunk; ++r) {
                    iq4_cpu_row(XS, gate + r * gs, H, T, input, a, CPUExpertPath::avx512);
                    iq4_cpu_row(XS, up + r * gs, H, T, input, b, CPUExpertPath::avx512);
                    for (int t = 0; t < T; ++t) mid[(e * T + t) * F + row + r] = expert_swiglu_clamp(a[t], b[t]);
                }
            } else {
                float a[Chunk], b[Chunk];
                // Selected tokens consume a hot 32-row chunk before advancing.
                // VNNI's native API is single-token; logical sample bytes are
                // counted once, NOT these repeated cache accesses or DRAM reads.
                for (int t = 0; t < T; ++t) {
                    const auto * act = reinterpret_cast<const unsigned char *>(qx.data()) + t * iq4_vnni_q8_bytes(H);
                    iq4_vnni_rows(XS, gate, H, act, Chunk, a, IQ4VNNIPath::vnni);
                    iq4_vnni_rows(XS, up, H, act, Chunk, b, IQ4VNNIPath::vnni);
                    for (int r = 0; r < Chunk; ++r) mid[(e * T + t) * F + row + r] = expert_swiglu_clamp(a[r], b[r]);
                }
            }
        });
        if (!f32) pool.run(E, [&](int e) {
            admission.check();
            for (int t = 0; t < T; ++t) {
                auto * dest = reinterpret_cast<unsigned char *>(qm.data()) + (e * T + t) * mid_stride();
                const auto * source = mid.data() + (e * T + t) * F;
                if (native) iq4_vnni_quantize_q8_0(source, F, dest);
                else iq4_vnni_quantize(source, F, dest);
            }
        });
        pool.run(E * H / Chunk, [&](int job) {
            admission.check();
            const int e = job / (H / Chunk), row = (job % (H / Chunk)) * Chunk;
            const auto * down = weights + e * blob + 2 * F * gs + row * ds;
            if (f32) {
                const float * input[FixtureT]; for (int t = 0; t < T; ++t) input[t] = mid.data() + (e * T + t) * F;
                float values[FixtureT];
                for (int r = 0; r < Chunk; ++r) {
                    iq4_cpu_row(NL, down + r * ds, F, T, input, values, CPUExpertPath::avx512);
                    for (int t = 0; t < T; ++t) out[(e * T + t) * H + row + r] = values[t];
                }
            } else for (int t = 0; t < T; ++t) {
                const auto * act = reinterpret_cast<const unsigned char *>(qm.data()) + (e * T + t) * mid_stride();
                auto * dest = out.data() + (e * T + t) * H + row;
                if (native) iq4_vnni_nl_q8_0_rows(down, F, act, Chunk, dest, IQ4VNNIPath::vnni);
                else iq4_vnni_rows(NL, down, F, act, Chunk, dest, IQ4VNNIPath::vnni);
            }
        });
    }
    void validate(Pool & pool, bool f32, const std::string & dir, Admission & admission) const {
        std::vector<float> expected(out.size());
        const char * basis = f32 ? "captured_output.f32" : "independent_scalar_quantized_model";
        const double atol = f32 ? 0.002 : 0.003;
        const double rtol = f32 ? 0.0005 : 0.001;
        if (f32) {
            std::vector<float> captured(E * FixtureT * H);
            load_into(dir + "/output.f32", captured.data(), captured.size() * sizeof(float));
            for (int e = 0; e < E; ++e)
                std::copy_n(captured.data() + e * FixtureT * H, T * H, expected.data() + e * T * H);
        } else pool.run(E, [&](int e) {
            // Independent complete expert evaluation: no optimized row batches,
            // no benchmark mid/qm reuse, and a separate scalar SwiGLU formula.
            std::vector<float> input_q((iq4_vnni_q8_bytes(H) + 3) / 4);
            std::vector<float> reference_mid(F), reference_q((mid_stride() + 3) / 4);
            const auto * gate = weights + e * blob;
            const auto * up = gate + F * gs;
            const auto * down = gate + 2 * F * gs;
            for (int t = 0; t < T; ++t) {
                admission.check();
                iq4_vnni_quantize(x.data() + t * H, H, input_q.data());
                for (int r = 0; r < F; ++r) {
                    if (r % Chunk == 0) admission.check();
                    const float a = iq4_vnni_row(XS, gate + r * gs, H, input_q.data(), IQ4VNNIPath::scalar);
                    const float b = iq4_vnni_row(XS, up + r * gs, H, input_q.data(), IQ4VNNIPath::scalar);
                    const float gc = std::min(a, 10.0f), uc = std::max(-10.0f, std::min(b, 10.0f));
                    reference_mid[r] = gc / (1.0f + std::exp(-gc)) * uc;
                }
                if (native) iq4_vnni_quantize_q8_0(reference_mid.data(), F, reference_q.data());
                else iq4_vnni_quantize(reference_mid.data(), F, reference_q.data());
                for (int r = 0; r < H; ++r) {
                    if (r % Chunk == 0) admission.check();
                    auto * dest = expected.data() + (e * T + t) * H + r;
                    if (native) iq4_vnni_nl_q8_0_rows(down + r * ds, F, reference_q.data(), 1, dest, IQ4VNNIPath::scalar);
                    else *dest = iq4_vnni_row(NL, down + r * ds, F, reference_q.data(), IQ4VNNIPath::scalar);
                }
            }
        });
        double max_abs = 0, max_rel = 0, dot = 0, aa = 0, bb = 0;
        bool pass = true;
        for (std::size_t i = 0; i < out.size(); ++i) {
            const double a = out[i], b = expected[i], error = std::abs(a - b);
            require(std::isfinite(a) && std::isfinite(b), "nonfinite same-weight numerical reference/output");
            pass &= error <= atol + rtol * std::abs(b);
            max_abs = std::max(max_abs, error);
            max_rel = std::max(max_rel, error / std::max(1e-6, std::abs(b)));
            dot += a * b; aa += a * a; bb += b * b;
        }
        const double cosine = aa && bb ? dot / std::sqrt(aa * bb) : (aa == bb ? 1 : 0);
        const double cosine_floor = f32 ? 0.999999 : 0.99999;
        pass &= cosine >= cosine_floor;
        std::cout << "{\"event\":\"correctness_proof\",\"basis\":\"" << basis
                  << "\",\"tokens\":" << T << ",\"fixture_tokens\":" << FixtureT
                  << ",\"same_weights\":true,\"atol\":" << atol << ",\"rtol\":" << rtol
                  << ",\"max_abs\":" << max_abs << ",\"max_rel\":" << max_rel
                  << ",\"cosine\":" << cosine << ",\"cosine_floor\":" << cosine_floor
                  << ",\"passed\":" << (pass ? "true" : "false") << "}" << std::endl;
        require(pass, "same-weight numerical validation failed before timing");
        admission.check();
    }
    double checksum() const {
        double sum = 0;
        for (std::size_t i = 0; i < out.size(); ++i) {
            require(std::isfinite(out[i]), "nonfinite benchmark output");
            sum += double(out[i]) * (1 + i % 17);
        }
        require(std::isfinite(sum), "nonfinite checksum"); return sum;
    }
};
} // namespace

int main(int argc, char ** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--self-test-admission") {
            admission_self_test(); return 0;
        }
        std::string dir, list, path = "vnni-native";
        int node = -1, repeats = 8, warmup = 2, tokens = 1;
        bool validate_only = false;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--validate-only") { validate_only = true; continue; }
            require(i + 1 < argc, "missing option value");
            const std::string value = argv[++i];
            if (arg == "--fixture-dir") dir = value;
            else if (arg == "--cpus") list = value;
            else if (arg == "--node") node = std::stoi(value);
            else if (arg == "--path") path = value;
            else if (arg == "--tokens") tokens = std::stoi(value);
            else if (arg == "--repeats") repeats = std::stoi(value);
            else if (arg == "--warmup") warmup = std::stoi(value);
            else throw std::runtime_error("unknown option " + arg);
        }
        require(!dir.empty() && node >= 0 && repeats >= 1 && repeats <= 40 && warmup >= 1 && warmup <= 4
                && (tokens == 1 || tokens == FixtureT),
                "usage: glm_cpu_bandwidth --fixture-dir DIR --node N --cpus LIST --path f32|vnni-native|vnni-q8k [--tokens 1|4] [--repeats 8 --warmup 2] [--validate-only]");
        require(path == "f32" || path == "vnni-native" || path == "vnni-q8k", "invalid path");
        const auto cpus = cpu_list(list);
        for (int c : cpus) require(std::filesystem::exists("/sys/devices/system/node/node" + std::to_string(node) + "/cpu" + std::to_string(c)),
                                  "selected CPU is not on requested node");
        require(path == "f32" ? iq4_cpu_has_avx512() : iq4_vnni_available(), "required ISA unavailable; scalar fallback forbidden");
        std::cout << std::setprecision(12);
        pin(cpus.front()); // Must precede arena allocation and every weight first touch.
        Admission admission(validate_only);
        const auto bytes = E * (2 * F * iq4_cpu_row_bytes(XS, H) + H * iq4_cpu_row_bytes(NL, F));
        Arena arena(bytes);
        load_into(dir + "/experts.bin", arena.data, bytes);
        arena.proof(node, "loaded_before_warmup"); admission.check();
        Model model(arena.data, dir, path, tokens);
        Pool pool(cpus);
        const bool f32 = path == "f32";
        for (int i = 0; i < warmup; ++i) { admission.check(); model.iteration(pool, f32, admission); }
        const auto reference = model.out;
        const double checksum = model.checksum();
        model.validate(pool, f32, dir, admission);
        if (validate_only) {
            // Explicit correctness-only rerun: no timer or bandwidth calculation.
            model.iteration(pool, f32, admission);
            require(model.out == reference && model.checksum() == checksum, "validation-only rerun is not deterministic");
            arena.proof(node, "after_validation_only");
            admission.finish(); admission.report("after"); admission.check();
            std::cout << "{\"event\":\"validation_result\",\"path\":\"" << path
                      << "\",\"node\":" << node << ",\"cpus\":\"" << list << "\",\"threads\":" << cpus.size()
                      << ",\"experts\":8,\"tokens\":" << tokens << ",\"fixture_tokens\":" << FixtureT
                      << ",\"row_chunk\":32,\"checksum\":" << checksum
                      << ",\"finite\":true,\"deterministic\":true,\"same_weight_correctness_validated\":true"
                      << ",\"numa_validated\":true,\"load_admission_skipped\":true"
                      << ",\"performance_measurement\":false,\"externally_qualified\":false,\"no_throughput_claim\":true}"
                      << std::endl;
            return 0;
        }
        arena.proof(node, "before_first_measured_iteration"); admission.check();
        std::vector<double> times;
        for (int i = 0; i < repeats; ++i) {
            admission.check();
            const auto start = Clock::now(); model.iteration(pool, f32, admission);
            times.push_back(std::chrono::duration<double, std::milli>(Clock::now() - start).count());
            require(model.out == reference, "output changed across identical iterations");
            require(model.checksum() == checksum, "checksum changed across iterations");
        }
        arena.proof(node, "after_measurement");
        admission.finish(); admission.report("after"); admission.check();
        std::sort(times.begin(), times.end());
        const double median = (times[(times.size() - 1) / 2] + times[times.size() / 2]) / 2;
        std::cout << "{\"event\":\"result\",\"path\":\"" << path << "\",\"node\":" << node
                  << ",\"cpus\":\"" << list << "\",\"threads\":" << cpus.size()
                  << ",\"experts\":8,\"tokens\":" << tokens << ",\"fixture_tokens\":" << FixtureT
                  << ",\"hidden\":4096,\"ff\":2048,\"row_chunk\":32"
                  << ",\"repeats\":" << repeats << ",\"warmup\":" << warmup << ",\"bytes\":" << bytes
                  << ",\"median_ms\":" << median << ",\"min_ms\":" << times.front() << ",\"max_ms\":" << times.back()
                  << ",\"effective_weight_GBps\":" << double(bytes) / 1e6 / median
                  << ",\"checksum\":" << checksum << ",\"finite\":true,\"deterministic\":true"
                  << ",\"same_weight_correctness_validated\":true"
                  << ",\"numa_validated\":true,\"sampled_load_admitted\":true,\"externally_qualified\":false"
                  << ",\"byte_basis\":\"unique weights per eight-expert iteration; VNNI visits each hot chunk once per selected token\""
                  << ",\"dram_counter_measured\":false,\"no_throughput_claim\":true}" << std::endl;
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "UNQUALIFIED: " << e.what() << '\n'; return 1;
    }
}
