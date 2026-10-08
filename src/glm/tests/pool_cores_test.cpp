// glm_serve's default --threads (pool.hpp physical_cores): one per physical core over all the NUMA nodes. No GPU.
// The live count is checked against an independent one: the distinct (package, core) pairs of sysfs's topology.
//
//   glm_pool_cores_test        (exit 0 = pass)
#include <cstdio>
#include <filesystem>
#include <set>
#include <string>
#include <thread>
#include <utility>

#include "pool.hpp"

using namespace strata::glm;

static int fails = 0;
static void check(bool ok, const char * what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++fails;
}

int main() {
    check(physical_cores({{1, 2, 3}, {24, 25}}) == 5, "the cores of every node are counted");
    check(physical_cores({}) == (int) std::max(1u, std::thread::hardware_concurrency()),
          "no NUMA topology: the logical CPU count");
    check(parse_cpulist("0-3,8,10-11\n") == std::vector<int>({0, 1, 2, 3, 8, 10, 11}), "cpulist ranges");

    std::set<std::pair<int, int>> cores;
    for (const auto & e : std::filesystem::directory_iterator("/sys/devices/system/cpu")) {
        const std::string name = e.path().filename().string();
        if (name.rfind("cpu", 0) != 0 || name.size() < 4 || name.find_first_not_of("0123456789", 3) != std::string::npos)
            continue;
        const std::string t = e.path().string() + "/topology/";
        const std::string pkg = read_file(t + "physical_package_id"), core = read_file(t + "core_id");
        if (!pkg.empty() && !core.empty()) cores.emplace(std::stoi(pkg), std::stoi(core));
    }
    const int live = physical_cores();
    std::printf("physical_cores() = %d, sysfs (package, core) pairs = %zu, logical CPUs = %u\n", live, cores.size(),
                std::thread::hardware_concurrency());
    if (!numa_cores().empty()) check(live == (int) cores.size(), "live count = the distinct (package, core) pairs");

    std::printf(fails ? "POOL CORES FAIL (%d)\n" : "POOL CORES OK\n", fails);
    return fails ? 1 : 0;
}
