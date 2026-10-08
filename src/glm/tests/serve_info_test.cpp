// glm_serve's INFO line (serve_info.hpp): the keys Strata's web app reads, with Strata's meanings. No GPU.
//
//   glm_serve_info_test        (exit 0 = pass)
#include <cstdio>
#include <map>
#include <sstream>
#include <string>

#include "../serve_info.hpp"

using namespace strata::glm;

static int fails = 0;
static void check(bool ok, const char * what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++fails;
}

static std::map<std::string, std::string> keys(const std::string & line) {
    std::map<std::string, std::string> kv;
    std::istringstream in(line);
    std::string w;
    in >> w;
    while (in >> w) {
        const size_t eq = w.find('=');
        kv[w.substr(0, eq)] = eq == std::string::npos ? "" : w.substr(eq + 1);
    }
    return kv;
}

int main() {
    ServeInfo s;
    s.spec_k = 1;
    s.adapt_every = 16;
    s.sessions = 2;
    s.snapshots = 4;
    s.q8 = true;
    s.r8 = s.vnni = s.down_q8_0 = true;
    s.slots = 1353;
    s.slot_bytes = 7340032;   // 7 MiB
    s.cpu_isa = "vnni";
    const std::string line = serve_info_line(s);
    std::printf("%s\n", line.c_str());
    auto k = keys(line);
    check(line.rfind("INFO ", 0) == 0 && line.find('\n') == std::string::npos, "one INFO line");
    check(k["engine"] == STRATA_VERSION && k["engine"] != "glm_serve", "engine = the version (About: Engine v...)");
    check(k["kv"] == "int8" && k["kv_latent"] == "q8", "kv int8 (Strata's spelling), kv_latent kept");
    check(k["expert_slots"] == "1353" && k["slots"] == "1353", "expert_slots = slots");
    check(k["expert_cache_mib"] == "9471", "expert_cache_mib = slots x blob");
    check(k["spec"] == "2" && k["mtp_max"] == "2" && k["spec_k"] == "1", "spec = verify window (1 draft + 1)");
    check(k["sampling"] == "greedy", "sampling=greedy");
    check(k["arena"] == "r8" && k["kernels"] == "vnni" && k["down"] == "q8_0", "the /health expert layout kept");
    check(k["cpu_isa"] == "vnni", "cpu_isa vnni");

    s.q8 = false;
    s.spec_k = 0;
    s.vnni = false;
    k = keys(serve_info_line(s));
    check(k["kv"] == "fp16" && k["kv_latent"] == "f16", "kv fp16");
    check(k["spec"] == "0" && k["mtp_max"] == "0", "no MTP: spec 0");
    check(k["kernels"] == "f32" && k["down"] == "f32", "f32 kernels");

    s.r8 = false;   // the AVX2 fallback: GGUF arena, f32 miss rows by `kernels`, told apart only by cpu_isa
    s.cpu_isa = "avx2";
    k = keys(serve_info_line(s));
    check(k["arena"] == "gguf" && k["kernels"] == "f32" && k["cpu_isa"] == "avx2", "AVX2 fallback: cpu_isa avx2");
    check(keys(serve_info_line(ServeInfo{}))["cpu_isa"] == "scalar", "cpu_isa default scalar");

    std::printf(fails ? "SERVE INFO FAIL (%d)\n" : "SERVE INFO OK\n", fails);
    return fails ? 1 : 0;
}
