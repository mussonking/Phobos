#pragma once
// Test helpers shared by the GLM parity programs (host only, header-only):
//   GgufSet  - (strata/glm/gguf.hpp) every shard of a split GGUF, tensor lookup by name.
//   Oracle   - the llama.cpp node dump (index.tsv + <seq>.f32), lookup by (phase, name, occurrence).
//   compare  - max abs / max rel / cosine between a result and the oracle.

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "strata/glm/gguf.hpp"
#include "strata/glm/types.hpp"

namespace strata::glm::test {

using strata::glm::ggml_nbytes;
using strata::glm::HostTensor;
using strata::glm::GgufSet;

struct OracleNode {
    int seq = -1, phase = -1;
    std::string name, op, type;
    int64_t ne[4] = {1, 1, 1, 1};
    std::string file;
    int64_t elems() const { return ne[0] * ne[1] * ne[2] * ne[3]; }
};

class Oracle {
public:
    explicit Oracle(std::string dir) : dir_(std::move(dir)) {
        std::ifstream in(dir_ + "/index.tsv");
        if (!in) throw std::runtime_error("oracle: cannot open " + dir_ + "/index.tsv");
        std::string line;
        while (std::getline(in, line)) {
            std::vector<std::string> c;
            std::stringstream ss(line);
            std::string f;
            while (std::getline(ss, f, '\t')) c.push_back(f);
            if (c.size() < 10) continue;
            OracleNode n;
            n.seq = std::stoi(c[0]);
            n.phase = std::stoi(c[1]);
            n.name = c[2];
            n.op = c[3];
            n.type = c[4];
            for (int i = 0; i < 4; ++i) n.ne[i] = std::stoll(c[5 + i]);
            n.file = c[9];
            nodes_.push_back(std::move(n));
        }
    }
    // The `occurrence`-th node (0-based) named `name` in `phase`.
    const OracleNode & node(int phase, const std::string & name, int occurrence = 0) const {
        for (const auto & n : nodes_)
            if (n.phase == phase && n.name == name && occurrence-- == 0) return n;
        throw std::runtime_error("oracle: no node '" + name + "' in phase " + std::to_string(phase));
    }
    const OracleNode & seq(int s) const {
        for (const auto & n : nodes_)
            if (n.seq == s) return n;
        throw std::runtime_error("oracle: no seq " + std::to_string(s));
    }
    std::vector<float> load(const OracleNode & n) const {
        std::ifstream in(dir_ + "/" + n.file, std::ios::binary);
        if (!in) throw std::runtime_error("oracle: cannot open " + n.file);
        std::vector<float> v((size_t) n.elems());
        in.read(reinterpret_cast<char *>(v.data()), (std::streamsize) (v.size() * sizeof(float)));
        if ((size_t) in.gcount() != v.size() * sizeof(float))
            throw std::runtime_error("oracle: short file " + n.file + " for " + n.name);
        return v;
    }
    std::vector<float> load(int phase, const std::string & name, int occurrence = 0) const {
        return load(node(phase, name, occurrence));
    }
    const std::vector<OracleNode> & nodes() const { return nodes_; }

private:
    std::string dir_;
    std::vector<OracleNode> nodes_;
};

struct Stats {
    double max_abs = 0, max_rel = 0, cosine = 0, ref_rms = 0;
    size_t n = 0;
    // max_rel: |a - b| / max(|b|, rel_floor * ref_rms), so near-zero references do not dominate.
    void print(const char * label) const {
        std::printf("%-34s n=%-9zu max_abs=%.3e max_rel=%.3e cos=%.9f ref_rms=%.3e\n", label, n, max_abs, max_rel,
                    cosine, ref_rms);
    }
};

inline Stats compare(const float * a, const float * ref, size_t n, double rel_floor = 1e-2) {
    Stats s;
    s.n = n;
    double dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < n; ++i) {
        dot += (double) a[i] * ref[i];
        na += (double) a[i] * a[i];
        nb += (double) ref[i] * ref[i];
    }
    s.ref_rms = n ? std::sqrt(nb / (double) n) : 0;
    s.cosine = (na > 0 && nb > 0) ? dot / std::sqrt(na * nb) : (na == nb ? 1.0 : 0.0);
    const double floor = rel_floor * s.ref_rms;
    for (size_t i = 0; i < n; ++i) {
        const double d = std::fabs((double) a[i] - ref[i]);
        if (!(d <= s.max_abs)) s.max_abs = d;   // NaN-propagating
        const double r = d / std::max(std::fabs((double) ref[i]), floor > 0 ? floor : 1e-30);
        if (!(r <= s.max_rel)) s.max_rel = r;
    }
    return s;
}

}  // namespace strata::glm::test
