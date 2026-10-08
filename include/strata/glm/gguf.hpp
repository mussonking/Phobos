#pragma once
// A split GGUF (all shards mmapped) for the GLM engine and its tests: tensor lookup by name, metadata access.

#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "strata/artifact/gguf_reader.hpp"
#include "strata/glm/types.hpp"

namespace strata::glm {

inline uint64_t ggml_nbytes(uint32_t type, const std::vector<uint64_t> & shape) {
    uint64_t n = 1;
    for (auto d : shape) n *= d;
    switch (type) {
        case 0: return n * 4;           // F32
        case 1: return n * 2;           // F16
        case 30: return n * 2;          // BF16
        case 8: return n / 32 * 34;     // Q8_0
        case 12: return n / 256 * 144;  // Q4_K
        case 13: return n / 256 * 176;  // Q5_K
        case 14: return n / 256 * 210;  // Q6_K
        case 20: return n / 32 * 18;    // IQ4_NL
        case 23: return n / 256 * 136;  // IQ4_XS
        default: throw std::runtime_error("ggml_nbytes: unhandled type " + std::to_string(type));
    }
}

struct HostTensor {
    const uint8_t * data = nullptr;   // mmapped GGUF bytes
    uint32_t type = 0;
    std::vector<uint64_t> shape;
    uint64_t nbytes = 0;
    // A WeightRef with this shape/type pointing at `device_data`.
    WeightRef ref(const void * device_data) const {
        WeightRef w;
        w.data = device_data;
        w.type = (int) type;
        for (size_t i = 0; i < shape.size() && i < 4; ++i) w.ne[i] = (int64_t) shape[i];
        w.nbytes = nbytes;
        return w;
    }
};

class GgufSet {
public:
    // `first` is shard 1 (".../X-00001-of-0000N.gguf") or a single-file GGUF.
    explicit GgufSet(const std::string & first) {
        const auto p = first.rfind("-00001-of-");
        if (p == std::string::npos) {
            files_.push_back(std::make_unique<strata::GgufFile>(first));
            paths_.push_back(first);
            return;
        }
        const int n = std::stoi(first.substr(p + 10, 5));
        for (int i = 1; i <= n; ++i) {
            char idx[16];
            std::snprintf(idx, sizeof idx, "%05d", i);
            std::string f = first;
            f.replace(p + 1, 5, idx);
            files_.push_back(std::make_unique<strata::GgufFile>(f));
            paths_.push_back(f);
        }
    }
    const strata::GgufFile & shard(size_t i) const { return *files_[i]; }
    size_t shards() const { return files_.size(); }
    bool has(const std::string & name) const {
        for (const auto & f : files_)
            if (f->find(name)) return true;
        return false;
    }
    // Metadata lives in shard 1.
    const MetaValue * meta(const std::string & key) const { return files_[0]->get(key); }
    HostTensor tensor(const std::string & name) const {
        for (const auto & f : files_)
            if (const auto * t = f->find(name)) {
                HostTensor h;
                h.data = f->tensor_data(*t);
                h.type = t->type;
                h.shape = t->shape;
                h.nbytes = ggml_nbytes(t->type, t->shape);
                return h;
            }
        throw std::runtime_error("GGUF: no tensor " + name);
    }
    // The shard path and file offset of a byte of mapped tensor data (for direct reads); false if not mapped here.
    bool locate(const void * p, std::string * path, uint64_t * offset) const {
        const uint8_t * q = static_cast<const uint8_t *>(p);
        for (size_t i = 0; i < files_.size(); ++i) {
            const auto & ts = files_[i]->tensors();
            if (ts.empty()) continue;
            const uint8_t * base = files_[i]->tensor_data(ts[0]) - files_[i]->data_start() - ts[0].offset;
            if (q >= base && q < base + files_[i]->file_size()) {
                *path = paths_[i];
                *offset = (uint64_t) (q - base);
                return true;
            }
        }
        return false;
    }

private:
    std::vector<std::unique_ptr<strata::GgufFile>> files_;
    std::vector<std::string> paths_;
};

}  // namespace strata::glm
