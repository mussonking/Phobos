#pragma once
// Context state file (two-GPU tests): a context prefilled once on one GPU (glm_prefill_test GLM_STATE_DUMP) is
// loaded into the attention state of each layer-split stage, on its own device. Layout: magic, n_layer, position,
// pending token, then per trunk block its kind (0 KDA, 1 MLA), byte count and bytes: kda_state_save, or
// mla_cache_export_rows [0, pos) followed by mla_cache_export_tail (the prefix cache's own blobs).

#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "strata/glm/geometry.hpp"
#include "strata/glm/kda.hpp"
#include "strata/glm/mla.hpp"

namespace strata::glm {

constexpr uint64_t kStateFileMagic = 0x31545354534d4c47ull;   // "GLMSTST1"

// kda / mla: one entry per trunk block (the block's own kind non-null). Synchronizes st.
inline void state_file_write(const std::string & path, const Geometry & g, int64_t pos, int32_t next,
                             const std::vector<KdaState *> & kda, const std::vector<MlaCache *> & mla,
                             cudaStream_t st) {
    FILE * f = std::fopen(path.c_str(), "wb");
    if (!f) throw std::runtime_error("state file: cannot create " + path);
    auto put = [&](const void * p, size_t n) {
        if (std::fwrite(p, 1, n, f) != n) {
            std::fclose(f);
            throw std::runtime_error("state file: write failed " + path);
        }
    };
    const int64_t head[4] = {(int64_t) kStateFileMagic, g.n_layer, pos, next};
    put(head, sizeof head);
    std::vector<uint8_t> buf;
    for (int il = 0; il < g.n_layer; ++il) {
        int64_t kind = 0;
        if (kda[il]) {
            buf.resize(kda_state_bytes(g));
            kda_state_save(kda[il], buf.data(), st);
        } else if (mla[il]) {
            kind = 1;
            const size_t rb = mla_cache_rows_bytes(mla[il], 0, pos), tb = mla_cache_tail_bytes(mla[il]);
            buf.assign(rb + tb, 0);
            mla_cache_export_rows(mla[il], 0, pos, buf.data(), st);
            mla_cache_export_tail(mla[il], pos, buf.data() + rb, st);
        } else {
            std::fclose(f);
            throw std::runtime_error("state file: block " + std::to_string(il) + " has no attention state");
        }
        if (cudaStreamSynchronize(st) != cudaSuccess) {
            std::fclose(f);
            throw std::runtime_error("state file: export failed");
        }
        const int64_t rec[2] = {kind, (int64_t) buf.size()};
        put(rec, sizeof rec);
        put(buf.data(), buf.size());
    }
    std::fclose(f);
}

// Loads the blocks whose state is present in kda / mla (null entries: another stage's blocks, skipped); st on
// their device. Returns (position, pending token). Synchronizes st.
inline std::pair<int64_t, int32_t> state_file_read(const std::string & path, const Geometry & g,
                                                   const std::vector<KdaState *> & kda,
                                                   const std::vector<MlaCache *> & mla, cudaStream_t st) {
    FILE * f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("state file: cannot open " + path);
    auto fail = [&](const std::string & why) {
        std::fclose(f);
        throw std::runtime_error("state file " + path + ": " + why);
    };
    auto get = [&](void * p, size_t n) {
        if (std::fread(p, 1, n, f) != n) fail("truncated");
    };
    int64_t head[4];
    get(head, sizeof head);
    if ((uint64_t) head[0] != kStateFileMagic) fail("bad magic");
    if (head[1] != g.n_layer) fail("written for another model");
    const int64_t pos = head[2];
    std::vector<uint8_t> buf;
    for (int il = 0; il < g.n_layer; ++il) {
        int64_t rec[2];
        get(rec, sizeof rec);
        const bool mine = rec[0] == 1 ? mla[il] != nullptr : kda[il] != nullptr;
        if (rec[0] == 1 && kda[il]) fail("block " + std::to_string(il) + " kind mismatch");
        if (rec[0] == 0 && mla[il]) fail("block " + std::to_string(il) + " kind mismatch");
        if (!mine) {
            if (std::fseek(f, (long) rec[1], SEEK_CUR) != 0) fail("seek");
            continue;
        }
        buf.resize((size_t) rec[1]);
        get(buf.data(), buf.size());
        if (rec[0] == 0) {
            if (buf.size() != kda_state_bytes(g)) fail("KDA state size");
            kda_state_load(kda[il], buf.data(), st);
        } else {
            const size_t rb = mla_cache_rows_bytes(mla[il], 0, pos);
            if (buf.size() != rb + mla_cache_tail_bytes(mla[il])) fail("MLA rows size (latent type or capacity)");
            mla_cache_truncate(mla[il], 0);
            if (!mla_cache_import_rows(mla[il], 0, pos, buf.data(), st)) fail("MLA rows import");
            if (!mla_cache_import_tail(mla[il], pos, buf.data() + rb, st)) fail("MLA tail import");
        }
        if (cudaStreamSynchronize(st) != cudaSuccess) fail("import failed");
    }
    std::fclose(f);
    return {pos, (int32_t) head[3]};
}

}  // namespace strata::glm
