// src/glm/engine/session.cpp - see session.hpp.
#include "session.hpp"

#include <nvtx3/nvToolsExt.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>

#include "decode.hpp"
#include "prefill_prompt.hpp"
#include "strata/glm/kda.hpp"
#include "strata/glm/mla.hpp"
#include "strata/glm/mtp.hpp"

namespace strata::glm {
namespace {

// NVTX range for nsys (glm_decode --profile-span): popped on every exit of its scope
struct NvtxRange {
    explicit NvtxRange(const char * n) { nvtxRangePushA(n); }
    ~NvtxRange() { nvtxRangePop(); }
    NvtxRange(const NvtxRange &) = delete;
    NvtxRange & operator=(const NvtxRange &) = delete;
};

void ck(cudaError_t e, const char * what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("glm session: ") + what + ": " + cudaGetErrorString(e));
}
double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
constexpr int kCatchUp = 8;   // MTP catch-up rows per mtp_draft call (and the session's workspace / routed max_T)

}  // namespace

struct Session {
    const Model * m = nullptr;
    SessionOptions opt;
    SessionExperts ex;
    cudaStream_t st = nullptr;
    std::unique_ptr<Workspace> ws;
    std::vector<KdaState *> kda;
    std::vector<MlaCache *> mla;
    KdaState * k0 = nullptr;
    void * scratch = nullptr;
    int64_t * d_pos0 = nullptr;
    int64_t * h_pos0 = nullptr;
    ForwardHooks gh, eh;   // graph hooks, eager hooks (MTP)
    std::unique_ptr<SegmentedGraph> g1[2], gv[2];   // T = 1 and T = k + 1, per KDA parity
    MtpState * mtp = nullptr;
    float * hcat = nullptr;            // device [kCatchUp + 1][n_embd]: h_prev rows for mtp_draft
    std::vector<int32_t> pend;         // prompt rows not yet caught up in the MTP cache
    int64_t pend_pos0 = 0;
    int64_t pos = 0;                   // tokens in the trunk caches
    std::vector<int32_t> toks;         // the fed tokens (pos of them)
    int32_t next = -1;                 // trunk's greedy token after the last fed one (not yet fed)
    std::vector<int32_t> drafts;       // drafts for the next round (empty: draft from next first)
    // Between calls: pend is empty and hcat row 0 = h[pos - 1] (zeros at pos 0), the MTP cache holds rows [0, pos]
    // (row pos = (next, h[pos - 1])) once next is drafted from.
    SessionStats stats;
    std::vector<float> margins, top1, row;   // record_margins
    // prefix cache: rows [0, chain->b) of every MLA cache are bitwise the chain's blobs (expired: none; the cache's
    // entries own the segments); a save in flight reads the caches until ev_save (fence() makes the stream wait
    // before the state changes)
    std::weak_ptr<PrefixSegment> chain;
    cudaEvent_t ev_save = nullptr, ev_ready = nullptr;
    bool save_pending = false;

    void fence() {
        if (!save_pending) return;
        ck(cudaStreamWaitEvent(st, ev_save, 0), "save fence");
        save_pending = false;
    }
    void trim_chain(int64_t n) {
        std::shared_ptr<PrefixSegment> c = chain.lock();
        while (c && c->b > n) c = c->parent;
        chain = c;
    }
    // the chain, if it lives in pc
    std::shared_ptr<PrefixSegment> chain_in(const PrefixCache & pc) const {
        std::shared_ptr<PrefixSegment> c = chain.lock();
        return c && c->pc == &pc ? c : nullptr;
    }
    // every MLA cache in the prefix cache's order: trunk layers, then the MTP block's
    std::vector<MlaCache *> caches() const {
        std::vector<MlaCache *> v;
        for (auto * c : mla)
            if (c) v.push_back(c);
        if (mtp) v.push_back(mtp_cache(mtp));
        return v;
    }

    // top-1 - top-2 logit gap of head rows [0, n) (synchronous; tests only)
    void record(int n) {
        const size_t V = (size_t) m->g.n_vocab;
        row.resize(V);
        for (int r = 0; r < n; ++r) {
            ck(cudaMemcpyAsync(row.data(), ws->logits + r * V, V * 4, cudaMemcpyDeviceToHost, st), "logits");
            ck(cudaStreamSynchronize(st), "logits");
            float a = -INFINITY, b = -INFINITY;
            for (float v : row) {
                if (v > a) { b = a; a = v; }
                else if (v > b) b = v;
            }
            margins.push_back(a - b);
            top1.push_back(a);
        }
    }

    void replay(SegmentedGraph & sg, const int32_t * ids, int T, int64_t pos0) {
        embed_upload(*m, *ws, ids, T, st);
        *h_pos0 = pos0;
        ck(cudaMemcpyAsync(d_pos0, h_pos0, sizeof(int64_t), cudaMemcpyHostToDevice, st), "pos0");
        replay_decode(sg, *ws, gh, st);
        for (auto * k : kda)
            if (k) kda_state_advance(k, T);
    }
    void flush_catch_up() {
        if (!mtp || pend.empty()) return;
        const int n = (int) pend.size();
        mtp_draft(mtp, *m, *ws, eh, pend.data(), hcat, n, pend_pos0, 0, nullptr, st);
        // row n (h of the last caught-up position) becomes row 0 of the next chunk
        const size_t D = (size_t) m->g.n_embd;
        ck(cudaMemcpyAsync(hcat, hcat + n * D, D * 4, cudaMemcpyDeviceToDevice, st), "hcat shift");
        pend_pos0 += n;
        pend.clear();
    }
    // first drafts of a round: MTP row pos = (next, h[pos - 1]) (hcat row 0)
    void draft_pending() {
        const int k = opt.spec_k;
        drafts.resize(k);
        const double d0 = now_ms();
        mtp_draft(mtp, *m, *ws, eh, &next, hcat, 1, pos, k, drafts.data(), st);
        stats.draft_ms += now_ms() - d0;
    }
};

struct SessionSnapshot {
    int64_t pos = 0;
    int32_t next = -1;
    std::vector<int32_t> toks;
    void * host = nullptr;   // pinned: [KDA layers][kda_state_bytes], then hcat row 0 [n_embd] F32
    size_t bytes = 0;
};

void session_destroy(Session * s) {
    if (!s) return;
    if (s->save_pending) cudaEventSynchronize(s->ev_save);
    for (auto e : {s->ev_save, s->ev_ready})
        if (e) cudaEventDestroy(e);
    if (s->mtp) mtp_destroy(s->mtp);
    for (auto * k : s->kda) if (k) kda_state_destroy(k);
    for (auto * c : s->mla) if (c) mla_cache_destroy(c);
    for (void * p : {s->scratch, (void *) s->d_pos0, (void *) s->hcat})
        if (p) cudaFree(p);
    if (s->h_pos0) cudaFreeHost(s->h_pos0);
    delete s;
}

SessionPtr session_create(const Model & m, const SessionOptions & opt, const SessionExperts & ex, cudaStream_t st) {
    const Geometry & g = m.g;
    if (opt.spec_k < 0 || opt.spec_k + 1 > kKdaLogT || opt.spec_k + 1 > kMaxHeadRows)
        throw std::runtime_error("session: spec_k + 1 must be <= " + std::to_string(std::min(kKdaLogT, kMaxHeadRows)));
    if (opt.spec_k > 0 && (int) m.layers.size() <= g.n_layer)
        throw std::runtime_error("session: spec_k > 0 needs the MTP block (LoadOptions::load_mtp)");
    if (!ex.start || !ex.eager) throw std::runtime_error("session: experts hooks missing");
    SessionPtr s(new Session);
    s->m = &m;
    s->opt = opt;
    s->ex = ex;
    s->st = st;
    s->ws = workspace_create(m, kCatchUp);
    s->kda.assign(g.n_layer, nullptr);
    s->mla.assign(g.n_layer, nullptr);
    for (int il = 0; il < g.n_layer; ++il) {
        if (m.layers[il].mla) s->mla[il] = mla_cache_create(g, opt.max_ctx, opt.mla_latent);
        else {
            s->kda[il] = kda_state_create(g);
            kda_state_reset(s->kda[il], st);
            if (!s->k0) s->k0 = s->kda[il];
        }
    }
    ck(cudaMalloc(&s->scratch, std::max(kda_scratch_bytes(g, kCatchUp), mla_scratch_bytes(g, kCatchUp, opt.max_ctx))),
       "scratch");
    ck(cudaMalloc(&s->d_pos0, sizeof(int64_t)), "d_pos0");
    ck(cudaMallocHost(&s->h_pos0, sizeof(int64_t)), "h_pos0");
    ck(cudaMalloc(&s->hcat, (size_t) (kCatchUp + 1) * g.n_embd * 4), "hcat");
    ck(cudaEventCreateWithFlags(&s->ev_save, cudaEventDisableTiming), "ev_save");
    ck(cudaEventCreateWithFlags(&s->ev_ready, cudaEventDisableTiming), "ev_ready");
    ck(cudaMemsetAsync(s->hcat, 0, (size_t) g.n_embd * 4, st), "hcat");   // h[-1] = 0

    Session * sp = s.get();
    s->gh.attn_in = [sp](int il, const float * y, const MatmulInput * y_in, float * out, const Batch & b,
                         cudaStream_t q) {
        const LayerWeights & L = sp->m->layers[il];
        if (L.mla)
            mla_forward_dev(sp->m->g, L.mla_w, sp->mla[il], b.T, sp->d_pos0, y, out, sp->scratch, q, nullptr, y_in);
        else
            kda_forward(sp->m->g, L.kda, sp->kda[il], b, y, out, sp->scratch, q, y_in);
    };
    s->gh.experts = ex.start;
    s->gh.experts_finish = ex.finish;
    s->eh.experts = ex.eager;

    // two parities per T (each capture flips the host KDA cur)
    for (int p = 0; p < 2; ++p) {
        const int c = kda_state_cur(s->k0);
        s->g1[c] = capture_decode(m, *s->ws, 1, s->gh, st, true);
    }
    if (opt.spec_k > 0) {
        for (int p = 0; p < 2; ++p) {
            const int c = kda_state_cur(s->k0);
            s->gv[c] = capture_decode(m, *s->ws, opt.spec_k + 1, s->gh, st, true);
        }
        // a round's draft steps write up to k - 1 rows past the last verified position (pos + c + k <= max_ctx + k)
        s->mtp = mtp_create(m, opt.max_ctx + opt.spec_k, kCatchUp, opt.mla_latent);
        mtp_set_share_selection(s->mtp, opt.mtp_share);
    }
    ck(cudaStreamSynchronize(st), "session create");
    return s;
}

int32_t session_prompt(Session & s, const int32_t * ids, int n, const std::function<void(int, int)> & on_progress,
                       const std::function<void(int32_t)> & on_next) {
    if (n <= 0) return s.next;
    if (s.pos + n > s.opt.max_ctx) throw std::runtime_error("session_prompt: context full");
    s.fence();
    const size_t D = (size_t) s.m->g.n_embd;
    const double t0 = now_ms();
    if (s.ex.prefill && n >= s.opt.prefill_min) {
        // chunked large-T prefill: hcat row 0 = h[pos - 1] (pend empty after the flush), updated in place to the
        // prompt's last h; the MTP cache follows the prompt
        s.flush_catch_up();
        PrefillPrompt pp;
        pp.kda = &s.kda;
        pp.mla = &s.mla;
        pp.max_ctx = s.opt.max_ctx;
        pp.mtp = s.mtp;
        pp.h_prev = s.hcat;
        pp.on_chunk = on_progress;
        pp.on_next = on_next;
        s.next = prefill_prompt(s.ex.prefill, *s.m, pp, ids, n, s.pos, s.st);
        s.toks.insert(s.toks.end(), ids, ids + n);
        s.pos += n;
        s.pend_pos0 = s.pos;
        // the loan's reclaim is in flight: later steps admit it (this one usually finds it still running)
        if (s.ex.step) s.ex.step(s.st);
    } else {
        for (int i = 0; i < n; ++i) {
            s.replay(*s.g1[kda_state_cur(s.k0)], ids + i, 1, s.pos);
            s.toks.push_back(ids[i]);
            for (auto * c : s.mla)
                if (c) mla_cache_set_size(c, s.pos + 1);
            if (s.mtp) {
                // MTP row pos = (ids[i], h[pos - 1]): h[pos - 1] is hcat row pend.size(); h[pos] goes one row further
                if (s.pend.empty()) s.pend_pos0 = s.pos;
                s.pend.push_back(ids[i]);
                ck(cudaMemcpyAsync(s.hcat + s.pend.size() * D, s.ws->hn, D * 4, cudaMemcpyDeviceToDevice, s.st),
                   "h row");
                if ((int) s.pend.size() == kCatchUp) s.flush_catch_up();
            }
            ++s.pos;
            if (s.ex.step) {
                ck(cudaStreamSynchronize(s.st), "prompt step");
                s.ex.step(s.st);
            }
            if (on_progress) on_progress(i + 1, n);
        }
        ck(cudaMemcpyAsync(&s.next, s.ws->tok, 4, cudaMemcpyDeviceToHost, s.st), "next");
        if (on_next) {
            ck(cudaStreamSynchronize(s.st), "next");
            on_next(s.next);
        }
    }
    ck(cudaStreamSynchronize(s.st), "prompt");
    s.drafts.clear();
    if (s.mtp) {
        s.flush_catch_up();
        s.draft_pending();
    }
    s.stats.prompt += n;
    s.stats.prompt_ms += now_ms() - t0;
    return s.next;
}

const char * finish_reason_name(FinishReason r) {
    switch (r) {
        case FinishReason::stop: return "stop";
        case FinishReason::length: return "length";
        case FinishReason::cancel: return "cancel";
        case FinishReason::ctx: return "ctx";
    }
    return "?";
}

FinishReason session_generate(Session & s, int max_new, std::vector<int32_t> & out,
                              const std::function<void(const int32_t *, int, double)> & on_round,
                              const GenerateControl & ctl) {
    if (max_new <= 0) return FinishReason::length;
    if (s.next < 0) throw std::runtime_error("session_generate: no pending token (session_prompt first)");
    s.fence();
    const Geometry & g = s.m->g;
    const size_t D = (size_t) g.n_embd;
    const int k = s.opt.spec_k;
    if (k > 0 && (int) s.drafts.size() != k) s.draft_pending();
    const auto is_stop = [&ctl](int32_t t) {
        return std::find(ctl.stop_ids.begin(), ctl.stop_ids.end(), t) != ctl.stop_ids.end();
    };
    int made = 0;
    std::vector<int32_t> vids(k + 1), o(k + 1), emit;
    for (;;) {
        if (made >= max_new) return FinishReason::length;
        if (ctl.cancel && ctl.cancel->load(std::memory_order_relaxed)) return FinishReason::cancel;
        if (s.pos + k + 1 > s.opt.max_ctx) return FinishReason::ctx;
        const double r0 = now_ms();
        std::optional<NvtxRange> nr(std::in_place, "round");   // ends before on_round
        emit.clear();
        bool stopped = false;
        if (k == 0) {
            s.replay(*s.g1[kda_state_cur(s.k0)], &s.next, 1, s.pos);
            for (auto * c : s.mla)
                if (c) mla_cache_set_size(c, s.pos + 1);
            emit.push_back(s.next);
            stopped = is_stop(s.next);
            ck(cudaMemcpyAsync(&s.next, s.ws->tok, 4, cudaMemcpyDeviceToHost, s.st), "next");
            ck(cudaStreamSynchronize(s.st), "step");
            if (s.opt.record_margins) s.record(1);
            ++s.pos;
        } else {
            // verify [next, d1..dk] at pos..pos+k
            if (s.opt.test_reject && s.stats.rounds % (k + 1) < k) {
                int32_t & d = s.drafts[s.stats.rounds % (k + 1)];
                d = (d + 1) % s.m->g.n_vocab;
            }
            vids[0] = s.next;
            for (int i = 0; i < k; ++i) vids[i + 1] = s.drafts[i];
            const double v0 = now_ms();
            {
                const NvtxRange nv("verify");
                s.replay(*s.gv[kda_state_cur(s.k0)], vids.data(), k + 1, s.pos);
                ck(cudaMemcpyAsync(o.data(), s.ws->tok, (size_t) (k + 1) * 4, cudaMemcpyDeviceToHost, s.st), "verify");
                ck(cudaStreamSynchronize(s.st), "verify");
            }
            s.stats.verify_ms += now_ms() - v0;
            int a = 0;
            while (a < k && s.drafts[a] == o[a]) ++a;
            // rows 0..a (tokens next, d1..da) are verified; keep the first c of them (cut after a stop id or at
            // max_new, the rest is neither emitted nor fed); the trunk's token after them is o[c - 1]
            int c = std::min(a + 1, max_new - made);
            for (int i = 0; i < c; ++i)
                if (is_stop(vids[i])) {
                    c = i + 1;
                    stopped = true;
                    break;
                }
            const bool last = stopped || made + c >= max_new;
            if (s.opt.record_margins) s.record(c);
            if (c < k + 1)
                for (auto * kd : s.kda)
                    if (kd) kda_state_rollback(g, kd, c, s.st);
            for (auto * cc : s.mla)
                if (cc) mla_cache_set_size(cc, s.pos + c);
            emit.assign(vids.begin(), vids.begin() + c);
            const int32_t t = o[c - 1];
            // MTP rows pos+1..pos+c = (vids[1..c), t) with h rows 0..c-1 of the verify; the last round of the call
            // only refreshes the cache (the drafts from t are made when generation resumes)
            ck(cudaMemcpyAsync(s.hcat, s.ws->hn, (size_t) c * D * 4, cudaMemcpyDeviceToDevice, s.st), "h rows");
            std::vector<int32_t> ids2(vids.begin() + 1, vids.begin() + c);
            ids2.push_back(t);
            const double d0 = now_ms();
            if (last) s.drafts.clear();
            {
                const NvtxRange nd("draft");
                mtp_draft(s.mtp, *s.m, *s.ws, s.eh, ids2.data(), s.hcat, c, s.pos + 1, last ? 0 : k,
                          last ? nullptr : s.drafts.data(), s.st);
            }
            s.stats.draft_ms += now_ms() - d0;
            if (c > 1)   // hcat row 0 = h[pos - 1] for the next prompt / snapshot
                ck(cudaMemcpyAsync(s.hcat, s.hcat + (size_t) (c - 1) * D, D * 4, cudaMemcpyDeviceToDevice, s.st),
                   "h row 0");
            s.stats.drafted += k;
            s.stats.accepted += a;
            s.pos += c;
            s.next = t;
        }
        s.toks.insert(s.toks.end(), emit.begin(), emit.end());
        ++s.stats.rounds;
        if (s.ex.step) {
            ck(cudaStreamSynchronize(s.st), "round");
            s.ex.step(s.st);
        }
        const int n = (int) emit.size();
        out.insert(out.end(), emit.begin(), emit.end());
        made += n;
        s.stats.generated += n;
        const double ms = now_ms() - r0;
        nr.reset();
        s.stats.gen_ms += ms;
        if (on_round) on_round(emit.data(), n, ms);
        if (stopped) return FinishReason::stop;
    }
}

size_t session_snapshot_bytes(const Session & s) {
    size_t n = 0;
    for (auto * kd : s.kda)
        if (kd) n += kda_state_bytes(s.m->g);
    return n + (size_t) s.m->g.n_embd * 4;
}

void session_snapshot_destroy(SessionSnapshot * p) {
    if (!p) return;
    if (p->host) cudaFreeHost(p->host);
    delete p;
}

void session_snapshot(Session & s, SessionSnapshotPtr & snap) {
    const size_t need = session_snapshot_bytes(s), kb = kda_state_bytes(s.m->g);
    if (!snap) snap.reset(new SessionSnapshot);
    if (snap->bytes != need) {
        if (snap->host) cudaFreeHost(snap->host);
        snap->host = nullptr;
        snap->bytes = 0;
        ck(cudaMallocHost(&snap->host, need), "snapshot alloc");
        snap->bytes = need;
    }
    char * p = static_cast<char *>(snap->host);
    for (auto * kd : s.kda)
        if (kd) {
            kda_state_save(kd, p, s.st);
            p += kb;
        }
    ck(cudaMemcpyAsync(p, s.hcat, (size_t) s.m->g.n_embd * 4, cudaMemcpyDeviceToHost, s.st), "snapshot h");
    ck(cudaStreamSynchronize(s.st), "snapshot");
    snap->pos = s.pos;
    snap->next = s.next;
    snap->toks = s.toks;
}

int64_t session_snapshot_pos(const SessionSnapshot & sn) { return sn.pos; }
const std::vector<int32_t> & session_snapshot_tokens(const SessionSnapshot & sn) { return sn.toks; }

bool session_restorable(const Session & s, const SessionSnapshot & sn) {
    if (sn.pos > s.pos || !std::equal(sn.toks.begin(), sn.toks.end(), s.toks.begin())) return false;
    for (auto * c : s.mla)
        if (c && !mla_cache_can_truncate(c, sn.pos)) return false;
    return !s.mtp || mla_cache_can_truncate(mtp_cache(s.mtp), std::min<int64_t>(sn.pos, mtp_size(s.mtp)));
}

void session_restore(Session & s, const SessionSnapshot & sn) {
    if (sn.bytes != session_snapshot_bytes(s)) throw std::runtime_error("session_restore: snapshot of another model");
    if (sn.pos > s.pos || !std::equal(sn.toks.begin(), sn.toks.end(), s.toks.begin()))
        throw std::runtime_error("session_restore: the snapshot's tokens are not a prefix of the session's");
    if (!session_restorable(s, sn))
        throw std::runtime_error("session_restore: the MLA caches cannot cut inside rows loaded from the prefix cache");
    s.fence();
    s.trim_chain(sn.pos);
    for (auto * c : s.mla)
        if (c) mla_cache_truncate(c, sn.pos);
    if (s.mtp) mtp_truncate(s.mtp, std::min<int64_t>(sn.pos, mtp_size(s.mtp)));
    const size_t kb = kda_state_bytes(s.m->g);
    const char * p = static_cast<const char *>(sn.host);
    for (auto * kd : s.kda)
        if (kd) {
            kda_state_load(kd, p, s.st);
            p += kb;
        }
    ck(cudaMemcpyAsync(s.hcat, p, (size_t) s.m->g.n_embd * 4, cudaMemcpyHostToDevice, s.st), "restore h");
    ck(cudaStreamSynchronize(s.st), "restore");
    s.pos = sn.pos;
    s.toks.resize(sn.pos);
    s.next = sn.next;
    s.drafts.clear();
    s.pend.clear();
}

void session_reset(Session & s) {
    s.fence();
    s.chain.reset();
    for (auto * kd : s.kda)
        if (kd) kda_state_reset(kd, s.st);
    for (auto * c : s.mla)
        if (c) mla_cache_truncate(c, 0);
    if (s.mtp) mtp_truncate(s.mtp, 0);
    ck(cudaMemsetAsync(s.hcat, 0, (size_t) s.m->g.n_embd * 4, s.st), "reset h");
    ck(cudaStreamSynchronize(s.st), "reset");
    s.pos = 0;
    s.toks.clear();
    s.next = -1;
    s.drafts.clear();
    s.pend.clear();
}

void session_save(Session & s, PrefixCache & pc) {
    const int64_t P = s.pos;
    if (P == 0) return;
    for (auto & e : pc.entries)
        if (e->toks == s.toks) {
            e->used = ++pc.clock;
            return;
        }
    const auto cs = s.caches();
    if (s.mtp && mtp_size(s.mtp) < P) {   // (the MTP cache follows the trunk between calls; never expected)
        ++pc.stats.save_skips;
        return;
    }
    s.trim_chain(P);
    const size_t kb = kda_state_bytes(s.m->g), tb = mla_cache_tail_bytes(cs[0]);
    auto e = std::make_unique<PrefixEntry>();
    // allocations first (evicting older entries: top keeps the session's chain alive), copies after
    std::vector<std::shared_ptr<PrefixSegment>> segs;
    std::shared_ptr<PrefixSegment> top = s.chain_in(pc);
    bool ok = true;
    size_t bytes = 0;
    for (int64_t a = top ? top->b : 0; a < P && ok;) {
        auto sg = std::make_shared<PrefixSegment>();
        sg->pc = &pc;
        ++pc.stats.segments;
        sg->a = a;
        sg->b = std::min(P, a + pc.opt.segment_max);
        sg->parent = top;
        for (auto * c : cs) {
            sg->rows.push_back(pc.alloc_evict(mla_cache_rows_bytes(c, sg->a, sg->b)));
            ok = ok && sg->rows.back().p;
            bytes += sg->rows.back().bytes;
        }
        segs.push_back(sg);
        top = sg;
        a = sg->b;
    }
    for (auto * kd : s.kda)
        if (kd && ok) {
            e->kda.push_back(pc.alloc_evict(kb));
            ok = e->kda.back().p != nullptr;
        }
    if (ok) {
        e->small = pc.alloc_evict(cs.size() * tb + (size_t) s.m->g.n_embd * 4);
        ok = e->small.p != nullptr;
    }
    if (!ok) {
        for (auto & k : e->kda) pc.release(k);
        pc.release(e->small);
        ++pc.stats.save_skips;
        return;   // segs go with their allocations
    }
    // the copies, on the cache's stream once the session's stream has written the state
    ck(cudaEventRecord(s.ev_ready, s.st), "save ready");
    ck(cudaStreamWaitEvent(pc.st, s.ev_ready, 0), "save wait");
    for (auto & sg : segs)
        for (size_t i = 0; i < cs.size(); ++i) mla_cache_export_rows(cs[i], sg->a, sg->b, sg->rows[i].p, pc.st);
    char * q = e->small.p;
    for (auto * c : cs) {
        mla_cache_export_tail(c, P, q, pc.st);
        q += tb;
    }
    ck(cudaMemcpyAsync(q, s.hcat, (size_t) s.m->g.n_embd * 4, cudaMemcpyDeviceToHost, pc.st), "save h");
    size_t j = 0;
    for (auto * kd : s.kda)
        if (kd) kda_state_save(kd, e->kda[j++].p, pc.st);
    ck(cudaEventCreateWithFlags(&e->done, cudaEventDisableTiming), "save event");
    ck(cudaEventRecord(e->done, pc.st), "save done");
    ck(cudaEventRecord(s.ev_save, pc.st), "save fence");
    s.save_pending = true;
    e->toks = s.toks;
    e->next = s.next;
    e->last = top;
    e->used = ++pc.clock;
    e->bytes = (size_t) j * kb + e->small.bytes;
    s.chain = top;
    pc.stats.bytes_saved += bytes + e->bytes;
    ++pc.stats.saves;
    ++pc.stats.entries;
    pc.entries.push_back(std::move(e));
}

int64_t session_load(Session & s, PrefixCache & pc, PrefixEntry & e) {
    const int64_t P = (int64_t) e.toks.size();
    if (P > s.opt.max_ctx) throw std::runtime_error("session_load: the entry does not fit the context");
    s.fence();
    ck(cudaStreamWaitEvent(s.st, e.done, 0), "load wait");
    const auto cs = s.caches();
    // rows to keep: the deepest boundary of a segment both chains hold (bitwise the same rows) where every cache can
    // cut (the MTP cache also needs the rows)
    int64_t keep = 0;
    {
        std::vector<const PrefixSegment *> mine;
        const std::shared_ptr<PrefixSegment> own = s.chain_in(pc);
        for (const PrefixSegment * p = own.get(); p; p = p->parent.get()) mine.push_back(p);
        const int64_t lim = s.mtp ? std::min(s.pos, mtp_size(s.mtp)) : s.pos;
        for (const PrefixSegment * p = e.last.get(); p && !keep; p = p->parent.get()) {
            if (p->b > lim || std::find(mine.begin(), mine.end(), p) == mine.end()) continue;
            bool cut = true;
            for (auto * c : cs) cut = cut && mla_cache_can_truncate(c, p->b);
            if (cut) keep = p->b;
        }
    }
    std::vector<const PrefixSegment *> segs;   // the entry's segments past keep, root first
    for (const PrefixSegment * p = e.last.get(); p && p->b > keep; p = p->parent.get()) segs.push_back(p);
    std::reverse(segs.begin(), segs.end());
    for (auto * c : cs) mla_cache_truncate(c, keep);
    const size_t kb = kda_state_bytes(s.m->g), tb = mla_cache_tail_bytes(cs[0]);
    bool ok = true;
    size_t bytes = 0;
    for (auto * sg : segs) {
        ok = ok && sg->rows.size() == cs.size();
        for (size_t i = 0; ok && i < cs.size(); ++i) {
            ok = mla_cache_import_rows(cs[i], sg->a, sg->b, sg->rows[i].p, s.st);
            bytes += sg->rows[i].bytes;
        }
    }
    const char * q = e.small.p;
    for (size_t i = 0; ok && i < cs.size(); ++i, q += tb) ok = mla_cache_import_tail(cs[i], P, q, s.st);
    ok = ok && e.kda.size() == (size_t) std::count_if(s.kda.begin(), s.kda.end(), [](auto * k) { return k; });
    if (!ok) {
        ck(cudaStreamSynchronize(s.st), "load");
        session_reset(s);
        throw std::runtime_error("session_load: the entry was saved from other cache types");
    }
    ck(cudaMemcpyAsync(s.hcat, q, (size_t) s.m->g.n_embd * 4, cudaMemcpyHostToDevice, s.st), "load h");
    size_t j = 0;
    for (auto * kd : s.kda)
        if (kd) kda_state_load(kd, e.kda[j++].p, s.st);
    ck(cudaStreamSynchronize(s.st), "load");
    s.pos = P;
    s.toks = e.toks;
    s.next = e.next;
    s.drafts.clear();
    s.pend.clear();
    s.pend_pos0 = P;
    s.chain = e.last;
    e.used = ++pc.clock;
    pc.stats.tokens_restored += P - keep;
    pc.stats.bytes_restored += bytes + j * kb + e.small.bytes;
    return P - keep;
}

std::vector<std::pair<std::string, uint64_t>> session_state_digest(Session & s) {
    s.fence();
    ck(cudaStreamSynchronize(s.st), "digest");
    const auto hash = [](const void * p, size_t n) {
        uint64_t h = 1469598103934665603ull;
        const auto * b = static_cast<const unsigned char *>(p);
        size_t i = 0;
        for (; i + 8 <= n; i += 8) {
            uint64_t w;
            std::memcpy(&w, b + i, 8);
            h = (h ^ w) * 1099511628211ull;
            h ^= h >> 29;
        }
        for (; i < n; ++i) h = (h ^ b[i]) * 1099511628211ull;
        return h;
    };
    std::vector<char> buf;
    const auto dev = [&](const void * src, size_t n) {   // device or mapped memory
        buf.resize(n);
        if (n) ck(cudaMemcpy(buf.data(), src, n, cudaMemcpyDefault), "digest copy");
        return hash(buf.data(), n);
    };
    std::vector<std::pair<std::string, uint64_t>> d;
    const int64_t P = s.pos, lo = P / 4 * 4;
    const auto cs = s.caches();
    for (size_t i = 0; i < cs.size(); ++i) {
        const MlaCache * c = cs[i];
        const std::string n = s.mtp && i + 1 == cs.size() ? "mtp" : "mla" + std::to_string(i);
        const bool q8 = mla_cache_latent_type(c) == MlaLatent::Q8;
        d.emplace_back(n + ".latent", dev(mla_cache_latent(c), (size_t) P * 512 * (q8 ? 1 : 2)));
        if (q8) d.emplace_back(n + ".scale", dev(mla_cache_latent_scale(c), (size_t) P * 8 * 2));
        d.emplace_back(n + ".pooled", dev(mla_cache_pooled(c), (size_t) (P / 4) * 128 * 2));
        d.emplace_back(n + ".kg_tail",
                       dev(static_cast<const char *>(mla_cache_kg(c)) + (size_t) lo * 512, (size_t) (P - lo) * 512));
    }
    const size_t kb = kda_state_bytes(s.m->g);
    buf.resize(kb);
    for (size_t il = 0; il < s.kda.size(); ++il)
        if (s.kda[il]) {
            kda_state_save(s.kda[il], buf.data(), s.st);
            ck(cudaStreamSynchronize(s.st), "digest kda");
            d.emplace_back("kda" + std::to_string(il), hash(buf.data(), kb));
        }
    d.emplace_back("hcat0", dev(s.hcat, (size_t) s.m->g.n_embd * 4));
    std::vector<int64_t> meta(s.toks.begin(), s.toks.end());
    meta.push_back(s.pos);
    meta.push_back(s.next);
    d.emplace_back("pos+tokens+next", hash(meta.data(), meta.size() * 8));
    return d;
}

int64_t session_pos(const Session & s) { return s.pos; }
const std::vector<int32_t> & session_tokens(const Session & s) { return s.toks; }
int32_t session_pending(const Session & s) { return s.next; }
const std::vector<float> & session_margins(const Session & s) { return s.margins; }
const std::vector<float> & session_top1(const Session & s) { return s.top1; }
SessionStats & session_stats(Session & s) { return s.stats; }

}  // namespace strata::glm
