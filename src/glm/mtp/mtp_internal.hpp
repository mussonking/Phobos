#pragma once
// MTP internals for the parity driver. Not an engine interface.
#include <functional>

#include "strata/glm/mtp.hpp"

namespace strata::glm::mtp {

// Test only: replace the block-45 attention (mla_forward) by this function (the MLA cache is then not written).
using AttnOverride = std::function<void(const float * y, float * out, int T, cudaStream_t)>;
void set_attn_override(AttnOverride f);
bool attn_overridden();

// Test only: teacher forcing of a draft chain. Called before chain step i >= 1 with that step's greedy input token
// and its h row (device [4096], the previous step's h_nextn); may replace either.
using ChainOverride = std::function<void(int step, int32_t & token, float * h, cudaStream_t)>;
void set_chain_override(ChainOverride f);

}  // namespace strata::glm::mtp
