#pragma once

#include <cstdint>

#include "cutlass/cutlass.h"
#include "cute/container/array.hpp"
#include "cute/int_tuple.hpp"

struct __align__(16) PowDiagnostics {
  uint32_t attempts;
  uint32_t best_msw32;
  uint32_t update_lock;
  uint32_t reserved;
  uint32_t best_hash[8];
  uint32_t best_tile_coord[3];
  uint32_t best_thread_idx;
};

static constexpr int pow_diagnostics_u32_size =
    sizeof(PowDiagnostics) / sizeof(uint32_t);

static_assert(sizeof(PowDiagnostics) == 64,
              "PowDiagnostics is expected to be one 64-byte cache line");

#if defined(__CUDACC__)
template <typename BlockCoord>
CUTLASS_DEVICE void record_pow_diagnostics(
    PowDiagnostics* diagnostics,
    const cute::array<uint32_t, 8>& hash, BlockCoord const& block_coord,
    int thread_idx) {
  if (diagnostics == nullptr) {
    return;
  }

  uint32_t old_attempts = atomicAdd(&diagnostics->attempts, 1U);
  bool first_attempt = old_attempts == 0U;

  // Word 7 is the most-significant uint32 in Pearl's little-endian uint256.
  uint32_t candidate_msw32 = hash[7];
  if (!first_attempt && candidate_msw32 > diagnostics->best_msw32) {
    return;
  }

  while (atomicCAS(&diagnostics->update_lock, 0U, 1U) != 0U) {
  }

  bool still_first = first_attempt && diagnostics->attempts == 1U;
  if (still_first || candidate_msw32 < diagnostics->best_msw32) {
    diagnostics->best_msw32 = candidate_msw32;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < 8; ++i) {
      diagnostics->best_hash[i] = hash[i];
    }
    diagnostics->best_tile_coord[0] =
        static_cast<uint32_t>(cute::get<0>(block_coord));
    diagnostics->best_tile_coord[1] =
        static_cast<uint32_t>(cute::get<1>(block_coord));
    diagnostics->best_tile_coord[2] =
        static_cast<uint32_t>(cute::get<2>(block_coord));
    diagnostics->best_thread_idx = static_cast<uint32_t>(thread_idx);
  }

  atomicExch(&diagnostics->update_lock, 0U);
}
#endif
