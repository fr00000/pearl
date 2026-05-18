#pragma once

#include <cstdint>
#include "blake3_constants.hpp"
#include "cute/layout.hpp"
#include "cute/tensor.hpp"

#include <cutlass/arch/memory.h>
#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/fast_math.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>
#include <cutlass/detail/layout.hpp>

using namespace cute;

// A single BLAKE3 round (i.e. 8 G operations -- see section 2.2 of BLAKE3 spec)
// laid out in a primitive form, to allow the compiler to do everything in registers.
// state0-15 are equivalent to v0-15 in the spec.
#define BLAKE3_ROUND()                                          \
  do {                                                          \
    rState(0) = add32(rState(0), add32(rState(4), rBlock(0)));  \
    rState(12) = rightrotate32(rState(12) ^ rState(0), 16);     \
    rState(8) = add32(rState(8), rState(12));                   \
    rState(4) = rightrotate32(rState(4) ^ rState(8), 12);       \
    rState(0) = add32(rState(0), add32(rState(4), rBlock(1)));  \
    rState(12) = rightrotate32(rState(12) ^ rState(0), 8);      \
    rState(8) = add32(rState(8), rState(12));                   \
    rState(4) = rightrotate32(rState(4) ^ rState(8), 7);        \
    rState(1) = add32(rState(1), add32(rState(5), rBlock(2)));  \
    rState(13) = rightrotate32(rState(13) ^ rState(1), 16);     \
    rState(9) = add32(rState(9), rState(13));                   \
    rState(5) = rightrotate32(rState(5) ^ rState(9), 12);       \
    rState(1) = add32(rState(1), add32(rState(5), rBlock(3)));  \
    rState(13) = rightrotate32(rState(13) ^ rState(1), 8);      \
    rState(9) = add32(rState(9), rState(13));                   \
    rState(5) = rightrotate32(rState(5) ^ rState(9), 7);        \
    rState(2) = add32(rState(2), add32(rState(6), rBlock(4)));  \
    rState(14) = rightrotate32(rState(14) ^ rState(2), 16);     \
    rState(10) = add32(rState(10), rState(14));                 \
    rState(6) = rightrotate32(rState(6) ^ rState(10), 12);      \
    rState(2) = add32(rState(2), add32(rState(6), rBlock(5)));  \
    rState(14) = rightrotate32(rState(14) ^ rState(2), 8);      \
    rState(10) = add32(rState(10), rState(14));                 \
    rState(6) = rightrotate32(rState(6) ^ rState(10), 7);       \
    rState(3) = add32(rState(3), add32(rState(7), rBlock(6)));  \
    rState(15) = rightrotate32(rState(15) ^ rState(3), 16);     \
    rState(11) = add32(rState(11), rState(15));                 \
    rState(7) = rightrotate32(rState(7) ^ rState(11), 12);      \
    rState(3) = add32(rState(3), add32(rState(7), rBlock(7)));  \
    rState(15) = rightrotate32(rState(15) ^ rState(3), 8);      \
    rState(11) = add32(rState(11), rState(15));                 \
    rState(7) = rightrotate32(rState(7) ^ rState(11), 7);       \
    rState(0) = add32(rState(0), add32(rState(5), rBlock(8)));  \
    rState(15) = rightrotate32(rState(15) ^ rState(0), 16);     \
    rState(10) = add32(rState(10), rState(15));                 \
    rState(5) = rightrotate32(rState(5) ^ rState(10), 12);      \
    rState(0) = add32(rState(0), add32(rState(5), rBlock(9)));  \
    rState(15) = rightrotate32(rState(15) ^ rState(0), 8);      \
    rState(10) = add32(rState(10), rState(15));                 \
    rState(5) = rightrotate32(rState(5) ^ rState(10), 7);       \
    rState(1) = add32(rState(1), add32(rState(6), rBlock(10))); \
    rState(12) = rightrotate32(rState(12) ^ rState(1), 16);     \
    rState(11) = add32(rState(11), rState(12));                 \
    rState(6) = rightrotate32(rState(6) ^ rState(11), 12);      \
    rState(1) = add32(rState(1), add32(rState(6), rBlock(11))); \
    rState(12) = rightrotate32(rState(12) ^ rState(1), 8);      \
    rState(11) = add32(rState(11), rState(12));                 \
    rState(6) = rightrotate32(rState(6) ^ rState(11), 7);       \
    rState(2) = add32(rState(2), add32(rState(7), rBlock(12))); \
    rState(13) = rightrotate32(rState(13) ^ rState(2), 16);     \
    rState(8) = add32(rState(8), rState(13));                   \
    rState(7) = rightrotate32(rState(7) ^ rState(8), 12);       \
    rState(2) = add32(rState(2), add32(rState(7), rBlock(13))); \
    rState(13) = rightrotate32(rState(13) ^ rState(2), 8);      \
    rState(8) = add32(rState(8), rState(13));                   \
    rState(7) = rightrotate32(rState(7) ^ rState(8), 7);        \
    rState(3) = add32(rState(3), add32(rState(4), rBlock(14))); \
    rState(14) = rightrotate32(rState(14) ^ rState(3), 16);     \
    rState(9) = add32(rState(9), rState(14));                   \
    rState(4) = rightrotate32(rState(4) ^ rState(9), 12);       \
    rState(3) = add32(rState(3), add32(rState(4), rBlock(15))); \
    rState(14) = rightrotate32(rState(14) ^ rState(3), 8);      \
    rState(9) = add32(rState(9), rState(14));                   \
    rState(4) = rightrotate32(rState(4) ^ rState(9), 7);        \
  } while (0)

// Explicit BLAKE3 state permutation (as above, this is done to allow the compiler to do everything in registers)
#define BLAKE3_PERMUTE()         \
  do {                           \
    rOrigBlock(0) = rBlock(0);   \
    rOrigBlock(1) = rBlock(1);   \
    rOrigBlock(2) = rBlock(2);   \
    rOrigBlock(3) = rBlock(3);   \
    rOrigBlock(4) = rBlock(4);   \
    rOrigBlock(5) = rBlock(5);   \
    rOrigBlock(6) = rBlock(6);   \
    rOrigBlock(7) = rBlock(7);   \
    rOrigBlock(8) = rBlock(8);   \
    rOrigBlock(9) = rBlock(9);   \
    rOrigBlock(10) = rBlock(10); \
    rOrigBlock(11) = rBlock(11); \
    rOrigBlock(12) = rBlock(12); \
    rOrigBlock(13) = rBlock(13); \
    rOrigBlock(14) = rBlock(14); \
    rOrigBlock(15) = rBlock(15); \
    rBlock(0) = rOrigBlock(2);   \
    rBlock(1) = rOrigBlock(6);   \
    rBlock(2) = rOrigBlock(3);   \
    rBlock(3) = rOrigBlock(10);  \
    rBlock(4) = rOrigBlock(7);   \
    rBlock(5) = rOrigBlock(0);   \
    rBlock(6) = rOrigBlock(4);   \
    rBlock(7) = rOrigBlock(13);  \
    rBlock(8) = rOrigBlock(1);   \
    rBlock(9) = rOrigBlock(11);  \
    rBlock(10) = rOrigBlock(12); \
    rBlock(11) = rOrigBlock(5);  \
    rBlock(12) = rOrigBlock(9);  \
    rBlock(13) = rOrigBlock(14); \
    rBlock(14) = rOrigBlock(15); \
    rBlock(15) = rOrigBlock(8);  \
  } while (0)

#define BLAKE3_G_MSG(a, b, c, d, mx, my)                         \
  do {                                                            \
    rState(a) = add32(rState(a), add32(rState(b), block(mx)));    \
    rState(d) = rightrotate32(rState(d) ^ rState(a), 16);         \
    rState(c) = add32(rState(c), rState(d));                      \
    rState(b) = rightrotate32(rState(b) ^ rState(c), 12);         \
    rState(a) = add32(rState(a), add32(rState(b), block(my)));    \
    rState(d) = rightrotate32(rState(d) ^ rState(a), 8);          \
    rState(c) = add32(rState(c), rState(d));                      \
    rState(b) = rightrotate32(rState(b) ^ rState(c), 7);          \
  } while (0)

#define BLAKE3_ROUND_MSG(m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, \
                         m11, m12, m13, m14, m15)                    \
  do {                                                               \
    BLAKE3_G_MSG(0, 4, 8, 12, m0, m1);                               \
    BLAKE3_G_MSG(1, 5, 9, 13, m2, m3);                               \
    BLAKE3_G_MSG(2, 6, 10, 14, m4, m5);                              \
    BLAKE3_G_MSG(3, 7, 11, 15, m6, m7);                              \
    BLAKE3_G_MSG(0, 5, 10, 15, m8, m9);                              \
    BLAKE3_G_MSG(1, 6, 11, 12, m10, m11);                            \
    BLAKE3_G_MSG(2, 7, 8, 13, m12, m13);                             \
    BLAKE3_G_MSG(3, 4, 9, 14, m14, m15);                             \
  } while (0)

using u32 = uint32_t;
using u64 = uint64_t;

namespace blake3 {
// BLAKE3 compress parameters
// counter: A 64-bit counter, t = t0, t1, with t0 the lower order word and t1 the higher order word.
// block_len: The number of input bytes in the block (32 bits).
// flags: A set of domain separation bit flags (32 bits).
struct CompressParams {
  u64 counter;
  u32 block_len;
  u32 flags;
};

__device__ __constant__ u32 IV[8] = {IV0, IV1, IV2, IV3, IV4, IV5, IV6, IV7};

// Some constexpr variants of CompressParams
// Regular inner node compression in the Merkle Tree stage
__device__ __constant__ constexpr CompressParams COMPRESS_PARAMS_INNER_NODE = {
    .counter = 0,
    .block_len = MSG_BLOCK_SIZE,
    .flags = KEYED_HASH | PARENT};
// Compression of the root of the Merkle Tree
__device__ __constant__ constexpr CompressParams COMPRESS_PARAMS_ROOT = {
    .counter = 0,
    .block_len = MSG_BLOCK_SIZE,
    .flags = KEYED_HASH | ROOT | PARENT};
// Compression for noise generation (single 64-byte message block with key)
__device__ __constant__ constexpr CompressParams
    COMPRESS_PARAMS_SINGLE_BLOCK_KEYED = {
        .counter = 0,
        .block_len = MSG_BLOCK_SIZE,
        .flags = KEYED_HASH | CHUNK_START | CHUNK_END | ROOT};

CUTLASS_DEVICE
u32 add32(u32 x, u32 y) {
  return x + y;
}

CUTLASS_DEVICE
u32 rightrotate32(u32 x, u32 n) {
  return (x << (32 - n)) | (x >> n);
}

// Compress a 64-byte message block
template <class RmemTensorBlock, class RmemTensorChainingValue>
CUTLASS_DEVICE void compress_msg_block_u32(
    RmemTensorBlock const& block, RmemTensorChainingValue& chaining_value,
    const CompressParams params) {
  Tensor rOrigBlock = make_tensor_like<uint32_t>(block);
  Tensor rState = make_tensor_like<uint32_t>(block);
  Tensor rBlock = make_tensor_like<uint32_t>(block);

  copy(block, rBlock);
  // Initialize state
  copy(chaining_value, rState);
  rState(8) = IV0;
  rState(9) = IV1;
  rState(10) = IV2;
  rState(11) = IV3;
  rState(12) = params.counter;
  rState(13) = params.counter >> 32;
  rState(14) = params.block_len;
  rState(15) = params.flags;

  // 6 rounds and permutations
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < 6; ++i) {
    BLAKE3_ROUND();
    BLAKE3_PERMUTE();
  }
  // Final round w/o permutation
  BLAKE3_ROUND();
  // Real BLAKE3 has some operations here on state8-15, but we don't care about these
  // so we can only change state0-7. Copy the result to the chaining value tensor.
  chaining_value(0) = rState(0) ^ rState(8);
  chaining_value(1) = rState(1) ^ rState(9);
  chaining_value(2) = rState(2) ^ rState(10);
  chaining_value(3) = rState(3) ^ rState(11);
  chaining_value(4) = rState(4) ^ rState(12);
  chaining_value(5) = rState(5) ^ rState(13);
  chaining_value(6) = rState(6) ^ rState(14);
  chaining_value(7) = rState(7) ^ rState(15);
}

// Compress the exact PoW message shape: one 64-byte block with a keyed BLAKE3
// chaining value. This avoids the generic compressor's rBlock copy and
// per-round permutation scratch. The message schedule below is P^round for
// BLAKE3's fixed permutation.
template <class RmemTensorBlock, class RmemTensorChainingValue>
CUTLASS_DEVICE void compress_single_block_keyed_u32_scheduled(
    RmemTensorBlock const& block, RmemTensorChainingValue& chaining_value) {
  Tensor rState = make_tensor<uint32_t>(Int<MSG_BLOCK_SIZE_U32>{});

  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < CHAINING_VALUE_SIZE_U32; ++i) {
    rState(i) = chaining_value(i);
  }
  rState(8) = IV0;
  rState(9) = IV1;
  rState(10) = IV2;
  rState(11) = IV3;
  rState(12) = 0;
  rState(13) = 0;
  rState(14) = MSG_BLOCK_SIZE;
  rState(15) = KEYED_HASH | CHUNK_START | CHUNK_END | ROOT;

  BLAKE3_ROUND_MSG(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
  BLAKE3_ROUND_MSG(2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8);
  BLAKE3_ROUND_MSG(3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1);
  BLAKE3_ROUND_MSG(10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6);
  BLAKE3_ROUND_MSG(12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4);
  BLAKE3_ROUND_MSG(9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7);
  BLAKE3_ROUND_MSG(11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13);

  chaining_value(0) = rState(0) ^ rState(8);
  chaining_value(1) = rState(1) ^ rState(9);
  chaining_value(2) = rState(2) ^ rState(10);
  chaining_value(3) = rState(3) ^ rState(11);
  chaining_value(4) = rState(4) ^ rState(12);
  chaining_value(5) = rState(5) ^ rState(13);
  chaining_value(6) = rState(6) ^ rState(14);
  chaining_value(7) = rState(7) ^ rState(15);
}
}  // namespace blake3
