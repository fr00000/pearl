#pragma once

#include <cuda_runtime.h>
#include <cstdint>

// Host function to launch the inner_hash kernel
void launch_inner_hash_kernel(uint32_t* input_buffer, int input_size,
                              uint32_t* output_hash, int64_t iterations,
                              cudaStream_t stream);

void launch_blake3_single_block_keyed_kernel(const uint32_t* block,
                                             const uint32_t* key,
                                             uint32_t* output_hash,
                                             cudaStream_t stream);
