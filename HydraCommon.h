#pragma once
/*
 * ======================================================================================
 * HYDRA - HydraCommon.h
 * ======================================================================================
 */

#include <cstdint>
#include <cuda_runtime.h>

// ============================================================================
// CONSTANTS
// ============================================================================
static constexpr int MAX_VAR_BITS = 64;

#ifndef HYDRA_WAVE_SIZE
#define HYDRA_WAVE_SIZE (1 << 22)
#endif

// LOW_BITS : number of variable bits handled by the intra-thread dictionary
// CPU precomputes 2^LOW_BITS affine points in __constant__ memory
// LOW_BITS=9 -> 512 points x 64 bytes = 32 KB (safe, limit is 64 KB)
#ifndef LOW_BITS
#define LOW_BITS 9
#endif
#define LOW_SIZE (1 << LOW_BITS)

// ============================================================================
// TARGET TYPE
// ============================================================================
enum class TargetType : uint32_t {
    BTC       = 0,
    ETH       = 1,
    BLOOM     = 2,  // Bloom filter -- all addresses (BTC + ETH)
    BLOOM_BTC = 3,  // Bloom filter -- BTC only (faster, skips ETH check)
    BLOOM_ETH = 4   // Bloom filter -- ETH only (faster, skips BTC check)
};

// ============================================================================
// STRUCTS
// ============================================================================

struct HydraData {
    uint32_t num_var_bits;
    uint32_t num_high_bits;     // = num_var_bits - LOW_BITS
    uint64_t total_candidates;  // 2^num_var_bits
    uint64_t high_candidates;   // 2^num_high_bits
    uint64_t gray_offset_start; // current wave offset (in LOW_SIZE units)

    uint64_t base_x[4];
    uint64_t base_y[4];

    uint64_t delta_x[MAX_VAR_BITS][4];
    uint64_t delta_y[MAX_VAR_BITS][4];
};

struct TargetData {
    TargetType      type;           // offset  0 (4 bytes)
    uint8_t         hash20[20];     // offset  4 (20 bytes)
    // offset 24 : already aligned on 8, no padding needed
    const uint64_t* d_bloom_filter; // offset 24 (8 bytes)
    uint64_t        bloom_m_bits;   // offset 32 (8 bytes)
};                                  // total  : 40 bytes, clean

struct HydraResult {
    int      found;
    uint32_t _pad;
    uint64_t index;
};

struct SeedResult {
    int     found;
    uint8_t entropy[16];
};