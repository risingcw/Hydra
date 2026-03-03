#pragma once
/*
 * ======================================================================================
 * HYDRA - Wif.cuh  (Mode WIF : brute force caractères $ inconnus)
 * ======================================================================================
 *
 * WIF compressé = 52 caractères Base58Check
 * Décodé = 38 bytes : [0x80][clé privée 32B][0x01][checksum 4B]
 * Checksum = SHA256(SHA256(bytes[0:34]))[0:4]
 *
 * Pipeline :
 *   K1 : décode Base58 → SHA256×2 → vérifie checksum 4 bytes → filtre ÷2^32
 *   K2 : extrait key[32] → ECC → Hash160 → compare
 *
 * ======================================================================================
 */

#include <cuda_runtime.h>
#include <stdint.h>
#include "HydraCommon.h"
#include "Bloom.h"
#include "ECC.h"
#include "Hash.cuh"

// ============================================================================
// CONSTANTES
// ============================================================================
#define WIF_LEN        52    // longueur WIF compressé en Base58
#define WIF_BYTES      38    // bytes décodés : 0x80 + 32B key + 0x01 + 4B checksum
#define WIF_KEY_OFFSET  1    // offset de la clé privée dans les bytes décodés
#define WIF_MAX_UNKN   12    // max positions $ raisonnables

// Alphabet Base58 : index 0='1', ..., 57='z'
// Table inverse précalculée en constant memory
__constant__ uint8_t B58_VAL[128] = {
    255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,  // 0x00-0x0F
    255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,  // 0x10-0x1F
    255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,  // 0x20-0x2F
    255,  0,  1,  2,  3,  4,  5,  6,  7,  8,255,255,255,255,255,255, // 0x30-0x3F '0'-'9'
    255,  9, 10, 11, 12, 13, 14, 15, 16,255, 17, 18, 19, 20, 21,255, // 0x40-0x4F 'A'-'O'
     22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,255,255,255,255,255, // 0x50-0x5F 'P'-'Z'
    255, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,255, 44, 45, 46, // 0x60-0x6F 'a'-'o'
     47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57,255,255,255,255,255  // 0x70-0x7F 'p'-'z'
};

// ============================================================================
// STRUCT WifMask
// ============================================================================
#define WIF_MAX_UNKN 12

struct WifMask {
    uint8_t  known_b58[WIF_LEN];  // valeur Base58 (0-57) pour chaque position
                                   // 0xFF = position $ inconnue
    uint8_t  num_chars;            // toujours 52
    uint8_t  num_unknown;          // nombre de $
    uint8_t  unknown_pos[WIF_MAX_UNKN]; // positions des $ dans known_b58
    uint8_t  _pad[1];
    uint64_t total_candidates;     // 58^num_unknown
};

// ============================================================================
// HELPER : décode un masque WIF (indices Base58) → 38 bytes
//
// Algorithme Base58 → big integer → bytes[38]
// WIF compressé commence toujours par 'K' ou 'L' (valeur ~22-24 en B58)
// Ce qui garantit que bytes[0] = 0x80 après décodage correct.
// ============================================================================
__device__ __forceinline__ void wif_decode_b58(
    const uint8_t b58vals[WIF_LEN],  // 52 valeurs 0-57
    uint8_t       out[WIF_BYTES])    // 38 bytes résultat
{
    // Arithmetic big-endian : résultat dans out[38]
    // On travaille en uint32_t pour éviter overflow
    uint32_t tmp[WIF_BYTES];
    #pragma unroll
    for(int i = 0; i < WIF_BYTES; i++) tmp[i] = 0;

    for(int i = 0; i < WIF_LEN; i++){
        uint32_t carry = b58vals[i];
        for(int j = WIF_BYTES - 1; j >= 0; j--){
            carry += 58u * tmp[j];
            tmp[j] = carry & 0xFF;
            carry >>= 8;
        }
    }

    #pragma unroll
    for(int i = 0; i < WIF_BYTES; i++) out[i] = (uint8_t)tmp[i];
}

// ============================================================================
// HELPER : SHA256 sur 34 bytes (payload WIF sans checksum)
// Format fixe : bytes[0..33] = 0x80 + key[32] + 0x01
// Message de 34 bytes → 1 bloc SHA256 (< 55 bytes)
// ============================================================================
__device__ __forceinline__ void sha256_34bytes(
    const uint8_t data[34],
    uint8_t       out[32])
{
    // Padding SHA256 pour 34 bytes :
    // W[0..7]  = data[0..31] big-endian
    // W[8]     = data[32..33] + 0x80 + 0x00 (big-endian)
    // W[9..14] = 0
    // W[15]    = 34 * 8 = 272 bits
    uint32_t W[16];

    // Charger les 32 premiers bytes (8 mots de 4 bytes, big-endian)
    #pragma unroll
    for(int i = 0; i < 8; i++){
        W[i] = ((uint32_t)data[i*4+0] << 24)
             | ((uint32_t)data[i*4+1] << 16)
             | ((uint32_t)data[i*4+2] <<  8)
             | ((uint32_t)data[i*4+3]);
    }
    // Bytes 32-33 + padding 0x80
    W[8] = ((uint32_t)data[32] << 24)
         | ((uint32_t)data[33] << 16)
         | 0x00008000u;
    #pragma unroll
    for(int i = 9; i < 15; i++) W[i] = 0;
    W[15] = 272u;  // 34 * 8

    uint32_t st[8];
    SHA256Initialize(st);
    SHA256Transform(st, W);

    #pragma unroll
    for(int i = 0; i < 8; i++){
        out[i*4+0] = (uint8_t)(st[i] >> 24);
        out[i*4+1] = (uint8_t)(st[i] >> 16);
        out[i*4+2] = (uint8_t)(st[i] >>  8);
        out[i*4+3] = (uint8_t)(st[i]);
    }
}

// ============================================================================
// HELPER : SHA256 sur 32 bytes (second passage du double-SHA256)
// ============================================================================
__device__ __forceinline__ void sha256_32bytes(
    const uint8_t data[32],
    uint8_t       out[32])
{
    uint32_t W[16];
    #pragma unroll
    for(int i = 0; i < 8; i++){
        W[i] = ((uint32_t)data[i*4+0] << 24)
             | ((uint32_t)data[i*4+1] << 16)
             | ((uint32_t)data[i*4+2] <<  8)
             | ((uint32_t)data[i*4+3]);
    }
    W[8] = 0x80000000u;
    #pragma unroll
    for(int i = 9; i < 15; i++) W[i] = 0;
    W[15] = 256u;  // 32 * 8

    uint32_t st[8];
    SHA256Initialize(st);
    SHA256Transform(st, W);

    #pragma unroll
    for(int i = 0; i < 8; i++){
        out[i*4+0] = (uint8_t)(st[i] >> 24);
        out[i*4+1] = (uint8_t)(st[i] >> 16);
        out[i*4+2] = (uint8_t)(st[i] >>  8);
        out[i*4+3] = (uint8_t)(st[i]);
    }
}

// ============================================================================
// HELPER : decode_wif_indices
// Transforme global_idx → b58vals[52] en remplaçant les positions $
// ============================================================================
__device__ __forceinline__ void decode_wif_indices(
    const WifMask* __restrict__ mask,
    uint64_t       global_idx,
    uint8_t        b58vals[WIF_LEN])
{
    // Copier les valeurs connues
    #pragma unroll
    for(int i = 0; i < WIF_LEN; i++) b58vals[i] = mask->known_b58[i];

    // Décoder global_idx → valeurs des positions inconnues
    uint64_t idx = global_idx;
    for(int x = (int)mask->num_unknown - 1; x >= 0; x--){
        b58vals[mask->unknown_pos[x]] = (uint8_t)(idx % 58);
        idx /= 58;
    }
}

// ============================================================================
// KERNEL 1 — FILTRE CHECKSUM WIF
// ~40 registres, haute occupancy
// Décode Base58 → SHA256×2 → compare checksum 4 bytes → filtre ÷2^32
// ============================================================================
__global__ void hydra_wif_checksum_kernel(
    const WifMask* __restrict__ mask,
    uint64_t       wave_offset,
    int            wave_size,
    uint64_t*      __restrict__ valid_indices,
    int*           __restrict__ valid_count)
{
    const int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if(tid >= wave_size) return;

    uint64_t global_idx = wave_offset + (uint64_t)tid;
    if(global_idx >= mask->total_candidates) return;

    // 1. Construire les 52 valeurs Base58
    uint8_t b58vals[WIF_LEN];
    decode_wif_indices(mask, global_idx, b58vals);

    // 2. Décoder Base58 → 38 bytes
    uint8_t wif_bytes[WIF_BYTES];
    wif_decode_b58(b58vals, wif_bytes);

    // Vérification sanity : byte[0] doit être 0x80, byte[33] doit être 0x01
    if(wif_bytes[0] != 0x80u || wif_bytes[33] != 0x01u) return;

    // 3. SHA256(SHA256(bytes[0..33])) → checksum attendu
    uint8_t h1[32], h2[32];
    sha256_34bytes(wif_bytes, h1);      // SHA256(payload 34 bytes)
    sha256_32bytes(h1, h2);             // SHA256(SHA256)

    // 4. Comparer les 4 premiers bytes du double-SHA256 avec bytes[34..37]
    if(h2[0] != wif_bytes[34]) return;
    if(h2[1] != wif_bytes[35]) return;
    if(h2[2] != wif_bytes[36]) return;
    if(h2[3] != wif_bytes[37]) return;

    // 5. Checksum valide → enregistrer
    int pos = atomicAdd(valid_count, 1);
    valid_indices[pos] = global_idx;
}

// ============================================================================
// KERNEL 2 — ECC + HASH160 + COMPARE
// Reçoit uniquement les indices validés par K1
// ============================================================================
__global__ __launch_bounds__(256, 1)
void hydra_wif_ecc_kernel(
    const WifMask*    __restrict__ mask,
    const TargetData* __restrict__ target,
    HydraResult*      __restrict__ result,
    const uint64_t*   __restrict__ valid_indices,
    int               valid_count)
{
    const int tid    = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int stride = (int)(blockDim.x * gridDim.x);

    for(int i = tid; i < valid_count; i += stride)
    {
        if(atomicAdd(&result->found, 0) != 0) return;

        uint64_t global_idx = valid_indices[i];

        // 1. Reconstruire les 52 valeurs Base58
        uint8_t b58vals[WIF_LEN];
        decode_wif_indices(mask, global_idx, b58vals);

        // 2. Décoder Base58 → 38 bytes
        uint8_t wif_bytes[WIF_BYTES];
        wif_decode_b58(b58vals, wif_bytes);

        // 3. Extraire la clé privée (bytes 1..32)
        uint64_t priv_limbs[4];
        #pragma unroll
        for(int j = 0; j < 4; j++){
            const uint8_t* p = wif_bytes + 1 + (3 - j) * 8;
            priv_limbs[j] = ((uint64_t)p[0] << 56) | ((uint64_t)p[1] << 48)
                          | ((uint64_t)p[2] << 40) | ((uint64_t)p[3] << 32)
                          | ((uint64_t)p[4] << 24) | ((uint64_t)p[5] << 16)
                          | ((uint64_t)p[6] <<  8) | ((uint64_t)p[7]);
        }

        // 4. ECC : clé privée → clé publique
        uint64_t ax[4], ay[4];
        scalarMulBaseAffine(priv_limbs, ax, ay);
        fieldNormalize(ax); fieldNormalize(ay);

        // 5. Hash160 + Keccak si nécessaire
        uint8_t h160[20];
        const uint8_t par = (ay[0] & 1) ? 0x03 : 0x02;
        getHash160_33_from_limbs(par, ax, h160);
        uint8_t computed[20];
        bool _hit = false;
        if(is_any_bloom(target)){
            // h160 déjà calculé avant ce bloc
            if(bloom_want_btc(target)){
                _hit = bloom_check(h160, target->d_bloom_filter, target->bloom_m_bits);
            }
            if(!_hit && bloom_want_eth(target)){
                uint8_t eth20[20]; getEthAddr_from_limbs(ax, ay, eth20);
                _hit = bloom_check(eth20, target->d_bloom_filter, target->bloom_m_bits);
            }
        } else {
            getHash160_33_from_limbs(par, ax, computed);
            _hit = hash20_matches(computed, target->hash20);
        }

        // 6. Comparer avec la cible (adresse unique ou bloom)
        if(_hit){
            if(atomicCAS(&result->found, 0, 1) == 0)
                result->index = global_idx;
            return;
        }
    }
}
