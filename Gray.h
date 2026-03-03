#pragma once
/*
 * ======================================================================================
 * HYDRA - Gray.h  (V4 : SPLIT HAUT/BAS — Affine Dict + Gray Jacobien)
 * ======================================================================================
 *
 * ARCHITECTURE :
 *
 *  CPU (Hydra.cu) :
 *    - Précalcule 2^LOW_BITS points affines R_k = (sum des bits bas actifs) * G
 *    - Stocke dans __constant__ DictX[LOW_SIZE][4], DictY[LOW_SIZE][4]
 *    - Pour les bits hauts : calcule les deltas Q_i = 2^(bit_haut_i) * G
 *
 *  GPU — un seul __global__ hydra_mega_kernel :
 *
 *  [Chaque thread = un P_base unique]
 *
 *  PHASE 1 — Gray Code sur bits HAUTS → P_base Jacobien
 *    Chaque thread avance d'un pas Gray sur les bits hauts.
 *    Résultat : P_base = (bits hauts) * G  en coordonnées Jacobiennes (X:Y:Z)
 *
 *  PHASE 2 — Normalisation affine de P_base (1 fieldInv par thread)
 *    P_base_affine = (X/Z², Y/Z³)
 *    → On profite de l'inversion pour rendre P_base utilisable en affine pur
 *
 *  PHASE 3 — Boucle intra-thread sur bits BAS (dictionnaire)
 *    Pour k = 0..LOW_SIZE-1 :
 *      P_k = P_base + R_k   (R_k depuis __constant__ DictX/DictY)
 *      ΔX_k = R_k.x - P_base.x  → accumulation Montgomery
 *    1 seul fieldInv pour tout le lot de LOW_SIZE candidats
 *    Back-prop → addition affine pure (2M + 1S par candidat)
 *    → hash + compare
 *
 *  COÛT PAR CANDIDAT (64 par thread) :
 *    Phase 1 : amortie (1 addition Jac pour LOW_SIZE candidats)
 *    Phase 2 : 1 fieldInv / LOW_SIZE = ~8 instr/candidat
 *    Phase 3 : 3M + 1S  (accum + backprop + lambda + x3/y3)
 *    Total   : ~3.5M + 1S   vs V2 : ~11M + 4S  → gain ~3×
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
// DICTIONNAIRE EN CONSTANT MEMORY
// R_k = combinaison des LOW_BITS bits bas, précalculé CPU
// DictX[0]/DictY[0] = point à l'infini (identité, k=0 → P_base lui-même)
// DictX[k]/DictY[k] = somme des bits bas actifs dans k, k=1..LOW_SIZE-1
// ============================================================================
__constant__ uint64_t DictX[LOW_SIZE][4];
__constant__ uint64_t DictY[LOW_SIZE][4];
__constant__ uint8_t  DictValid[LOW_SIZE];  // 1 = point valide, 0 = infini (k=0)

// ============================================================================
// ECC MIXED JAC+AFF → JAC  (Phase 1 : Gray sur bits hauts)
// ============================================================================
__device__ __forceinline__ void point_add_mixed_jac(
    uint64_t Rx[4], uint64_t Ry[4], uint64_t Rz[4],
    const uint64_t Px[4], const uint64_t Py[4], const uint64_t Pz[4],
    const uint64_t Qx[4], const uint64_t Qy[4])
{
    if ((Pz[0]|Pz[1]|Pz[2]|Pz[3])==0){
        fieldCopy(Qx,Rx); fieldCopy(Qy,Ry);
        Rz[0]=1;Rz[1]=0;Rz[2]=0;Rz[3]=0; return;
    }
    uint64_t Z1Z1[4],U2[4],S2[4],H[4],R2[4],HH[4],I[4],J[4],V[4],tmp[4];
    fieldSqr(Pz,Z1Z1);
    fieldMul(Qx,Z1Z1,U2);
    fieldMul(Pz,Z1Z1,S2); fieldMul(Qy,S2,S2);
    fieldSub(U2,Px,H);
    fieldSub(S2,Py,R2); fieldAdd(R2,R2,R2);
    if ((H[0]|H[1]|H[2]|H[3])==0){Rz[0]=Rz[1]=Rz[2]=Rz[3]=0;return;}
    fieldSqr(H,HH);
    fieldAdd(HH,HH,I); fieldAdd(I,I,I);
    fieldMul(H,I,J); fieldMul(Px,I,V);
    fieldSqr(R2,Rx); fieldSub(Rx,J,Rx); fieldSub(Rx,V,Rx); fieldSub(Rx,V,Rx);
    fieldSub(V,Rx,tmp); fieldMul(R2,tmp,Ry);
    fieldMul(Py,J,tmp); fieldAdd(tmp,tmp,tmp); fieldSub(Ry,tmp,Ry);
    fieldMul(Pz,H,Rz); fieldAdd(Rz,Rz,Rz);
}

__device__ __forceinline__ void point_sub_mixed_jac(
    uint64_t Rx[4], uint64_t Ry[4], uint64_t Rz[4],
    const uint64_t Px[4], const uint64_t Py[4], const uint64_t Pz[4],
    const uint64_t Qx[4], const uint64_t Qy[4])
{
    uint64_t nQy[4]; fieldNeg(Qy,nQy);
    point_add_mixed_jac(Rx,Ry,Rz,Px,Py,Pz,Qx,nQy);
}

// ============================================================================
// HASH comparaison
// ============================================================================
__device__ __forceinline__ bool hash20_matches(
    const uint8_t *a, const uint8_t *b)
{
    const uint32_t *x=(const uint32_t*)a;
    const uint32_t *y=(const uint32_t*)b;
    if (x[0]!=y[0]) return false;
    return (x[1]==y[1])&&(x[2]==y[2])&&(x[3]==y[3])&&(x[4]==y[4]);
}

// ============================================================================
// MEGA KERNEL V4
// ============================================================================
__global__ __launch_bounds__(256, 2)
void hydra_mega_kernel(
    const HydraData*  __restrict__ fd,
    const TargetData* __restrict__ target,
    HydraResult*      __restrict__ result,
    int wave_size)   // nombre de P_base (chunks hauts) dans cette wave
{
    const int tid    = (int)(blockIdx.x*blockDim.x+threadIdx.x);
    const int stride = (int)(blockDim.x*gridDim.x);

    for (int high_step = tid; high_step < wave_size; high_step += stride)
    {
        // Index global du P_base dans les bits hauts
        const uint64_t high_idx = fd->gray_offset_start + (uint64_t)high_step;
        if (high_idx >= fd->high_candidates) return;
        if (atomicAdd(&result->found,0)!=0) return;

        // ----------------------------------------------------------------
        // PHASE 1 : GRAY CODE sur bits HAUTS → P_base Jacobien
        // ----------------------------------------------------------------
        uint64_t Px[4],Py[4],Pz[4];
        #pragma unroll 4
        for(int i=0;i<4;i++){Px[i]=fd->base_x[i];Py[i]=fd->base_y[i];}
        Pz[0]=1;Pz[1]=0;Pz[2]=0;Pz[3]=0;

        {
            const uint64_t g0 = high_idx ^ (high_idx>>1);
            for(int b=0;b<(int)fd->num_high_bits;b++){
                if((g0>>b)&1ULL){
                    uint64_t tx[4],ty[4],tz[4];
                    point_add_mixed_jac(tx,ty,tz,Px,Py,Pz,
                                        fd->delta_x[b],fd->delta_y[b]);
                    #pragma unroll 4
                    for(int i=0;i<4;i++){Px[i]=tx[i];Py[i]=ty[i];Pz[i]=tz[i];}
                }
            }
        }

        // ----------------------------------------------------------------
        // PHASE 2 : NORMALISATION AFFINE de P_base (1 fieldInv)
        // ----------------------------------------------------------------
        uint64_t z_inv[4], z2[4], z3[4], bx[4], by[4];
        {
            uint64_t znorm[4];
            #pragma unroll 4
            for(int i=0;i<4;i++) znorm[i]=Pz[i];
            fieldNormalize(znorm);
            fieldInv(znorm, z_inv);
        }
        fieldSqr(z_inv,z2);
        fieldMul(z_inv,z2,z3);
        fieldMul(Px,z2,bx);
        fieldMul(Py,z3,by);
        fieldNormalize(bx);
        fieldNormalize(by);
        // bx, by = P_base en coordonnées affines ✓

        // ----------------------------------------------------------------
        // PHASE 3 : BOUCLE SUR BITS BAS (dictionnaire __constant__)
        //
        // Candidat k : P_k = P_base + R_k
        //   k=0 → P_base lui-même (R_0 = point à l'infini)
        //   k=1..LOW_SIZE-1 → addition affine via batch inversion
        //
        // Accumulation des ΔX_k = R_k.x - P_base.x
        // ----------------------------------------------------------------

        // Tableau local des ΔX et ΔY (pour back-prop)
        uint64_t l_dX [LOW_SIZE][4];
        uint64_t l_dY [LOW_SIZE][4];
        uint64_t l_acc[LOW_SIZE][4];  // acc[k] = dX[1]*...*dX[k]

        int valid_k = 0;  // nb de points avec ΔX valide (k≥1 et pas colinéaire)

        for(int k=1; k<LOW_SIZE; k++){
            // ΔX = R_k.x - P_base.x
            uint64_t dx[4], dy[4];
            fieldSub(DictX[k], bx, dx);

            // Cas dégénéré : ΔX==0 (R_k.x == P_base.x)
            // → P_base + R_k = doublement ou annulation, on skip (très rare)
            if ((dx[0]|dx[1]|dx[2]|dx[3])==0){
                // On met un marker : dX=0 → skip en back-prop
                #pragma unroll 4
                for(int i=0;i<4;i++){l_dX[k][i]=0; l_dY[k][i]=0;}
                // Copie acc[k] = acc[k-1] pour ne pas casser la chaîne
                if(valid_k>0){
                    #pragma unroll 4
                    for(int i=0;i<4;i++) l_acc[k][i]=l_acc[k-1][i];
                } else {
                    l_acc[k][0]=1;l_acc[k][1]=0;l_acc[k][2]=0;l_acc[k][3]=0;
                }
                continue;
            }

            // ΔY = R_k.y - P_base.y
            fieldSub(DictY[k], by, dy);
            #pragma unroll 4
            for(int i=0;i<4;i++){l_dX[k][i]=dx[i]; l_dY[k][i]=dy[i];}

            // Accumulation
            if(valid_k==0){
                #pragma unroll 4
                for(int i=0;i<4;i++) l_acc[k][i]=dx[i];
            } else {
                // Trouve le dernier acc valide
                fieldMul(l_acc[k-1], dx, l_acc[k]);
            }
            valid_k++;
        }

        // k=0 : pas d'addition, P_k = P_base → hash directement
        {
            uint8_t h160[20];
            uint8_t computed[20];
            if(is_any_bloom(target)){
                if(bloom_want_btc(target)){
                    getHash160_33_from_limbs((by[0]&1)?0x03:0x02, bx, h160);
                    if(bloom_check(h160, target->d_bloom_filter, target->bloom_m_bits)){
                        if(atomicCAS(&result->found,0,1)==0)
                            result->index = high_idx*(uint64_t)LOW_SIZE + 0;
                        return;
                    }
                }
                if(bloom_want_eth(target)){
                    uint8_t eth20[20]; getEthAddr_from_limbs(bx, by, eth20);
                    if(bloom_check(eth20, target->d_bloom_filter, target->bloom_m_bits)){
                        if(atomicCAS(&result->found,0,1)==0)
                            result->index = high_idx*(uint64_t)LOW_SIZE + 0;
                        return;
                    }
                }
            } else {
                if(target->type==TargetType::BTC){
                    const uint8_t par=(by[0]&1)?0x03:0x02;
                    getHash160_33_from_limbs(par,bx,computed);
                } else {
                    getEthAddr_from_limbs(bx,by,computed);
                }
                if(hash20_matches(computed,target->hash20)){
                    if(atomicCAS(&result->found,0,1)==0)
                        result->index = high_idx*(uint64_t)LOW_SIZE + 0;
                    return;
                }
            }
        }

        if(valid_k==0) continue;

        // ----------------------------------------------------------------
        // PHASE 3b : 1 seule fieldInv pour tout le lot de ΔX
        // ----------------------------------------------------------------
        // Trouve le dernier acc valide (peut ne pas être LOW_SIZE-1 si skips)
        uint64_t inv[4];
        {
            // Cherche le dernier k avec un ΔX valide
            int last_valid = 0;
            for(int k=LOW_SIZE-1;k>=1;k--){
                if((l_dX[k][0]|l_dX[k][1]|l_dX[k][2]|l_dX[k][3])!=0){
                    last_valid=k; break;
                }
            }
            fieldNormalize(l_acc[last_valid]);
            fieldInv(l_acc[last_valid], inv);
        }

        // ----------------------------------------------------------------
        // PHASE 3c : BACK-PROP + ADDITION AFFINE + HASH
        // ----------------------------------------------------------------
        // On parcourt k de LOW_SIZE-1 à 1
        // Pour chaque k valide :
        //   dx_inv[k] = inv * acc[k-1]
        //   inv       = inv * dX[k]
        //   lambda    = dY[k] * dx_inv[k]
        //   x3 = lambda² - bx - Rk.x
        //   y3 = lambda*(bx - x3) - by
        //   → hash(x3, y3)

        // Pour gérer les trous (k avec dX=0), on maintient
        // l'invariant : inv * acc[k] = 1  à chaque étape valide

        uint64_t running_inv[4];
        #pragma unroll 4
        for(int i=0;i<4;i++) running_inv[i]=inv[i];

        for(int k=LOW_SIZE-1;k>=1;k--){
            // Skip si dX==0 (cas dégénéré)
            if((l_dX[k][0]|l_dX[k][1]|l_dX[k][2]|l_dX[k][3])==0) continue;

            uint64_t dx_inv[4];
            // Cherche l'acc du step précédent valide
            // acc[k-1] : valide si k>1 et qu'il y a eu au moins un valid avant k
            // Pour k==1 (premier) : dx_inv = running_inv
            bool is_first = true;
            for(int j=k-1;j>=1;j--){
                if((l_dX[j][0]|l_dX[j][1]|l_dX[j][2]|l_dX[j][3])!=0){
                    is_first=false; break;
                }
            }

            if(!is_first){
                fieldMul(running_inv, l_acc[k-1], dx_inv);
                uint64_t t[4]; fieldMul(running_inv, l_dX[k], t);
                #pragma unroll 4
                for(int i=0;i<4;i++) running_inv[i]=t[i];
            } else {
                #pragma unroll 4
                for(int i=0;i<4;i++) dx_inv[i]=running_inv[i];
            }

            // Addition affine : P_k = P_base + R_k
            uint64_t lam[4],lam2[4],x3[4],y3[4],tmp[4];
            fieldMul(l_dY[k], dx_inv, lam);        // λ = ΔY / ΔX
            fieldSqr(lam, lam2);                    // λ²
            fieldSub(lam2, bx, x3);
            fieldSub(x3, DictX[k], x3);             // x3 = λ² - bx - Rk.x
            fieldSub(bx, x3, tmp);
            fieldMul(lam, tmp, y3);
            fieldSub(y3, by, y3);                   // y3 = λ(bx-x3) - by
            fieldNormalize(x3); fieldNormalize(y3);

            uint8_t h160[20];
            uint8_t computed[20];
            if(is_any_bloom(target)){
                if(bloom_want_btc(target)){
                    getHash160_33_from_limbs((y3[0]&1)?0x03:0x02, x3, h160);
                    if(bloom_check(h160, target->d_bloom_filter, target->bloom_m_bits)){
                        if(atomicCAS(&result->found,0,1)==0)
                            result->index = high_idx*(uint64_t)LOW_SIZE + (uint64_t)k;
                        return;
                    }
                }
                if(bloom_want_eth(target)){
                    uint8_t eth20[20]; getEthAddr_from_limbs(x3, y3, eth20);
                    if(bloom_check(eth20, target->d_bloom_filter, target->bloom_m_bits)){
                        if(atomicCAS(&result->found,0,1)==0)
                            result->index = high_idx*(uint64_t)LOW_SIZE + (uint64_t)k;
                        return;
                    }
                }
            } else {
                if(target->type==TargetType::BTC){
                    const uint8_t par=(y3[0]&1)?0x03:0x02;
                    getHash160_33_from_limbs(par,x3,computed);
                } else {
                    getEthAddr_from_limbs(x3,y3,computed);
                }
                if(hash20_matches(computed,target->hash20)){
                    if(atomicCAS(&result->found,0,1)==0)
                        result->index = high_idx*(uint64_t)LOW_SIZE + (uint64_t)k;
                    return;
                }
            }
        }
    }
}
