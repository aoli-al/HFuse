//#include "ethash_cuda_miner_kernel.h"
//
//#include "ethash_cuda_miner_kernel_globals.h"
//
//#include "cuda_helper.h"
//
//#include "fnv.cuh"
//
//#include <string.h>
//#include <stdint.h>
//
//#include "keccak.cuh"
//
//#include "dagger_shuffled.cuh"
//
//
//#define TPB_blake 128
//#define NBN 2
//
//namespace
//{
//__device__ uint64_t dd_target[1];
//static uint32_t* d_sha256_resNonces[MAX_GPUS] = {0};
//__constant__ static uint32_t __align__(8) c_target[2];
//const __constant__ uint32_t __align__(8) c_H256[8] = {0x6A09E667U, 0xBB67AE85U, 0x3C6EF372U,
//    0xA54FF53AU, 0x510E527FU, 0x9B05688CU, 0x1F83D9ABU, 0x5BE0CD19U};
//
//static const uint32_t cpu_K[64] = {0x428A2F98, 0x71374491, 0xB5C0FBCF, 0xE9B5DBA5, 0x3956C25B,
//    0x59F111F1, 0x923F82A4, 0xAB1C5ED5, 0xD807AA98, 0x12835B01, 0x243185BE, 0x550C7DC3, 0x72BE5D74,
//    0x80DEB1FE, 0x9BDC06A7, 0xC19BF174, 0xE49B69C1, 0xEFBE4786, 0x0FC19DC6, 0x240CA1CC, 0x2DE92C6F,
//    0x4A7484AA, 0x5CB0A9DC, 0x76F988DA, 0x983E5152, 0xA831C66D, 0xB00327C8, 0xBF597FC7, 0xC6E00BF3,
//    0xD5A79147, 0x06CA6351, 0x14292967, 0x27B70A85, 0x2E1B2138, 0x4D2C6DFC, 0x53380D13, 0x650A7354,
//    0x766A0ABB, 0x81C2C92E, 0x92722C85, 0xA2BFE8A1, 0xA81A664B, 0xC24B8B70, 0xC76C51A3, 0xD192E819,
//    0xD6990624, 0xF40E3585, 0x106AA070, 0x19A4C116, 0x1E376C08, 0x2748774C, 0x34B0BCB5, 0x391C0CB3,
//    0x4ED8AA4A, 0x5B9CCA4F, 0x682E6FF3, 0x748F82EE, 0x78A5636F, 0x84C87814, 0x8CC70208, 0x90BEFFFA,
//    0xA4506CEB, 0xBEF9A3F7, 0xC67178F2};
//__constant__ static uint32_t __align__(8) c_K[64];
//__constant__ static uint32_t __align__(8) c_midstate76[8];
//__constant__ static uint32_t __align__(8) c_dataEnd80[4];
//
//#define xor3b(a, b, c) (a ^ b ^ c)
//
//__device__ __forceinline__ uint32_t bsg2_0(const uint32_t x)
//{
//    return xor3b(ROTR32(x, 2), ROTR32(x, 13), ROTR32(x, 22));
//}
//
//__device__ __forceinline__ uint32_t bsg2_1(const uint32_t x)
//{
//    return xor3b(ROTR32(x, 6), ROTR32(x, 11), ROTR32(x, 25));
//}
//
//__device__ __forceinline__ uint32_t ssg2_0(const uint32_t x)
//{
//    return xor3b(ROTR32(x, 7), ROTR32(x, 18), (x >> 3));
//}
//
//__device__ __forceinline__ uint32_t ssg2_1(const uint32_t x)
//{
//    return xor3b(ROTR32(x, 17), ROTR32(x, 19), (x >> 10));
//}
//
//
//__device__ static void sha2_step1(uint32_t a, uint32_t b, uint32_t c, uint32_t& d, uint32_t e,
//    uint32_t f, uint32_t g, uint32_t& h, uint32_t in, const uint32_t Kshared)
//{
//    uint32_t t1, t2;
//    uint32_t vxandx = xandx(e, f, g);
//    uint32_t bsg21 = bsg2_1(e);
//    uint32_t bsg20 = bsg2_0(a);
//    uint32_t andorv = andor32(a, b, c);
//
//    t1 = h + bsg21 + vxandx + Kshared + in;
//    t2 = bsg20 + andorv;
//    d = d + t1;
//    h = t1 + t2;
//}
//
//__device__ static void sha2_step2(uint32_t a, uint32_t b, uint32_t c, uint32_t& d, uint32_t e,
//    uint32_t f, uint32_t g, uint32_t& h, uint32_t* in, uint32_t pc, const uint32_t Kshared)
//{
//    uint32_t t1, t2;
//
//    int pcidx1 = (pc - 2) & 0xF;
//    int pcidx2 = (pc - 7) & 0xF;
//    int pcidx3 = (pc - 15) & 0xF;
//
//    uint32_t inx0 = in[pc];
//    uint32_t inx1 = in[pcidx1];
//    uint32_t inx2 = in[pcidx2];
//    uint32_t inx3 = in[pcidx3];
//
//    uint32_t ssg21 = ssg2_1(inx1);
//    uint32_t ssg20 = ssg2_0(inx3);
//    uint32_t vxandx = xandx(e, f, g);
//    uint32_t bsg21 = bsg2_1(e);
//    uint32_t bsg20 = bsg2_0(a);
//    uint32_t andorv = andor32(a, b, c);
//
//    in[pc] = ssg21 + inx2 + ssg20 + inx0;
//
//    t1 = h + bsg21 + vxandx + Kshared + in[pc];
//    t2 = bsg20 + andorv;
//    d = d + t1;
//    h = t1 + t2;
//}
//
//__device__ static void sha256_round_body(uint32_t* in, uint32_t* state, uint32_t* const Kshared)
//{
//    uint32_t a = state[0];
//    uint32_t b = state[1];
//    uint32_t c = state[2];
//    uint32_t d = state[3];
//    uint32_t e = state[4];
//    uint32_t f = state[5];
//    uint32_t g = state[6];
//    uint32_t h = state[7];
//
//    sha2_step1(a, b, c, d, e, f, g, h, in[0], Kshared[0]);
//    sha2_step1(h, a, b, c, d, e, f, g, in[1], Kshared[1]);
//    sha2_step1(g, h, a, b, c, d, e, f, in[2], Kshared[2]);
//    sha2_step1(f, g, h, a, b, c, d, e, in[3], Kshared[3]);
//    sha2_step1(e, f, g, h, a, b, c, d, in[4], Kshared[4]);
//    sha2_step1(d, e, f, g, h, a, b, c, in[5], Kshared[5]);
//    sha2_step1(c, d, e, f, g, h, a, b, in[6], Kshared[6]);
//    sha2_step1(b, c, d, e, f, g, h, a, in[7], Kshared[7]);
//    sha2_step1(a, b, c, d, e, f, g, h, in[8], Kshared[8]);
//    sha2_step1(h, a, b, c, d, e, f, g, in[9], Kshared[9]);
//    sha2_step1(g, h, a, b, c, d, e, f, in[10], Kshared[10]);
//    sha2_step1(f, g, h, a, b, c, d, e, in[11], Kshared[11]);
//    sha2_step1(e, f, g, h, a, b, c, d, in[12], Kshared[12]);
//    sha2_step1(d, e, f, g, h, a, b, c, in[13], Kshared[13]);
//    sha2_step1(c, d, e, f, g, h, a, b, in[14], Kshared[14]);
//    sha2_step1(b, c, d, e, f, g, h, a, in[15], Kshared[15]);
//
//#pragma unroll
//    for (int i = 0; i < 3; i++)
//    {
//        sha2_step2(a, b, c, d, e, f, g, h, in, 0, Kshared[16 + 16 * i]);
//        sha2_step2(h, a, b, c, d, e, f, g, in, 1, Kshared[17 + 16 * i]);
//        sha2_step2(g, h, a, b, c, d, e, f, in, 2, Kshared[18 + 16 * i]);
//        sha2_step2(f, g, h, a, b, c, d, e, in, 3, Kshared[19 + 16 * i]);
//        sha2_step2(e, f, g, h, a, b, c, d, in, 4, Kshared[20 + 16 * i]);
//        sha2_step2(d, e, f, g, h, a, b, c, in, 5, Kshared[21 + 16 * i]);
//        sha2_step2(c, d, e, f, g, h, a, b, in, 6, Kshared[22 + 16 * i]);
//        sha2_step2(b, c, d, e, f, g, h, a, in, 7, Kshared[23 + 16 * i]);
//        sha2_step2(a, b, c, d, e, f, g, h, in, 8, Kshared[24 + 16 * i]);
//        sha2_step2(h, a, b, c, d, e, f, g, in, 9, Kshared[25 + 16 * i]);
//        sha2_step2(g, h, a, b, c, d, e, f, in, 10, Kshared[26 + 16 * i]);
//        sha2_step2(f, g, h, a, b, c, d, e, in, 11, Kshared[27 + 16 * i]);
//        sha2_step2(e, f, g, h, a, b, c, d, in, 12, Kshared[28 + 16 * i]);
//        sha2_step2(d, e, f, g, h, a, b, c, in, 13, Kshared[29 + 16 * i]);
//        sha2_step2(c, d, e, f, g, h, a, b, in, 14, Kshared[30 + 16 * i]);
//        sha2_step2(b, c, d, e, f, g, h, a, in, 15, Kshared[31 + 16 * i]);
//    }
//
//    state[0] += a;
//    state[1] += b;
//    state[2] += c;
//    state[3] += d;
//    state[4] += e;
//    state[5] += f;
//    state[6] += g;
//    state[7] += h;
//}
//
//__device__ static void sha256_round_last(uint32_t* in, uint32_t* state, uint32_t* const Kshared)
//{
//    uint32_t a = state[0];
//    uint32_t b = state[1];
//    uint32_t c = state[2];
//    uint32_t d = state[3];
//    uint32_t e = state[4];
//    uint32_t f = state[5];
//    uint32_t g = state[6];
//    uint32_t h = state[7];
//
//    sha2_step1(a, b, c, d, e, f, g, h, in[0], Kshared[0]);
//    sha2_step1(h, a, b, c, d, e, f, g, in[1], Kshared[1]);
//    sha2_step1(g, h, a, b, c, d, e, f, in[2], Kshared[2]);
//    sha2_step1(f, g, h, a, b, c, d, e, in[3], Kshared[3]);
//    sha2_step1(e, f, g, h, a, b, c, d, in[4], Kshared[4]);
//    sha2_step1(d, e, f, g, h, a, b, c, in[5], Kshared[5]);
//    sha2_step1(c, d, e, f, g, h, a, b, in[6], Kshared[6]);
//    sha2_step1(b, c, d, e, f, g, h, a, in[7], Kshared[7]);
//    sha2_step1(a, b, c, d, e, f, g, h, in[8], Kshared[8]);
//    sha2_step1(h, a, b, c, d, e, f, g, in[9], Kshared[9]);
//    sha2_step1(g, h, a, b, c, d, e, f, in[10], Kshared[10]);
//    sha2_step1(f, g, h, a, b, c, d, e, in[11], Kshared[11]);
//    sha2_step1(e, f, g, h, a, b, c, d, in[12], Kshared[12]);
//    sha2_step1(d, e, f, g, h, a, b, c, in[13], Kshared[13]);
//    sha2_step1(c, d, e, f, g, h, a, b, in[14], Kshared[14]);
//    sha2_step1(b, c, d, e, f, g, h, a, in[15], Kshared[15]);
//
//#pragma unroll 2
//    for (int i = 0; i < 2; i++)
//    {
//        sha2_step2(a, b, c, d, e, f, g, h, in, 0, Kshared[16 + 16 * i]);
//        sha2_step2(h, a, b, c, d, e, f, g, in, 1, Kshared[17 + 16 * i]);
//        sha2_step2(g, h, a, b, c, d, e, f, in, 2, Kshared[18 + 16 * i]);
//        sha2_step2(f, g, h, a, b, c, d, e, in, 3, Kshared[19 + 16 * i]);
//        sha2_step2(e, f, g, h, a, b, c, d, in, 4, Kshared[20 + 16 * i]);
//        sha2_step2(d, e, f, g, h, a, b, c, in, 5, Kshared[21 + 16 * i]);
//        sha2_step2(c, d, e, f, g, h, a, b, in, 6, Kshared[22 + 16 * i]);
//        sha2_step2(b, c, d, e, f, g, h, a, in, 7, Kshared[23 + 16 * i]);
//        sha2_step2(a, b, c, d, e, f, g, h, in, 8, Kshared[24 + 16 * i]);
//        sha2_step2(h, a, b, c, d, e, f, g, in, 9, Kshared[25 + 16 * i]);
//        sha2_step2(g, h, a, b, c, d, e, f, in, 10, Kshared[26 + 16 * i]);
//        sha2_step2(f, g, h, a, b, c, d, e, in, 11, Kshared[27 + 16 * i]);
//        sha2_step2(e, f, g, h, a, b, c, d, in, 12, Kshared[28 + 16 * i]);
//        sha2_step2(d, e, f, g, h, a, b, c, in, 13, Kshared[29 + 16 * i]);
//        sha2_step2(c, d, e, f, g, h, a, b, in, 14, Kshared[30 + 16 * i]);
//        sha2_step2(b, c, d, e, f, g, h, a, in, 15, Kshared[31 + 16 * i]);
//    }
//
//    sha2_step2(a, b, c, d, e, f, g, h, in, 0, Kshared[16 + 16 * 2]);
//    sha2_step2(h, a, b, c, d, e, f, g, in, 1, Kshared[17 + 16 * 2]);
//    sha2_step2(g, h, a, b, c, d, e, f, in, 2, Kshared[18 + 16 * 2]);
//    sha2_step2(f, g, h, a, b, c, d, e, in, 3, Kshared[19 + 16 * 2]);
//    sha2_step2(e, f, g, h, a, b, c, d, in, 4, Kshared[20 + 16 * 2]);
//    sha2_step2(d, e, f, g, h, a, b, c, in, 5, Kshared[21 + 16 * 2]);
//    sha2_step2(c, d, e, f, g, h, a, b, in, 6, Kshared[22 + 16 * 2]);
//    sha2_step2(b, c, d, e, f, g, h, a, in, 7, Kshared[23 + 16 * 2]);
//    sha2_step2(a, b, c, d, e, f, g, h, in, 8, Kshared[24 + 16 * 2]);
//    sha2_step2(h, a, b, c, d, e, f, g, in, 9, Kshared[25 + 16 * 2]);
//    sha2_step2(g, h, a, b, c, d, e, f, in, 10, Kshared[26 + 16 * 2]);
//    sha2_step2(f, g, h, a, b, c, d, e, in, 11, Kshared[27 + 16 * 2]);
//    sha2_step2(e, f, g, h, a, b, c, d, in, 12, Kshared[28 + 16 * 2]);
//    sha2_step2(d, e, f, g, h, a, b, c, in, 13, Kshared[29 + 16 * 2]);
//
//    state[6] += g;
//    state[7] += h;
//}
//
//__device__ __forceinline__ uint64_t cuda_swab32ll(uint64_t x)
//{
//    return MAKE_ULONGLONG(cuda_swab32(_LODWORD(x)), cuda_swab32(_HIDWORD(x)));
//}
//
//
//__global__
//    /*__launch_bounds__(256,3)*/
//    void
//    sha256d_gpu_hash_shared(const uint32_t threads, const uint32_t startNonce, uint32_t* resNonces)
//{
//    for (int i = 0; i < 40; i++)
//    {
//        const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x) * 40 + i;
//
//        __shared__ uint32_t s_K[64 * 4];
//        // s_K[thread & 63] = c_K[thread & 63];
//        if (threadIdx.x < 64U)
//            s_K[threadIdx.x] = c_K[threadIdx.x];
//
//        if (thread < threads)
//        {
//            const uint32_t nonce = startNonce + thread;
//
//            uint32_t dat[16];
//            AS_UINT2(dat) = AS_UINT2(c_dataEnd80);
//            dat[2] = c_dataEnd80[2];
//            dat[3] = nonce;
//            dat[4] = 0x80000000;
//            dat[15] = 0x280;
//#pragma unroll 10
//            for (int i = 5; i < 15; i++)
//                dat[i] = 0;
//
//            uint32_t buf[8];
//#pragma unroll 4
//            for (int i = 0; i < 8; i += 2)
//                AS_UINT2(&buf[i]) = AS_UINT2(&c_midstate76[i]);
//            // for (int i=0; i<8; i++) buf[i] = c_midstate76[i];
//
//            sha256_round_body(dat, buf, s_K);
//
//            // second sha256
//
//#pragma unroll 8
//            for (int i = 0; i < 8; i++)
//                dat[i] = buf[i];
//            dat[8] = 0x80000000;
//#pragma unroll 6
//            for (int i = 9; i < 15; i++)
//                dat[i] = 0;
//            dat[15] = 0x100;
//
//#pragma unroll 8
//            for (int i = 0; i < 8; i++)
//                buf[i] = c_H256[i];
//
//            sha256_round_last(dat, buf, s_K);
//
//            // valid nonces
//            uint64_t high = cuda_swab32ll(((uint64_t*)buf)[3]);
//            if (high <= c_target[0])
//            {
//                // printf("%08x %08x - %016llx %016llx - %08x %08x\n", buf[7], buf[6], high, d_target[0], c_target[1], c_target[0]);
//                resNonces[1] = atomicExch(resNonces, nonce);
//                // d_target[0] = high;
//            }
//        }
//    }
//}
//
//static uint32_t* d_resNonces_blake[MAX_GPUS];
//
//__device__ uint64_t d_blake_data[10];
//
//#define AS_U32(addr) *((uint32_t*)(addr))
//
//static __constant__ const int8_t blake2b_sigma_blake[12][16] = {
//    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
//    {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3},
//    {11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4},
//    {7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8},
//    {9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13},
//    {2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9},
//    {12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11},
//    {13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10},
//    {6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5},
//    {10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0},
//    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
//    {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3}};
//
//// host mem align
//#define A 64
////
//// extern "C" void blake2b_hash(void *output, const void *input)
////{
////    uint8_t _ALIGN(A) hash[32];
////    blake2b_ctx ctx;
////
////    blake2b_init(&ctx, 32, NULL, 0);
////    blake2b_update(&ctx, input, 80);
////    blake2b_final(&ctx, hash);
////
////    memcpy(output, hash, 32);
////}
//
//// ----------------------------------------------------------------
//
//__device__ __forceinline__ static void G_b(const int r, const int i, uint64_t& a, uint64_t& b,
//    uint64_t& c, uint64_t& d, uint64_t const m[16])
//{
//    a = a + b + m[blake2b_sigma_blake[r][2 * i]];
//    ((uint2*)&d)[0] = SWAPUINT2(((uint2*)&d)[0] ^ ((uint2*)&a)[0]);
//    c = c + d;
//    ((uint2*)&b)[0] = ROR24(((uint2*)&b)[0] ^ ((uint2*)&c)[0]);
//    a = a + b + m[blake2b_sigma_blake[r][2 * i + 1]];
//    ((uint2*)&d)[0] = ROR16(((uint2*)&d)[0] ^ ((uint2*)&a)[0]);
//    c = c + d;
//    ((uint2*)&b)[0] = ROR2(((uint2*)&b)[0] ^ ((uint2*)&c)[0], 63U);
//}
//
//#define ROUND(r) \
//	G_b(r, 0, v[0], v[4], v[8], v[12], m); \
//	G_b(r, 1, v[1], v[5], v[9], v[13], m); \
//	G_b(r, 2, v[2], v[6], v[10], v[14], m); \
//	G_b(r, 3, v[3], v[7], v[11], v[15], m); \
//	G_b(r, 4, v[0], v[5], v[10], v[15], m); \
//	G_b(r, 5, v[1], v[6], v[11], v[12], m); \
//	G_b(r, 6, v[2], v[7], v[8], v[13], m); \
//	G_b(r, 7, v[3], v[4], v[9], v[14], m);
//
//__global__ void blake2b_gpu_hash(
//    const uint32_t threads, const uint32_t startNonce, uint32_t* resNonce, const uint2 target2)
//{
//    for (int i = 0; i < 80; i++)
//    {
//        const uint32_t nonce = (blockDim.x * blockIdx.x + threadIdx.x) * 80 + i + startNonce;
//
//        uint64_t m[16];
//
//        m[0] = d_blake_data[0];
//        m[1] = d_blake_data[1];
//        m[2] = d_blake_data[2];
//        m[3] = d_blake_data[3];
//        m[4] = d_blake_data[4];
//        m[5] = d_blake_data[5];
//        m[6] = d_blake_data[6];
//        m[7] = d_blake_data[7];
//        m[8] = d_blake_data[8];
//        ((uint32_t*)m)[18] = AS_U32(&d_blake_data[9]);
//        ((uint32_t*)m)[19] = nonce;
//
//        m[10] = m[11] = 0;
//        m[12] = m[13] = 0;
//        m[14] = m[15] = 0;
//
//        uint64_t v[16] = {0x6a09e667f2bdc928, 0xbb67ae8584caa73b, 0x3c6ef372fe94f82b,
//            0xa54ff53a5f1d36f1, 0x510e527fade682d1, 0x9b05688c2b3e6c1f, 0x1f83d9abfb41bd6b,
//            0x5be0cd19137e2179, 0x6a09e667f3bcc908, 0xbb67ae8584caa73b, 0x3c6ef372fe94f82b,
//            0xa54ff53a5f1d36f1, 0x510e527fade68281, 0x9b05688c2b3e6c1f, 0xe07c265404be4294,
//            0x5be0cd19137e2179};
//
//        ROUND(0);
//        ROUND(1);
//        ROUND(2);
//        ROUND(3);
//        ROUND(4);
//        ROUND(5);
//        ROUND(6);
//        ROUND(7);
//        ROUND(8);
//        ROUND(9);
//        ROUND(10);
//        ROUND(11);
//
//        uint2 last = vectorize(v[3] ^ v[11] ^ 0xa54ff53a5f1d36f1);
//        if (last.y <= target2.y && last.x <= target2.x)
//        {
//            resNonce[1] = resNonce[0];
//            resNonce[0] = nonce;
//        }
//    }
//}
//
//#define TPB 128
//#define NBN 2
//
//static uint32_t *d_resNonces[MAX_GPUS];
//
//__device__ uint64_t d_data2[10];
//
//static __constant__ const int8_t blake2b_sigma[12][16] = {
//    { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15 } ,
//    { 14, 10, 4,  8,  9,  15, 13, 6,  1,  12, 0,  2,  11, 7,  5,  3  } ,
//    { 11, 8,  12, 0,  5,  2,  15, 13, 10, 14, 3,  6,  7,  1,  9,  4  } ,
//    { 7,  9,  3,  1,  13, 12, 11, 14, 2,  6,  5,  10, 4,  0,  15, 8  } ,
//    { 9,  0,  5,  7,  2,  4,  10, 15, 14, 1,  11, 12, 6,  8,  3,  13 } ,
//    { 2,  12, 6,  10, 0,  11, 8,  3,  4,  13, 7,  5,  15, 14, 1,  9  } ,
//    { 12, 5,  1,  15, 14, 13, 4,  10, 0,  7,  6,  3,  9,  2,  8,  11 } ,
//    { 13, 11, 7,  14, 12, 1,  3,  9,  5,  0,  15, 4,  8,  6,  2,  10 } ,
//    { 6,  15, 14, 9,  11, 3,  0,  8,  12, 2,  13, 7,  1,  4,  10, 5  } ,
//    { 10, 2,  8,  4,  7,  6,  1,  5,  15, 11, 9,  14, 3,  12, 13, 0  } ,
//    { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15 } ,
//    { 14, 10, 4,  8,  9,  15, 13, 6,  1,  12, 0,  2,  11, 7,  5,  3  }
//};
//
//// host mem align
//#define A 64
////
////extern "C" void sia_blake2b_hash(void *output, const void *input)
////{
////    uint8_t _ALIGN(A) hash[32];
////    blake2b_ctx ctx;
////
////    blake2b_init(&ctx, 32, NULL, 0);
////    blake2b_update(&ctx, input, 80);
////    blake2b_final(&ctx, hash);
////
////    memcpy(output, hash, 32);
////}
////
//// ----------------------------------------------------------------
//
//__device__ __forceinline__
//static void G(const int r, const int i, uint64_t &a, uint64_t &b, uint64_t &c, uint64_t &d, uint64_t const m[16])
//{
//    a = a + b + m[ blake2b_sigma[r][2*i] ];
//    ((uint2*)&d)[0] = SWAPUINT2( ((uint2*)&d)[0] ^ ((uint2*)&a)[0] );
//    c = c + d;
//    ((uint2*)&b)[0] = ROR24( ((uint2*)&b)[0] ^ ((uint2*)&c)[0] );
//    a = a + b + m[ blake2b_sigma[r][2*i+1] ];
//    ((uint2*)&d)[0] = ROR16( ((uint2*)&d)[0] ^ ((uint2*)&a)[0] );
//    c = c + d;
//    ((uint2*)&b)[0] = ROR2( ((uint2*)&b)[0] ^ ((uint2*)&c)[0], 63U);
//}
//
//#define ROUND(r) \
//	G(r, 0, v[0], v[4], v[ 8], v[12], m); \
//	G(r, 1, v[1], v[5], v[ 9], v[13], m); \
//	G(r, 2, v[2], v[6], v[10], v[14], m); \
//	G(r, 3, v[3], v[7], v[11], v[15], m); \
//	G(r, 4, v[0], v[5], v[10], v[15], m); \
//	G(r, 5, v[1], v[6], v[11], v[12], m); \
//	G(r, 6, v[2], v[7], v[ 8], v[13], m); \
//	G(r, 7, v[3], v[4], v[ 9], v[14], m);
//
//// simplified for the last round
//__device__ __forceinline__
//static void H(const int r, const int i, uint64_t &a, uint64_t &b, uint64_t &c, uint64_t &d, uint64_t const m[16])
//{
//    a = a + b + m[ blake2b_sigma[r][2*i] ];
//    ((uint2*)&d)[0] = SWAPUINT2( ((uint2*)&d)[0] ^ ((uint2*)&a)[0] );
//    c = c + d;
//    ((uint2*)&b)[0] = ROR24( ((uint2*)&b)[0] ^ ((uint2*)&c)[0] );
//    a = a + b + m[ blake2b_sigma[r][2*i+1] ];
//    ((uint2*)&d)[0] = ROR16( ((uint2*)&d)[0] ^ ((uint2*)&a)[0] );
//    c = c + d;
//}
//
//// we only check v[0] and v[8]
//#define ROUND_F(r) \
//	G(r, 0, v[0], v[4], v[ 8], v[12], m); \
//	G(r, 1, v[1], v[5], v[ 9], v[13], m); \
//	G(r, 2, v[2], v[6], v[10], v[14], m); \
//	G(r, 3, v[3], v[7], v[11], v[15], m); \
//	G(r, 4, v[0], v[5], v[10], v[15], m); \
//	G(r, 5, v[1], v[6], v[11], v[12], m); \
//	H(r, 6, v[2], v[7], v[ 8], v[13], m);
//
//__global__
////__launch_bounds__(128, 8) /* to force 64 regs */
//void sia_blake2b_gpu_hash(const uint32_t threads, const uint32_t startNonce, uint32_t *resNonce, const uint2 target2)
//{
//
//    for (int i = 0; i < 80; i++) {
//        const uint32_t nonce = (blockDim.x * blockIdx.x + threadIdx.x) * 80 + i + startNonce;
//        __shared__ uint64_t s_target;
//        if (!threadIdx.x) s_target = devectorize(target2);
//        uint64_t m[16];
//
//        m[0] = d_data2[0];
//        m[1] = d_data2[1];
//        m[2] = d_data2[2];
//        m[3] = d_data2[3];
//        m[4] = d_data2[4] | nonce;
//        m[5] = d_data2[5];
//        m[6] = d_data2[6];
//        m[7] = d_data2[7];
//        m[8] = d_data2[8];
//        m[9] = d_data2[9];
//
//        m[10] = m[11] = 0;
//        m[12] = m[13] = m[14] = m[15] = 0;
//
//        uint64_t v[16] = {
//            0x6a09e667f2bdc928, 0xbb67ae8584caa73b, 0x3c6ef372fe94f82b, 0xa54ff53a5f1d36f1,
//            0x510e527fade682d1, 0x9b05688c2b3e6c1f, 0x1f83d9abfb41bd6b, 0x5be0cd19137e2179,
//            0x6a09e667f3bcc908, 0xbb67ae8584caa73b, 0x3c6ef372fe94f82b, 0xa54ff53a5f1d36f1,
//            0x510e527fade68281, 0x9b05688c2b3e6c1f, 0xe07c265404be4294, 0x5be0cd19137e2179
//        };
//
//        ROUND( 0 );
//        ROUND( 1 );
//        ROUND( 2 );
//        ROUND( 3 );
//        ROUND( 4 );
//        ROUND( 5 );
//        ROUND( 6 );
//        ROUND( 7 );
//        ROUND( 8 );
//        ROUND( 9 );
//        ROUND( 10 );
//        ROUND_F( 11 );
//
//        uint64_t h64 = cuda_swab64(0x6a09e667f2bdc928 ^ v[0] ^ v[8]);
//        if (h64 <= s_target) {
//            resNonce[1] = resNonce[0];
//            resNonce[0] = nonce;
//            s_target = h64;
//        }
//    }
//    // if (!nonce) printf("%016lx ", s_target);
//}
//
//__global__ void ethash_search4(volatile Search_results* g_output, uint64_t start_nonce)
//{
//    uint32_t const gid = blockIdx.x * blockDim.x + threadIdx.x;
//    uint2 mix[4];
//    uint64_t nonce = start_nonce + gid;
//    uint2* mix_hash = mix;
//    bool result = false;
//
//    uint2 state[12];
//
//    state[4] = vectorize(nonce);
//
//    keccak_f1600_init(state);
//
//    // Threads work together in this phase in groups of 8.
//    const int thread_id = threadIdx.x & (THREADS_PER_HASH - 1);
//    const int mix_idx = thread_id & 3;
//
//    for (int i = 0; i < THREADS_PER_HASH; i += _PARALLEL_HASH)
//    {
//        uint4 mix[_PARALLEL_HASH];
//        uint32_t offset[_PARALLEL_HASH];
//        uint32_t init0[_PARALLEL_HASH];
//
//        // share init among threads
//        for (int p = 0; p < _PARALLEL_HASH; p++)
//        {
//            uint2 shuffle[8];
//            for (int j = 0; j < 8; j++)
//            {
//                shuffle[j].x = SHFL(state[j].x, i + p, THREADS_PER_HASH);
//                shuffle[j].y = SHFL(state[j].y, i + p, THREADS_PER_HASH);
//            }
//            switch (mix_idx)
//            {
//            case 0:
//                mix[p] = vectorize2(shuffle[0], shuffle[1]);
//                break;
//            case 1:
//                mix[p] = vectorize2(shuffle[2], shuffle[3]);
//                break;
//            case 2:
//                mix[p] = vectorize2(shuffle[4], shuffle[5]);
//                break;
//            case 3:
//                mix[p] = vectorize2(shuffle[6], shuffle[7]);
//                break;
//            }
//            init0[p] = SHFL(shuffle[0].x, 0, THREADS_PER_HASH);
//        }
//
//        for (uint32_t a = 0; a < ACCESSES; a += 4)
//        {
//            int t = bfe(a, 2u, 3u);
//
//            for (uint32_t b = 0; b < 4; b++)
//            {
//                for (int p = 0; p < _PARALLEL_HASH; p++)
//                {
//                    offset[p] = fnv(init0[p] ^ (a + b), ((uint32_t*)&mix[p])[b]) % d_dag_size;
//                    offset[p] = SHFL(offset[p], t, THREADS_PER_HASH);
//                    mix[p] = fnv4(mix[p], d_dag[offset[p]].uint4s[thread_id]);
//                }
//            }
//        }
//
//        for (int p = 0; p < _PARALLEL_HASH; p++)
//        {
//            uint2 shuffle[4];
//            uint32_t thread_mix = fnv_reduce(mix[p]);
//
//            // update mix across threads
//            shuffle[0].x = SHFL(thread_mix, 0, THREADS_PER_HASH);
//            shuffle[0].y = SHFL(thread_mix, 1, THREADS_PER_HASH);
//            shuffle[1].x = SHFL(thread_mix, 2, THREADS_PER_HASH);
//            shuffle[1].y = SHFL(thread_mix, 3, THREADS_PER_HASH);
//            shuffle[2].x = SHFL(thread_mix, 4, THREADS_PER_HASH);
//            shuffle[2].y = SHFL(thread_mix, 5, THREADS_PER_HASH);
//            shuffle[3].x = SHFL(thread_mix, 6, THREADS_PER_HASH);
//            shuffle[3].y = SHFL(thread_mix, 7, THREADS_PER_HASH);
//
//            if ((i + p) == thread_id)
//            {
//                // move mix into state:
//                state[8] = shuffle[0];
//                state[9] = shuffle[1];
//                state[10] = shuffle[2];
//                state[11] = shuffle[3];
//            }
//        }
//    }
//
//    // keccak_256(keccak_512(header..nonce) .. mix);
//    if (!(cuda_swab64(keccak_f1600_final(state)) > d_target)) {
//        mix_hash[0] = state[8];
//        mix_hash[1] = state[9];
//        mix_hash[2] = state[10];
//        mix_hash[3] = state[11];
//        return;
//    }
//
//    uint32_t index = atomicInc((uint32_t*)&g_output->count, 0xffffffff);
//    if (index >= MAX_SEARCH_RESULTS)
//        return;
//    g_output->result[index].gid = gid;
//    g_output->result[index].mix[0] = mix[0].x;
//    g_output->result[index].mix[1] = mix[0].y;
//    g_output->result[index].mix[2] = mix[1].x;
//    g_output->result[index].mix[3] = mix[1].y;
//    g_output->result[index].mix[4] = mix[2].x;
//    g_output->result[index].mix[5] = mix[2].y;
//    g_output->result[index].mix[6] = mix[3].x;
//    g_output->result[index].mix[7] = mix[3].y;
//}
//
//template <typename scalar_t0, typename accscalar_t1, typename output_t60, typename input_t61, typename IndexType62, int ADims63, int PDims64, int BDims65, at::native::CUDAHistogramMemoryType MemoryType66 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op67, template <typename T> class VarTransform31, typename input_scalar_t32, typename stat_scalar_t33, typename stat_accscalar_t34, typename index_t35>
//__attribute__((global)) void FUNC(const int nthreads2, const scalar_t0 *bottom_data3, const int num4, const int channels5, const int height6, const int width7, const int pooled_height8, const int pooled_width9, const int kernel_h10, const int kernel_w11, const int stride_h12, const int stride_w13, const int pad_h14, const int pad_w15, const int dilation_h16, const int dilation_w17, scalar_t0 *top_data18, int64_t *top_mask19, TensorInfo<output_t60, IndexType62> a68, TensorInfo<output_t60, IndexType62> p69, TensorInfo<input_t61, IndexType62> b70, int nbins71, input_t61 minvalue72, input_t61 maxvalue73, IndexType62 totalElements74, Op67 getOp75, const PackedTensorAccessor<input_scalar_t32, 3, RestrictPtrTraits, index_t35> input36, const stat_accscalar_t34 epsilon37, const stat_accscalar_t34 momentum38, PackedTensorAccessor<stat_scalar_t33, 1, RestrictPtrTraits, index_t35> running_mean39, PackedTensorAccessor<stat_scalar_t33, 1, RestrictPtrTraits, index_t35> running_var40, PackedTensorAccessor<stat_accscalar_t34, 1, RestrictPtrTraits, index_t35> save_mean41, PackedTensorAccessor<stat_accscalar_t34, 1, RestrictPtrTraits, index_t35> save_transformed_var42)
//{
//    if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=256 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 768)) goto label_1;
//    unsigned int blockDim_x_2;
//    blockDim_x_2 = 512;
//    unsigned int threadIdx_x_2;
//    threadIdx_x_2 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 256) % 512;
//    unsigned int blockDim_y_2;
//    blockDim_y_2 = 1;
//    unsigned int threadIdx_y_2;
//    threadIdx_y_2 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 256) / 512 % 1;
//    unsigned int blockDim_z_2;
//    blockDim_z_2 = 1;
//    unsigned int threadIdx_z_2;
//    threadIdx_z_2 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 256) / 512;
//    extern unsigned char my_smem76[] __attribute__((shared));
//    output_t60 *smem77;
//    smem77 = nullptr;
//    smem77 = reinterpret_cast<output_t60 *>(my_smem76);
//    for (IndexType62 i = threadIdx_x_2; i < a68.sizes[0]; i += blockDim_x_2) {
//        smem77[i] = 0;
//    }
//    label_1:;
//    __syncthreads();
//    if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=256 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 768)) goto label_2;
//    for (IndexType62 linearIndex = blockIdx.x * blockDim_x_2 + threadIdx_x_2; linearIndex < totalElements74; linearIndex += gridDim.x * blockDim_x_2) {
//        IndexType62 bOffset78;
//        bOffset78 = IndexToOffset<input_t61, IndexType62, BDims65>::get(linearIndex, b70);
//        input_t61 bVal79;
//        bVal79 = b70.data[bOffset78];
//        if (bVal79 >= minvalue72 && bVal79 <= maxvalue73) {
//            IndexType62 bin80;
//            bin80 = getBin<input_t61, IndexType62>(bVal79, minvalue72, maxvalue73, nbins71);
//            atomicAdd(&smem77[bin80], getOp75(linearIndex));
//        }
//    }
//    label_2:;
//    if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 256)) goto label_0;
//    unsigned int blockDim_x_0;
//    blockDim_x_0 = 256;
//    unsigned int threadIdx_x_0;
//    threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 256;
//    unsigned int blockDim_y_0;
//    blockDim_y_0 = 1;
//    unsigned int threadIdx_y_0;
//    threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256 % 1;
//    unsigned int blockDim_z_0;
//    blockDim_z_0 = 1;
//    unsigned int threadIdx_z_0;
//    threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256;
//    for (int index = blockIdx.x * blockDim_x_0 + threadIdx_x_0; index < (nthreads2); index += blockDim_x_0 * gridDim.x) {
//        int pw20;
//        pw20 = index % pooled_width9;
//        int ph21;
//        ph21 = (index / pooled_width9) % pooled_height8;
//        int c22;
//        c22 = (index / pooled_width9 / pooled_height8) % channels5;
//        int n23;
//        n23 = index / pooled_width9 / pooled_height8 / channels5;
//        int hstart24;
//        hstart24 = ph21 * stride_h12 - pad_h14;
//        int wstart25;
//        wstart25 = pw20 * stride_w13 - pad_w15;
//        int hend26;
//        hend26 = min(hstart24 + (kernel_h10 - 1) * dilation_h16 + 1, height6);
//        int wend27;
//        wend27 = min(wstart25 + (kernel_w11 - 1) * dilation_w17 + 1, width7);
//        while (hstart24 < 0)
//            hstart24 += dilation_h16;
//        while (wstart25 < 0)
//            wstart25 += dilation_w17;
//        accscalar_t1 maxval28;
//        maxval28 = at::numeric_limits<accscalar_t1>::lower_bound();
//        int maxidx29;
//        maxidx29 = hstart24 * width7 + wstart25;
//        bottom_data3 += (n23 * channels5 + c22) * height6 * width7;
//        for (int h = hstart24; h < hend26; h += dilation_h16) {
//            for (int w = wstart25; w < wend27; w += dilation_w17) {
//                scalar_t0 val30;
//                val30 = bottom_data3[h * width7 + w];
//                if ((ScalarConvert<scalar_t0, accscalar_t1>::to(val30) > maxval28) || THCNumerics<scalar_t0>::isnan(val30)) {
//                    maxidx29 = h * width7 + w;
//                    maxval28 = ScalarConvert<scalar_t0, accscalar_t1>::to(val30);
//                }
//            }
//        }
//        top_data18[index] = ScalarConvert<scalar_t0, accscalar_t1>::to(maxval28);
//        top_mask19[index] = maxidx29;
//    }
//    label_0:;
//    if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=768 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_4;
//    unsigned int blockDim_x_1;
//    blockDim_x_1 = 16;
//    unsigned int threadIdx_x_1;
//    threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 768) % 16;
//    unsigned int blockDim_y_1;
//    blockDim_y_1 = 16;
//    unsigned int threadIdx_y_1;
//    threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 768) / 16 % 16;
//    unsigned int blockDim_z_1;
//    blockDim_z_1 = 1;
//    unsigned int threadIdx_z_1;
//    threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 768) / 256;
//    static int shared_n43[160] __attribute__((shared));
//    int plane44;
//    plane44 = blockIdx.x;
//    int N45;
//    N45 = input36.size(0) * input36.size(2);
//    int tid46;
//    tid46 = threadIdx_x_1 + threadIdx_y_1 * blockDim_x_1;
//    stat_accscalar_t34 *shared_avg_var47;
//    shared_avg_var47 = (stat_accscalar_t34 *)&shared_n43[WARP_SIZE];
//    stat_accscalar_t34 avg48;
//    avg48 = 0;
//    stat_accscalar_t34 var_n49;
//    var_n49 = 0;
//    int n50;
//    n50 = 0;
//    for (int batch = threadIdx_y_1; batch < input36.size(0); batch += blockDim_y_1) {
//        for (int x = threadIdx_x_1; x < input36.size(2); x += blockDim_x_1) {
//            stat_accscalar_t34 v51;
//            v51 = input36[batch][plane44][x];
//            stat_accscalar_t34 d152;
//            d152 = v51 - avg48;
//            n50++;
//            avg48 += d152 / n50;
//            var_n49 += d152 * (v51 - avg48);
//        }
//    }
//    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
//        stat_accscalar_t34 o_avg53;
//        o_avg53 = WARP_SHFL_XOR(avg48, 1 << i, WARP_SIZE);
//        int o_n54;
//        o_n54 = WARP_SHFL_XOR(n50, 1 << i, WARP_SIZE);
//        stat_accscalar_t34 factor55;
//        factor55 = 1. / fmaxf(1., n50 + o_n54);
//        var_n49 += WARP_SHFL_XOR(var_n49, 1 << i, WARP_SIZE) + (avg48 - o_avg53) * (avg48 - o_avg53) * n50 * o_n54 * factor55;
//        avg48 = (n50 * avg48 + o_n54 * o_avg53) * factor55;
//        n50 += o_n54;
//    }
//    label_4:;
//    __syncthreads();
//    if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=256 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 768)) goto label_3;
//    for (IndexType62 i = threadIdx_x_2; i < a68.sizes[0]; i += blockDim_x_2) {
//        IndexType62 aOffset81;
//        aOffset81 = IndexToOffset<output_t60, IndexType62, ADims63>::get(i, a68);
//        atomicAdd(&a68.data[aOffset81], smem77[i]);
//    }
//    label_3:;
//    if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=768 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_5;
//    if (tid46 % WARP_SIZE == 0) {
//        shared_n43[tid46 / WARP_SIZE] = n50;
//        shared_avg_var47[tid46 / WARP_SIZE * 2] = avg48;
//        shared_avg_var47[tid46 / WARP_SIZE * 2 + 1] = var_n49;
//    }
//    label_5:;
//    __syncthreads();
//    if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=768 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)) goto label_6;
//    if (tid46 < WARP_SIZE) {
//        n50 = (tid46 < blockDim_x_1 * blockDim_y_1 / WARP_SIZE ? shared_n43[tid46] : 0);
//        avg48 = (tid46 < blockDim_x_1 * blockDim_y_1 / WARP_SIZE ? shared_avg_var47[2 * tid46] : stat_accscalar_t34(0));
//        var_n49 = (tid46 < blockDim_x_1 * blockDim_y_1 / WARP_SIZE ? shared_avg_var47[2 * tid46 + 1] : stat_accscalar_t34(0));
//    }
//    for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
//        stat_accscalar_t34 o_avg56;
//        o_avg56 = WARP_SHFL_XOR(avg48, 1 << i, WARP_SIZE);
//        int o_n57;
//        o_n57 = WARP_SHFL_XOR(n50, 1 << i, WARP_SIZE);
//        stat_accscalar_t34 factor58;
//        factor58 = 1. / fmaxf(1., n50 + o_n57);
//        var_n49 += WARP_SHFL_XOR(var_n49, 1 << i, WARP_SIZE) + (avg48 - o_avg56) * (avg48 - o_avg56) * n50 * o_n57 * factor58;
//        avg48 = (n50 * avg48 + o_n57 * o_avg56) * factor58;
//        n50 += o_n57;
//    }
//    if (tid46 == 0) {
//        if (save_mean41.data() != __null) {
//            save_mean41[plane44] = avg48;
//        }
//        if (save_transformed_var42.data() != __null) {
//            save_transformed_var42[plane44] = VarTransform31<stat_accscalar_t34>({})(var_n49 / N45, epsilon37);
//        }
//        if (running_mean39.data() != __null) {
//            running_mean39[plane44] = static_cast<stat_scalar_t33>((1 - momentum38) * running_mean39[plane44] + momentum38 * avg48);
//        }
//        if (running_var40.data() != __null) {
//            stat_accscalar_t34 unbiasedVar59;
//            unbiasedVar59 = var_n49 / (N45 - 1);
//            running_var40[plane44] = static_cast<stat_scalar_t33>((1 - momentum38) * running_var40[plane44] + momentum38 * unbiasedVar59);
//        }
//    }
//    label_6:;
//}
//
//__attribute__((global)) void FUNC(const uint32_t threads9, const uint32_t startNonce10, uint32_t *resNonce11, const uint2 target212, const uint32_t threads0, const uint32_t startNonce1, uint32_t *resNonces2, volatile Search_results *g_output17, uint64_t start_nonce18)
//{
//    if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 128)) goto label_0;
//    unsigned int blockDim_x_1;
//    blockDim_x_1 = 128;
//    unsigned int threadIdx_x_1;
//    threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 128;
//    unsigned int blockDim_y_1;
//    blockDim_y_1 = 1;
//    unsigned int threadIdx_y_1;
//    threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128 % 1;
//    unsigned int blockDim_z_1;
//    blockDim_z_1 = 1;
//    unsigned int threadIdx_z_1;
//    threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128;
//    for (int i = 0; i < 80; i++) {
//        uint32_t nonce13;
//        nonce13 = (blockDim_x_1 * blockIdx.x + threadIdx_x_1) * 80 + i + startNonce10;
//        uint64_t m14[16];
//        m14[0] = d_blake_data[0];
//        m14[1] = d_blake_data[1];
//        m14[2] = d_blake_data[2];
//        m14[3] = d_blake_data[3];
//        m14[4] = d_blake_data[4];
//        m14[5] = d_blake_data[5];
//        m14[6] = d_blake_data[6];
//        m14[7] = d_blake_data[7];
//        m14[8] = d_blake_data[8];
//        ((uint32_t *)m14)[18] = *((uint32_t *)(&d_blake_data[9]));
//        ((uint32_t *)m14)[19] = nonce13;
//        m14[10] = m14[11] = 0;
//        m14[12] = m14[13] = 0;
//        m14[14] = m14[15] = 0;
//        uint64_t v15[16] = {7640891576939301160L, 13503953896175478587UL, 4354685564936845355L, 11912009170470909681UL, 5840696475078001361L, 11170449401992604703UL, 2270897969802886507L, 6620516959819538809L, 7640891576956012808L, 13503953896175478587UL, 4354685564936845355L, 11912009170470909681UL, 5840696475078001281L, 11170449401992604703UL, 16175846103906665108UL, 6620516959819538809L};
//        G_b(0, 0, v15[0], v15[4], v15[8], v15[12], m14);
//        G_b(0, 1, v15[1], v15[5], v15[9], v15[13], m14);
//        G_b(0, 2, v15[2], v15[6], v15[10], v15[14], m14);
//        G_b(0, 3, v15[3], v15[7], v15[11], v15[15], m14);
//        G_b(0, 4, v15[0], v15[5], v15[10], v15[15], m14);
//        G_b(0, 5, v15[1], v15[6], v15[11], v15[12], m14);
//        G_b(0, 6, v15[2], v15[7], v15[8], v15[13], m14);
//        G_b(0, 7, v15[3], v15[4], v15[9], v15[14], m14);
//        ;
//        G_b(1, 0, v15[0], v15[4], v15[8], v15[12], m14);
//        G_b(1, 1, v15[1], v15[5], v15[9], v15[13], m14);
//        G_b(1, 2, v15[2], v15[6], v15[10], v15[14], m14);
//        G_b(1, 3, v15[3], v15[7], v15[11], v15[15], m14);
//        G_b(1, 4, v15[0], v15[5], v15[10], v15[15], m14);
//        G_b(1, 5, v15[1], v15[6], v15[11], v15[12], m14);
//        G_b(1, 6, v15[2], v15[7], v15[8], v15[13], m14);
//        G_b(1, 7, v15[3], v15[4], v15[9], v15[14], m14);
//        ;
//        G_b(2, 0, v15[0], v15[4], v15[8], v15[12], m14);
//        G_b(2, 1, v15[1], v15[5], v15[9], v15[13], m14);
//        G_b(2, 2, v15[2], v15[6], v15[10], v15[14], m14);
//        G_b(2, 3, v15[3], v15[7], v15[11], v15[15], m14);
//        G_b(2, 4, v15[0], v15[5], v15[10], v15[15], m14);
//        G_b(2, 5, v15[1], v15[6], v15[11], v15[12], m14);
//        G_b(2, 6, v15[2], v15[7], v15[8], v15[13], m14);
//        G_b(2, 7, v15[3], v15[4], v15[9], v15[14], m14);
//        ;
//        G_b(3, 0, v15[0], v15[4], v15[8], v15[12], m14);
//        G_b(3, 1, v15[1], v15[5], v15[9], v15[13], m14);
//        G_b(3, 2, v15[2], v15[6], v15[10], v15[14], m14);
//        G_b(3, 3, v15[3], v15[7], v15[11], v15[15], m14);
//        G_b(3, 4, v15[0], v15[5], v15[10], v15[15], m14);
//        G_b(3, 5, v15[1], v15[6], v15[11], v15[12], m14);
//        G_b(3, 6, v15[2], v15[7], v15[8], v15[13], m14);
//        G_b(3, 7, v15[3], v15[4], v15[9], v15[14], m14);
//        ;
//        G_b(4, 0, v15[0], v15[4], v15[8], v15[12], m14);
//        G_b(4, 1, v15[1], v15[5], v15[9], v15[13], m14);
//        G_b(4, 2, v15[2], v15[6], v15[10], v15[14], m14);
//        G_b(4, 3, v15[3], v15[7], v15[11], v15[15], m14);
//        G_b(4, 4, v15[0], v15[5], v15[10], v15[15], m14);
//        G_b(4, 5, v15[1], v15[6], v15[11], v15[12], m14);
//        G_b(4, 6, v15[2], v15[7], v15[8], v15[13], m14);
//        G_b(4, 7, v15[3], v15[4], v15[9], v15[14], m14);
//        ;
//        G_b(5, 0, v15[0], v15[4], v15[8], v15[12], m14);
//        G_b(5, 1, v15[1], v15[5], v15[9], v15[13], m14);
//        G_b(5, 2, v15[2], v15[6], v15[10], v15[14], m14);
//        G_b(5, 3, v15[3], v15[7], v15[11], v15[15], m14);
//        G_b(5, 4, v15[0], v15[5], v15[10], v15[15], m14);
//        G_b(5, 5, v15[1], v15[6], v15[11], v15[12], m14);
//        G_b(5, 6, v15[2], v15[7], v15[8], v15[13], m14);
//        G_b(5, 7, v15[3], v15[4], v15[9], v15[14], m14);
//        ;
//        G_b(6, 0, v15[0], v15[4], v15[8], v15[12], m14);
//        G_b(6, 1, v15[1], v15[5], v15[9], v15[13], m14);
//        G_b(6, 2, v15[2], v15[6], v15[10], v15[14], m14);
//        G_b(6, 3, v15[3], v15[7], v15[11], v15[15], m14);
//        G_b(6, 4, v15[0], v15[5], v15[10], v15[15], m14);
//        G_b(6, 5, v15[1], v15[6], v15[11], v15[12], m14);
//        G_b(6, 6, v15[2], v15[7], v15[8], v15[13], m14);
//        G_b(6, 7, v15[3], v15[4], v15[9], v15[14], m14);
//        ;
//        G_b(7, 0, v15[0], v15[4], v15[8], v15[12], m14);
//        G_b(7, 1, v15[1], v15[5], v15[9], v15[13], m14);
//        G_b(7, 2, v15[2], v15[6], v15[10], v15[14], m14);
//        G_b(7, 3, v15[3], v15[7], v15[11], v15[15], m14);
//        G_b(7, 4, v15[0], v15[5], v15[10], v15[15], m14);
//        G_b(7, 5, v15[1], v15[6], v15[11], v15[12], m14);
//        G_b(7, 6, v15[2], v15[7], v15[8], v15[13], m14);
//        G_b(7, 7, v15[3], v15[4], v15[9], v15[14], m14);
//        ;
//        G_b(8, 0, v15[0], v15[4], v15[8], v15[12], m14);
//        G_b(8, 1, v15[1], v15[5], v15[9], v15[13], m14);
//        G_b(8, 2, v15[2], v15[6], v15[10], v15[14], m14);
//        G_b(8, 3, v15[3], v15[7], v15[11], v15[15], m14);
//        G_b(8, 4, v15[0], v15[5], v15[10], v15[15], m14);
//        G_b(8, 5, v15[1], v15[6], v15[11], v15[12], m14);
//        G_b(8, 6, v15[2], v15[7], v15[8], v15[13], m14);
//        G_b(8, 7, v15[3], v15[4], v15[9], v15[14], m14);
//        ;
//        G_b(9, 0, v15[0], v15[4], v15[8], v15[12], m14);
//        G_b(9, 1, v15[1], v15[5], v15[9], v15[13], m14);
//        G_b(9, 2, v15[2], v15[6], v15[10], v15[14], m14);
//        G_b(9, 3, v15[3], v15[7], v15[11], v15[15], m14);
//        G_b(9, 4, v15[0], v15[5], v15[10], v15[15], m14);
//        G_b(9, 5, v15[1], v15[6], v15[11], v15[12], m14);
//        G_b(9, 6, v15[2], v15[7], v15[8], v15[13], m14);
//        G_b(9, 7, v15[3], v15[4], v15[9], v15[14], m14);
//        ;
//        G_b(10, 0, v15[0], v15[4], v15[8], v15[12], m14);
//        G_b(10, 1, v15[1], v15[5], v15[9], v15[13], m14);
//        G_b(10, 2, v15[2], v15[6], v15[10], v15[14], m14);
//        G_b(10, 3, v15[3], v15[7], v15[11], v15[15], m14);
//        G_b(10, 4, v15[0], v15[5], v15[10], v15[15], m14);
//        G_b(10, 5, v15[1], v15[6], v15[11], v15[12], m14);
//        G_b(10, 6, v15[2], v15[7], v15[8], v15[13], m14);
//        G_b(10, 7, v15[3], v15[4], v15[9], v15[14], m14);
//        ;
//        G_b(11, 0, v15[0], v15[4], v15[8], v15[12], m14);
//        G_b(11, 1, v15[1], v15[5], v15[9], v15[13], m14);
//        G_b(11, 2, v15[2], v15[6], v15[10], v15[14], m14);
//        G_b(11, 3, v15[3], v15[7], v15[11], v15[15], m14);
//        G_b(11, 4, v15[0], v15[5], v15[10], v15[15], m14);
//        G_b(11, 5, v15[1], v15[6], v15[11], v15[12], m14);
//        G_b(11, 6, v15[2], v15[7], v15[8], v15[13], m14);
//        G_b(11, 7, v15[3], v15[4], v15[9], v15[14], m14);
//        ;
//        uint2 last16;
//        last16 = vectorize(v15[3] ^ v15[11] ^ 11912009170470909681UL);
//        if (last16.y <= target212.y && last16.x <= target212.x) {
//            resNonce11[1] = resNonce11[0];
//            resNonce11[0] = nonce13;
//        }
//    }
//    label_0:;
//    if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=128 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 256)) goto label_1;
//    unsigned int blockDim_x_0;
//    blockDim_x_0 = 128;
//    unsigned int threadIdx_x_0;
//    threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) % 128;
//    unsigned int blockDim_y_0;
//    blockDim_y_0 = 1;
//    unsigned int threadIdx_y_0;
//    threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) / 128 % 1;
//    unsigned int blockDim_z_0;
//    blockDim_z_0 = 1;
//    unsigned int threadIdx_z_0;
//    threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) / 128;
//    for (int i = 0; i < 40; i++) {
//        uint32_t thread3;
//        thread3 = (blockDim_x_0 * blockIdx.x + threadIdx_x_0) * 40 + i;
//        static uint32_t s_K4[256] __attribute__((shared));
//        if (threadIdx_x_0 < 64U)
//            s_K4[threadIdx_x_0] = c_K[threadIdx_x_0];
//        if (thread3 < threads0) {
//            uint32_t nonce5;
//            nonce5 = startNonce1 + thread3;
//            uint32_t dat6[16];
//            *((uint2 *)(dat6)) = *((uint2 *)(c_dataEnd80));
//            dat6[2] = c_dataEnd80[2];
//            dat6[3] = nonce5;
//            dat6[4] = 2147483648U;
//            dat6[15] = 640;
//#pragma unroll (10)
//            for (int i = 5; i < 15; i++)
//                dat6[i] = 0;
//            uint32_t buf7[8];
//#pragma unroll (4)
//            for (int i = 0; i < 8; i += 2)
//                *((uint2 *)(&buf7[i])) = *((uint2 *)(&c_midstate76[i]));
//            sha256_round_body(dat6, buf7, s_K4);
//#pragma unroll (8)
//            for (int i = 0; i < 8; i++)
//                dat6[i] = buf7[i];
//            dat6[8] = 2147483648U;
//#pragma unroll (6)
//            for (int i = 9; i < 15; i++)
//                dat6[i] = 0;
//            dat6[15] = 256;
//#pragma unroll (8)
//            for (int i = 0; i < 8; i++)
//                buf7[i] = c_H256[i];
//            sha256_round_last(dat6, buf7, s_K4);
//            uint64_t high8;
//            high8 = cuda_swab32ll(((uint64_t *)buf7)[3]);
//            if (high8 <= c_target[0]) {
//                resNonces2[1] = atomicExch(resNonces2, nonce5);
//            }
//        }
//    }
//    label_1:;
//    if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=256 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 384)) goto label_2;
//    unsigned int blockDim_x_2;
//    blockDim_x_2 = 128;
//    unsigned int threadIdx_x_2;
//    threadIdx_x_2 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 256) % 128;
//    unsigned int blockDim_y_2;
//    blockDim_y_2 = 1;
//    unsigned int threadIdx_y_2;
//    threadIdx_y_2 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 256) / 128 % 1;
//    unsigned int blockDim_z_2;
//    blockDim_z_2 = 1;
//    unsigned int threadIdx_z_2;
//    threadIdx_z_2 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 256) / 128;
//    uint32_t gid19;
//    gid19 = blockIdx.x * blockDim_x_2 + threadIdx_x_2;
//    uint2 mix20[4];
//    uint64_t nonce21;
//    nonce21 = start_nonce18 + gid19;
//    uint2 *mix_hash22;
//    mix_hash22 = mix20;
//    bool result23;
//    result23 = false;
//    uint2 state24[12];
//    state24[4] = vectorize(nonce21);
//    keccak_f1600_init(state24);
//    int thread_id25;
//    thread_id25 = threadIdx_x_2 & ((128 / 16) - 1);
//    int mix_idx26;
//    mix_idx26 = thread_id25 & 3;
//    for (int i = 0; i < (128 / 16); i += 4) {
//        uint4 mix28[4];
//        uint32_t offset29[4];
//        uint32_t init030[4];
//        for (int p = 0; p < 4; p++) {
//            uint2 shuffle31[8];
//            for (int j = 0; j < 8; j++) {
//                shuffle31[j].x = __shfl_sync(4294967295U, (state24[j].x), (i + p), ((128 / 16)));
//                shuffle31[j].y = __shfl_sync(4294967295U, (state24[j].y), (i + p), ((128 / 16)));
//            }
//            switch (mix_idx26) {
//            case 0:
//                mix28[p] = vectorize2(shuffle31[0], shuffle31[1]);
//                break;
//            case 1:
//                mix28[p] = vectorize2(shuffle31[2], shuffle31[3]);
//                break;
//            case 2:
//                mix28[p] = vectorize2(shuffle31[4], shuffle31[5]);
//                break;
//            case 3:
//                mix28[p] = vectorize2(shuffle31[6], shuffle31[7]);
//                break;
//            }
//            init030[p] = __shfl_sync(4294967295U, (shuffle31[0].x), (0), ((128 / 16)));
//        }
//        for (uint32_t a = 0; a < 64; a += 4) {
//            int t32;
//            t32 = bfe(a, 2U, 3U);
//            for (uint32_t b = 0; b < 4; b++) {
//                for (int p = 0; p < 4; p++) {
//                    offset29[p] = ((init030[p] ^ (a + b)) * 16777619 ^ (((uint32_t *)&mix28[p])[b])) % d_dag_size;
//                    offset29[p] = __shfl_sync(4294967295U, (offset29[p]), (t32), ((128 / 16)));
//                    mix28[p] = fnv4(mix28[p], d_dag[offset29[p]].uint4s[thread_id25]);
//                }
//            }
//        }
//        for (int p = 0; p < 4; p++) {
//            uint2 shuffle33[4];
//            uint32_t thread_mix34;
//            thread_mix34 = fnv_reduce(mix28[p]);
//            shuffle33[0].x = __shfl_sync(4294967295U, (thread_mix34), (0), ((128 / 16)));
//            shuffle33[0].y = __shfl_sync(4294967295U, (thread_mix34), (1), ((128 / 16)));
//            shuffle33[1].x = __shfl_sync(4294967295U, (thread_mix34), (2), ((128 / 16)));
//            shuffle33[1].y = __shfl_sync(4294967295U, (thread_mix34), (3), ((128 / 16)));
//            shuffle33[2].x = __shfl_sync(4294967295U, (thread_mix34), (4), ((128 / 16)));
//            shuffle33[2].y = __shfl_sync(4294967295U, (thread_mix34), (5), ((128 / 16)));
//            shuffle33[3].x = __shfl_sync(4294967295U, (thread_mix34), (6), ((128 / 16)));
//            shuffle33[3].y = __shfl_sync(4294967295U, (thread_mix34), (7), ((128 / 16)));
//            if ((i + p) == thread_id25) {
//                state24[8] = shuffle33[0];
//                state24[9] = shuffle33[1];
//                state24[10] = shuffle33[2];
//                state24[11] = shuffle33[3];
//            }
//        }
//    }
//    if (!(cuda_swab64(keccak_f1600_final(state24)) > d_target)) {
//        mix_hash22[0] = state24[8];
//        mix_hash22[1] = state24[9];
//        mix_hash22[2] = state24[10];
//        mix_hash22[3] = state24[11];
//        return;
//    }
//    uint32_t index27;
//    index27 = atomicInc((uint32_t *)&g_output17->count, 4294967295U);
//    if (index27 >= 4U)
//        return;
//    g_output17->result[index27].gid = gid19;
//    g_output17->result[index27].mix[0] = mix20[0].x;
//    g_output17->result[index27].mix[1] = mix20[0].y;
//    g_output17->result[index27].mix[2] = mix20[1].x;
//    g_output17->result[index27].mix[3] = mix20[1].y;
//    g_output17->result[index27].mix[4] = mix20[2].x;
//    g_output17->result[index27].mix[5] = mix20[2].y;
//    g_output17->result[index27].mix[6] = mix20[3].x;
//    g_output17->result[index27].mix[7] = mix20[3].y;
//    label_2:;
//}
//
//
//__attribute__((global)) void FUNC_nosha3(const uint32_t threads0, const uint32_t startNonce1, uint32_t *resNonce2, const uint2 target23, const uint32_t threads8, const uint32_t startNonce9, uint32_t *resNonce10, const uint2 target211, volatile Search_results *g_output17, uint64_t start_nonce18)
//{
//    if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 128)) goto label_0;
//    unsigned int blockDim_x_0;
//    blockDim_x_0 = 128;
//    unsigned int threadIdx_x_0;
//    threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 128;
//    unsigned int blockDim_y_0;
//    blockDim_y_0 = 1;
//    unsigned int threadIdx_y_0;
//    threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128 % 1;
//    unsigned int blockDim_z_0;
//    blockDim_z_0 = 1;
//    unsigned int threadIdx_z_0;
//    threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128;
//    for (int i = 0; i < 80; i++) {
//        uint32_t nonce4;
//        nonce4 = (blockDim_x_0 * blockIdx.x + threadIdx_x_0) * 80 + i + startNonce1;
//        uint64_t m5[16];
//        m5[0] = d_blake_data[0];
//        m5[1] = d_blake_data[1];
//        m5[2] = d_blake_data[2];
//        m5[3] = d_blake_data[3];
//        m5[4] = d_blake_data[4];
//        m5[5] = d_blake_data[5];
//        m5[6] = d_blake_data[6];
//        m5[7] = d_blake_data[7];
//        m5[8] = d_blake_data[8];
//        ((uint32_t *)m5)[18] = *((uint32_t *)(&d_blake_data[9]));
//        ((uint32_t *)m5)[19] = nonce4;
//        m5[10] = m5[11] = 0;
//        m5[12] = m5[13] = 0;
//        m5[14] = m5[15] = 0;
//        uint64_t v6[16] = {7640891576939301160L, 13503953896175478587UL, 4354685564936845355L, 11912009170470909681UL, 5840696475078001361L, 11170449401992604703UL, 2270897969802886507L, 6620516959819538809L, 7640891576956012808L, 13503953896175478587UL, 4354685564936845355L, 11912009170470909681UL, 5840696475078001281L, 11170449401992604703UL, 16175846103906665108UL, 6620516959819538809L};
//        G_b(0, 0, v6[0], v6[4], v6[8], v6[12], m5);
//        G_b(0, 1, v6[1], v6[5], v6[9], v6[13], m5);
//        G_b(0, 2, v6[2], v6[6], v6[10], v6[14], m5);
//        G_b(0, 3, v6[3], v6[7], v6[11], v6[15], m5);
//        G_b(0, 4, v6[0], v6[5], v6[10], v6[15], m5);
//        G_b(0, 5, v6[1], v6[6], v6[11], v6[12], m5);
//        G_b(0, 6, v6[2], v6[7], v6[8], v6[13], m5);
//        G_b(0, 7, v6[3], v6[4], v6[9], v6[14], m5);
//        ;
//        G_b(1, 0, v6[0], v6[4], v6[8], v6[12], m5);
//        G_b(1, 1, v6[1], v6[5], v6[9], v6[13], m5);
//        G_b(1, 2, v6[2], v6[6], v6[10], v6[14], m5);
//        G_b(1, 3, v6[3], v6[7], v6[11], v6[15], m5);
//        G_b(1, 4, v6[0], v6[5], v6[10], v6[15], m5);
//        G_b(1, 5, v6[1], v6[6], v6[11], v6[12], m5);
//        G_b(1, 6, v6[2], v6[7], v6[8], v6[13], m5);
//        G_b(1, 7, v6[3], v6[4], v6[9], v6[14], m5);
//        ;
//        G_b(2, 0, v6[0], v6[4], v6[8], v6[12], m5);
//        G_b(2, 1, v6[1], v6[5], v6[9], v6[13], m5);
//        G_b(2, 2, v6[2], v6[6], v6[10], v6[14], m5);
//        G_b(2, 3, v6[3], v6[7], v6[11], v6[15], m5);
//        G_b(2, 4, v6[0], v6[5], v6[10], v6[15], m5);
//        G_b(2, 5, v6[1], v6[6], v6[11], v6[12], m5);
//        G_b(2, 6, v6[2], v6[7], v6[8], v6[13], m5);
//        G_b(2, 7, v6[3], v6[4], v6[9], v6[14], m5);
//        ;
//        G_b(3, 0, v6[0], v6[4], v6[8], v6[12], m5);
//        G_b(3, 1, v6[1], v6[5], v6[9], v6[13], m5);
//        G_b(3, 2, v6[2], v6[6], v6[10], v6[14], m5);
//        G_b(3, 3, v6[3], v6[7], v6[11], v6[15], m5);
//        G_b(3, 4, v6[0], v6[5], v6[10], v6[15], m5);
//        G_b(3, 5, v6[1], v6[6], v6[11], v6[12], m5);
//        G_b(3, 6, v6[2], v6[7], v6[8], v6[13], m5);
//        G_b(3, 7, v6[3], v6[4], v6[9], v6[14], m5);
//        ;
//        G_b(4, 0, v6[0], v6[4], v6[8], v6[12], m5);
//        G_b(4, 1, v6[1], v6[5], v6[9], v6[13], m5);
//        G_b(4, 2, v6[2], v6[6], v6[10], v6[14], m5);
//        G_b(4, 3, v6[3], v6[7], v6[11], v6[15], m5);
//        G_b(4, 4, v6[0], v6[5], v6[10], v6[15], m5);
//        G_b(4, 5, v6[1], v6[6], v6[11], v6[12], m5);
//        G_b(4, 6, v6[2], v6[7], v6[8], v6[13], m5);
//        G_b(4, 7, v6[3], v6[4], v6[9], v6[14], m5);
//        ;
//        G_b(5, 0, v6[0], v6[4], v6[8], v6[12], m5);
//        G_b(5, 1, v6[1], v6[5], v6[9], v6[13], m5);
//        G_b(5, 2, v6[2], v6[6], v6[10], v6[14], m5);
//        G_b(5, 3, v6[3], v6[7], v6[11], v6[15], m5);
//        G_b(5, 4, v6[0], v6[5], v6[10], v6[15], m5);
//        G_b(5, 5, v6[1], v6[6], v6[11], v6[12], m5);
//        G_b(5, 6, v6[2], v6[7], v6[8], v6[13], m5);
//        G_b(5, 7, v6[3], v6[4], v6[9], v6[14], m5);
//        ;
//        G_b(6, 0, v6[0], v6[4], v6[8], v6[12], m5);
//        G_b(6, 1, v6[1], v6[5], v6[9], v6[13], m5);
//        G_b(6, 2, v6[2], v6[6], v6[10], v6[14], m5);
//        G_b(6, 3, v6[3], v6[7], v6[11], v6[15], m5);
//        G_b(6, 4, v6[0], v6[5], v6[10], v6[15], m5);
//        G_b(6, 5, v6[1], v6[6], v6[11], v6[12], m5);
//        G_b(6, 6, v6[2], v6[7], v6[8], v6[13], m5);
//        G_b(6, 7, v6[3], v6[4], v6[9], v6[14], m5);
//        ;
//        G_b(7, 0, v6[0], v6[4], v6[8], v6[12], m5);
//        G_b(7, 1, v6[1], v6[5], v6[9], v6[13], m5);
//        G_b(7, 2, v6[2], v6[6], v6[10], v6[14], m5);
//        G_b(7, 3, v6[3], v6[7], v6[11], v6[15], m5);
//        G_b(7, 4, v6[0], v6[5], v6[10], v6[15], m5);
//        G_b(7, 5, v6[1], v6[6], v6[11], v6[12], m5);
//        G_b(7, 6, v6[2], v6[7], v6[8], v6[13], m5);
//        G_b(7, 7, v6[3], v6[4], v6[9], v6[14], m5);
//        ;
//        G_b(8, 0, v6[0], v6[4], v6[8], v6[12], m5);
//        G_b(8, 1, v6[1], v6[5], v6[9], v6[13], m5);
//        G_b(8, 2, v6[2], v6[6], v6[10], v6[14], m5);
//        G_b(8, 3, v6[3], v6[7], v6[11], v6[15], m5);
//        G_b(8, 4, v6[0], v6[5], v6[10], v6[15], m5);
//        G_b(8, 5, v6[1], v6[6], v6[11], v6[12], m5);
//        G_b(8, 6, v6[2], v6[7], v6[8], v6[13], m5);
//        G_b(8, 7, v6[3], v6[4], v6[9], v6[14], m5);
//        ;
//        G_b(9, 0, v6[0], v6[4], v6[8], v6[12], m5);
//        G_b(9, 1, v6[1], v6[5], v6[9], v6[13], m5);
//        G_b(9, 2, v6[2], v6[6], v6[10], v6[14], m5);
//        G_b(9, 3, v6[3], v6[7], v6[11], v6[15], m5);
//        G_b(9, 4, v6[0], v6[5], v6[10], v6[15], m5);
//        G_b(9, 5, v6[1], v6[6], v6[11], v6[12], m5);
//        G_b(9, 6, v6[2], v6[7], v6[8], v6[13], m5);
//        G_b(9, 7, v6[3], v6[4], v6[9], v6[14], m5);
//        ;
//        G_b(10, 0, v6[0], v6[4], v6[8], v6[12], m5);
//        G_b(10, 1, v6[1], v6[5], v6[9], v6[13], m5);
//        G_b(10, 2, v6[2], v6[6], v6[10], v6[14], m5);
//        G_b(10, 3, v6[3], v6[7], v6[11], v6[15], m5);
//        G_b(10, 4, v6[0], v6[5], v6[10], v6[15], m5);
//        G_b(10, 5, v6[1], v6[6], v6[11], v6[12], m5);
//        G_b(10, 6, v6[2], v6[7], v6[8], v6[13], m5);
//        G_b(10, 7, v6[3], v6[4], v6[9], v6[14], m5);
//        ;
//        G_b(11, 0, v6[0], v6[4], v6[8], v6[12], m5);
//        G_b(11, 1, v6[1], v6[5], v6[9], v6[13], m5);
//        G_b(11, 2, v6[2], v6[6], v6[10], v6[14], m5);
//        G_b(11, 3, v6[3], v6[7], v6[11], v6[15], m5);
//        G_b(11, 4, v6[0], v6[5], v6[10], v6[15], m5);
//        G_b(11, 5, v6[1], v6[6], v6[11], v6[12], m5);
//        G_b(11, 6, v6[2], v6[7], v6[8], v6[13], m5);
//        G_b(11, 7, v6[3], v6[4], v6[9], v6[14], m5);
//        ;
//        uint2 last7;
//        last7 = vectorize(v6[3] ^ v6[11] ^ 11912009170470909681UL);
//        if (last7.y <= target23.y && last7.x <= target23.x) {
//            resNonce2[1] = resNonce2[0];
//            resNonce2[0] = nonce4;
//        }
//    }
//    label_0:;
//    if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=128 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 256)) goto label_1;
//    unsigned int blockDim_x_1;
//    blockDim_x_1 = 128;
//    unsigned int threadIdx_x_1;
//    threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) % 128;
//    unsigned int blockDim_y_1;
//    blockDim_y_1 = 1;
//    unsigned int threadIdx_y_1;
//    threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) / 128 % 1;
//    unsigned int blockDim_z_1;
//    blockDim_z_1 = 1;
//    unsigned int threadIdx_z_1;
//    threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) / 128;
//    for (int i = 0; i < 80; i++) {
//        uint32_t nonce12;
//        nonce12 = (blockDim_x_1 * blockIdx.x + threadIdx_x_1) * 80 + i + startNonce9;
//        static uint64_t s_target13 __attribute__((shared));
//        if (!threadIdx_x_1)
//            s_target13 = devectorize(target211);
//        uint64_t m14[16];
//        m14[0] = d_data2[0];
//        m14[1] = d_data2[1];
//        m14[2] = d_data2[2];
//        m14[3] = d_data2[3];
//        m14[4] = d_data2[4] | nonce12;
//        m14[5] = d_data2[5];
//        m14[6] = d_data2[6];
//        m14[7] = d_data2[7];
//        m14[8] = d_data2[8];
//        m14[9] = d_data2[9];
//        m14[10] = m14[11] = 0;
//        m14[12] = m14[13] = m14[14] = m14[15] = 0;
//        uint64_t v15[16] = {7640891576939301160L, 13503953896175478587UL, 4354685564936845355L, 11912009170470909681UL, 5840696475078001361L, 11170449401992604703UL, 2270897969802886507L, 6620516959819538809L, 7640891576956012808L, 13503953896175478587UL, 4354685564936845355L, 11912009170470909681UL, 5840696475078001281L, 11170449401992604703UL, 16175846103906665108UL, 6620516959819538809L};
//        G(0, 0, v15[0], v15[4], v15[8], v15[12], m14);
//        G(0, 1, v15[1], v15[5], v15[9], v15[13], m14);
//        G(0, 2, v15[2], v15[6], v15[10], v15[14], m14);
//        G(0, 3, v15[3], v15[7], v15[11], v15[15], m14);
//        G(0, 4, v15[0], v15[5], v15[10], v15[15], m14);
//        G(0, 5, v15[1], v15[6], v15[11], v15[12], m14);
//        G(0, 6, v15[2], v15[7], v15[8], v15[13], m14);
//        G(0, 7, v15[3], v15[4], v15[9], v15[14], m14);
//        ;
//        G(1, 0, v15[0], v15[4], v15[8], v15[12], m14);
//        G(1, 1, v15[1], v15[5], v15[9], v15[13], m14);
//        G(1, 2, v15[2], v15[6], v15[10], v15[14], m14);
//        G(1, 3, v15[3], v15[7], v15[11], v15[15], m14);
//        G(1, 4, v15[0], v15[5], v15[10], v15[15], m14);
//        G(1, 5, v15[1], v15[6], v15[11], v15[12], m14);
//        G(1, 6, v15[2], v15[7], v15[8], v15[13], m14);
//        G(1, 7, v15[3], v15[4], v15[9], v15[14], m14);
//        ;
//        G(2, 0, v15[0], v15[4], v15[8], v15[12], m14);
//        G(2, 1, v15[1], v15[5], v15[9], v15[13], m14);
//        G(2, 2, v15[2], v15[6], v15[10], v15[14], m14);
//        G(2, 3, v15[3], v15[7], v15[11], v15[15], m14);
//        G(2, 4, v15[0], v15[5], v15[10], v15[15], m14);
//        G(2, 5, v15[1], v15[6], v15[11], v15[12], m14);
//        G(2, 6, v15[2], v15[7], v15[8], v15[13], m14);
//        G(2, 7, v15[3], v15[4], v15[9], v15[14], m14);
//        ;
//        G(3, 0, v15[0], v15[4], v15[8], v15[12], m14);
//        G(3, 1, v15[1], v15[5], v15[9], v15[13], m14);
//        G(3, 2, v15[2], v15[6], v15[10], v15[14], m14);
//        G(3, 3, v15[3], v15[7], v15[11], v15[15], m14);
//        G(3, 4, v15[0], v15[5], v15[10], v15[15], m14);
//        G(3, 5, v15[1], v15[6], v15[11], v15[12], m14);
//        G(3, 6, v15[2], v15[7], v15[8], v15[13], m14);
//        G(3, 7, v15[3], v15[4], v15[9], v15[14], m14);
//        ;
//        G(4, 0, v15[0], v15[4], v15[8], v15[12], m14);
//        G(4, 1, v15[1], v15[5], v15[9], v15[13], m14);
//        G(4, 2, v15[2], v15[6], v15[10], v15[14], m14);
//        G(4, 3, v15[3], v15[7], v15[11], v15[15], m14);
//        G(4, 4, v15[0], v15[5], v15[10], v15[15], m14);
//        G(4, 5, v15[1], v15[6], v15[11], v15[12], m14);
//        G(4, 6, v15[2], v15[7], v15[8], v15[13], m14);
//        G(4, 7, v15[3], v15[4], v15[9], v15[14], m14);
//        ;
//        G(5, 0, v15[0], v15[4], v15[8], v15[12], m14);
//        G(5, 1, v15[1], v15[5], v15[9], v15[13], m14);
//        G(5, 2, v15[2], v15[6], v15[10], v15[14], m14);
//        G(5, 3, v15[3], v15[7], v15[11], v15[15], m14);
//        G(5, 4, v15[0], v15[5], v15[10], v15[15], m14);
//        G(5, 5, v15[1], v15[6], v15[11], v15[12], m14);
//        G(5, 6, v15[2], v15[7], v15[8], v15[13], m14);
//        G(5, 7, v15[3], v15[4], v15[9], v15[14], m14);
//        ;
//        G(6, 0, v15[0], v15[4], v15[8], v15[12], m14);
//        G(6, 1, v15[1], v15[5], v15[9], v15[13], m14);
//        G(6, 2, v15[2], v15[6], v15[10], v15[14], m14);
//        G(6, 3, v15[3], v15[7], v15[11], v15[15], m14);
//        G(6, 4, v15[0], v15[5], v15[10], v15[15], m14);
//        G(6, 5, v15[1], v15[6], v15[11], v15[12], m14);
//        G(6, 6, v15[2], v15[7], v15[8], v15[13], m14);
//        G(6, 7, v15[3], v15[4], v15[9], v15[14], m14);
//        ;
//        G(7, 0, v15[0], v15[4], v15[8], v15[12], m14);
//        G(7, 1, v15[1], v15[5], v15[9], v15[13], m14);
//        G(7, 2, v15[2], v15[6], v15[10], v15[14], m14);
//        G(7, 3, v15[3], v15[7], v15[11], v15[15], m14);
//        G(7, 4, v15[0], v15[5], v15[10], v15[15], m14);
//        G(7, 5, v15[1], v15[6], v15[11], v15[12], m14);
//        G(7, 6, v15[2], v15[7], v15[8], v15[13], m14);
//        G(7, 7, v15[3], v15[4], v15[9], v15[14], m14);
//        ;
//        G(8, 0, v15[0], v15[4], v15[8], v15[12], m14);
//        G(8, 1, v15[1], v15[5], v15[9], v15[13], m14);
//        G(8, 2, v15[2], v15[6], v15[10], v15[14], m14);
//        G(8, 3, v15[3], v15[7], v15[11], v15[15], m14);
//        G(8, 4, v15[0], v15[5], v15[10], v15[15], m14);
//        G(8, 5, v15[1], v15[6], v15[11], v15[12], m14);
//        G(8, 6, v15[2], v15[7], v15[8], v15[13], m14);
//        G(8, 7, v15[3], v15[4], v15[9], v15[14], m14);
//        ;
//        G(9, 0, v15[0], v15[4], v15[8], v15[12], m14);
//        G(9, 1, v15[1], v15[5], v15[9], v15[13], m14);
//        G(9, 2, v15[2], v15[6], v15[10], v15[14], m14);
//        G(9, 3, v15[3], v15[7], v15[11], v15[15], m14);
//        G(9, 4, v15[0], v15[5], v15[10], v15[15], m14);
//        G(9, 5, v15[1], v15[6], v15[11], v15[12], m14);
//        G(9, 6, v15[2], v15[7], v15[8], v15[13], m14);
//        G(9, 7, v15[3], v15[4], v15[9], v15[14], m14);
//        ;
//        G(10, 0, v15[0], v15[4], v15[8], v15[12], m14);
//        G(10, 1, v15[1], v15[5], v15[9], v15[13], m14);
//        G(10, 2, v15[2], v15[6], v15[10], v15[14], m14);
//        G(10, 3, v15[3], v15[7], v15[11], v15[15], m14);
//        G(10, 4, v15[0], v15[5], v15[10], v15[15], m14);
//        G(10, 5, v15[1], v15[6], v15[11], v15[12], m14);
//        G(10, 6, v15[2], v15[7], v15[8], v15[13], m14);
//        G(10, 7, v15[3], v15[4], v15[9], v15[14], m14);
//        ;
//        G(11, 0, v15[0], v15[4], v15[8], v15[12], m14);
//        G(11, 1, v15[1], v15[5], v15[9], v15[13], m14);
//        G(11, 2, v15[2], v15[6], v15[10], v15[14], m14);
//        G(11, 3, v15[3], v15[7], v15[11], v15[15], m14);
//        G(11, 4, v15[0], v15[5], v15[10], v15[15], m14);
//        G(11, 5, v15[1], v15[6], v15[11], v15[12], m14);
//        H(11, 6, v15[2], v15[7], v15[8], v15[13], m14);
//        ;
//        uint64_t h6416;
//        h6416 = cuda_swab64(7640891576939301160L ^ v15[0] ^ v15[8]);
//        if (h6416 <= s_target13) {
//            resNonce10[1] = resNonce10[0];
//            resNonce10[0] = nonce12;
//            s_target13 = h6416;
//        }
//    }
//    label_1:;
//    if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=256 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 384)) goto label_2;
//    unsigned int blockDim_x_2;
//    blockDim_x_2 = 128;
//    unsigned int threadIdx_x_2;
//    threadIdx_x_2 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 256) % 128;
//    unsigned int blockDim_y_2;
//    blockDim_y_2 = 1;
//    unsigned int threadIdx_y_2;
//    threadIdx_y_2 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 256) / 128 % 1;
//    unsigned int blockDim_z_2;
//    blockDim_z_2 = 1;
//    unsigned int threadIdx_z_2;
//    threadIdx_z_2 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 256) / 128;
//    uint32_t gid19;
//    gid19 = blockIdx.x * blockDim_x_2 + threadIdx_x_2;
//    uint2 mix20[4];
//    uint64_t nonce21;
//    nonce21 = start_nonce18 + gid19;
//    uint2 *mix_hash22;
//    mix_hash22 = mix20;
//    bool result23;
//    result23 = false;
//    uint2 state24[12];
//    state24[4] = vectorize(nonce21);
//    keccak_f1600_init(state24);
//    int thread_id25;
//    thread_id25 = threadIdx_x_2 & ((128 / 16) - 1);
//    int mix_idx26;
//    mix_idx26 = thread_id25 & 3;
//    for (int i = 0; i < (128 / 16); i += 4) {
//        uint4 mix28[4];
//        uint32_t offset29[4];
//        uint32_t init030[4];
//        for (int p = 0; p < 4; p++) {
//            uint2 shuffle31[8];
//            for (int j = 0; j < 8; j++) {
//                shuffle31[j].x = __shfl_sync(4294967295U, (state24[j].x), (i + p), ((128 / 16)));
//                shuffle31[j].y = __shfl_sync(4294967295U, (state24[j].y), (i + p), ((128 / 16)));
//            }
//            switch (mix_idx26) {
//            case 0:
//                mix28[p] = vectorize2(shuffle31[0], shuffle31[1]);
//                break;
//            case 1:
//                mix28[p] = vectorize2(shuffle31[2], shuffle31[3]);
//                break;
//            case 2:
//                mix28[p] = vectorize2(shuffle31[4], shuffle31[5]);
//                break;
//            case 3:
//                mix28[p] = vectorize2(shuffle31[6], shuffle31[7]);
//                break;
//            }
//            init030[p] = __shfl_sync(4294967295U, (shuffle31[0].x), (0), ((128 / 16)));
//        }
//        for (uint32_t a = 0; a < 64; a += 4) {
//            int t32;
//            t32 = bfe(a, 2U, 3U);
//            for (uint32_t b = 0; b < 4; b++) {
//                for (int p = 0; p < 4; p++) {
//                    offset29[p] = ((init030[p] ^ (a + b)) * 16777619 ^ (((uint32_t *)&mix28[p])[b])) % d_dag_size;
//                    offset29[p] = __shfl_sync(4294967295U, (offset29[p]), (t32), ((128 / 16)));
//                    mix28[p] = fnv4(mix28[p], d_dag[offset29[p]].uint4s[thread_id25]);
//                }
//            }
//        }
//        for (int p = 0; p < 4; p++) {
//            uint2 shuffle33[4];
//            uint32_t thread_mix34;
//            thread_mix34 = fnv_reduce(mix28[p]);
//            shuffle33[0].x = __shfl_sync(4294967295U, (thread_mix34), (0), ((128 / 16)));
//            shuffle33[0].y = __shfl_sync(4294967295U, (thread_mix34), (1), ((128 / 16)));
//            shuffle33[1].x = __shfl_sync(4294967295U, (thread_mix34), (2), ((128 / 16)));
//            shuffle33[1].y = __shfl_sync(4294967295U, (thread_mix34), (3), ((128 / 16)));
//            shuffle33[2].x = __shfl_sync(4294967295U, (thread_mix34), (4), ((128 / 16)));
//            shuffle33[2].y = __shfl_sync(4294967295U, (thread_mix34), (5), ((128 / 16)));
//            shuffle33[3].x = __shfl_sync(4294967295U, (thread_mix34), (6), ((128 / 16)));
//            shuffle33[3].y = __shfl_sync(4294967295U, (thread_mix34), (7), ((128 / 16)));
//            if ((i + p) == thread_id25) {
//                state24[8] = shuffle33[0];
//                state24[9] = shuffle33[1];
//                state24[10] = shuffle33[2];
//                state24[11] = shuffle33[3];
//            }
//        }
//    }
//    if (!(cuda_swab64(keccak_f1600_final(state24)) > d_target)) {
//        mix_hash22[0] = state24[8];
//        mix_hash22[1] = state24[9];
//        mix_hash22[2] = state24[10];
//        mix_hash22[3] = state24[11];
//        return;
//    }
//    uint32_t index27;
//    index27 = atomicInc((uint32_t *)&g_output17->count, 4294967295U);
//    if (index27 >= 4U)
//        return;
//    g_output17->result[index27].gid = gid19;
//    g_output17->result[index27].mix[0] = mix20[0].x;
//    g_output17->result[index27].mix[1] = mix20[0].y;
//    g_output17->result[index27].mix[2] = mix20[1].x;
//    g_output17->result[index27].mix[3] = mix20[1].y;
//    g_output17->result[index27].mix[4] = mix20[2].x;
//    g_output17->result[index27].mix[5] = mix20[2].y;
//    g_output17->result[index27].mix[6] = mix20[3].x;
//    g_output17->result[index27].mix[7] = mix20[3].y;
//    label_2:;
//}
//
//
//__attribute__((global)) void FUNC4(const uint32_t threads915, const uint32_t startNonce1016, uint32_t *resNonce1117, const uint2 target21218, const uint32_t threads1729, const uint32_t startNonce1830, uint32_t *resNonce1931, const uint2 target22032, const uint32_t threads00, const uint32_t startNonce11, uint32_t *resNonces22, volatile Search_results *g_output2644, uint64_t start_nonce2745)
//{
//    if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 128)) goto label_0;
//    unsigned int blockDim_x_1;
//    blockDim_x_1 = 128;
//    unsigned int threadIdx_x_1;
//    threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 128;
//    unsigned int blockDim_y_1;
//    blockDim_y_1 = 1;
//    unsigned int threadIdx_y_1;
//    threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128 % 1;
//    unsigned int blockDim_z_1;
//    blockDim_z_1 = 1;
//    unsigned int threadIdx_z_1;
//    threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128;
//    unsigned int blockDim_x_119;
//    blockDim_x_119 = 128;
//    unsigned int threadIdx_x_120;
//    threadIdx_x_120 = ((threadIdx_x_1 + threadIdx_y_1 * blockDim_x_1 + threadIdx_z_1 * blockDim_x_1 * blockDim_y_1) - 0) % 128;
//    unsigned int blockDim_y_121;
//    blockDim_y_121 = 1;
//    unsigned int threadIdx_y_122;
//    threadIdx_y_122 = ((threadIdx_x_1 + threadIdx_y_1 * blockDim_x_1 + threadIdx_z_1 * blockDim_x_1 * blockDim_y_1) - 0) / 128 % 1;
//    unsigned int blockDim_z_123;
//    blockDim_z_123 = 1;
//    unsigned int threadIdx_z_124;
//    threadIdx_z_124 = ((threadIdx_x_1 + threadIdx_y_1 * blockDim_x_1 + threadIdx_z_1 * blockDim_x_1 * blockDim_y_1) - 0) / 128;
//    for (int i = 0; i < 80; i++) {
//        uint32_t nonce1325;
//        nonce1325 = (blockDim_x_119 * blockIdx.x + threadIdx_x_120) * 80 + i + startNonce1016;
//        uint64_t m1426[16];
//        m1426[0] = d_blake_data[0];
//        m1426[1] = d_blake_data[1];
//        m1426[2] = d_blake_data[2];
//        m1426[3] = d_blake_data[3];
//        m1426[4] = d_blake_data[4];
//        m1426[5] = d_blake_data[5];
//        m1426[6] = d_blake_data[6];
//        m1426[7] = d_blake_data[7];
//        m1426[8] = d_blake_data[8];
//        ((uint32_t *)m1426)[18] = *((uint32_t *)(&d_blake_data[9]));
//        ((uint32_t *)m1426)[19] = nonce1325;
//        m1426[10] = m1426[11] = 0;
//        m1426[12] = m1426[13] = 0;
//        m1426[14] = m1426[15] = 0;
//        uint64_t v1527[16] = {7640891576939301160L, 13503953896175478587UL, 4354685564936845355L, 11912009170470909681UL, 5840696475078001361L, 11170449401992604703UL, 2270897969802886507L, 6620516959819538809L, 7640891576956012808L, 13503953896175478587UL, 4354685564936845355L, 11912009170470909681UL, 5840696475078001281L, 11170449401992604703UL, 16175846103906665108UL, 6620516959819538809L};
//        G_b(0, 0, v1527[0], v1527[4], v1527[8], v1527[12], m1426);
//        G_b(0, 1, v1527[1], v1527[5], v1527[9], v1527[13], m1426);
//        G_b(0, 2, v1527[2], v1527[6], v1527[10], v1527[14], m1426);
//        G_b(0, 3, v1527[3], v1527[7], v1527[11], v1527[15], m1426);
//        G_b(0, 4, v1527[0], v1527[5], v1527[10], v1527[15], m1426);
//        G_b(0, 5, v1527[1], v1527[6], v1527[11], v1527[12], m1426);
//        G_b(0, 6, v1527[2], v1527[7], v1527[8], v1527[13], m1426);
//        G_b(0, 7, v1527[3], v1527[4], v1527[9], v1527[14], m1426);
//        ;
//        G_b(1, 0, v1527[0], v1527[4], v1527[8], v1527[12], m1426);
//        G_b(1, 1, v1527[1], v1527[5], v1527[9], v1527[13], m1426);
//        G_b(1, 2, v1527[2], v1527[6], v1527[10], v1527[14], m1426);
//        G_b(1, 3, v1527[3], v1527[7], v1527[11], v1527[15], m1426);
//        G_b(1, 4, v1527[0], v1527[5], v1527[10], v1527[15], m1426);
//        G_b(1, 5, v1527[1], v1527[6], v1527[11], v1527[12], m1426);
//        G_b(1, 6, v1527[2], v1527[7], v1527[8], v1527[13], m1426);
//        G_b(1, 7, v1527[3], v1527[4], v1527[9], v1527[14], m1426);
//        ;
//        G_b(2, 0, v1527[0], v1527[4], v1527[8], v1527[12], m1426);
//        G_b(2, 1, v1527[1], v1527[5], v1527[9], v1527[13], m1426);
//        G_b(2, 2, v1527[2], v1527[6], v1527[10], v1527[14], m1426);
//        G_b(2, 3, v1527[3], v1527[7], v1527[11], v1527[15], m1426);
//        G_b(2, 4, v1527[0], v1527[5], v1527[10], v1527[15], m1426);
//        G_b(2, 5, v1527[1], v1527[6], v1527[11], v1527[12], m1426);
//        G_b(2, 6, v1527[2], v1527[7], v1527[8], v1527[13], m1426);
//        G_b(2, 7, v1527[3], v1527[4], v1527[9], v1527[14], m1426);
//        ;
//        G_b(3, 0, v1527[0], v1527[4], v1527[8], v1527[12], m1426);
//        G_b(3, 1, v1527[1], v1527[5], v1527[9], v1527[13], m1426);
//        G_b(3, 2, v1527[2], v1527[6], v1527[10], v1527[14], m1426);
//        G_b(3, 3, v1527[3], v1527[7], v1527[11], v1527[15], m1426);
//        G_b(3, 4, v1527[0], v1527[5], v1527[10], v1527[15], m1426);
//        G_b(3, 5, v1527[1], v1527[6], v1527[11], v1527[12], m1426);
//        G_b(3, 6, v1527[2], v1527[7], v1527[8], v1527[13], m1426);
//        G_b(3, 7, v1527[3], v1527[4], v1527[9], v1527[14], m1426);
//        ;
//        G_b(4, 0, v1527[0], v1527[4], v1527[8], v1527[12], m1426);
//        G_b(4, 1, v1527[1], v1527[5], v1527[9], v1527[13], m1426);
//        G_b(4, 2, v1527[2], v1527[6], v1527[10], v1527[14], m1426);
//        G_b(4, 3, v1527[3], v1527[7], v1527[11], v1527[15], m1426);
//        G_b(4, 4, v1527[0], v1527[5], v1527[10], v1527[15], m1426);
//        G_b(4, 5, v1527[1], v1527[6], v1527[11], v1527[12], m1426);
//        G_b(4, 6, v1527[2], v1527[7], v1527[8], v1527[13], m1426);
//        G_b(4, 7, v1527[3], v1527[4], v1527[9], v1527[14], m1426);
//        ;
//        G_b(5, 0, v1527[0], v1527[4], v1527[8], v1527[12], m1426);
//        G_b(5, 1, v1527[1], v1527[5], v1527[9], v1527[13], m1426);
//        G_b(5, 2, v1527[2], v1527[6], v1527[10], v1527[14], m1426);
//        G_b(5, 3, v1527[3], v1527[7], v1527[11], v1527[15], m1426);
//        G_b(5, 4, v1527[0], v1527[5], v1527[10], v1527[15], m1426);
//        G_b(5, 5, v1527[1], v1527[6], v1527[11], v1527[12], m1426);
//        G_b(5, 6, v1527[2], v1527[7], v1527[8], v1527[13], m1426);
//        G_b(5, 7, v1527[3], v1527[4], v1527[9], v1527[14], m1426);
//        ;
//        G_b(6, 0, v1527[0], v1527[4], v1527[8], v1527[12], m1426);
//        G_b(6, 1, v1527[1], v1527[5], v1527[9], v1527[13], m1426);
//        G_b(6, 2, v1527[2], v1527[6], v1527[10], v1527[14], m1426);
//        G_b(6, 3, v1527[3], v1527[7], v1527[11], v1527[15], m1426);
//        G_b(6, 4, v1527[0], v1527[5], v1527[10], v1527[15], m1426);
//        G_b(6, 5, v1527[1], v1527[6], v1527[11], v1527[12], m1426);
//        G_b(6, 6, v1527[2], v1527[7], v1527[8], v1527[13], m1426);
//        G_b(6, 7, v1527[3], v1527[4], v1527[9], v1527[14], m1426);
//        ;
//        G_b(7, 0, v1527[0], v1527[4], v1527[8], v1527[12], m1426);
//        G_b(7, 1, v1527[1], v1527[5], v1527[9], v1527[13], m1426);
//        G_b(7, 2, v1527[2], v1527[6], v1527[10], v1527[14], m1426);
//        G_b(7, 3, v1527[3], v1527[7], v1527[11], v1527[15], m1426);
//        G_b(7, 4, v1527[0], v1527[5], v1527[10], v1527[15], m1426);
//        G_b(7, 5, v1527[1], v1527[6], v1527[11], v1527[12], m1426);
//        G_b(7, 6, v1527[2], v1527[7], v1527[8], v1527[13], m1426);
//        G_b(7, 7, v1527[3], v1527[4], v1527[9], v1527[14], m1426);
//        ;
//        G_b(8, 0, v1527[0], v1527[4], v1527[8], v1527[12], m1426);
//        G_b(8, 1, v1527[1], v1527[5], v1527[9], v1527[13], m1426);
//        G_b(8, 2, v1527[2], v1527[6], v1527[10], v1527[14], m1426);
//        G_b(8, 3, v1527[3], v1527[7], v1527[11], v1527[15], m1426);
//        G_b(8, 4, v1527[0], v1527[5], v1527[10], v1527[15], m1426);
//        G_b(8, 5, v1527[1], v1527[6], v1527[11], v1527[12], m1426);
//        G_b(8, 6, v1527[2], v1527[7], v1527[8], v1527[13], m1426);
//        G_b(8, 7, v1527[3], v1527[4], v1527[9], v1527[14], m1426);
//        ;
//        G_b(9, 0, v1527[0], v1527[4], v1527[8], v1527[12], m1426);
//        G_b(9, 1, v1527[1], v1527[5], v1527[9], v1527[13], m1426);
//        G_b(9, 2, v1527[2], v1527[6], v1527[10], v1527[14], m1426);
//        G_b(9, 3, v1527[3], v1527[7], v1527[11], v1527[15], m1426);
//        G_b(9, 4, v1527[0], v1527[5], v1527[10], v1527[15], m1426);
//        G_b(9, 5, v1527[1], v1527[6], v1527[11], v1527[12], m1426);
//        G_b(9, 6, v1527[2], v1527[7], v1527[8], v1527[13], m1426);
//        G_b(9, 7, v1527[3], v1527[4], v1527[9], v1527[14], m1426);
//        ;
//        G_b(10, 0, v1527[0], v1527[4], v1527[8], v1527[12], m1426);
//        G_b(10, 1, v1527[1], v1527[5], v1527[9], v1527[13], m1426);
//        G_b(10, 2, v1527[2], v1527[6], v1527[10], v1527[14], m1426);
//        G_b(10, 3, v1527[3], v1527[7], v1527[11], v1527[15], m1426);
//        G_b(10, 4, v1527[0], v1527[5], v1527[10], v1527[15], m1426);
//        G_b(10, 5, v1527[1], v1527[6], v1527[11], v1527[12], m1426);
//        G_b(10, 6, v1527[2], v1527[7], v1527[8], v1527[13], m1426);
//        G_b(10, 7, v1527[3], v1527[4], v1527[9], v1527[14], m1426);
//        ;
//        G_b(11, 0, v1527[0], v1527[4], v1527[8], v1527[12], m1426);
//        G_b(11, 1, v1527[1], v1527[5], v1527[9], v1527[13], m1426);
//        G_b(11, 2, v1527[2], v1527[6], v1527[10], v1527[14], m1426);
//        G_b(11, 3, v1527[3], v1527[7], v1527[11], v1527[15], m1426);
//        G_b(11, 4, v1527[0], v1527[5], v1527[10], v1527[15], m1426);
//        G_b(11, 5, v1527[1], v1527[6], v1527[11], v1527[12], m1426);
//        G_b(11, 6, v1527[2], v1527[7], v1527[8], v1527[13], m1426);
//        G_b(11, 7, v1527[3], v1527[4], v1527[9], v1527[14], m1426);
//        ;
//        uint2 last1628;
//        last1628 = vectorize(v1527[3] ^ v1527[11] ^ 11912009170470909681UL);
//        if (last1628.y <= target21218.y && last1628.x <= target21218.x) {
//            resNonce1117[1] = resNonce1117[0];
//            resNonce1117[0] = nonce1325;
//        }
//    }
//    label_0:;
//    if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=128 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 256)) goto label_1;
//    unsigned int blockDim_x_2;
//    blockDim_x_2 = 128;
//    unsigned int threadIdx_x_2;
//    threadIdx_x_2 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) % 128;
//    unsigned int blockDim_y_2;
//    blockDim_y_2 = 1;
//    unsigned int threadIdx_y_2;
//    threadIdx_y_2 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) / 128 % 1;
//    unsigned int blockDim_z_2;
//    blockDim_z_2 = 1;
//    unsigned int threadIdx_z_2;
//    threadIdx_z_2 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) / 128;
//    unsigned int blockDim_x_233;
//    blockDim_x_233 = 128;
//    unsigned int threadIdx_x_234;
//    threadIdx_x_234 = ((threadIdx_x_2 + threadIdx_y_2 * blockDim_x_2 + threadIdx_z_2 * blockDim_x_2 * blockDim_y_2) - 128) % 128;
//    unsigned int blockDim_y_235;
//    blockDim_y_235 = 1;
//    unsigned int threadIdx_y_236;
//    threadIdx_y_236 = ((threadIdx_x_2 + threadIdx_y_2 * blockDim_x_2 + threadIdx_z_2 * blockDim_x_2 * blockDim_y_2) - 128) / 128 % 1;
//    unsigned int blockDim_z_237;
//    blockDim_z_237 = 1;
//    unsigned int threadIdx_z_238;
//    threadIdx_z_238 = ((threadIdx_x_2 + threadIdx_y_2 * blockDim_x_2 + threadIdx_z_2 * blockDim_x_2 * blockDim_y_2) - 128) / 128;
//    for (int i = 0; i < 80; i++) {
//        uint32_t nonce2139;
//        nonce2139 = (blockDim_x_233 * blockIdx.x + threadIdx_x_234) * 80 + i + startNonce1830;
//        static uint64_t s_target2240 __attribute__((shared));
//        if (!threadIdx_x_234)
//            s_target2240 = devectorize(target22032);
//        uint64_t m2341[16];
//        m2341[0] = d_data2[0];
//        m2341[1] = d_data2[1];
//        m2341[2] = d_data2[2];
//        m2341[3] = d_data2[3];
//        m2341[4] = d_data2[4] | nonce2139;
//        m2341[5] = d_data2[5];
//        m2341[6] = d_data2[6];
//        m2341[7] = d_data2[7];
//        m2341[8] = d_data2[8];
//        m2341[9] = d_data2[9];
//        m2341[10] = m2341[11] = 0;
//        m2341[12] = m2341[13] = m2341[14] = m2341[15] = 0;
//        uint64_t v2442[16] = {7640891576939301160L, 13503953896175478587UL, 4354685564936845355L, 11912009170470909681UL, 5840696475078001361L, 11170449401992604703UL, 2270897969802886507L, 6620516959819538809L, 7640891576956012808L, 13503953896175478587UL, 4354685564936845355L, 11912009170470909681UL, 5840696475078001281L, 11170449401992604703UL, 16175846103906665108UL, 6620516959819538809L};
//        G(0, 0, v2442[0], v2442[4], v2442[8], v2442[12], m2341);
//        G(0, 1, v2442[1], v2442[5], v2442[9], v2442[13], m2341);
//        G(0, 2, v2442[2], v2442[6], v2442[10], v2442[14], m2341);
//        G(0, 3, v2442[3], v2442[7], v2442[11], v2442[15], m2341);
//        G(0, 4, v2442[0], v2442[5], v2442[10], v2442[15], m2341);
//        G(0, 5, v2442[1], v2442[6], v2442[11], v2442[12], m2341);
//        G(0, 6, v2442[2], v2442[7], v2442[8], v2442[13], m2341);
//        G(0, 7, v2442[3], v2442[4], v2442[9], v2442[14], m2341);
//        ;
//        G(1, 0, v2442[0], v2442[4], v2442[8], v2442[12], m2341);
//        G(1, 1, v2442[1], v2442[5], v2442[9], v2442[13], m2341);
//        G(1, 2, v2442[2], v2442[6], v2442[10], v2442[14], m2341);
//        G(1, 3, v2442[3], v2442[7], v2442[11], v2442[15], m2341);
//        G(1, 4, v2442[0], v2442[5], v2442[10], v2442[15], m2341);
//        G(1, 5, v2442[1], v2442[6], v2442[11], v2442[12], m2341);
//        G(1, 6, v2442[2], v2442[7], v2442[8], v2442[13], m2341);
//        G(1, 7, v2442[3], v2442[4], v2442[9], v2442[14], m2341);
//        ;
//        G(2, 0, v2442[0], v2442[4], v2442[8], v2442[12], m2341);
//        G(2, 1, v2442[1], v2442[5], v2442[9], v2442[13], m2341);
//        G(2, 2, v2442[2], v2442[6], v2442[10], v2442[14], m2341);
//        G(2, 3, v2442[3], v2442[7], v2442[11], v2442[15], m2341);
//        G(2, 4, v2442[0], v2442[5], v2442[10], v2442[15], m2341);
//        G(2, 5, v2442[1], v2442[6], v2442[11], v2442[12], m2341);
//        G(2, 6, v2442[2], v2442[7], v2442[8], v2442[13], m2341);
//        G(2, 7, v2442[3], v2442[4], v2442[9], v2442[14], m2341);
//        ;
//        G(3, 0, v2442[0], v2442[4], v2442[8], v2442[12], m2341);
//        G(3, 1, v2442[1], v2442[5], v2442[9], v2442[13], m2341);
//        G(3, 2, v2442[2], v2442[6], v2442[10], v2442[14], m2341);
//        G(3, 3, v2442[3], v2442[7], v2442[11], v2442[15], m2341);
//        G(3, 4, v2442[0], v2442[5], v2442[10], v2442[15], m2341);
//        G(3, 5, v2442[1], v2442[6], v2442[11], v2442[12], m2341);
//        G(3, 6, v2442[2], v2442[7], v2442[8], v2442[13], m2341);
//        G(3, 7, v2442[3], v2442[4], v2442[9], v2442[14], m2341);
//        ;
//        G(4, 0, v2442[0], v2442[4], v2442[8], v2442[12], m2341);
//        G(4, 1, v2442[1], v2442[5], v2442[9], v2442[13], m2341);
//        G(4, 2, v2442[2], v2442[6], v2442[10], v2442[14], m2341);
//        G(4, 3, v2442[3], v2442[7], v2442[11], v2442[15], m2341);
//        G(4, 4, v2442[0], v2442[5], v2442[10], v2442[15], m2341);
//        G(4, 5, v2442[1], v2442[6], v2442[11], v2442[12], m2341);
//        G(4, 6, v2442[2], v2442[7], v2442[8], v2442[13], m2341);
//        G(4, 7, v2442[3], v2442[4], v2442[9], v2442[14], m2341);
//        ;
//        G(5, 0, v2442[0], v2442[4], v2442[8], v2442[12], m2341);
//        G(5, 1, v2442[1], v2442[5], v2442[9], v2442[13], m2341);
//        G(5, 2, v2442[2], v2442[6], v2442[10], v2442[14], m2341);
//        G(5, 3, v2442[3], v2442[7], v2442[11], v2442[15], m2341);
//        G(5, 4, v2442[0], v2442[5], v2442[10], v2442[15], m2341);
//        G(5, 5, v2442[1], v2442[6], v2442[11], v2442[12], m2341);
//        G(5, 6, v2442[2], v2442[7], v2442[8], v2442[13], m2341);
//        G(5, 7, v2442[3], v2442[4], v2442[9], v2442[14], m2341);
//        ;
//        G(6, 0, v2442[0], v2442[4], v2442[8], v2442[12], m2341);
//        G(6, 1, v2442[1], v2442[5], v2442[9], v2442[13], m2341);
//        G(6, 2, v2442[2], v2442[6], v2442[10], v2442[14], m2341);
//        G(6, 3, v2442[3], v2442[7], v2442[11], v2442[15], m2341);
//        G(6, 4, v2442[0], v2442[5], v2442[10], v2442[15], m2341);
//        G(6, 5, v2442[1], v2442[6], v2442[11], v2442[12], m2341);
//        G(6, 6, v2442[2], v2442[7], v2442[8], v2442[13], m2341);
//        G(6, 7, v2442[3], v2442[4], v2442[9], v2442[14], m2341);
//        ;
//        G(7, 0, v2442[0], v2442[4], v2442[8], v2442[12], m2341);
//        G(7, 1, v2442[1], v2442[5], v2442[9], v2442[13], m2341);
//        G(7, 2, v2442[2], v2442[6], v2442[10], v2442[14], m2341);
//        G(7, 3, v2442[3], v2442[7], v2442[11], v2442[15], m2341);
//        G(7, 4, v2442[0], v2442[5], v2442[10], v2442[15], m2341);
//        G(7, 5, v2442[1], v2442[6], v2442[11], v2442[12], m2341);
//        G(7, 6, v2442[2], v2442[7], v2442[8], v2442[13], m2341);
//        G(7, 7, v2442[3], v2442[4], v2442[9], v2442[14], m2341);
//        ;
//        G(8, 0, v2442[0], v2442[4], v2442[8], v2442[12], m2341);
//        G(8, 1, v2442[1], v2442[5], v2442[9], v2442[13], m2341);
//        G(8, 2, v2442[2], v2442[6], v2442[10], v2442[14], m2341);
//        G(8, 3, v2442[3], v2442[7], v2442[11], v2442[15], m2341);
//        G(8, 4, v2442[0], v2442[5], v2442[10], v2442[15], m2341);
//        G(8, 5, v2442[1], v2442[6], v2442[11], v2442[12], m2341);
//        G(8, 6, v2442[2], v2442[7], v2442[8], v2442[13], m2341);
//        G(8, 7, v2442[3], v2442[4], v2442[9], v2442[14], m2341);
//        ;
//        G(9, 0, v2442[0], v2442[4], v2442[8], v2442[12], m2341);
//        G(9, 1, v2442[1], v2442[5], v2442[9], v2442[13], m2341);
//        G(9, 2, v2442[2], v2442[6], v2442[10], v2442[14], m2341);
//        G(9, 3, v2442[3], v2442[7], v2442[11], v2442[15], m2341);
//        G(9, 4, v2442[0], v2442[5], v2442[10], v2442[15], m2341);
//        G(9, 5, v2442[1], v2442[6], v2442[11], v2442[12], m2341);
//        G(9, 6, v2442[2], v2442[7], v2442[8], v2442[13], m2341);
//        G(9, 7, v2442[3], v2442[4], v2442[9], v2442[14], m2341);
//        ;
//        G(10, 0, v2442[0], v2442[4], v2442[8], v2442[12], m2341);
//        G(10, 1, v2442[1], v2442[5], v2442[9], v2442[13], m2341);
//        G(10, 2, v2442[2], v2442[6], v2442[10], v2442[14], m2341);
//        G(10, 3, v2442[3], v2442[7], v2442[11], v2442[15], m2341);
//        G(10, 4, v2442[0], v2442[5], v2442[10], v2442[15], m2341);
//        G(10, 5, v2442[1], v2442[6], v2442[11], v2442[12], m2341);
//        G(10, 6, v2442[2], v2442[7], v2442[8], v2442[13], m2341);
//        G(10, 7, v2442[3], v2442[4], v2442[9], v2442[14], m2341);
//        ;
//        G(11, 0, v2442[0], v2442[4], v2442[8], v2442[12], m2341);
//        G(11, 1, v2442[1], v2442[5], v2442[9], v2442[13], m2341);
//        G(11, 2, v2442[2], v2442[6], v2442[10], v2442[14], m2341);
//        G(11, 3, v2442[3], v2442[7], v2442[11], v2442[15], m2341);
//        G(11, 4, v2442[0], v2442[5], v2442[10], v2442[15], m2341);
//        G(11, 5, v2442[1], v2442[6], v2442[11], v2442[12], m2341);
//        H(11, 6, v2442[2], v2442[7], v2442[8], v2442[13], m2341);
//        ;
//        uint64_t h642543;
//        h642543 = cuda_swab64(7640891576939301160L ^ v2442[0] ^ v2442[8]);
//        if (h642543 <= s_target2240) {
//            resNonce1931[1] = resNonce1931[0];
//            resNonce1931[0] = nonce2139;
//            s_target2240 = h642543;
//        }
//    }
//    label_1:;
//    if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=256 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 384)) goto label_2;
//    unsigned int blockDim_x_0;
//    blockDim_x_0 = 128;
//    unsigned int threadIdx_x_0;
//    threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 256) % 128;
//    unsigned int blockDim_y_0;
//    blockDim_y_0 = 1;
//    unsigned int threadIdx_y_0;
//    threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 256) / 128 % 1;
//    unsigned int blockDim_z_0;
//    blockDim_z_0 = 1;
//    unsigned int threadIdx_z_0;
//    threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 256) / 128;
//    unsigned int blockDim_x_03;
//    blockDim_x_03 = 128;
//    unsigned int threadIdx_x_04;
//    threadIdx_x_04 = ((threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0 + threadIdx_z_0 * blockDim_x_0 * blockDim_y_0) - 256) % 128;
//    unsigned int blockDim_y_05;
//    blockDim_y_05 = 1;
//    unsigned int threadIdx_y_06;
//    threadIdx_y_06 = ((threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0 + threadIdx_z_0 * blockDim_x_0 * blockDim_y_0) - 256) / 128 % 1;
//    unsigned int blockDim_z_07;
//    blockDim_z_07 = 1;
//    unsigned int threadIdx_z_08;
//    threadIdx_z_08 = ((threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0 + threadIdx_z_0 * blockDim_x_0 * blockDim_y_0) - 256) / 128;
//    for (int i = 0; i < 40; i++) {
//        uint32_t thread39;
//        thread39 = (blockDim_x_03 * blockIdx.x + threadIdx_x_04) * 40 + i;
//        static uint32_t s_K410[256] __attribute__((shared));
//        if (threadIdx_x_04 < 64U)
//            s_K410[threadIdx_x_04] = c_K[threadIdx_x_04];
//        if (thread39 < threads00) {
//            uint32_t nonce511;
//            nonce511 = startNonce11 + thread39;
//            uint32_t dat612[16];
//            *((uint2 *)(dat612)) = *((uint2 *)(c_dataEnd80));
//            dat612[2] = c_dataEnd80[2];
//            dat612[3] = nonce511;
//            dat612[4] = 2147483648U;
//            dat612[15] = 640;
//#pragma unroll (10)
//            for (int i = 5; i < 15; i++)
//                dat612[i] = 0;
//            uint32_t buf713[8];
//#pragma unroll (4)
//            for (int i = 0; i < 8; i += 2)
//                *((uint2 *)(&buf713[i])) = *((uint2 *)(&c_midstate76[i]));
//            sha256_round_body(dat612, buf713, s_K410);
//#pragma unroll (8)
//            for (int i = 0; i < 8; i++)
//                dat612[i] = buf713[i];
//            dat612[8] = 2147483648U;
//#pragma unroll (6)
//            for (int i = 9; i < 15; i++)
//                dat612[i] = 0;
//            dat612[15] = 256;
//#pragma unroll (8)
//            for (int i = 0; i < 8; i++)
//                buf713[i] = c_H256[i];
//            sha256_round_last(dat612, buf713, s_K410);
//            uint64_t high814;
//            high814 = cuda_swab32ll(((uint64_t *)buf713)[3]);
//            if (high814 <= c_target[0]) {
//                resNonces22[1] = atomicExch(resNonces22, nonce511);
//            }
//        }
//    }
//    label_2:;
//    if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=384 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)) goto label_3;
//    unsigned int blockDim_x_3;
//    blockDim_x_3 = 128;
//    unsigned int threadIdx_x_3;
//    threadIdx_x_3 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 384) % 128;
//    unsigned int blockDim_y_3;
//    blockDim_y_3 = 1;
//    unsigned int threadIdx_y_3;
//    threadIdx_y_3 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 384) / 128 % 1;
//    unsigned int blockDim_z_3;
//    blockDim_z_3 = 1;
//    unsigned int threadIdx_z_3;
//    threadIdx_z_3 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 384) / 128;
//    unsigned int blockDim_x_346;
//    blockDim_x_346 = 128;
//    unsigned int threadIdx_x_347;
//    threadIdx_x_347 = ((threadIdx_x_3 + threadIdx_y_3 * blockDim_x_3 + threadIdx_z_3 * blockDim_x_3 * blockDim_y_3) - 384) % 128;
//    unsigned int blockDim_y_348;
//    blockDim_y_348 = 1;
//    unsigned int threadIdx_y_349;
//    threadIdx_y_349 = ((threadIdx_x_3 + threadIdx_y_3 * blockDim_x_3 + threadIdx_z_3 * blockDim_x_3 * blockDim_y_3) - 384) / 128 % 1;
//    unsigned int blockDim_z_350;
//    blockDim_z_350 = 1;
//    unsigned int threadIdx_z_351;
//    threadIdx_z_351 = ((threadIdx_x_3 + threadIdx_y_3 * blockDim_x_3 + threadIdx_z_3 * blockDim_x_3 * blockDim_y_3) - 384) / 128;
//    uint32_t gid2852;
//    gid2852 = blockIdx.x * blockDim_x_346 + threadIdx_x_347;
//    uint2 mix2953[4];
//    uint64_t nonce3054;
//    nonce3054 = start_nonce2745 + gid2852;
//    uint2 *mix_hash3155;
//    mix_hash3155 = mix2953;
//    bool result3256;
//    result3256 = false;
//    uint2 state3357[12];
//    state3357[4] = vectorize(nonce3054);
//    keccak_f1600_init(state3357);
//    int thread_id3458;
//    thread_id3458 = threadIdx_x_347 & ((128 / 16) - 1);
//    int mix_idx3559;
//    mix_idx3559 = thread_id3458 & 3;
//    for (int i = 0; i < (128 / 16); i += 4) {
//        uint4 mix3761[4];
//        uint32_t offset3862[4];
//        uint32_t init03963[4];
//        for (int p = 0; p < 4; p++) {
//            uint2 shuffle4064[8];
//            for (int j = 0; j < 8; j++) {
//                shuffle4064[j].x = __shfl_sync(4294967295U, (state3357[j].x), (i + p), ((128 / 16)));
//                shuffle4064[j].y = __shfl_sync(4294967295U, (state3357[j].y), (i + p), ((128 / 16)));
//            }
//            switch (mix_idx3559) {
//            case 0:
//                mix3761[p] = vectorize2(shuffle4064[0], shuffle4064[1]);
//                break;
//            case 1:
//                mix3761[p] = vectorize2(shuffle4064[2], shuffle4064[3]);
//                break;
//            case 2:
//                mix3761[p] = vectorize2(shuffle4064[4], shuffle4064[5]);
//                break;
//            case 3:
//                mix3761[p] = vectorize2(shuffle4064[6], shuffle4064[7]);
//                break;
//            }
//            init03963[p] = __shfl_sync(4294967295U, (shuffle4064[0].x), (0), ((128 / 16)));
//        }
//        for (uint32_t a = 0; a < 64; a += 4) {
//            int t4165;
//            t4165 = bfe(a, 2U, 3U);
//            for (uint32_t b = 0; b < 4; b++) {
//                for (int p = 0; p < 4; p++) {
//                    offset3862[p] = ((init03963[p] ^ (a + b)) * 16777619 ^ (((uint32_t *)&mix3761[p])[b])) % d_dag_size;
//                    offset3862[p] = __shfl_sync(4294967295U, (offset3862[p]), (t4165), ((128 / 16)));
//                    mix3761[p] = fnv4(mix3761[p], d_dag[offset3862[p]].uint4s[thread_id3458]);
//                }
//            }
//        }
//        for (int p = 0; p < 4; p++) {
//            uint2 shuffle4266[4];
//            uint32_t thread_mix4367;
//            thread_mix4367 = fnv_reduce(mix3761[p]);
//            shuffle4266[0].x = __shfl_sync(4294967295U, (thread_mix4367), (0), ((128 / 16)));
//            shuffle4266[0].y = __shfl_sync(4294967295U, (thread_mix4367), (1), ((128 / 16)));
//            shuffle4266[1].x = __shfl_sync(4294967295U, (thread_mix4367), (2), ((128 / 16)));
//            shuffle4266[1].y = __shfl_sync(4294967295U, (thread_mix4367), (3), ((128 / 16)));
//            shuffle4266[2].x = __shfl_sync(4294967295U, (thread_mix4367), (4), ((128 / 16)));
//            shuffle4266[2].y = __shfl_sync(4294967295U, (thread_mix4367), (5), ((128 / 16)));
//            shuffle4266[3].x = __shfl_sync(4294967295U, (thread_mix4367), (6), ((128 / 16)));
//            shuffle4266[3].y = __shfl_sync(4294967295U, (thread_mix4367), (7), ((128 / 16)));
//            if ((i + p) == thread_id3458) {
//                state3357[8] = shuffle4266[0];
//                state3357[9] = shuffle4266[1];
//                state3357[10] = shuffle4266[2];
//                state3357[11] = shuffle4266[3];
//            }
//        }
//    }
//    if (!(cuda_swab64(keccak_f1600_final(state3357)) > d_target)) {
//        mix_hash3155[0] = state3357[8];
//        mix_hash3155[1] = state3357[9];
//        mix_hash3155[2] = state3357[10];
//        mix_hash3155[3] = state3357[11];
//        return;
//    }
//    uint32_t index3660;
//    index3660 = atomicInc((uint32_t *)&g_output2644->count, 4294967295U);
//    if (index3660 >= 4U)
//        return;
//    g_output2644->result[index3660].gid = gid2852;
//    g_output2644->result[index3660].mix[0] = mix2953[0].x;
//    g_output2644->result[index3660].mix[1] = mix2953[0].y;
//    g_output2644->result[index3660].mix[2] = mix2953[1].x;
//    g_output2644->result[index3660].mix[3] = mix2953[1].y;
//    g_output2644->result[index3660].mix[4] = mix2953[2].x;
//    g_output2644->result[index3660].mix[5] = mix2953[2].y;
//    g_output2644->result[index3660].mix[6] = mix2953[3].x;
//    g_output2644->result[index3660].mix[7] = mix2953[3].y;
//    label_3:;
//}
//
//
//
//
//
//}
//
//void four(uint32_t gridSize, uint32_t blockSize, cudaStream_t stream,
//                           volatile Search_results* g_output, uint64_t start_nonce)
//{
//
//    {
//        uint32_t resNonces[NBN] = { UINT32_MAX, UINT32_MAX };
//        uint32_t result = UINT32_MAX;
//
//        uint32_t threads = 1048576;
//        dim3 grid((threads + TPB-1)/TPB);
//        dim3 block(TPB);
//        int thr_id = 0;
//
//        cudaMalloc(&d_resNonces[thr_id], NBN * sizeof(uint32_t));
//        /* Check error on Ctrl+C or kill to prevent segfaults on exit */
//        if (cudaMemset(d_resNonces[thr_id], 0xff, NBN*sizeof(uint32_t)) != cudaSuccess)
//            return;
//
//        const uint2 target2 = make_uint2(3, 5);
//
////        uint32_t threads = 1048576;
////        dim3 grid((threads + TPB_blake -1)/ TPB_blake);
////        dim3 block(TPB_blake);
//        const uint32_t threads_sha256 = 1048576;
//        const uint32_t threadsperblock = 128;
//        dim3 grid_sha256(threads_sha256 /threadsperblock);
//        dim3 block_sha256(threadsperblock);
////        auto thr_id = 0;
//        CUDA_SAFE_CALL(cudaMalloc(&d_resNonces_blake[thr_id], NBN * sizeof(uint32_t)));
//        CUDA_SAFE_CALL(cudaMalloc(&d_sha256_resNonces[0], 2*sizeof(uint32_t)));
//        CUDA_SAFE_CALL(cudaMemset(d_sha256_resNonces[0], 0xFF, 2 * sizeof(uint32_t)));
//
//        if (cudaMemset(d_resNonces_blake[thr_id], 0xff, NBN*sizeof(uint32_t)) != cudaSuccess)
//            return;
//
////        const uint2 target2 = make_uint2(0, 1);
//        cudaStream_t t1;
//        cudaStream_t t2;
//        cudaStreamCreate ( &t1);
//        cudaStreamCreate ( &t2);
//        cudaStream_t t3;
//        cudaStream_t t4;
//        cudaStreamCreate ( &t3);
//        cudaStreamCreate ( &t4);
//
//        ethash_search<<<gridSize, blockSize, 0, t1>>>(
//            g_output, start_nonce
//            );
////        sia_blake2b_gpu_hash <<<grid, block, 8, t2>>> (
////            threads, 0, d_resNonces[thr_id], target2
////            );
//        blake2b_gpu_hash <<<grid, block, 8, t3>>> (
//            threads, 0, d_sha256_resNonces[thr_id], target2
//            );
//        sha256d_gpu_hash_shared <<<grid_sha256, block_sha256, 0, t4>>> (
//            threads_sha256 * 40, 0, d_sha256_resNonces[0]
//            );
//
//        cudaDeviceSynchronize();
//        FUNC<<<grid, blockSize * 3, 8>>>(
//            threads, 0, d_sha256_resNonces[thr_id], target2,
////                threads, 0, d_resNonces[thr_id], target2,
//                threads_sha256 * 40, 0, d_sha256_resNonces[0],
//                g_output, start_nonce
//        );
//
//        cudaThreadSynchronize();
//    }
//    CUDA_SAFE_CALL(cudaGetLastError());
//}
