#include "ethash_cuda_miner_kernel.h"

#include "ethash_cuda_miner_kernel_globals.h"

#include "cuda_helper.h"

#include "fnv.cuh"

#include <string.h>
#include <stdint.h>

#include "keccak.cuh"

#include "dagger_shuffled.cuh"

//#include <sph/blake2b.h>
#define TPB_sia 128
#define TPB_blake 128

static uint32_t* d_blake_resNonces[MAX_GPUS];

namespace foo {

static __constant__ const int8_t blake2b_sigma[12][16] = {
    { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15 } ,
    { 14, 10, 4,  8,  9,  15, 13, 6,  1,  12, 0,  2,  11, 7,  5,  3  } ,
    { 11, 8,  12, 0,  5,  2,  15, 13, 10, 14, 3,  6,  7,  1,  9,  4  } ,
    { 7,  9,  3,  1,  13, 12, 11, 14, 2,  6,  5,  10, 4,  0,  15, 8  } ,
    { 9,  0,  5,  7,  2,  4,  10, 15, 14, 1,  11, 12, 6,  8,  3,  13 } ,
    { 2,  12, 6,  10, 0,  11, 8,  3,  4,  13, 7,  5,  15, 14, 1,  9  } ,
    { 12, 5,  1,  15, 14, 13, 4,  10, 0,  7,  6,  3,  9,  2,  8,  11 } ,
    { 13, 11, 7,  14, 12, 1,  3,  9,  5,  0,  15, 4,  8,  6,  2,  10 } ,
    { 6,  15, 14, 9,  11, 3,  0,  8,  12, 2,  13, 7,  1,  4,  10, 5  } ,
    { 10, 2,  8,  4,  7,  6,  1,  5,  15, 11, 9,  14, 3,  12, 13, 0  } ,
    { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15 } ,
    { 14, 10, 4,  8,  9,  15, 13, 6,  1,  12, 0,  2,  11, 7,  5,  3  }
};


static uint32_t * d_sia_resNonces[MAX_GPUS];

__device__ uint64_t d_data2[10];

// host mem align
#define A 64
//
//extern "C" void sia_blake2b_hash(void *output, const void *input)
//{
//    uint8_t _ALIGN(A) hash[32];
//    blake2b_ctx ctx;
//
//    blake2b_init(&ctx, 32, NULL, 0);
//    blake2b_update(&ctx, input, 80);
//    blake2b_final(&ctx, hash);
//
//    memcpy(output, hash, 32);
//}
//
// ----------------------------------------------------------------

__device__ __forceinline__
static void G(const int r, const int i, uint64_t &a, uint64_t &b, uint64_t &c, uint64_t &d, uint64_t const m[16])
{
    a = a + b + m[ blake2b_sigma[r][2*i] ];
    ((uint2*)&d)[0] = SWAPUINT2( ((uint2*)&d)[0] ^ ((uint2*)&a)[0] );
    c = c + d;
    ((uint2*)&b)[0] = ROR24( ((uint2*)&b)[0] ^ ((uint2*)&c)[0] );
    a = a + b + m[ blake2b_sigma[r][2*i+1] ];
    ((uint2*)&d)[0] = ROR16( ((uint2*)&d)[0] ^ ((uint2*)&a)[0] );
    c = c + d;
    ((uint2*)&b)[0] = ROR2( ((uint2*)&b)[0] ^ ((uint2*)&c)[0], 63U);
}

#define ROUND(r) \
	G(r, 0, v[0], v[4], v[ 8], v[12], m); \
	G(r, 1, v[1], v[5], v[ 9], v[13], m); \
	G(r, 2, v[2], v[6], v[10], v[14], m); \
	G(r, 3, v[3], v[7], v[11], v[15], m); \
	G(r, 4, v[0], v[5], v[10], v[15], m); \
	G(r, 5, v[1], v[6], v[11], v[12], m); \
	G(r, 6, v[2], v[7], v[ 8], v[13], m); \
	G(r, 7, v[3], v[4], v[ 9], v[14], m);

// simplified for the last round
__device__ __forceinline__
static void H(const int r, const int i, uint64_t &a, uint64_t &b, uint64_t &c, uint64_t &d, uint64_t const m[16])
{
    a = a + b + m[ blake2b_sigma[r][2*i] ];
    ((uint2*)&d)[0] = SWAPUINT2( ((uint2*)&d)[0] ^ ((uint2*)&a)[0] );
    c = c + d;
    ((uint2*)&b)[0] = ROR24( ((uint2*)&b)[0] ^ ((uint2*)&c)[0] );
    a = a + b + m[ blake2b_sigma[r][2*i+1] ];
    ((uint2*)&d)[0] = ROR16( ((uint2*)&d)[0] ^ ((uint2*)&a)[0] );
    c = c + d;
}

// we only check v[0] and v[8]
#define ROUND_F(r) \
	G(r, 0, v[0], v[4], v[ 8], v[12], m); \
	G(r, 1, v[1], v[5], v[ 9], v[13], m); \
	G(r, 2, v[2], v[6], v[10], v[14], m); \
	G(r, 3, v[3], v[7], v[11], v[15], m); \
	G(r, 4, v[0], v[5], v[10], v[15], m); \
	G(r, 5, v[1], v[6], v[11], v[12], m); \
	H(r, 6, v[2], v[7], v[ 8], v[13], m);

__global__
//__launch_bounds__(128, 8) /* to force 64 regs */
void sia_blake2b_gpu_hash(const uint32_t threads, const uint32_t startNonce, uint32_t *resNonce, const uint2 target2)
{
    for (int i = 0; i < 80; i++) {
        const uint32_t nonce = (blockDim.x * blockIdx.x + threadIdx.x) * 80 + i + startNonce;
        __shared__ uint64_t s_target;
        if (!threadIdx.x) s_target = devectorize(target2);

        uint64_t m[16];

        m[0] = d_data2[0];
        m[1] = d_data2[1];
        m[2] = d_data2[2];
        m[3] = d_data2[3];
        m[4] = d_data2[4] | nonce;
        m[5] = d_data2[5];
        m[6] = d_data2[6];
        m[7] = d_data2[7];
        m[8] = d_data2[8];
        m[9] = d_data2[9];

        m[10] = m[11] = 0;
        m[12] = m[13] = m[14] = m[15] = 0;

        uint64_t v[16] = {
            0x6a09e667f2bdc928, 0xbb67ae8584caa73b, 0x3c6ef372fe94f82b, 0xa54ff53a5f1d36f1,
            0x510e527fade682d1, 0x9b05688c2b3e6c1f, 0x1f83d9abfb41bd6b, 0x5be0cd19137e2179,
            0x6a09e667f3bcc908, 0xbb67ae8584caa73b, 0x3c6ef372fe94f82b, 0xa54ff53a5f1d36f1,
            0x510e527fade68281, 0x9b05688c2b3e6c1f, 0xe07c265404be4294, 0x5be0cd19137e2179
        };

        ROUND( 0 );
        ROUND( 1 );
        ROUND( 2 );
        ROUND( 3 );
        ROUND( 4 );
        ROUND( 5 );
        ROUND( 6 );
        ROUND( 7 );
        ROUND( 8 );
        ROUND( 9 );
        ROUND( 10 );
        ROUND_F( 11 );

        uint64_t h64 = cuda_swab64(0x6a09e667f2bdc928 ^ v[0] ^ v[8]);
        if (h64 <= s_target) {
            resNonce[1] = resNonce[0];
            resNonce[0] = nonce;
            s_target = h64;
        }
    }
    // if (!nonce) printf("%016lx ", s_target);
}

static uint32_t* d_resNonces_blake[MAX_GPUS];

__device__ uint64_t d_blake_data[10];

#define AS_U32(addr) *((uint32_t*)(addr))

static __constant__ const int8_t blake2b_sigma_blake[12][16] = {
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3},
    {11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4},
    {7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8},
    {9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13},
    {2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9},
    {12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11},
    {13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10},
    {6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5},
    {10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0},
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3}};

// host mem align
#define A 64
//
// extern "C" void blake2b_hash(void *output, const void *input)
//{
//    uint8_t _ALIGN(A) hash[32];
//    blake2b_ctx ctx;
//
//    blake2b_init(&ctx, 32, NULL, 0);
//    blake2b_update(&ctx, input, 80);
//    blake2b_final(&ctx, hash);
//
//    memcpy(output, hash, 32);
//}

// ----------------------------------------------------------------

__device__ __forceinline__ static void G_b(const int r, const int i, uint64_t& a, uint64_t& b,
                                           uint64_t& c, uint64_t& d, uint64_t const m[16])
{
    a = a + b + m[blake2b_sigma_blake[r][2 * i]];
    ((uint2*)&d)[0] = SWAPUINT2(((uint2*)&d)[0] ^ ((uint2*)&a)[0]);
    c = c + d;
    ((uint2*)&b)[0] = ROR24(((uint2*)&b)[0] ^ ((uint2*)&c)[0]);
    a = a + b + m[blake2b_sigma_blake[r][2 * i + 1]];
    ((uint2*)&d)[0] = ROR16(((uint2*)&d)[0] ^ ((uint2*)&a)[0]);
    c = c + d;
    ((uint2*)&b)[0] = ROR2(((uint2*)&b)[0] ^ ((uint2*)&c)[0], 63U);
}

#define ROUND(r) \
	G_b(r, 0, v[0], v[4], v[8], v[12], m); \
	G_b(r, 1, v[1], v[5], v[9], v[13], m); \
	G_b(r, 2, v[2], v[6], v[10], v[14], m); \
	G_b(r, 3, v[3], v[7], v[11], v[15], m); \
	G_b(r, 4, v[0], v[5], v[10], v[15], m); \
	G_b(r, 5, v[1], v[6], v[11], v[12], m); \
	G_b(r, 6, v[2], v[7], v[8], v[13], m); \
	G_b(r, 7, v[3], v[4], v[9], v[14], m);

__global__ void blake2b_gpu_hash(
    const uint32_t threads, const uint32_t startNonce, uint32_t* resNonce, const uint2 target2)
{
    for (int i = 0; i < 80; i++)
    {
        const uint32_t nonce = (blockDim.x * blockIdx.x + threadIdx.x) * 80 + i + startNonce;

        uint64_t m[16];

        m[0] = d_blake_data[0];
        m[1] = d_blake_data[1];
        m[2] = d_blake_data[2];
        m[3] = d_blake_data[3];
        m[4] = d_blake_data[4];
        m[5] = d_blake_data[5];
        m[6] = d_blake_data[6];
        m[7] = d_blake_data[7];
        m[8] = d_blake_data[8];
        ((uint32_t*)m)[18] = AS_U32(&d_blake_data[9]);
        ((uint32_t*)m)[19] = nonce;

        m[10] = m[11] = 0;
        m[12] = m[13] = 0;
        m[14] = m[15] = 0;

        uint64_t v[16] = {0x6a09e667f2bdc928, 0xbb67ae8584caa73b, 0x3c6ef372fe94f82b,
                          0xa54ff53a5f1d36f1, 0x510e527fade682d1, 0x9b05688c2b3e6c1f, 0x1f83d9abfb41bd6b,
                          0x5be0cd19137e2179, 0x6a09e667f3bcc908, 0xbb67ae8584caa73b, 0x3c6ef372fe94f82b,
                          0xa54ff53a5f1d36f1, 0x510e527fade68281, 0x9b05688c2b3e6c1f, 0xe07c265404be4294,
                          0x5be0cd19137e2179};

        ROUND(0);
        ROUND(1);
        ROUND(2);
        ROUND(3);
        ROUND(4);
        ROUND(5);
        ROUND(6);
        ROUND(7);
        ROUND(8);
        ROUND(9);
        ROUND(10);
        ROUND(11);

        uint2 last = vectorize(v[3] ^ v[11] ^ 0xa54ff53a5f1d36f1);
        if (last.y <= target2.y && last.x <= target2.x)
        {
            resNonce[1] = resNonce[0];
            resNonce[0] = nonce;
        }
    }
}

__attribute__((global)) void blake2b_gpu_hash_sia_blake2b_gpu_hash_100(const uint32_t threads9, const uint32_t startNonce10, uint32_t *resNonce11, const uint2 target212, const uint32_t threads0, const uint32_t startNonce1, uint32_t *resNonce2, const uint2 target23)
{
    if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 128)){
        unsigned int blockDim_x_1;
        blockDim_x_1 = 128;
        unsigned int threadIdx_x_1;
        threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 128;
        unsigned int blockDim_y_1;
        blockDim_y_1 = 1;
        unsigned int threadIdx_y_1;
        threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128 % 1;
        unsigned int blockDim_z_1;
        blockDim_z_1 = 1;
        unsigned int threadIdx_z_1;
        threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128;
        for (int i = 0; i < 80; i++) {
            uint32_t nonce13;
            nonce13 = (blockDim_x_1 * blockIdx.x + threadIdx_x_1) * 80 + i + startNonce10;
            uint64_t m14[16];
            m14[0] = d_blake_data[0];
            m14[1] = d_blake_data[1];
            m14[2] = d_blake_data[2];
            m14[3] = d_blake_data[3];
            m14[4] = d_blake_data[4];
            m14[5] = d_blake_data[5];
            m14[6] = d_blake_data[6];
            m14[7] = d_blake_data[7];
            m14[8] = d_blake_data[8];
            ((uint32_t *)m14)[18] = *((uint32_t *)(&d_blake_data[9]));
            ((uint32_t *)m14)[19] = nonce13;
            m14[10] = m14[11] = 0;
            m14[12] = m14[13] = 0;
            m14[14] = m14[15] = 0;
            uint64_t v15[16] = {7640891576939301160L, 13503953896175478587UL, 4354685564936845355L, 11912009170470909681UL, 5840696475078001361L, 11170449401992604703UL, 2270897969802886507L, 6620516959819538809L, 7640891576956012808L, 13503953896175478587UL, 4354685564936845355L, 11912009170470909681UL, 5840696475078001281L, 11170449401992604703UL, 16175846103906665108UL, 6620516959819538809L};
            G_b(0, 0, v15[0], v15[4], v15[8], v15[12], m14);
            G_b(0, 1, v15[1], v15[5], v15[9], v15[13], m14);
            G_b(0, 2, v15[2], v15[6], v15[10], v15[14], m14);
            G_b(0, 3, v15[3], v15[7], v15[11], v15[15], m14);
            G_b(0, 4, v15[0], v15[5], v15[10], v15[15], m14);
            G_b(0, 5, v15[1], v15[6], v15[11], v15[12], m14);
            G_b(0, 6, v15[2], v15[7], v15[8], v15[13], m14);
            G_b(0, 7, v15[3], v15[4], v15[9], v15[14], m14);
            ;
            G_b(1, 0, v15[0], v15[4], v15[8], v15[12], m14);
            G_b(1, 1, v15[1], v15[5], v15[9], v15[13], m14);
            G_b(1, 2, v15[2], v15[6], v15[10], v15[14], m14);
            G_b(1, 3, v15[3], v15[7], v15[11], v15[15], m14);
            G_b(1, 4, v15[0], v15[5], v15[10], v15[15], m14);
            G_b(1, 5, v15[1], v15[6], v15[11], v15[12], m14);
            G_b(1, 6, v15[2], v15[7], v15[8], v15[13], m14);
            G_b(1, 7, v15[3], v15[4], v15[9], v15[14], m14);
            ;
            G_b(2, 0, v15[0], v15[4], v15[8], v15[12], m14);
            G_b(2, 1, v15[1], v15[5], v15[9], v15[13], m14);
            G_b(2, 2, v15[2], v15[6], v15[10], v15[14], m14);
            G_b(2, 3, v15[3], v15[7], v15[11], v15[15], m14);
            G_b(2, 4, v15[0], v15[5], v15[10], v15[15], m14);
            G_b(2, 5, v15[1], v15[6], v15[11], v15[12], m14);
            G_b(2, 6, v15[2], v15[7], v15[8], v15[13], m14);
            G_b(2, 7, v15[3], v15[4], v15[9], v15[14], m14);
            ;
            G_b(3, 0, v15[0], v15[4], v15[8], v15[12], m14);
            G_b(3, 1, v15[1], v15[5], v15[9], v15[13], m14);
            G_b(3, 2, v15[2], v15[6], v15[10], v15[14], m14);
            G_b(3, 3, v15[3], v15[7], v15[11], v15[15], m14);
            G_b(3, 4, v15[0], v15[5], v15[10], v15[15], m14);
            G_b(3, 5, v15[1], v15[6], v15[11], v15[12], m14);
            G_b(3, 6, v15[2], v15[7], v15[8], v15[13], m14);
            G_b(3, 7, v15[3], v15[4], v15[9], v15[14], m14);
            ;
            G_b(4, 0, v15[0], v15[4], v15[8], v15[12], m14);
            G_b(4, 1, v15[1], v15[5], v15[9], v15[13], m14);
            G_b(4, 2, v15[2], v15[6], v15[10], v15[14], m14);
            G_b(4, 3, v15[3], v15[7], v15[11], v15[15], m14);
            G_b(4, 4, v15[0], v15[5], v15[10], v15[15], m14);
            G_b(4, 5, v15[1], v15[6], v15[11], v15[12], m14);
            G_b(4, 6, v15[2], v15[7], v15[8], v15[13], m14);
            G_b(4, 7, v15[3], v15[4], v15[9], v15[14], m14);
            ;
            G_b(5, 0, v15[0], v15[4], v15[8], v15[12], m14);
            G_b(5, 1, v15[1], v15[5], v15[9], v15[13], m14);
            G_b(5, 2, v15[2], v15[6], v15[10], v15[14], m14);
            G_b(5, 3, v15[3], v15[7], v15[11], v15[15], m14);
            G_b(5, 4, v15[0], v15[5], v15[10], v15[15], m14);
            G_b(5, 5, v15[1], v15[6], v15[11], v15[12], m14);
            G_b(5, 6, v15[2], v15[7], v15[8], v15[13], m14);
            G_b(5, 7, v15[3], v15[4], v15[9], v15[14], m14);
            ;
            G_b(6, 0, v15[0], v15[4], v15[8], v15[12], m14);
            G_b(6, 1, v15[1], v15[5], v15[9], v15[13], m14);
            G_b(6, 2, v15[2], v15[6], v15[10], v15[14], m14);
            G_b(6, 3, v15[3], v15[7], v15[11], v15[15], m14);
            G_b(6, 4, v15[0], v15[5], v15[10], v15[15], m14);
            G_b(6, 5, v15[1], v15[6], v15[11], v15[12], m14);
            G_b(6, 6, v15[2], v15[7], v15[8], v15[13], m14);
            G_b(6, 7, v15[3], v15[4], v15[9], v15[14], m14);
            ;
            G_b(7, 0, v15[0], v15[4], v15[8], v15[12], m14);
            G_b(7, 1, v15[1], v15[5], v15[9], v15[13], m14);
            G_b(7, 2, v15[2], v15[6], v15[10], v15[14], m14);
            G_b(7, 3, v15[3], v15[7], v15[11], v15[15], m14);
            G_b(7, 4, v15[0], v15[5], v15[10], v15[15], m14);
            G_b(7, 5, v15[1], v15[6], v15[11], v15[12], m14);
            G_b(7, 6, v15[2], v15[7], v15[8], v15[13], m14);
            G_b(7, 7, v15[3], v15[4], v15[9], v15[14], m14);
            ;
            G_b(8, 0, v15[0], v15[4], v15[8], v15[12], m14);
            G_b(8, 1, v15[1], v15[5], v15[9], v15[13], m14);
            G_b(8, 2, v15[2], v15[6], v15[10], v15[14], m14);
            G_b(8, 3, v15[3], v15[7], v15[11], v15[15], m14);
            G_b(8, 4, v15[0], v15[5], v15[10], v15[15], m14);
            G_b(8, 5, v15[1], v15[6], v15[11], v15[12], m14);
            G_b(8, 6, v15[2], v15[7], v15[8], v15[13], m14);
            G_b(8, 7, v15[3], v15[4], v15[9], v15[14], m14);
            ;
            G_b(9, 0, v15[0], v15[4], v15[8], v15[12], m14);
            G_b(9, 1, v15[1], v15[5], v15[9], v15[13], m14);
            G_b(9, 2, v15[2], v15[6], v15[10], v15[14], m14);
            G_b(9, 3, v15[3], v15[7], v15[11], v15[15], m14);
            G_b(9, 4, v15[0], v15[5], v15[10], v15[15], m14);
            G_b(9, 5, v15[1], v15[6], v15[11], v15[12], m14);
            G_b(9, 6, v15[2], v15[7], v15[8], v15[13], m14);
            G_b(9, 7, v15[3], v15[4], v15[9], v15[14], m14);
            ;
            G_b(10, 0, v15[0], v15[4], v15[8], v15[12], m14);
            G_b(10, 1, v15[1], v15[5], v15[9], v15[13], m14);
            G_b(10, 2, v15[2], v15[6], v15[10], v15[14], m14);
            G_b(10, 3, v15[3], v15[7], v15[11], v15[15], m14);
            G_b(10, 4, v15[0], v15[5], v15[10], v15[15], m14);
            G_b(10, 5, v15[1], v15[6], v15[11], v15[12], m14);
            G_b(10, 6, v15[2], v15[7], v15[8], v15[13], m14);
            G_b(10, 7, v15[3], v15[4], v15[9], v15[14], m14);
            ;
            G_b(11, 0, v15[0], v15[4], v15[8], v15[12], m14);
            G_b(11, 1, v15[1], v15[5], v15[9], v15[13], m14);
            G_b(11, 2, v15[2], v15[6], v15[10], v15[14], m14);
            G_b(11, 3, v15[3], v15[7], v15[11], v15[15], m14);
            G_b(11, 4, v15[0], v15[5], v15[10], v15[15], m14);
            G_b(11, 5, v15[1], v15[6], v15[11], v15[12], m14);
            G_b(11, 6, v15[2], v15[7], v15[8], v15[13], m14);
            G_b(11, 7, v15[3], v15[4], v15[9], v15[14], m14);
            ;
            uint2 last16;
            last16 = vectorize(v15[3] ^ v15[11] ^ 11912009170470909681UL);
            if (last16.y <= target212.y && last16.x <= target212.x) {
                resNonce11[1] = resNonce11[0];
                resNonce11[0] = nonce13;
            }
        }
    }
    if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 128)){
        unsigned int blockDim_x_0;
        blockDim_x_0 = 128;
        unsigned int threadIdx_x_0;
        threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 128;
        unsigned int blockDim_y_0;
        blockDim_y_0 = 1;
        unsigned int threadIdx_y_0;
        threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128 % 1;
        unsigned int blockDim_z_0;
        blockDim_z_0 = 1;
        unsigned int threadIdx_z_0;
        threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128;
        for (int i = 0; i < 80; i++) {
            uint32_t nonce4;
            nonce4 = (blockDim_x_0 * blockIdx.x + threadIdx_x_0) * 80 + i + startNonce1;
            static uint64_t s_target5 __attribute__((shared));
            if (!threadIdx_x_0)
                s_target5 = devectorize(target23);
            uint64_t m6[16];
            m6[0] = d_data2[0];
            m6[1] = d_data2[1];
            m6[2] = d_data2[2];
            m6[3] = d_data2[3];
            m6[4] = d_data2[4] | nonce4;
            m6[5] = d_data2[5];
            m6[6] = d_data2[6];
            m6[7] = d_data2[7];
            m6[8] = d_data2[8];
            m6[9] = d_data2[9];
            m6[10] = m6[11] = 0;
            m6[12] = m6[13] = m6[14] = m6[15] = 0;
            uint64_t v7[16] = {7640891576939301160L, 13503953896175478587UL, 4354685564936845355L, 11912009170470909681UL, 5840696475078001361L, 11170449401992604703UL, 2270897969802886507L, 6620516959819538809L, 7640891576956012808L, 13503953896175478587UL, 4354685564936845355L, 11912009170470909681UL, 5840696475078001281L, 11170449401992604703UL, 16175846103906665108UL, 6620516959819538809L};
            G(0, 0, v7[0], v7[4], v7[8], v7[12], m6);
            G(0, 1, v7[1], v7[5], v7[9], v7[13], m6);
            G(0, 2, v7[2], v7[6], v7[10], v7[14], m6);
            G(0, 3, v7[3], v7[7], v7[11], v7[15], m6);
            G(0, 4, v7[0], v7[5], v7[10], v7[15], m6);
            G(0, 5, v7[1], v7[6], v7[11], v7[12], m6);
            G(0, 6, v7[2], v7[7], v7[8], v7[13], m6);
            G(0, 7, v7[3], v7[4], v7[9], v7[14], m6);
            ;
            G(1, 0, v7[0], v7[4], v7[8], v7[12], m6);
            G(1, 1, v7[1], v7[5], v7[9], v7[13], m6);
            G(1, 2, v7[2], v7[6], v7[10], v7[14], m6);
            G(1, 3, v7[3], v7[7], v7[11], v7[15], m6);
            G(1, 4, v7[0], v7[5], v7[10], v7[15], m6);
            G(1, 5, v7[1], v7[6], v7[11], v7[12], m6);
            G(1, 6, v7[2], v7[7], v7[8], v7[13], m6);
            G(1, 7, v7[3], v7[4], v7[9], v7[14], m6);
            ;
            G(2, 0, v7[0], v7[4], v7[8], v7[12], m6);
            G(2, 1, v7[1], v7[5], v7[9], v7[13], m6);
            G(2, 2, v7[2], v7[6], v7[10], v7[14], m6);
            G(2, 3, v7[3], v7[7], v7[11], v7[15], m6);
            G(2, 4, v7[0], v7[5], v7[10], v7[15], m6);
            G(2, 5, v7[1], v7[6], v7[11], v7[12], m6);
            G(2, 6, v7[2], v7[7], v7[8], v7[13], m6);
            G(2, 7, v7[3], v7[4], v7[9], v7[14], m6);
            ;
            G(3, 0, v7[0], v7[4], v7[8], v7[12], m6);
            G(3, 1, v7[1], v7[5], v7[9], v7[13], m6);
            G(3, 2, v7[2], v7[6], v7[10], v7[14], m6);
            G(3, 3, v7[3], v7[7], v7[11], v7[15], m6);
            G(3, 4, v7[0], v7[5], v7[10], v7[15], m6);
            G(3, 5, v7[1], v7[6], v7[11], v7[12], m6);
            G(3, 6, v7[2], v7[7], v7[8], v7[13], m6);
            G(3, 7, v7[3], v7[4], v7[9], v7[14], m6);
            ;
            G(4, 0, v7[0], v7[4], v7[8], v7[12], m6);
            G(4, 1, v7[1], v7[5], v7[9], v7[13], m6);
            G(4, 2, v7[2], v7[6], v7[10], v7[14], m6);
            G(4, 3, v7[3], v7[7], v7[11], v7[15], m6);
            G(4, 4, v7[0], v7[5], v7[10], v7[15], m6);
            G(4, 5, v7[1], v7[6], v7[11], v7[12], m6);
            G(4, 6, v7[2], v7[7], v7[8], v7[13], m6);
            G(4, 7, v7[3], v7[4], v7[9], v7[14], m6);
            ;
            G(5, 0, v7[0], v7[4], v7[8], v7[12], m6);
            G(5, 1, v7[1], v7[5], v7[9], v7[13], m6);
            G(5, 2, v7[2], v7[6], v7[10], v7[14], m6);
            G(5, 3, v7[3], v7[7], v7[11], v7[15], m6);
            G(5, 4, v7[0], v7[5], v7[10], v7[15], m6);
            G(5, 5, v7[1], v7[6], v7[11], v7[12], m6);
            G(5, 6, v7[2], v7[7], v7[8], v7[13], m6);
            G(5, 7, v7[3], v7[4], v7[9], v7[14], m6);
            ;
            G(6, 0, v7[0], v7[4], v7[8], v7[12], m6);
            G(6, 1, v7[1], v7[5], v7[9], v7[13], m6);
            G(6, 2, v7[2], v7[6], v7[10], v7[14], m6);
            G(6, 3, v7[3], v7[7], v7[11], v7[15], m6);
            G(6, 4, v7[0], v7[5], v7[10], v7[15], m6);
            G(6, 5, v7[1], v7[6], v7[11], v7[12], m6);
            G(6, 6, v7[2], v7[7], v7[8], v7[13], m6);
            G(6, 7, v7[3], v7[4], v7[9], v7[14], m6);
            ;
            G(7, 0, v7[0], v7[4], v7[8], v7[12], m6);
            G(7, 1, v7[1], v7[5], v7[9], v7[13], m6);
            G(7, 2, v7[2], v7[6], v7[10], v7[14], m6);
            G(7, 3, v7[3], v7[7], v7[11], v7[15], m6);
            G(7, 4, v7[0], v7[5], v7[10], v7[15], m6);
            G(7, 5, v7[1], v7[6], v7[11], v7[12], m6);
            G(7, 6, v7[2], v7[7], v7[8], v7[13], m6);
            G(7, 7, v7[3], v7[4], v7[9], v7[14], m6);
            ;
            G(8, 0, v7[0], v7[4], v7[8], v7[12], m6);
            G(8, 1, v7[1], v7[5], v7[9], v7[13], m6);
            G(8, 2, v7[2], v7[6], v7[10], v7[14], m6);
            G(8, 3, v7[3], v7[7], v7[11], v7[15], m6);
            G(8, 4, v7[0], v7[5], v7[10], v7[15], m6);
            G(8, 5, v7[1], v7[6], v7[11], v7[12], m6);
            G(8, 6, v7[2], v7[7], v7[8], v7[13], m6);
            G(8, 7, v7[3], v7[4], v7[9], v7[14], m6);
            ;
            G(9, 0, v7[0], v7[4], v7[8], v7[12], m6);
            G(9, 1, v7[1], v7[5], v7[9], v7[13], m6);
            G(9, 2, v7[2], v7[6], v7[10], v7[14], m6);
            G(9, 3, v7[3], v7[7], v7[11], v7[15], m6);
            G(9, 4, v7[0], v7[5], v7[10], v7[15], m6);
            G(9, 5, v7[1], v7[6], v7[11], v7[12], m6);
            G(9, 6, v7[2], v7[7], v7[8], v7[13], m6);
            G(9, 7, v7[3], v7[4], v7[9], v7[14], m6);
            ;
            G(10, 0, v7[0], v7[4], v7[8], v7[12], m6);
            G(10, 1, v7[1], v7[5], v7[9], v7[13], m6);
            G(10, 2, v7[2], v7[6], v7[10], v7[14], m6);
            G(10, 3, v7[3], v7[7], v7[11], v7[15], m6);
            G(10, 4, v7[0], v7[5], v7[10], v7[15], m6);
            G(10, 5, v7[1], v7[6], v7[11], v7[12], m6);
            G(10, 6, v7[2], v7[7], v7[8], v7[13], m6);
            G(10, 7, v7[3], v7[4], v7[9], v7[14], m6);
            ;
            G(11, 0, v7[0], v7[4], v7[8], v7[12], m6);
            G(11, 1, v7[1], v7[5], v7[9], v7[13], m6);
            G(11, 2, v7[2], v7[6], v7[10], v7[14], m6);
            G(11, 3, v7[3], v7[7], v7[11], v7[15], m6);
            G(11, 4, v7[0], v7[5], v7[10], v7[15], m6);
            G(11, 5, v7[1], v7[6], v7[11], v7[12], m6);
            H(11, 6, v7[2], v7[7], v7[8], v7[13], m6);
            ;
            uint64_t h648;
            h648 = cuda_swab64(7640891576939301160L ^ v7[0] ^ v7[8]);
            if (h648 <= s_target5) {
                resNonce2[1] = resNonce2[0];
                resNonce2[0] = nonce4;
                s_target5 = h648;
            }
        }
    }
}


__attribute__((global)) void blake2b_gpu_hash_sia_blake2b_gpu_hash_0(const uint32_t threads9, const uint32_t startNonce10, uint32_t *resNonce11, const uint2 target212, const uint32_t threads0, const uint32_t startNonce1, uint32_t *resNonce2, const uint2 target23)
{
    if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 128)) goto label_0;
    unsigned int blockDim_x_1;
    blockDim_x_1 = 128;
    unsigned int threadIdx_x_1;
    threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 128;
    unsigned int blockDim_y_1;
    blockDim_y_1 = 1;
    unsigned int threadIdx_y_1;
    threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128 % 1;
    unsigned int blockDim_z_1;
    blockDim_z_1 = 1;
    unsigned int threadIdx_z_1;
    threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128;
    for (int i = 0; i < 80; i++) {
        uint32_t nonce13;
        nonce13 = (blockDim_x_1 * blockIdx.x + threadIdx_x_1) * 80 + i + startNonce10;
        uint64_t m14[16];
        m14[0] = d_blake_data[0];
        m14[1] = d_blake_data[1];
        m14[2] = d_blake_data[2];
        m14[3] = d_blake_data[3];
        m14[4] = d_blake_data[4];
        m14[5] = d_blake_data[5];
        m14[6] = d_blake_data[6];
        m14[7] = d_blake_data[7];
        m14[8] = d_blake_data[8];
        ((uint32_t *)m14)[18] = *((uint32_t *)(&d_blake_data[9]));
        ((uint32_t *)m14)[19] = nonce13;
        m14[10] = m14[11] = 0;
        m14[12] = m14[13] = 0;
        m14[14] = m14[15] = 0;
        uint64_t v15[16] = {7640891576939301160L, 13503953896175478587UL, 4354685564936845355L, 11912009170470909681UL, 5840696475078001361L, 11170449401992604703UL, 2270897969802886507L, 6620516959819538809L, 7640891576956012808L, 13503953896175478587UL, 4354685564936845355L, 11912009170470909681UL, 5840696475078001281L, 11170449401992604703UL, 16175846103906665108UL, 6620516959819538809L};
        G_b(0, 0, v15[0], v15[4], v15[8], v15[12], m14);
        G_b(0, 1, v15[1], v15[5], v15[9], v15[13], m14);
        G_b(0, 2, v15[2], v15[6], v15[10], v15[14], m14);
        G_b(0, 3, v15[3], v15[7], v15[11], v15[15], m14);
        G_b(0, 4, v15[0], v15[5], v15[10], v15[15], m14);
        G_b(0, 5, v15[1], v15[6], v15[11], v15[12], m14);
        G_b(0, 6, v15[2], v15[7], v15[8], v15[13], m14);
        G_b(0, 7, v15[3], v15[4], v15[9], v15[14], m14);
        ;
        G_b(1, 0, v15[0], v15[4], v15[8], v15[12], m14);
        G_b(1, 1, v15[1], v15[5], v15[9], v15[13], m14);
        G_b(1, 2, v15[2], v15[6], v15[10], v15[14], m14);
        G_b(1, 3, v15[3], v15[7], v15[11], v15[15], m14);
        G_b(1, 4, v15[0], v15[5], v15[10], v15[15], m14);
        G_b(1, 5, v15[1], v15[6], v15[11], v15[12], m14);
        G_b(1, 6, v15[2], v15[7], v15[8], v15[13], m14);
        G_b(1, 7, v15[3], v15[4], v15[9], v15[14], m14);
        ;
        G_b(2, 0, v15[0], v15[4], v15[8], v15[12], m14);
        G_b(2, 1, v15[1], v15[5], v15[9], v15[13], m14);
        G_b(2, 2, v15[2], v15[6], v15[10], v15[14], m14);
        G_b(2, 3, v15[3], v15[7], v15[11], v15[15], m14);
        G_b(2, 4, v15[0], v15[5], v15[10], v15[15], m14);
        G_b(2, 5, v15[1], v15[6], v15[11], v15[12], m14);
        G_b(2, 6, v15[2], v15[7], v15[8], v15[13], m14);
        G_b(2, 7, v15[3], v15[4], v15[9], v15[14], m14);
        ;
        G_b(3, 0, v15[0], v15[4], v15[8], v15[12], m14);
        G_b(3, 1, v15[1], v15[5], v15[9], v15[13], m14);
        G_b(3, 2, v15[2], v15[6], v15[10], v15[14], m14);
        G_b(3, 3, v15[3], v15[7], v15[11], v15[15], m14);
        G_b(3, 4, v15[0], v15[5], v15[10], v15[15], m14);
        G_b(3, 5, v15[1], v15[6], v15[11], v15[12], m14);
        G_b(3, 6, v15[2], v15[7], v15[8], v15[13], m14);
        G_b(3, 7, v15[3], v15[4], v15[9], v15[14], m14);
        ;
        G_b(4, 0, v15[0], v15[4], v15[8], v15[12], m14);
        G_b(4, 1, v15[1], v15[5], v15[9], v15[13], m14);
        G_b(4, 2, v15[2], v15[6], v15[10], v15[14], m14);
        G_b(4, 3, v15[3], v15[7], v15[11], v15[15], m14);
        G_b(4, 4, v15[0], v15[5], v15[10], v15[15], m14);
        G_b(4, 5, v15[1], v15[6], v15[11], v15[12], m14);
        G_b(4, 6, v15[2], v15[7], v15[8], v15[13], m14);
        G_b(4, 7, v15[3], v15[4], v15[9], v15[14], m14);
        ;
        G_b(5, 0, v15[0], v15[4], v15[8], v15[12], m14);
        G_b(5, 1, v15[1], v15[5], v15[9], v15[13], m14);
        G_b(5, 2, v15[2], v15[6], v15[10], v15[14], m14);
        G_b(5, 3, v15[3], v15[7], v15[11], v15[15], m14);
        G_b(5, 4, v15[0], v15[5], v15[10], v15[15], m14);
        G_b(5, 5, v15[1], v15[6], v15[11], v15[12], m14);
        G_b(5, 6, v15[2], v15[7], v15[8], v15[13], m14);
        G_b(5, 7, v15[3], v15[4], v15[9], v15[14], m14);
        ;
        G_b(6, 0, v15[0], v15[4], v15[8], v15[12], m14);
        G_b(6, 1, v15[1], v15[5], v15[9], v15[13], m14);
        G_b(6, 2, v15[2], v15[6], v15[10], v15[14], m14);
        G_b(6, 3, v15[3], v15[7], v15[11], v15[15], m14);
        G_b(6, 4, v15[0], v15[5], v15[10], v15[15], m14);
        G_b(6, 5, v15[1], v15[6], v15[11], v15[12], m14);
        G_b(6, 6, v15[2], v15[7], v15[8], v15[13], m14);
        G_b(6, 7, v15[3], v15[4], v15[9], v15[14], m14);
        ;
        G_b(7, 0, v15[0], v15[4], v15[8], v15[12], m14);
        G_b(7, 1, v15[1], v15[5], v15[9], v15[13], m14);
        G_b(7, 2, v15[2], v15[6], v15[10], v15[14], m14);
        G_b(7, 3, v15[3], v15[7], v15[11], v15[15], m14);
        G_b(7, 4, v15[0], v15[5], v15[10], v15[15], m14);
        G_b(7, 5, v15[1], v15[6], v15[11], v15[12], m14);
        G_b(7, 6, v15[2], v15[7], v15[8], v15[13], m14);
        G_b(7, 7, v15[3], v15[4], v15[9], v15[14], m14);
        ;
        G_b(8, 0, v15[0], v15[4], v15[8], v15[12], m14);
        G_b(8, 1, v15[1], v15[5], v15[9], v15[13], m14);
        G_b(8, 2, v15[2], v15[6], v15[10], v15[14], m14);
        G_b(8, 3, v15[3], v15[7], v15[11], v15[15], m14);
        G_b(8, 4, v15[0], v15[5], v15[10], v15[15], m14);
        G_b(8, 5, v15[1], v15[6], v15[11], v15[12], m14);
        G_b(8, 6, v15[2], v15[7], v15[8], v15[13], m14);
        G_b(8, 7, v15[3], v15[4], v15[9], v15[14], m14);
        ;
        G_b(9, 0, v15[0], v15[4], v15[8], v15[12], m14);
        G_b(9, 1, v15[1], v15[5], v15[9], v15[13], m14);
        G_b(9, 2, v15[2], v15[6], v15[10], v15[14], m14);
        G_b(9, 3, v15[3], v15[7], v15[11], v15[15], m14);
        G_b(9, 4, v15[0], v15[5], v15[10], v15[15], m14);
        G_b(9, 5, v15[1], v15[6], v15[11], v15[12], m14);
        G_b(9, 6, v15[2], v15[7], v15[8], v15[13], m14);
        G_b(9, 7, v15[3], v15[4], v15[9], v15[14], m14);
        ;
        G_b(10, 0, v15[0], v15[4], v15[8], v15[12], m14);
        G_b(10, 1, v15[1], v15[5], v15[9], v15[13], m14);
        G_b(10, 2, v15[2], v15[6], v15[10], v15[14], m14);
        G_b(10, 3, v15[3], v15[7], v15[11], v15[15], m14);
        G_b(10, 4, v15[0], v15[5], v15[10], v15[15], m14);
        G_b(10, 5, v15[1], v15[6], v15[11], v15[12], m14);
        G_b(10, 6, v15[2], v15[7], v15[8], v15[13], m14);
        G_b(10, 7, v15[3], v15[4], v15[9], v15[14], m14);
        ;
        G_b(11, 0, v15[0], v15[4], v15[8], v15[12], m14);
        G_b(11, 1, v15[1], v15[5], v15[9], v15[13], m14);
        G_b(11, 2, v15[2], v15[6], v15[10], v15[14], m14);
        G_b(11, 3, v15[3], v15[7], v15[11], v15[15], m14);
        G_b(11, 4, v15[0], v15[5], v15[10], v15[15], m14);
        G_b(11, 5, v15[1], v15[6], v15[11], v15[12], m14);
        G_b(11, 6, v15[2], v15[7], v15[8], v15[13], m14);
        G_b(11, 7, v15[3], v15[4], v15[9], v15[14], m14);
        ;
        uint2 last16;
        last16 = vectorize(v15[3] ^ v15[11] ^ 11912009170470909681UL);
        if (last16.y <= target212.y && last16.x <= target212.x) {
            resNonce11[1] = resNonce11[0];
            resNonce11[0] = nonce13;
        }
    }
    label_0:;
    if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=128 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 256)) goto label_1;
    unsigned int blockDim_x_0;
    blockDim_x_0 = 128;
    unsigned int threadIdx_x_0;
    threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) % 128;
    unsigned int blockDim_y_0;
    blockDim_y_0 = 1;
    unsigned int threadIdx_y_0;
    threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) / 128 % 1;
    unsigned int blockDim_z_0;
    blockDim_z_0 = 1;
    unsigned int threadIdx_z_0;
    threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) / 128;
    for (int i = 0; i < 80; i++) {
        uint32_t nonce4;
        nonce4 = (blockDim_x_0 * blockIdx.x + threadIdx_x_0) * 80 + i + startNonce1;
        static uint64_t s_target5 __attribute__((shared));
        if (!threadIdx_x_0)
            s_target5 = devectorize(target23);
        uint64_t m6[16];
        m6[0] = d_data2[0];
        m6[1] = d_data2[1];
        m6[2] = d_data2[2];
        m6[3] = d_data2[3];
        m6[4] = d_data2[4] | nonce4;
        m6[5] = d_data2[5];
        m6[6] = d_data2[6];
        m6[7] = d_data2[7];
        m6[8] = d_data2[8];
        m6[9] = d_data2[9];
        m6[10] = m6[11] = 0;
        m6[12] = m6[13] = m6[14] = m6[15] = 0;
        uint64_t v7[16] = {7640891576939301160L, 13503953896175478587UL, 4354685564936845355L, 11912009170470909681UL, 5840696475078001361L, 11170449401992604703UL, 2270897969802886507L, 6620516959819538809L, 7640891576956012808L, 13503953896175478587UL, 4354685564936845355L, 11912009170470909681UL, 5840696475078001281L, 11170449401992604703UL, 16175846103906665108UL, 6620516959819538809L};
        G(0, 0, v7[0], v7[4], v7[8], v7[12], m6);
        G(0, 1, v7[1], v7[5], v7[9], v7[13], m6);
        G(0, 2, v7[2], v7[6], v7[10], v7[14], m6);
        G(0, 3, v7[3], v7[7], v7[11], v7[15], m6);
        G(0, 4, v7[0], v7[5], v7[10], v7[15], m6);
        G(0, 5, v7[1], v7[6], v7[11], v7[12], m6);
        G(0, 6, v7[2], v7[7], v7[8], v7[13], m6);
        G(0, 7, v7[3], v7[4], v7[9], v7[14], m6);
        ;
        G(1, 0, v7[0], v7[4], v7[8], v7[12], m6);
        G(1, 1, v7[1], v7[5], v7[9], v7[13], m6);
        G(1, 2, v7[2], v7[6], v7[10], v7[14], m6);
        G(1, 3, v7[3], v7[7], v7[11], v7[15], m6);
        G(1, 4, v7[0], v7[5], v7[10], v7[15], m6);
        G(1, 5, v7[1], v7[6], v7[11], v7[12], m6);
        G(1, 6, v7[2], v7[7], v7[8], v7[13], m6);
        G(1, 7, v7[3], v7[4], v7[9], v7[14], m6);
        ;
        G(2, 0, v7[0], v7[4], v7[8], v7[12], m6);
        G(2, 1, v7[1], v7[5], v7[9], v7[13], m6);
        G(2, 2, v7[2], v7[6], v7[10], v7[14], m6);
        G(2, 3, v7[3], v7[7], v7[11], v7[15], m6);
        G(2, 4, v7[0], v7[5], v7[10], v7[15], m6);
        G(2, 5, v7[1], v7[6], v7[11], v7[12], m6);
        G(2, 6, v7[2], v7[7], v7[8], v7[13], m6);
        G(2, 7, v7[3], v7[4], v7[9], v7[14], m6);
        ;
        G(3, 0, v7[0], v7[4], v7[8], v7[12], m6);
        G(3, 1, v7[1], v7[5], v7[9], v7[13], m6);
        G(3, 2, v7[2], v7[6], v7[10], v7[14], m6);
        G(3, 3, v7[3], v7[7], v7[11], v7[15], m6);
        G(3, 4, v7[0], v7[5], v7[10], v7[15], m6);
        G(3, 5, v7[1], v7[6], v7[11], v7[12], m6);
        G(3, 6, v7[2], v7[7], v7[8], v7[13], m6);
        G(3, 7, v7[3], v7[4], v7[9], v7[14], m6);
        ;
        G(4, 0, v7[0], v7[4], v7[8], v7[12], m6);
        G(4, 1, v7[1], v7[5], v7[9], v7[13], m6);
        G(4, 2, v7[2], v7[6], v7[10], v7[14], m6);
        G(4, 3, v7[3], v7[7], v7[11], v7[15], m6);
        G(4, 4, v7[0], v7[5], v7[10], v7[15], m6);
        G(4, 5, v7[1], v7[6], v7[11], v7[12], m6);
        G(4, 6, v7[2], v7[7], v7[8], v7[13], m6);
        G(4, 7, v7[3], v7[4], v7[9], v7[14], m6);
        ;
        G(5, 0, v7[0], v7[4], v7[8], v7[12], m6);
        G(5, 1, v7[1], v7[5], v7[9], v7[13], m6);
        G(5, 2, v7[2], v7[6], v7[10], v7[14], m6);
        G(5, 3, v7[3], v7[7], v7[11], v7[15], m6);
        G(5, 4, v7[0], v7[5], v7[10], v7[15], m6);
        G(5, 5, v7[1], v7[6], v7[11], v7[12], m6);
        G(5, 6, v7[2], v7[7], v7[8], v7[13], m6);
        G(5, 7, v7[3], v7[4], v7[9], v7[14], m6);
        ;
        G(6, 0, v7[0], v7[4], v7[8], v7[12], m6);
        G(6, 1, v7[1], v7[5], v7[9], v7[13], m6);
        G(6, 2, v7[2], v7[6], v7[10], v7[14], m6);
        G(6, 3, v7[3], v7[7], v7[11], v7[15], m6);
        G(6, 4, v7[0], v7[5], v7[10], v7[15], m6);
        G(6, 5, v7[1], v7[6], v7[11], v7[12], m6);
        G(6, 6, v7[2], v7[7], v7[8], v7[13], m6);
        G(6, 7, v7[3], v7[4], v7[9], v7[14], m6);
        ;
        G(7, 0, v7[0], v7[4], v7[8], v7[12], m6);
        G(7, 1, v7[1], v7[5], v7[9], v7[13], m6);
        G(7, 2, v7[2], v7[6], v7[10], v7[14], m6);
        G(7, 3, v7[3], v7[7], v7[11], v7[15], m6);
        G(7, 4, v7[0], v7[5], v7[10], v7[15], m6);
        G(7, 5, v7[1], v7[6], v7[11], v7[12], m6);
        G(7, 6, v7[2], v7[7], v7[8], v7[13], m6);
        G(7, 7, v7[3], v7[4], v7[9], v7[14], m6);
        ;
        G(8, 0, v7[0], v7[4], v7[8], v7[12], m6);
        G(8, 1, v7[1], v7[5], v7[9], v7[13], m6);
        G(8, 2, v7[2], v7[6], v7[10], v7[14], m6);
        G(8, 3, v7[3], v7[7], v7[11], v7[15], m6);
        G(8, 4, v7[0], v7[5], v7[10], v7[15], m6);
        G(8, 5, v7[1], v7[6], v7[11], v7[12], m6);
        G(8, 6, v7[2], v7[7], v7[8], v7[13], m6);
        G(8, 7, v7[3], v7[4], v7[9], v7[14], m6);
        ;
        G(9, 0, v7[0], v7[4], v7[8], v7[12], m6);
        G(9, 1, v7[1], v7[5], v7[9], v7[13], m6);
        G(9, 2, v7[2], v7[6], v7[10], v7[14], m6);
        G(9, 3, v7[3], v7[7], v7[11], v7[15], m6);
        G(9, 4, v7[0], v7[5], v7[10], v7[15], m6);
        G(9, 5, v7[1], v7[6], v7[11], v7[12], m6);
        G(9, 6, v7[2], v7[7], v7[8], v7[13], m6);
        G(9, 7, v7[3], v7[4], v7[9], v7[14], m6);
        ;
        G(10, 0, v7[0], v7[4], v7[8], v7[12], m6);
        G(10, 1, v7[1], v7[5], v7[9], v7[13], m6);
        G(10, 2, v7[2], v7[6], v7[10], v7[14], m6);
        G(10, 3, v7[3], v7[7], v7[11], v7[15], m6);
        G(10, 4, v7[0], v7[5], v7[10], v7[15], m6);
        G(10, 5, v7[1], v7[6], v7[11], v7[12], m6);
        G(10, 6, v7[2], v7[7], v7[8], v7[13], m6);
        G(10, 7, v7[3], v7[4], v7[9], v7[14], m6);
        ;
        G(11, 0, v7[0], v7[4], v7[8], v7[12], m6);
        G(11, 1, v7[1], v7[5], v7[9], v7[13], m6);
        G(11, 2, v7[2], v7[6], v7[10], v7[14], m6);
        G(11, 3, v7[3], v7[7], v7[11], v7[15], m6);
        G(11, 4, v7[0], v7[5], v7[10], v7[15], m6);
        G(11, 5, v7[1], v7[6], v7[11], v7[12], m6);
        H(11, 6, v7[2], v7[7], v7[8], v7[13], m6);
        ;
        uint64_t h648;
        h648 = cuda_swab64(7640891576939301160L ^ v7[0] ^ v7[8]);
        if (h648 <= s_target5) {
            resNonce2[1] = resNonce2[0];
            resNonce2[0] = nonce4;
            s_target5 = h648;
        }
    }
    label_1:;
}

}

#define NBN 2

using namespace foo;
void blake_sia()
{
    {
        uint32_t threads_sia = 1048576;
        dim3 grid_sia((threads_sia + TPB_sia -1)/ TPB_sia);
        dim3 block_sia(TPB_sia);
        uint32_t resNonces[NBN] = { UINT32_MAX, UINT32_MAX };
        uint32_t result = UINT32_MAX;
        uint32_t threads_blake = 1048576;
        dim3 grid_blake((threads_blake + TPB_blake -1)/ TPB_blake);
        dim3 block_blake(TPB_blake);
        auto thr_id = 0;
        cudaMalloc(&d_sia_resNonces[thr_id], NBN * sizeof(uint32_t));
        /* Check error on Ctrl+C or kill to prevent segfaults on exit */
        if (cudaMemset(d_sia_resNonces[thr_id], 0xff, NBN*sizeof(uint32_t)) != cudaSuccess)
            return;
        CUDA_SAFE_CALL(cudaMalloc(&d_blake_resNonces[thr_id], NBN * sizeof(uint32_t)));
        if (cudaMemset(d_blake_resNonces[thr_id], 0xff, NBN*sizeof(uint32_t)) != cudaSuccess)
            return;
        const uint2 target2 = make_uint2(0, 1);
        cudaStream_t t1;
        cudaStream_t t2;
        cudaStreamCreate ( &t1);
        cudaStreamCreate ( &t2);
        blake2b_gpu_hash <<<grid_blake, block_blake, 8, t1>>> (
            threads_blake, 0, d_blake_resNonces[thr_id], target2);
        sia_blake2b_gpu_hash <<<grid_sia, block_sia, 8, t2>>> (
            threads_sia, 0, d_sia_resNonces[thr_id], target2);



        cudaDeviceSynchronize();

        blake2b_gpu_hash_sia_blake2b_gpu_hash_0
            <<<grid_blake, block_blake.x + block_sia.x, 16, t1>>> (
            threads_blake, 0, d_blake_resNonces[thr_id], target2,
                threads_sia, 0, d_sia_resNonces[thr_id], target2);

        cudaDeviceSynchronize();

        blake2b_gpu_hash_sia_blake2b_gpu_hash_100
            <<<grid_blake, 128, 16, t1>>> (
            threads_blake, 0, d_blake_resNonces[thr_id], target2,
                threads_sia, 0, d_sia_resNonces[thr_id], target2);

    }



    cudaDeviceSynchronize();
    CUDA_SAFE_CALL(cudaGetLastError());
}

