#include "ethash_cuda_miner_kernel.h"

#include "ethash_cuda_miner_kernel_globals.h"

#include "cuda_helper.h"

#include "fnv.cuh"

#define copy(dst, src, count)        \
    for (int i = 0; i != count; ++i) \
    {                                \
        (dst)[i] = (src)[i];         \
    }

#include "keccak.cuh"

#include "dagger_shuffled.cuh"

#define TPB 128
#define NBN 2

static uint32_t *d_resNonces[MAX_GPUS];

__device__ uint64_t d_data2[10];

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

__global__ void ethash_search4(volatile Search_results* g_output, uint64_t start_nonce)
{
    uint32_t const gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint2 mix[4];
    uint64_t nonce = start_nonce + gid;
    uint2* mix_hash = mix;
    bool result = false;

    uint2 state[12];

    state[4] = vectorize(nonce);

    keccak_f1600_init(state);

    // Threads work together in this phase in groups of 8.
    const int thread_id = threadIdx.x & (THREADS_PER_HASH - 1);
    const int mix_idx = thread_id & 3;

    for (int i = 0; i < THREADS_PER_HASH; i += _PARALLEL_HASH)
    {
        uint4 mix[_PARALLEL_HASH];
        uint32_t offset[_PARALLEL_HASH];
        uint32_t init0[_PARALLEL_HASH];

        // share init among threads
        for (int p = 0; p < _PARALLEL_HASH; p++)
        {
            uint2 shuffle[8];
            for (int j = 0; j < 8; j++)
            {
                shuffle[j].x = SHFL(state[j].x, i + p, THREADS_PER_HASH);
                shuffle[j].y = SHFL(state[j].y, i + p, THREADS_PER_HASH);
            }
            switch (mix_idx)
            {
            case 0:
                mix[p] = vectorize2(shuffle[0], shuffle[1]);
                break;
            case 1:
                mix[p] = vectorize2(shuffle[2], shuffle[3]);
                break;
            case 2:
                mix[p] = vectorize2(shuffle[4], shuffle[5]);
                break;
            case 3:
                mix[p] = vectorize2(shuffle[6], shuffle[7]);
                break;
            }
            init0[p] = SHFL(shuffle[0].x, 0, THREADS_PER_HASH);
        }

        for (uint32_t a = 0; a < ACCESSES; a += 4)
        {
            int t = bfe(a, 2u, 3u);

            for (uint32_t b = 0; b < 4; b++)
            {
                for (int p = 0; p < _PARALLEL_HASH; p++)
                {
                    offset[p] = fnv(init0[p] ^ (a + b), ((uint32_t*)&mix[p])[b]) % d_dag_size;
                    offset[p] = SHFL(offset[p], t, THREADS_PER_HASH);
                    mix[p] = fnv4(mix[p], d_dag[offset[p]].uint4s[thread_id]);
                }
            }
        }

        for (int p = 0; p < _PARALLEL_HASH; p++)
        {
            uint2 shuffle[4];
            uint32_t thread_mix = fnv_reduce(mix[p]);

            // update mix across threads
            shuffle[0].x = SHFL(thread_mix, 0, THREADS_PER_HASH);
            shuffle[0].y = SHFL(thread_mix, 1, THREADS_PER_HASH);
            shuffle[1].x = SHFL(thread_mix, 2, THREADS_PER_HASH);
            shuffle[1].y = SHFL(thread_mix, 3, THREADS_PER_HASH);
            shuffle[2].x = SHFL(thread_mix, 4, THREADS_PER_HASH);
            shuffle[2].y = SHFL(thread_mix, 5, THREADS_PER_HASH);
            shuffle[3].x = SHFL(thread_mix, 6, THREADS_PER_HASH);
            shuffle[3].y = SHFL(thread_mix, 7, THREADS_PER_HASH);

            if ((i + p) == thread_id)
            {
                // move mix into state:
                state[8] = shuffle[0];
                state[9] = shuffle[1];
                state[10] = shuffle[2];
                state[11] = shuffle[3];
            }
        }
    }

    // keccak_256(keccak_512(header..nonce) .. mix);
    if (!(cuda_swab64(keccak_f1600_final(state)) > d_target)) {
        mix_hash[0] = state[8];
        mix_hash[1] = state[9];
        mix_hash[2] = state[10];
        mix_hash[3] = state[11];
        return;
    }

    uint32_t index = atomicInc((uint32_t*)&g_output->count, 0xffffffff);
    if (index >= MAX_SEARCH_RESULTS)
        return;
    g_output->result[index].gid = gid;
    g_output->result[index].mix[0] = mix[0].x;
    g_output->result[index].mix[1] = mix[0].y;
    g_output->result[index].mix[2] = mix[1].x;
    g_output->result[index].mix[3] = mix[1].y;
    g_output->result[index].mix[4] = mix[2].x;
    g_output->result[index].mix[5] = mix[2].y;
    g_output->result[index].mix[6] = mix[3].x;
    g_output->result[index].mix[7] = mix[3].y;
}




__attribute__((global)) void sia_blake2b_gpu_hash_ethash_search4_0(const uint32_t threads0, const uint32_t startNonce1, uint32_t *resNonce2, const uint2 target23, volatile Search_results *g_output9, uint64_t start_nonce10)
{
    if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 128)) goto label_0;
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
    label_0:;
    if (!((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=128 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 256)) goto label_1;
    unsigned int blockDim_x_1;
    blockDim_x_1 = 128;
    unsigned int threadIdx_x_1;
    threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) % 128;
    unsigned int blockDim_y_1;
    blockDim_y_1 = 1;
    unsigned int threadIdx_y_1;
    threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) / 128 % 1;
    unsigned int blockDim_z_1;
    blockDim_z_1 = 1;
    unsigned int threadIdx_z_1;
    threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) / 128;
    uint32_t gid11;
    gid11 = blockIdx.x * blockDim_x_1 + threadIdx_x_1;
    uint2 mix12[4];
    uint64_t nonce13;
    nonce13 = start_nonce10 + gid11;
    uint2 *mix_hash14;
    mix_hash14 = mix12;
    bool result15;
    result15 = false;
    uint2 state16[12];
    state16[4] = vectorize(nonce13);
    keccak_f1600_init(state16);
    int thread_id17;
    thread_id17 = threadIdx_x_1 & ((128 / 16) - 1);
    int mix_idx18;
    mix_idx18 = thread_id17 & 3;
    for (int i = 0; i < (128 / 16); i += 4) {
        uint4 mix20[4];
        uint32_t offset21[4];
        uint32_t init022[4];
        for (int p = 0; p < 4; p++) {
            uint2 shuffle23[8];
            for (int j = 0; j < 8; j++) {
                shuffle23[j].x = __shfl_sync(4294967295U, (state16[j].x), (i + p), ((128 / 16)));
                shuffle23[j].y = __shfl_sync(4294967295U, (state16[j].y), (i + p), ((128 / 16)));
            }
            switch (mix_idx18) {
            case 0:
                mix20[p] = vectorize2(shuffle23[0], shuffle23[1]);
                break;
            case 1:
                mix20[p] = vectorize2(shuffle23[2], shuffle23[3]);
                break;
            case 2:
                mix20[p] = vectorize2(shuffle23[4], shuffle23[5]);
                break;
            case 3:
                mix20[p] = vectorize2(shuffle23[6], shuffle23[7]);
                break;
            }
            init022[p] = __shfl_sync(4294967295U, (shuffle23[0].x), (0), ((128 / 16)));
        }
        for (uint32_t a = 0; a < 64; a += 4) {
            int t24;
            t24 = bfe(a, 2U, 3U);
            for (uint32_t b = 0; b < 4; b++) {
                for (int p = 0; p < 4; p++) {
                    offset21[p] = ((init022[p] ^ (a + b)) * 16777619 ^ (((uint32_t *)&mix20[p])[b])) % d_dag_size;
                    offset21[p] = __shfl_sync(4294967295U, (offset21[p]), (t24), ((128 / 16)));
                    mix20[p] = fnv4(mix20[p], d_dag[offset21[p]].uint4s[thread_id17]);
                }
            }
        }
        for (int p = 0; p < 4; p++) {
            uint2 shuffle25[4];
            uint32_t thread_mix26;
            thread_mix26 = fnv_reduce(mix20[p]);
            shuffle25[0].x = __shfl_sync(4294967295U, (thread_mix26), (0), ((128 / 16)));
            shuffle25[0].y = __shfl_sync(4294967295U, (thread_mix26), (1), ((128 / 16)));
            shuffle25[1].x = __shfl_sync(4294967295U, (thread_mix26), (2), ((128 / 16)));
            shuffle25[1].y = __shfl_sync(4294967295U, (thread_mix26), (3), ((128 / 16)));
            shuffle25[2].x = __shfl_sync(4294967295U, (thread_mix26), (4), ((128 / 16)));
            shuffle25[2].y = __shfl_sync(4294967295U, (thread_mix26), (5), ((128 / 16)));
            shuffle25[3].x = __shfl_sync(4294967295U, (thread_mix26), (6), ((128 / 16)));
            shuffle25[3].y = __shfl_sync(4294967295U, (thread_mix26), (7), ((128 / 16)));
            if ((i + p) == thread_id17) {
                state16[8] = shuffle25[0];
                state16[9] = shuffle25[1];
                state16[10] = shuffle25[2];
                state16[11] = shuffle25[3];
            }
        }
    }
    if (!(cuda_swab64(keccak_f1600_final(state16)) > d_target)) {
        mix_hash14[0] = state16[8];
        mix_hash14[1] = state16[9];
        mix_hash14[2] = state16[10];
        mix_hash14[3] = state16[11];
        return;
    }
    uint32_t index19;
    index19 = atomicInc((uint32_t *)&g_output9->count, 4294967295U);
    if (index19 >= 4U)
        return;
    g_output9->result[index19].gid = gid11;
    g_output9->result[index19].mix[0] = mix12[0].x;
    g_output9->result[index19].mix[1] = mix12[0].y;
    g_output9->result[index19].mix[2] = mix12[1].x;
    g_output9->result[index19].mix[3] = mix12[1].y;
    g_output9->result[index19].mix[4] = mix12[2].x;
    g_output9->result[index19].mix[5] = mix12[2].y;
    g_output9->result[index19].mix[6] = mix12[3].x;
    g_output9->result[index19].mix[7] = mix12[3].y;
    label_1:;
}


__attribute__((global)) void sia_blake2b_gpu_hash_ethash_search4_100(const uint32_t threads0, const uint32_t startNonce1, uint32_t *resNonce2, const uint2 target23, volatile Search_results *g_output9, uint64_t start_nonce10)
{
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
        uint32_t gid11;
        gid11 = blockIdx.x * blockDim_x_1 + threadIdx_x_1;
        uint2 mix12[4];
        uint64_t nonce13;
        nonce13 = start_nonce10 + gid11;
        uint2 *mix_hash14;
        mix_hash14 = mix12;
        bool result15;
        result15 = false;
        uint2 state16[12];
        state16[4] = vectorize(nonce13);
        keccak_f1600_init(state16);
        int thread_id17;
        thread_id17 = threadIdx_x_1 & ((128 / 16) - 1);
        int mix_idx18;
        mix_idx18 = thread_id17 & 3;
        for (int i = 0; i < (128 / 16); i += 4) {
            uint4 mix20[4];
            uint32_t offset21[4];
            uint32_t init022[4];
            for (int p = 0; p < 4; p++) {
                uint2 shuffle23[8];
                for (int j = 0; j < 8; j++) {
                    shuffle23[j].x = __shfl_sync(4294967295U, (state16[j].x), (i + p), ((128 / 16)));
                    shuffle23[j].y = __shfl_sync(4294967295U, (state16[j].y), (i + p), ((128 / 16)));
                }
                switch (mix_idx18) {
                case 0:
                    mix20[p] = vectorize2(shuffle23[0], shuffle23[1]);
                    break;
                case 1:
                    mix20[p] = vectorize2(shuffle23[2], shuffle23[3]);
                    break;
                case 2:
                    mix20[p] = vectorize2(shuffle23[4], shuffle23[5]);
                    break;
                case 3:
                    mix20[p] = vectorize2(shuffle23[6], shuffle23[7]);
                    break;
                }
                init022[p] = __shfl_sync(4294967295U, (shuffle23[0].x), (0), ((128 / 16)));
            }
            for (uint32_t a = 0; a < 64; a += 4) {
                int t24;
                t24 = bfe(a, 2U, 3U);
                for (uint32_t b = 0; b < 4; b++) {
                    for (int p = 0; p < 4; p++) {
                        offset21[p] = ((init022[p] ^ (a + b)) * 16777619 ^ (((uint32_t *)&mix20[p])[b])) % d_dag_size;
                        offset21[p] = __shfl_sync(4294967295U, (offset21[p]), (t24), ((128 / 16)));
                        mix20[p] = fnv4(mix20[p], d_dag[offset21[p]].uint4s[thread_id17]);
                    }
                }
            }
            for (int p = 0; p < 4; p++) {
                uint2 shuffle25[4];
                uint32_t thread_mix26;
                thread_mix26 = fnv_reduce(mix20[p]);
                shuffle25[0].x = __shfl_sync(4294967295U, (thread_mix26), (0), ((128 / 16)));
                shuffle25[0].y = __shfl_sync(4294967295U, (thread_mix26), (1), ((128 / 16)));
                shuffle25[1].x = __shfl_sync(4294967295U, (thread_mix26), (2), ((128 / 16)));
                shuffle25[1].y = __shfl_sync(4294967295U, (thread_mix26), (3), ((128 / 16)));
                shuffle25[2].x = __shfl_sync(4294967295U, (thread_mix26), (4), ((128 / 16)));
                shuffle25[2].y = __shfl_sync(4294967295U, (thread_mix26), (5), ((128 / 16)));
                shuffle25[3].x = __shfl_sync(4294967295U, (thread_mix26), (6), ((128 / 16)));
                shuffle25[3].y = __shfl_sync(4294967295U, (thread_mix26), (7), ((128 / 16)));
                if ((i + p) == thread_id17) {
                    state16[8] = shuffle25[0];
                    state16[9] = shuffle25[1];
                    state16[10] = shuffle25[2];
                    state16[11] = shuffle25[3];
                }
            }
        }
        if (!(cuda_swab64(keccak_f1600_final(state16)) > d_target)) {
            mix_hash14[0] = state16[8];
            mix_hash14[1] = state16[9];
            mix_hash14[2] = state16[10];
            mix_hash14[3] = state16[11];
            return;
        }
        uint32_t index19;
        index19 = atomicInc((uint32_t *)&g_output9->count, 4294967295U);
        if (index19 >= 4U)
            return;
        g_output9->result[index19].gid = gid11;
        g_output9->result[index19].mix[0] = mix12[0].x;
        g_output9->result[index19].mix[1] = mix12[0].y;
        g_output9->result[index19].mix[2] = mix12[1].x;
        g_output9->result[index19].mix[3] = mix12[1].y;
        g_output9->result[index19].mix[4] = mix12[2].x;
        g_output9->result[index19].mix[5] = mix12[2].y;
        g_output9->result[index19].mix[6] = mix12[3].x;
        g_output9->result[index19].mix[7] = mix12[3].y;
    }
}


void run_ethash_search_sia(uint32_t gridSize, uint32_t blockSize, cudaStream_t stream,
                           volatile Search_results* g_output, uint64_t start_nonce)
{

    {
        uint32_t resNonces[NBN] = { UINT32_MAX, UINT32_MAX };
        uint32_t result = UINT32_MAX;

        uint32_t threads = 1048576;
        dim3 grid((threads + TPB-1)/TPB);
        dim3 block(TPB);
        int thr_id = 0;

        cudaMalloc(&d_resNonces[thr_id], NBN * sizeof(uint32_t));
        /* Check error on Ctrl+C or kill to prevent segfaults on exit */
        if (cudaMemset(d_resNonces[thr_id], 0xff, NBN*sizeof(uint32_t)) != cudaSuccess)
            return;

        const uint2 target2 = make_uint2(3, 5);

        cudaThreadSynchronize();
        cudaStream_t t1;
        cudaStream_t t2;
        cudaStreamCreate ( &t1);
        cudaStreamCreate ( &t2);
        ethash_search<<<gridSize, blockSize, 0, t1>>>(g_output, start_nonce);
        sia_blake2b_gpu_hash <<<grid, block, 8, t2>>> (threads, 0, d_resNonces[thr_id], target2);
        cudaThreadSynchronize();
        sia_blake2b_gpu_hash_ethash_search4_0<<<grid, block.x + blockSize, 8>>>(
            threads, 0, d_resNonces[thr_id], target2,
                g_output, start_nonce
        );
        cudaThreadSynchronize();
        sia_blake2b_gpu_hash_ethash_search4_100<<<grid, block.x, 8>>>(
            threads, 0, d_resNonces[thr_id], target2,
                g_output, start_nonce
        );
    }
    CUDA_SAFE_CALL(cudaGetLastError());
}
