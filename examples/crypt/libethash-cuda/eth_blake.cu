//
// Created by Leo Li on 2019-11-14.
//
#include "ethash_cuda_miner_kernel.h"

#include "ethash_cuda_miner_kernel_globals.h"

#include "cuda_helper.h"

#include "fnv.cuh"

#include <string.h>
#include <stdint.h>

#include "keccak.cuh"

#include "dagger_shuffled.cuh"

//#include <sph/blake2b.h>

#define TPB 128
#define NBN 2

static uint32_t *d_resNonces[MAX_GPUS];

__device__ uint64_t d_data[10];

#define AS_U32(addr)   *((uint32_t*)(addr))

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
//extern "C" void blake2b_hash(void *output, const void *input)
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

__global__
void blake2b_gpu_hash(
    const uint32_t threads, const uint32_t startNonce, uint32_t *resNonce, const uint2 target2)
{
    for (int i = 0; i < 80; i++)
    {
        const uint32_t nonce = (blockDim.x * blockIdx.x + threadIdx.x) * 80 + i + startNonce;

        uint64_t m[16];

        m[0] = d_data[0];
        m[1] = d_data[1];
        m[2] = d_data[2];
        m[3] = d_data[3];
        m[4] = d_data[4];
        m[5] = d_data[5];
        m[6] = d_data[6];
        m[7] = d_data[7];
        m[8] = d_data[8];
        ((uint32_t*)m)[18] = AS_U32(&d_data[9]);
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

#define copy(dst, src, count)        \
    for (int i = 0; i != count; ++i) \
    {                                \
        (dst)[i] = (src)[i];         \
    }


__global__ void ethash_search2(volatile Search_results* g_output, uint64_t start_nonce)
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

__attribute__((global))
void ethash_search2_blake2b_gpu_hash_0(volatile Search_results *g_output8, uint64_t start_nonce9, const uint32_t threads0, const uint32_t startNonce1, uint32_t *resNonce2, const uint2 target23)
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
    uint32_t gid10;
    gid10 = blockIdx.x * blockDim_x_1 + threadIdx_x_1;
    uint2 mix11[4];
    uint64_t nonce12;
    nonce12 = start_nonce9 + gid10;
    uint2 *mix_hash13;
    mix_hash13 = mix11;
    bool result14;
    result14 = false;
    uint2 state15[12];
    state15[4] = vectorize(nonce12);
    keccak_f1600_init(state15);
    int thread_id16;
    thread_id16 = threadIdx_x_1 & ((128 / 16) - 1);
    int mix_idx17;
    mix_idx17 = thread_id16 & 3;
    for (int i = 0; i < (128 / 16); i += 4) {
        uint4 mix19[4];
        uint32_t offset20[4];
        uint32_t init021[4];
        for (int p = 0; p < 4; p++) {
            uint2 shuffle22[8];
            for (int j = 0; j < 8; j++) {
                shuffle22[j].x = __shfl_sync(4294967295U, (state15[j].x), (i + p), ((128 / 16)));
                shuffle22[j].y = __shfl_sync(4294967295U, (state15[j].y), (i + p), ((128 / 16)));
            }
            switch (mix_idx17) {
            case 0:
                mix19[p] = vectorize2(shuffle22[0], shuffle22[1]);
                break;
            case 1:
                mix19[p] = vectorize2(shuffle22[2], shuffle22[3]);
                break;
            case 2:
                mix19[p] = vectorize2(shuffle22[4], shuffle22[5]);
                break;
            case 3:
                mix19[p] = vectorize2(shuffle22[6], shuffle22[7]);
                break;
            }
            init021[p] = __shfl_sync(4294967295U, (shuffle22[0].x), (0), ((128 / 16)));
        }
        for (uint32_t a = 0; a < 64; a += 4) {
            int t23;
            t23 = bfe(a, 2U, 3U);
            for (uint32_t b = 0; b < 4; b++) {
                for (int p = 0; p < 4; p++) {
                    offset20[p] = ((init021[p] ^ (a + b)) * 16777619 ^ (((uint32_t *)&mix19[p])[b])) % d_dag_size;
                    offset20[p] = __shfl_sync(4294967295U, (offset20[p]), (t23), ((128 / 16)));
                    mix19[p] = fnv4(mix19[p], d_dag[offset20[p]].uint4s[thread_id16]);
                }
            }
        }
        for (int p = 0; p < 4; p++) {
            uint2 shuffle24[4];
            uint32_t thread_mix25;
            thread_mix25 = fnv_reduce(mix19[p]);
            shuffle24[0].x = __shfl_sync(4294967295U, (thread_mix25), (0), ((128 / 16)));
            shuffle24[0].y = __shfl_sync(4294967295U, (thread_mix25), (1), ((128 / 16)));
            shuffle24[1].x = __shfl_sync(4294967295U, (thread_mix25), (2), ((128 / 16)));
            shuffle24[1].y = __shfl_sync(4294967295U, (thread_mix25), (3), ((128 / 16)));
            shuffle24[2].x = __shfl_sync(4294967295U, (thread_mix25), (4), ((128 / 16)));
            shuffle24[2].y = __shfl_sync(4294967295U, (thread_mix25), (5), ((128 / 16)));
            shuffle24[3].x = __shfl_sync(4294967295U, (thread_mix25), (6), ((128 / 16)));
            shuffle24[3].y = __shfl_sync(4294967295U, (thread_mix25), (7), ((128 / 16)));
            if ((i + p) == thread_id16) {
                state15[8] = shuffle24[0];
                state15[9] = shuffle24[1];
                state15[10] = shuffle24[2];
                state15[11] = shuffle24[3];
            }
        }
    }
    if (!(cuda_swab64(keccak_f1600_final(state15)) > d_target)) {
        mix_hash13[0] = state15[8];
        mix_hash13[1] = state15[9];
        mix_hash13[2] = state15[10];
        mix_hash13[3] = state15[11];
        return;
    }
    uint32_t index18;
    index18 = atomicInc((uint32_t *)&g_output8->count, 4294967295U);
    if (index18 >= 4U)
        return;
    g_output8->result[index18].gid = gid10;
    g_output8->result[index18].mix[0] = mix11[0].x;
    g_output8->result[index18].mix[1] = mix11[0].y;
    g_output8->result[index18].mix[2] = mix11[1].x;
    g_output8->result[index18].mix[3] = mix11[1].y;
    g_output8->result[index18].mix[4] = mix11[2].x;
    g_output8->result[index18].mix[5] = mix11[2].y;
    g_output8->result[index18].mix[6] = mix11[3].x;
    g_output8->result[index18].mix[7] = mix11[3].y;
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
        nonce4 = ((blockDim_x_0 * blockIdx.x + threadIdx_x_0) * 80 + i) + startNonce1;
        uint64_t m5[16];
        m5[0] = d_data[0];
        m5[1] = d_data[1];
        m5[2] = d_data[2];
        m5[3] = d_data[3];
        m5[4] = d_data[4];
        m5[5] = d_data[5];
        m5[6] = d_data[6];
        m5[7] = d_data[7];
        m5[8] = d_data[8];
        ((uint32_t *)m5)[18] = *((uint32_t *)(&d_data[9]));
        ((uint32_t *)m5)[19] = nonce4;
        m5[10] = m5[11] = 0;
        m5[12] = m5[13] = 0;
        m5[14] = m5[15] = 0;
        uint64_t v6[16] = {7640891576939301160L, 13503953896175478587UL, 4354685564936845355L, 11912009170470909681UL, 5840696475078001361L, 11170449401992604703UL, 2270897969802886507L, 6620516959819538809L, 7640891576956012808L, 13503953896175478587UL, 4354685564936845355L, 11912009170470909681UL, 5840696475078001281L, 11170449401992604703UL, 16175846103906665108UL, 6620516959819538809L};
        G(0, 0, v6[0], v6[4], v6[8], v6[12], m5);
        G(0, 1, v6[1], v6[5], v6[9], v6[13], m5);
        G(0, 2, v6[2], v6[6], v6[10], v6[14], m5);
        G(0, 3, v6[3], v6[7], v6[11], v6[15], m5);
        G(0, 4, v6[0], v6[5], v6[10], v6[15], m5);
        G(0, 5, v6[1], v6[6], v6[11], v6[12], m5);
        G(0, 6, v6[2], v6[7], v6[8], v6[13], m5);
        G(0, 7, v6[3], v6[4], v6[9], v6[14], m5);
        ;
        G(1, 0, v6[0], v6[4], v6[8], v6[12], m5);
        G(1, 1, v6[1], v6[5], v6[9], v6[13], m5);
        G(1, 2, v6[2], v6[6], v6[10], v6[14], m5);
        G(1, 3, v6[3], v6[7], v6[11], v6[15], m5);
        G(1, 4, v6[0], v6[5], v6[10], v6[15], m5);
        G(1, 5, v6[1], v6[6], v6[11], v6[12], m5);
        G(1, 6, v6[2], v6[7], v6[8], v6[13], m5);
        G(1, 7, v6[3], v6[4], v6[9], v6[14], m5);
        ;
        G(2, 0, v6[0], v6[4], v6[8], v6[12], m5);
        G(2, 1, v6[1], v6[5], v6[9], v6[13], m5);
        G(2, 2, v6[2], v6[6], v6[10], v6[14], m5);
        G(2, 3, v6[3], v6[7], v6[11], v6[15], m5);
        G(2, 4, v6[0], v6[5], v6[10], v6[15], m5);
        G(2, 5, v6[1], v6[6], v6[11], v6[12], m5);
        G(2, 6, v6[2], v6[7], v6[8], v6[13], m5);
        G(2, 7, v6[3], v6[4], v6[9], v6[14], m5);
        ;
        G(3, 0, v6[0], v6[4], v6[8], v6[12], m5);
        G(3, 1, v6[1], v6[5], v6[9], v6[13], m5);
        G(3, 2, v6[2], v6[6], v6[10], v6[14], m5);
        G(3, 3, v6[3], v6[7], v6[11], v6[15], m5);
        G(3, 4, v6[0], v6[5], v6[10], v6[15], m5);
        G(3, 5, v6[1], v6[6], v6[11], v6[12], m5);
        G(3, 6, v6[2], v6[7], v6[8], v6[13], m5);
        G(3, 7, v6[3], v6[4], v6[9], v6[14], m5);
        ;
        G(4, 0, v6[0], v6[4], v6[8], v6[12], m5);
        G(4, 1, v6[1], v6[5], v6[9], v6[13], m5);
        G(4, 2, v6[2], v6[6], v6[10], v6[14], m5);
        G(4, 3, v6[3], v6[7], v6[11], v6[15], m5);
        G(4, 4, v6[0], v6[5], v6[10], v6[15], m5);
        G(4, 5, v6[1], v6[6], v6[11], v6[12], m5);
        G(4, 6, v6[2], v6[7], v6[8], v6[13], m5);
        G(4, 7, v6[3], v6[4], v6[9], v6[14], m5);
        ;
        G(5, 0, v6[0], v6[4], v6[8], v6[12], m5);
        G(5, 1, v6[1], v6[5], v6[9], v6[13], m5);
        G(5, 2, v6[2], v6[6], v6[10], v6[14], m5);
        G(5, 3, v6[3], v6[7], v6[11], v6[15], m5);
        G(5, 4, v6[0], v6[5], v6[10], v6[15], m5);
        G(5, 5, v6[1], v6[6], v6[11], v6[12], m5);
        G(5, 6, v6[2], v6[7], v6[8], v6[13], m5);
        G(5, 7, v6[3], v6[4], v6[9], v6[14], m5);
        ;
        G(6, 0, v6[0], v6[4], v6[8], v6[12], m5);
        G(6, 1, v6[1], v6[5], v6[9], v6[13], m5);
        G(6, 2, v6[2], v6[6], v6[10], v6[14], m5);
        G(6, 3, v6[3], v6[7], v6[11], v6[15], m5);
        G(6, 4, v6[0], v6[5], v6[10], v6[15], m5);
        G(6, 5, v6[1], v6[6], v6[11], v6[12], m5);
        G(6, 6, v6[2], v6[7], v6[8], v6[13], m5);
        G(6, 7, v6[3], v6[4], v6[9], v6[14], m5);
        ;
        G(7, 0, v6[0], v6[4], v6[8], v6[12], m5);
        G(7, 1, v6[1], v6[5], v6[9], v6[13], m5);
        G(7, 2, v6[2], v6[6], v6[10], v6[14], m5);
        G(7, 3, v6[3], v6[7], v6[11], v6[15], m5);
        G(7, 4, v6[0], v6[5], v6[10], v6[15], m5);
        G(7, 5, v6[1], v6[6], v6[11], v6[12], m5);
        G(7, 6, v6[2], v6[7], v6[8], v6[13], m5);
        G(7, 7, v6[3], v6[4], v6[9], v6[14], m5);
        ;
        G(8, 0, v6[0], v6[4], v6[8], v6[12], m5);
        G(8, 1, v6[1], v6[5], v6[9], v6[13], m5);
        G(8, 2, v6[2], v6[6], v6[10], v6[14], m5);
        G(8, 3, v6[3], v6[7], v6[11], v6[15], m5);
        G(8, 4, v6[0], v6[5], v6[10], v6[15], m5);
        G(8, 5, v6[1], v6[6], v6[11], v6[12], m5);
        G(8, 6, v6[2], v6[7], v6[8], v6[13], m5);
        G(8, 7, v6[3], v6[4], v6[9], v6[14], m5);
        ;
        G(9, 0, v6[0], v6[4], v6[8], v6[12], m5);
        G(9, 1, v6[1], v6[5], v6[9], v6[13], m5);
        G(9, 2, v6[2], v6[6], v6[10], v6[14], m5);
        G(9, 3, v6[3], v6[7], v6[11], v6[15], m5);
        G(9, 4, v6[0], v6[5], v6[10], v6[15], m5);
        G(9, 5, v6[1], v6[6], v6[11], v6[12], m5);
        G(9, 6, v6[2], v6[7], v6[8], v6[13], m5);
        G(9, 7, v6[3], v6[4], v6[9], v6[14], m5);
        ;
        G(10, 0, v6[0], v6[4], v6[8], v6[12], m5);
        G(10, 1, v6[1], v6[5], v6[9], v6[13], m5);
        G(10, 2, v6[2], v6[6], v6[10], v6[14], m5);
        G(10, 3, v6[3], v6[7], v6[11], v6[15], m5);
        G(10, 4, v6[0], v6[5], v6[10], v6[15], m5);
        G(10, 5, v6[1], v6[6], v6[11], v6[12], m5);
        G(10, 6, v6[2], v6[7], v6[8], v6[13], m5);
        G(10, 7, v6[3], v6[4], v6[9], v6[14], m5);
        ;
        G(11, 0, v6[0], v6[4], v6[8], v6[12], m5);
        G(11, 1, v6[1], v6[5], v6[9], v6[13], m5);
        G(11, 2, v6[2], v6[6], v6[10], v6[14], m5);
        G(11, 3, v6[3], v6[7], v6[11], v6[15], m5);
        G(11, 4, v6[0], v6[5], v6[10], v6[15], m5);
        G(11, 5, v6[1], v6[6], v6[11], v6[12], m5);
        G(11, 6, v6[2], v6[7], v6[8], v6[13], m5);
        G(11, 7, v6[3], v6[4], v6[9], v6[14], m5);
        ;
        uint2 last7;
        last7 = vectorize(v6[3] ^ v6[11] ^ 11912009170470909681UL);
        if (last7.y <= target23.y && last7.x <= target23.x) {
            resNonce2[1] = resNonce2[0];
            resNonce2[0] = nonce4;
        }
    }
    label_1:;
}

__attribute__((global)) void ethash_search2_blake2b_gpu_hash_100(volatile Search_results *g_output814, uint64_t start_nonce915, const uint32_t threads00, const uint32_t startNonce11, uint32_t *resNonce22, const uint2 target233)
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
        unsigned int blockDim_x_116;
        blockDim_x_116 = 128;
        unsigned int threadIdx_x_117;
        threadIdx_x_117 = ((threadIdx_x_1 + threadIdx_y_1 * blockDim_x_1 + threadIdx_z_1 * blockDim_x_1 * blockDim_y_1) - 0) % 128;
        unsigned int blockDim_y_118;
        blockDim_y_118 = 1;
        unsigned int threadIdx_y_119;
        threadIdx_y_119 = ((threadIdx_x_1 + threadIdx_y_1 * blockDim_x_1 + threadIdx_z_1 * blockDim_x_1 * blockDim_y_1) - 0) / 128 % 1;
        unsigned int blockDim_z_120;
        blockDim_z_120 = 1;
        unsigned int threadIdx_z_121;
        threadIdx_z_121 = ((threadIdx_x_1 + threadIdx_y_1 * blockDim_x_1 + threadIdx_z_1 * blockDim_x_1 * blockDim_y_1) - 0) / 128;
        uint32_t gid1022;
        gid1022 = blockIdx.x * blockDim_x_116 + threadIdx_x_117;
        uint2 mix1123[4];
        uint64_t nonce1224;
        nonce1224 = start_nonce915 + gid1022;
        uint2 *mix_hash1325;
        mix_hash1325 = mix1123;
        bool result1426;
        result1426 = false;
        uint2 state1527[12];
        state1527[4] = vectorize(nonce1224);
        keccak_f1600_init(state1527);
        int thread_id1628;
        thread_id1628 = threadIdx_x_117 & ((128 / 16) - 1);
        int mix_idx1729;
        mix_idx1729 = thread_id1628 & 3;
        for (int i = 0; i < (128 / 16); i += 4) {
            uint4 mix1931[4];
            uint32_t offset2032[4];
            uint32_t init02133[4];
            for (int p = 0; p < 4; p++) {
                uint2 shuffle2234[8];
                for (int j = 0; j < 8; j++) {
                    shuffle2234[j].x = __shfl_sync(4294967295U, (state1527[j].x), (i + p), ((128 / 16)));
                    shuffle2234[j].y = __shfl_sync(4294967295U, (state1527[j].y), (i + p), ((128 / 16)));
                }
                switch (mix_idx1729) {
                case 0:
                    mix1931[p] = vectorize2(shuffle2234[0], shuffle2234[1]);
                    break;
                case 1:
                    mix1931[p] = vectorize2(shuffle2234[2], shuffle2234[3]);
                    break;
                case 2:
                    mix1931[p] = vectorize2(shuffle2234[4], shuffle2234[5]);
                    break;
                case 3:
                    mix1931[p] = vectorize2(shuffle2234[6], shuffle2234[7]);
                    break;
                }
                init02133[p] = __shfl_sync(4294967295U, (shuffle2234[0].x), (0), ((128 / 16)));
            }
            for (uint32_t a = 0; a < 64; a += 4) {
                int t2335;
                t2335 = bfe(a, 2U, 3U);
                for (uint32_t b = 0; b < 4; b++) {
                    for (int p = 0; p < 4; p++) {
                        offset2032[p] = ((init02133[p] ^ (a + b)) * 16777619 ^ (((uint32_t *)&mix1931[p])[b])) % d_dag_size;
                        offset2032[p] = __shfl_sync(4294967295U, (offset2032[p]), (t2335), ((128 / 16)));
                        mix1931[p] = fnv4(mix1931[p], d_dag[offset2032[p]].uint4s[thread_id1628]);
                    }
                }
            }
            for (int p = 0; p < 4; p++) {
                uint2 shuffle2436[4];
                uint32_t thread_mix2537;
                thread_mix2537 = fnv_reduce(mix1931[p]);
                shuffle2436[0].x = __shfl_sync(4294967295U, (thread_mix2537), (0), ((128 / 16)));
                shuffle2436[0].y = __shfl_sync(4294967295U, (thread_mix2537), (1), ((128 / 16)));
                shuffle2436[1].x = __shfl_sync(4294967295U, (thread_mix2537), (2), ((128 / 16)));
                shuffle2436[1].y = __shfl_sync(4294967295U, (thread_mix2537), (3), ((128 / 16)));
                shuffle2436[2].x = __shfl_sync(4294967295U, (thread_mix2537), (4), ((128 / 16)));
                shuffle2436[2].y = __shfl_sync(4294967295U, (thread_mix2537), (5), ((128 / 16)));
                shuffle2436[3].x = __shfl_sync(4294967295U, (thread_mix2537), (6), ((128 / 16)));
                shuffle2436[3].y = __shfl_sync(4294967295U, (thread_mix2537), (7), ((128 / 16)));
                if ((i + p) == thread_id1628) {
                    state1527[8] = shuffle2436[0];
                    state1527[9] = shuffle2436[1];
                    state1527[10] = shuffle2436[2];
                    state1527[11] = shuffle2436[3];
                }
            }
        }
        if (!(cuda_swab64(keccak_f1600_final(state1527)) > d_target)) {
            mix_hash1325[0] = state1527[8];
            mix_hash1325[1] = state1527[9];
            mix_hash1325[2] = state1527[10];
            mix_hash1325[3] = state1527[11];
            return;
        }
        uint32_t index1830;
        index1830 = atomicInc((uint32_t *)&g_output814->count, 4294967295U);
        if (index1830 >= 4U)
            return;
        g_output814->result[index1830].gid = gid1022;
        g_output814->result[index1830].mix[0] = mix1123[0].x;
        g_output814->result[index1830].mix[1] = mix1123[0].y;
        g_output814->result[index1830].mix[2] = mix1123[1].x;
        g_output814->result[index1830].mix[3] = mix1123[1].y;
        g_output814->result[index1830].mix[4] = mix1123[2].x;
        g_output814->result[index1830].mix[5] = mix1123[2].y;
        g_output814->result[index1830].mix[6] = mix1123[3].x;
        g_output814->result[index1830].mix[7] = mix1123[3].y;
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
        unsigned int blockDim_x_04;
        blockDim_x_04 = 128;
        unsigned int threadIdx_x_05;
        threadIdx_x_05 = ((threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0 + threadIdx_z_0 * blockDim_x_0 * blockDim_y_0) - 128) % 128;
        unsigned int blockDim_y_06;
        blockDim_y_06 = 1;
        unsigned int threadIdx_y_07;
        threadIdx_y_07 = ((threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0 + threadIdx_z_0 * blockDim_x_0 * blockDim_y_0) - 128) / 128 % 1;
        unsigned int blockDim_z_08;
        blockDim_z_08 = 1;
        unsigned int threadIdx_z_09;
        threadIdx_z_09 = ((threadIdx_x_0 + threadIdx_y_0 * blockDim_x_0 + threadIdx_z_0 * blockDim_x_0 * blockDim_y_0) - 128) / 128;
        for (int i = 0; i < 80; i++) {
            uint32_t nonce410;
            nonce410 = (blockDim_x_04 * blockIdx.x + threadIdx_x_05) * 80 + i + startNonce11;
            uint64_t m511[16];
            m511[0] = d_data[0];
            m511[1] = d_data[1];
            m511[2] = d_data[2];
            m511[3] = d_data[3];
            m511[4] = d_data[4];
            m511[5] = d_data[5];
            m511[6] = d_data[6];
            m511[7] = d_data[7];
            m511[8] = d_data[8];
            ((uint32_t *)m511)[18] = *((uint32_t *)(&d_data[9]));
            ((uint32_t *)m511)[19] = nonce410;
            m511[10] = m511[11] = 0;
            m511[12] = m511[13] = 0;
            m511[14] = m511[15] = 0;
            uint64_t v612[16] = {7640891576939301160L, 13503953896175478587UL, 4354685564936845355L, 11912009170470909681UL, 5840696475078001361L, 11170449401992604703UL, 2270897969802886507L, 6620516959819538809L, 7640891576956012808L, 13503953896175478587UL, 4354685564936845355L, 11912009170470909681UL, 5840696475078001281L, 11170449401992604703UL, 16175846103906665108UL, 6620516959819538809L};
            G(0, 0, v612[0], v612[4], v612[8], v612[12], m511);
            G(0, 1, v612[1], v612[5], v612[9], v612[13], m511);
            G(0, 2, v612[2], v612[6], v612[10], v612[14], m511);
            G(0, 3, v612[3], v612[7], v612[11], v612[15], m511);
            G(0, 4, v612[0], v612[5], v612[10], v612[15], m511);
            G(0, 5, v612[1], v612[6], v612[11], v612[12], m511);
            G(0, 6, v612[2], v612[7], v612[8], v612[13], m511);
            G(0, 7, v612[3], v612[4], v612[9], v612[14], m511);
            ;
            G(1, 0, v612[0], v612[4], v612[8], v612[12], m511);
            G(1, 1, v612[1], v612[5], v612[9], v612[13], m511);
            G(1, 2, v612[2], v612[6], v612[10], v612[14], m511);
            G(1, 3, v612[3], v612[7], v612[11], v612[15], m511);
            G(1, 4, v612[0], v612[5], v612[10], v612[15], m511);
            G(1, 5, v612[1], v612[6], v612[11], v612[12], m511);
            G(1, 6, v612[2], v612[7], v612[8], v612[13], m511);
            G(1, 7, v612[3], v612[4], v612[9], v612[14], m511);
            ;
            G(2, 0, v612[0], v612[4], v612[8], v612[12], m511);
            G(2, 1, v612[1], v612[5], v612[9], v612[13], m511);
            G(2, 2, v612[2], v612[6], v612[10], v612[14], m511);
            G(2, 3, v612[3], v612[7], v612[11], v612[15], m511);
            G(2, 4, v612[0], v612[5], v612[10], v612[15], m511);
            G(2, 5, v612[1], v612[6], v612[11], v612[12], m511);
            G(2, 6, v612[2], v612[7], v612[8], v612[13], m511);
            G(2, 7, v612[3], v612[4], v612[9], v612[14], m511);
            ;
            G(3, 0, v612[0], v612[4], v612[8], v612[12], m511);
            G(3, 1, v612[1], v612[5], v612[9], v612[13], m511);
            G(3, 2, v612[2], v612[6], v612[10], v612[14], m511);
            G(3, 3, v612[3], v612[7], v612[11], v612[15], m511);
            G(3, 4, v612[0], v612[5], v612[10], v612[15], m511);
            G(3, 5, v612[1], v612[6], v612[11], v612[12], m511);
            G(3, 6, v612[2], v612[7], v612[8], v612[13], m511);
            G(3, 7, v612[3], v612[4], v612[9], v612[14], m511);
            ;
            G(4, 0, v612[0], v612[4], v612[8], v612[12], m511);
            G(4, 1, v612[1], v612[5], v612[9], v612[13], m511);
            G(4, 2, v612[2], v612[6], v612[10], v612[14], m511);
            G(4, 3, v612[3], v612[7], v612[11], v612[15], m511);
            G(4, 4, v612[0], v612[5], v612[10], v612[15], m511);
            G(4, 5, v612[1], v612[6], v612[11], v612[12], m511);
            G(4, 6, v612[2], v612[7], v612[8], v612[13], m511);
            G(4, 7, v612[3], v612[4], v612[9], v612[14], m511);
            ;
            G(5, 0, v612[0], v612[4], v612[8], v612[12], m511);
            G(5, 1, v612[1], v612[5], v612[9], v612[13], m511);
            G(5, 2, v612[2], v612[6], v612[10], v612[14], m511);
            G(5, 3, v612[3], v612[7], v612[11], v612[15], m511);
            G(5, 4, v612[0], v612[5], v612[10], v612[15], m511);
            G(5, 5, v612[1], v612[6], v612[11], v612[12], m511);
            G(5, 6, v612[2], v612[7], v612[8], v612[13], m511);
            G(5, 7, v612[3], v612[4], v612[9], v612[14], m511);
            ;
            G(6, 0, v612[0], v612[4], v612[8], v612[12], m511);
            G(6, 1, v612[1], v612[5], v612[9], v612[13], m511);
            G(6, 2, v612[2], v612[6], v612[10], v612[14], m511);
            G(6, 3, v612[3], v612[7], v612[11], v612[15], m511);
            G(6, 4, v612[0], v612[5], v612[10], v612[15], m511);
            G(6, 5, v612[1], v612[6], v612[11], v612[12], m511);
            G(6, 6, v612[2], v612[7], v612[8], v612[13], m511);
            G(6, 7, v612[3], v612[4], v612[9], v612[14], m511);
            ;
            G(7, 0, v612[0], v612[4], v612[8], v612[12], m511);
            G(7, 1, v612[1], v612[5], v612[9], v612[13], m511);
            G(7, 2, v612[2], v612[6], v612[10], v612[14], m511);
            G(7, 3, v612[3], v612[7], v612[11], v612[15], m511);
            G(7, 4, v612[0], v612[5], v612[10], v612[15], m511);
            G(7, 5, v612[1], v612[6], v612[11], v612[12], m511);
            G(7, 6, v612[2], v612[7], v612[8], v612[13], m511);
            G(7, 7, v612[3], v612[4], v612[9], v612[14], m511);
            ;
            G(8, 0, v612[0], v612[4], v612[8], v612[12], m511);
            G(8, 1, v612[1], v612[5], v612[9], v612[13], m511);
            G(8, 2, v612[2], v612[6], v612[10], v612[14], m511);
            G(8, 3, v612[3], v612[7], v612[11], v612[15], m511);
            G(8, 4, v612[0], v612[5], v612[10], v612[15], m511);
            G(8, 5, v612[1], v612[6], v612[11], v612[12], m511);
            G(8, 6, v612[2], v612[7], v612[8], v612[13], m511);
            G(8, 7, v612[3], v612[4], v612[9], v612[14], m511);
            ;
            G(9, 0, v612[0], v612[4], v612[8], v612[12], m511);
            G(9, 1, v612[1], v612[5], v612[9], v612[13], m511);
            G(9, 2, v612[2], v612[6], v612[10], v612[14], m511);
            G(9, 3, v612[3], v612[7], v612[11], v612[15], m511);
            G(9, 4, v612[0], v612[5], v612[10], v612[15], m511);
            G(9, 5, v612[1], v612[6], v612[11], v612[12], m511);
            G(9, 6, v612[2], v612[7], v612[8], v612[13], m511);
            G(9, 7, v612[3], v612[4], v612[9], v612[14], m511);
            ;
            G(10, 0, v612[0], v612[4], v612[8], v612[12], m511);
            G(10, 1, v612[1], v612[5], v612[9], v612[13], m511);
            G(10, 2, v612[2], v612[6], v612[10], v612[14], m511);
            G(10, 3, v612[3], v612[7], v612[11], v612[15], m511);
            G(10, 4, v612[0], v612[5], v612[10], v612[15], m511);
            G(10, 5, v612[1], v612[6], v612[11], v612[12], m511);
            G(10, 6, v612[2], v612[7], v612[8], v612[13], m511);
            G(10, 7, v612[3], v612[4], v612[9], v612[14], m511);
            ;
            G(11, 0, v612[0], v612[4], v612[8], v612[12], m511);
            G(11, 1, v612[1], v612[5], v612[9], v612[13], m511);
            G(11, 2, v612[2], v612[6], v612[10], v612[14], m511);
            G(11, 3, v612[3], v612[7], v612[11], v612[15], m511);
            G(11, 4, v612[0], v612[5], v612[10], v612[15], m511);
            G(11, 5, v612[1], v612[6], v612[11], v612[12], m511);
            G(11, 6, v612[2], v612[7], v612[8], v612[13], m511);
            G(11, 7, v612[3], v612[4], v612[9], v612[14], m511);
            ;
            uint2 last713;
            last713 = vectorize(v612[3] ^ v612[11] ^ 11912009170470909681UL);
            if (last713.y <= target233.y && last713.x <= target233.x) {
                resNonce22[1] = resNonce22[0];
                resNonce22[0] = nonce410;
            }
        }
    }
}

extern __global__ void ethash_search(volatile Search_results* g_output, uint64_t start_nonce);



void run_ethash_search_blake(uint32_t gridSize, uint32_t blockSize, cudaStream_t stream,
                       volatile Search_results* g_output, uint64_t start_nonce)
{
    {
        uint32_t threads = 1048576;
        dim3 grid((threads + TPB-1)/TPB);
        dim3 block(TPB);
        auto thr_id = 0;
        CUDA_SAFE_CALL(cudaMalloc(&d_resNonces[thr_id], NBN * sizeof(uint32_t)));
        if (cudaMemset(d_resNonces[thr_id], 0xff, NBN*sizeof(uint32_t)) != cudaSuccess)
            return;
        const uint2 target2 = make_uint2(0, 1);
        cudaStream_t t1;
        cudaStream_t t2;
        cudaStreamCreate ( &t1);
        cudaStreamCreate ( &t2);
        ethash_search<<<gridSize, blockSize, 0, t2>>>(g_output, start_nonce);
        blake2b_gpu_hash <<<grid, block, 8, t1>>> (threads, 0, d_resNonces[thr_id], target2);
        cudaDeviceSynchronize();
        ethash_search2_blake2b_gpu_hash_0<<<grid, 256, 8>>>
        (g_output, start_nonce, threads, 0, d_resNonces[thr_id], target2);
        cudaDeviceSynchronize();
        ethash_search2_blake2b_gpu_hash_100<<<grid, 128, 8>>>
            (g_output, start_nonce, threads, 0, d_resNonces[thr_id], target2);
    }



    CUDA_SAFE_CALL(cudaGetLastError());
}

