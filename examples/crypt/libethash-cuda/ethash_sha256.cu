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

__device__ uint64_t dd_target[1];
static uint32_t* d_sha256_resNonces[MAX_GPUS] = { 0 };
__constant__ static uint32_t __align__(8) c_target[2];
const __constant__  uint32_t __align__(8) c_H256[8] = {
    0x6A09E667U, 0xBB67AE85U, 0x3C6EF372U, 0xA54FF53AU,
    0x510E527FU, 0x9B05688CU, 0x1F83D9ABU, 0x5BE0CD19U
};

static const uint32_t cpu_K[64] = {
    0x428A2F98, 0x71374491, 0xB5C0FBCF, 0xE9B5DBA5, 0x3956C25B, 0x59F111F1, 0x923F82A4, 0xAB1C5ED5,
    0xD807AA98, 0x12835B01, 0x243185BE, 0x550C7DC3, 0x72BE5D74, 0x80DEB1FE, 0x9BDC06A7, 0xC19BF174,
    0xE49B69C1, 0xEFBE4786, 0x0FC19DC6, 0x240CA1CC, 0x2DE92C6F, 0x4A7484AA, 0x5CB0A9DC, 0x76F988DA,
    0x983E5152, 0xA831C66D, 0xB00327C8, 0xBF597FC7, 0xC6E00BF3, 0xD5A79147, 0x06CA6351, 0x14292967,
    0x27B70A85, 0x2E1B2138, 0x4D2C6DFC, 0x53380D13, 0x650A7354, 0x766A0ABB, 0x81C2C92E, 0x92722C85,
    0xA2BFE8A1, 0xA81A664B, 0xC24B8B70, 0xC76C51A3, 0xD192E819, 0xD6990624, 0xF40E3585, 0x106AA070,
    0x19A4C116, 0x1E376C08, 0x2748774C, 0x34B0BCB5, 0x391C0CB3, 0x4ED8AA4A, 0x5B9CCA4F, 0x682E6FF3,
    0x748F82EE, 0x78A5636F, 0x84C87814, 0x8CC70208, 0x90BEFFFA, 0xA4506CEB, 0xBEF9A3F7, 0xC67178F2
};
__constant__ static uint32_t __align__(8) c_K[64];
__constant__ static uint32_t __align__(8) c_midstate76[8];
__constant__ static uint32_t __align__(8) c_dataEnd80[4];

#define xor3b(a,b,c) (a ^ b ^ c)

__device__ __forceinline__ uint32_t bsg2_0(const uint32_t x)
{
    return xor3b(ROTR32(x,2),ROTR32(x,13),ROTR32(x,22));
}

__device__ __forceinline__ uint32_t bsg2_1(const uint32_t x)
{
    return xor3b(ROTR32(x,6),ROTR32(x,11),ROTR32(x,25));
}

__device__ __forceinline__ uint32_t ssg2_0(const uint32_t x)
{
    return xor3b(ROTR32(x,7),ROTR32(x,18),(x>>3));
}

__device__ __forceinline__ uint32_t ssg2_1(const uint32_t x)
{
    return xor3b(ROTR32(x,17),ROTR32(x,19),(x>>10));
}


__device__
static void sha2_step1(uint32_t a, uint32_t b, uint32_t c, uint32_t &d, uint32_t e, uint32_t f, uint32_t g, uint32_t &h,
                       uint32_t in, const uint32_t Kshared)
{
    uint32_t t1,t2;
    uint32_t vxandx = xandx(e, f, g);
    uint32_t bsg21 = bsg2_1(e);
    uint32_t bsg20 = bsg2_0(a);
    uint32_t andorv = andor32(a,b,c);

    t1 = h + bsg21 + vxandx + Kshared + in;
    t2 = bsg20 + andorv;
    d = d + t1;
    h = t1 + t2;
}

__device__
static void sha2_step2(uint32_t a, uint32_t b, uint32_t c, uint32_t &d, uint32_t e, uint32_t f, uint32_t g, uint32_t &h,
                       uint32_t* in, uint32_t pc, const uint32_t Kshared)
{
    uint32_t t1,t2;

    int pcidx1 = (pc-2) & 0xF;
    int pcidx2 = (pc-7) & 0xF;
    int pcidx3 = (pc-15) & 0xF;

    uint32_t inx0 = in[pc];
    uint32_t inx1 = in[pcidx1];
    uint32_t inx2 = in[pcidx2];
    uint32_t inx3 = in[pcidx3];

    uint32_t ssg21 = ssg2_1(inx1);
    uint32_t ssg20 = ssg2_0(inx3);
    uint32_t vxandx = xandx(e, f, g);
    uint32_t bsg21 = bsg2_1(e);
    uint32_t bsg20 = bsg2_0(a);
    uint32_t andorv = andor32(a,b,c);

    in[pc] = ssg21 + inx2 + ssg20 + inx0;

    t1 = h + bsg21 + vxandx + Kshared + in[pc];
    t2 = bsg20 + andorv;
    d =  d + t1;
    h = t1 + t2;
}

__device__
static void sha256_round_body(uint32_t* in, uint32_t* state, uint32_t* const Kshared)
{
    uint32_t a = state[0];
    uint32_t b = state[1];
    uint32_t c = state[2];
    uint32_t d = state[3];
    uint32_t e = state[4];
    uint32_t f = state[5];
    uint32_t g = state[6];
    uint32_t h = state[7];

    sha2_step1(a,b,c,d,e,f,g,h,in[ 0], Kshared[ 0]);
    sha2_step1(h,a,b,c,d,e,f,g,in[ 1], Kshared[ 1]);
    sha2_step1(g,h,a,b,c,d,e,f,in[ 2], Kshared[ 2]);
    sha2_step1(f,g,h,a,b,c,d,e,in[ 3], Kshared[ 3]);
    sha2_step1(e,f,g,h,a,b,c,d,in[ 4], Kshared[ 4]);
    sha2_step1(d,e,f,g,h,a,b,c,in[ 5], Kshared[ 5]);
    sha2_step1(c,d,e,f,g,h,a,b,in[ 6], Kshared[ 6]);
    sha2_step1(b,c,d,e,f,g,h,a,in[ 7], Kshared[ 7]);
    sha2_step1(a,b,c,d,e,f,g,h,in[ 8], Kshared[ 8]);
    sha2_step1(h,a,b,c,d,e,f,g,in[ 9], Kshared[ 9]);
    sha2_step1(g,h,a,b,c,d,e,f,in[10], Kshared[10]);
    sha2_step1(f,g,h,a,b,c,d,e,in[11], Kshared[11]);
    sha2_step1(e,f,g,h,a,b,c,d,in[12], Kshared[12]);
    sha2_step1(d,e,f,g,h,a,b,c,in[13], Kshared[13]);
    sha2_step1(c,d,e,f,g,h,a,b,in[14], Kshared[14]);
    sha2_step1(b,c,d,e,f,g,h,a,in[15], Kshared[15]);

#pragma unroll
    for (int i=0; i<3; i++)
    {
        sha2_step2(a,b,c,d,e,f,g,h,in,0, Kshared[16+16*i]);
        sha2_step2(h,a,b,c,d,e,f,g,in,1, Kshared[17+16*i]);
        sha2_step2(g,h,a,b,c,d,e,f,in,2, Kshared[18+16*i]);
        sha2_step2(f,g,h,a,b,c,d,e,in,3, Kshared[19+16*i]);
        sha2_step2(e,f,g,h,a,b,c,d,in,4, Kshared[20+16*i]);
        sha2_step2(d,e,f,g,h,a,b,c,in,5, Kshared[21+16*i]);
        sha2_step2(c,d,e,f,g,h,a,b,in,6, Kshared[22+16*i]);
        sha2_step2(b,c,d,e,f,g,h,a,in,7, Kshared[23+16*i]);
        sha2_step2(a,b,c,d,e,f,g,h,in,8, Kshared[24+16*i]);
        sha2_step2(h,a,b,c,d,e,f,g,in,9, Kshared[25+16*i]);
        sha2_step2(g,h,a,b,c,d,e,f,in,10,Kshared[26+16*i]);
        sha2_step2(f,g,h,a,b,c,d,e,in,11,Kshared[27+16*i]);
        sha2_step2(e,f,g,h,a,b,c,d,in,12,Kshared[28+16*i]);
        sha2_step2(d,e,f,g,h,a,b,c,in,13,Kshared[29+16*i]);
        sha2_step2(c,d,e,f,g,h,a,b,in,14,Kshared[30+16*i]);
        sha2_step2(b,c,d,e,f,g,h,a,in,15,Kshared[31+16*i]);
    }

    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
}

__device__
static void sha256_round_last(uint32_t* in, uint32_t* state, uint32_t* const Kshared)
{
    uint32_t a = state[0];
    uint32_t b = state[1];
    uint32_t c = state[2];
    uint32_t d = state[3];
    uint32_t e = state[4];
    uint32_t f = state[5];
    uint32_t g = state[6];
    uint32_t h = state[7];

    sha2_step1(a,b,c,d, e,f,g,h, in[ 0], Kshared[ 0]);
    sha2_step1(h,a,b,c, d,e,f,g, in[ 1], Kshared[ 1]);
    sha2_step1(g,h,a,b, c,d,e,f, in[ 2], Kshared[ 2]);
    sha2_step1(f,g,h,a, b,c,d,e, in[ 3], Kshared[ 3]);
    sha2_step1(e,f,g,h, a,b,c,d, in[ 4], Kshared[ 4]);
    sha2_step1(d,e,f,g, h,a,b,c, in[ 5], Kshared[ 5]);
    sha2_step1(c,d,e,f, g,h,a,b, in[ 6], Kshared[ 6]);
    sha2_step1(b,c,d,e, f,g,h,a, in[ 7], Kshared[ 7]);
    sha2_step1(a,b,c,d, e,f,g,h, in[ 8], Kshared[ 8]);
    sha2_step1(h,a,b,c, d,e,f,g, in[ 9], Kshared[ 9]);
    sha2_step1(g,h,a,b, c,d,e,f, in[10], Kshared[10]);
    sha2_step1(f,g,h,a, b,c,d,e, in[11], Kshared[11]);
    sha2_step1(e,f,g,h, a,b,c,d, in[12], Kshared[12]);
    sha2_step1(d,e,f,g, h,a,b,c, in[13], Kshared[13]);
    sha2_step1(c,d,e,f, g,h,a,b, in[14], Kshared[14]);
    sha2_step1(b,c,d,e, f,g,h,a, in[15], Kshared[15]);

#pragma unroll 2
    for (int i=0; i<2; i++)
    {
        sha2_step2(a,b,c,d, e,f,g,h, in, 0, Kshared[16+16*i]);
        sha2_step2(h,a,b,c, d,e,f,g, in, 1, Kshared[17+16*i]);
        sha2_step2(g,h,a,b, c,d,e,f, in, 2, Kshared[18+16*i]);
        sha2_step2(f,g,h,a, b,c,d,e, in, 3, Kshared[19+16*i]);
        sha2_step2(e,f,g,h, a,b,c,d, in, 4, Kshared[20+16*i]);
        sha2_step2(d,e,f,g, h,a,b,c, in, 5, Kshared[21+16*i]);
        sha2_step2(c,d,e,f, g,h,a,b, in, 6, Kshared[22+16*i]);
        sha2_step2(b,c,d,e, f,g,h,a, in, 7, Kshared[23+16*i]);
        sha2_step2(a,b,c,d, e,f,g,h, in, 8, Kshared[24+16*i]);
        sha2_step2(h,a,b,c, d,e,f,g, in, 9, Kshared[25+16*i]);
        sha2_step2(g,h,a,b, c,d,e,f, in,10, Kshared[26+16*i]);
        sha2_step2(f,g,h,a, b,c,d,e, in,11, Kshared[27+16*i]);
        sha2_step2(e,f,g,h, a,b,c,d, in,12, Kshared[28+16*i]);
        sha2_step2(d,e,f,g, h,a,b,c, in,13, Kshared[29+16*i]);
        sha2_step2(c,d,e,f, g,h,a,b, in,14, Kshared[30+16*i]);
        sha2_step2(b,c,d,e, f,g,h,a, in,15, Kshared[31+16*i]);
    }

    sha2_step2(a,b,c,d, e,f,g,h, in, 0, Kshared[16+16*2]);
    sha2_step2(h,a,b,c, d,e,f,g, in, 1, Kshared[17+16*2]);
    sha2_step2(g,h,a,b, c,d,e,f, in, 2, Kshared[18+16*2]);
    sha2_step2(f,g,h,a, b,c,d,e, in, 3, Kshared[19+16*2]);
    sha2_step2(e,f,g,h, a,b,c,d, in, 4, Kshared[20+16*2]);
    sha2_step2(d,e,f,g, h,a,b,c, in, 5, Kshared[21+16*2]);
    sha2_step2(c,d,e,f, g,h,a,b, in, 6, Kshared[22+16*2]);
    sha2_step2(b,c,d,e, f,g,h,a, in, 7, Kshared[23+16*2]);
    sha2_step2(a,b,c,d, e,f,g,h, in, 8, Kshared[24+16*2]);
    sha2_step2(h,a,b,c, d,e,f,g, in, 9, Kshared[25+16*2]);
    sha2_step2(g,h,a,b, c,d,e,f, in,10, Kshared[26+16*2]);
    sha2_step2(f,g,h,a, b,c,d,e, in,11, Kshared[27+16*2]);
    sha2_step2(e,f,g,h, a,b,c,d, in,12, Kshared[28+16*2]);
    sha2_step2(d,e,f,g, h,a,b,c, in,13, Kshared[29+16*2]);

    state[6] += g;
    state[7] += h;
}

__device__ __forceinline__
uint64_t cuda_swab32ll(uint64_t x) {
    return MAKE_ULONGLONG(cuda_swab32(_LODWORD(x)), cuda_swab32(_HIDWORD(x)));
}


__global__
/*__launch_bounds__(256,3)*/
void sha256d_gpu_hash_shared(const uint32_t threads, const uint32_t startNonce, uint32_t *resNonces)
{
    for (int i = 0; i < 40; i++) {
        const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x) * 40 + i;

        __shared__ uint32_t s_K[64*4];
        //s_K[thread & 63] = c_K[thread & 63];
        if (threadIdx.x < 64U) s_K[threadIdx.x] = c_K[threadIdx.x];

        if (thread < threads)
        {
            const uint32_t nonce = startNonce + thread;

            uint32_t dat[16];
            AS_UINT2(dat) = AS_UINT2(c_dataEnd80);
            dat[ 2] = c_dataEnd80[2];
            dat[ 3] = nonce;
            dat[ 4] = 0x80000000;
            dat[15] = 0x280;
#pragma unroll 10
            for (int i=5; i<15; i++) dat[i] = 0;

            uint32_t buf[8];
#pragma unroll 4
            for (int i=0; i<8; i+=2) AS_UINT2(&buf[i]) = AS_UINT2(&c_midstate76[i]);
            //for (int i=0; i<8; i++) buf[i] = c_midstate76[i];

            sha256_round_body(dat, buf, s_K);

            // second sha256

#pragma unroll 8
            for (int i=0; i<8; i++) dat[i] = buf[i];
            dat[8] = 0x80000000;
#pragma unroll 6
            for (int i=9; i<15; i++) dat[i] = 0;
            dat[15] = 0x100;

#pragma unroll 8
            for (int i=0; i<8; i++) buf[i] = c_H256[i];

            sha256_round_last(dat, buf, s_K);

            // valid nonces
            uint64_t high = cuda_swab32ll(((uint64_t*)buf)[3]);
            if (high <= c_target[0]) {
                //printf("%08x %08x - %016llx %016llx - %08x %08x\n", buf[7], buf[6], high, d_target[0], c_target[1], c_target[0]);
                resNonces[1] = atomicExch(resNonces, nonce);
                //d_target[0] = high;
            }
        }
    }
}


#include "dagger_shuffled.cuh"

__global__ void ethash_search3(volatile Search_results* g_output, uint64_t start_nonce)
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

__attribute__((global)) void sha256d_gpu_hash_shared_ethash_search3_0(const uint32_t threads0, const uint32_t startNonce1, uint32_t *resNonces2, volatile Search_results *g_output9, uint64_t start_nonce10)
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
    for (int i = 0; i < 40; i++) {
        uint32_t thread3;
        thread3 = (blockDim_x_0 * blockIdx.x + threadIdx_x_0) * 40 + i;
        static uint32_t s_K4[256] __attribute__((shared));
        if (threadIdx_x_0 < 64U)
            s_K4[threadIdx_x_0] = c_K[threadIdx_x_0];
        if (thread3 < threads0) {
            uint32_t nonce5;
            nonce5 = startNonce1 + thread3;
            uint32_t dat6[16];
            *((uint2 *)(dat6)) = *((uint2 *)(c_dataEnd80));
            dat6[2] = c_dataEnd80[2];
            dat6[3] = nonce5;
            dat6[4] = 2147483648U;
            dat6[15] = 640;
#pragma unroll (10)
            for (int i = 5; i < 15; i++)
                dat6[i] = 0;
            uint32_t buf7[8];
#pragma unroll (4)
            for (int i = 0; i < 8; i += 2)
                *((uint2 *)(&buf7[i])) = *((uint2 *)(&c_midstate76[i]));
            sha256_round_body(dat6, buf7, s_K4);
#pragma unroll (8)
            for (int i = 0; i < 8; i++)
                dat6[i] = buf7[i];
            dat6[8] = 2147483648U;
#pragma unroll (6)
            for (int i = 9; i < 15; i++)
                dat6[i] = 0;
            dat6[15] = 256;
#pragma unroll (8)
            for (int i = 0; i < 8; i++)
                buf7[i] = c_H256[i];
            sha256_round_last(dat6, buf7, s_K4);
            uint64_t high8;
            high8 = cuda_swab32ll(((uint64_t *)buf7)[3]);
            if (high8 <= c_target[0]) {
                resNonces2[1] = atomicExch(resNonces2, nonce5);
            }
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


__attribute__((global)) void sha256d_gpu_hash_shared_ethash_search3_100(const uint32_t threads0, const uint32_t startNonce1, uint32_t *resNonces2, volatile Search_results *g_output9, uint64_t start_nonce10)
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
        for (int i = 0; i < 40; i++) {
            uint32_t thread3;
            thread3 = (blockDim_x_0 * blockIdx.x + threadIdx_x_0) * 40 + i;
            static uint32_t s_K4[256] __attribute__((shared));
            if (threadIdx_x_0 < 64U)
                s_K4[threadIdx_x_0] = c_K[threadIdx_x_0];
            if (thread3 < threads0) {
                uint32_t nonce5;
                nonce5 = startNonce1 + thread3;
                uint32_t dat6[16];
                *((uint2 *)(dat6)) = *((uint2 *)(c_dataEnd80));
                dat6[2] = c_dataEnd80[2];
                dat6[3] = nonce5;
                dat6[4] = 2147483648U;
                dat6[15] = 640;
#pragma unroll (10)
                for (int i = 5; i < 15; i++)
                    dat6[i] = 0;
                uint32_t buf7[8];
#pragma unroll (4)
                for (int i = 0; i < 8; i += 2)
                    *((uint2 *)(&buf7[i])) = *((uint2 *)(&c_midstate76[i]));
                sha256_round_body(dat6, buf7, s_K4);
#pragma unroll (8)
                for (int i = 0; i < 8; i++)
                    dat6[i] = buf7[i];
                dat6[8] = 2147483648U;
#pragma unroll (6)
                for (int i = 9; i < 15; i++)
                    dat6[i] = 0;
                dat6[15] = 256;
#pragma unroll (8)
                for (int i = 0; i < 8; i++)
                    buf7[i] = c_H256[i];
                sha256_round_last(dat6, buf7, s_K4);
                uint64_t high8;
                high8 = cuda_swab32ll(((uint64_t *)buf7)[3]);
                if (high8 <= c_target[0]) {
                    resNonces2[1] = atomicExch(resNonces2, nonce5);
                }
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


void run_ethash_search_sha256(uint32_t gridSize, uint32_t blockSize, cudaStream_t stream,
    volatile Search_results* g_output, uint64_t start_nonce)
{
//    ethash_search<<<gridSize, blockSize>>>(g_output, start_nonce);
    {
        cudaMemcpyToSymbol(c_K, cpu_K, sizeof(cpu_K), 0, cudaMemcpyHostToDevice);
        CUDA_SAFE_CALL(cudaMalloc(&d_sha256_resNonces[0], 2*sizeof(uint32_t)));
        const uint32_t threads_sha256 = 1048576;
        const uint32_t threadsperblock = 128;
        dim3 grid_sha256(threads_sha256 /threadsperblock);
        dim3 block_sha256(threadsperblock);

        cudaStream_t t1;
        cudaStream_t t2;
        cudaStreamCreate ( &t1);
        cudaStreamCreate ( &t2);
        CUDA_SAFE_CALL(cudaMemset(d_sha256_resNonces[0], 0xFF, 2 * sizeof(uint32_t)));
        cudaThreadSynchronize();
        ethash_search<<<gridSize, blockSize, 0, t2>>>(g_output, start_nonce);
        sha256d_gpu_hash_shared <<<grid_sha256, block_sha256, 0, t1>>> (threads_sha256 * 40, 0, d_sha256_resNonces[0]);
        cudaThreadSynchronize();
        sha256d_gpu_hash_shared_ethash_search3_0<<<grid_sha256, block_sha256.x+blockSize>>>(
            threads_sha256 * 40, 0, d_sha256_resNonces[0],
            g_output, start_nonce);
        cudaThreadSynchronize();
        sha256d_gpu_hash_shared_ethash_search3_100<<<grid_sha256, block_sha256>>>(
            threads_sha256 * 40, 0, d_sha256_resNonces[0],
                g_output, start_nonce);
        cudaThreadSynchronize();
    }
    cudaDeviceSynchronize();
    CUDA_SAFE_CALL(cudaGetLastError());
}

