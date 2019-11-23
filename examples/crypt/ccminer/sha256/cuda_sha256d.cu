/*
 * sha256d CUDA implementation.
 * tpruvot 2017
 */

#include <stdio.h>
#include <stdint.h>
#include <memory.h>

#include <cuda_helper.h>
#include <miner.h>


const __constant__  uint32_t __align__(8) c_H256[8] = {
	0x6A09E667U, 0xBB67AE85U, 0x3C6EF372U, 0xA54FF53AU,
	0x510E527FU, 0x9B05688CU, 0x1F83D9ABU, 0x5BE0CD19U
};
__device__ uint64_t dd_target[1];


// ------------------------------------------------------------------------------------------------

static const uint32_t cpu_H256[8] = {
	0x6A09E667U, 0xBB67AE85U, 0x3C6EF372U, 0xA54FF53AU,
	0x510E527FU, 0x9B05688CU, 0x1F83D9ABU, 0x5BE0CD19U
};


#define ROTR ROTR32

__host__
static void sha256_step1_host(uint32_t a, uint32_t b, uint32_t c, uint32_t &d,
	uint32_t e, uint32_t f, uint32_t g, uint32_t &h,
	uint32_t in, const uint32_t Kshared)
{
	uint32_t t1,t2;
	uint32_t vxandx = (((f) ^ (g)) & (e)) ^ (g); // xandx(e, f, g);
	uint32_t bsg21 = ROTR(e, 6) ^ ROTR(e, 11) ^ ROTR(e, 25); // bsg2_1(e);
	uint32_t bsg20 = ROTR(a, 2) ^ ROTR(a, 13) ^ ROTR(a, 22); //bsg2_0(a);
	uint32_t andorv = ((b) & (c)) | (((b) | (c)) & (a)); //andor32(a,b,c);

	t1 = h + bsg21 + vxandx + Kshared + in;
	t2 = bsg20 + andorv;
	d = d + t1;
	h = t1 + t2;
}

__host__
static void sha256_step2_host(uint32_t a, uint32_t b, uint32_t c, uint32_t &d,
	uint32_t e, uint32_t f, uint32_t g, uint32_t &h,
	uint32_t* in, uint32_t pc, const uint32_t Kshared)
{
	uint32_t t1,t2;

	int pcidx1 = (pc-2)  & 0xF;
	int pcidx2 = (pc-7)  & 0xF;
	int pcidx3 = (pc-15) & 0xF;

	uint32_t inx0 = in[pc];
	uint32_t inx1 = in[pcidx1];
	uint32_t inx2 = in[pcidx2];
	uint32_t inx3 = in[pcidx3];

	uint32_t ssg21 = ROTR(inx1, 17) ^ ROTR(inx1, 19) ^ SPH_T32((inx1) >> 10); //ssg2_1(inx1);
	uint32_t ssg20 = ROTR(inx3, 7) ^ ROTR(inx3, 18) ^ SPH_T32((inx3) >> 3); //ssg2_0(inx3);
	uint32_t vxandx = (((f) ^ (g)) & (e)) ^ (g); // xandx(e, f, g);
	uint32_t bsg21 = ROTR(e, 6) ^ ROTR(e, 11) ^ ROTR(e, 25); // bsg2_1(e);
	uint32_t bsg20 = ROTR(a, 2) ^ ROTR(a, 13) ^ ROTR(a, 22); //bsg2_0(a);
	uint32_t andorv = ((b) & (c)) | (((b) | (c)) & (a)); //andor32(a,b,c);

	in[pc] = ssg21 + inx2 + ssg20 + inx0;

	t1 = h + bsg21 + vxandx + Kshared + in[pc];
	t2 = bsg20 + andorv;
	d =  d + t1;
	h = t1 + t2;
}

__host__
static void sha256_round_body_host(uint32_t* in, uint32_t* state, const uint32_t* Kshared)
{
	uint32_t a = state[0];
	uint32_t b = state[1];
	uint32_t c = state[2];
	uint32_t d = state[3];
	uint32_t e = state[4];
	uint32_t f = state[5];
	uint32_t g = state[6];
	uint32_t h = state[7];

	sha256_step1_host(a,b,c,d,e,f,g,h,in[ 0], Kshared[ 0]);
	sha256_step1_host(h,a,b,c,d,e,f,g,in[ 1], Kshared[ 1]);
	sha256_step1_host(g,h,a,b,c,d,e,f,in[ 2], Kshared[ 2]);
	sha256_step1_host(f,g,h,a,b,c,d,e,in[ 3], Kshared[ 3]);
	sha256_step1_host(e,f,g,h,a,b,c,d,in[ 4], Kshared[ 4]);
	sha256_step1_host(d,e,f,g,h,a,b,c,in[ 5], Kshared[ 5]);
	sha256_step1_host(c,d,e,f,g,h,a,b,in[ 6], Kshared[ 6]);
	sha256_step1_host(b,c,d,e,f,g,h,a,in[ 7], Kshared[ 7]);
	sha256_step1_host(a,b,c,d,e,f,g,h,in[ 8], Kshared[ 8]);
	sha256_step1_host(h,a,b,c,d,e,f,g,in[ 9], Kshared[ 9]);
	sha256_step1_host(g,h,a,b,c,d,e,f,in[10], Kshared[10]);
	sha256_step1_host(f,g,h,a,b,c,d,e,in[11], Kshared[11]);
	sha256_step1_host(e,f,g,h,a,b,c,d,in[12], Kshared[12]);
	sha256_step1_host(d,e,f,g,h,a,b,c,in[13], Kshared[13]);
	sha256_step1_host(c,d,e,f,g,h,a,b,in[14], Kshared[14]);
	sha256_step1_host(b,c,d,e,f,g,h,a,in[15], Kshared[15]);

	for (int i=0; i<3; i++)
	{
		sha256_step2_host(a,b,c,d,e,f,g,h,in,0, Kshared[16+16*i]);
		sha256_step2_host(h,a,b,c,d,e,f,g,in,1, Kshared[17+16*i]);
		sha256_step2_host(g,h,a,b,c,d,e,f,in,2, Kshared[18+16*i]);
		sha256_step2_host(f,g,h,a,b,c,d,e,in,3, Kshared[19+16*i]);
		sha256_step2_host(e,f,g,h,a,b,c,d,in,4, Kshared[20+16*i]);
		sha256_step2_host(d,e,f,g,h,a,b,c,in,5, Kshared[21+16*i]);
		sha256_step2_host(c,d,e,f,g,h,a,b,in,6, Kshared[22+16*i]);
		sha256_step2_host(b,c,d,e,f,g,h,a,in,7, Kshared[23+16*i]);
		sha256_step2_host(a,b,c,d,e,f,g,h,in,8, Kshared[24+16*i]);
		sha256_step2_host(h,a,b,c,d,e,f,g,in,9, Kshared[25+16*i]);
		sha256_step2_host(g,h,a,b,c,d,e,f,in,10,Kshared[26+16*i]);
		sha256_step2_host(f,g,h,a,b,c,d,e,in,11,Kshared[27+16*i]);
		sha256_step2_host(e,f,g,h,a,b,c,d,in,12,Kshared[28+16*i]);
		sha256_step2_host(d,e,f,g,h,a,b,c,in,13,Kshared[29+16*i]);
		sha256_step2_host(c,d,e,f,g,h,a,b,in,14,Kshared[30+16*i]);
		sha256_step2_host(b,c,d,e,f,g,h,a,in,15,Kshared[31+16*i]);
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

__global__
/*__launch_bounds__(256,3)*/
void sha256d_gpu_hash_shared(const uint32_t threads, const uint32_t startNonce, uint32_t *resNonces)
{
	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);

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
		#pragma unroll
		for (int i=5; i<15; i++) dat[i] = 0;

		uint32_t buf[8];
		#pragma unroll
		for (int i=0; i<8; i+=2) AS_UINT2(&buf[i]) = AS_UINT2(&c_midstate76[i]);
		//for (int i=0; i<8; i++) buf[i] = c_midstate76[i];

		sha256_round_body(dat, buf, s_K);

		// second sha256

		#pragma unroll
		for (int i=0; i<8; i++) dat[i] = buf[i];
		dat[8] = 0x80000000;
		#pragma unroll
		for (int i=9; i<15; i++) dat[i] = 0;
		dat[15] = 0x100;

		#pragma unroll
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

__host__
void sha256d_init(int thr_id)
{
	cuda_get_arch(thr_id);
	cudaMemcpyToSymbol(c_K, cpu_K, sizeof(cpu_K), 0, cudaMemcpyHostToDevice);
	CUDA_SAFE_CALL(cudaMalloc(&d_resNonces[thr_id], 2*sizeof(uint32_t)));
}

__host__
void sha256d_free(int thr_id)
{
	if (d_resNonces[thr_id]) cudaFree(d_resNonces[thr_id]);
	d_resNonces[thr_id] = NULL;
}

__host__
void sha256d_setBlock_80(uint32_t *pdata, uint32_t *ptarget)
{
	uint32_t _ALIGN(64) in[16], buf[8], end[4];
	for (int i=0;i<16;i++) in[i] = cuda_swab32(pdata[i]);
	for (int i=0;i<8;i++) buf[i] = cpu_H256[i];
	for (int i=0;i<4;i++) end[i] = cuda_swab32(pdata[16+i]);
	sha256_round_body_host(in, buf, cpu_K);

	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_midstate76, buf, 32, 0, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_dataEnd80,  end, sizeof(end), 0, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(c_target, &ptarget[6], 8, 0, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(dd_target, &ptarget[6], 8, 0, cudaMemcpyHostToDevice));
}

__host__
void sha256d_hash_80(int thr_id, uint32_t threads, uint32_t startNonce, uint32_t *resNonces)
{
	const uint32_t threadsperblock = 256;

	dim3 grid(threads/threadsperblock);
	dim3 block(threadsperblock);

	CUDA_SAFE_CALL(cudaMemset(d_resNonces[thr_id], 0xFF, 2 * sizeof(uint32_t)));
	cudaThreadSynchronize();
	sha256d_gpu_hash_shared <<<grid, block>>> (threads, startNonce, d_resNonces[thr_id]);
	cudaThreadSynchronize();

	CUDA_SAFE_CALL(cudaMemcpy(resNonces, d_resNonces[thr_id], 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
	if (resNonces[0] == resNonces[1]) {
		resNonces[1] = UINT32_MAX;
	}
}
