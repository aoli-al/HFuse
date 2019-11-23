#pragma once

extern __constant__ uint32_t d_dag_size;
extern __constant__ hash128_t* d_dag;
extern __constant__ uint32_t d_light_size;
extern __constant__ hash64_t* d_light;
extern __constant__ hash32_t d_header;
extern __constant__ uint64_t d_target;

#define SHFL(x, y, z) __shfl_sync(0xFFFFFFFF, (x), (y), (z))

#define LDG(x) (x)
