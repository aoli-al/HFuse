template <typename output_t30, typename input_t31, typename IndexType32, int ADims33, int PDims34, int BDims35, at::native::CUDAHistogramMemoryType MemoryType36 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op37, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(512, 2) void kernelHistogram1D_upsample_bilinear2d_out_frame_fused_kernel_vfuse_lb_idx_0(TensorInfo<output_t30, IndexType32> a38, TensorInfo<output_t30, IndexType32> p39, TensorInfo<input_t31, IndexType32> b40, int nbins41, input_t31 minvalue42, input_t31 maxvalue43, IndexType32 totalElements44, Op37 getOp45, const int ns2, const accscalar_t1 rheight3, const accscalar_t1 rwidth4, const bool align_corners5, const PackedTensorAccessor<scalar_t0, 4> idata6, PackedTensorAccessor<scalar_t0, 4> odata7)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)){
    unsigned int blockDim_x_1 = 512;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 512;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512;
    extern unsigned char my_smem46[] __attribute__((shared));
    output_t30 *smem47 = nullptr;
    smem47 = reinterpret_cast<output_t30 *>(my_smem46);
    for (IndexType32 i = threadIdx_x_1; i < a38.sizes[0]; i += blockDim_x_1) {
        smem47[i] = 0;
    }
    asm ("bar.sync 1,512;");
    ;
    for (IndexType32 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements44; linearIndex += gridDim.x * blockDim_x_1) {
        const IndexType32 bOffset48 = IndexToOffset<input_t31, IndexType32, BDims35>::get(linearIndex, b40);
        const input_t31 bVal49 = b40.data[bOffset48];
        if (bVal49 >= minvalue42 && bVal49 <= maxvalue43) {
            const IndexType32 bin50 = getBin<input_t31, IndexType32>(bVal49, minvalue42, maxvalue43, nbins41);
            atomicAdd(& smem47[bin50], getOp45(linearIndex));
        }
    }
    asm ("bar.sync 1,512;");
    ;
    for (IndexType32 i = threadIdx_x_1; i < a38.sizes[0]; i += blockDim_x_1) {
        const IndexType32 aOffset51 = IndexToOffset<output_t30, IndexType32, ADims33>::get(i, a38);
        atomicAdd(& a38.data[aOffset51], smem47[i]);
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)){
    unsigned int blockDim_x_0 = 512;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 512;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512;
    const int batchsize8 = idata6.size(0);
    const int channels9 = idata6.size(1);
    const int height110 = idata6.size(2);
    const int width111 = idata6.size(3);
    const int height212 = odata7.size(2);
    const int width213 = odata7.size(3);
    for (int index = blockIdx.x * blockDim_x_0 + threadIdx_x_0; index < (ns2); index += blockDim_x_0 * gridDim.x) {
        const int w214 = index % width213;
        const int h215 = index / width213;
        if (height110 == height212 && width111 == width213) {
            const int h126 = h215;
            const int w127 = w214;
            for (int n = 0; n < batchsize8; n++) {
                for (int c = 0; c < channels9; ++c) {
                    const scalar_t0 val28 = idata6[n][c][h126][w127];
                    odata7[n][c][h215][w214] = val28;
                }
            }
            return;
        }
        const accscalar_t1 h1r16 = area_pixel_compute_source_index<accscalar_t1>(rheight3, h215, align_corners5, false);
        const int h117 = h1r16;
        const int h1p18 = (h117 < height110 - 1) ? 1 : 0;
        const accscalar_t1 h1lambda19 = h1r16 - h117;
        const accscalar_t1 h0lambda20 = static_cast<accscalar_t1>(1) - h1lambda19;
        const accscalar_t1 w1r21 = area_pixel_compute_source_index<accscalar_t1>(rwidth4, w214, align_corners5, false);
        const int w122 = w1r21;
        const int w1p23 = (w122 < width111 - 1) ? 1 : 0;
        const accscalar_t1 w1lambda24 = w1r21 - w122;
        const accscalar_t1 w0lambda25 = static_cast<accscalar_t1>(1) - w1lambda24;
        for (int n = 0; n < batchsize8; n++) {
            for (int c = 0; c < channels9; ++c) {
                const accscalar_t1 val29 = h0lambda20 * (w0lambda25 * idata6[n][c][h117][w122] + w1lambda24 * idata6[n][c][h117][w122 + w1p23]) + h1lambda19 * (w0lambda25 * idata6[n][c][h117 + h1p18][w122] + w1lambda24 * idata6[n][c][h117 + h1p18][w122 + w1p23]);
                odata7[n][c][h215][w214] = static_cast<scalar_t0>(val29);
            }
        }
    }
}
}
template <typename output_t30, typename input_t31, typename IndexType32, int ADims33, int PDims34, int BDims35, at::native::CUDAHistogramMemoryType MemoryType36 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op37, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(512, 0) void kernelHistogram1D_upsample_bilinear2d_out_frame_fused_kernel_vfuse_idx_0(TensorInfo<output_t30, IndexType32> a38, TensorInfo<output_t30, IndexType32> p39, TensorInfo<input_t31, IndexType32> b40, int nbins41, input_t31 minvalue42, input_t31 maxvalue43, IndexType32 totalElements44, Op37 getOp45, const int ns2, const accscalar_t1 rheight3, const accscalar_t1 rwidth4, const bool align_corners5, const PackedTensorAccessor<scalar_t0, 4> idata6, PackedTensorAccessor<scalar_t0, 4> odata7)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)){
    unsigned int blockDim_x_1 = 512;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 512;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512;
    extern unsigned char my_smem46[] __attribute__((shared));
    output_t30 *smem47 = nullptr;
    smem47 = reinterpret_cast<output_t30 *>(my_smem46);
    for (IndexType32 i = threadIdx_x_1; i < a38.sizes[0]; i += blockDim_x_1) {
        smem47[i] = 0;
    }
    asm ("bar.sync 1,512;");
    ;
    for (IndexType32 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements44; linearIndex += gridDim.x * blockDim_x_1) {
        const IndexType32 bOffset48 = IndexToOffset<input_t31, IndexType32, BDims35>::get(linearIndex, b40);
        const input_t31 bVal49 = b40.data[bOffset48];
        if (bVal49 >= minvalue42 && bVal49 <= maxvalue43) {
            const IndexType32 bin50 = getBin<input_t31, IndexType32>(bVal49, minvalue42, maxvalue43, nbins41);
            atomicAdd(& smem47[bin50], getOp45(linearIndex));
        }
    }
    asm ("bar.sync 1,512;");
    ;
    for (IndexType32 i = threadIdx_x_1; i < a38.sizes[0]; i += blockDim_x_1) {
        const IndexType32 aOffset51 = IndexToOffset<output_t30, IndexType32, ADims33>::get(i, a38);
        atomicAdd(& a38.data[aOffset51], smem47[i]);
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)){
    unsigned int blockDim_x_0 = 512;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 512;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512;
    const int batchsize8 = idata6.size(0);
    const int channels9 = idata6.size(1);
    const int height110 = idata6.size(2);
    const int width111 = idata6.size(3);
    const int height212 = odata7.size(2);
    const int width213 = odata7.size(3);
    for (int index = blockIdx.x * blockDim_x_0 + threadIdx_x_0; index < (ns2); index += blockDim_x_0 * gridDim.x) {
        const int w214 = index % width213;
        const int h215 = index / width213;
        if (height110 == height212 && width111 == width213) {
            const int h126 = h215;
            const int w127 = w214;
            for (int n = 0; n < batchsize8; n++) {
                for (int c = 0; c < channels9; ++c) {
                    const scalar_t0 val28 = idata6[n][c][h126][w127];
                    odata7[n][c][h215][w214] = val28;
                }
            }
            return;
        }
        const accscalar_t1 h1r16 = area_pixel_compute_source_index<accscalar_t1>(rheight3, h215, align_corners5, false);
        const int h117 = h1r16;
        const int h1p18 = (h117 < height110 - 1) ? 1 : 0;
        const accscalar_t1 h1lambda19 = h1r16 - h117;
        const accscalar_t1 h0lambda20 = static_cast<accscalar_t1>(1) - h1lambda19;
        const accscalar_t1 w1r21 = area_pixel_compute_source_index<accscalar_t1>(rwidth4, w214, align_corners5, false);
        const int w122 = w1r21;
        const int w1p23 = (w122 < width111 - 1) ? 1 : 0;
        const accscalar_t1 w1lambda24 = w1r21 - w122;
        const accscalar_t1 w0lambda25 = static_cast<accscalar_t1>(1) - w1lambda24;
        for (int n = 0; n < batchsize8; n++) {
            for (int c = 0; c < channels9; ++c) {
                const accscalar_t1 val29 = h0lambda20 * (w0lambda25 * idata6[n][c][h117][w122] + w1lambda24 * idata6[n][c][h117][w122 + w1p23]) + h1lambda19 * (w0lambda25 * idata6[n][c][h117 + h1p18][w122] + w1lambda24 * idata6[n][c][h117 + h1p18][w122 + w1p23]);
                odata7[n][c][h215][w214] = static_cast<scalar_t0>(val29);
            }
        }
    }
}
}
template <typename output_t30, typename input_t31, typename IndexType32, int ADims33, int PDims34, int BDims35, at::native::CUDAHistogramMemoryType MemoryType36 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op37, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(1024, 0) void kernelHistogram1D_upsample_bilinear2d_out_frame_fused_kernel_hfuse_idx_0(TensorInfo<output_t30, IndexType32> a38, TensorInfo<output_t30, IndexType32> p39, TensorInfo<input_t31, IndexType32> b40, int nbins41, input_t31 minvalue42, input_t31 maxvalue43, IndexType32 totalElements44, Op37 getOp45, const int ns2, const accscalar_t1 rheight3, const accscalar_t1 rwidth4, const bool align_corners5, const PackedTensorAccessor<scalar_t0, 4> idata6, PackedTensorAccessor<scalar_t0, 4> odata7)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 128)){
    unsigned int blockDim_x_1 = 128;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 128;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128;
    extern unsigned char my_smem46[] __attribute__((shared));
    output_t30 *smem47 = nullptr;
    smem47 = reinterpret_cast<output_t30 *>(my_smem46);
    for (IndexType32 i = threadIdx_x_1; i < a38.sizes[0]; i += blockDim_x_1) {
        smem47[i] = 0;
    }
    asm ("bar.sync 1,128;");
    ;
    for (IndexType32 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements44; linearIndex += gridDim.x * blockDim_x_1) {
        const IndexType32 bOffset48 = IndexToOffset<input_t31, IndexType32, BDims35>::get(linearIndex, b40);
        const input_t31 bVal49 = b40.data[bOffset48];
        if (bVal49 >= minvalue42 && bVal49 <= maxvalue43) {
            const IndexType32 bin50 = getBin<input_t31, IndexType32>(bVal49, minvalue42, maxvalue43, nbins41);
            atomicAdd(& smem47[bin50], getOp45(linearIndex));
        }
    }
    asm ("bar.sync 1,128;");
    ;
    for (IndexType32 i = threadIdx_x_1; i < a38.sizes[0]; i += blockDim_x_1) {
        const IndexType32 aOffset51 = IndexToOffset<output_t30, IndexType32, ADims33>::get(i, a38);
        atomicAdd(& a38.data[aOffset51], smem47[i]);
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=128 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)){
    unsigned int blockDim_x_0 = 896;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) % 896;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) / 896 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) / 896;
    const int batchsize8 = idata6.size(0);
    const int channels9 = idata6.size(1);
    const int height110 = idata6.size(2);
    const int width111 = idata6.size(3);
    const int height212 = odata7.size(2);
    const int width213 = odata7.size(3);
    for (int index = blockIdx.x * blockDim_x_0 + threadIdx_x_0; index < (ns2); index += blockDim_x_0 * gridDim.x) {
        const int w214 = index % width213;
        const int h215 = index / width213;
        if (height110 == height212 && width111 == width213) {
            const int h126 = h215;
            const int w127 = w214;
            for (int n = 0; n < batchsize8; n++) {
                for (int c = 0; c < channels9; ++c) {
                    const scalar_t0 val28 = idata6[n][c][h126][w127];
                    odata7[n][c][h215][w214] = val28;
                }
            }
            return;
        }
        const accscalar_t1 h1r16 = area_pixel_compute_source_index<accscalar_t1>(rheight3, h215, align_corners5, false);
        const int h117 = h1r16;
        const int h1p18 = (h117 < height110 - 1) ? 1 : 0;
        const accscalar_t1 h1lambda19 = h1r16 - h117;
        const accscalar_t1 h0lambda20 = static_cast<accscalar_t1>(1) - h1lambda19;
        const accscalar_t1 w1r21 = area_pixel_compute_source_index<accscalar_t1>(rwidth4, w214, align_corners5, false);
        const int w122 = w1r21;
        const int w1p23 = (w122 < width111 - 1) ? 1 : 0;
        const accscalar_t1 w1lambda24 = w1r21 - w122;
        const accscalar_t1 w0lambda25 = static_cast<accscalar_t1>(1) - w1lambda24;
        for (int n = 0; n < batchsize8; n++) {
            for (int c = 0; c < channels9; ++c) {
                const accscalar_t1 val29 = h0lambda20 * (w0lambda25 * idata6[n][c][h117][w122] + w1lambda24 * idata6[n][c][h117][w122 + w1p23]) + h1lambda19 * (w0lambda25 * idata6[n][c][h117 + h1p18][w122] + w1lambda24 * idata6[n][c][h117 + h1p18][w122 + w1p23]);
                odata7[n][c][h215][w214] = static_cast<scalar_t0>(val29);
            }
        }
    }
}
}
template <typename output_t30, typename input_t31, typename IndexType32, int ADims33, int PDims34, int BDims35, at::native::CUDAHistogramMemoryType MemoryType36 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op37, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(1024, 0) void kernelHistogram1D_upsample_bilinear2d_out_frame_fused_kernel_hfuse_idx_1(TensorInfo<output_t30, IndexType32> a38, TensorInfo<output_t30, IndexType32> p39, TensorInfo<input_t31, IndexType32> b40, int nbins41, input_t31 minvalue42, input_t31 maxvalue43, IndexType32 totalElements44, Op37 getOp45, const int ns2, const accscalar_t1 rheight3, const accscalar_t1 rwidth4, const bool align_corners5, const PackedTensorAccessor<scalar_t0, 4> idata6, PackedTensorAccessor<scalar_t0, 4> odata7)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 256)){
    unsigned int blockDim_x_1 = 256;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 256;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256;
    extern unsigned char my_smem46[] __attribute__((shared));
    output_t30 *smem47 = nullptr;
    smem47 = reinterpret_cast<output_t30 *>(my_smem46);
    for (IndexType32 i = threadIdx_x_1; i < a38.sizes[0]; i += blockDim_x_1) {
        smem47[i] = 0;
    }
    asm ("bar.sync 1,256;");
    ;
    for (IndexType32 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements44; linearIndex += gridDim.x * blockDim_x_1) {
        const IndexType32 bOffset48 = IndexToOffset<input_t31, IndexType32, BDims35>::get(linearIndex, b40);
        const input_t31 bVal49 = b40.data[bOffset48];
        if (bVal49 >= minvalue42 && bVal49 <= maxvalue43) {
            const IndexType32 bin50 = getBin<input_t31, IndexType32>(bVal49, minvalue42, maxvalue43, nbins41);
            atomicAdd(& smem47[bin50], getOp45(linearIndex));
        }
    }
    asm ("bar.sync 1,256;");
    ;
    for (IndexType32 i = threadIdx_x_1; i < a38.sizes[0]; i += blockDim_x_1) {
        const IndexType32 aOffset51 = IndexToOffset<output_t30, IndexType32, ADims33>::get(i, a38);
        atomicAdd(& a38.data[aOffset51], smem47[i]);
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=256 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)){
    unsigned int blockDim_x_0 = 768;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 256) % 768;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 256) / 768 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 256) / 768;
    const int batchsize8 = idata6.size(0);
    const int channels9 = idata6.size(1);
    const int height110 = idata6.size(2);
    const int width111 = idata6.size(3);
    const int height212 = odata7.size(2);
    const int width213 = odata7.size(3);
    for (int index = blockIdx.x * blockDim_x_0 + threadIdx_x_0; index < (ns2); index += blockDim_x_0 * gridDim.x) {
        const int w214 = index % width213;
        const int h215 = index / width213;
        if (height110 == height212 && width111 == width213) {
            const int h126 = h215;
            const int w127 = w214;
            for (int n = 0; n < batchsize8; n++) {
                for (int c = 0; c < channels9; ++c) {
                    const scalar_t0 val28 = idata6[n][c][h126][w127];
                    odata7[n][c][h215][w214] = val28;
                }
            }
            return;
        }
        const accscalar_t1 h1r16 = area_pixel_compute_source_index<accscalar_t1>(rheight3, h215, align_corners5, false);
        const int h117 = h1r16;
        const int h1p18 = (h117 < height110 - 1) ? 1 : 0;
        const accscalar_t1 h1lambda19 = h1r16 - h117;
        const accscalar_t1 h0lambda20 = static_cast<accscalar_t1>(1) - h1lambda19;
        const accscalar_t1 w1r21 = area_pixel_compute_source_index<accscalar_t1>(rwidth4, w214, align_corners5, false);
        const int w122 = w1r21;
        const int w1p23 = (w122 < width111 - 1) ? 1 : 0;
        const accscalar_t1 w1lambda24 = w1r21 - w122;
        const accscalar_t1 w0lambda25 = static_cast<accscalar_t1>(1) - w1lambda24;
        for (int n = 0; n < batchsize8; n++) {
            for (int c = 0; c < channels9; ++c) {
                const accscalar_t1 val29 = h0lambda20 * (w0lambda25 * idata6[n][c][h117][w122] + w1lambda24 * idata6[n][c][h117][w122 + w1p23]) + h1lambda19 * (w0lambda25 * idata6[n][c][h117 + h1p18][w122] + w1lambda24 * idata6[n][c][h117 + h1p18][w122 + w1p23]);
                odata7[n][c][h215][w214] = static_cast<scalar_t0>(val29);
            }
        }
    }
}
}
template <typename output_t30, typename input_t31, typename IndexType32, int ADims33, int PDims34, int BDims35, at::native::CUDAHistogramMemoryType MemoryType36 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op37, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(1024, 0) void kernelHistogram1D_upsample_bilinear2d_out_frame_fused_kernel_hfuse_idx_2(TensorInfo<output_t30, IndexType32> a38, TensorInfo<output_t30, IndexType32> p39, TensorInfo<input_t31, IndexType32> b40, int nbins41, input_t31 minvalue42, input_t31 maxvalue43, IndexType32 totalElements44, Op37 getOp45, const int ns2, const accscalar_t1 rheight3, const accscalar_t1 rwidth4, const bool align_corners5, const PackedTensorAccessor<scalar_t0, 4> idata6, PackedTensorAccessor<scalar_t0, 4> odata7)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 384)){
    unsigned int blockDim_x_1 = 384;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 384;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 384 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 384;
    extern unsigned char my_smem46[] __attribute__((shared));
    output_t30 *smem47 = nullptr;
    smem47 = reinterpret_cast<output_t30 *>(my_smem46);
    for (IndexType32 i = threadIdx_x_1; i < a38.sizes[0]; i += blockDim_x_1) {
        smem47[i] = 0;
    }
    asm ("bar.sync 1,384;");
    ;
    for (IndexType32 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements44; linearIndex += gridDim.x * blockDim_x_1) {
        const IndexType32 bOffset48 = IndexToOffset<input_t31, IndexType32, BDims35>::get(linearIndex, b40);
        const input_t31 bVal49 = b40.data[bOffset48];
        if (bVal49 >= minvalue42 && bVal49 <= maxvalue43) {
            const IndexType32 bin50 = getBin<input_t31, IndexType32>(bVal49, minvalue42, maxvalue43, nbins41);
            atomicAdd(& smem47[bin50], getOp45(linearIndex));
        }
    }
    asm ("bar.sync 1,384;");
    ;
    for (IndexType32 i = threadIdx_x_1; i < a38.sizes[0]; i += blockDim_x_1) {
        const IndexType32 aOffset51 = IndexToOffset<output_t30, IndexType32, ADims33>::get(i, a38);
        atomicAdd(& a38.data[aOffset51], smem47[i]);
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=384 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)){
    unsigned int blockDim_x_0 = 640;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 384) % 640;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 384) / 640 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 384) / 640;
    const int batchsize8 = idata6.size(0);
    const int channels9 = idata6.size(1);
    const int height110 = idata6.size(2);
    const int width111 = idata6.size(3);
    const int height212 = odata7.size(2);
    const int width213 = odata7.size(3);
    for (int index = blockIdx.x * blockDim_x_0 + threadIdx_x_0; index < (ns2); index += blockDim_x_0 * gridDim.x) {
        const int w214 = index % width213;
        const int h215 = index / width213;
        if (height110 == height212 && width111 == width213) {
            const int h126 = h215;
            const int w127 = w214;
            for (int n = 0; n < batchsize8; n++) {
                for (int c = 0; c < channels9; ++c) {
                    const scalar_t0 val28 = idata6[n][c][h126][w127];
                    odata7[n][c][h215][w214] = val28;
                }
            }
            return;
        }
        const accscalar_t1 h1r16 = area_pixel_compute_source_index<accscalar_t1>(rheight3, h215, align_corners5, false);
        const int h117 = h1r16;
        const int h1p18 = (h117 < height110 - 1) ? 1 : 0;
        const accscalar_t1 h1lambda19 = h1r16 - h117;
        const accscalar_t1 h0lambda20 = static_cast<accscalar_t1>(1) - h1lambda19;
        const accscalar_t1 w1r21 = area_pixel_compute_source_index<accscalar_t1>(rwidth4, w214, align_corners5, false);
        const int w122 = w1r21;
        const int w1p23 = (w122 < width111 - 1) ? 1 : 0;
        const accscalar_t1 w1lambda24 = w1r21 - w122;
        const accscalar_t1 w0lambda25 = static_cast<accscalar_t1>(1) - w1lambda24;
        for (int n = 0; n < batchsize8; n++) {
            for (int c = 0; c < channels9; ++c) {
                const accscalar_t1 val29 = h0lambda20 * (w0lambda25 * idata6[n][c][h117][w122] + w1lambda24 * idata6[n][c][h117][w122 + w1p23]) + h1lambda19 * (w0lambda25 * idata6[n][c][h117 + h1p18][w122] + w1lambda24 * idata6[n][c][h117 + h1p18][w122 + w1p23]);
                odata7[n][c][h215][w214] = static_cast<scalar_t0>(val29);
            }
        }
    }
}
}
template <typename output_t30, typename input_t31, typename IndexType32, int ADims33, int PDims34, int BDims35, at::native::CUDAHistogramMemoryType MemoryType36 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op37, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(1024, 0) void kernelHistogram1D_upsample_bilinear2d_out_frame_fused_kernel_hfuse_idx_3(TensorInfo<output_t30, IndexType32> a38, TensorInfo<output_t30, IndexType32> p39, TensorInfo<input_t31, IndexType32> b40, int nbins41, input_t31 minvalue42, input_t31 maxvalue43, IndexType32 totalElements44, Op37 getOp45, const int ns2, const accscalar_t1 rheight3, const accscalar_t1 rwidth4, const bool align_corners5, const PackedTensorAccessor<scalar_t0, 4> idata6, PackedTensorAccessor<scalar_t0, 4> odata7)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)){
    unsigned int blockDim_x_1 = 512;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 512;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512;
    extern unsigned char my_smem46[] __attribute__((shared));
    output_t30 *smem47 = nullptr;
    smem47 = reinterpret_cast<output_t30 *>(my_smem46);
    for (IndexType32 i = threadIdx_x_1; i < a38.sizes[0]; i += blockDim_x_1) {
        smem47[i] = 0;
    }
    asm ("bar.sync 1,512;");
    ;
    for (IndexType32 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements44; linearIndex += gridDim.x * blockDim_x_1) {
        const IndexType32 bOffset48 = IndexToOffset<input_t31, IndexType32, BDims35>::get(linearIndex, b40);
        const input_t31 bVal49 = b40.data[bOffset48];
        if (bVal49 >= minvalue42 && bVal49 <= maxvalue43) {
            const IndexType32 bin50 = getBin<input_t31, IndexType32>(bVal49, minvalue42, maxvalue43, nbins41);
            atomicAdd(& smem47[bin50], getOp45(linearIndex));
        }
    }
    asm ("bar.sync 1,512;");
    ;
    for (IndexType32 i = threadIdx_x_1; i < a38.sizes[0]; i += blockDim_x_1) {
        const IndexType32 aOffset51 = IndexToOffset<output_t30, IndexType32, ADims33>::get(i, a38);
        atomicAdd(& a38.data[aOffset51], smem47[i]);
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)){
    unsigned int blockDim_x_0 = 512;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) % 512;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 512 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 512;
    const int batchsize8 = idata6.size(0);
    const int channels9 = idata6.size(1);
    const int height110 = idata6.size(2);
    const int width111 = idata6.size(3);
    const int height212 = odata7.size(2);
    const int width213 = odata7.size(3);
    for (int index = blockIdx.x * blockDim_x_0 + threadIdx_x_0; index < (ns2); index += blockDim_x_0 * gridDim.x) {
        const int w214 = index % width213;
        const int h215 = index / width213;
        if (height110 == height212 && width111 == width213) {
            const int h126 = h215;
            const int w127 = w214;
            for (int n = 0; n < batchsize8; n++) {
                for (int c = 0; c < channels9; ++c) {
                    const scalar_t0 val28 = idata6[n][c][h126][w127];
                    odata7[n][c][h215][w214] = val28;
                }
            }
            return;
        }
        const accscalar_t1 h1r16 = area_pixel_compute_source_index<accscalar_t1>(rheight3, h215, align_corners5, false);
        const int h117 = h1r16;
        const int h1p18 = (h117 < height110 - 1) ? 1 : 0;
        const accscalar_t1 h1lambda19 = h1r16 - h117;
        const accscalar_t1 h0lambda20 = static_cast<accscalar_t1>(1) - h1lambda19;
        const accscalar_t1 w1r21 = area_pixel_compute_source_index<accscalar_t1>(rwidth4, w214, align_corners5, false);
        const int w122 = w1r21;
        const int w1p23 = (w122 < width111 - 1) ? 1 : 0;
        const accscalar_t1 w1lambda24 = w1r21 - w122;
        const accscalar_t1 w0lambda25 = static_cast<accscalar_t1>(1) - w1lambda24;
        for (int n = 0; n < batchsize8; n++) {
            for (int c = 0; c < channels9; ++c) {
                const accscalar_t1 val29 = h0lambda20 * (w0lambda25 * idata6[n][c][h117][w122] + w1lambda24 * idata6[n][c][h117][w122 + w1p23]) + h1lambda19 * (w0lambda25 * idata6[n][c][h117 + h1p18][w122] + w1lambda24 * idata6[n][c][h117 + h1p18][w122 + w1p23]);
                odata7[n][c][h215][w214] = static_cast<scalar_t0>(val29);
            }
        }
    }
}
}
template <typename output_t30, typename input_t31, typename IndexType32, int ADims33, int PDims34, int BDims35, at::native::CUDAHistogramMemoryType MemoryType36 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op37, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(1024, 0) void kernelHistogram1D_upsample_bilinear2d_out_frame_fused_kernel_hfuse_idx_4(TensorInfo<output_t30, IndexType32> a38, TensorInfo<output_t30, IndexType32> p39, TensorInfo<input_t31, IndexType32> b40, int nbins41, input_t31 minvalue42, input_t31 maxvalue43, IndexType32 totalElements44, Op37 getOp45, const int ns2, const accscalar_t1 rheight3, const accscalar_t1 rwidth4, const bool align_corners5, const PackedTensorAccessor<scalar_t0, 4> idata6, PackedTensorAccessor<scalar_t0, 4> odata7)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 640)){
    unsigned int blockDim_x_1 = 640;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 640;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 640 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 640;
    extern unsigned char my_smem46[] __attribute__((shared));
    output_t30 *smem47 = nullptr;
    smem47 = reinterpret_cast<output_t30 *>(my_smem46);
    for (IndexType32 i = threadIdx_x_1; i < a38.sizes[0]; i += blockDim_x_1) {
        smem47[i] = 0;
    }
    asm ("bar.sync 1,640;");
    ;
    for (IndexType32 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements44; linearIndex += gridDim.x * blockDim_x_1) {
        const IndexType32 bOffset48 = IndexToOffset<input_t31, IndexType32, BDims35>::get(linearIndex, b40);
        const input_t31 bVal49 = b40.data[bOffset48];
        if (bVal49 >= minvalue42 && bVal49 <= maxvalue43) {
            const IndexType32 bin50 = getBin<input_t31, IndexType32>(bVal49, minvalue42, maxvalue43, nbins41);
            atomicAdd(& smem47[bin50], getOp45(linearIndex));
        }
    }
    asm ("bar.sync 1,640;");
    ;
    for (IndexType32 i = threadIdx_x_1; i < a38.sizes[0]; i += blockDim_x_1) {
        const IndexType32 aOffset51 = IndexToOffset<output_t30, IndexType32, ADims33>::get(i, a38);
        atomicAdd(& a38.data[aOffset51], smem47[i]);
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=640 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)){
    unsigned int blockDim_x_0 = 384;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 640) % 384;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 640) / 384 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 640) / 384;
    const int batchsize8 = idata6.size(0);
    const int channels9 = idata6.size(1);
    const int height110 = idata6.size(2);
    const int width111 = idata6.size(3);
    const int height212 = odata7.size(2);
    const int width213 = odata7.size(3);
    for (int index = blockIdx.x * blockDim_x_0 + threadIdx_x_0; index < (ns2); index += blockDim_x_0 * gridDim.x) {
        const int w214 = index % width213;
        const int h215 = index / width213;
        if (height110 == height212 && width111 == width213) {
            const int h126 = h215;
            const int w127 = w214;
            for (int n = 0; n < batchsize8; n++) {
                for (int c = 0; c < channels9; ++c) {
                    const scalar_t0 val28 = idata6[n][c][h126][w127];
                    odata7[n][c][h215][w214] = val28;
                }
            }
            return;
        }
        const accscalar_t1 h1r16 = area_pixel_compute_source_index<accscalar_t1>(rheight3, h215, align_corners5, false);
        const int h117 = h1r16;
        const int h1p18 = (h117 < height110 - 1) ? 1 : 0;
        const accscalar_t1 h1lambda19 = h1r16 - h117;
        const accscalar_t1 h0lambda20 = static_cast<accscalar_t1>(1) - h1lambda19;
        const accscalar_t1 w1r21 = area_pixel_compute_source_index<accscalar_t1>(rwidth4, w214, align_corners5, false);
        const int w122 = w1r21;
        const int w1p23 = (w122 < width111 - 1) ? 1 : 0;
        const accscalar_t1 w1lambda24 = w1r21 - w122;
        const accscalar_t1 w0lambda25 = static_cast<accscalar_t1>(1) - w1lambda24;
        for (int n = 0; n < batchsize8; n++) {
            for (int c = 0; c < channels9; ++c) {
                const accscalar_t1 val29 = h0lambda20 * (w0lambda25 * idata6[n][c][h117][w122] + w1lambda24 * idata6[n][c][h117][w122 + w1p23]) + h1lambda19 * (w0lambda25 * idata6[n][c][h117 + h1p18][w122] + w1lambda24 * idata6[n][c][h117 + h1p18][w122 + w1p23]);
                odata7[n][c][h215][w214] = static_cast<scalar_t0>(val29);
            }
        }
    }
}
}
template <typename output_t30, typename input_t31, typename IndexType32, int ADims33, int PDims34, int BDims35, at::native::CUDAHistogramMemoryType MemoryType36 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op37, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(1024, 0) void kernelHistogram1D_upsample_bilinear2d_out_frame_fused_kernel_hfuse_idx_5(TensorInfo<output_t30, IndexType32> a38, TensorInfo<output_t30, IndexType32> p39, TensorInfo<input_t31, IndexType32> b40, int nbins41, input_t31 minvalue42, input_t31 maxvalue43, IndexType32 totalElements44, Op37 getOp45, const int ns2, const accscalar_t1 rheight3, const accscalar_t1 rwidth4, const bool align_corners5, const PackedTensorAccessor<scalar_t0, 4> idata6, PackedTensorAccessor<scalar_t0, 4> odata7)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 768)){
    unsigned int blockDim_x_1 = 768;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 768;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 768 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 768;
    extern unsigned char my_smem46[] __attribute__((shared));
    output_t30 *smem47 = nullptr;
    smem47 = reinterpret_cast<output_t30 *>(my_smem46);
    for (IndexType32 i = threadIdx_x_1; i < a38.sizes[0]; i += blockDim_x_1) {
        smem47[i] = 0;
    }
    asm ("bar.sync 1,768;");
    ;
    for (IndexType32 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements44; linearIndex += gridDim.x * blockDim_x_1) {
        const IndexType32 bOffset48 = IndexToOffset<input_t31, IndexType32, BDims35>::get(linearIndex, b40);
        const input_t31 bVal49 = b40.data[bOffset48];
        if (bVal49 >= minvalue42 && bVal49 <= maxvalue43) {
            const IndexType32 bin50 = getBin<input_t31, IndexType32>(bVal49, minvalue42, maxvalue43, nbins41);
            atomicAdd(& smem47[bin50], getOp45(linearIndex));
        }
    }
    asm ("bar.sync 1,768;");
    ;
    for (IndexType32 i = threadIdx_x_1; i < a38.sizes[0]; i += blockDim_x_1) {
        const IndexType32 aOffset51 = IndexToOffset<output_t30, IndexType32, ADims33>::get(i, a38);
        atomicAdd(& a38.data[aOffset51], smem47[i]);
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=768 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)){
    unsigned int blockDim_x_0 = 256;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 768) % 256;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 768) / 256 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 768) / 256;
    const int batchsize8 = idata6.size(0);
    const int channels9 = idata6.size(1);
    const int height110 = idata6.size(2);
    const int width111 = idata6.size(3);
    const int height212 = odata7.size(2);
    const int width213 = odata7.size(3);
    for (int index = blockIdx.x * blockDim_x_0 + threadIdx_x_0; index < (ns2); index += blockDim_x_0 * gridDim.x) {
        const int w214 = index % width213;
        const int h215 = index / width213;
        if (height110 == height212 && width111 == width213) {
            const int h126 = h215;
            const int w127 = w214;
            for (int n = 0; n < batchsize8; n++) {
                for (int c = 0; c < channels9; ++c) {
                    const scalar_t0 val28 = idata6[n][c][h126][w127];
                    odata7[n][c][h215][w214] = val28;
                }
            }
            return;
        }
        const accscalar_t1 h1r16 = area_pixel_compute_source_index<accscalar_t1>(rheight3, h215, align_corners5, false);
        const int h117 = h1r16;
        const int h1p18 = (h117 < height110 - 1) ? 1 : 0;
        const accscalar_t1 h1lambda19 = h1r16 - h117;
        const accscalar_t1 h0lambda20 = static_cast<accscalar_t1>(1) - h1lambda19;
        const accscalar_t1 w1r21 = area_pixel_compute_source_index<accscalar_t1>(rwidth4, w214, align_corners5, false);
        const int w122 = w1r21;
        const int w1p23 = (w122 < width111 - 1) ? 1 : 0;
        const accscalar_t1 w1lambda24 = w1r21 - w122;
        const accscalar_t1 w0lambda25 = static_cast<accscalar_t1>(1) - w1lambda24;
        for (int n = 0; n < batchsize8; n++) {
            for (int c = 0; c < channels9; ++c) {
                const accscalar_t1 val29 = h0lambda20 * (w0lambda25 * idata6[n][c][h117][w122] + w1lambda24 * idata6[n][c][h117][w122 + w1p23]) + h1lambda19 * (w0lambda25 * idata6[n][c][h117 + h1p18][w122] + w1lambda24 * idata6[n][c][h117 + h1p18][w122 + w1p23]);
                odata7[n][c][h215][w214] = static_cast<scalar_t0>(val29);
            }
        }
    }
}
}
template <typename output_t30, typename input_t31, typename IndexType32, int ADims33, int PDims34, int BDims35, at::native::CUDAHistogramMemoryType MemoryType36 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op37, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(1024, 0) void kernelHistogram1D_upsample_bilinear2d_out_frame_fused_kernel_hfuse_idx_6(TensorInfo<output_t30, IndexType32> a38, TensorInfo<output_t30, IndexType32> p39, TensorInfo<input_t31, IndexType32> b40, int nbins41, input_t31 minvalue42, input_t31 maxvalue43, IndexType32 totalElements44, Op37 getOp45, const int ns2, const accscalar_t1 rheight3, const accscalar_t1 rwidth4, const bool align_corners5, const PackedTensorAccessor<scalar_t0, 4> idata6, PackedTensorAccessor<scalar_t0, 4> odata7)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 896)){
    unsigned int blockDim_x_1 = 896;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 896;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 896 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 896;
    extern unsigned char my_smem46[] __attribute__((shared));
    output_t30 *smem47 = nullptr;
    smem47 = reinterpret_cast<output_t30 *>(my_smem46);
    for (IndexType32 i = threadIdx_x_1; i < a38.sizes[0]; i += blockDim_x_1) {
        smem47[i] = 0;
    }
    asm ("bar.sync 1,896;");
    ;
    for (IndexType32 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements44; linearIndex += gridDim.x * blockDim_x_1) {
        const IndexType32 bOffset48 = IndexToOffset<input_t31, IndexType32, BDims35>::get(linearIndex, b40);
        const input_t31 bVal49 = b40.data[bOffset48];
        if (bVal49 >= minvalue42 && bVal49 <= maxvalue43) {
            const IndexType32 bin50 = getBin<input_t31, IndexType32>(bVal49, minvalue42, maxvalue43, nbins41);
            atomicAdd(& smem47[bin50], getOp45(linearIndex));
        }
    }
    asm ("bar.sync 1,896;");
    ;
    for (IndexType32 i = threadIdx_x_1; i < a38.sizes[0]; i += blockDim_x_1) {
        const IndexType32 aOffset51 = IndexToOffset<output_t30, IndexType32, ADims33>::get(i, a38);
        atomicAdd(& a38.data[aOffset51], smem47[i]);
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=896 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)){
    unsigned int blockDim_x_0 = 128;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 896) % 128;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 896) / 128 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 896) / 128;
    const int batchsize8 = idata6.size(0);
    const int channels9 = idata6.size(1);
    const int height110 = idata6.size(2);
    const int width111 = idata6.size(3);
    const int height212 = odata7.size(2);
    const int width213 = odata7.size(3);
    for (int index = blockIdx.x * blockDim_x_0 + threadIdx_x_0; index < (ns2); index += blockDim_x_0 * gridDim.x) {
        const int w214 = index % width213;
        const int h215 = index / width213;
        if (height110 == height212 && width111 == width213) {
            const int h126 = h215;
            const int w127 = w214;
            for (int n = 0; n < batchsize8; n++) {
                for (int c = 0; c < channels9; ++c) {
                    const scalar_t0 val28 = idata6[n][c][h126][w127];
                    odata7[n][c][h215][w214] = val28;
                }
            }
            return;
        }
        const accscalar_t1 h1r16 = area_pixel_compute_source_index<accscalar_t1>(rheight3, h215, align_corners5, false);
        const int h117 = h1r16;
        const int h1p18 = (h117 < height110 - 1) ? 1 : 0;
        const accscalar_t1 h1lambda19 = h1r16 - h117;
        const accscalar_t1 h0lambda20 = static_cast<accscalar_t1>(1) - h1lambda19;
        const accscalar_t1 w1r21 = area_pixel_compute_source_index<accscalar_t1>(rwidth4, w214, align_corners5, false);
        const int w122 = w1r21;
        const int w1p23 = (w122 < width111 - 1) ? 1 : 0;
        const accscalar_t1 w1lambda24 = w1r21 - w122;
        const accscalar_t1 w0lambda25 = static_cast<accscalar_t1>(1) - w1lambda24;
        for (int n = 0; n < batchsize8; n++) {
            for (int c = 0; c < channels9; ++c) {
                const accscalar_t1 val29 = h0lambda20 * (w0lambda25 * idata6[n][c][h117][w122] + w1lambda24 * idata6[n][c][h117][w122 + w1p23]) + h1lambda19 * (w0lambda25 * idata6[n][c][h117 + h1p18][w122] + w1lambda24 * idata6[n][c][h117 + h1p18][w122 + w1p23]);
                odata7[n][c][h215][w214] = static_cast<scalar_t0>(val29);
            }
        }
    }
}
}
template <typename output_t30, typename input_t31, typename IndexType32, int ADims33, int PDims34, int BDims35, at::native::CUDAHistogramMemoryType MemoryType36 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op37, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(1024, 1) void kernelHistogram1D_upsample_bilinear2d_out_frame_fused_kernel_hfuse_lb_idx_0(TensorInfo<output_t30, IndexType32> a38, TensorInfo<output_t30, IndexType32> p39, TensorInfo<input_t31, IndexType32> b40, int nbins41, input_t31 minvalue42, input_t31 maxvalue43, IndexType32 totalElements44, Op37 getOp45, const int ns2, const accscalar_t1 rheight3, const accscalar_t1 rwidth4, const bool align_corners5, const PackedTensorAccessor<scalar_t0, 4> idata6, PackedTensorAccessor<scalar_t0, 4> odata7)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 128)){
    unsigned int blockDim_x_1 = 128;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 128;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128;
    extern unsigned char my_smem46[] __attribute__((shared));
    output_t30 *smem47 = nullptr;
    smem47 = reinterpret_cast<output_t30 *>(my_smem46);
    for (IndexType32 i = threadIdx_x_1; i < a38.sizes[0]; i += blockDim_x_1) {
        smem47[i] = 0;
    }
    asm ("bar.sync 1,128;");
    ;
    for (IndexType32 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements44; linearIndex += gridDim.x * blockDim_x_1) {
        const IndexType32 bOffset48 = IndexToOffset<input_t31, IndexType32, BDims35>::get(linearIndex, b40);
        const input_t31 bVal49 = b40.data[bOffset48];
        if (bVal49 >= minvalue42 && bVal49 <= maxvalue43) {
            const IndexType32 bin50 = getBin<input_t31, IndexType32>(bVal49, minvalue42, maxvalue43, nbins41);
            atomicAdd(& smem47[bin50], getOp45(linearIndex));
        }
    }
    asm ("bar.sync 1,128;");
    ;
    for (IndexType32 i = threadIdx_x_1; i < a38.sizes[0]; i += blockDim_x_1) {
        const IndexType32 aOffset51 = IndexToOffset<output_t30, IndexType32, ADims33>::get(i, a38);
        atomicAdd(& a38.data[aOffset51], smem47[i]);
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=128 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)){
    unsigned int blockDim_x_0 = 896;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) % 896;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) / 896 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) / 896;
    const int batchsize8 = idata6.size(0);
    const int channels9 = idata6.size(1);
    const int height110 = idata6.size(2);
    const int width111 = idata6.size(3);
    const int height212 = odata7.size(2);
    const int width213 = odata7.size(3);
    for (int index = blockIdx.x * blockDim_x_0 + threadIdx_x_0; index < (ns2); index += blockDim_x_0 * gridDim.x) {
        const int w214 = index % width213;
        const int h215 = index / width213;
        if (height110 == height212 && width111 == width213) {
            const int h126 = h215;
            const int w127 = w214;
            for (int n = 0; n < batchsize8; n++) {
                for (int c = 0; c < channels9; ++c) {
                    const scalar_t0 val28 = idata6[n][c][h126][w127];
                    odata7[n][c][h215][w214] = val28;
                }
            }
            return;
        }
        const accscalar_t1 h1r16 = area_pixel_compute_source_index<accscalar_t1>(rheight3, h215, align_corners5, false);
        const int h117 = h1r16;
        const int h1p18 = (h117 < height110 - 1) ? 1 : 0;
        const accscalar_t1 h1lambda19 = h1r16 - h117;
        const accscalar_t1 h0lambda20 = static_cast<accscalar_t1>(1) - h1lambda19;
        const accscalar_t1 w1r21 = area_pixel_compute_source_index<accscalar_t1>(rwidth4, w214, align_corners5, false);
        const int w122 = w1r21;
        const int w1p23 = (w122 < width111 - 1) ? 1 : 0;
        const accscalar_t1 w1lambda24 = w1r21 - w122;
        const accscalar_t1 w0lambda25 = static_cast<accscalar_t1>(1) - w1lambda24;
        for (int n = 0; n < batchsize8; n++) {
            for (int c = 0; c < channels9; ++c) {
                const accscalar_t1 val29 = h0lambda20 * (w0lambda25 * idata6[n][c][h117][w122] + w1lambda24 * idata6[n][c][h117][w122 + w1p23]) + h1lambda19 * (w0lambda25 * idata6[n][c][h117 + h1p18][w122] + w1lambda24 * idata6[n][c][h117 + h1p18][w122 + w1p23]);
                odata7[n][c][h215][w214] = static_cast<scalar_t0>(val29);
            }
        }
    }
}
}
template <typename output_t30, typename input_t31, typename IndexType32, int ADims33, int PDims34, int BDims35, at::native::CUDAHistogramMemoryType MemoryType36 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op37, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(1024, 1) void kernelHistogram1D_upsample_bilinear2d_out_frame_fused_kernel_hfuse_lb_idx_1(TensorInfo<output_t30, IndexType32> a38, TensorInfo<output_t30, IndexType32> p39, TensorInfo<input_t31, IndexType32> b40, int nbins41, input_t31 minvalue42, input_t31 maxvalue43, IndexType32 totalElements44, Op37 getOp45, const int ns2, const accscalar_t1 rheight3, const accscalar_t1 rwidth4, const bool align_corners5, const PackedTensorAccessor<scalar_t0, 4> idata6, PackedTensorAccessor<scalar_t0, 4> odata7)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 256)){
    unsigned int blockDim_x_1 = 256;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 256;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256;
    extern unsigned char my_smem46[] __attribute__((shared));
    output_t30 *smem47 = nullptr;
    smem47 = reinterpret_cast<output_t30 *>(my_smem46);
    for (IndexType32 i = threadIdx_x_1; i < a38.sizes[0]; i += blockDim_x_1) {
        smem47[i] = 0;
    }
    asm ("bar.sync 1,256;");
    ;
    for (IndexType32 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements44; linearIndex += gridDim.x * blockDim_x_1) {
        const IndexType32 bOffset48 = IndexToOffset<input_t31, IndexType32, BDims35>::get(linearIndex, b40);
        const input_t31 bVal49 = b40.data[bOffset48];
        if (bVal49 >= minvalue42 && bVal49 <= maxvalue43) {
            const IndexType32 bin50 = getBin<input_t31, IndexType32>(bVal49, minvalue42, maxvalue43, nbins41);
            atomicAdd(& smem47[bin50], getOp45(linearIndex));
        }
    }
    asm ("bar.sync 1,256;");
    ;
    for (IndexType32 i = threadIdx_x_1; i < a38.sizes[0]; i += blockDim_x_1) {
        const IndexType32 aOffset51 = IndexToOffset<output_t30, IndexType32, ADims33>::get(i, a38);
        atomicAdd(& a38.data[aOffset51], smem47[i]);
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=256 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)){
    unsigned int blockDim_x_0 = 768;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 256) % 768;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 256) / 768 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 256) / 768;
    const int batchsize8 = idata6.size(0);
    const int channels9 = idata6.size(1);
    const int height110 = idata6.size(2);
    const int width111 = idata6.size(3);
    const int height212 = odata7.size(2);
    const int width213 = odata7.size(3);
    for (int index = blockIdx.x * blockDim_x_0 + threadIdx_x_0; index < (ns2); index += blockDim_x_0 * gridDim.x) {
        const int w214 = index % width213;
        const int h215 = index / width213;
        if (height110 == height212 && width111 == width213) {
            const int h126 = h215;
            const int w127 = w214;
            for (int n = 0; n < batchsize8; n++) {
                for (int c = 0; c < channels9; ++c) {
                    const scalar_t0 val28 = idata6[n][c][h126][w127];
                    odata7[n][c][h215][w214] = val28;
                }
            }
            return;
        }
        const accscalar_t1 h1r16 = area_pixel_compute_source_index<accscalar_t1>(rheight3, h215, align_corners5, false);
        const int h117 = h1r16;
        const int h1p18 = (h117 < height110 - 1) ? 1 : 0;
        const accscalar_t1 h1lambda19 = h1r16 - h117;
        const accscalar_t1 h0lambda20 = static_cast<accscalar_t1>(1) - h1lambda19;
        const accscalar_t1 w1r21 = area_pixel_compute_source_index<accscalar_t1>(rwidth4, w214, align_corners5, false);
        const int w122 = w1r21;
        const int w1p23 = (w122 < width111 - 1) ? 1 : 0;
        const accscalar_t1 w1lambda24 = w1r21 - w122;
        const accscalar_t1 w0lambda25 = static_cast<accscalar_t1>(1) - w1lambda24;
        for (int n = 0; n < batchsize8; n++) {
            for (int c = 0; c < channels9; ++c) {
                const accscalar_t1 val29 = h0lambda20 * (w0lambda25 * idata6[n][c][h117][w122] + w1lambda24 * idata6[n][c][h117][w122 + w1p23]) + h1lambda19 * (w0lambda25 * idata6[n][c][h117 + h1p18][w122] + w1lambda24 * idata6[n][c][h117 + h1p18][w122 + w1p23]);
                odata7[n][c][h215][w214] = static_cast<scalar_t0>(val29);
            }
        }
    }
}
}
template <typename output_t30, typename input_t31, typename IndexType32, int ADims33, int PDims34, int BDims35, at::native::CUDAHistogramMemoryType MemoryType36 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op37, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(1024, 2) void kernelHistogram1D_upsample_bilinear2d_out_frame_fused_kernel_hfuse_lb_idx_2(TensorInfo<output_t30, IndexType32> a38, TensorInfo<output_t30, IndexType32> p39, TensorInfo<input_t31, IndexType32> b40, int nbins41, input_t31 minvalue42, input_t31 maxvalue43, IndexType32 totalElements44, Op37 getOp45, const int ns2, const accscalar_t1 rheight3, const accscalar_t1 rwidth4, const bool align_corners5, const PackedTensorAccessor<scalar_t0, 4> idata6, PackedTensorAccessor<scalar_t0, 4> odata7)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 384)){
    unsigned int blockDim_x_1 = 384;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 384;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 384 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 384;
    extern unsigned char my_smem46[] __attribute__((shared));
    output_t30 *smem47 = nullptr;
    smem47 = reinterpret_cast<output_t30 *>(my_smem46);
    for (IndexType32 i = threadIdx_x_1; i < a38.sizes[0]; i += blockDim_x_1) {
        smem47[i] = 0;
    }
    asm ("bar.sync 1,384;");
    ;
    for (IndexType32 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements44; linearIndex += gridDim.x * blockDim_x_1) {
        const IndexType32 bOffset48 = IndexToOffset<input_t31, IndexType32, BDims35>::get(linearIndex, b40);
        const input_t31 bVal49 = b40.data[bOffset48];
        if (bVal49 >= minvalue42 && bVal49 <= maxvalue43) {
            const IndexType32 bin50 = getBin<input_t31, IndexType32>(bVal49, minvalue42, maxvalue43, nbins41);
            atomicAdd(& smem47[bin50], getOp45(linearIndex));
        }
    }
    asm ("bar.sync 1,384;");
    ;
    for (IndexType32 i = threadIdx_x_1; i < a38.sizes[0]; i += blockDim_x_1) {
        const IndexType32 aOffset51 = IndexToOffset<output_t30, IndexType32, ADims33>::get(i, a38);
        atomicAdd(& a38.data[aOffset51], smem47[i]);
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=384 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)){
    unsigned int blockDim_x_0 = 640;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 384) % 640;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 384) / 640 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 384) / 640;
    const int batchsize8 = idata6.size(0);
    const int channels9 = idata6.size(1);
    const int height110 = idata6.size(2);
    const int width111 = idata6.size(3);
    const int height212 = odata7.size(2);
    const int width213 = odata7.size(3);
    for (int index = blockIdx.x * blockDim_x_0 + threadIdx_x_0; index < (ns2); index += blockDim_x_0 * gridDim.x) {
        const int w214 = index % width213;
        const int h215 = index / width213;
        if (height110 == height212 && width111 == width213) {
            const int h126 = h215;
            const int w127 = w214;
            for (int n = 0; n < batchsize8; n++) {
                for (int c = 0; c < channels9; ++c) {
                    const scalar_t0 val28 = idata6[n][c][h126][w127];
                    odata7[n][c][h215][w214] = val28;
                }
            }
            return;
        }
        const accscalar_t1 h1r16 = area_pixel_compute_source_index<accscalar_t1>(rheight3, h215, align_corners5, false);
        const int h117 = h1r16;
        const int h1p18 = (h117 < height110 - 1) ? 1 : 0;
        const accscalar_t1 h1lambda19 = h1r16 - h117;
        const accscalar_t1 h0lambda20 = static_cast<accscalar_t1>(1) - h1lambda19;
        const accscalar_t1 w1r21 = area_pixel_compute_source_index<accscalar_t1>(rwidth4, w214, align_corners5, false);
        const int w122 = w1r21;
        const int w1p23 = (w122 < width111 - 1) ? 1 : 0;
        const accscalar_t1 w1lambda24 = w1r21 - w122;
        const accscalar_t1 w0lambda25 = static_cast<accscalar_t1>(1) - w1lambda24;
        for (int n = 0; n < batchsize8; n++) {
            for (int c = 0; c < channels9; ++c) {
                const accscalar_t1 val29 = h0lambda20 * (w0lambda25 * idata6[n][c][h117][w122] + w1lambda24 * idata6[n][c][h117][w122 + w1p23]) + h1lambda19 * (w0lambda25 * idata6[n][c][h117 + h1p18][w122] + w1lambda24 * idata6[n][c][h117 + h1p18][w122 + w1p23]);
                odata7[n][c][h215][w214] = static_cast<scalar_t0>(val29);
            }
        }
    }
}
}
template <typename output_t30, typename input_t31, typename IndexType32, int ADims33, int PDims34, int BDims35, at::native::CUDAHistogramMemoryType MemoryType36 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op37, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(1024, 2) void kernelHistogram1D_upsample_bilinear2d_out_frame_fused_kernel_hfuse_lb_idx_3(TensorInfo<output_t30, IndexType32> a38, TensorInfo<output_t30, IndexType32> p39, TensorInfo<input_t31, IndexType32> b40, int nbins41, input_t31 minvalue42, input_t31 maxvalue43, IndexType32 totalElements44, Op37 getOp45, const int ns2, const accscalar_t1 rheight3, const accscalar_t1 rwidth4, const bool align_corners5, const PackedTensorAccessor<scalar_t0, 4> idata6, PackedTensorAccessor<scalar_t0, 4> odata7)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)){
    unsigned int blockDim_x_1 = 512;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 512;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512;
    extern unsigned char my_smem46[] __attribute__((shared));
    output_t30 *smem47 = nullptr;
    smem47 = reinterpret_cast<output_t30 *>(my_smem46);
    for (IndexType32 i = threadIdx_x_1; i < a38.sizes[0]; i += blockDim_x_1) {
        smem47[i] = 0;
    }
    asm ("bar.sync 1,512;");
    ;
    for (IndexType32 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements44; linearIndex += gridDim.x * blockDim_x_1) {
        const IndexType32 bOffset48 = IndexToOffset<input_t31, IndexType32, BDims35>::get(linearIndex, b40);
        const input_t31 bVal49 = b40.data[bOffset48];
        if (bVal49 >= minvalue42 && bVal49 <= maxvalue43) {
            const IndexType32 bin50 = getBin<input_t31, IndexType32>(bVal49, minvalue42, maxvalue43, nbins41);
            atomicAdd(& smem47[bin50], getOp45(linearIndex));
        }
    }
    asm ("bar.sync 1,512;");
    ;
    for (IndexType32 i = threadIdx_x_1; i < a38.sizes[0]; i += blockDim_x_1) {
        const IndexType32 aOffset51 = IndexToOffset<output_t30, IndexType32, ADims33>::get(i, a38);
        atomicAdd(& a38.data[aOffset51], smem47[i]);
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)){
    unsigned int blockDim_x_0 = 512;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) % 512;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 512 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 512;
    const int batchsize8 = idata6.size(0);
    const int channels9 = idata6.size(1);
    const int height110 = idata6.size(2);
    const int width111 = idata6.size(3);
    const int height212 = odata7.size(2);
    const int width213 = odata7.size(3);
    for (int index = blockIdx.x * blockDim_x_0 + threadIdx_x_0; index < (ns2); index += blockDim_x_0 * gridDim.x) {
        const int w214 = index % width213;
        const int h215 = index / width213;
        if (height110 == height212 && width111 == width213) {
            const int h126 = h215;
            const int w127 = w214;
            for (int n = 0; n < batchsize8; n++) {
                for (int c = 0; c < channels9; ++c) {
                    const scalar_t0 val28 = idata6[n][c][h126][w127];
                    odata7[n][c][h215][w214] = val28;
                }
            }
            return;
        }
        const accscalar_t1 h1r16 = area_pixel_compute_source_index<accscalar_t1>(rheight3, h215, align_corners5, false);
        const int h117 = h1r16;
        const int h1p18 = (h117 < height110 - 1) ? 1 : 0;
        const accscalar_t1 h1lambda19 = h1r16 - h117;
        const accscalar_t1 h0lambda20 = static_cast<accscalar_t1>(1) - h1lambda19;
        const accscalar_t1 w1r21 = area_pixel_compute_source_index<accscalar_t1>(rwidth4, w214, align_corners5, false);
        const int w122 = w1r21;
        const int w1p23 = (w122 < width111 - 1) ? 1 : 0;
        const accscalar_t1 w1lambda24 = w1r21 - w122;
        const accscalar_t1 w0lambda25 = static_cast<accscalar_t1>(1) - w1lambda24;
        for (int n = 0; n < batchsize8; n++) {
            for (int c = 0; c < channels9; ++c) {
                const accscalar_t1 val29 = h0lambda20 * (w0lambda25 * idata6[n][c][h117][w122] + w1lambda24 * idata6[n][c][h117][w122 + w1p23]) + h1lambda19 * (w0lambda25 * idata6[n][c][h117 + h1p18][w122] + w1lambda24 * idata6[n][c][h117 + h1p18][w122 + w1p23]);
                odata7[n][c][h215][w214] = static_cast<scalar_t0>(val29);
            }
        }
    }
}
}
template <typename output_t30, typename input_t31, typename IndexType32, int ADims33, int PDims34, int BDims35, at::native::CUDAHistogramMemoryType MemoryType36 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op37, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(1024, 2) void kernelHistogram1D_upsample_bilinear2d_out_frame_fused_kernel_hfuse_lb_idx_4(TensorInfo<output_t30, IndexType32> a38, TensorInfo<output_t30, IndexType32> p39, TensorInfo<input_t31, IndexType32> b40, int nbins41, input_t31 minvalue42, input_t31 maxvalue43, IndexType32 totalElements44, Op37 getOp45, const int ns2, const accscalar_t1 rheight3, const accscalar_t1 rwidth4, const bool align_corners5, const PackedTensorAccessor<scalar_t0, 4> idata6, PackedTensorAccessor<scalar_t0, 4> odata7)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 640)){
    unsigned int blockDim_x_1 = 640;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 640;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 640 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 640;
    extern unsigned char my_smem46[] __attribute__((shared));
    output_t30 *smem47 = nullptr;
    smem47 = reinterpret_cast<output_t30 *>(my_smem46);
    for (IndexType32 i = threadIdx_x_1; i < a38.sizes[0]; i += blockDim_x_1) {
        smem47[i] = 0;
    }
    asm ("bar.sync 1,640;");
    ;
    for (IndexType32 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements44; linearIndex += gridDim.x * blockDim_x_1) {
        const IndexType32 bOffset48 = IndexToOffset<input_t31, IndexType32, BDims35>::get(linearIndex, b40);
        const input_t31 bVal49 = b40.data[bOffset48];
        if (bVal49 >= minvalue42 && bVal49 <= maxvalue43) {
            const IndexType32 bin50 = getBin<input_t31, IndexType32>(bVal49, minvalue42, maxvalue43, nbins41);
            atomicAdd(& smem47[bin50], getOp45(linearIndex));
        }
    }
    asm ("bar.sync 1,640;");
    ;
    for (IndexType32 i = threadIdx_x_1; i < a38.sizes[0]; i += blockDim_x_1) {
        const IndexType32 aOffset51 = IndexToOffset<output_t30, IndexType32, ADims33>::get(i, a38);
        atomicAdd(& a38.data[aOffset51], smem47[i]);
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=640 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)){
    unsigned int blockDim_x_0 = 384;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 640) % 384;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 640) / 384 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 640) / 384;
    const int batchsize8 = idata6.size(0);
    const int channels9 = idata6.size(1);
    const int height110 = idata6.size(2);
    const int width111 = idata6.size(3);
    const int height212 = odata7.size(2);
    const int width213 = odata7.size(3);
    for (int index = blockIdx.x * blockDim_x_0 + threadIdx_x_0; index < (ns2); index += blockDim_x_0 * gridDim.x) {
        const int w214 = index % width213;
        const int h215 = index / width213;
        if (height110 == height212 && width111 == width213) {
            const int h126 = h215;
            const int w127 = w214;
            for (int n = 0; n < batchsize8; n++) {
                for (int c = 0; c < channels9; ++c) {
                    const scalar_t0 val28 = idata6[n][c][h126][w127];
                    odata7[n][c][h215][w214] = val28;
                }
            }
            return;
        }
        const accscalar_t1 h1r16 = area_pixel_compute_source_index<accscalar_t1>(rheight3, h215, align_corners5, false);
        const int h117 = h1r16;
        const int h1p18 = (h117 < height110 - 1) ? 1 : 0;
        const accscalar_t1 h1lambda19 = h1r16 - h117;
        const accscalar_t1 h0lambda20 = static_cast<accscalar_t1>(1) - h1lambda19;
        const accscalar_t1 w1r21 = area_pixel_compute_source_index<accscalar_t1>(rwidth4, w214, align_corners5, false);
        const int w122 = w1r21;
        const int w1p23 = (w122 < width111 - 1) ? 1 : 0;
        const accscalar_t1 w1lambda24 = w1r21 - w122;
        const accscalar_t1 w0lambda25 = static_cast<accscalar_t1>(1) - w1lambda24;
        for (int n = 0; n < batchsize8; n++) {
            for (int c = 0; c < channels9; ++c) {
                const accscalar_t1 val29 = h0lambda20 * (w0lambda25 * idata6[n][c][h117][w122] + w1lambda24 * idata6[n][c][h117][w122 + w1p23]) + h1lambda19 * (w0lambda25 * idata6[n][c][h117 + h1p18][w122] + w1lambda24 * idata6[n][c][h117 + h1p18][w122 + w1p23]);
                odata7[n][c][h215][w214] = static_cast<scalar_t0>(val29);
            }
        }
    }
}
}
template <typename output_t30, typename input_t31, typename IndexType32, int ADims33, int PDims34, int BDims35, at::native::CUDAHistogramMemoryType MemoryType36 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op37, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(1024, 2) void kernelHistogram1D_upsample_bilinear2d_out_frame_fused_kernel_hfuse_lb_idx_5(TensorInfo<output_t30, IndexType32> a38, TensorInfo<output_t30, IndexType32> p39, TensorInfo<input_t31, IndexType32> b40, int nbins41, input_t31 minvalue42, input_t31 maxvalue43, IndexType32 totalElements44, Op37 getOp45, const int ns2, const accscalar_t1 rheight3, const accscalar_t1 rwidth4, const bool align_corners5, const PackedTensorAccessor<scalar_t0, 4> idata6, PackedTensorAccessor<scalar_t0, 4> odata7)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 768)){
    unsigned int blockDim_x_1 = 768;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 768;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 768 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 768;
    extern unsigned char my_smem46[] __attribute__((shared));
    output_t30 *smem47 = nullptr;
    smem47 = reinterpret_cast<output_t30 *>(my_smem46);
    for (IndexType32 i = threadIdx_x_1; i < a38.sizes[0]; i += blockDim_x_1) {
        smem47[i] = 0;
    }
    asm ("bar.sync 1,768;");
    ;
    for (IndexType32 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements44; linearIndex += gridDim.x * blockDim_x_1) {
        const IndexType32 bOffset48 = IndexToOffset<input_t31, IndexType32, BDims35>::get(linearIndex, b40);
        const input_t31 bVal49 = b40.data[bOffset48];
        if (bVal49 >= minvalue42 && bVal49 <= maxvalue43) {
            const IndexType32 bin50 = getBin<input_t31, IndexType32>(bVal49, minvalue42, maxvalue43, nbins41);
            atomicAdd(& smem47[bin50], getOp45(linearIndex));
        }
    }
    asm ("bar.sync 1,768;");
    ;
    for (IndexType32 i = threadIdx_x_1; i < a38.sizes[0]; i += blockDim_x_1) {
        const IndexType32 aOffset51 = IndexToOffset<output_t30, IndexType32, ADims33>::get(i, a38);
        atomicAdd(& a38.data[aOffset51], smem47[i]);
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=768 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)){
    unsigned int blockDim_x_0 = 256;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 768) % 256;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 768) / 256 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 768) / 256;
    const int batchsize8 = idata6.size(0);
    const int channels9 = idata6.size(1);
    const int height110 = idata6.size(2);
    const int width111 = idata6.size(3);
    const int height212 = odata7.size(2);
    const int width213 = odata7.size(3);
    for (int index = blockIdx.x * blockDim_x_0 + threadIdx_x_0; index < (ns2); index += blockDim_x_0 * gridDim.x) {
        const int w214 = index % width213;
        const int h215 = index / width213;
        if (height110 == height212 && width111 == width213) {
            const int h126 = h215;
            const int w127 = w214;
            for (int n = 0; n < batchsize8; n++) {
                for (int c = 0; c < channels9; ++c) {
                    const scalar_t0 val28 = idata6[n][c][h126][w127];
                    odata7[n][c][h215][w214] = val28;
                }
            }
            return;
        }
        const accscalar_t1 h1r16 = area_pixel_compute_source_index<accscalar_t1>(rheight3, h215, align_corners5, false);
        const int h117 = h1r16;
        const int h1p18 = (h117 < height110 - 1) ? 1 : 0;
        const accscalar_t1 h1lambda19 = h1r16 - h117;
        const accscalar_t1 h0lambda20 = static_cast<accscalar_t1>(1) - h1lambda19;
        const accscalar_t1 w1r21 = area_pixel_compute_source_index<accscalar_t1>(rwidth4, w214, align_corners5, false);
        const int w122 = w1r21;
        const int w1p23 = (w122 < width111 - 1) ? 1 : 0;
        const accscalar_t1 w1lambda24 = w1r21 - w122;
        const accscalar_t1 w0lambda25 = static_cast<accscalar_t1>(1) - w1lambda24;
        for (int n = 0; n < batchsize8; n++) {
            for (int c = 0; c < channels9; ++c) {
                const accscalar_t1 val29 = h0lambda20 * (w0lambda25 * idata6[n][c][h117][w122] + w1lambda24 * idata6[n][c][h117][w122 + w1p23]) + h1lambda19 * (w0lambda25 * idata6[n][c][h117 + h1p18][w122] + w1lambda24 * idata6[n][c][h117 + h1p18][w122 + w1p23]);
                odata7[n][c][h215][w214] = static_cast<scalar_t0>(val29);
            }
        }
    }
}
}
template <typename output_t30, typename input_t31, typename IndexType32, int ADims33, int PDims34, int BDims35, at::native::CUDAHistogramMemoryType MemoryType36 = CUDAHistogramMemoryType::MULTI_BLOCK, typename Op37, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(1024, 2) void kernelHistogram1D_upsample_bilinear2d_out_frame_fused_kernel_hfuse_lb_idx_6(TensorInfo<output_t30, IndexType32> a38, TensorInfo<output_t30, IndexType32> p39, TensorInfo<input_t31, IndexType32> b40, int nbins41, input_t31 minvalue42, input_t31 maxvalue43, IndexType32 totalElements44, Op37 getOp45, const int ns2, const accscalar_t1 rheight3, const accscalar_t1 rwidth4, const bool align_corners5, const PackedTensorAccessor<scalar_t0, 4> idata6, PackedTensorAccessor<scalar_t0, 4> odata7)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 896)){
    unsigned int blockDim_x_1 = 896;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 896;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 896 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 896;
    extern unsigned char my_smem46[] __attribute__((shared));
    output_t30 *smem47 = nullptr;
    smem47 = reinterpret_cast<output_t30 *>(my_smem46);
    for (IndexType32 i = threadIdx_x_1; i < a38.sizes[0]; i += blockDim_x_1) {
        smem47[i] = 0;
    }
    asm ("bar.sync 1,896;");
    ;
    for (IndexType32 linearIndex = blockIdx.x * blockDim_x_1 + threadIdx_x_1; linearIndex < totalElements44; linearIndex += gridDim.x * blockDim_x_1) {
        const IndexType32 bOffset48 = IndexToOffset<input_t31, IndexType32, BDims35>::get(linearIndex, b40);
        const input_t31 bVal49 = b40.data[bOffset48];
        if (bVal49 >= minvalue42 && bVal49 <= maxvalue43) {
            const IndexType32 bin50 = getBin<input_t31, IndexType32>(bVal49, minvalue42, maxvalue43, nbins41);
            atomicAdd(& smem47[bin50], getOp45(linearIndex));
        }
    }
    asm ("bar.sync 1,896;");
    ;
    for (IndexType32 i = threadIdx_x_1; i < a38.sizes[0]; i += blockDim_x_1) {
        const IndexType32 aOffset51 = IndexToOffset<output_t30, IndexType32, ADims33>::get(i, a38);
        atomicAdd(& a38.data[aOffset51], smem47[i]);
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=896 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 1024)){
    unsigned int blockDim_x_0 = 128;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 896) % 128;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 896) / 128 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 896) / 128;
    const int batchsize8 = idata6.size(0);
    const int channels9 = idata6.size(1);
    const int height110 = idata6.size(2);
    const int width111 = idata6.size(3);
    const int height212 = odata7.size(2);
    const int width213 = odata7.size(3);
    for (int index = blockIdx.x * blockDim_x_0 + threadIdx_x_0; index < (ns2); index += blockDim_x_0 * gridDim.x) {
        const int w214 = index % width213;
        const int h215 = index / width213;
        if (height110 == height212 && width111 == width213) {
            const int h126 = h215;
            const int w127 = w214;
            for (int n = 0; n < batchsize8; n++) {
                for (int c = 0; c < channels9; ++c) {
                    const scalar_t0 val28 = idata6[n][c][h126][w127];
                    odata7[n][c][h215][w214] = val28;
                }
            }
            return;
        }
        const accscalar_t1 h1r16 = area_pixel_compute_source_index<accscalar_t1>(rheight3, h215, align_corners5, false);
        const int h117 = h1r16;
        const int h1p18 = (h117 < height110 - 1) ? 1 : 0;
        const accscalar_t1 h1lambda19 = h1r16 - h117;
        const accscalar_t1 h0lambda20 = static_cast<accscalar_t1>(1) - h1lambda19;
        const accscalar_t1 w1r21 = area_pixel_compute_source_index<accscalar_t1>(rwidth4, w214, align_corners5, false);
        const int w122 = w1r21;
        const int w1p23 = (w122 < width111 - 1) ? 1 : 0;
        const accscalar_t1 w1lambda24 = w1r21 - w122;
        const accscalar_t1 w0lambda25 = static_cast<accscalar_t1>(1) - w1lambda24;
        for (int n = 0; n < batchsize8; n++) {
            for (int c = 0; c < channels9; ++c) {
                const accscalar_t1 val29 = h0lambda20 * (w0lambda25 * idata6[n][c][h117][w122] + w1lambda24 * idata6[n][c][h117][w122 + w1p23]) + h1lambda19 * (w0lambda25 * idata6[n][c][h117 + h1p18][w122] + w1lambda24 * idata6[n][c][h117 + h1p18][w122 + w1p23]);
                odata7[n][c][h215][w214] = static_cast<scalar_t0>(val29);
            }
        }
    }
}
}
