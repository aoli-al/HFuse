template <typename scalar_t30, typename accscalar_t31, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(512, 2) void MaxPoolForward_upsample_bilinear2d_out_frame_fused_kernel_vfuse_lb_idx_0(const int nthreads32, const scalar_t30 *bottom_data33, const int num34, const int channels35, const int height36, const int width37, const int pooled_height38, const int pooled_width39, const int kernel_h40, const int kernel_w41, const int stride_h42, const int stride_w43, const int pad_h44, const int pad_w45, const int dilation_h46, const int dilation_w47, scalar_t30 *top_data48, int64_t *top_mask49, const int ns2, const accscalar_t1 rheight3, const accscalar_t1 rwidth4, const bool align_corners5, const PackedTensorAccessor<scalar_t0, 4> idata6, PackedTensorAccessor<scalar_t0, 4> odata7)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 256)){
    unsigned int blockDim_x_1 = 256;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 256;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256;
    for (int index = blockIdx.x * blockDim_x_1 + threadIdx_x_1; index < (nthreads32); index += blockDim_x_1 * gridDim.x) {
        int pw50 = index % pooled_width39;
        int ph51 = (index / pooled_width39) % pooled_height38;
        int c52 = (index / pooled_width39 / pooled_height38) % channels35;
        int n53 = index / pooled_width39 / pooled_height38 / channels35;
        int hstart54 = ph51 * stride_h42 - pad_h44;
        int wstart55 = pw50 * stride_w43 - pad_w45;
        int hend56 = min(hstart54 + (kernel_h40 - 1) * dilation_h46 + 1, height36);
        int wend57 = min(wstart55 + (kernel_w41 - 1) * dilation_w47 + 1, width37);
        while (hstart54 < 0)
            hstart54 += dilation_h46;
        while (wstart55 < 0)
            wstart55 += dilation_w47;
        accscalar_t31 maxval58 = at::numeric_limits<accscalar_t31>::lower_bound();
        int maxidx59 = hstart54 * width37 + wstart55;
        const scalar_t30 *btm_data60 = bottom_data33 + (n53 * channels35 + c52) * height36 * width37;
        for (int h = hstart54; h < hend56; h += dilation_h46) {
            for (int w = wstart55; w < wend57; w += dilation_w47) {
                scalar_t30 val61 = btm_data60[h * width37 + w];
                if ((ScalarConvert<scalar_t30, accscalar_t31>::to(val61) > maxval58) || THCNumerics<scalar_t30>::isnan(val61)) {
                    maxidx59 = h * width37 + w;
                    maxval58 = ScalarConvert<scalar_t30, accscalar_t31>::to(val61);
                }
            }
        }
        top_data48[index] = ScalarConvert<scalar_t30, accscalar_t31>::to(maxval58);
        top_mask49[index] = maxidx59;
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
template <typename scalar_t30, typename accscalar_t31, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(512, 0) void MaxPoolForward_upsample_bilinear2d_out_frame_fused_kernel_vfuse_idx_0(const int nthreads32, const scalar_t30 *bottom_data33, const int num34, const int channels35, const int height36, const int width37, const int pooled_height38, const int pooled_width39, const int kernel_h40, const int kernel_w41, const int stride_h42, const int stride_w43, const int pad_h44, const int pad_w45, const int dilation_h46, const int dilation_w47, scalar_t30 *top_data48, int64_t *top_mask49, const int ns2, const accscalar_t1 rheight3, const accscalar_t1 rwidth4, const bool align_corners5, const PackedTensorAccessor<scalar_t0, 4> idata6, PackedTensorAccessor<scalar_t0, 4> odata7)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 256)){
    unsigned int blockDim_x_1 = 256;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 256;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256;
    for (int index = blockIdx.x * blockDim_x_1 + threadIdx_x_1; index < (nthreads32); index += blockDim_x_1 * gridDim.x) {
        int pw50 = index % pooled_width39;
        int ph51 = (index / pooled_width39) % pooled_height38;
        int c52 = (index / pooled_width39 / pooled_height38) % channels35;
        int n53 = index / pooled_width39 / pooled_height38 / channels35;
        int hstart54 = ph51 * stride_h42 - pad_h44;
        int wstart55 = pw50 * stride_w43 - pad_w45;
        int hend56 = min(hstart54 + (kernel_h40 - 1) * dilation_h46 + 1, height36);
        int wend57 = min(wstart55 + (kernel_w41 - 1) * dilation_w47 + 1, width37);
        while (hstart54 < 0)
            hstart54 += dilation_h46;
        while (wstart55 < 0)
            wstart55 += dilation_w47;
        accscalar_t31 maxval58 = at::numeric_limits<accscalar_t31>::lower_bound();
        int maxidx59 = hstart54 * width37 + wstart55;
        const scalar_t30 *btm_data60 = bottom_data33 + (n53 * channels35 + c52) * height36 * width37;
        for (int h = hstart54; h < hend56; h += dilation_h46) {
            for (int w = wstart55; w < wend57; w += dilation_w47) {
                scalar_t30 val61 = btm_data60[h * width37 + w];
                if ((ScalarConvert<scalar_t30, accscalar_t31>::to(val61) > maxval58) || THCNumerics<scalar_t30>::isnan(val61)) {
                    maxidx59 = h * width37 + w;
                    maxval58 = ScalarConvert<scalar_t30, accscalar_t31>::to(val61);
                }
            }
        }
        top_data48[index] = ScalarConvert<scalar_t30, accscalar_t31>::to(maxval58);
        top_mask49[index] = maxidx59;
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
template <typename scalar_t30, typename accscalar_t31, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(768, 0) void MaxPoolForward_upsample_bilinear2d_out_frame_fused_kernel_hfuse_idx_0(const int nthreads32, const scalar_t30 *bottom_data33, const int num34, const int channels35, const int height36, const int width37, const int pooled_height38, const int pooled_width39, const int kernel_h40, const int kernel_w41, const int stride_h42, const int stride_w43, const int pad_h44, const int pad_w45, const int dilation_h46, const int dilation_w47, scalar_t30 *top_data48, int64_t *top_mask49, const int ns2, const accscalar_t1 rheight3, const accscalar_t1 rwidth4, const bool align_corners5, const PackedTensorAccessor<scalar_t0, 4> idata6, PackedTensorAccessor<scalar_t0, 4> odata7)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 128)){
    unsigned int blockDim_x_1 = 128;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 128;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128;
    for (int index = blockIdx.x * blockDim_x_1 + threadIdx_x_1; index < (nthreads32); index += blockDim_x_1 * gridDim.x) {
        int pw50 = index % pooled_width39;
        int ph51 = (index / pooled_width39) % pooled_height38;
        int c52 = (index / pooled_width39 / pooled_height38) % channels35;
        int n53 = index / pooled_width39 / pooled_height38 / channels35;
        int hstart54 = ph51 * stride_h42 - pad_h44;
        int wstart55 = pw50 * stride_w43 - pad_w45;
        int hend56 = min(hstart54 + (kernel_h40 - 1) * dilation_h46 + 1, height36);
        int wend57 = min(wstart55 + (kernel_w41 - 1) * dilation_w47 + 1, width37);
        while (hstart54 < 0)
            hstart54 += dilation_h46;
        while (wstart55 < 0)
            wstart55 += dilation_w47;
        accscalar_t31 maxval58 = at::numeric_limits<accscalar_t31>::lower_bound();
        int maxidx59 = hstart54 * width37 + wstart55;
        const scalar_t30 *btm_data60 = bottom_data33 + (n53 * channels35 + c52) * height36 * width37;
        for (int h = hstart54; h < hend56; h += dilation_h46) {
            for (int w = wstart55; w < wend57; w += dilation_w47) {
                scalar_t30 val61 = btm_data60[h * width37 + w];
                if ((ScalarConvert<scalar_t30, accscalar_t31>::to(val61) > maxval58) || THCNumerics<scalar_t30>::isnan(val61)) {
                    maxidx59 = h * width37 + w;
                    maxval58 = ScalarConvert<scalar_t30, accscalar_t31>::to(val61);
                }
            }
        }
        top_data48[index] = ScalarConvert<scalar_t30, accscalar_t31>::to(maxval58);
        top_mask49[index] = maxidx59;
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=128 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 768)){
    unsigned int blockDim_x_0 = 640;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) % 640;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) / 640 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) / 640;
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
template <typename scalar_t30, typename accscalar_t31, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(768, 0) void MaxPoolForward_upsample_bilinear2d_out_frame_fused_kernel_hfuse_idx_1(const int nthreads32, const scalar_t30 *bottom_data33, const int num34, const int channels35, const int height36, const int width37, const int pooled_height38, const int pooled_width39, const int kernel_h40, const int kernel_w41, const int stride_h42, const int stride_w43, const int pad_h44, const int pad_w45, const int dilation_h46, const int dilation_w47, scalar_t30 *top_data48, int64_t *top_mask49, const int ns2, const accscalar_t1 rheight3, const accscalar_t1 rwidth4, const bool align_corners5, const PackedTensorAccessor<scalar_t0, 4> idata6, PackedTensorAccessor<scalar_t0, 4> odata7)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 256)){
    unsigned int blockDim_x_1 = 256;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 256;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256;
    for (int index = blockIdx.x * blockDim_x_1 + threadIdx_x_1; index < (nthreads32); index += blockDim_x_1 * gridDim.x) {
        int pw50 = index % pooled_width39;
        int ph51 = (index / pooled_width39) % pooled_height38;
        int c52 = (index / pooled_width39 / pooled_height38) % channels35;
        int n53 = index / pooled_width39 / pooled_height38 / channels35;
        int hstart54 = ph51 * stride_h42 - pad_h44;
        int wstart55 = pw50 * stride_w43 - pad_w45;
        int hend56 = min(hstart54 + (kernel_h40 - 1) * dilation_h46 + 1, height36);
        int wend57 = min(wstart55 + (kernel_w41 - 1) * dilation_w47 + 1, width37);
        while (hstart54 < 0)
            hstart54 += dilation_h46;
        while (wstart55 < 0)
            wstart55 += dilation_w47;
        accscalar_t31 maxval58 = at::numeric_limits<accscalar_t31>::lower_bound();
        int maxidx59 = hstart54 * width37 + wstart55;
        const scalar_t30 *btm_data60 = bottom_data33 + (n53 * channels35 + c52) * height36 * width37;
        for (int h = hstart54; h < hend56; h += dilation_h46) {
            for (int w = wstart55; w < wend57; w += dilation_w47) {
                scalar_t30 val61 = btm_data60[h * width37 + w];
                if ((ScalarConvert<scalar_t30, accscalar_t31>::to(val61) > maxval58) || THCNumerics<scalar_t30>::isnan(val61)) {
                    maxidx59 = h * width37 + w;
                    maxval58 = ScalarConvert<scalar_t30, accscalar_t31>::to(val61);
                }
            }
        }
        top_data48[index] = ScalarConvert<scalar_t30, accscalar_t31>::to(maxval58);
        top_mask49[index] = maxidx59;
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=256 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 768)){
    unsigned int blockDim_x_0 = 512;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 256) % 512;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 256) / 512 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 256) / 512;
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
template <typename scalar_t30, typename accscalar_t31, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(768, 0) void MaxPoolForward_upsample_bilinear2d_out_frame_fused_kernel_hfuse_idx_2(const int nthreads32, const scalar_t30 *bottom_data33, const int num34, const int channels35, const int height36, const int width37, const int pooled_height38, const int pooled_width39, const int kernel_h40, const int kernel_w41, const int stride_h42, const int stride_w43, const int pad_h44, const int pad_w45, const int dilation_h46, const int dilation_w47, scalar_t30 *top_data48, int64_t *top_mask49, const int ns2, const accscalar_t1 rheight3, const accscalar_t1 rwidth4, const bool align_corners5, const PackedTensorAccessor<scalar_t0, 4> idata6, PackedTensorAccessor<scalar_t0, 4> odata7)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 384)){
    unsigned int blockDim_x_1 = 384;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 384;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 384 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 384;
    for (int index = blockIdx.x * blockDim_x_1 + threadIdx_x_1; index < (nthreads32); index += blockDim_x_1 * gridDim.x) {
        int pw50 = index % pooled_width39;
        int ph51 = (index / pooled_width39) % pooled_height38;
        int c52 = (index / pooled_width39 / pooled_height38) % channels35;
        int n53 = index / pooled_width39 / pooled_height38 / channels35;
        int hstart54 = ph51 * stride_h42 - pad_h44;
        int wstart55 = pw50 * stride_w43 - pad_w45;
        int hend56 = min(hstart54 + (kernel_h40 - 1) * dilation_h46 + 1, height36);
        int wend57 = min(wstart55 + (kernel_w41 - 1) * dilation_w47 + 1, width37);
        while (hstart54 < 0)
            hstart54 += dilation_h46;
        while (wstart55 < 0)
            wstart55 += dilation_w47;
        accscalar_t31 maxval58 = at::numeric_limits<accscalar_t31>::lower_bound();
        int maxidx59 = hstart54 * width37 + wstart55;
        const scalar_t30 *btm_data60 = bottom_data33 + (n53 * channels35 + c52) * height36 * width37;
        for (int h = hstart54; h < hend56; h += dilation_h46) {
            for (int w = wstart55; w < wend57; w += dilation_w47) {
                scalar_t30 val61 = btm_data60[h * width37 + w];
                if ((ScalarConvert<scalar_t30, accscalar_t31>::to(val61) > maxval58) || THCNumerics<scalar_t30>::isnan(val61)) {
                    maxidx59 = h * width37 + w;
                    maxval58 = ScalarConvert<scalar_t30, accscalar_t31>::to(val61);
                }
            }
        }
        top_data48[index] = ScalarConvert<scalar_t30, accscalar_t31>::to(maxval58);
        top_mask49[index] = maxidx59;
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=384 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 768)){
    unsigned int blockDim_x_0 = 384;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 384) % 384;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 384) / 384 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 384) / 384;
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
template <typename scalar_t30, typename accscalar_t31, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(768, 0) void MaxPoolForward_upsample_bilinear2d_out_frame_fused_kernel_hfuse_idx_3(const int nthreads32, const scalar_t30 *bottom_data33, const int num34, const int channels35, const int height36, const int width37, const int pooled_height38, const int pooled_width39, const int kernel_h40, const int kernel_w41, const int stride_h42, const int stride_w43, const int pad_h44, const int pad_w45, const int dilation_h46, const int dilation_w47, scalar_t30 *top_data48, int64_t *top_mask49, const int ns2, const accscalar_t1 rheight3, const accscalar_t1 rwidth4, const bool align_corners5, const PackedTensorAccessor<scalar_t0, 4> idata6, PackedTensorAccessor<scalar_t0, 4> odata7)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)){
    unsigned int blockDim_x_1 = 512;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 512;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512;
    for (int index = blockIdx.x * blockDim_x_1 + threadIdx_x_1; index < (nthreads32); index += blockDim_x_1 * gridDim.x) {
        int pw50 = index % pooled_width39;
        int ph51 = (index / pooled_width39) % pooled_height38;
        int c52 = (index / pooled_width39 / pooled_height38) % channels35;
        int n53 = index / pooled_width39 / pooled_height38 / channels35;
        int hstart54 = ph51 * stride_h42 - pad_h44;
        int wstart55 = pw50 * stride_w43 - pad_w45;
        int hend56 = min(hstart54 + (kernel_h40 - 1) * dilation_h46 + 1, height36);
        int wend57 = min(wstart55 + (kernel_w41 - 1) * dilation_w47 + 1, width37);
        while (hstart54 < 0)
            hstart54 += dilation_h46;
        while (wstart55 < 0)
            wstart55 += dilation_w47;
        accscalar_t31 maxval58 = at::numeric_limits<accscalar_t31>::lower_bound();
        int maxidx59 = hstart54 * width37 + wstart55;
        const scalar_t30 *btm_data60 = bottom_data33 + (n53 * channels35 + c52) * height36 * width37;
        for (int h = hstart54; h < hend56; h += dilation_h46) {
            for (int w = wstart55; w < wend57; w += dilation_w47) {
                scalar_t30 val61 = btm_data60[h * width37 + w];
                if ((ScalarConvert<scalar_t30, accscalar_t31>::to(val61) > maxval58) || THCNumerics<scalar_t30>::isnan(val61)) {
                    maxidx59 = h * width37 + w;
                    maxval58 = ScalarConvert<scalar_t30, accscalar_t31>::to(val61);
                }
            }
        }
        top_data48[index] = ScalarConvert<scalar_t30, accscalar_t31>::to(maxval58);
        top_mask49[index] = maxidx59;
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 768)){
    unsigned int blockDim_x_0 = 256;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) % 256;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 256 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 256;
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
template <typename scalar_t30, typename accscalar_t31, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(768, 0) void MaxPoolForward_upsample_bilinear2d_out_frame_fused_kernel_hfuse_idx_4(const int nthreads32, const scalar_t30 *bottom_data33, const int num34, const int channels35, const int height36, const int width37, const int pooled_height38, const int pooled_width39, const int kernel_h40, const int kernel_w41, const int stride_h42, const int stride_w43, const int pad_h44, const int pad_w45, const int dilation_h46, const int dilation_w47, scalar_t30 *top_data48, int64_t *top_mask49, const int ns2, const accscalar_t1 rheight3, const accscalar_t1 rwidth4, const bool align_corners5, const PackedTensorAccessor<scalar_t0, 4> idata6, PackedTensorAccessor<scalar_t0, 4> odata7)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 640)){
    unsigned int blockDim_x_1 = 640;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 640;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 640 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 640;
    for (int index = blockIdx.x * blockDim_x_1 + threadIdx_x_1; index < (nthreads32); index += blockDim_x_1 * gridDim.x) {
        int pw50 = index % pooled_width39;
        int ph51 = (index / pooled_width39) % pooled_height38;
        int c52 = (index / pooled_width39 / pooled_height38) % channels35;
        int n53 = index / pooled_width39 / pooled_height38 / channels35;
        int hstart54 = ph51 * stride_h42 - pad_h44;
        int wstart55 = pw50 * stride_w43 - pad_w45;
        int hend56 = min(hstart54 + (kernel_h40 - 1) * dilation_h46 + 1, height36);
        int wend57 = min(wstart55 + (kernel_w41 - 1) * dilation_w47 + 1, width37);
        while (hstart54 < 0)
            hstart54 += dilation_h46;
        while (wstart55 < 0)
            wstart55 += dilation_w47;
        accscalar_t31 maxval58 = at::numeric_limits<accscalar_t31>::lower_bound();
        int maxidx59 = hstart54 * width37 + wstart55;
        const scalar_t30 *btm_data60 = bottom_data33 + (n53 * channels35 + c52) * height36 * width37;
        for (int h = hstart54; h < hend56; h += dilation_h46) {
            for (int w = wstart55; w < wend57; w += dilation_w47) {
                scalar_t30 val61 = btm_data60[h * width37 + w];
                if ((ScalarConvert<scalar_t30, accscalar_t31>::to(val61) > maxval58) || THCNumerics<scalar_t30>::isnan(val61)) {
                    maxidx59 = h * width37 + w;
                    maxval58 = ScalarConvert<scalar_t30, accscalar_t31>::to(val61);
                }
            }
        }
        top_data48[index] = ScalarConvert<scalar_t30, accscalar_t31>::to(maxval58);
        top_mask49[index] = maxidx59;
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=640 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 768)){
    unsigned int blockDim_x_0 = 128;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 640) % 128;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 640) / 128 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 640) / 128;
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
template <typename scalar_t30, typename accscalar_t31, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(768, 2) void MaxPoolForward_upsample_bilinear2d_out_frame_fused_kernel_hfuse_lb_idx_0(const int nthreads32, const scalar_t30 *bottom_data33, const int num34, const int channels35, const int height36, const int width37, const int pooled_height38, const int pooled_width39, const int kernel_h40, const int kernel_w41, const int stride_h42, const int stride_w43, const int pad_h44, const int pad_w45, const int dilation_h46, const int dilation_w47, scalar_t30 *top_data48, int64_t *top_mask49, const int ns2, const accscalar_t1 rheight3, const accscalar_t1 rwidth4, const bool align_corners5, const PackedTensorAccessor<scalar_t0, 4> idata6, PackedTensorAccessor<scalar_t0, 4> odata7)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 128)){
    unsigned int blockDim_x_1 = 128;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 128;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 128;
    for (int index = blockIdx.x * blockDim_x_1 + threadIdx_x_1; index < (nthreads32); index += blockDim_x_1 * gridDim.x) {
        int pw50 = index % pooled_width39;
        int ph51 = (index / pooled_width39) % pooled_height38;
        int c52 = (index / pooled_width39 / pooled_height38) % channels35;
        int n53 = index / pooled_width39 / pooled_height38 / channels35;
        int hstart54 = ph51 * stride_h42 - pad_h44;
        int wstart55 = pw50 * stride_w43 - pad_w45;
        int hend56 = min(hstart54 + (kernel_h40 - 1) * dilation_h46 + 1, height36);
        int wend57 = min(wstart55 + (kernel_w41 - 1) * dilation_w47 + 1, width37);
        while (hstart54 < 0)
            hstart54 += dilation_h46;
        while (wstart55 < 0)
            wstart55 += dilation_w47;
        accscalar_t31 maxval58 = at::numeric_limits<accscalar_t31>::lower_bound();
        int maxidx59 = hstart54 * width37 + wstart55;
        const scalar_t30 *btm_data60 = bottom_data33 + (n53 * channels35 + c52) * height36 * width37;
        for (int h = hstart54; h < hend56; h += dilation_h46) {
            for (int w = wstart55; w < wend57; w += dilation_w47) {
                scalar_t30 val61 = btm_data60[h * width37 + w];
                if ((ScalarConvert<scalar_t30, accscalar_t31>::to(val61) > maxval58) || THCNumerics<scalar_t30>::isnan(val61)) {
                    maxidx59 = h * width37 + w;
                    maxval58 = ScalarConvert<scalar_t30, accscalar_t31>::to(val61);
                }
            }
        }
        top_data48[index] = ScalarConvert<scalar_t30, accscalar_t31>::to(maxval58);
        top_mask49[index] = maxidx59;
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=128 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 768)){
    unsigned int blockDim_x_0 = 640;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) % 640;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) / 640 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 128) / 640;
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
template <typename scalar_t30, typename accscalar_t31, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(768, 2) void MaxPoolForward_upsample_bilinear2d_out_frame_fused_kernel_hfuse_lb_idx_1(const int nthreads32, const scalar_t30 *bottom_data33, const int num34, const int channels35, const int height36, const int width37, const int pooled_height38, const int pooled_width39, const int kernel_h40, const int kernel_w41, const int stride_h42, const int stride_w43, const int pad_h44, const int pad_w45, const int dilation_h46, const int dilation_w47, scalar_t30 *top_data48, int64_t *top_mask49, const int ns2, const accscalar_t1 rheight3, const accscalar_t1 rwidth4, const bool align_corners5, const PackedTensorAccessor<scalar_t0, 4> idata6, PackedTensorAccessor<scalar_t0, 4> odata7)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 256)){
    unsigned int blockDim_x_1 = 256;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 256;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 256;
    for (int index = blockIdx.x * blockDim_x_1 + threadIdx_x_1; index < (nthreads32); index += blockDim_x_1 * gridDim.x) {
        int pw50 = index % pooled_width39;
        int ph51 = (index / pooled_width39) % pooled_height38;
        int c52 = (index / pooled_width39 / pooled_height38) % channels35;
        int n53 = index / pooled_width39 / pooled_height38 / channels35;
        int hstart54 = ph51 * stride_h42 - pad_h44;
        int wstart55 = pw50 * stride_w43 - pad_w45;
        int hend56 = min(hstart54 + (kernel_h40 - 1) * dilation_h46 + 1, height36);
        int wend57 = min(wstart55 + (kernel_w41 - 1) * dilation_w47 + 1, width37);
        while (hstart54 < 0)
            hstart54 += dilation_h46;
        while (wstart55 < 0)
            wstart55 += dilation_w47;
        accscalar_t31 maxval58 = at::numeric_limits<accscalar_t31>::lower_bound();
        int maxidx59 = hstart54 * width37 + wstart55;
        const scalar_t30 *btm_data60 = bottom_data33 + (n53 * channels35 + c52) * height36 * width37;
        for (int h = hstart54; h < hend56; h += dilation_h46) {
            for (int w = wstart55; w < wend57; w += dilation_w47) {
                scalar_t30 val61 = btm_data60[h * width37 + w];
                if ((ScalarConvert<scalar_t30, accscalar_t31>::to(val61) > maxval58) || THCNumerics<scalar_t30>::isnan(val61)) {
                    maxidx59 = h * width37 + w;
                    maxval58 = ScalarConvert<scalar_t30, accscalar_t31>::to(val61);
                }
            }
        }
        top_data48[index] = ScalarConvert<scalar_t30, accscalar_t31>::to(maxval58);
        top_mask49[index] = maxidx59;
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=256 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 768)){
    unsigned int blockDim_x_0 = 512;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 256) % 512;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 256) / 512 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 256) / 512;
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
template <typename scalar_t30, typename accscalar_t31, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(768, 2) void MaxPoolForward_upsample_bilinear2d_out_frame_fused_kernel_hfuse_lb_idx_2(const int nthreads32, const scalar_t30 *bottom_data33, const int num34, const int channels35, const int height36, const int width37, const int pooled_height38, const int pooled_width39, const int kernel_h40, const int kernel_w41, const int stride_h42, const int stride_w43, const int pad_h44, const int pad_w45, const int dilation_h46, const int dilation_w47, scalar_t30 *top_data48, int64_t *top_mask49, const int ns2, const accscalar_t1 rheight3, const accscalar_t1 rwidth4, const bool align_corners5, const PackedTensorAccessor<scalar_t0, 4> idata6, PackedTensorAccessor<scalar_t0, 4> odata7)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 384)){
    unsigned int blockDim_x_1 = 384;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 384;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 384 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 384;
    for (int index = blockIdx.x * blockDim_x_1 + threadIdx_x_1; index < (nthreads32); index += blockDim_x_1 * gridDim.x) {
        int pw50 = index % pooled_width39;
        int ph51 = (index / pooled_width39) % pooled_height38;
        int c52 = (index / pooled_width39 / pooled_height38) % channels35;
        int n53 = index / pooled_width39 / pooled_height38 / channels35;
        int hstart54 = ph51 * stride_h42 - pad_h44;
        int wstart55 = pw50 * stride_w43 - pad_w45;
        int hend56 = min(hstart54 + (kernel_h40 - 1) * dilation_h46 + 1, height36);
        int wend57 = min(wstart55 + (kernel_w41 - 1) * dilation_w47 + 1, width37);
        while (hstart54 < 0)
            hstart54 += dilation_h46;
        while (wstart55 < 0)
            wstart55 += dilation_w47;
        accscalar_t31 maxval58 = at::numeric_limits<accscalar_t31>::lower_bound();
        int maxidx59 = hstart54 * width37 + wstart55;
        const scalar_t30 *btm_data60 = bottom_data33 + (n53 * channels35 + c52) * height36 * width37;
        for (int h = hstart54; h < hend56; h += dilation_h46) {
            for (int w = wstart55; w < wend57; w += dilation_w47) {
                scalar_t30 val61 = btm_data60[h * width37 + w];
                if ((ScalarConvert<scalar_t30, accscalar_t31>::to(val61) > maxval58) || THCNumerics<scalar_t30>::isnan(val61)) {
                    maxidx59 = h * width37 + w;
                    maxval58 = ScalarConvert<scalar_t30, accscalar_t31>::to(val61);
                }
            }
        }
        top_data48[index] = ScalarConvert<scalar_t30, accscalar_t31>::to(maxval58);
        top_mask49[index] = maxidx59;
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=384 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 768)){
    unsigned int blockDim_x_0 = 384;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 384) % 384;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 384) / 384 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 384) / 384;
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
template <typename scalar_t30, typename accscalar_t31, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(768, 2) void MaxPoolForward_upsample_bilinear2d_out_frame_fused_kernel_hfuse_lb_idx_3(const int nthreads32, const scalar_t30 *bottom_data33, const int num34, const int channels35, const int height36, const int width37, const int pooled_height38, const int pooled_width39, const int kernel_h40, const int kernel_w41, const int stride_h42, const int stride_w43, const int pad_h44, const int pad_w45, const int dilation_h46, const int dilation_w47, scalar_t30 *top_data48, int64_t *top_mask49, const int ns2, const accscalar_t1 rheight3, const accscalar_t1 rwidth4, const bool align_corners5, const PackedTensorAccessor<scalar_t0, 4> idata6, PackedTensorAccessor<scalar_t0, 4> odata7)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 512)){
    unsigned int blockDim_x_1 = 512;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 512;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 512;
    for (int index = blockIdx.x * blockDim_x_1 + threadIdx_x_1; index < (nthreads32); index += blockDim_x_1 * gridDim.x) {
        int pw50 = index % pooled_width39;
        int ph51 = (index / pooled_width39) % pooled_height38;
        int c52 = (index / pooled_width39 / pooled_height38) % channels35;
        int n53 = index / pooled_width39 / pooled_height38 / channels35;
        int hstart54 = ph51 * stride_h42 - pad_h44;
        int wstart55 = pw50 * stride_w43 - pad_w45;
        int hend56 = min(hstart54 + (kernel_h40 - 1) * dilation_h46 + 1, height36);
        int wend57 = min(wstart55 + (kernel_w41 - 1) * dilation_w47 + 1, width37);
        while (hstart54 < 0)
            hstart54 += dilation_h46;
        while (wstart55 < 0)
            wstart55 += dilation_w47;
        accscalar_t31 maxval58 = at::numeric_limits<accscalar_t31>::lower_bound();
        int maxidx59 = hstart54 * width37 + wstart55;
        const scalar_t30 *btm_data60 = bottom_data33 + (n53 * channels35 + c52) * height36 * width37;
        for (int h = hstart54; h < hend56; h += dilation_h46) {
            for (int w = wstart55; w < wend57; w += dilation_w47) {
                scalar_t30 val61 = btm_data60[h * width37 + w];
                if ((ScalarConvert<scalar_t30, accscalar_t31>::to(val61) > maxval58) || THCNumerics<scalar_t30>::isnan(val61)) {
                    maxidx59 = h * width37 + w;
                    maxval58 = ScalarConvert<scalar_t30, accscalar_t31>::to(val61);
                }
            }
        }
        top_data48[index] = ScalarConvert<scalar_t30, accscalar_t31>::to(maxval58);
        top_mask49[index] = maxidx59;
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=512 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 768)){
    unsigned int blockDim_x_0 = 256;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) % 256;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 256 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 512) / 256;
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
template <typename scalar_t30, typename accscalar_t31, typename scalar_t0, typename accscalar_t1>
 __global__ __launch_bounds__(768, 2) void MaxPoolForward_upsample_bilinear2d_out_frame_fused_kernel_hfuse_lb_idx_4(const int nthreads32, const scalar_t30 *bottom_data33, const int num34, const int channels35, const int height36, const int width37, const int pooled_height38, const int pooled_width39, const int kernel_h40, const int kernel_w41, const int stride_h42, const int stride_w43, const int pad_h44, const int pad_w45, const int dilation_h46, const int dilation_w47, scalar_t30 *top_data48, int64_t *top_mask49, const int ns2, const accscalar_t1 rheight3, const accscalar_t1 rwidth4, const bool align_corners5, const PackedTensorAccessor<scalar_t0, 4> idata6, PackedTensorAccessor<scalar_t0, 4> odata7)
 {
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=0 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 640)){
    unsigned int blockDim_x_1 = 640;
    unsigned int threadIdx_x_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) % 640;
    unsigned int blockDim_y_1 = 1;
    unsigned int threadIdx_y_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 640 % 1;
    unsigned int blockDim_z_1 = 1;
    unsigned int threadIdx_z_1 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 0) / 640;
    for (int index = blockIdx.x * blockDim_x_1 + threadIdx_x_1; index < (nthreads32); index += blockDim_x_1 * gridDim.x) {
        int pw50 = index % pooled_width39;
        int ph51 = (index / pooled_width39) % pooled_height38;
        int c52 = (index / pooled_width39 / pooled_height38) % channels35;
        int n53 = index / pooled_width39 / pooled_height38 / channels35;
        int hstart54 = ph51 * stride_h42 - pad_h44;
        int wstart55 = pw50 * stride_w43 - pad_w45;
        int hend56 = min(hstart54 + (kernel_h40 - 1) * dilation_h46 + 1, height36);
        int wend57 = min(wstart55 + (kernel_w41 - 1) * dilation_w47 + 1, width37);
        while (hstart54 < 0)
            hstart54 += dilation_h46;
        while (wstart55 < 0)
            wstart55 += dilation_w47;
        accscalar_t31 maxval58 = at::numeric_limits<accscalar_t31>::lower_bound();
        int maxidx59 = hstart54 * width37 + wstart55;
        const scalar_t30 *btm_data60 = bottom_data33 + (n53 * channels35 + c52) * height36 * width37;
        for (int h = hstart54; h < hend56; h += dilation_h46) {
            for (int w = wstart55; w < wend57; w += dilation_w47) {
                scalar_t30 val61 = btm_data60[h * width37 + w];
                if ((ScalarConvert<scalar_t30, accscalar_t31>::to(val61) > maxval58) || THCNumerics<scalar_t30>::isnan(val61)) {
                    maxidx59 = h * width37 + w;
                    maxval58 = ScalarConvert<scalar_t30, accscalar_t31>::to(val61);
                }
            }
        }
        top_data48[index] = ScalarConvert<scalar_t30, accscalar_t31>::to(maxval58);
        top_mask49[index] = maxidx59;
    }
}
if (((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y)>=640 && (threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) < 768)){
    unsigned int blockDim_x_0 = 128;
    unsigned int threadIdx_x_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 640) % 128;
    unsigned int blockDim_y_0 = 1;
    unsigned int threadIdx_y_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 640) / 128 % 1;
    unsigned int blockDim_z_0 = 1;
    unsigned int threadIdx_z_0 = ((threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y) - 640) / 128;
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
