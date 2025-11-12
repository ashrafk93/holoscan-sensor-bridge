// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "image_decoder.hpp"

#include <hololink/core/logging_internal.hpp>
#include <hololink/common/cuda_helper.hpp>
#include <holoscan/holoscan.hpp>
#include <holoscan/utils/cuda_stream_handler.hpp>

namespace {
const char* source = R"(
extern "C" {

typedef unsigned char  uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;


typedef struct {
    int width;
    int height;
    float ppx;
    float ppy;
    float fx;
    float fy;
    float model; // not used
    float coeffs[5]; // not used
} rs2_intrinsics;

typedef struct {
    float rotation[9];
    float translation[3];
} rs2_extrinsics;


__global__ void frameReconstructionZ16(unsigned short* out,
                                       const unsigned char* in,
                                       int per_line_size,
                                       int width,
                                       int height)
{
    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
    if ((idx_x >= width) || (idx_y >= height)) return;
    int out_index = idx_y * width + idx_x;
    int in_index = (per_line_size * idx_y) + idx_x * 2;
    unsigned short val = static_cast<unsigned short>(in[in_index]) |
                         (static_cast<unsigned short>(in[in_index + 1]) << 8);
    out[out_index] = val;
}

__global__ void frameReconstructionYUYV(uint8_t* out_rgb,
                                        const uint8_t* in_yuyv,
                                        int per_line_size,
                                        int width,
                                        int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if ((x >= width) || (y >= height)) return;
    int pixel_pair_idx = x / 2;
    int in_idx = y * per_line_size + pixel_pair_idx * 4;
    uint8_t Y0 = in_yuyv[in_idx + 0];
    uint8_t U  = in_yuyv[in_idx + 1];
    uint8_t Y1 = in_yuyv[in_idx + 2];
    uint8_t V  = in_yuyv[in_idx + 3];
    int c = x % 2;
    uint8_t Y = (c == 0) ? Y0 : Y1;
    int C = Y - 16, D = U - 128, E = V - 128;
    int R = (298 * C + 409 * E + 128) >> 8;
    int G = (298 * C - 100 * D - 208 * E + 128) >> 8;
    int B = (298 * C + 516 * D + 128) >> 8;
    R = R < 0 ? 0 : (R > 255 ? 255 : R);
    G = G < 0 ? 0 : (G > 255 ? 255 : G);
    B = B < 0 ? 0 : (B > 255 ? 255 : B);
    int out_idx = (y * width + x) * 3;
    out_rgb[out_idx + 0] = R;
    out_rgb[out_idx + 1] = G;
    out_rgb[out_idx + 2] = B;

}

__global__ void compute_histogram(const uint16_t* depth, int* hist, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    uint16_t d = depth[idx];
    if (d > 0 && d < 65536) atomicAdd(&hist[d], 1);
}

__global__ void prefix_sum_histogram(int* hist, int size) {
    for (int i = 1; i < size; ++i) {
        hist[i] += hist[i - 1];
    }
}

__device__ inline float3 interpolate_colormap(float value, const float3* colormap, int colormap_size) {
    float t = fminf(fmaxf(value, 0.f), 1.f) * (colormap_size - 1);
    int idx = (int)t;
    float frac = t - idx;
    float3 lo = colormap[idx];
    float3 hi = colormap[min(idx + 1, colormap_size - 1)];
    return make_float3(lo.x * (1.f - frac) + hi.x * frac,
                       lo.y * (1.f - frac) + hi.y * frac,
                       lo.z * (1.f - frac) + hi.z * frac);
}

__global__ void depthToRGB(uint8_t* out_rgb,
                           const uint16_t* depth,
                           const int* hist,
                           int width,
                           int height,
                           float depth_units,
                           float min_m,
                           float max_m,
                           bool equalize,
                           const float3* colormap,
                           int colormap_size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    uint16_t d = depth[idx];

    if (d == 0) {
        out_rgb[3 * idx + 0] = 0;
        out_rgb[3 * idx + 1] = 0;
        out_rgb[3 * idx + 2] = 0;
        return;
    }

    float norm;
    if (equalize) {
        int total_hist = hist[65535];
        norm = (total_hist > 0) ? (float)(hist[d]) / total_hist : 0.f;
    } else {
        float depth_m = d * depth_units;
        norm = (depth_m - min_m) / (max_m - min_m);
        norm = fminf(fmaxf(norm, 0.f), 1.f);
    }

    float3 c = interpolate_colormap(norm, colormap, colormap_size);
    out_rgb[3 * idx + 0] = (uint8_t)(c.x);
    out_rgb[3 * idx + 1] = (uint8_t)(c.y);
    out_rgb[3 * idx + 2] = (uint8_t)(c.z);
}

__global__ void projectDepthToRGB(uint32_t* aligned_depth,
                                  const uint16_t* raw_depth,
                                  const rs2_intrinsics* depth_intr,
                                  const rs2_intrinsics* rgb_intr,
                                  const rs2_extrinsics* depth_to_rgb,
                                  int width,
                                  int height,
                                  float depth_scale) {
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;
    if (dx >= width || dy >= height) return;

    int depth_idx = dy * width + dx;
    uint16_t d = raw_depth[depth_idx];
    if (d == 0) return;

    float z = d * depth_scale;
    float x = (dx - depth_intr->ppx) / depth_intr->fx * z;
    float y = (dy - depth_intr->ppy) / depth_intr->fy * z;

    float rx = depth_to_rgb->rotation[0] * x + depth_to_rgb->rotation[3] * y + depth_to_rgb->rotation[6] * z + depth_to_rgb->translation[0];
    float ry = depth_to_rgb->rotation[1] * x + depth_to_rgb->rotation[4] * y + depth_to_rgb->rotation[7] * z + depth_to_rgb->translation[1];
    float rz = depth_to_rgb->rotation[2] * x + depth_to_rgb->rotation[5] * y + depth_to_rgb->rotation[8] * z + depth_to_rgb->translation[2];

    if (rz <= 0.f) return;

    int rx_pixel = static_cast<int>((rx / rz) * rgb_intr->fx + rgb_intr->ppx);
    int ry_pixel = static_cast<int>((ry / rz) * rgb_intr->fy + rgb_intr->ppy);

    if (rx_pixel < 0 || rx_pixel >= rgb_intr->width || ry_pixel < 0 || ry_pixel >= rgb_intr->height) return;

    int rgb_idx = ry_pixel * rgb_intr->width + rx_pixel;

    atomicMin(&aligned_depth[rgb_idx], static_cast<uint32_t>(d));
}


__global__ void colorizeAlignedDepth(uint8_t* out_rgb,
                                     const uint32_t* aligned_depth,
                                     int width,
                                     int height,
                                     float depth_scale,
                                     float min_m,
                                     float max_m,
                                     const float3* colormap,
                                     int colormap_size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    uint16_t d = static_cast<uint16_t>(aligned_depth[idx]);
    if (aligned_depth[idx] == 0xFFFFFFFF || d == 0) {
        out_rgb[3 * idx + 0] = 0;
        out_rgb[3 * idx + 1] = 0;
        out_rgb[3 * idx + 2] = 0;
        return;
    }

    float depth_m = d * depth_scale;
    float norm = (depth_m - min_m) / (max_m - min_m);
    norm = fminf(fmaxf(norm, 0.f), 1.f);
    float3 c = interpolate_colormap(norm, colormap, colormap_size);
    out_rgb[3 * idx + 0] = (uint8_t)c.x;
    out_rgb[3 * idx + 1] = (uint8_t)c.y;
    out_rgb[3 * idx + 2] = (uint8_t)c.z;
}

}
)";
} // namespace

namespace hololink::operators {

void ImageDecoder::setup(holoscan::OperatorSpec& spec) {
    spec.input<holoscan::gxf::Entity>("input");
    spec.output<holoscan::gxf::Entity>("output");
    spec.param(allocator_, "allocator", "Allocator", "Memory allocator");
    spec.param(cuda_device_ordinal_, "cuda_device_ordinal", "CudaDeviceOrdinal", "CUDA device");
    spec.param(out_tensor_name_, "out_tensor_name", "OutputTensorName", "Name of output tensor");
    spec.param(align_depth_to_rgb, "align_depth_to_rgb", "AlignDepthToRGB", "Align Depth to RGB");
    cuda_stream_handler_.define_params(spec);
}

void ImageDecoder::start() {
    if (pixel_format_ == PixelFormat::INVALID) throw std::runtime_error("Decoder not configured");
    CudaCheck(cuInit(0));
    CudaCheck(cuDeviceGet(&cuda_device_, cuda_device_ordinal_.get()));
    CudaCheck(cuDevicePrimaryCtxRetain(&cuda_context_, cuda_device_));
    hololink::common::CudaContextScopedPush cur_cuda_context(cuda_context_);
    cuda_function_launcher_.reset(new hololink::common::CudaFunctionLauncher(
        source, {"frameReconstructionZ16", "frameReconstructionYUYV","depthToRGB", "compute_histogram",
             "prefix_sum_histogram", "projectDepthToRGB", "colorizeAlignedDepth"}));
    
    // Allocate d_hist_ (256KB)
    cudaMalloc(&d_hist_, sizeof(int) * 0x10000);

    // Allocate and upload colormap
    std::vector<float3> colormap = {
        {0.f, 0.f, 255.f},    // Blue
        {0.f, 255.f, 255.f},  // Cyan
        {255.f, 255.f, 0.f},  // Yellow
        {255.f, 0.f, 0.f},    // Red
        {50.f, 0.f, 0.f}      // Dark red
    };
    colormap_size_ = colormap.size();
    cudaMalloc(&d_colormap_, colormap_size_ * sizeof(float3));
    cudaMemcpy(d_colormap_, colormap.data(), colormap_size_ * sizeof(float3), cudaMemcpyHostToDevice);

   // Depth intrinsics (original 1280x800 values)
    const float fx_d_orig = 642.30053711f;
    const float fy_d_orig = 642.30053711f;
    const float ppx_d_orig = 644.71191406f;
    const float ppy_d_orig = 396.334198f;

    // RGB intrinsics (original 1280x800 values)
    const float fx_rgb_orig = 639.31732178f;
    const float fy_rgb_orig = 637.15985107f;
    const float ppx_rgb_orig = 636.95349121f;
    const float ppy_rgb_orig = 402.9045105f;

    const int orig_width = 1280;
    const int orig_height = 800;
    const int new_width = width_;
    const int new_height = height_;
    const float scale_ratio = std::max(
        float(new_width) / orig_width, float(new_height) / orig_height); // 0.5

    const float crop_x = (orig_width * scale_ratio - new_width) * 0.5f;  // 0
    const float crop_y = (orig_height * scale_ratio - new_height) * 0.5f; // 20

    // Depth intrinsics (scaled to widthxheight)
    h_depth_intrin_.width  = new_width;
    h_depth_intrin_.height = new_height;
    h_depth_intrin_.fx  = fx_d_orig * scale_ratio;  // 321.15027
    h_depth_intrin_.fy  = fy_d_orig * scale_ratio;  // 321.15027
    h_depth_intrin_.ppx = (ppx_d_orig + 0.5f) * scale_ratio - crop_x - 0.5f;  // 322.35596
    h_depth_intrin_.ppy = (ppy_d_orig + 0.5f) * scale_ratio - crop_y - 1.0f;  // 178.35039

    // RGB intrinsics (scaled to widthxheight)
    h_rgb_intrin_.width  = new_width;
    h_rgb_intrin_.height = new_height;
    h_rgb_intrin_.fx  = fx_rgb_orig * scale_ratio * 0.98;  // 319.65866
    h_rgb_intrin_.fy  = fy_rgb_orig * scale_ratio;  // 318.57993
    h_rgb_intrin_.ppx = (ppx_rgb_orig + 0.5f) * scale_ratio - crop_x - 2.5f; // 313.32538
    h_rgb_intrin_.ppy = (ppy_rgb_orig + 0.5f) * scale_ratio - crop_y - 8.5f; // 177.434

    // Extrinsics (no change, keep in meters)
    // Transpose of rotation
    float rot_inv[] = {
        0.99999851f, -0.00101535f, -0.00141438f,
        0.00101646f,  0.99999917f,  0.00078718f,
        0.00141358f, -0.00078861f,  0.99999869f
    };

    // T_inv = -Rᵗ * T
    float tx = 0.05817929f;
    float ty = 0.00017164f;
    float tz = 0.00025427f;
    float trans_inv[] = {
        -(rot_inv[0] * tx + rot_inv[1] * ty + rot_inv[2] * tz),
        -(rot_inv[3] * tx + rot_inv[4] * ty + rot_inv[5] * tz),
        -(rot_inv[6] * tx + rot_inv[7] * ty + rot_inv[8] * tz)
    };

    memcpy(h_extrinsics_.rotation, rot_inv, sizeof(rot_inv));
    memcpy(h_extrinsics_.translation, trans_inv, sizeof(trans_inv));

    // Allocate device memory and copy to GPU
    cudaMalloc(&d_depth_intrinsics_, sizeof(rs2_intrinsics));
    cudaMalloc(&d_rgb_intrinsics_, sizeof(rs2_intrinsics));
    cudaMalloc(&d_depth_to_rgb_extrinsics_, sizeof(rs2_extrinsics));

    cudaMemcpy(d_depth_intrinsics_, &h_depth_intrin_, sizeof(rs2_intrinsics), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rgb_intrinsics_, &h_rgb_intrin_, sizeof(rs2_intrinsics), cudaMemcpyHostToDevice);
    cudaMemcpy(d_depth_to_rgb_extrinsics_, &h_extrinsics_, sizeof(rs2_extrinsics), cudaMemcpyHostToDevice);

    cudaMalloc(&d_aligned_depth_, sizeof(uint32_t) * h_rgb_intrin_.width * h_rgb_intrin_.height);
}

void ImageDecoder::stop() {
    hololink::common::CudaContextScopedPush cur_cuda_context(cuda_context_);
    cuda_function_launcher_.reset();

    if (d_hist_) {
        cudaFree(d_hist_);
        d_hist_ = nullptr;
    }

    if (d_colormap_) {
        cudaFree(d_colormap_);
        d_colormap_ = nullptr;
        colormap_size_ = 0;
    }
    if (d_depth_intrinsics_) {
        cudaFree(d_depth_intrinsics_);
        d_depth_intrinsics_ = nullptr;
    }
    if (d_rgb_intrinsics_) {
        cudaFree(d_rgb_intrinsics_);
        d_rgb_intrinsics_ = nullptr;
    }
    if (d_depth_to_rgb_extrinsics_) {
        cudaFree(d_depth_to_rgb_extrinsics_);
        d_depth_to_rgb_extrinsics_ = nullptr;
    }
    if (d_aligned_depth_) {
        cudaFree(d_aligned_depth_);
        d_aligned_depth_ = nullptr;
    }

    CudaCheck(cuDevicePrimaryCtxRelease(cuda_device_));
    cuda_context_ = nullptr;
}

void ImageDecoder::compute(holoscan::InputContext& input, holoscan::OutputContext& output,
                           holoscan::ExecutionContext& context) {
    auto maybe_entity = input.receive<holoscan::gxf::Entity>("input");
    if (!maybe_entity) throw std::runtime_error("No input entity");
    auto& entity = static_cast<nvidia::gxf::Entity&>(maybe_entity.value());
    gxf_result_t stream_handler_result = cuda_stream_handler_.from_message(context.context(), entity);
    if (stream_handler_result != GXF_SUCCESS) throw std::runtime_error("Failed to get stream");
    auto input_tensor = entity.get<nvidia::gxf::Tensor>().value();

    if (input_tensor->storage_type() == nvidia::gxf::MemoryStorageType::kHost) {
        if (!is_integrated_ && !host_memory_warning_) {
            host_memory_warning_ = true;
            HSB_LOG_WARN(
                "The input tensor is stored in host memory, this will reduce performance of this "
                "operator. For best performance store the input tensor in device memory.");
        }
    } else if (input_tensor->storage_type() != nvidia::gxf::MemoryStorageType::kDevice) {
        throw std::runtime_error(
            fmt::format("Unsupported storage type {}", (int)input_tensor->storage_type()));
    }

    if (input_tensor->rank() != 1) throw std::runtime_error("Tensor must be 1D");

    const int32_t size = input_tensor->shape().dimension(0);
    auto allocator = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(
        fragment()->executor().context(), allocator_->gxf_cid());
    const uint32_t per_line_size = line_start_size_ + bytes_per_line_ + line_end_size_;

    switch (pixel_format_) {
    case PixelFormat::Z16: {
        // 1. Allocate depth tensor (device)
        nvidia::gxf::Shape depth_shape{int(height_), int(width_), 1};
        auto depth_message = CreateTensorMap(context.context(), allocator.value(), {{
            "depth", nvidia::gxf::MemoryStorageType::kDevice, depth_shape,
            nvidia::gxf::PrimitiveType::kUnsigned16, 0,
            nvidia::gxf::ComputeTrivialStrides(depth_shape, 2)}}, false);
        auto depth_tensor = depth_message.value().get<nvidia::gxf::Tensor>("depth");

        // 2. Reconstruct depth frame from CSI
        cuda_function_launcher_->launch("frameReconstructionZ16", {width_, height_, 1},
            cuda_stream_handler_.get_cuda_stream(context.context()),
            depth_tensor.value()->pointer(),
            input_tensor->pointer() + frame_start_size_ + line_start_size_,
            per_line_size, width_, height_);

        // 3. Allocate RGB tensor (device)
        nvidia::gxf::Shape rgb_shape{int(height_), int(width_), 3};
        auto out_message = align_depth_to_rgb.get()
            ? CreateTensorMap(context.context(), allocator.value(), {
                {
                    out_tensor_name_.get(), nvidia::gxf::MemoryStorageType::kDevice, rgb_shape,
                    nvidia::gxf::PrimitiveType::kUnsigned8, 0,
                    nvidia::gxf::ComputeTrivialStrides(rgb_shape, 1)
                },
                {
                    "depth_raw", nvidia::gxf::MemoryStorageType::kDevice, depth_shape,
                    nvidia::gxf::PrimitiveType::kUnsigned16, 0,
                    nvidia::gxf::ComputeTrivialStrides(depth_shape,
                    nvidia::gxf::PrimitiveTypeSize(nvidia::gxf::PrimitiveType::kUnsigned16))
                },
            }, false)
            : CreateTensorMap(context.context(), allocator.value(), {
                {
                    out_tensor_name_.get(), nvidia::gxf::MemoryStorageType::kDevice, rgb_shape,
                    nvidia::gxf::PrimitiveType::kUnsigned8, 0,
                    nvidia::gxf::ComputeTrivialStrides(rgb_shape, 1)
                },
            }, false);
        auto rgb_tensor = out_message.value().get<nvidia::gxf::Tensor>(out_tensor_name_.get().c_str());

        // fill depth_raw sensor
        if (align_depth_to_rgb.get()) {
            auto raw_tensor = out_message.value().get<nvidia::gxf::Tensor>("depth_raw");
            cudaMemcpyAsync(raw_tensor.value()->pointer(), depth_tensor.value()->pointer(),
                            sizeof(uint16_t) * width_ * height_,
                            cudaMemcpyDeviceToDevice,
                            cuda_stream_handler_.get_cuda_stream(context.context()));
        }

        // 4. GPU histogram
        cudaMemsetAsync(d_hist_, 0, sizeof(int) * 0x10000, cuda_stream_handler_.get_cuda_stream(context.context()));

        cuda_function_launcher_->launch("compute_histogram", {width_, height_, 1},
            cuda_stream_handler_.get_cuda_stream(context.context()),
            depth_tensor.value()->pointer(), d_hist_, width_ * height_);

        cuda_function_launcher_->launch("prefix_sum_histogram", {1, 1, 1},
            cuda_stream_handler_.get_cuda_stream(context.context()),
            d_hist_,
            0x10000);

        // 5. Run GPU depth → RGB colorizer
        if (align_depth_to_rgb.get()) {
            cudaMemsetAsync(d_aligned_depth_, 0xFF, sizeof(uint32_t) * width_ * height_, cuda_stream_handler_.get_cuda_stream(context.context()));

            cuda_function_launcher_->launch("projectDepthToRGB", {width_, height_, 1},
                cuda_stream_handler_.get_cuda_stream(context.context()),
                d_aligned_depth_,
                depth_tensor.value()->pointer(),
                d_depth_intrinsics_,
                d_rgb_intrinsics_,
                d_depth_to_rgb_extrinsics_,
                width_, height_, 0.001f);

            cuda_function_launcher_->launch("colorizeAlignedDepth", {width_, height_, 1},
                cuda_stream_handler_.get_cuda_stream(context.context()),
                rgb_tensor.value()->pointer(),
                d_aligned_depth_,
                width_, height_,
                0.001f, 0.3f, 4.0f,
                d_colormap_, colormap_size_);

            // std::vector<uint32_t> aligned_preview(width_ * height_);
            // cudaMemcpy(aligned_preview.data(), d_aligned_depth_, sizeof(uint32_t) * width_ * height_, cudaMemcpyDeviceToHost);

            // int valid_count = 0;
            // int min_d = 0xFFFF, max_d = 0;
            // for (auto d : aligned_preview) {
            //     if (d != 0xFFFFFFFF) {
            //         valid_count++;
            //         if (d < min_d) min_d = d;
            //         if (d > max_d) max_d = d;
            //     }
            // }
            // HSB_LOG_INFO("Aligned depth: valid={}  min={}  max={}", valid_count, min_d, max_d);

        } else {
            cuda_function_launcher_->launch("depthToRGB", {width_, height_, 1},
                cuda_stream_handler_.get_cuda_stream(context.context()),
                rgb_tensor.value()->pointer(),
                depth_tensor.value()->pointer(),
                d_hist_,
                width_, height_,
                0.001f,
                0.3f, 4.0f,
                false,
                d_colormap_,
                static_cast<int>(colormap_size_));
        }


        // 6. Emit output
        stream_handler_result = cuda_stream_handler_.to_message(out_message);
        if (stream_handler_result != GXF_SUCCESS) {
            throw std::runtime_error("Failed to emit RGB image");
        }
        // auto& out_entity = out_message.value();
        auto out_entity = holoscan::gxf::Entity(std::move(out_message.value()));
        output.emit(out_entity);
        return;
    }
    case PixelFormat::YUYV: {
        nvidia::gxf::Shape shape{int(height_), int(width_), 3};
        auto out_message = CreateTensorMap(context.context(), allocator.value(), {{
            out_tensor_name_.get(), nvidia::gxf::MemoryStorageType::kDevice, shape,
            nvidia::gxf::PrimitiveType::kUnsigned8, 0,
            nvidia::gxf::ComputeTrivialStrides(shape, 1)}}, false);
        auto tensor = out_message.value().get<nvidia::gxf::Tensor>(out_tensor_name_.get().c_str());
        cuda_function_launcher_->launch("frameReconstructionYUYV", {width_, height_, 1},
            cuda_stream_handler_.get_cuda_stream(context.context()),
            tensor.value()->pointer(),
            input_tensor->pointer() + frame_start_size_ + line_start_size_,
            per_line_size, width_, height_);
        stream_handler_result = cuda_stream_handler_.to_message(out_message);
        auto out_entity = holoscan::gxf::Entity(std::move(out_message.value()));
        output.emit(out_entity);
        return;
    }
    default:
        throw std::runtime_error("Unsupported pixel format");
    }
}

void ImageDecoder::configure(uint32_t width, uint32_t height, PixelFormat pixel_format,
                             uint32_t frame_start_size, uint32_t frame_end_size,
                             uint32_t line_start_size, uint32_t line_end_size,
                             uint32_t margin_left, uint32_t margin_top,
                             uint32_t margin_right, uint32_t margin_bottom) {
    width_ = width;
    height_ = height;
    pixel_format_ = pixel_format;
    frame_start_size_ = frame_start_size;
    frame_end_size_ = frame_end_size;
    line_start_size_ = line_start_size;
    line_end_size_ = line_end_size;
    switch (pixel_format_) {
    case PixelFormat::Z16:
        bytes_per_line_ = width * 2;
        line_start_size_ += margin_left * 2;
        line_end_size_ += margin_right * 2;
        break;
    case PixelFormat::YUYV:
        bytes_per_line_ = width * 2;
        line_start_size_ += margin_left * 2;
        line_end_size_ += margin_right * 2;
        break;
    default:
        throw std::runtime_error("Unsupported pixel format");
    }
    const uint32_t line_size = line_start_size_ + bytes_per_line_ + line_end_size_;
    frame_start_size_ += margin_top * line_size;
    frame_end_size_ += margin_bottom * line_size;
    csi_length_ = (frame_start_size_ + line_size * height_ + frame_end_size_ + 7) & ~7;
}

size_t ImageDecoder::get_csi_length() {
    if (pixel_format_ == PixelFormat::INVALID) {
        throw std::runtime_error("ImageDecoder is not configured.");
    }
    return csi_length_;
}

} // namespace hololink::operators