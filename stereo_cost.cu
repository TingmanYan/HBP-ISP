#include "stereo_cost.hpp"

#define CENSUS_WIDTH 9
#define CENSUS_HEIGHT 7
#define TOP (CENSUS_HEIGHT - 1) / 2
#define LEFT (CENSUS_WIDTH - 1) / 2
#define CENSUS_SIZE 63

__global__ void
ad_cost(const float *const imgL_f1_d, const float *const imgR_f1_d,
        float *const cost_d, const int width, const int height,
        const int labels, const float tau)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    int index = y * width + x;
    int im_size = width * height;

    if (x < labels - 1)
    {
        // TODO: other process methods for this border
        //       i.e. fill of volume
        for (int label = 0; label < labels; ++label)
            cost_d[im_size * label + index] = 0;
    }
    else
    {
        float ref = imgL_f1_d[index];
        // __syncthreads();
        for (int label = 0; label < labels; ++label)
        {
            float diff = ref - imgR_f1_d[index - label];
            cost_d[im_size * label + index] = fminf(fabsf(diff), tau);
            // TODO: try if this __syncthreads help
            // for global memory coalescing
            // the answer is no, there has no affects to runtime
            // __syncthreads();
        }
    }
}

__global__ void __launch_bounds__(1024, 2)
    CenterSymmetricCensusKernelSM2(const uint8_t *im, const uint8_t *im2,
                                   uint32_t *transform, uint32_t *transform2,
                                   const uint32_t rows, const uint32_t cols)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;

    const int win_cols = (32 + LEFT * 2); // 32+4*2 = 40
    const int win_rows = (32 + TOP * 2);  // 32+3*2 = 38

    __shared__ uint8_t window[win_cols * win_rows];
    __shared__ uint8_t window2[win_cols * win_rows];

    const int id = threadIdx.y * blockDim.x + threadIdx.x;
    const int sm_row = id / win_cols;
    const int sm_col = id % win_cols;

    const int im_row = blockIdx.y * blockDim.y + sm_row - TOP;
    const int im_col = blockIdx.x * blockDim.x + sm_col - LEFT;
    const bool boundaries = (im_row >= 0 && im_col >= 0 && im_row < rows && im_col < cols);
    window[sm_row * win_cols + sm_col] =
        boundaries ? im[im_row * cols + im_col] : 0;
    window2[sm_row * win_cols + sm_col] =
        boundaries ? im2[im_row * cols + im_col] : 0;

    // Not enough threads to fill window and window2
    const int block_size = blockDim.x * blockDim.y;
    if (id < (win_cols * win_rows - block_size))
    {
        const int id = threadIdx.y * blockDim.x + threadIdx.x + block_size;
        const int sm_row = id / win_cols;
        const int sm_col = id % win_cols;

        const int im_row = blockIdx.y * blockDim.y + sm_row - TOP;
        const int im_col = blockIdx.x * blockDim.x + sm_col - LEFT;
        const bool boundaries = (im_row >= 0 && im_col >= 0 && im_row < rows && im_col < cols);
        window[sm_row * win_cols + sm_col] =
            boundaries ? im[im_row * cols + im_col] : 0;
        window2[sm_row * win_cols + sm_col] =
            boundaries ? im2[im_row * cols + im_col] : 0;
    }

    __syncthreads();
    uint32_t census = 0;
    uint32_t census2 = 0;
    if (idy < rows && idx < cols)
    {
        for (int k = 0; k < CENSUS_HEIGHT / 2; k++)
        {
            for (int m = 0; m < CENSUS_WIDTH; m++)
            {
                const uint8_t e1 = window[(threadIdx.y + k) * win_cols + threadIdx.x + m];
                const uint8_t e2 = window[(threadIdx.y + 2 * TOP - k) * win_cols + threadIdx.x + 2 * LEFT - m];
                const uint8_t i1 = window2[(threadIdx.y + k) * win_cols + threadIdx.x + m];
                const uint8_t i2 = window2[(threadIdx.y + 2 * TOP - k) * win_cols + threadIdx.x + 2 * LEFT - m];

                const int shft = k * CENSUS_WIDTH + m;
                // Compare to the center
                uint32_t tmp = (e1 >= e2);
                // Shift to the desired position
                tmp <<= shft;
                // Add it to its place
                census |= tmp;
                // Compare to the center
                uint32_t tmp2 = (i1 >= i2);
                // Shift to the desired position
                tmp2 <<= shft;
                // Add it to its place
                census2 |= tmp2;
            }
        }
        if (CENSUS_HEIGHT % 2 != 0)
        {
            const int k = CENSUS_HEIGHT / 2;
            for (int m = 0; m < CENSUS_WIDTH / 2; m++)
            {
                const uint8_t e1 = window[(threadIdx.y + k) * win_cols + threadIdx.x + m];
                const uint8_t e2 = window[(threadIdx.y + 2 * TOP - k) * win_cols + threadIdx.x + 2 * LEFT - m];
                const uint8_t i1 = window2[(threadIdx.y + k) * win_cols + threadIdx.x + m];
                const uint8_t i2 = window2[(threadIdx.y + 2 * TOP - k) * win_cols + threadIdx.x + 2 * LEFT - m];

                const int shft = k * CENSUS_WIDTH + m;
                // Compare to the center
                uint32_t tmp = (e1 >= e2);
                // Shift to the desired position
                tmp <<= shft;
                // Add it to its place
                census |= tmp;
                // Compare to the center
                uint32_t tmp2 = (i1 >= i2);
                // Shift to the desired position
                tmp2 <<= shft;
                // Add it to its place
                census2 |= tmp2;
            }
        }

        transform[idy * cols + idx] = census;
        transform2[idy * cols + idx] = census2;
    }
}

__global__ void
HammingDistanceCostKernel(const uint32_t *d_transform0,
                          const uint32_t *d_transform1, uint8_t *d_cost,
                          const int rows, const int cols, const int max_disp)
{
    // const int Dmax = blockDim.x;  // Dmax is CTA size
    const int y = blockIdx.x;      // y is CTA Identifier
    const int THRid = threadIdx.x; // THRid is Thread Identifier

    extern __shared__ uint32_t S[];
    uint32_t *SharedMatch = S;
    uint32_t *SharedBase = &S[2 * max_disp];
    // __shared__ uint32_t SharedMatch[2 * max_disp];
    // __shared__ uint32_t SharedBase[max_disp];

    SharedMatch[max_disp + THRid] = d_transform1[y * cols + 0]; // init position

    int n_iter = cols / max_disp;
    for (int ix = 0; ix < n_iter; ix++)
    {
        const int x = ix * max_disp;
        SharedMatch[THRid] = SharedMatch[THRid + max_disp];
        SharedMatch[THRid + max_disp] = d_transform1[y * cols + x + THRid];
        SharedBase[THRid] = d_transform0[y * cols + x + THRid];

        __syncthreads();
        for (int i = 0; i < max_disp; i++)
        {
            const uint32_t base = SharedBase[i];
            const uint32_t match = SharedMatch[(max_disp - 1 - THRid) + 1 + i];
            d_cost[(y * cols + x + i) * max_disp + THRid] =
                // popcount(base ^ match);
                __popc(base ^ match);
        }
        __syncthreads();
    }
    // For images with cols not multiples of max_disp
    const int x = max_disp * (cols / max_disp);
    const int left = cols - x;
    if (left > 0)
    {
        SharedMatch[THRid] = SharedMatch[THRid + max_disp];
        if (THRid < left)
        {
            SharedMatch[THRid + max_disp] = d_transform1[y * cols + x + THRid];
            SharedBase[THRid] = d_transform0[y * cols + x + THRid];
        }

        __syncthreads();
        for (int i = 0; i < left; i++)
        {
            const uint32_t base = SharedBase[i];
            const uint32_t match = SharedMatch[(max_disp - 1 - THRid) + 1 + i];
            d_cost[(y * cols + x + i) * max_disp + THRid] =
                // popcount(base ^ match);
                __popc(base ^ match);
        }
        __syncthreads();
    }
}

__global__ void
cost_u1_to_f1_transpose(const uchar *const img_cost_u1_d,
                        float *const img_cost_d, const int width,
                        const int height, const int m_labels,
                        const int max_disp, const float tau)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    int id = y * width + x;
    int im_size = width * height;
    for (int i = 0; i < m_labels; ++i)
    {
        float cost = img_cost_u1_d[i + id * max_disp];
        cost = cost < tau ? cost : tau;
        img_cost_d[id + i * im_size] = cost;
    }
}

__global__ void
truncate_data_k(float *const img_cost_d, const int m_labels, const int width,
                const int height, const float tau) // tau of float type causes bugs
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    int id = y * width + x;
    int im_size = width * height;
    for (int i = 0; i < m_labels; ++i)
    {
        float cost = img_cost_d[id + i * im_size];
        img_cost_d[id + i * im_size] = fminf(cost, tau);
    }
}

__global__ void
WTA_D_k(const float *const img_cost_d, float *const img_D_d, const int im_size,
        const int m_labels)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= im_size)
        return;

    float min_cost = FLT_MAX;
    int min_id = 0;
    for (int i = 0; i < m_labels; ++i)
    {
        if (img_cost_d[id + i * im_size] < min_cost)
        {
            min_cost = img_cost_d[id + i * im_size];
            min_id = i;
        }
    }
    img_D_d[id] = min_id;
    // TODO: this is suitable for normalized cost only
    /*
    if (min_id <= 1 || min_id >= m_labels - 2)
        img_D_d[id] = min_id;
    else
    {
        float c = 1 - img_cost_d[id + min_id * im_size];
        float c_m = 1 - img_cost_d[id + (min_id - 1) * im_size];
        float c_p = 1 - img_cost_d[id + (min_id + 1) * im_size];
        // img_D_d[id] = min_id - (c_p - c_m) / (c_p - 2 * c + c_m) / 2;
        img_D_d[id] = (c * min_id + c_m * (min_id - 1) + c_p * (min_id + 1)) / (c + c_m + c_p);
    }
    */
}

__global__ void
img_BGR_to_grey(const uchar3 *, float *, const int, const int);

__global__ void
img_float1_to_uchar1(const float *, uchar *, const int, const int);

/*****************************************************************/

void StereoCost::img_CPU_to_GPU(const cv::Mat &img, uchar3 *const img_u3_d,
                                float *const img_f1_d)
{
    cudaMemcpy(img_u3_d, img.data, sizeof(uchar3) * im_size,
               cudaMemcpyHostToDevice);

    img_BGR_to_grey<<<m_grids, m_blocks>>>(img_u3_d, img_f1_d, width, height);
}

void StereoCost::census_cost(float *const img_cost_d)
{
    // first census transform
    // then compute hamming distance

    // image_float_to_uchar
    img_float1_to_uchar1<<<m_grids, m_blocks>>>(
        imgL_f1_d,
        imgL_u1_d, width, height);
    img_float1_to_uchar1<<<m_grids, m_blocks>>>(
        imgR_f1_d,
        imgR_u1_d, width, height);

    dim3 block_size(32, 32);

    dim3 grid_size((width + block_size.x - 1) / block_size.x,
                   (height + block_size.y - 1) / block_size.y);
    CenterSymmetricCensusKernelSM2<<<grid_size, block_size>>>(
        imgL_u1_d,
        imgR_u1_d, imgL_trans_d, imgR_trans_d, height,
        width);

    HammingDistanceCostKernel<<<height,
                                m_labels_align,
                                3 * m_labels_align * sizeof(uint32_t)>>>(
        imgL_trans_d,
        imgR_trans_d, img_cost_u1_d, height, width,
        m_labels_align);

    // convert cost u1 to f1
    cost_u1_to_f1_transpose<<<m_grids, m_blocks>>>(img_cost_u1_d, img_cost_d,
                                                   width, height, m_labels, m_labels_align, m_tau);
}

void fillOutOfView(cv::Mat &volume, const bool is_left)
{
    int D = volume.size.p[0];
    int H = volume.size.p[1];
    int W = volume.size.p[2];

    if (is_left)
        for (int d = 0; d < D; d++)
            for (int y = 0; y < H; y++)
            {
                auto p = volume.ptr<float>(d, y);
                auto q = p + d;
                float v = *q;
                while (p != q)
                {
                    *p = v;
                    p++;
                }
            }
    else
        for (int d = 0; d < D; d++)
            for (int y = 0; y < H; y++)
            {
                auto q = volume.ptr<float>(d, y) + W;
                auto p = q - d;
                float v = p[-1];
                while (p != q)
                {
                    *p = v;
                    p++;
                }
            }
}
__global__ void
fill_out_of_view_k(float *const img_cost_d, const int D, const int H,
                   const int W, const bool is_left)
{
    int y = blockDim.x * blockIdx.x + threadIdx.x;
    int d = blockDim.y * blockIdx.y + threadIdx.y;

    if (y >= H || d >= D)
        return;

    int im_size = H * W;
    if (is_left)
    {
        int x_ptr = d * im_size + y * W;
        float c = img_cost_d[x_ptr + d];
        for (int x = 0; x <= d; ++x)
        {
            img_cost_d[x_ptr + x] = c;
        }
    }
    else
    {
        int x_ptr = d * im_size + y * W + W - 1;
        float c = img_cost_d[x_ptr - d];
        for (int x = 0; x <= d; ++x)
        {
            img_cost_d[x_ptr - x] = c;
        }
    }
}
void StereoCost::read_volume_from_file(const std::string vol_dir,
                                       float *const img_cost_d)
{
    int sizes[] =
        {m_labels, height, width};
    cv::Mat vol = cv::Mat_<float>(3, sizes);
    if (cvutils::io::loadMatBinary(vol_dir, vol, false) == false)
    {
        printf("Cost volume file im0.acrt not found\n");
        return;
    }
    // fillOutOfView (vol, true);
    cudaMemcpy(img_cost_d, vol.data, sizeof(float) * m_labels * height * width,
               cudaMemcpyHostToDevice);

    dim3 grids((height + m_blocks.x - 1) / m_blocks.x,
               (m_labels + m_blocks.y - 1) / m_blocks.y);
    fill_out_of_view_k<<<grids, m_blocks>>>(img_cost_d, m_labels, height, width,
                                            true);

    truncate_data_k<<<m_grids, m_blocks>>>(
        img_cost_d,
        m_labels, width, height, m_tau);
}

void StereoCost::compute_stereo_cost(const std::string algo)
{
    // TODO: add census
    if (algo == "AD")
    {
        m_tau = 30;
        // absolute difference
        ad_cost<<<m_grids, m_blocks>>>(imgL_f1_d, imgR_f1_d, img_cost_d, width,
                                       height, m_labels, m_tau);
    }
    else if (algo == "Census")
    {
        m_tau = 25;
        census_cost(img_cost_d);
    }
    else if (algo == "MC-CNN")
    {
        m_tau = 0.5;
        // read from file
        read_volume_from_file(vol_dir, img_cost_d);
    }
    // WTA depth from cost
    int grid = (im_size + m_block - 1) / m_block;
    WTA_D_k<<<grid, m_block>>>(img_cost_d, img_D_d, im_size, m_labels);
}

cv::Mat
convertVolumeL2R(cv::Mat &volSrc, int margin = 0)
{
    int D = volSrc.size[0];
    int H = volSrc.size[1];
    int W = volSrc.size[2];
    cv::Mat volDst = volSrc.clone();

    for (int d = 0; d < D; d++)
    {
        cv::Mat_<float> s0(H, W, volSrc.ptr<float>() + H * W * d);
        cv::Mat_<float> s1(H, W, volDst.ptr<float>() + H * W * d);
        s0(cv::Rect(d, 0, W - d, H)).copyTo(s1(cv::Rect(0, 0, W - d, H)));

        cv::Mat edge1 = s0(cv::Rect(W - 1 - margin, 0, 1, H)).clone();
        cv::Mat edge0 = s0(cv::Rect(d + margin, 0, 1, H)).clone();
        for (int x = W - 1 - d - margin; x < W; x++)
            edge1.copyTo(s1.col(x));
        for (int x = 0; x < margin; x++)
            edge0.copyTo(s1.col(x));
    }
    return volDst;
}

__global__ void
convert_vol_L2R_k(float *const img_cost_d, const int D, const int H,
                  const int W)
{
    int y = blockDim.x * blockIdx.x + threadIdx.x;
    int d = blockDim.y * blockIdx.y + threadIdx.y;

    if (y >= H || d >= D)
        return;

    int x_ptr = d * H * W + y * W;
    for (int x = 0; x + d < W; ++x)
    {
        img_cost_d[x_ptr + x] = img_cost_d[x_ptr + x + d];
    }
}

void StereoCost::cost_to_right()
{
    // CPU version
    /*
     int sizes[] =
     { m_labels, height, width };
     cv::Mat vol = cv::Mat_<float> (3, sizes);
     cudaMemcpy (vol.data, img_cost_d, sizeof(float) * m_labels * height * width,
     cudaMemcpyDeviceToHost);
     cv::Mat vol_r = convertVolumeL2R (vol);

     // fillOutOfView (vol_r, false);
     cudaMemcpy (img_cost_d, vol_r.data,
     sizeof(float) * m_labels * height * width,
     cudaMemcpyHostToDevice);
     */

    // GPU version
    dim3 grids((height + m_blocks.x - 1) / m_blocks.x,
               (m_labels + m_blocks.y - 1) / m_blocks.y);

    convert_vol_L2R_k<<<grids, m_blocks>>>(img_cost_d, m_labels, height, width);

    fill_out_of_view_k<<<grids, m_blocks>>>(img_cost_d, m_labels, height, width,
                                            false);
    // WTA depth from cost
    int grid = (im_size + m_block - 1) / m_block;
    WTA_D_k<<<grid, m_block>>>(img_cost_d, img_D_d, im_size, m_labels);
}
