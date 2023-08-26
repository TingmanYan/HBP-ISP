#include "post_process.hpp"
#include <omp.h>

void PostProcess::doConsistencyCheck(const cv::Mat &dispL, const cv::Mat &dispR,
                                     cv::Mat &failL, cv::Mat &failR,
                                     double dispThreshold)
{
    cv::Mat fail[2];
    cv::Mat disp[2] =
        {dispL, dispR};
    for (int i = 0; i < 2; i++)
    {
        fail[i] = cv::Mat::zeros(cv::Size(width, height), CV_8U);
    }

    for (int i = 0; i < 2; i++)
    {
        float sign = (i ? -1.0f : 1.0f);
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
            {
                cv::Point p(x, y);

                float ds = disp[i].at<float>(p);
                int rx = int((float)x - ds * sign + 0.5f);

                cv::Point q(rx, y);
                if (imageDomain.contains(q))
                {
                    float dsr = disp[1 - i].at<float>(q);
                    if (fabs(dsr - ds) > dispThreshold)
                    {
                        fail[i].at<uchar>(p) = 255;
                    }
                }
            }
    }

    failL = fail[0];
    failR = fail[1];
}

float GetZ(cv::Vec3f &v, cv::Point &p)
{
    return v[0] * p.x + v[1] * p.y + v[2];
}
void PostProcess::left_right_check(std::pair<cv::Mat, cv::Mat> &results_left,
                                   std::pair<cv::Mat, cv::Mat> &results_right)
{
    cv::Mat disp[2] =
        {results_left.first, results_right.first};
    cv::Mat LR[2] =
        {results_left.second, results_right.second};

    // LR-consistency check
    cv::Mat fail2[2];

    for (int i = 0; i < 2; i++)
    {
        fail[i] = cv::Mat::zeros(cv::Size(width, height), CV_8U);
        fail2[i] = cv::Mat::zeros(cv::Size(width, height), CV_8U);
    }

    doConsistencyCheck(disp[0], disp[1], fail[0], fail[1], lrc_thresh);
    fail[0] = fail[0] > 0;
    fail[1] = fail[1] > 0;

    cv::dilate(fail[0], fail2[0], cv::Mat());
    cv::dilate(fail[1], fail2[1], cv::Mat());

    //// horizontal NN-interpolation
    for (int i = 0; i < 1; i++)
    {
#pragma omp parallel for
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
            {
                cv::Point p(x, y);

                if (fail[i].at<uchar>(p) == 0)
                    continue;

                cv::Vec3f *pl = NULL, *pr = NULL;
                int xx;
                for (xx = x; xx >= 0 && fail2[i].at<uchar>(y, xx) == 255;
                     xx--)
                    ;
                if (xx >= 0)
                    pl = &LR[i].at<cv::Vec3f>(y, xx);

                for (xx = x; xx < width && fail2[i].at<uchar>(y, xx) == 255;
                     xx++)
                    ;
                if (xx < width)
                    pr = &LR[i].at<cv::Vec3f>(y, xx);

                if (pl == NULL && pr == NULL)
                    // LR[i][s] = *pr;
                    ;
                else if (pl == NULL)
                    LR[i].at<cv::Vec3f>(p) = *pr;
                else if (pr == NULL)
                    LR[i].at<cv::Vec3f>(p) = *pl;
                else if (GetZ(*pl, p) < GetZ(*pr, p))
                    LR[i].at<cv::Vec3f>(p) = *pl;
                else
                    LR[i].at<cv::Vec3f>(p) = *pr;
            }
    }
}

float PostProcess::computePatchWeight(const cv::Point &s, const cv::Point &t,
                                      const int mode) const
{
    const cv::Mat &I = this->I[mode];
    cv::Vec3f dI = I.at<cv::Vec3f>(s) - I.at<cv::Vec3f>(t);
    float absdiff = fabs(dI[0]) + fabs(dI[1]) + fabs(dI[2]);
    return std::exp(-absdiff / omega);
}

cv::Rect
getLargerRect(cv::Rect rect, int margin)
{
    return cv::Rect(rect.x - margin, rect.y - margin, rect.width + margin * 2,
                    rect.height + margin * 2);
}
void PostProcess::median_filter(std::pair<cv::Mat, cv::Mat> &results_left,
                                std::pair<cv::Mat, cv::Mat> &results_right)
{
    cv::Mat LR[2] =
        {results_left.second, results_right.second};

    using Triplet = std::tuple<cv::Vec3f, float, float>;

    //// median filter
    for (int i = 0; i < 1; i++)
    {
        cv::Mat LRcopy = LR[i].clone();

#pragma omp parallel for
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
            {
                cv::Point p(x, y);

                if (fail[i].at<uchar>(p) == 0)
                    continue;
                std::vector<Triplet> median;
                double sumw = 0;

                cv::Rect patch = getLargerRect(cv::Rect(p, cv::Size(1, 1)),
                                               windR) &
                                 imageDomain;

                for (int yy = patch.y; yy < patch.br().y; yy++)
                    for (int xx = patch.x; xx < patch.br().x; xx++)
                    {
                        cv::Point q(xx, yy);

                        float w = computePatchWeight(p, q, i);
                        sumw += w;
                        // median.push_back(Triplet(LR[i].at<Plane>(q), w, LR[i].at<Plane>(q).GetZ(p)));
                        median.push_back(
                            Triplet(
                                LRcopy.at<cv::Vec3f>(q), w,
                                GetZ(LRcopy.at<cv::Vec3f>(q), p)));
                    }

                std::sort(begin(median), end(median), [](const Triplet &a, const Triplet &b)
                          { return std::get<2>(a) < std::get<2>(b); });

                double center = sumw / 2.0;
                sumw = 0;
                for (int j = 0; j < median.size(); j++)
                {
                    sumw += std::get<1>(median[j]);
                    if (sumw > center)
                    {
                        LR[i].at<cv::Vec3f>(p) = std::get<0>(median[j]);
                        break;
                    }
                }
            }
    }
}

__device__ float4
get_larger_rect(const int x, const int y, const int width, const int height,
                const int windR)
{
    return make_float4(max(x - windR, 0), max(y - windR, 0),
                       min(x + windR, width - 1), min(y + windR, height - 1));
}
__forceinline__ __device__ float
compute_patch_weight(const int x, const int y, const int xx, const int yy,
                     const float *const img_d, const int width,
                     const int height, const float omega)
{
    int s_id = y * width + x;
    int t_id = yy * width + xx;
    float s_0 = img_d[s_id * 3];
    float s_1 = img_d[s_id * 3 + 1];
    float s_2 = img_d[s_id * 3 + 2];
    float t_0 = img_d[t_id * 3];
    float t_1 = img_d[t_id * 3 + 1];
    float t_2 = img_d[t_id * 3 + 2];
    float absdiff = fabsf(s_0 - t_0) + fabsf(s_1 - t_1) + fabsf(s_2 - t_2);
    return expf(-absdiff / omega);
}
__forceinline__ __device__ float
get_z(const int x, const int y, const int xx, const int yy,
      const float *const label_buf_d, const int width)
{
    int l_id = yy * width + xx;
    float a = label_buf_d[l_id * 3];
    float b = label_buf_d[l_id * 3 + 1];
    float c = label_buf_d[l_id * 3 + 2];
    return a * x + b * y + c;
}
__device__ void
bubble_sort(float3 *const median, const int size)
{
    for (int i = 0; i < size - 1; ++i)
    {
        for (int j = i + 1; j < size; ++j)
        {
            if (median[j].z < median[i].z)
            {
                float3 tmp = median[i];
                median[i] = median[j];
                median[j] = tmp;
            }
        }
    }
}
__forceinline__ __device__ int
get_new_id(const int l_id, const float4 patch, const int width)
{
    int patch_w = patch.z - patch.x + 1;
    int x = l_id % patch_w + patch.x;
    int y = l_id / patch_w + patch.y;
    return y * width + x;
}

__global__ void
median_filter_k(float3 *const median_d, float *const label_d,
                const float *const label_buf_d, const float *const img_d,
                const int *const im_id_d, const int num_occ, const int width,
                const int height, const int windR, const float omega)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= num_occ)
        return;

    int im_id = im_id_d[id];
    int y = im_id / width;
    int x = im_id % width;

    float sumw = 0.f;
    float4 patch = get_larger_rect(x, y, width, height, windR);
    int size = (patch.z - patch.x + 1) * (patch.w - patch.y + 1);
    int win_step = (windR * 2 + 1) * (windR * 2 + 1);
    float3 *median = &median_d[id * win_step];

    int i = 0;
    for (int yy = patch.y; yy <= patch.w; yy++)
        for (int xx = patch.x; xx <= patch.z; xx++)
        {
            float w = compute_patch_weight(x, y, xx, yy, img_d, width, height,
                                           omega);
            sumw += w;
            median[i] = make_float3(i, w,
                                    get_z(x, y, xx, yy, label_buf_d, width));
            ++i;
        }

    bubble_sort(median, size);

    float center = sumw / 2;
    sumw = 0;
    for (int j = 0; j < size; j++)
    {
        sumw += median[j].y;
        if (sumw > center)
        {
            int im_id_ = get_new_id(median[j].x, patch, width);
            // printf ("%d %d, ", im_id, im_id_);
            label_d[im_id * 3] = label_buf_d[im_id_ * 3];
            label_d[im_id * 3 + 1] = label_buf_d[im_id_ * 3 + 1];
            label_d[im_id * 3 + 2] = label_buf_d[im_id_ * 3 + 2];
            break;
        }
    }
}
__global__ void
set_im_id_k(int *const im_id_d, int *const predicate_d,
            const uchar *const mask_d, const int im_size)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= im_size)
        return;

    im_id_d[id] = id;
    predicate_d[id] = mask_d[id] ? 1 : 0;
}
void PostProcess::median_filter_gpu(std::pair<cv::Mat, cv::Mat> &results_left,
                                    std::pair<cv::Mat, cv::Mat> &results_right)
{
    cv::Mat label = results_left.second.clone();
    cv::Mat mask = fail[0];
    cv::Mat img = I[0];

    float *label_d = nullptr;
    float *label_buf_d = nullptr;
    float *img_d = nullptr;
    uchar *mask_d = nullptr;

    cudaMalloc(&label_d, sizeof(float) * im_size * 3);
    cudaMalloc(&label_buf_d, sizeof(float) * im_size * 3);
    cudaMalloc(&img_d, sizeof(float) * im_size * 3);
    cudaMalloc(&mask_d, sizeof(uchar) * im_size);

    cudaMemcpy(label_d, label.data, sizeof(float) * im_size * 3,
               cudaMemcpyHostToDevice);
    cudaMemcpy(label_buf_d, label_d, sizeof(float) * im_size * 3,
               cudaMemcpyDeviceToDevice);
    cudaMemcpy(img_d, img.data, sizeof(float) * im_size * 3,
               cudaMemcpyHostToDevice);
    cudaMemcpy(mask_d, mask.data, sizeof(uchar) * im_size,
               cudaMemcpyHostToDevice);

    int *predicate_d = nullptr;
    int *pos_scan_d = nullptr;
    int *im_id_d = nullptr;
    int *im_id_rd_d = nullptr;
    cudaMalloc(&predicate_d, sizeof(int) * im_size);
    cudaMalloc(&pos_scan_d, sizeof(int) * im_size);
    cudaMalloc(&im_id_d, sizeof(int) * im_size);
    cudaMalloc(&im_id_rd_d, sizeof(int) * im_size);

    int grid = (im_size + m_block - 1) / m_block;
    set_im_id_k<<<grid, m_block>>>(im_id_d, predicate_d, mask_d, im_size);
    thrust::exclusive_scan(thrust::device, predicate_d, predicate_d + im_size,
                           pos_scan_d);
    int num_occ = get_num_from_scan(predicate_d, pos_scan_d, im_size);
    thrust::scatter_if(thrust::device, im_id_d, im_id_d + im_size, pos_scan_d,
                       predicate_d, im_id_rd_d);

    float3 *median_d = nullptr;
    cudaMalloc(&median_d,
               sizeof(float3) * num_occ * (windR * 2 + 1) * (windR * 2 + 1));

    grid = (num_occ + m_block - 1) / m_block;
    median_filter_k<<<grid, m_block>>>(median_d, label_d, label_buf_d, img_d,
                                       im_id_rd_d, num_occ, width, height, windR, omega);

    cv::Mat label_img_h(height, width, CV_32FC3);
    cudaMemcpy(label_img_h.data, label_d, sizeof(float) * im_size * 3,
               cudaMemcpyDeviceToHost);
    results_left.second = label_img_h.clone();
    cudaFree(mask_d);
    cudaFree(img_d);
    cudaFree(label_buf_d);
    cudaFree(label_d);

    cudaFree(median_d);
    cudaFree(im_id_rd_d);
    cudaFree(im_id_d);
    cudaFree(pos_scan_d);
    cudaFree(predicate_d);
}

void PostProcess::plane_label_to_disp(std::pair<cv::Mat, cv::Mat> &results_left,
                                      std::pair<cv::Mat, cv::Mat> &results_right)
{
    cv::Mat disp[2] =
        {results_left.first, results_right.first};
    cv::Mat LR[2] =
        {results_left.second, results_right.second};

    for (int i = 0; i < 1; i++)
    {
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
            {
                cv::Point p(x, y);
                disp[i].at<float>(p) = GetZ(LR[i].at<cv::Vec3f>(p), p);
            }
    }
}
