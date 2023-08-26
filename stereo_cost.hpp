#ifndef _STEREO_COST_HPP_
#define _STEREO_COST_HPP_

#include "util.h"
#include "cvutils.hpp"
#include <opencv2/opencv.hpp>

class StereoCost
{
public:
    void
    compute_stereo_cost (const std::string);
    void
    cost_to_right ();

private:
    void
    img_CPU_to_GPU (const cv::Mat&, uchar3*, float*);

    void
    census_cost (float*);
    void
    read_volume_from_file (const std::string, float*);

public:
    // interfaces
    int
    g_labels () const
    {
        return this->m_labels;
    }
    int
    g_im_size () const
    {
        return this->im_size;
    }
    float
    g_tau () const
    {
        return this->m_tau;
    }
    float*
    g_img_cost_d () const
    {
        return this->img_cost_d;
    }
    float*
    g_img_D_d () const
    {
        return this->img_D_d;
    }

protected:
    const int width;
    const int height;
    const int m_labels;

    cv::Mat imgL;
    cv::Mat imgR;

    float *imgL_f1_d = nullptr;
    float *imgR_f1_d = nullptr;

    uchar *imgL_u1_d = nullptr;
    uchar *imgR_u1_d = nullptr;

    uint32_t *imgL_trans_d = nullptr;
    uint32_t *imgR_trans_d = nullptr;

    float *img_D_d = nullptr;
    float *img_cost_d = nullptr;
    uchar *img_cost_u1_d = nullptr;

private:
    // help variables and functions
    const std::string vol_dir;
    int im_size;
    int m_labels_align;

    const int m_block = 32;
    const dim3 m_blocks = dim3 (32, 2);
    dim3 m_grids;

    uchar3 *imgL_u3_d;
    uchar3 *imgR_u3_d;

    float m_tau;

public:
    StereoCost (const cv::Mat &img_left, const cv::Mat &img_right,
                const int labels, const std::string vol_dir) :
            width (img_left.cols), height (img_left.rows), m_labels (labels), vol_dir (
                    vol_dir), im_size (width * height), m_grids (
                    (width + m_blocks.x - 1) / m_blocks.x,
                    (height + m_blocks.y - 1) / m_blocks.y)
    {
        imgL = img_left.clone ();
        imgR = img_right.clone ();

        m_labels_align = (m_labels + 31) / 32 * 32;

        cudaMalloc (&img_D_d, sizeof(float) * im_size);
        cudaMalloc (&img_cost_d, sizeof(float) * im_size * m_labels);
        cudaMalloc (&img_cost_u1_d, sizeof(uchar) * im_size * m_labels_align);
        cudaMalloc (&imgL_f1_d, sizeof(float) * im_size);
        cudaMalloc (&imgR_f1_d, sizeof(float) * im_size);
        cudaMalloc (&imgL_u1_d, sizeof(uchar) * im_size);
        cudaMalloc (&imgR_u1_d, sizeof(uchar) * im_size);
        cudaMalloc (&imgL_trans_d, sizeof(uint32_t) * im_size);
        cudaMalloc (&imgR_trans_d, sizeof(uint32_t) * im_size);
        cudaMalloc (&imgL_u3_d, sizeof(uchar3) * im_size);
        cudaMalloc (&imgR_u3_d, sizeof(uchar3) * im_size);

        img_CPU_to_GPU (imgL, imgL_u3_d, imgL_f1_d);
        img_CPU_to_GPU (imgR, imgR_u3_d, imgR_f1_d);
    }

    ~StereoCost ()
    {
        cudaFree (imgR_u3_d);
        cudaFree (imgL_u3_d);
        cudaFree (imgR_trans_d);
        cudaFree (imgL_trans_d);
        cudaFree (imgR_u1_d);
        cudaFree (imgL_u1_d);
        cudaFree (imgR_f1_d);
        cudaFree (imgL_f1_d);
        cudaFree (img_cost_u1_d);
        cudaFree (img_cost_d);
        cudaFree (img_D_d);
    }
};

#endif
