#ifndef _HBP_ISP_HPP_
#define _HBP_ISP_HPP_

#include "segment.hpp"
#include "stereo_cost.hpp"
#include "util.h"
#include <opencv2/opencv.hpp>

class ISPData
{
public:
    // interface functions
    int g_width() const { return this->width; }
    int g_height() const { return this->height; }
    int g_mean_channels() const { return this->mean_channels; }
    std::vector<int> g_num_clus_isp() const { return this->num_clus_isp; }
    std::vector<int> g_num_bd_isp() const { return this->num_bd_isp; }
    int *g_isp_clus_d() const { return this->isp_clus_d; }
    float *g_isp_mean_d() const { return this->isp_mean_d; }
    int2 *g_isp_bd_d() const { return this->isp_bd_d; }
    int *g_isp_bdl_d() const { return this->isp_bdl_d; }

public:
    ISPData(const SegHAC *const seg_hac)
        : width(seg_hac->g_width()), height(seg_hac->g_height()),
          mean_channels(seg_hac->g_mean_channels()),
          num_clus_isp(seg_hac->g_num_clus_isp()),
          num_bd_isp(seg_hac->g_num_bd_isp())
    {
        int total_clus = get_vec_sum(num_clus_isp);
        int total_bd = get_vec_sum(num_bd_isp);
        cudaMalloc(&isp_clus_d, sizeof(int) * total_clus);
        cudaMalloc(&isp_mean_d, sizeof(float) * total_clus * mean_channels);
        cudaMalloc(&isp_bd_d, sizeof(int2) * total_bd);
        cudaMalloc(&isp_bdl_d, sizeof(int) * total_bd);

        cudaMemcpy(isp_clus_d, seg_hac->g_isp_clus_d(), sizeof(int) * total_clus,
                   cudaMemcpyDeviceToDevice);
        cudaMemcpy(isp_mean_d, seg_hac->g_isp_mean_d(),
                   sizeof(float) * total_clus * mean_channels,
                   cudaMemcpyDeviceToDevice);
        cudaMemcpy(isp_bd_d, seg_hac->g_isp_bd_d(), sizeof(int2) * total_bd,
                   cudaMemcpyDeviceToDevice);
        cudaMemcpy(isp_bdl_d, seg_hac->g_isp_bdl_d(), sizeof(int) * total_bd,
                   cudaMemcpyDeviceToDevice);
    }
    ~ISPData()
    {
        cudaFree(isp_bdl_d);
        cudaFree(isp_bd_d);
        cudaFree(isp_mean_d);
        cudaFree(isp_clus_d);
    }

private:
    int width;
    int height;
    int mean_channels;
    std::vector<int> num_clus_isp;
    std::vector<int> num_bd_isp;

    int *isp_clus_d = nullptr;
    float *isp_mean_d = nullptr;
    int2 *isp_bd_d = nullptr;
    int *isp_bdl_d = nullptr;

private:
    int get_vec_sum(const std::vector<int> &vec)
    {
        int sum = 0;
        for (int i = 0; i < vec.size(); ++i)
        {
            sum += vec[i];
        }
        return sum;
    }
};

class StereoData
{
public:
    // interface functions
    int g_labels() const { return this->m_labels; }
    int g_im_size() const { return this->im_size; }
    float *g_im_D_d() const { return this->im_D_d; }
    bool g_use_cost_vol() const { return this->use_cost_vol; }
    float *g_img_cost_d() const { return this->im_cost_d; }
    float g_tau() const { return this->m_tau; }

public:
    StereoData(const StereoCost *const stereo_cost, const bool use_cost_vol)
        : m_labels(stereo_cost->g_labels()), im_size(stereo_cost->g_im_size()),
          use_cost_vol(use_cost_vol)
    {
        cudaMalloc(&im_D_d, sizeof(float) * im_size);
        cudaMemcpy(im_D_d, stereo_cost->g_img_D_d(), sizeof(float) * im_size,
                   cudaMemcpyDeviceToDevice);
        if (use_cost_vol)
        {
            cudaMalloc(&im_cost_d, sizeof(float) * im_size * m_labels);
            cudaMemcpy(im_cost_d, stereo_cost->g_img_cost_d(),
                       sizeof(float) * im_size * m_labels, cudaMemcpyDeviceToDevice);
            m_tau = stereo_cost->g_tau();
        }
    }
    ~StereoData()
    {
        if (use_cost_vol)
            cudaFree(im_cost_d);
        cudaFree(im_D_d);
    }

private:
    int m_labels;
    int im_size;
    float *im_D_d = nullptr;
    float *im_cost_d = nullptr;
    bool use_cost_vol = false;
    float m_tau;
};

class HBP_ISP
{
public:
    std::pair<cv::Mat, cv::Mat> run_ms(const int);
    std::vector<std::tuple<cv::Mat, cv::Mat, cv::Mat>> run_ms_vis(const int);

protected:
    void switch_level(const int);
    void compute_dist_color_pos(float *, float *);
    void sort_clus(int *, int *, int *);
    void hbp_g(float4 *, float *);
    std::pair<cv::Mat, cv::Mat> map_msg(const float4 *, const float *);
    cv::Mat draw_spx_mean(const float *);

private:
    float set_lambda_by_KL_divergence(const float *, float *);
    void setup_D4(int **, float4 **);
    int remove_non_fbd(const int2 *, const float *, int2 *, float *, const int);

    void reverse_bd_id(const int2 *, int *);
    void reduce_num_pos(const int *, int *, const int, const int);

    bool set_candidate_plane(float4 *, const int);
    void compute_plane_cost(const int);
    void compute_inlier_ratio(float *);
    void set_occlusion_cost(const float *, float *);
    void occlusion_test(const int);
    void belief_propagation(float4 *, float *, float *, float *);
    void passing_msg(const float4 *, const float *, float *, float *);
    void aggregate_belief(const float *, const float *, float *);
    void ping_pang_msg(float *, const float *);
    void select_candidate_plane(float4 *, float *, float *, float *);

    void init_bp(float4 *, float *, float *);
    bool generate_plane(int &);

    void compute_plane_confidence(float *);
    void reset_candidate_plane(float4 *, const float *);
    void candidate_test(float4 *);

protected:
    int width;
    int height;

    const int im_size;
    const int m_labels;
    const int isp_levels;

    float *im_D_d = nullptr;
    float *im_cost_d = nullptr;

    int **isp_clus_d = nullptr;
    int **isp_im_clus_d = nullptr;
    float **isp_mean_d = nullptr;
    float **isp_cost_d = nullptr;
    float **isp_belief_d = nullptr;
    float4 **isp_plane_d = nullptr;
    float4 **isp_im_D4_d = nullptr;
    int4 **isp_point_id_d = nullptr;

    int2 **isp_bd_d = nullptr;
    float **isp_dc_d = nullptr;
    int **isp_bdl_d = nullptr;
    int **isp_clus_sort_d = nullptr;
    int **isp_clus_num_pos_d = nullptr;
    int **isp_im_clus_num_pos_d = nullptr;
    int **isp_nb_num_pos_d = nullptr;
    int **isp_reverse_id_d = nullptr;

    std::vector<int> num_clus_isp;
    std::vector<int> num_bd_isp;

private:
    const int mean_channels = 3 + 2 + 1;
    const int dc_channels = 1 + 1;

    // help pointers within one level
    int *clus_d = nullptr;
    int *im_clus_d = nullptr;
    float *mean_d = nullptr;
    float *cost_d = nullptr;
    float *belief_d = nullptr;
    float4 *plane_d = nullptr;

    int2 *bd_d = nullptr;
    float *dc_d = nullptr;
    int *bdl_d = nullptr;
    int *clus_sort_d = nullptr;
    int *clus_num_pos_d = nullptr;
    int *im_clus_num_pos_d = nullptr;
    int *nb_num_pos_d = nullptr;
    int *reverse_id_d = nullptr;

    int *im_clus_h_d = nullptr;
    float *mean_h_d = nullptr;
    float *cost_h_d = nullptr;
    float *belief_h_d = nullptr;
    float4 *plane_h_d = nullptr;

    float4 *im_D4_d = nullptr;
    int4 *point_id_d = nullptr;

    // help pointers need to be allocated
    float *im_plane_cost_d = nullptr;

    float4 *plane_buf_d = nullptr;
    float *msg_d = nullptr;
    float *msg_t_d = nullptr;

    float *KL_d = nullptr;

    int2 *bd_buf_d = nullptr;
    int *bds_d = nullptr;

    int *order_d = nullptr;

    int *predicate_d = nullptr;
    int *pos_scan_d = nullptr;
    float *min_belief_d = nullptr;
    float4 *min_plane_d = nullptr;
    float *disp_img_d = nullptr;
    float *label_img_d = nullptr;

    float *inlier_ratio_d = nullptr;
    float *plane_cfd_d = nullptr;

    // random state
    int *s_child_d = nullptr;
    int *s_neighbor_d = nullptr;
    int *s_ransac_d = nullptr;

    bool *flag_d = nullptr;

    int num_clus;
    int num_bd;
    int num_clus_h;

    int m_level;
    int m_plane_iter = 0;
    int iter_spatial_prop;
    int iter_ransac_search;
    int iter_random_adjust;

    const float end_dz = 0.01f;
    float max_dz;
    float max_dn;

    // kernel cfg
    const dim3 m_blocks = dim3(32, 2);
    const int m_block = 32 * 2;

    // params
    float m_lambda;
    const float m_alpha;
    const float m_tau_smooth;
    const float inlier_thrsh = 1.f;
    const double m_gamma = 20;
    const int P = 4;
    const int K = P * 3;
    bool use_cost_vol = false;
    const float inlier_ratio = 0.3f;
    float m_tau;

    // P x 3, P particles, three for best, prior, and candidate

public:
    HBP_ISP(const ISPData *const isp_data, const StereoData *const stereo_data,
            const float alpha, const float tau_smooth)
        : width(isp_data->g_width()), height(isp_data->g_height()),
          im_size(width * height), m_labels(stereo_data->g_labels()),
          isp_levels(isp_data->g_num_clus_isp().size()),
          num_clus_isp(isp_data->g_num_clus_isp()),
          num_bd_isp(isp_data->g_num_bd_isp()), im_D_d(stereo_data->g_im_D_d()),
          m_alpha(alpha), m_tau_smooth(tau_smooth), use_cost_vol(stereo_data->g_use_cost_vol())
    {
        isp_clus_d = new int *[isp_levels];
        isp_im_clus_d = new int *[isp_levels];
        isp_mean_d = new float *[isp_levels];
        isp_cost_d = new float *[isp_levels];
        isp_belief_d = new float *[isp_levels];
        isp_plane_d = new float4 *[isp_levels];
        isp_im_D4_d = new float4 *[isp_levels];
        isp_point_id_d = new int4 *[isp_levels];
        isp_bd_d = new int2 *[isp_levels];
        isp_dc_d = new float *[isp_levels];
        isp_bdl_d = new int *[isp_levels];
        isp_clus_sort_d = new int *[isp_levels];
        isp_clus_num_pos_d = new int *[isp_levels];
        isp_im_clus_num_pos_d = new int *[isp_levels];
        isp_nb_num_pos_d = new int *[isp_levels];
        isp_reverse_id_d = new int *[isp_levels];
        // read node info
        assert(mean_channels == isp_data->g_mean_channels());
        int total_clus = 0;
        int *v_isp_clus_d = isp_data->g_isp_clus_d();
        float *v_isp_mean_d = isp_data->g_isp_mean_d();
        for (int i = 0; i < isp_levels; ++i)
        {
            isp_clus_d[i] = &v_isp_clus_d[total_clus];
            isp_mean_d[i] = &v_isp_mean_d[total_clus * mean_channels];
            total_clus += num_clus_isp[i];
        }
        // read edge info
        int total_bd = 0;
        int2 *v_isp_bd_d = isp_data->g_isp_bd_d();
        int *v_isp_bdl_d = isp_data->g_isp_bdl_d();
        for (int i = 0; i < isp_levels; ++i)
        {
            isp_bd_d[i] = &v_isp_bd_d[total_bd];
            isp_bdl_d[i] = &v_isp_bdl_d[total_bd];
            total_bd += num_bd_isp[i];
        }

        for (int i = 0; i < isp_levels; ++i)
        {
            cudaMalloc(&isp_im_clus_d[i], sizeof(int) * im_size);
            cudaMalloc(&isp_cost_d[i], sizeof(float) * num_clus_isp[i] * K);
            cudaMalloc(&isp_belief_d[i], sizeof(float) * num_clus_isp[i] * K);
            cudaMalloc(&isp_plane_d[i], sizeof(float4) * num_clus_isp[i] * K);
            cudaMalloc(&isp_point_id_d[i], sizeof(int4) * num_clus_isp[i]);
            cudaMalloc(&isp_im_D4_d[i], sizeof(float4) * im_size);
            cudaMalloc(&isp_dc_d[i], sizeof(float) * num_bd_isp[i] * dc_channels);
            cudaMalloc(&isp_clus_sort_d[i], sizeof(int) * num_clus_isp[i] * 2);
            cudaMalloc(&isp_clus_num_pos_d[i],
                       sizeof(int) * num_clus_isp[i] *
                           2); // this shall be num_clus_isp[i+1]
            cudaMalloc(&isp_im_clus_num_pos_d[i], sizeof(int) * num_clus_isp[i] * 2);
            cudaMalloc(&isp_nb_num_pos_d[i], sizeof(int) * (num_clus_isp[i] + 1) * 2);
            cudaMalloc(&isp_reverse_id_d[i], sizeof(int) * num_bd_isp[i]);
        }

        if (use_cost_vol)
        {
            im_cost_d = stereo_data->g_img_cost_d();
            m_tau = stereo_data->g_tau();

            // cudaMalloc(&KL_d, sizeof(float) * num_bd_isp[0]);
            // m_lambda = set_lambda_by_KL_divergence(im_cost_d, KL_d);
            m_lambda = alpha;
        }
        else
            m_lambda = alpha;

        cudaMalloc(&im_plane_cost_d, sizeof(float) * im_size * K);
        cudaMalloc(&plane_buf_d, sizeof(float4) * num_clus_isp[0] * P);
        cudaMalloc(&msg_d, sizeof(float) * num_bd_isp[0] * K);
        cudaMalloc(&msg_t_d, sizeof(float) * num_bd_isp[0] * K);
        cudaMalloc(&bd_buf_d, sizeof(int2) * num_bd_isp[0]);
        cudaMalloc(&bds_d, sizeof(int) * num_bd_isp[0]);
        cudaMalloc(&order_d, sizeof(int) * num_clus_isp[0] * K);
        cudaMalloc(&inlier_ratio_d, sizeof(float) * num_clus_isp[0] * K);
        cudaMalloc(&plane_cfd_d, sizeof(int) * num_clus_isp[0] * K);
        cudaMalloc(&predicate_d, sizeof(int) * num_bd_isp[0]);
        cudaMalloc(&pos_scan_d, sizeof(int) * num_bd_isp[0]);
        cudaMalloc(&min_belief_d, sizeof(float) * num_clus_isp[0]);
        cudaMalloc(&min_plane_d, sizeof(float4) * num_clus_isp[0]);
        cudaMalloc(&disp_img_d, sizeof(float) * im_size);
        cudaMalloc(&label_img_d, sizeof(float) * im_size * 3);
        cudaMalloc(&s_child_d, sizeof(int) * num_clus_isp[0]);
        cudaMalloc(&s_neighbor_d, sizeof(int) * num_clus_isp[0]);
        cudaMalloc(&s_ransac_d, sizeof(int) * num_clus_isp[0]);
        cudaMalloc(&flag_d, sizeof(bool));

        setup_D4(isp_im_clus_d, isp_im_D4_d);
    }

    ~HBP_ISP()
    {
        cudaFree(flag_d);
        cudaFree(s_ransac_d);
        cudaFree(s_neighbor_d);
        cudaFree(s_child_d);
        cudaFree(label_img_d);
        cudaFree(disp_img_d);
        cudaFree(min_plane_d);
        cudaFree(min_belief_d);
        cudaFree(pos_scan_d);
        cudaFree(predicate_d);
        cudaFree(plane_cfd_d);
        cudaFree(inlier_ratio_d);
        cudaFree(order_d);
        cudaFree(bds_d);
        cudaFree(bd_buf_d);
        cudaFree(KL_d);
        cudaFree(msg_t_d);
        cudaFree(msg_d);
        cudaFree(plane_buf_d);
        cudaFree(im_plane_cost_d);
        for (int i = 0; i < isp_levels; ++i)
        {
            cudaFree(isp_reverse_id_d[i]);
            cudaFree(isp_nb_num_pos_d[i]);
            cudaFree(isp_im_clus_num_pos_d[i]);
            cudaFree(isp_clus_num_pos_d[i]);
            cudaFree(isp_clus_sort_d[i]);
            cudaFree(isp_dc_d[i]);
            cudaFree(isp_point_id_d[i]);
            cudaFree(isp_im_D4_d[i]);
            cudaFree(isp_plane_d[i]);
            cudaFree(isp_belief_d[i]);
            cudaFree(isp_cost_d[i]);
            cudaFree(isp_im_clus_d[i]);
        }
        delete[] isp_reverse_id_d;
        delete[] isp_bdl_d;
        delete[] isp_nb_num_pos_d;
        delete[] isp_im_clus_num_pos_d;
        delete[] isp_clus_num_pos_d;
        delete[] isp_clus_sort_d;
        delete[] isp_dc_d;
        delete[] isp_bd_d;
        delete[] isp_point_id_d;
        delete[] isp_im_D4_d;
        delete[] isp_plane_d;
        delete[] isp_belief_d;
        delete[] isp_cost_d;
        delete[] isp_mean_d;
        delete[] isp_im_clus_d;
        delete[] isp_clus_d;
    }
};

#endif
