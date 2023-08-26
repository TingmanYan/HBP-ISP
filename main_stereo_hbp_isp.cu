#include "hbp_isp.hpp"
#include "segment.hpp"
#include "stereo_cost.hpp"
#include "post_process.hpp"

void medianFilter(cv::Mat &);

int main(int argc, char **argv)
{
	if (argc != 13 && argc != 14)
	{
		std::cerr << "usage: " << argv[0]
				  << " img_left img_right disp_labels linkage num_nb sigma alpha tau_smooth "
					 "out_name use_cost_vol min_level cost_name (cost_vol) \n"
					 "linkage: 0 - MinLink, 1 - MaxLink, 2- CentoridLink, 3 - "
					 "WardLink\n";
		exit(1);
	}
	// load input as color images
	cv::Mat img_left = cv::imread(argv[1], cv::IMREAD_COLOR);
	cv::Mat img_right = cv::imread(argv[2], cv::IMREAD_COLOR);
	// cv::resize(img_left, img_left, cv::Size(342, 228));
	// cv::resize(img_right, img_right, cv::Size(342, 228));
	int disp_labels = atoi(argv[3]);

	Linkage link = static_cast<Linkage>(atoi(argv[4]));
	int num_nb = atoi(argv[5]);
	double sigma = atof(argv[6]);
	const int target_clus = 1;

	cudaSetDevice(0);
	cudaDeviceSynchronize();
	SegHAC *seg_hac = new SegHAC(img_left, link, num_nb, sigma, LabelISPFormat,
								 target_clus, false);
	cudaDeviceSynchronize();
	auto c0 = std::chrono::steady_clock::now();
	seg_hac->run_ms();
	cudaDeviceSynchronize();
	auto c1 = std::chrono::steady_clock::now();
	std::cout << "time for hierarchical segmentation: "
			  << std::chrono::duration_cast<std::chrono::milliseconds>(c1 - c0).count() << " ms" << std::endl;
	ISPData *isp_data = new ISPData(seg_hac);
	delete seg_hac;

	std::string cost_name(argv[12]);
	std::string vol_dir = "";
	if (argc == 14)
		vol_dir = std::string(argv[13]);
	StereoCost *stereo_cost = new StereoCost(img_left, img_right, disp_labels, vol_dir);
	cudaDeviceSynchronize();
	c0 = std::chrono::steady_clock::now();
	stereo_cost->compute_stereo_cost(cost_name);
	cudaDeviceSynchronize();
	c1 = std::chrono::steady_clock::now();
	std::cout << "time for matching cost computation: "
			  << std::chrono::duration_cast<std::chrono::milliseconds>(c1 - c0).count() << " ms" << std::endl;
	bool use_cost_vol = atoi(argv[10]);
	StereoData *stereo_data = new StereoData(stereo_cost, use_cost_vol);
	delete stereo_cost;

	float alpha = atof(argv[7]);
	float tau_smooth = atof(argv[8]);
	int min_level = atoi(argv[11]);
	HBP_ISP *hbp_isp = new HBP_ISP(isp_data, stereo_data, alpha, tau_smooth);
	cudaDeviceSynchronize();
	c0 = std::chrono::steady_clock::now();
	auto results_left = hbp_isp->run_ms(min_level);
	cv::Mat disp_left = results_left.first;
	cudaDeviceSynchronize();
	c1 = std::chrono::steady_clock::now();
	std::cout << "time for beilef propagation on ISP: "
			  << std::chrono::duration_cast<std::chrono::milliseconds>(c1 - c0).count() << " ms" << std::endl;

	cv::String out_name(argv[9]);
	medianFilter(disp_left);
	cvutils::io::save_pfm_file(out_name + "_hbp_isp_disp.pfm", disp_left);

	delete hbp_isp;
	delete stereo_data;
	delete isp_data;

	// right
	/*
	SegHAC *seg_hac_r = new SegHAC(img_right, link, num_nb, sigma,
									LabelISPFormat, target_clus, false);
	seg_hac_r->run_ms();
	ISPData *isp_data_r = new ISPData(seg_hac_r);
	delete seg_hac_r;

	StereoCost *stereo_cost_r = new StereoCost(img_left, img_right, disp_labels, vol_dir);
	stereo_cost_r->compute_stereo_cost(cost_name);
	stereo_cost_r->cost_to_right();
	StereoData *stereo_data_r = new StereoData(stereo_cost_r,
												use_cost_vol);
	delete stereo_cost_r;

	HBP_ISP *pmbp_isp_r = new HBP_ISP(isp_data_r, stereo_data_r, alpha, tau_smooth);
	auto results_right = pmbp_isp_r->run_ms(min_level);
	cv::Mat disp_right = results_right.first;

	delete pmbp_isp_r;
	delete stereo_data_r;
	delete isp_data_r;

	PostProcess *post_process = new PostProcess(img_left, img_right);
	post_process->left_right_check(results_left, results_right);
	post_process->plane_label_to_disp(results_left, results_right);

	post_process->median_filter(results_left, results_right);
	post_process->plane_label_to_disp(results_left, results_right);

	cv::Mat disp_img = results_left.first.clone();

	cvutils::io::save_pfm_file(out_name + "_hbp_isp_disp_lr.pfm", disp_img);
	*/

	return 0;
}

void medianFilter(cv::Mat &D)
{
    // get disparity image dimensions
    int32_t D_width = D.cols;
    int32_t D_height = D.rows;

    // temporary memory
    cv::Mat D_temp(D.rows, D.cols, CV_32FC1, cv::Scalar::all(0));
    int32_t window_size = 3;

    float *vals = new float[window_size * 2 + 1];
    int32_t i, j;
    float temp;

    // first step: horizontal median filter
    for (int32_t u = window_size; u < D_width - window_size; u++)
    {
        for (int32_t v = window_size; v < D_height - window_size; v++)
        {
            if (D.at<float>(v, u) >= 0)
            {
                j = 0;
                for (int32_t u2 = u - window_size; u2 <= u + window_size; u2++)
                {
                    temp = D.at<float>(v, u2);
                    i = j - 1;
                    while (i >= 0 && *(vals + i) > temp)
                    {
                        *(vals + i + 1) = *(vals + i);
                        i--;
                    }
                    *(vals + i + 1) = temp;
                    j++;
                }
                D_temp.at<float>(v, u) = *(vals + window_size);
            }
            else
            {
                D_temp.at<float>(v, u) = D.at<float>(v, u);
            }
        }
    }

    // second step: vertical median filter
    for (int32_t u = window_size; u < D_width - window_size; u++)
    {
        for (int32_t v = window_size; v < D_height - window_size; v++)
        {
            if (D.at<float>(v, u) >= 0)
            {
                j = 0;
                for (int32_t v2 = v - window_size; v2 <= v + window_size; v2++)
                {
                    temp = D_temp.at<float>(v2, u);
                    i = j - 1;
                    while (i >= 0 && *(vals + i) > temp)
                    {
                        *(vals + i + 1) = *(vals + i);
                        i--;
                    }
                    *(vals + i + 1) = temp;
                    j++;
                }
                D.at<float>(v, u) = *(vals + window_size);
            }
            else
            {
                D.at<float>(v, u) = D.at<float>(v, u);
            }
        }
    }

    delete[] vals;
}