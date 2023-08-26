#include "ArgsParser.h"
#include "Evaluator.h"
#include "hbp_isp.hpp"
#include "segment.hpp"
#include "stereo_cost.hpp"
#include "post_process.hpp"
#include <sys/stat.h>
#include <xmmintrin.h>

void medianFilter(cv::Mat &);

bool loadData(const std::string input_dir, cv::Mat &im0, cv::Mat &im1,
              cv::Mat &dispGT, cv::Mat &nonocc, Calib &calib)
{
    calib = Calib(input_dir + "calib.txt");
    if (calib.ndisp <= 0)
    {
        printf("ndisp is not speficied.\n");
        return false;
    }

    im0 = cv::imread(input_dir + "im0.png");
    if (im0.empty())
    {
        printf("Image im0.png not found in\n");
        printf("%s\n", input_dir.c_str());
        return false;
    }
    im1 = cv::imread(input_dir + "im1.png");
    if (im1.empty())
    {
        printf("Image im1.png not found in\n");
        printf("%s\n", input_dir.c_str());
        return false;
    }

    dispGT = cvutils::io::read_pfm_file(input_dir + "disp0GT.pfm");
    if (dispGT.empty())
        dispGT = cv::Mat_<float>::zeros(im0.size());

    nonocc = cv::imread(input_dir + "mask0nocc.png", cv::IMREAD_GRAYSCALE);
    if (!nonocc.empty())
        nonocc = nonocc == 255;
    else
        nonocc = cv::Mat_<uchar>(im0.size(), 255);

    return true;
}

void MidV3(const std::string input_dir, const std::string output_dir, char **argv)
{
    cv::Mat im0, im1, disp_WTA, dispGT, nonocc;
    Calib calib;

    if (!loadData(input_dir, im0, im1, dispGT, nonocc, calib))
        return;
    printf("ndisp = %d\n", calib.ndisp);

    int maxdisp = calib.ndisp;
    double errorThresh = 1.0;
    if (cvutils::contains(input_dir, "trainingQ") || cvutils::contains(input_dir, "testQ"))
        errorThresh = errorThresh / 2.0;
    else if (cvutils::contains(input_dir, "trainingF") || cvutils::contains(input_dir, "testF"))
        errorThresh = errorThresh * 2.0;

    Linkage link = static_cast<Linkage>(atoi(argv[3]));
    int num_nb = atoi(argv[4]);
    double sigma = atof(argv[5]);
    const int target_clus = 1;
    float alpha = atof(argv[6]);
    float tau_smooth = atof(argv[7]);
    std::string cost_name(argv[8]);
    std::string vol_dir = input_dir + "im0.acrt";
    bool use_cost_vol = atoi(argv[9]);
    int min_level = atoi(argv[10]);

    {
        mkdir((output_dir + "debug").c_str(), 0755);

        Evaluator *eval = new Evaluator(dispGT, nonocc, "result",
                                        output_dir + "debug/");
        eval->setErrorThreshold(errorThresh);
        eval->start();

        SegHAC *seg_hac = new SegHAC(im0, link, num_nb, sigma, LabelISPFormat,
                                     target_clus, false);
        seg_hac->run_ms();
        ISPData *isp_data = new ISPData(seg_hac);
        delete seg_hac;

        StereoCost *stereo_cost = new StereoCost(im0, im1, maxdisp, vol_dir);
        stereo_cost->compute_stereo_cost(cost_name);
        StereoData *stereo_data = new StereoData(stereo_cost, use_cost_vol);
        delete stereo_cost;

        HBP_ISP *hbp_isp = new HBP_ISP(isp_data, stereo_data, alpha, tau_smooth);
        auto results = hbp_isp->run_ms_vis(min_level);
        for (int l = 0; l < results.size(); ++l)
        {
            int level = isp_data->g_num_clus_isp().size() - l;
            cv::Mat disp_img = std::get<0>(results[l]);
            cv::Mat label_img = std::get<1>(results[l]);
            cv::Mat spx_img = std::get<2>(results[l]);
            cvutils::io::save_pfm_file(output_dir + "disp0HBP_ISP_" + std::to_string(level) + ".pfm", disp_img);
            cv::imwrite(output_dir + "label0HBP_ISP_" + std::to_string(level) + ".tiff", label_img);
            cv::imwrite(output_dir + "spx0HBP_ISP_" + std::to_string(level) + ".png", spx_img);
            if (cvutils::contains(input_dir, "training"))
                eval->evaluate(disp_img, true, true, level);
        }

        {
            FILE *fp = fopen((output_dir + "timeHBP_ISP.txt").c_str(), "w");
            if (fp != nullptr)
            {
                fprintf(fp, "%lf\n", eval->getCurrentTime());
                fclose(fp);
            }
        }

        delete hbp_isp;
        delete stereo_data;
        delete isp_data;
        delete eval;
    }
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

int main(int argc, char **argv)
{
    if (argc != 11)
    {
        std::cerr << "usage: " << argv[0]
                  << " input_dir output_dir linkage num_nb sigma alpha tau_smooth "
                     "cost_name use_cost_vol min_level\n"
                     "linkage: 0 - MinLink, 1 - MaxLink, 2- CentoridLink, 3 - "
                     "WardLink\n";
        exit(1);
    }
    std::string input_dir(argv[1]);
    std::string output_dir(argv[2]);

    std::cout << "processing: " << input_dir << std::endl;

    if (output_dir.length())
        mkdir((output_dir).c_str(), 0755);

	cudaSetDevice(0);
	cudaDeviceSynchronize();
    MidV3(input_dir, output_dir, argv);

    return 0;
}
