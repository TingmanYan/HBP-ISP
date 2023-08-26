#ifndef _POST_PROCESS_HPP_
#define _POST_PROCESS_HPP_

#include "util.h"
#include "CRTrees.hpp"
#include <opencv2/opencv.hpp>

class PostProcess
{
public:
    void
    left_right_check (std::pair<cv::Mat, cv::Mat>&,
                      std::pair<cv::Mat, cv::Mat>&);
    void
    median_filter (std::pair<cv::Mat, cv::Mat>&, std::pair<cv::Mat, cv::Mat>&);
    void
    median_filter_gpu (std::pair<cv::Mat, cv::Mat>&,
                       std::pair<cv::Mat, cv::Mat>&);
    void
    plane_label_to_disp (std::pair<cv::Mat, cv::Mat>&,
                         std::pair<cv::Mat, cv::Mat>&);
private:
    void
    doConsistencyCheck (const cv::Mat&, const cv::Mat&, cv::Mat&, cv::Mat&,
                        double dispThreshold = 1.5);
    float
    computePatchWeight (const cv::Point&, const cv::Point&, int mode = 0) const;

private:
    cv::Mat I[2];
    cv::Mat fail[2];

    const double lrc_thresh = 1.5;
    const float omega = 10.0f;
    const int windR = 20;
    const int width;
    const int height;
    const int im_size;
    const cv::Rect imageDomain;

    const int m_block = 32;
    const dim3 m_blocks = dim3 (32, 2);

public:
    // TODO:
    // a naive implementation of weighted median filter
    PostProcess (const cv::Mat &imL, const cv::Mat &imR) :
            width (imL.cols), height (imL.rows), im_size (width * height), imageDomain (
                    0, 0, imL.cols, imL.rows)
    {
        I[0] = cv::Mat (width, height, CV_32FC3);
        I[1] = cv::Mat (width, height, CV_32FC3);
        imL.convertTo (I[0], I[0].type ());
        imR.convertTo (I[1], I[1].type ());
    }
    ~PostProcess ()
    {
        ;
    }

};

#endif
