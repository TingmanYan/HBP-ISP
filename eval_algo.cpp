#include "ArgsParser.h"
#include "Evaluator.h"
#include "cvutils.hpp"
#include <sys/stat.h>
#include <xmmintrin.h>

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

void eval_MidV3(const std::string input_dir, const std::string output_dir, const std::string algo_name, double errorThresh)
{
    cv::Mat im0, im1, disp_WTA, dispGT, nonocc;
    Calib calib;

    if (!loadData(input_dir, im0, im1, dispGT, nonocc, calib))
        return;
    printf("ndisp = %d\n", calib.ndisp);

    int maxdisp = calib.ndisp;
    if (cvutils::contains(input_dir, "trainingQ") || cvutils::contains(input_dir, "testQ"))
        errorThresh = errorThresh / 2.0;
    else if (cvutils::contains(input_dir, "trainingF") || cvutils::contains(input_dir, "testF"))
        errorThresh = errorThresh * 2.0;

    {
        mkdir((output_dir + "debug").c_str(), 0755);

        Evaluator *eval = new Evaluator(dispGT, nonocc, "result",
                                        output_dir + "debug/");
        eval->setErrorThreshold(errorThresh);
        eval->start();

        cv::Mat disp_img = cvutils::io::read_pfm_file(output_dir + "disp0" + algo_name + ".pfm");

        if (cvutils::contains(input_dir, "training"))
            eval->evaluate(disp_img, true, true);

        delete eval;
    }
}

int main(int argc, char **argv)
{
    if (argc != 5)
    {
        std::cerr << "usage: " << argv[0]
                  << " input_dir output_dir thresh algo_name\n";
        exit(1);
    }
    std::string input_dir(argv[1]);
    std::string output_dir(argv[2]);
    double errorThresh = atof(argv[3]);
    std::string algo_name(argv[4]);

    std::cout << "evaluating " << algo_name << " on " << input_dir << std::endl;

    if (output_dir.length())
        mkdir((output_dir).c_str(), 0755);

    eval_MidV3(input_dir, output_dir, algo_name, errorThresh);

    return 0;
}
