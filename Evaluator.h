#pragma once

#include "TimeStamper.h"
#include <opencv2/opencv.hpp>

class Evaluator
{

protected:
	TimeStamper timer;
	const cv::Mat dispGT;
	const cv::Mat nonoccMask;
	cv::Mat occMask;
	cv::Mat validMask;
	std::string saveDir;
	std::string header;
	int validPixels;
	int nonoccPixels;
	int occPixels;
	FILE *fp_output;
	double errorThreshold;

public:
	bool showProgress;
	bool saveProgress;
	bool printProgress;

	std::string getSaveDirectory()
	{
		return saveDir;
	}

	Evaluator(cv::Mat dispGT, cv::Mat nonoccMask, std::string header = "result", std::string saveDir = "./", bool show = true, bool print = true, bool save = true)
		: dispGT(dispGT), nonoccMask(nonoccMask), header(header), saveDir(saveDir), showProgress(show), saveProgress(save), printProgress(print), fp_output(nullptr)
	{

		if (save)
		{
			fp_output = fopen((saveDir + "log_output.txt").c_str(), "w");
			if (fp_output != nullptr)
			{
				fprintf(fp_output, "%s\t%s\t%s\n", "Time", "all", "nonocc");
				fflush(fp_output);
			}
		}

		errorThreshold = 0.5;

		validMask = (dispGT > 0.0) & (dispGT != INFINITY);
		validPixels = cv::countNonZero(validMask);
		occMask = ~nonoccMask & validMask;
		nonoccPixels = cv::countNonZero(nonoccMask);
		occPixels = cv::countNonZero(occMask);
	}
	~Evaluator()
	{
		if (fp_output != nullptr)
			fclose(fp_output);
	}

	void setErrorThreshold(double t)
	{
		errorThreshold = t;
	}

	void evaluate(cv::Mat disp, bool save, bool print, int level = -1)
	{
		cv::Mat errorMap = cv::abs(disp - dispGT) <= errorThreshold;
		cv::Mat errorMapVis = errorMap | (~validMask);
		cv::Mat occErrorMapVis = errorMapVis.clone();
		errorMapVis.setTo(cv::Scalar(200), occMask & (~errorMapVis));
		occErrorMapVis.setTo(cv::Scalar(200), nonoccMask);

		double all = 1.0 - (double)cv::countNonZero(errorMap & validMask) / validPixels;
		double nonocc = 1.0 - (double)cv::countNonZero(errorMap & nonoccMask) / nonoccPixels;
		double occ = 1.0 - (double)cv::countNonZero(errorMap & occMask) / occPixels;
		all *= 100.0;
		nonocc *= 100.0;
		occ *= 100.0;

		if (saveProgress && save)
		{
			if (level == -1) {
				cv::imwrite(saveDir + cv::format("%s_E.png", header.c_str(), level), errorMapVis);
				cv::imwrite(saveDir + cv::format("%s_occ_E.png", header.c_str(), level), occErrorMapVis);
			}
			else {
				cv::imwrite(saveDir + cv::format("%s_%dE.png", header.c_str(), level), errorMapVis);
				cv::imwrite(saveDir + cv::format("%s_occ_%dE.png", header.c_str(), level), occErrorMapVis);
			}

			if (fp_output != nullptr)
			{
				fprintf(fp_output, "%lf\t%lf\t%lf\t%lf\n", getCurrentTime(), all, nonocc, occ);
				fflush(fp_output);
			}
		}

		if (printProgress && print)
			std::cout << cv::format("%5.1lf\t%4.2lf\t%4.2lf\t%4.2lf", getCurrentTime(), all, nonocc, occ) << std::endl;
	}

	void start()
	{
		timer.start();
	}

	void stop()
	{
		timer.stop();
	}

	double getCurrentTime()
	{
		return timer.getCurrentTime();
	}
};
