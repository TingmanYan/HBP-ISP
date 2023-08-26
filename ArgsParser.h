#ifndef __ARGSPARSER_H__
#define __ARGSPARSER_H__

#include <vector>
#include <string>

struct Calib
{
	float cam0[3][3];
	float cam1[3][3];
	float doffs;
	float baseline;
	int width;
	int height;
	int ndisp;
	int isint;
	int vmin;
	int vmax;
	float dyavg;
	float dymax;
				   // ----------- format of calib.txt ----------
				   //cam0 = [2852.758 0 1424.085; 0 2852.758 953.053; 0 0 1]
				   //cam1 = [2852.758 0 1549.445; 0 2852.758 953.053; 0 0 1]
				   //doffs = 125.36
				   //baseline = 178.089
				   //width = 2828
				   //height = 1924
				   //ndisp = 260
				   //isint = 0
				   //vmin = 36
				   //vmax = 218
				   //dyavg = 0.408
				   //dymax = 1.923

	Calib()
		: doffs(0)
		, baseline(0)
		, width(0)
		, height(0)
		, ndisp(0)
		, isint(0)
		, vmin(0)
		, vmax(0)
		, dyavg(0)
		, dymax(0)
	{
	}

	Calib(std::string filename)
		: Calib()
	{
		FILE* fp = fopen(filename.c_str(), "r");
		char buff[512];

		if (fp != nullptr)
		{
			if (fgets(buff, sizeof(buff), fp) != nullptr) sscanf(buff, "cam0 = [%f %f %f; %f %f %f; %f %f %f]\n", &cam0[0][0], &cam0[0][1], &cam0[0][2], &cam0[1][0], &cam0[1][1], &cam0[1][2], &cam0[2][0], &cam0[2][1], &cam0[2][2]);
			if (fgets(buff, sizeof(buff), fp) != nullptr) sscanf(buff, "cam1 = [%f %f %f; %f %f %f; %f %f %f]\n", &cam1[0][0], &cam1[0][1], &cam1[0][2], &cam1[1][0], &cam1[1][1], &cam1[1][2], &cam1[2][0], &cam1[2][1], &cam1[2][2]);
			if (fgets(buff, sizeof(buff), fp) != nullptr) sscanf(buff, "doffs = %f\n", &doffs);
			if (fgets(buff, sizeof(buff), fp) != nullptr) sscanf(buff, "baseline = %f\n", &baseline);
			if (fgets(buff, sizeof(buff), fp) != nullptr) sscanf(buff, "width = %d\n", &width);
			if (fgets(buff, sizeof(buff), fp) != nullptr) sscanf(buff, "height = %d\n", &height);
			if (fgets(buff, sizeof(buff), fp) != nullptr) sscanf(buff, "ndisp = %d\n", &ndisp);
			if (fgets(buff, sizeof(buff), fp) != nullptr) sscanf(buff, "isint = %d\n", &isint);
			if (fgets(buff, sizeof(buff), fp) != nullptr) sscanf(buff, "vmin = %d\n", &vmin);
			if (fgets(buff, sizeof(buff), fp) != nullptr) sscanf(buff, "vmax = %d\n", &vmax);
			if (fgets(buff, sizeof(buff), fp) != nullptr) sscanf(buff, "dyavg = %f\n", &dyavg);
			if (fgets(buff, sizeof(buff), fp) != nullptr) sscanf(buff, "dymax = %f\n", &dymax);
			fclose(fp);
		}
	}
};

#endif
