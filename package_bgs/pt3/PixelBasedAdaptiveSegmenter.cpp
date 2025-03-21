/*
This file is part of BGSLibrary.

BGSLibrary is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

BGSLibrary is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with BGSLibrary.  If not, see <http://www.gnu.org/licenses/>.
*/
#include "PixelBasedAdaptiveSegmenter.h"

PixelBasedAdaptiveSegmenter::PixelBasedAdaptiveSegmenter() : firstTime(true), showOutput(true), enableInputBlur(true), enableOutputBlur(true),
alpha(7.0), beta(1.0), N(20), Raute_min(2), R_incdec(0.05), R_lower(18),
R_scale(5), T_dec(0.05), T_inc(1), T_init(18), T_lower(2), T_upper(200)
{
	std::cout << "PixelBasedAdaptiveSegmenter()" << std::endl;
}

PixelBasedAdaptiveSegmenter::~PixelBasedAdaptiveSegmenter()
{
	std::cout << "~PixelBasedAdaptiveSegmenter()" << std::endl;
}

void PixelBasedAdaptiveSegmenter::process(const cv::Mat &img_input, cv::Mat &img_output, cv::Mat &img_bgmodel)
{
	if (img_input.empty())
		return;

	loadConfig();

	if (firstTime)
	{
		pbas.setAlpha(alpha);
		pbas.setBeta(beta);
		pbas.setN(N);
		pbas.setRaute_min(Raute_min);
		pbas.setR_incdec(R_incdec);
		pbas.setR_lower(R_lower);
		pbas.setR_scale(R_scale);
		pbas.setT_dec(T_dec);
		pbas.setT_inc(T_inc);
		pbas.setT_init(T_init);
		pbas.setT_lower(T_lower);
		pbas.setT_upper(T_upper);

		saveConfig();
	}

	cv::Mat img_input_new;
	if (enableInputBlur)
		cv::GaussianBlur(img_input, img_input_new, cv::Size(5, 5), 1.5);
	else
		img_input.copyTo(img_input_new);

	cv::Mat img_foreground;
	pbas.process(&img_input_new, &img_foreground);

	if (enableOutputBlur)
		cv::medianBlur(img_foreground, img_foreground, 5);

	if (showOutput)
		cv::imshow("PBAS", img_foreground);

	img_foreground.copyTo(img_output);

	firstTime = false;
}

void PixelBasedAdaptiveSegmenter::saveConfig()
{
	CvFileStorage* fs = cvOpenFileStorage("./config/PixelBasedAdaptiveSegmenter.xml", 0, CV_STORAGE_WRITE);

	cvWriteInt(fs, "enableInputBlur", enableInputBlur);
	cvWriteInt(fs, "enableOutputBlur", enableOutputBlur);

	cvWriteReal(fs, "alpha", alpha);
	cvWriteReal(fs, "beta", beta);
	cvWriteInt(fs, "N", N);
	cvWriteInt(fs, "Raute_min", Raute_min);
	cvWriteReal(fs, "R_incdec", R_incdec);
	cvWriteInt(fs, "R_lower", R_lower);
	cvWriteInt(fs, "R_scale", R_scale);
	cvWriteReal(fs, "T_dec", T_dec);
	cvWriteInt(fs, "T_inc", T_inc);
	cvWriteInt(fs, "T_init", T_init);
	cvWriteInt(fs, "T_lower", T_lower);
	cvWriteInt(fs, "T_upper", T_upper);

	cvWriteInt(fs, "showOutput", showOutput);

	cvReleaseFileStorage(&fs);
}

void PixelBasedAdaptiveSegmenter::loadConfig()
{
	CvFileStorage* fs = cvOpenFileStorage("./config/PixelBasedAdaptiveSegmenter.xml", 0, CV_STORAGE_READ);

	enableInputBlur = cvReadIntByName(fs, 0, "enableInputBlur", true);
	enableOutputBlur = cvReadIntByName(fs, 0, "enableOutputBlur", true);

	alpha = cvReadRealByName(fs, 0, "alpha", 7.0);
	beta = cvReadRealByName(fs, 0, "beta", 1.0);
	N = cvReadIntByName(fs, 0, "N", 20);
	Raute_min = cvReadIntByName(fs, 0, "Raute_min", 2);
	R_incdec = cvReadRealByName(fs, 0, "R_incdec", 0.05);
	R_lower = cvReadIntByName(fs, 0, "R_lower", 18);
	R_scale = cvReadIntByName(fs, 0, "R_scale", 5);
	T_dec = cvReadRealByName(fs, 0, "T_dec", 0.05);
	T_inc = cvReadIntByName(fs, 0, "T_inc", 1);
	T_init = cvReadIntByName(fs, 0, "T_init", 18);
	T_lower = cvReadIntByName(fs, 0, "T_lower", 2);
	T_upper = cvReadIntByName(fs, 0, "T_upper", 200);

	showOutput = cvReadIntByName(fs, 0, "showOutput", true);

	cvReleaseFileStorage(&fs);
}