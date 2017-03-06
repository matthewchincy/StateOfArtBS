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
#include "FuzzyChoquetIntegral.h"
#include <opencv2/legacy/compat.hpp>
#include <iostream>
#include <fstream>

FuzzyChoquetIntegral::FuzzyChoquetIntegral() : firstTime(true), frameNumber(0),
framesToLearn(10), alphaLearn(0.1), alphaUpdate(0.01), colorSpace(1), option(2), smooth(true), threshold(0.67)
{
}
FuzzyChoquetIntegral::FuzzyChoquetIntegral(long frameNumber, int framesToLearn,
	double alphaLearn, double alphaUpdate, int colorSpace, int option,
	bool smooth, double threshold)
	: firstTime(true), frameNumber(frameNumber),
	framesToLearn(framesToLearn), alphaLearn(alphaLearn), alphaUpdate(alphaUpdate), colorSpace(colorSpace), option(option), smooth(smooth), threshold(threshold)
{
}
FuzzyChoquetIntegral::~FuzzyChoquetIntegral()
{
}

void FuzzyChoquetIntegral::process(const cv::Mat &img_input, cv::Mat &img_output, cv::Mat &img_bgmodel)
{
	if (img_input.empty())
		return;

	cv::Mat img_input_f3(img_input.size(), CV_32F);
	img_input.convertTo(img_input_f3, CV_32F, 1. / 255.);

	if (firstTime)
	{
		std::string colorSpaceName = "";
		switch (colorSpace)
		{
		case 1: colorSpaceName = "RGB";  break;
		case 2: colorSpaceName = "OHTA"; break;
		case 3: colorSpaceName = "HSV";  break;
		case 4: colorSpaceName = "YCrCb"; break;
		}
	}

	if (frameNumber <= framesToLearn)
	{

		if (img_background_f3.empty())
			img_input_f3.copyTo(img_background_f3);
		else
			img_background_f3 = alphaLearn*img_input_f3 + (1 - alphaLearn)*img_background_f3;
	}
	else
	{
		cv::Mat img_input_f1;
		cv::cvtColor(img_input_f3, img_input_f1, CV_BGR2GRAY);

		cv::Mat img_background_f1;
		cv::cvtColor(img_background_f3, img_background_f1, CV_BGR2GRAY);

		IplImage* input_f3 = new IplImage(img_input_f3);
		IplImage* input_f1 = new IplImage(img_input_f1);
		IplImage* background_f3 = new IplImage(img_background_f3);
		IplImage* background_f1 = new IplImage(img_background_f1);

		IplImage* lbp_input_f1 = cvCreateImage(cvSize(input_f1->width, input_f1->height), IPL_DEPTH_32F, 1);
		cvFillImage(lbp_input_f1, 0.0);
		fu.LBP(input_f1, lbp_input_f1);

		IplImage* lbp_background_f1 = cvCreateImage(cvSize(background_f1->width, background_f1->height), IPL_DEPTH_32F, 1);
		cvFillImage(lbp_background_f1, 0.0);
		fu.LBP(background_f1, lbp_background_f1);

		IplImage* sim_texture_f1 = cvCreateImage(cvSize(input_f1->width, input_f1->height), IPL_DEPTH_32F, 1);
		fu.SimilarityDegreesImage(lbp_input_f1, lbp_background_f1, sim_texture_f1, 1, colorSpace);

		IplImage* sim_color_f3 = cvCreateImage(cvSize(input_f3->width, input_f3->height), IPL_DEPTH_32F, 3);
		fu.SimilarityDegreesImage(input_f3, background_f3, sim_color_f3, 3, colorSpace);

		float* measureG = (float*)malloc(3 * (sizeof(float)));
		IplImage* integral_choquet_f1 = cvCreateImage(cvSize(input_f1->width, input_f1->height), IPL_DEPTH_32F, 1);

		// 3 color components
		if (option == 1)
		{
			fu.FuzzyMeasureG(0.4f, 0.3f, 0.3f, measureG);
			fu.getFuzzyIntegralChoquet(sim_texture_f1, sim_color_f3, option, measureG, integral_choquet_f1);
		}

		// 2 color components + 1 texture component
		if (option == 2)
		{
			fu.FuzzyMeasureG(0.6f, 0.3f, 0.1f, measureG);
			fu.getFuzzyIntegralChoquet(sim_texture_f1, sim_color_f3, option, measureG, integral_choquet_f1);
		}

		free(measureG);
		cv::Mat img_integral_choquet_f1(integral_choquet_f1);

		if (smooth)
			cv::medianBlur(img_integral_choquet_f1, img_integral_choquet_f1, 3);

		cv::Mat img_foreground_f1(img_input.size(), CV_32F);
		cv::threshold(img_integral_choquet_f1, img_foreground_f1, threshold, 255, cv::THRESH_BINARY_INV);

		cv::Mat img_foreground_u1(img_input.size(), CV_8U);
		double minVal = 0., maxVal = 1.;
		img_foreground_f1.convertTo(img_foreground_u1, CV_8U, 255.0 / (maxVal - minVal), -minVal);
		img_foreground_u1.copyTo(img_output);

		cv::Mat img_background_u3(img_input.size(), CV_8U);
		//double minVal = 0., maxVal = 1.;
		img_background_f3.convertTo(img_background_u3, CV_8U, 255.0 / (maxVal - minVal), -minVal);
		img_background_u3.copyTo(img_bgmodel);

		IplImage* updated_background_f3 = cvCreateImage(cvSize(input_f1->width, input_f1->height), IPL_DEPTH_32F, 3);
		cvFillImage(updated_background_f3, 0.0);
		fu.AdaptativeSelectiveBackgroundModelUpdate(input_f3, background_f3, updated_background_f3, integral_choquet_f1, threshold, alphaUpdate);
		cv::Mat img_updated_background_f3(updated_background_f3);
		img_updated_background_f3.copyTo(img_background_f3);

		cvReleaseImage(&lbp_input_f1);
		cvReleaseImage(&lbp_background_f1);
		cvReleaseImage(&sim_texture_f1);
		cvReleaseImage(&sim_color_f3);
		cvReleaseImage(&integral_choquet_f1);
		cvReleaseImage(&updated_background_f3);

		delete background_f1;
		delete background_f3;
		delete input_f1;
		delete input_f3;
	}

	firstTime = false;
	frameNumber++;
}

// Save parameters
void FuzzyChoquetIntegral::SaveParameter(std::string folderName) {
	std::ofstream myfile;
	myfile.open(folderName + "/parameter.txt", std::ios::app);
	myfile << "\n\n----METHOD THRESHOLD----" << std::endl;
	myfile << "Frames to learn: " << framesToLearn << std::endl;
	myfile << "Alpha learn: " << alphaLearn << std::endl;
	myfile << "Alpha update: " << alphaUpdate << std::endl;
	myfile << "Color space: " << colorSpace << std::endl;
	myfile << "Option: " << option << std::endl;
	myfile << "Smooth: " << smooth << std::endl;
	myfile << "Threshold: " << threshold << std::endl;

	myfile.close();
}

