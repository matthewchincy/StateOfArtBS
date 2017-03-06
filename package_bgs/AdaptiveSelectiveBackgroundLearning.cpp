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
#include "AdaptiveSelectiveBackgroundLearning.h"
#include <iostream>
#include <fstream>

AdaptiveSelectiveBackgroundLearning::AdaptiveSelectiveBackgroundLearning() : firstTime(true),
alphaLearn(0.05), alphaDetection(0.05), learningFrames(-1), counter(0), minVal(0.0), maxVal(1.0),
threshold(15)
{
}
AdaptiveSelectiveBackgroundLearning::AdaptiveSelectiveBackgroundLearning(double alphaLearn, double alphaDetection, long learningFrames, long counter,
	double minVal, double maxVal, int threshold)
	: firstTime(false), alphaLearn(alphaLearn), alphaDetection(alphaDetection), learningFrames(learningFrames),
	counter(counter), minVal(minVal), maxVal(maxVal), threshold(threshold)
{
}
AdaptiveSelectiveBackgroundLearning::~AdaptiveSelectiveBackgroundLearning()
{
}

void AdaptiveSelectiveBackgroundLearning::process(const cv::Mat &img_input_, cv::Mat &img_output, cv::Mat &img_bgmodel)
{
	if (img_input_.empty())
		return;

	cv::Mat img_input;
	if (img_input_.channels() == 3)
		cv::cvtColor(img_input_, img_input, CV_BGR2GRAY);
	else
		img_input_.copyTo(img_input);

	if (img_background.empty())
		img_input.copyTo(img_background);

	cv::Mat img_input_f(img_input.size(), CV_32F);
	img_input.convertTo(img_input_f, CV_32F, 1. / 255.);

	cv::Mat img_background_f(img_background.size(), CV_32F);
	img_background.convertTo(img_background_f, CV_32F, 1. / 255.);

	cv::Mat img_diff_f(img_input.size(), CV_32F);
	cv::absdiff(img_input_f, img_background_f, img_diff_f);

	cv::Mat img_foreground(img_input.size(), CV_8U);
	img_diff_f.convertTo(img_foreground, CV_8U, 255.0 / (maxVal - minVal), -minVal);

	cv::threshold(img_foreground, img_foreground, threshold, 255, cv::THRESH_BINARY);
	cv::medianBlur(img_foreground, img_foreground, 3);

	if (learningFrames > 0 && counter <= learningFrames)
	{
		// Only Adaptive update of the background model
		img_background_f = alphaLearn * img_input_f + (1 - alphaLearn) * img_background_f;
		counter++;
	}
	else
	{
		int rows = img_input.rows;
		int cols = img_input.cols;

		for (int i = 0; i < rows; i++)
		{
			for (int j = 0; j < cols; j++)
			{
				// Adaptive and Selective update of the background model
				if (img_foreground.at<uchar>(i, j) == 0)
				{
					img_background_f.at<float>(i, j) = alphaDetection * img_input_f.at<float>(i, j) + (1 - alphaDetection) * img_background_f.at<float>(i, j);
				}
			}
		}
	}

	cv::Mat img_new_background(img_input.size(), CV_8U);
	img_background_f.convertTo(img_new_background, CV_8U, 255.0 / (maxVal - minVal), -minVal);
	img_new_background.copyTo(img_background);

	img_foreground.copyTo(img_output);
	img_background.copyTo(img_bgmodel);

	firstTime = false;
}

// Save parameters
void AdaptiveSelectiveBackgroundLearning::SaveParameter(std::string folderName) {
	std::ofstream myfile;
	myfile.open(folderName + "/parameter.txt", std::ios::app);
	myfile << "\n\n----METHOD THRESHOLD----" << std::endl;
	myfile << "Learning frames: " << learningFrames << std::endl;
	myfile << "Alpha learn: " << alphaLearn << std::endl;
	myfile << "Alpha detection: " << alphaDetection << std::endl;
	myfile << "Threshold: " << threshold << std::endl;
	myfile.close();
}