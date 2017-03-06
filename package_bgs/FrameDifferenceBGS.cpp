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
#include "FrameDifferenceBGS.h"
#include <iostream>
#include <fstream>

FrameDifferenceBGS::FrameDifferenceBGS() : firstTime(true), enableThreshold(true), threshold(15)
{
}
FrameDifferenceBGS::FrameDifferenceBGS(bool enableThreshold,int threshold)
	: firstTime(false), enableThreshold(enableThreshold), threshold(threshold)
{
}
FrameDifferenceBGS::~FrameDifferenceBGS()
{
}

void FrameDifferenceBGS::process(const cv::Mat &img_input, cv::Mat &img_output, cv::Mat &img_bgmodel)
{
	if (img_input.empty())
		return;

	if (img_input_prev.empty())
	{
		img_input.copyTo(img_input_prev);
		return;
	}

	cv::absdiff(img_input_prev, img_input, img_foreground);

	if (img_foreground.channels() == 3)
		cv::cvtColor(img_foreground, img_foreground, CV_BGR2GRAY);

	if (enableThreshold)
		cv::threshold(img_foreground, img_foreground, threshold, 255, cv::THRESH_BINARY);

	img_foreground.copyTo(img_output);

	img_input.copyTo(img_input_prev);

	firstTime = false;
}

// Save parameters
void FrameDifferenceBGS::SaveParameter(std::string folderName) {
	std::ofstream myfile;
	myfile.open(folderName + "/parameter.txt", std::ios::app);
	myfile << "\n\n----METHOD THRESHOLD----" << std::endl;
	myfile << "Enable threshold: " << enableThreshold << std::endl;
	myfile << "Threshold: " << threshold << std::endl;
	myfile.close();
}