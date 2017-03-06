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
#include "GMG.h"
#include <iostream>
#include <fstream>

GMG::GMG() : firstTime(true), initializationFrames(20), decisionThreshold(0.7)
{
	cv::initModule_video();
	cv::setUseOptimized(true);
	cv::setNumThreads(8);

	fgbg = cv::Algorithm::create<cv::BackgroundSubtractorGMG>("BackgroundSubtractor.GMG");
}
GMG::GMG(int initializationFrames, double decisionThreshold)
	: firstTime(true), initializationFrames(initializationFrames), decisionThreshold(decisionThreshold)
{
	cv::initModule_video();
	cv::setUseOptimized(true);
	cv::setNumThreads(8);

	fgbg = cv::Algorithm::create<cv::BackgroundSubtractorGMG>("BackgroundSubtractor.GMG");
}
GMG::~GMG()
{
}

void GMG::process(const cv::Mat &img_input, cv::Mat &img_output, cv::Mat &img_bgmodel)
{
	if (img_input.empty())
		return;


	if (firstTime)
	{
		fgbg->set("initializationFrames", initializationFrames);
		fgbg->set("decisionThreshold", decisionThreshold);

	}

	if (fgbg.empty())
	{
		std::cerr << "Failed to create BackgroundSubtractor.GMG Algorithm." << std::endl;
		return;
	}

	(*fgbg)(img_input, img_foreground);

	cv::Mat img_background;
	(*fgbg).getBackgroundImage(img_background);

	img_input.copyTo(img_segmentation);
	cv::add(img_input, cv::Scalar(100, 100, 0), img_segmentation, img_foreground);


	img_foreground.copyTo(img_output);
	img_background.copyTo(img_bgmodel);

	firstTime = false;
}

// Save parameters
void GMG::SaveParameter(std::string folderName) {
	std::ofstream myfile;
	myfile.open(folderName + "/parameter.txt", std::ios::app);
	myfile << "\n\n----METHOD THRESHOLD----" << std::endl;
	myfile << "Initialization frames: " << initializationFrames << std::endl;
	myfile << "Decision threshold: " << decisionThreshold << std::endl;
	myfile.close();
}
