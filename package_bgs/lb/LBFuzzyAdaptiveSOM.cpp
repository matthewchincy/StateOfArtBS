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
#include "LBFuzzyAdaptiveSOM.h"
#include <iostream>
#include <fstream>

LBFuzzyAdaptiveSOM::LBFuzzyAdaptiveSOM() : firstTime(true), sensitivity(90), trainingSensitivity(240), learningRate(38), trainingLearningRate(255), trainingSteps(81)
{
}
LBFuzzyAdaptiveSOM::LBFuzzyAdaptiveSOM(int sensitivity, int trainingSensitivity, int learningRate, int trainingLearningRate, int trainingSteps)
	: firstTime(true), sensitivity(sensitivity), trainingSensitivity(trainingSensitivity), learningRate(learningRate), trainingLearningRate(trainingLearningRate), trainingSteps(81)
{
}
LBFuzzyAdaptiveSOM::~LBFuzzyAdaptiveSOM()
{
	delete m_pBGModel;
}

void LBFuzzyAdaptiveSOM::process(const cv::Mat &img_input, cv::Mat &img_output, cv::Mat &img_bgmodel)
{
	if (img_input.empty())
		return;
	IplImage *frame = new IplImage(img_input);

	if (firstTime)
	{

		int w = cvGetSize(frame).width;
		int h = cvGetSize(frame).height;

		m_pBGModel = new BGModelFuzzySom(w, h);
		m_pBGModel->InitModel(frame);
	}

	m_pBGModel->setBGModelParameter(0, sensitivity);
	m_pBGModel->setBGModelParameter(1, trainingSensitivity);
	m_pBGModel->setBGModelParameter(2, learningRate);
	m_pBGModel->setBGModelParameter(3, trainingLearningRate);
	m_pBGModel->setBGModelParameter(5, trainingSteps);

	m_pBGModel->UpdateModel(frame);

	img_foreground = cv::Mat(m_pBGModel->GetFG());
	img_background = cv::Mat(m_pBGModel->GetBG());

	img_foreground.copyTo(img_output);
	img_background.copyTo(img_bgmodel);

	delete frame;

	firstTime = false;
}

// Save parameters
void LBFuzzyAdaptiveSOM::SaveParameter(std::string folderName) {
	std::ofstream myfile;
	myfile.open(folderName + "/parameter.txt", std::ios::app);
	myfile << "\n\n----METHOD THRESHOLD----" << std::endl;
	myfile << "Sensitivity: " << sensitivity << std::endl;
	myfile << "Training sensitivity: " << trainingSensitivity << std::endl;
	myfile << "Learning rate: " << learningRate << std::endl;
	myfile << "Training learning rate: " << trainingLearningRate << std::endl;
	myfile << "Training steps: " << trainingSteps << std::endl;

	myfile.close();
}