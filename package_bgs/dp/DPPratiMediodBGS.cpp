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
#include "DPPratiMediodBGS.h"
#include <iostream>
#include <fstream>

DPPratiMediodBGS::DPPratiMediodBGS() : firstTime(true), frameNumber(0), threshold(30), samplingRate(5), historySize(16), weight(5)
{
}
DPPratiMediodBGS::DPPratiMediodBGS(long frameNumber, int threshold, int samplingRate, int historySize, int weight) :
	firstTime(true), frameNumber(frameNumber), threshold(threshold), samplingRate(samplingRate), historySize(historySize), weight(weight)
{
}

DPPratiMediodBGS::~DPPratiMediodBGS()
{
}

void DPPratiMediodBGS::process(const cv::Mat &img_input, cv::Mat &img_output, cv::Mat &img_bgmodel)
{
  if(img_input.empty())
    return;

  frame = new IplImage(img_input);
  
  if(firstTime)
    frame_data.ReleaseMemory(false);
  frame_data = frame;

  if(firstTime)
  {
    int width	= img_input.size().width;
    int height = img_input.size().height;

    lowThresholdMask = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 1);
    lowThresholdMask.Ptr()->origin = IPL_ORIGIN_BL;

    highThresholdMask = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 1);
    highThresholdMask.Ptr()->origin = IPL_ORIGIN_BL;

    params.SetFrameSize(width, height);
    params.LowThreshold() = threshold;
    params.HighThreshold() = 2*params.LowThreshold();	// Note: high threshold is used by post-processing 
    params.SamplingRate() = samplingRate;
    params.HistorySize() = historySize;
    params.Weight() = weight;

    bgs.Initalize(params);
    bgs.InitModel(frame_data);
  }

  bgs.Subtract(frameNumber, frame_data, lowThresholdMask, highThresholdMask);
  lowThresholdMask.Clear();
  bgs.Update(frameNumber, frame_data, lowThresholdMask);
  
  cv::Mat foreground(highThresholdMask.Ptr());
  cv::Mat background(bgs.Background()->Ptr());


  foreground.copyTo(img_output);
  background.copyTo(img_bgmodel);

  delete frame;
  firstTime = false;
  frameNumber++;
}

// Save parameters
void DPPratiMediodBGS::SaveParameter(std::string folderName) {
	std::ofstream myfile;
	myfile.open(folderName + "/parameter.txt", std::ios::app);
	myfile << "\n\n----METHOD THRESHOLD----" << std::endl;
	myfile << "Threshold: " << threshold << std::endl;
	myfile << "Sampling Rate: " << samplingRate << std::endl;
	myfile << "History Size: " << historySize << std::endl;
	myfile << "Weight: " << weight << std::endl;
	myfile.close();
}
