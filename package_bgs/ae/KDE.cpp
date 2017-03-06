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
#include <iostream>
#include <fstream>
#include "KDE.h"


KDE::KDE() : SequenceLength(50), TimeWindowSize(100), SDEstimationFlag(1), lUseColorRatiosFlag(1),
  th(10e-8), alpha(0.3), framesToLearn(10), frameNumber(0), firstTime(true)
{
  p = new NPBGSubtractor;
}
KDE::KDE(int SequenceLength, int TimeWindowSize, int SDEstimationFlag, int lUseColorRatiosFlag, double th, double alpha,
	int framesToLearn, int frameNumber)
	: SequenceLength(SequenceLength), TimeWindowSize(TimeWindowSize), SDEstimationFlag(SDEstimationFlag), lUseColorRatiosFlag(lUseColorRatiosFlag),
th(th), alpha(alpha), framesToLearn(framesToLearn), frameNumber(frameNumber), firstTime(true)
{

	p = new NPBGSubtractor;
}
KDE::~KDE()
{
  delete FGImage;
  delete p;
}

void KDE::process(const cv::Mat &img_input, cv::Mat &img_output, cv::Mat &img_bgmodel)
{
  if(img_input.empty())
    return;


  if(firstTime)
  {
    rows = img_input.size().height;
    cols = img_input.size().width;
    color_channels = img_input.channels();

    // SequenceLength: number of samples for each pixel.
    // TimeWindowSize: Time window for sampling. for example in the call above, the bg will sample 50 points out of 100 frames. 
    // this rate will affect how fast the model adapt.
    // SDEstimationFlag: True means to estimate suitable kernel bandwidth to each pixel, False uses a default value.
    // lUseColorRatiosFlag: True means use normalized RGB for color (recommended.)
    p->Intialize(rows,cols,color_channels,SequenceLength,TimeWindowSize,SDEstimationFlag,lUseColorRatiosFlag);
    // th: 0-1 is the probability threshold for a pixel to be a foregroud. typically make it small as 10e-8. the smaller the value the less false positive and more false negative.
    // alpha: 0-1, for color. typically set to 0.3. this affect shadow suppression.
    p->SetThresholds(th,alpha);

    FGImage = new unsigned char[rows*cols];
    //FilteredFGImage = new unsigned char[rows*cols];
    FilteredFGImage = 0;
    DisplayBuffers = 0;

    img_foreground = cv::Mat::zeros(rows,cols,CV_8UC1);

    frameNumber = 0;
    firstTime = false;
  }

  // Stores the first N frames to build the background model
  if(frameNumber < framesToLearn)
  {
    p->AddFrame(img_input.data);
    frameNumber++;
    return;
  }

  // Build the background model with first 10 frames
  if(frameNumber == framesToLearn)
  {
    p->Estimation();
    frameNumber++;
  }

  // Now, we can subtract the background
  ((NPBGSubtractor *)p)->NBBGSubtraction(img_input.data,FGImage,FilteredFGImage,DisplayBuffers);
  
  // At each frame also you can call the update function to adapt the bg
  // here you pass a mask where pixels with true value will be masked out of the update.
  ((NPBGSubtractor *)p)->Update(FGImage);

  img_foreground.data = FGImage;

  img_foreground.copyTo(img_output);
}

// Save parameters
void KDE::SaveParameter(std::string folderName) {
	std::ofstream myfile;
	myfile.open(folderName + "/parameter.txt", std::ios::app);
	myfile << "\n\n----METHOD THRESHOLD----" << std::endl;
	myfile << "Frames to learn: " << framesToLearn << std::endl;
	myfile << "Sequence length: " << SequenceLength << std::endl;
	myfile << "Time window size: " << TimeWindowSize << std::endl;
	myfile << "SD estimation flag: " << SDEstimationFlag << std::endl;
	myfile << "l use color ratio flag: " << lUseColorRatiosFlag << std::endl;
	myfile << "Th: " << th << std::endl;
	myfile << "Alpha: " << alpha << std::endl;

	myfile.close();
}