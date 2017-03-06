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
#include "VuMeter.h"
#include <iostream>
#include <fstream>

VuMeter::VuMeter() : firstTime(true), enableFilter(true), binSize(8), alpha(0.995), threshold(0.03)
{
}
VuMeter::VuMeter( bool enableFilter, int binSize, double alpha, double threshold)
	: firstTime(true),enableFilter(enableFilter), binSize(binSize), alpha(alpha), threshold(threshold)
{
}

VuMeter::~VuMeter()
{
  cvReleaseImage(&mask);
  cvReleaseImage(&background);
  cvReleaseImage(&gray);

}

void VuMeter::process(const cv::Mat &img_input, cv::Mat &img_output, cv::Mat &img_bgmodel)
{
  if(img_input.empty())
    return;
  else
    frame = new IplImage(img_input);


  if(firstTime)
  {
    bgs.SetAlpha(alpha);
    bgs.SetBinSize(binSize);
    bgs.SetThreshold(threshold);

    gray = cvCreateImage(cvGetSize(frame),IPL_DEPTH_8U,1);
    cvCvtColor(frame,gray,CV_RGB2GRAY);

    background = cvCreateImage(cvGetSize(gray),IPL_DEPTH_8U,1);
    cvCopy(gray, background);

    mask = cvCreateImage(cvGetSize(gray),IPL_DEPTH_8U,1);
    cvZero(mask);
  }
  else
    cvCvtColor(frame,gray,CV_RGB2GRAY);
  
  bgs.UpdateBackground(gray,background,mask);
  cv::Mat img_foreground(mask);
  cv::Mat img_bkg(background);

  if(enableFilter)
  {
    cv::erode(img_foreground,img_foreground,cv::Mat());
    cv::medianBlur(img_foreground, img_foreground, 5);
  }


  img_foreground.copyTo(img_output);
  img_bkg.copyTo(img_bgmodel);
  
  delete frame;
  firstTime = false;
}

// Save parameters
void VuMeter::SaveParameter(std::string folderName) {
	std::ofstream myfile;
	myfile.open(folderName + "/parameter.txt", std::ios::app);
	myfile << "\n\n----METHOD THRESHOLD----" << std::endl;
	myfile << "Enable filter: " << enableFilter << std::endl;
	myfile << "Bin size: " << binSize << std::endl;
	myfile << "Alpha: " << alpha << std::endl;
	myfile << "Threshold: " << threshold << std::endl;
	myfile.close();
}
