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
#include "T2FMRF_UM.h"
#include <iostream>
#include <fstream>

T2FMRF_UM::T2FMRF_UM() : firstTime(true), frameNumber(0), threshold(9.0), alpha(0.01), 
km(2.f), kv(0.9f), gaussians(3)
{
}
T2FMRF_UM::T2FMRF_UM(long frameNumber, double threshold, double alpha, float km, float kv, int gaussians)
	: firstTime(true), frameNumber(frameNumber), threshold(threshold), alpha(alpha),
	km(km), kv(kv), gaussians(gaussians)
{
}
T2FMRF_UM::~T2FMRF_UM()
{
}

void T2FMRF_UM::process(const cv::Mat &img_input, cv::Mat &img_output, cv::Mat &img_bgmodel)
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
    params.HighThreshold() = 2*params.LowThreshold();
    params.Alpha() = alpha;
    params.MaxModes() = gaussians;
    params.Type() = TYPE_T2FMRF_UM;
    params.KM() = km; // Factor control for the T2FMRF-UM [0,3] default: 2
    params.KV() = kv; // Factor control for the T2FMRF-UV [0.3,1] default: 0.9

    bgs.Initalize(params);
    bgs.InitModel(frame_data);

    old_labeling = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 1);
    old = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 1);

    mrf.height = height;
	  mrf.width = width;
    mrf.Build_Classes_OldLabeling_InImage_LocalEnergy();

    firstTime = false;
  }

  bgs.Subtract(frameNumber, frame_data, lowThresholdMask, highThresholdMask);
  cvCopy(lowThresholdMask.Ptr(), old);

  /************************************************************************/
	/* the code for MRF, it can be noted when using other methods   */
	/************************************************************************/
	//the optimization process is done when the foreground detection is stable, 
	if(frameNumber >= 10)
	{
		gmm = bgs.gmm();
		hmm = bgs.hmm();
		mrf.background2 = frame_data.Ptr();
		mrf.in_image = lowThresholdMask.Ptr();
		mrf.out_image = lowThresholdMask.Ptr();
		mrf.InitEvidence2(gmm,hmm,old_labeling);
		mrf.ICM2();
		cvCopy(mrf.out_image, lowThresholdMask.Ptr());
	}

  cvCopy(old, old_labeling);

  lowThresholdMask.Clear();
  bgs.Update(frameNumber, frame_data, lowThresholdMask);
  
  cv::Mat foreground(highThresholdMask.Ptr());
  
  foreground.copyTo(img_output);

  delete frame;
  frameNumber++;
}

// Save parameters
void T2FMRF_UM::SaveParameter(std::string folderName) {
	std::ofstream myfile;
	myfile.open(folderName + "/parameter.txt", std::ios::app);
	myfile << "\n\n----METHOD THRESHOLD----" << std::endl;
	myfile << "Threshold: " << threshold << std::endl;
	myfile << "Alpha: " << alpha << std::endl;
	myfile << "km: " << km << std::endl;
	myfile << "kv: " << kv << std::endl;
	myfile << "Gaussians: " << gaussians << std::endl;

	myfile.close();
}
