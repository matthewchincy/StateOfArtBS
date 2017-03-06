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
#include "MixtureOfGaussianV2BGS.h"
#include <iostream>
#include <fstream>
MixtureOfGaussianV2BGS::MixtureOfGaussianV2BGS() : firstTime(true), alpha(0.05), enableThreshold(true), threshold(15)
{
}
MixtureOfGaussianV2BGS::MixtureOfGaussianV2BGS(double alpha, bool enableThreshold, int threshold)
	: firstTime(true), alpha(alpha), enableThreshold(enableThreshold), threshold(threshold)
{
}
MixtureOfGaussianV2BGS::~MixtureOfGaussianV2BGS()
{
}

void MixtureOfGaussianV2BGS::process(const cv::Mat &img_input, cv::Mat &img_output, cv::Mat &img_bgmodel)
{
  if(img_input.empty())
    return;

  //------------------------------------------------------------------
  // BackgroundSubtractorMOG2
  // http://opencv.itseez.com/modules/video/doc/motion_analysis_and_object_tracking.html#backgroundsubtractormog2
  //
  // Gaussian Mixture-based Backbround/Foreground Segmentation Algorithm.
  //
  // The class implements the Gaussian mixture model background subtraction described in:
  //  (1) Z.Zivkovic, Improved adaptive Gausian mixture model for background subtraction, International Conference Pattern Recognition, UK, August, 2004, 
  //  The code is very fast and performs also shadow detection. Number of Gausssian components is adapted per pixel.
  //
  //  (2) Z.Zivkovic, F. van der Heijden, Efficient Adaptive Density Estimation per Image Pixel for the Task of Background Subtraction, 
  //  Pattern Recognition Letters, vol. 27, no. 7, pages 773-780, 2006. 
  //  The algorithm similar to the standard Stauffer&Grimson algorithm with additional selection of the number of the Gaussian components based on: 
  //    Z.Zivkovic, F.van der Heijden, Recursive unsupervised learning of finite mixture models, IEEE Trans. on Pattern Analysis and Machine Intelligence, 
  //    vol.26, no.5, pages 651-656, 2004.
  //------------------------------------------------------------------

  mog(img_input, img_foreground, alpha);
  
  cv::Mat img_background;
  mog.getBackgroundImage(img_background);

  if(enableThreshold)
    cv::threshold(img_foreground, img_foreground, threshold, 255, cv::THRESH_BINARY);


  img_foreground.copyTo(img_output);
  img_background.copyTo(img_bgmodel);

  firstTime = false;
}

// Save parameters
void MixtureOfGaussianV2BGS::SaveParameter(std::string folderName) {
	std::ofstream myfile;
	myfile.open(folderName + "/parameter.txt", std::ios::app);
	myfile << "\n\n----METHOD THRESHOLD----" << std::endl;
	myfile << "Enable threshold: " << enableThreshold << std::endl;
	myfile << "Alpha: " << alpha << std::endl;
	myfile << "Threshold: " << threshold << std::endl;
	myfile.close();
}

