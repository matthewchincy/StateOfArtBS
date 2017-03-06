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
#include "PixelBasedAdaptiveSegmenter.h"
#include "PBAS.h"
#include <iostream>
#include <fstream>

PixelBasedAdaptiveSegmenterBGS::PixelBasedAdaptiveSegmenterBGS() : firstTime(true), enableInputBlur(true), enableOutputBlur(true),
alpha(7.0), beta(1.0), N(20), Raute_min(2), R_incdec(0.05), R_lower(18),
R_scale(5), T_dec(0.05), T_inc(1), T_init(18), T_lower(2), T_upper(200)
{
}
PixelBasedAdaptiveSegmenterBGS::PixelBasedAdaptiveSegmenterBGS(bool enableInputBlur,
	bool enableOutputBlur, float alpha, float beta, int N, int Raute_min,
	float R_incdec, int R_lower, int R_scale, float T_dec, int T_inc, int T_init, int T_lower, int T_upper)
	: firstTime(true), enableInputBlur(enableInputBlur), enableOutputBlur(enableOutputBlur),
alpha(alpha), beta(beta), N(N), Raute_min(Raute_min), R_incdec(R_incdec), R_lower(R_lower),
R_scale(R_scale), T_dec(T_dec), T_inc(T_inc), T_init(T_init), T_lower(T_lower), T_upper(T_upper)
{
}
PixelBasedAdaptiveSegmenterBGS::~PixelBasedAdaptiveSegmenterBGS()
{
}

void PixelBasedAdaptiveSegmenterBGS::process(const cv::Mat &img_input, cv::Mat &img_output, cv::Mat &img_bgmodel)
{
  if(img_input.empty())
    return;

  if(firstTime)
  {
	  pPBAS = new PBAS(alpha, beta, N, Raute_min, R_incdec, R_lower, R_scale, T_dec, T_inc, T_init, T_lower, T_upper);

  }

  cv::Mat img_input_new;
  if(enableInputBlur)
    cv::GaussianBlur(img_input, img_input_new, cv::Size(5,5), 1.5);
  else
    img_input.copyTo(img_input_new);

  cv::Mat img_foreground;
  (*pPBAS)(img_input_new, img_foreground);

  if(enableOutputBlur)
    cv::medianBlur(img_foreground, img_foreground, 5);

  img_foreground.copyTo(img_output);

  firstTime = false;
}

// Save parameters
void PixelBasedAdaptiveSegmenterBGS::SaveParameter(std::string folderName) {
	std::ofstream myfile;
	myfile.open(folderName + "/parameter.txt", std::ios::app);
	myfile << "\n\n----METHOD THRESHOLD----" << std::endl;
	myfile << "Enable input blur: " << enableInputBlur << std::endl;
	myfile << "Enable output blur: " << enableOutputBlur << std::endl;
	myfile << "Alpha: " << alpha << std::endl;
	myfile << "Beta: " << beta << std::endl;
	myfile << "N: " << N << std::endl;
	myfile << "Raute min: " << Raute_min << std::endl;
	myfile << "R incdec: " << R_incdec << std::endl;
	myfile << "R lower: " << R_lower << std::endl;
	myfile << "R scale: " << R_scale << std::endl;
	myfile << "T dec: " << T_dec << std::endl;
	myfile << "T inc: " << T_inc << std::endl;
	myfile << "T init: " << T_init << std::endl;
	myfile << "T lower: " << T_lower << std::endl;
	myfile << "T upper: " << T_upper << std::endl;
	myfile.close();
}