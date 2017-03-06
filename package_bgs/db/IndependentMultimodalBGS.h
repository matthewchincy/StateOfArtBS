#pragma once

#include <opencv2/opencv.hpp>


#include "imbs.hpp"

#include "../IBGS.h"

class IndependentMultimodalBGS : public IBGS
{
private:
  BackgroundSubtractorIMBS* pIMBS;
  int fps;
  bool firstTime;
  cv::Mat img_foreground;
  cv::Mat img_background;
  
public:
  IndependentMultimodalBGS();
  IndependentMultimodalBGS(int fps);
  ~IndependentMultimodalBGS();

  void process(const cv::Mat &img_input, cv::Mat &img_output, cv::Mat &img_bgmodel);
  void SaveParameter(std::string folderName);
};