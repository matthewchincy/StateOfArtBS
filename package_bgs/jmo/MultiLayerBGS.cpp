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
#include "MultiLayerBGS.h"
#include <iostream>
#include <fstream>

MultiLayerBGS::MultiLayerBGS() : firstTime(true), frameNumber(0), 
saveModel(false), disableDetectMode(true), disableLearning(false), detectAfter(0), bg_model_preload(""), loadDefaultParams(true)
{
}
MultiLayerBGS::MultiLayerBGS(long long frameNumber, bool saveModel, bool disableDetectMode, bool disableLearning, int detectAfter,
	bool loadDefaultParams, int max_mode_num, float weight_updating_constant, float texture_weight, float bg_mode_percent,
	int pattern_neig_half_size, float pattern_neig_gaus_sigma, float bg_prob_threshold, float bg_prob_updating_threshold,
	int robust_LBP_constant, float min_noised_angle, float shadow_rate, float highlight_rate, float bilater_filter_sigma_s,
	float bilater_filter_sigma_r, float frame_duration, float learn_mode_learn_rate_per_second, float learn_weight_learn_rate_per_second,
	float learn_init_mode_weight, float detect_mode_learn_rate_per_second, float detect_weight_learn_rate_per_second, float detect_init_mode_weight)
	: firstTime(true), frameNumber(frameNumber), 
saveModel(saveModel), disableDetectMode(disableDetectMode), disableLearning(disableLearning), detectAfter(detectAfter), bg_model_preload(""),
loadDefaultParams(loadDefaultParams), max_mode_num(max_mode_num), weight_updating_constant(weight_updating_constant), texture_weight(texture_weight), bg_mode_percent(bg_mode_percent),
pattern_neig_half_size(pattern_neig_half_size), pattern_neig_gaus_sigma(pattern_neig_gaus_sigma), bg_prob_threshold(bg_prob_threshold), bg_prob_updating_threshold(bg_prob_updating_threshold),
robust_LBP_constant(robust_LBP_constant), min_noised_angle(min_noised_angle), shadow_rate(shadow_rate), highlight_rate(highlight_rate), bilater_filter_sigma_s(bilater_filter_sigma_s),
bilater_filter_sigma_r(bilater_filter_sigma_r), frame_duration(frame_duration), learn_mode_learn_rate_per_second(learn_mode_learn_rate_per_second), learn_weight_learn_rate_per_second(learn_weight_learn_rate_per_second),
learn_init_mode_weight(learn_init_mode_weight), detect_mode_learn_rate_per_second(detect_mode_learn_rate_per_second), detect_weight_learn_rate_per_second(detect_weight_learn_rate_per_second), detect_init_mode_weight(detect_init_mode_weight)
{
}
MultiLayerBGS::~MultiLayerBGS()
{
  finish();
}

void MultiLayerBGS::setStatus(Status _status)
{
  status = _status;
}

void MultiLayerBGS::finish(void)
{
  if (bg_model_preload.empty())
  {
    bg_model_preload = "./models/MultiLayerBGSModel.yml";
  }

  if (status == MLBGS_LEARN && saveModel == true)
  {
    BGS->Save(bg_model_preload.c_str());
  }

  cvReleaseImage(&fg_img);
  cvReleaseImage(&bg_img);
  cvReleaseImage(&fg_prob_img);
  cvReleaseImage(&fg_mask_img);
  cvReleaseImage(&fg_prob_img3);
  cvReleaseImage(&merged_img);

  delete BGS;
}

void MultiLayerBGS::process(const cv::Mat &img_input, cv::Mat &img_output, cv::Mat &img_bgmodel)
{
  if (img_input.empty())
    return;


  CvSize img_size = cvSize(cvCeil((double)img_input.size().width), cvCeil((double)img_input.size().height));

  if (firstTime)
  {
    if (disableDetectMode)
      status = MLBGS_LEARN;

    if (status == MLBGS_LEARN)
      std::cout << "MultiLayerBGS in LEARN mode" << std::endl;

    if (status == MLBGS_DETECT)
      std::cout << "MultiLayerBGS in DETECT mode" << std::endl;

    org_img = new IplImage(img_input);

    fg_img = cvCreateImage(img_size, org_img->depth, org_img->nChannels);
    bg_img = cvCreateImage(img_size, org_img->depth, org_img->nChannels);
    fg_prob_img = cvCreateImage(img_size, org_img->depth, 1);
    fg_mask_img = cvCreateImage(img_size, org_img->depth, 1);
    fg_prob_img3 = cvCreateImage(img_size, org_img->depth, org_img->nChannels);
    merged_img = cvCreateImage(cvSize(img_size.width * 2, img_size.height * 2), org_img->depth, org_img->nChannels);

    BGS = new CMultiLayerBGS();
    BGS->Init(img_size.width, img_size.height);
    BGS->SetForegroundMaskImage(fg_mask_img);
    BGS->SetForegroundProbImage(fg_prob_img);

    if (bg_model_preload.empty() == false)
    {
      BGS->Load(bg_model_preload.c_str());
    }

    if (status == MLBGS_DETECT)
    {
      BGS->m_disableLearning = disableLearning;
    }

    if (loadDefaultParams)
    {

      max_mode_num = 5;
      weight_updating_constant = 5.0;
      texture_weight = 0.5;
      bg_mode_percent = 0.6f;
      pattern_neig_half_size = 4;
      pattern_neig_gaus_sigma = 3.0f;
      bg_prob_threshold = 0.2f;
      bg_prob_updating_threshold = 0.2f;
      robust_LBP_constant = 3;
      min_noised_angle = 10.0 / 180.0 * PI; //0,01768
      shadow_rate = 0.6f;
      highlight_rate = 1.2f;
      bilater_filter_sigma_s = 3.0f;
      bilater_filter_sigma_r = 0.1f;
    }

    BGS->m_nMaxLBPModeNum = max_mode_num;
    BGS->m_fWeightUpdatingConstant = weight_updating_constant;
    BGS->m_fTextureWeight = texture_weight;
    BGS->m_fBackgroundModelPercent = bg_mode_percent;
    BGS->m_nPatternDistSmoothNeigHalfSize = pattern_neig_half_size;
    BGS->m_fPatternDistConvGaussianSigma = pattern_neig_gaus_sigma;
    BGS->m_fPatternColorDistBgThreshold = bg_prob_threshold;
    BGS->m_fPatternColorDistBgUpdatedThreshold = bg_prob_updating_threshold;
    BGS->m_fRobustColorOffset = robust_LBP_constant;
    BGS->m_fMinNoisedAngle = min_noised_angle;
    BGS->m_fRobustShadowRate = shadow_rate;
    BGS->m_fRobustHighlightRate = highlight_rate;
    BGS->m_fSigmaS = bilater_filter_sigma_s;
    BGS->m_fSigmaR = bilater_filter_sigma_r;

    if (loadDefaultParams)
    {
      //frame_duration = 1.0 / 30.0;
      //frame_duration = 1.0 / 25.0;
      frame_duration = 1.0f / 10.0f;
    }

    BGS->SetFrameRate(frame_duration);

    if (status == MLBGS_LEARN)
    {
      if (loadDefaultParams)
      {
        mode_learn_rate_per_second = 0.5;
        weight_learn_rate_per_second = 0.5;
        init_mode_weight = 0.05f;
      }
      else
      {
        mode_learn_rate_per_second = learn_mode_learn_rate_per_second;
        weight_learn_rate_per_second = learn_weight_learn_rate_per_second;
        init_mode_weight = learn_init_mode_weight;
      }
    }

    if (status == MLBGS_DETECT)
    {
      if (loadDefaultParams)
      {
        mode_learn_rate_per_second = 0.01f;
        weight_learn_rate_per_second = 0.01f;
        init_mode_weight = 0.001f;
      }
      else
      {
        mode_learn_rate_per_second = detect_mode_learn_rate_per_second;
        weight_learn_rate_per_second = detect_weight_learn_rate_per_second;
        init_mode_weight = detect_init_mode_weight;
      }
    }

    BGS->SetParameters(max_mode_num, mode_learn_rate_per_second, weight_learn_rate_per_second, init_mode_weight);


    delete org_img;
  }

  //IplImage* inputImage = new IplImage(img_input);
  //IplImage* img = cvCreateImage(img_size, IPL_DEPTH_8U, 3);
  //cvCopy(inputImage, img);
  //delete inputImage;

  if (detectAfter > 0 && detectAfter == frameNumber)
  {
    status = MLBGS_DETECT;

    mode_learn_rate_per_second = 0.01f;
    weight_learn_rate_per_second = 0.01f;
    init_mode_weight = 0.001f;

    BGS->SetParameters(max_mode_num, mode_learn_rate_per_second, weight_learn_rate_per_second, init_mode_weight);

    BGS->m_disableLearning = disableLearning;
  }

  IplImage* img = new IplImage(img_input);

  BGS->SetRGBInputImage(img);
  BGS->Process();

  BGS->GetBackgroundImage(bg_img);
  BGS->GetForegroundImage(fg_img);
  BGS->GetForegroundProbabilityImage(fg_prob_img3);
  BGS->GetForegroundMaskImage(fg_mask_img);
  BGS->MergeImages(4, img, bg_img, fg_prob_img3, fg_img, merged_img);

  img_merged = cv::Mat(merged_img);
  img_foreground = cv::Mat(fg_mask_img);
  img_background = cv::Mat(bg_img);

  img_foreground.copyTo(img_output);
  img_background.copyTo(img_bgmodel);

  delete img;
  //cvReleaseImage(&img);

  firstTime = false;
  frameNumber++;
}

// Save parameters
void MultiLayerBGS::SaveParameter(std::string folderName) {
	std::ofstream myfile;
	myfile.open(folderName + "/parameter.txt", std::ios::app);
	myfile << "\n\n----METHOD THRESHOLD----" << std::endl;
	myfile << "Preload model: " << bg_model_preload.c_str() << std::endl;
	myfile << "Save model: " << saveModel << std::endl;
	myfile << "Detect after: " << detectAfter << std::endl;
	myfile << "Disable detect mode: " << disableDetectMode << std::endl;
	myfile << "Disable learning: " << disableLearning << std::endl;
	myfile << "Load default params: " << loadDefaultParams << std::endl;

	myfile << "Max mode number: " << max_mode_num << std::endl;
	myfile << "Weight updating constant: " << weight_updating_constant << std::endl;
	myfile << "Texture weight: " << texture_weight << std::endl;
	myfile << "Bg mode percent: " << bg_mode_percent << std::endl;
	myfile << "Pattern neighbour half size: " << pattern_neig_half_size << std::endl;
	myfile << "Pattern neighbour gaussian sigma: " << pattern_neig_gaus_sigma << std::endl;
	myfile << "BG probability threshold: " << bg_prob_threshold << std::endl;
	myfile << "BG probability updating threshold: " << bg_prob_updating_threshold << std::endl;
	myfile << "Robust LBP constant: " << robust_LBP_constant << std::endl;
	myfile << "Min noised angle: " << min_noised_angle << std::endl;
	myfile << "Shadow rate: " << shadow_rate << std::endl;
	myfile << "highlight rate: " << highlight_rate << std::endl;
	myfile << "Bilater filter sigma s: " << bilater_filter_sigma_s << std::endl;
	myfile << "Bilater filter sigma r: " << bilater_filter_sigma_r << std::endl;

	myfile << "Frame duration: " << frame_duration << std::endl;
	myfile << "Learn mode learn rate per second: " << learn_mode_learn_rate_per_second << std::endl;
	myfile << "Learn weight learn rate per second: " << learn_weight_learn_rate_per_second << std::endl;
	myfile << "Learn init mode weight: " << learn_init_mode_weight << std::endl;

	myfile << "Detect mode learn rate per second: " << detect_mode_learn_rate_per_second << std::endl;
	myfile << "Detect weight learn rate per second: " << detect_weight_learn_rate_per_second << std::endl;
	myfile << "Detect init mode weight: " << detect_init_mode_weight << std::endl;
	myfile.close();
}
