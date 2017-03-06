#pragma once

#include <opencv2/opencv.hpp>

#include "../IBGS.h"

class BackgroundSubtractorSuBSENSE;

class SuBSENSEBGS: public IBGS {
private:
	BackgroundSubtractorSuBSENSE* pSubsense;
	bool firstTime;

	float fRelLBSPThreshold;
	size_t nDescDistThresholdOffset;
	size_t nMinColorDistThreshold;
	size_t nBGSamples;
	size_t nRequiredBGSamples;
	size_t nSamplesForMovingAvgs;

public:
	SuBSENSEBGS();
	SuBSENSEBGS(float fRelLBSPThreshold, size_t nDescDistThresholdOffset, size_t nMinColorDistThreshold
		, size_t nBGSamples, size_t nRequiredBGSamples, size_t nSamplesForMovingAvgs);
	~SuBSENSEBGS();

	void process(const cv::Mat &img_input, cv::Mat &img_output,
			cv::Mat &img_bgmodel);
	void SaveParameter(std::string folderName);
};
