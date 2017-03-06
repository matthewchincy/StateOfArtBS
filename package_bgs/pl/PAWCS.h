#pragma once

#include <opencv2/opencv.hpp>

#include "../IBGS.h"

class BackgroundSubtractorPAWCS;

class PAWCSBGS : public IBGS {
private:
	BackgroundSubtractorPAWCS* pPAWCS;
	bool firstTime;

	float fRelLBSPThreshold;
	size_t nDescDistThresholdOffset;
	size_t nMinColorDistThreshold;
	size_t nMaxNbWords;
	size_t nSamplesForMovingAvgs;

public:
	PAWCSBGS();
	PAWCSBGS(float fRelLBSPThreshold, size_t nDescDistThresholdOffset, size_t nMinColorDistThreshold
		, size_t nMaxNbWords, size_t nSamplesForMovingAvgs);
	~PAWCSBGS();

	void process(const cv::Mat &img_input, cv::Mat &img_output,
		cv::Mat &img_bgmodel);
	void SaveParameter(std::string folderName);
};
