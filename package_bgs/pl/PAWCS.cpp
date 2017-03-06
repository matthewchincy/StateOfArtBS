#include "PAWCS.h"
#include "BackgroundSubtractorPAWCS.h"
#include <iostream>
#include <fstream>

PAWCSBGS::PAWCSBGS() :
	pPAWCS(0), firstTime(true),
	fRelLBSPThreshold(BGSPAWCS_DEFAULT_LBSP_REL_SIMILARITY_THRESHOLD),
	nDescDistThresholdOffset(BGSPAWCS_DEFAULT_DESC_DIST_THRESHOLD_OFFSET),
	nMinColorDistThreshold(BGSPAWCS_DEFAULT_MIN_COLOR_DIST_THRESHOLD),
	nMaxNbWords(BGSPAWCS_DEFAULT_MAX_NB_WORDS),
	nSamplesForMovingAvgs(BGSPAWCS_DEFAULT_N_SAMPLES_FOR_MV_AVGS)
{
}
PAWCSBGS::PAWCSBGS(float fRelLBSPThreshold, size_t nDescDistThresholdOffset, size_t nMinColorDistThreshold
	, size_t nMaxNbWords, size_t nSamplesForMovingAvgs) :
	pPAWCS(0), firstTime(true), fRelLBSPThreshold(fRelLBSPThreshold),
	nDescDistThresholdOffset(nDescDistThresholdOffset), nMinColorDistThreshold(nMinColorDistThreshold),
	nMaxNbWords(nMaxNbWords), nSamplesForMovingAvgs(nSamplesForMovingAvgs)
{
}
PAWCSBGS::~PAWCSBGS() {
	if (pPAWCS)
		delete pPAWCS;
}

void PAWCSBGS::process(const cv::Mat &img_input, cv::Mat &img_output, cv::Mat &img_bgmodel)
{
	if (img_input.empty())
		return;
	
	if (firstTime) {
		pPAWCS = new BackgroundSubtractorPAWCS(
			fRelLBSPThreshold, nDescDistThresholdOffset, nMinColorDistThreshold,
			nMaxNbWords, nSamplesForMovingAvgs);

		pPAWCS->initialize(img_input, cv::Mat(img_input.size(), CV_8UC1, cv::Scalar_<uchar>(255)));
		firstTime = false;
	}

	(*pPAWCS)(img_input, img_output);
	pPAWCS->getBackgroundImage(img_bgmodel);
}

// Save parameters
void PAWCSBGS::SaveParameter(std::string folderName) {
	std::ofstream myfile;
	myfile.open(folderName + "/parameter.txt", std::ios::app);
	myfile << "\n\n----METHOD THRESHOLD----" << std::endl;
	myfile << "LBSP threshold: " << fRelLBSPThreshold << std::endl;
	myfile << "Absolute descriptor distance threshold offset: " << nDescDistThresholdOffset << std::endl;
	myfile << "Absolute minimal color distance threshold ('R'): " << nMinColorDistThreshold << std::endl;
	myfile << "Max number of local words used to build background submodels: " << nMaxNbWords << std::endl;
	myfile << "Number of samples to use to compute the learning rate of moving averages: " << nSamplesForMovingAvgs << std::endl;
	myfile.close();
}
