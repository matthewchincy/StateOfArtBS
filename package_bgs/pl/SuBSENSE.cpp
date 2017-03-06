#include "SuBSENSE.h"
#include "BackgroundSubtractorSuBSENSE.h"
#include <iostream>
#include <fstream>

SuBSENSEBGS::SuBSENSEBGS() :
	pSubsense(0), firstTime(true),
	fRelLBSPThreshold(BGSSUBSENSE_DEFAULT_LBSP_REL_SIMILARITY_THRESHOLD),
	nDescDistThresholdOffset(BGSSUBSENSE_DEFAULT_DESC_DIST_THRESHOLD_OFFSET),
	nMinColorDistThreshold(BGSSUBSENSE_DEFAULT_MIN_COLOR_DIST_THRESHOLD),
	nBGSamples(BGSSUBSENSE_DEFAULT_NB_BG_SAMPLES),
	nRequiredBGSamples(BGSSUBSENSE_DEFAULT_REQUIRED_NB_BG_SAMPLES),
	nSamplesForMovingAvgs(BGSSUBSENSE_DEFAULT_N_SAMPLES_FOR_MV_AVGS)
{
}
SuBSENSEBGS::SuBSENSEBGS(float fRelLBSPThreshold, size_t nDescDistThresholdOffset, size_t nMinColorDistThreshold
	, size_t nBGSamples, size_t nRequiredBGSamples, size_t nSamplesForMovingAvgs) :
	pSubsense(0), firstTime(true),
	fRelLBSPThreshold(fRelLBSPThreshold),
	nDescDistThresholdOffset(nDescDistThresholdOffset),
	nMinColorDistThreshold(nMinColorDistThreshold),
	nBGSamples(nBGSamples),
	nRequiredBGSamples(nRequiredBGSamples),
	nSamplesForMovingAvgs(nSamplesForMovingAvgs)
{
}
SuBSENSEBGS::~SuBSENSEBGS() {
	if (pSubsense)
		delete pSubsense;
}

void SuBSENSEBGS::process(const cv::Mat &img_input, cv::Mat &img_output, cv::Mat &img_bgmodel)
{
	if (img_input.empty())
		return;

	if (firstTime) {
		pSubsense = new BackgroundSubtractorSuBSENSE(
			fRelLBSPThreshold, nDescDistThresholdOffset, nMinColorDistThreshold,
			nBGSamples, nRequiredBGSamples, nSamplesForMovingAvgs);

		pSubsense->initialize(img_input, cv::Mat(img_input.size(), CV_8UC1, cv::Scalar_<uchar>(255)));
		firstTime = false;
	}

	(*pSubsense)(img_input, img_output);
	pSubsense->getBackgroundImage(img_bgmodel);

}

// Save parameters
void SuBSENSEBGS::SaveParameter(std::string folderName) {
	std::ofstream myfile;
	myfile.open(folderName + "/parameter.txt", std::ios::app);
	myfile << "\n\n----METHOD THRESHOLD----" << std::endl;
	myfile << "LBSP threshold: " << fRelLBSPThreshold << std::endl;
	myfile << "Absolute descriptor distance threshold offset: " << nDescDistThresholdOffset << std::endl;
	myfile << "Absolute minimal color distance threshold ('R'): " << nMinColorDistThreshold << std::endl;
	myfile << "Number of different samples per pixel: " << nBGSamples << std::endl;
	myfile << "Number of similar samples needed to consider the current pixel/block as 'background': " << nRequiredBGSamples << std::endl;
	myfile << "Number of samples to use to compute the learning rate of moving averages: " << nSamplesForMovingAvgs << std::endl;
	myfile.close();
}
