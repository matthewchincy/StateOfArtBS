#include "LOBSTER.h"
#include "BackgroundSubtractorLOBSTER.h"
#include <iostream>
#include <fstream>

LOBSTERBGS::LOBSTERBGS() :
	pLOBSTER(0), firstTime(true),
	fRelLBSPThreshold(BGSLOBSTER_DEFAULT_LBSP_REL_SIMILARITY_THRESHOLD),
	nLBSPThresholdOffset(BGSLOBSTER_DEFAULT_LBSP_OFFSET_SIMILARITY_THRESHOLD),
	nDescDistThreshold(BGSLOBSTER_DEFAULT_DESC_DIST_THRESHOLD),
	nColorDistThreshold(BGSLOBSTER_DEFAULT_COLOR_DIST_THRESHOLD),
	nBGSamples(BGSLOBSTER_DEFAULT_NB_BG_SAMPLES),
	nRequiredBGSamples(BGSLOBSTER_DEFAULT_REQUIRED_NB_BG_SAMPLES)
{
}
LOBSTERBGS::LOBSTERBGS(float fRelLBSPThreshold, size_t nLBSPThresholdOffset, size_t nDescDistThreshold
	, size_t nColorDistThreshold, size_t nBGSamples, size_t nRequiredBGSamples) :
	pLOBSTER(0), firstTime(true),
	fRelLBSPThreshold(fRelLBSPThreshold),
	nLBSPThresholdOffset(nLBSPThresholdOffset),
	nDescDistThreshold(nDescDistThreshold),
	nColorDistThreshold(nColorDistThreshold),
	nBGSamples(nBGSamples),
	nRequiredBGSamples(nRequiredBGSamples)
{
}
LOBSTERBGS::~LOBSTERBGS() {
	if (pLOBSTER)
		delete pLOBSTER;
}

void LOBSTERBGS::process(const cv::Mat &img_input, cv::Mat &img_output, cv::Mat &img_bgmodel)
{
	if (img_input.empty())
		return;


	if (firstTime) {
		pLOBSTER = new BackgroundSubtractorLOBSTER(
			fRelLBSPThreshold, nLBSPThresholdOffset, nDescDistThreshold,
			nColorDistThreshold, nBGSamples, nRequiredBGSamples);

		pLOBSTER->initialize(img_input, cv::Mat(img_input.size(), CV_8UC1, cv::Scalar_<uchar>(255)));
		firstTime = false;
	}

	(*pLOBSTER)(img_input, img_output);
	pLOBSTER->getBackgroundImage(img_bgmodel);

}

// Save parameters
void LOBSTERBGS::SaveParameter(std::string folderName) {
	std::ofstream myfile;
	myfile.open(folderName + "/parameter.txt", std::ios::app);
	myfile << "\n\n----METHOD THRESHOLD----" << std::endl;
	myfile << "LBSP threshold: " << fRelLBSPThreshold << std::endl;
	myfile << "LBSP threshold offset: " << nLBSPThresholdOffset << std::endl;
	myfile << "Absolute descriptor distance threshold: " << nDescDistThreshold << std::endl;
	myfile << "Absolute color distance threshold: " << nColorDistThreshold << std::endl;
	myfile << "Number of different samples per pixel: " << nBGSamples << std::endl;
	myfile << "Number of similar samples needed to consider the current pixel/block as 'background': " << nRequiredBGSamples << std::endl;
	myfile.close();
}
