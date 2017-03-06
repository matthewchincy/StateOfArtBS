#include "SigmaDeltaBGS.h"
#include <iostream>
#include <fstream>

SigmaDeltaBGS::SigmaDeltaBGS() :
	firstTime(true),
	ampFactor(1),
	minVar(15),
	maxVar(255),
	algorithm(sdLaMa091New()) {

	applyParams();
}
SigmaDeltaBGS::SigmaDeltaBGS(unsigned int ampFactor, unsigned int minVar, unsigned int maxVar) :
	firstTime(true),
	ampFactor(ampFactor),
	minVar(minVar),
	maxVar(maxVar),
	algorithm(sdLaMa091New()) {

	applyParams();
}
SigmaDeltaBGS::~SigmaDeltaBGS() {
	sdLaMa091Free(algorithm);
}

void SigmaDeltaBGS::process(
	const cv::Mat &img_input,
	cv::Mat &img_output,
	cv::Mat &img_bgmodel
) {
	if (img_input.empty())
		return;
		firstTime = false;

	img_output = cv::Mat(img_input.rows, img_input.cols, CV_8UC1);
	cv::Mat img_output_tmp(img_input.rows, img_input.cols, CV_8UC3);

	sdLaMa091Update_8u_C3R(algorithm, img_input.data, img_output_tmp.data);

	unsigned char* tmpBuffer = (unsigned char*)img_output_tmp.data;
	unsigned char* outBuffer = (unsigned char*)img_output.data;

	for (size_t i = 0; i < img_output.total(); ++i) {
		*outBuffer = *tmpBuffer;

		++outBuffer;
		tmpBuffer += img_output_tmp.channels();
	}

}


// Save parameters
void SigmaDeltaBGS::SaveParameter(std::string folderName) {
	std::ofstream myfile;
	myfile.open(folderName + "/parameter.txt", std::ios::app);
	myfile << "\n\n----METHOD THRESHOLD----" << std::endl;
	myfile << "Amp factor: " << ampFactor << std::endl;
	myfile << "Min var: " << minVar << std::endl;
	myfile << "Max var: " << maxVar << std::endl;
	myfile.close();
}
void SigmaDeltaBGS::applyParams() {
	sdLaMa091SetAmplificationFactor(algorithm, ampFactor);
	sdLaMa091SetMinimalVariance(algorithm, minVar);
	sdLaMa091SetMaximalVariance(algorithm, maxVar);
}
