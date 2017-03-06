#include "IndependentMultimodalBGS.h"
#include <iostream>
#include <fstream>

IndependentMultimodalBGS::IndependentMultimodalBGS() : fps(10), firstTime(true){
  pIMBS = new BackgroundSubtractorIMBS(fps);
}
IndependentMultimodalBGS::IndependentMultimodalBGS(int fps) 
	: fps(fps), firstTime(true) {
	pIMBS = new BackgroundSubtractorIMBS(fps);
}
IndependentMultimodalBGS::~IndependentMultimodalBGS(){
  delete pIMBS;
}

void IndependentMultimodalBGS::process(const cv::Mat &img_input, cv::Mat &img_output, cv::Mat &img_bgmodel)
{
  if(img_input.empty())
    return;
  
  //get the fgmask and update the background model
  pIMBS->apply(img_input, img_foreground);
  
  //get background image
  pIMBS->getBackgroundImage(img_background);

  img_foreground.copyTo(img_output);
  img_background.copyTo(img_bgmodel);
  
  firstTime = false;
}

// Save parameters
void IndependentMultimodalBGS::SaveParameter(std::string folderName) {
	std::ofstream myfile;
	myfile.open(folderName + "/parameter.txt", std::ios::app);
	myfile << "\n\n----METHOD THRESHOLD----" << std::endl;
	myfile << "FPS: " << fps << std::endl;
	myfile.close();
}
