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
#include <iostream>

#include "../package_bgs/FrameDifferenceBGS.h"
#include "../package_bgs/StaticFrameDifferenceBGS.h"
#include "../package_bgs/WeightedMovingMeanBGS.h"
#include "../package_bgs/WeightedMovingVarianceBGS.h"
#include "../package_bgs/MixtureOfGaussianV1BGS.h"
#include "../package_bgs/MixtureOfGaussianV2BGS.h"
#include "../package_bgs/AdaptiveBackgroundLearning.h"
#include "../package_bgs/AdaptiveSelectiveBackgroundLearning.h"

#if CV_MAJOR_VERSION >= 2 && CV_MINOR_VERSION >= 4 && CV_SUBMINOR_VERSION >= 3
#include "../package_bgs/GMG.h"
#endif

#include "../package_bgs/dp/DPAdaptiveMedianBGS.h"
#include "../package_bgs/dp/DPGrimsonGMMBGS.h"
#include "../package_bgs/dp/DPZivkovicAGMMBGS.h"
#include "../package_bgs/dp/DPMeanBGS.h"
#include "../package_bgs/dp/DPWrenGABGS.h"
#include "../package_bgs/dp/DPPratiMediodBGS.h"
#include "../package_bgs/dp/DPEigenbackgroundBGS.h"
#include "../package_bgs/dp/DPTextureBGS.h"

#include "../package_bgs/tb/T2FGMM_UM.h"
#include "../package_bgs/tb/T2FGMM_UV.h"
#include "../package_bgs/tb/T2FMRF_UM.h"
#include "../package_bgs/tb/T2FMRF_UV.h"
#include "../package_bgs/tb/FuzzySugenoIntegral.h"
#include "../package_bgs/tb/FuzzyChoquetIntegral.h"

#include "../package_bgs/lb/LBSimpleGaussian.h"
#include "../package_bgs/lb/LBFuzzyGaussian.h"
#include "../package_bgs/lb/LBMixtureOfGaussians.h"
#include "../package_bgs/lb/LBAdaptiveSOM.h"
#include "../package_bgs/lb/LBFuzzyAdaptiveSOM.h"

#include "../package_bgs/ck/LbpMrf.h"
#include "../package_bgs/jmo/MultiLayerBGS.h"
#include "../package_bgs/pt/PixelBasedAdaptiveSegmenter.h"
#include "../package_bgs/av/VuMeter.h"
#include "../package_bgs/ae/KDE.h"
#include "../package_bgs/db/IndependentMultimodalBGS.h"
#include "../package_bgs/sjn/SJN_MultiCueBGS.h"
#include "../package_bgs/bl/SigmaDeltaBGS.h"

#include "../package_bgs/pl/SuBSENSE.h"
#include "../package_bgs/pl/LOBSTER.h"
#include "../package_bgs/pl/PAWCS.h"

#include <direct.h>
#include <opencv2/opencv.hpp>

#include <time.h>
#include <fstream>
#include <iomanip>
#include <conio.h>
#include <windows.h>

/****Read input methods****/
// Read integer value input
int readIntInput(std::string question);
// Read double value input
double readDoubleInput(std::string question);
// Read boolean value input
bool readBoolInput(std::string question);
// Read video input
cv::VideoCapture readVideoInput(std::string question, std::string *filename,
	double *FPS, double *FRAME_COUNT, cv::Size *FRAME_SIZE);
// Get current date/time, format is DDMMYYHHmmss
const std::string currentDateTimeStamp(time_t * now);
// Get current date/time, format is DD-MM-YY HH:mm:ss
const std::string currentDateTime(time_t * now);
// Save current's process parameter
void SaveParameter(std::string folderName);
// Evaluate results
void EvaluateResult(std::string filename, std::string folderName, std::string currFolderName);
// Calculate processing time
void GenerateProcessTime(double FRAME_COUNT, std::string currFolderName);

// Read method information
IBGS* readMethodInfo();

/****Global variable declaration****/
// Program version
const std::string programVersion = "E1-0";
// Method name
std::string methodName;
// Method type
std::string methodType;
// Method author
std::string methodAuthor;
// Show input frame switch
bool showInputSwitch;
// Show output frame switch
bool showOutputSwitch;
// Save result switch
bool saveResultSwitch;
// Evaluate result switch
bool evaluateResultSwitch;
// Method ID
int method;
// Frames per second (FPS) of the input video
double FPS;
// Total number of frame of the input video
double FRAME_COUNT;
// Frame size of the input video
cv::Size FRAME_SIZE;
//// Debug switch
//bool debugSwitch;
// Program start time
time_t tempStartTime;
// Program finish time
time_t tempFinishTime;

using namespace std;

int main()
{
	// Video file name
	std::string filename;
	// Read video input from user
	cv::VideoCapture videoCapture = readVideoInput("Video folder", &filename, &FPS, &FRAME_COUNT, &FRAME_SIZE);
	// Show input frame switch
	showInputSwitch = readBoolInput("Show input frame(1/0)");
	// Show output frame switch
	showOutputSwitch = readBoolInput("Show output frame(1/0)");
	// Save result switch
	saveResultSwitch = readBoolInput("Save result(1/0)");
	// Evaluate result switch
	evaluateResultSwitch = readBoolInput("Evaluate result(1/0)");

	// Input frame
	cv::Mat inputFrame;
	// Foreground Mask
	cv::Mat fgMask;
	// Background image
	cv::Mat bgImage;
	// Region of interest frame
	cv::Mat ROI_FRAME;
	ROI_FRAME.create(FRAME_SIZE, CV_8UC1);
	ROI_FRAME = cv::Scalar_<uchar>(255);

	char *p;
	/* Background Subtraction Methods */
	IBGS *bgs = nullptr;
	bgs = readMethodInfo();

	// Read first frame from video
	videoCapture.set(CV_CAP_PROP_POS_FRAMES, 0);
	videoCapture >> inputFrame;
	// Current date/time based on current system
	tempStartTime = time(0);
	// Program start time
	std::string startTime = currentDateTimeStamp(&tempStartTime);
	// Current process result folder name
	std::string folderName = filename + "/" + methodName + "-" + programVersion + "-" + startTime;
	const std::string currFolderName = folderName;

	if (showInputSwitch) {
		// Display input video windows
		cv::namedWindow("Input Video");
	}
	if (showOutputSwitch) {
		// Display results windows
		cv::namedWindow("Results");
	}

	const char *s1;
	if (saveResultSwitch) {
		s1 = folderName.c_str();
		_mkdir(s1);
		SaveParameter(folderName);
		bgs->SaveParameter(folderName);
		folderName += "/results";
		s1 = folderName.c_str();
		_mkdir(s1);
	}

	char s[25];
	for (int currFrameIndex = 1; currFrameIndex <= FRAME_COUNT; currFrameIndex++) {


		bgs->process(inputFrame, fgMask, bgImage); // by default, it shows automatically the foreground mask image

		if (showInputSwitch) {
			cv::imshow("Input Video", inputFrame);
		}
		if (showOutputSwitch) {
			cv::imshow("Results", fgMask);
		}
		if (saveResultSwitch) {
			std::string saveFolder;
			sprintf(s, "/bin%06d.png", (currFrameIndex));
			saveFolder = folderName + s;
			cv::imwrite(saveFolder, fgMask);
		}
		// If 'esc' key is pressed, break loop
		if (cv::waitKey(1) == 27)
		{
			std::cout << "Program ended by users." << std::endl;
			break;
		}
		bool inputCheck = videoCapture.read(inputFrame);
		if (!inputCheck && (currFrameIndex < FRAME_COUNT)) {
			std::cout << "Video having problem. Cannot read the frame from video file." << std::endl;
			return -1;
		}
	}

	delete bgs;
	videoCapture.release();
	tempFinishTime = time(0);
	GenerateProcessTime(FRAME_COUNT, currFolderName);

	std::cout << "Background subtraction completed" << std::endl;

	if (evaluateResultSwitch) {
		if (saveResultSwitch) {

			std::cout << "Now starting evaluate the processed result..." << std::endl;
			EvaluateResult(filename, folderName, currFolderName);
		}
		else {
			std::cout << "No saved results for evaluation" << std::endl;
		}
	}

	std::cout << "Program Completed!" << std::endl;
	Beep(1568, 200);
	Beep(1568, 200);
	Beep(1568, 200);
	Beep(1245, 1000);
	Beep(1397, 200);
	Beep(1397, 200);
	Beep(1397, 200);
	Beep(1175, 1000);
	system("pause");
	return 0;
}
// Get current date/time, format is DDMMYYHHmmss
const std::string currentDateTimeStamp(time_t * now) {
	//time_t     now = time(0);
	struct tm  tstruct;
	char       buf[80];
	tstruct = *localtime(now);
	strftime(buf, sizeof(buf), "%d%m%y-%H%M%S", &tstruct);

	return buf;
}
// Get current date/time, format is DD-MM-YY HH:mm:ss
const std::string currentDateTime(time_t * now) {
	struct tm  tstruct;
	char       buf[80];
	tstruct = *localtime(now);
	strftime(buf, sizeof(buf), "%d-%b-%G %X", &tstruct);

	return buf;
}
// Save current process's parameter
void SaveParameter(std::string folderName) {
	std::ofstream myfile;
	myfile.open(folderName + "/parameter.txt", std::ios::app);
	myfile << "----VERSION----\n";
	myfile << programVersion << std::endl;

	myfile << "----METHOD----\n";
	myfile << "Method name: " << methodName << std::endl;
	myfile << "Method type: " << methodType << std::endl;
	myfile << "Method author: " << methodAuthor << std::endl;
	myfile << "----MAIN PROCESS PARAMETER----\n";
	myfile << "RESULT FOLDER: " << folderName << std::endl;
	myfile << "\n----VIDEO PARAMETER----\n";
	myfile << "VIDEO WIDTH:" << FRAME_SIZE.width << std::endl;
	myfile << "VIDEO HEIGHT:" << FRAME_SIZE.height << std::endl;
	myfile << "VIDEO LENGTH:" << FRAME_COUNT << std::endl;
	myfile << "VIDEO FPS:" << FPS << std::endl;
	myfile.close();
}
// Evaluate results
void EvaluateResult(std::string filename, std::string folderName, std::string currFolderName) {
	// A video folder should contain 2 folders['input', 'groundtruth']
	//and the "temporalROI.txt" file to be valid.The choosen method will be 
	// applied to all the frames specified in \temporalROI.txt

	// Read index from temporalROI.txt
	std::ifstream infile(filename + "/temporalROI.txt");
	int idxFrom, idxTo;
	double TotalShadow = 0, TP = 0, TN = 0, FP = 0, FN = 0, SE = 0;
	infile >> idxFrom >> idxTo;
	infile.close();
	std::string groundtruthFolder = filename + "/groundtruth";
	char s[25];
	for (size_t startIndex = idxFrom; startIndex <= idxTo; startIndex++) {
		sprintf(s, "%06d.png", (startIndex));
		//cv::Mat gtImg = cv::imread( "bungalows/groundtruth/gt000001.png");
		cv::Mat gtImg = cv::imread(groundtruthFolder + "/gt" + s, CV_LOAD_IMAGE_GRAYSCALE);
		cv::Mat resultImg = cv::imread(folderName + "/bin" + s, CV_LOAD_IMAGE_GRAYSCALE);
		for (size_t pxPointer = 0; pxPointer < (resultImg.rows*resultImg.cols); pxPointer++) {

			double gtValue = gtImg.data[pxPointer];
			double resValue = resultImg.data[pxPointer];
			if (gtValue == 255) {
				// TP
				if (resValue == 255) {
					TP++;
				}
				// FN
				else {
					FN++;
				}
			}
			else if (gtValue == 50) {
				TotalShadow++;
				// TN
				if (resValue == 0) {
					TN++;
				}
				// SE FP
				else {
					SE++;
					FP++;
				}
			}
			else if (gtValue < 50) {
				// TN
				if (resValue == 0) {
					TN++;
				}
				// SE FP
				else {
					FP++;
				}
			}
		}
	}
	const double recall = TP / (TP + FN);
	const double precision = TP / (TP + FP);
	const double FMeasure = (2.0*recall*precision) / (recall + precision);

	const double specficity = TN / (TN + FP);
	const double FPR = FP / (FP + TN);
	const double FNR = FN / (TP + FN);
	const double PBC = 100.0 * (FN + FP) / (TP + FP + FN + TN);

	std::ofstream myfile;
	myfile.open(currFolderName + "/parameter.txt", std::ios::app);
	myfile << "\n<<<<<-STATISTICS  RESULTS->>>>>\n";
	myfile << "TRUE POSITIVE(TP): " << std::setprecision(0) << std::fixed << TP << std::endl;
	myfile << "FALSE POSITIVE(FP): " << std::setprecision(0) << std::fixed << FP << std::endl;
	myfile << "TRUE NEGATIVE(TN): " << std::setprecision(0) << std::fixed << TN << std::endl;
	myfile << "FALSE NEGATIVE(FN): " << std::setprecision(0) << std::fixed << FN << std::endl;
	myfile << "SHADOW ERROR(SE): " << std::setprecision(0) << std::fixed << SE << std::endl;
	myfile << "RECALL: " << std::setprecision(3) << std::fixed << recall << std::endl;
	myfile << "PRECISION: " << std::setprecision(3) << std::fixed << precision << std::endl;
	myfile << "F-MEASURE: " << std::setprecision(3) << std::fixed << FMeasure << std::endl;
	myfile << "SPECIFICITY: " << std::setprecision(3) << std::fixed << specficity << std::endl;
	myfile << "FPR: " << std::setprecision(3) << std::fixed << FPR << std::endl;
	myfile << "FNR: " << std::setprecision(3) << std::fixed << FNR << std::endl;
	myfile << "PBC: " << std::setprecision(3) << std::fixed << PBC << std::endl;
	myfile << "TOTAL SHADOW: " << std::setprecision(0) << std::fixed << TotalShadow << std::endl;
	myfile.close();
}
// Calculate processing time
void GenerateProcessTime(double FRAME_COUNT, std::string currFolderName) {

	double diffSeconds = difftime(tempFinishTime, tempStartTime);
	int seconds, hours, minutes;
	minutes = diffSeconds / 60;
	hours = minutes / 60;
	seconds = int(diffSeconds) % 60;
	double fpsProcess = FRAME_COUNT / diffSeconds;
	std::cout << "<<<<<-TOTAL PROGRAM TIME->>>>>\n" << "PROGRAM START TIME:" << currentDateTime(&tempStartTime) << std::endl;
	std::cout << "PROGRAM END  TIME:" << currentDateTime(&tempFinishTime) << std::endl;
	std::cout << "TOTAL SPEND TIME:" << hours << " H " << minutes << " M " << seconds << " S" << std::endl;
	std::cout << "AVERAGE FPS:" << fpsProcess << std::endl;
	if (saveResultSwitch) {
		std::ofstream myfile;
		myfile.open(currFolderName + "/parameter.txt", std::ios::app);
		myfile << "\n\n<<<<<-TOTAL PROGRAM TIME->>>>>\n";
		myfile << "PROGRAM START TIME:";
		myfile << currentDateTime(&tempStartTime);
		myfile << "\n";
		myfile << "PROGRAM END  TIME:";
		myfile << currentDateTime(&tempFinishTime);
		myfile << "\n";
		myfile << "TOTAL SPEND TIME:";
		myfile << hours << " H " << minutes << " M " << seconds << " S";
		myfile << "\n";
		myfile << "AVERAGE FPS:";
		myfile << fpsProcess;
		myfile << "\n";
		myfile.close();
	}
}

/****Read input methods****/
// Read integer value input
int readIntInput(std::string question)
{
	int input = -1;
	bool valid = false;
	do
	{
		std::cout << question << " :" << std::flush;
		std::cin >> input;
		if (std::cin.good())
		{
			valid = true;
		}
		else
		{
			std::cin.clear();
			std::cin.ignore();
			std::cout << "Invalid input; please re-enter double value only." << std::endl;
		}
	} while (!valid);

	return (input);
}
// Read double value input
double readDoubleInput(std::string question)
{
	double input = -1;
	bool valid = false;
	do
	{
		std::cout << question << " :" << std::flush;
		std::cin >> input;
		if (std::cin.good())
		{
			valid = true;
		}
		else
		{
			std::cin.clear();
			std::cin.ignore();
			std::cout << "Invalid input; please re-enter double value only." << std::endl;
		}
	} while (!valid);

	return (input);
}
// Read boolean value input
bool readBoolInput(std::string question)
{
	int input = 3;
	bool valid = false;
	bool result = false;
	do
	{
		std::cout << question << " :" << std::flush;
		std::cin >> input;
		if (std::cin.good())
		{
			switch (input) {
			case 0:	result = false;
				valid = true;
				break;
			case 1:	result = true;
				valid = true;
				break;
			default:
				std::cin.clear();
				std::cin.ignore(true, '\n');
				std::cout << "Invalid input; please re-enter boolean (0/1) only." << std::endl;
				break;
			}
		}
		else
		{
			std::cin.clear();
			std::cin.ignore(true, '\n');
			std::cout << "Invalid input; please re-enter boolean (0/1) only." << std::endl;
		}
	} while (!valid);

	return (result);
}
// Read video input
cv::VideoCapture readVideoInput(std::string question, std::string *filename, double *FPS,
	double *FRAME_COUNT, cv::Size *FRAME_SIZE)
{
	// Video capture variable
	cv::VideoCapture videoCapture;
	std::string videoName;

	int input = 3;
	bool valid = false;
	bool result = false;
	do
	{
		std::cout << question << " :" << std::flush;
		getline(std::cin, (*filename));
		bool check = false;
		for (size_t formatIndex = 0; formatIndex < 2; formatIndex++) {
			switch (formatIndex) {
			case 0:
				videoName = (*filename) + "/" + (*filename) + ".avi";
				break;
			case 1:
				videoName = (*filename) + "/" + (*filename) + ".mp4";
				break;
			}
			videoCapture.open(videoName);
			// Input frame
			cv::Mat inputFrames;
			videoCapture >> inputFrames;

			// Checking video whether successful be opened
			if (videoCapture.isOpened() && !inputFrames.empty()) {
				check = true;
				break;
			}
		}
		if (check) {
			std::cout << "Video successful loaded!" << std::endl;
			valid = true;
			// Getting frames per second (FPS) of the input video
			(*FPS) = videoCapture.get(CV_CAP_PROP_FPS);
			// Getting total number of frame of the input video
			(*FRAME_COUNT) = videoCapture.get(CV_CAP_PROP_FRAME_COUNT);
			// Getting size of the input video
			(*FRAME_SIZE) = cv::Size(videoCapture.get(CV_CAP_PROP_FRAME_WIDTH), videoCapture.get(CV_CAP_PROP_FRAME_HEIGHT));
			videoCapture.set(CV_CAP_PROP_POS_FRAMES, 0);
			break;
		}
		else {
			std::cout << "\nVideo having problem. Cannot open the video file. Please re-enter valid filename" << std::endl;
			std::cin.sync();
		}
	} while (!valid);

	return (videoCapture);
}

// Read method information
IBGS * readMethodInfo() {
	IBGS * bgs = nullptr;
	int threshold = 0;
	bool enableThreshold = true;
	bool success = false;
	while (!success) {
		std::cout << "1. Static Frame Difference \n2. Frame Difference \n3. Weighted Moving Mean\n";
		std::cout << "4. Weighted Moving Variance \n5. Adaptive Background Learning \n6. Adaptive-Selective Background Learning\n";
		std::cout << "7. Temporal Mean \n8. Adaptive Median \n9. Temporal Median\n";
		std::cout << "10. Sigma-Delta \n11. Fuzzy Sugeno Integral \n12. Fuzzy Choquet Integral\n";
		std::cout << "13. Fuzzy Gaussian \n14. Gaussian Average \n15. Simple Gaussian\n";
		std::cout << "16. Gaussian Mixture Model(Stauffer and Grimson) \n17. Gaussian Mixture Model(KadewTraKuPong and Bowden) \n18. Gaussian Mixture Model 1 (Zivkovic)\n";
		std::cout << "19. Gaussian Mixture Model 2 (Zivkovic) \n20. Gaussian Mixture Model(Laurence Bender) \n21. Type-2 Fuzzy GMM-UM\n";
		std::cout << "22. Type-2 Fuzzy GMM-UV \n23. Type-2 Fuzzy GMM-UM with MRF \n24. Type-2 Fuzzy GMM-UV with MRF\n";
		std::cout << "25. Texture BGS \n26. Texture-Based Foreground Detection with MRF \n27. Multi-Layer BGS\n";
		std::cout << "28. MultiCue BGS \n29. SuBSENSE \n30. LOBSTER\n";
		std::cout << "31. PBAS \n32. PAWCS \n33. GMG\n";
		std::cout << "34. VuMeter \n35. KDE \n36. IMBS\n";
		std::cout << "37. Eigenbackground / SL-PCA \n38. Adaptive SOM \n39. Fuzzy Adaptive SOM\n";

		success = true;
		// Input method id
		int method = readIntInput("Method ID");
		switch (method) {
			// Basic Type
		case 1:
			// bool enableThreshold, int threshold
			threshold = readIntInput("Threshold (Default:15))");
			bgs = new StaticFrameDifferenceBGS(enableThreshold, threshold);
			methodName = "Static Frame Difference";
			methodType = "Basic";
			methodAuthor = "OpenCV";
			break;
		case 2:
			// bool enableThreshold, int threshold
			threshold = readIntInput("Threshold (Default:15))");
			bgs = new FrameDifferenceBGS(enableThreshold, threshold);
			methodName = "Frame Difference";
			methodType = "Basic";
			methodAuthor = "OpenCV";
			break;
		case 3:
			// bool enableWeight, bool enableThreshold, int threshold
			bgs = new WeightedMovingMeanBGS(true, true, 15);
			methodName = "Weighted Moving Mean";
			methodType = "Basic";
			methodAuthor = "OpenCV";
			break;
		case 4:
			// bool enableWeight, bool enableThreshold, int threshold
			threshold = readIntInput("Threshold (Default:15))");
			bgs = new WeightedMovingVarianceBGS(true, enableThreshold, threshold);
			methodName = "Weighted Moving Variance";
			methodType = "Basic";
			methodAuthor = "OpenCV";
			break;
		case 5:
			// double alpha,long limit,long counter,double minVal,double maxVal,
			// bool enableThreshold, int threshold
			threshold = readIntInput("Threshold (Default:15))");
			bgs = new AdaptiveBackgroundLearning(0.05, -1, 0, 0.0, 1.0, enableThreshold, threshold);
			methodName = "Adaptive Background Learning";
			methodType = "Basic";
			methodAuthor = "OpenCV";
			break;
		case 6:
			// double alphaLearn, double alphaDetection, long learningFrames, long counter,
			// double minVal, double maxVal, int threshold
			threshold = readIntInput("Threshold (Default:15))");
			bgs = new AdaptiveSelectiveBackgroundLearning(0.05, 0.05, -1, 0, 0.0, 1.0, threshold);
			methodName = "Adaptive-Selective Background Learning";
			methodType = "Basic";
			methodAuthor = "OpenCV";
			break;
		case 7:
			// long framenumber, int threshold, double alpha, int learningframes
			bgs = new DPMeanBGS(0, 2700, 1e-6f, 30);
			methodName = "Temporal Mean";
			methodType = "Basic";
			methodAuthor = "OpenCV";
			break;
		case 8:
			// long frameNumber, int threshold, double samplingRate, int learningFrames
			bgs = new DPAdaptiveMedianBGS(0, 40, 7, 30);
			methodName = "Adaptive Median";
			methodType = "Basic";
			methodAuthor = "McFarlane and Schofield (1995)";
			break;
		case 9:
			// long frameNumber, int threshold, int samplingRate, int historySize, int weight
			bgs = new DPPratiMediodBGS(0, 30, 5, 16, 5);
			methodName = "Temporal Median";
			methodType = "Basic";
			methodAuthor = "Cucchiara et al. (2003) and Calderara et al. (2006)";
			break;
		case 10:
			// unsigned int ampFactor, unsigned int minVar, unsigned int maxVar
			bgs = new SigmaDeltaBGS(1, 15, 255);
			methodName = "Sigma-Delta";
			methodType = "Basic";
			methodAuthor = "Manzanera and Richefeu (2004))";
			break;
			// Fuzzy Type
		case 11:
			// long long frameNumber,int framesToLearn, double alphaLearn, double alphaUpdate,
			// int colorSpace, int option, bool smooth, double threshold
			bgs = new FuzzySugenoIntegral(0, 10, 0.1f, 0.01f, 1, 2, 1, 0.67f);
			methodName = "Fuzzy Sugeno Integral";
			methodType = "Fuzzy";
			methodAuthor = "Hongxun Zhang and De Xu (2006)";
			break;
		case 12:
			// long long frameNumber,int framesToLearn, double alphaLearn, double alphaUpdate,
			// int colorSpace, int option, bool smooth, double threshold
			bgs = new FuzzyChoquetIntegral(0, 10, 0.1f, 0.01f, 1, 2, 1, 0.67f);
			methodName = "Fuzzy Choquet Integral";
			methodType = "Fuzzy";
			methodAuthor = "Baf et al. (2008)";
			break;
		case 13:
			// int sensitivity, int bgThreshold, int learningRate,int noiseVariance
			bgs = new LBFuzzyGaussian(72, 162, 49, 195);
			methodName = "Fuzzy Gaussian";
			methodType = "Fuzzy";
			methodAuthor = "Wren (1997) with Sigari et al. (2008)";
			break;
			// Single Gaussian Type
		case 14:
			// long frameNumber, double threshold, double alpha, int learningFrames
			bgs = new DPWrenGABGS(0, 12.25f, 0.005f, 30);
			methodName = "Gaussian Average";
			methodType = "Single gaussian";
			methodAuthor = "Wren (1997)";
			break;
		case 15:
			// int sensitivity, int noiseVariance, int learningRate
			bgs = new LBSimpleGaussian(66, 162, 18);
			methodName = "Simple Gaussian";
			methodType = "Single gaussian";
			methodAuthor = "Benezeth et al. (2008)";
			break;
			// Multiple Gaussians Type
		case 16:
			// long frameNumber, double threshold, double alpha, int gaussians
			bgs = new DPGrimsonGMMBGS(0, 9.0f, 0.01f, 3);
			methodName = "Gaussian Mixture Model(Stauffer and Grimson)";
			methodType = "Multiple gaussian";
			methodAuthor = "Stauffer and Grimson (1999)";
			break;
		case 17:
			// double alpha, bool enableThreshold, int threshold
			bgs = new MixtureOfGaussianV1BGS(0.05f, 1, 15);
			methodName = "Gaussian Mixture Model(KadewTraKuPong and Bowden)";
			methodType = "Multiple gaussian";
			methodAuthor = "KadewTraKuPong and Bowden (2001)";
			break;
		case 18:
			// double alpha, bool enableThreshold, int threshold
			bgs = new MixtureOfGaussianV2BGS(0.05f, 1, 15);
			methodName = "Gaussian Mixture Model 1 (Zivkovic)";
			methodType = "Multiple gaussian";
			methodAuthor = "Zivkovic (2004)";
			break;
		case 19:
			// long frameNumber, double threshold, double alpha, int gaussians
			bgs = new DPZivkovicAGMMBGS(0, 25.0f, 0.001f, 3);
			methodName = "Gaussian Mixture Model 2 (Zivkovic)";
			methodType = "Multiple gaussian";
			methodAuthor = "Zivkovic (2004)";
			break;
		case 20:
			//int sensitivity, int bgThreshold, int learningRate, int noiseVariance
			bgs = new LBMixtureOfGaussians(81, 83, 59, 206);
			methodName = "Gaussian Mixture Model(Laurence Bender";
			methodType = "Multiple gaussian";
			methodAuthor = "Laurence Bender implementation (GMM with Mahalanobis distance)";
			break;
			// Type-2 Fuzzy Type
		case 21:
			// long frameNumber, double threshold, double alpha, float km, float kv, int gaussians
			bgs = new T2FGMM_UM(0, 9.0f, 0.01f, 1.5f, 0.6f, 3);
			methodName = "Type-2 Fuzzy GMM-UM";
			methodType = "Type-2 Fuzzy Type";
			methodAuthor = "Baf et al. (2008)";
			break;
		case 22:
			// long frameNumber, double threshold, double alpha, float km, float kv, int gaussians
			bgs = new T2FGMM_UV(0, 9.0f, 0.01f, 1.5f, 0.6f, 3);
			methodName = "Type-2 Fuzzy GMM-UV";
			methodType = "Type-2 Fuzzy Type";
			methodAuthor = "Baf et al. (2008)";
			break;
		case 23:
			// long frameNumber, double threshold, double alpha, float km, float kv, int gaussians
			bgs = new T2FMRF_UM(0, 9.0f, 0.01f, 2.0f, 0.9f, 3);
			methodName = "Type-2 Fuzzy GMM-UM with MRF";
			methodType = "Type-2 Fuzzy Type";
			methodAuthor = "Zhao et al. (2012)";
			break;
		case 24:
			// long frameNumber, double threshold, double alpha, float km, float kv, int gaussians, 
			bgs = new T2FMRF_UV(0, 9.0f, 0.01f, 2.0f, 0.9f, 3);
			methodName = "Type-2 Fuzzy GMM-UV with MRF";
			methodType = "Type-2 Fuzzy Type";
			methodAuthor = "Zhao et al. (2012)";
			break;
			// Multiple Features Type
		case 25:
			bgs = new DPTextureBGS();
			methodName = "Texture BGS";
			methodType = "Multiple features";
			methodAuthor = "Heikkila et al. (2006)";
			break;
		case 26:
			bgs = new LbpMrf();
			methodName = "Texture-Based Foreground Detection with MRF";
			methodType = "Multiple features";
			methodAuthor = "Csaba Kertesz (2011)";
			break;
		case 27:
			// long long frameNumber, bool saveModel, bool disableDetectMode, bool disableLearning, int detectAfter,
			// bool loadDefaultParams, int max_mode_num, float weight_updating_constant, float texture_weight, float bg_mode_percent,
			// int pattern_neig_half_size, float pattern_neig_gaus_sigma, float bg_prob_threshold, float bg_prob_updating_threshold,
			// int robust_LBP_constant, float min_noised_angle, float shadow_rate, float highlight_rate, float bilater_filter_sigma_s,
			// float bilater_filter_sigma_r, float frame_duration, float learn_mode_learn_rate_per_second, float learn_weight_learn_rate_per_second,
			// float learn_init_mode_weight, float detect_mode_learn_rate_per_second, float detect_weight_learn_rate_per_second, float detect_init_mode_weight
			bgs = new MultiLayerBGS(0, 0, 1, 0, 0, 1, 5, 5.0f, 0.5f, 0.6f, 4, 3.0f, 0.2f, 0.2f, 3, 0.01768f, 0.6f, 1.2f, 3.0f,
				0.1f, 0.1f, 0.5f, 0.5f, 0.05f, 0.01f, 0.01f, 0.001f);
			methodName = "Multi-Layer BGS";
			methodType = "Multiple features";
			methodAuthor = "Jian Yao and Jean-Marc Odobez (2007)";
			break;
		case 28:
			// int g_iTrainingPeriod2, int g_iT_ModelThreshold2, int g_iC_ModelThreshold2, float g_fLearningRate2,
			// short g_nTextureTrainVolRange2, short g_nColorTrainVolRange2, int g_iAbsortionPeriod2, int g_iRWidth2, int g_iRHeight2,
			// int g_iBackClearPeriod2, int g_iCacheClearPeriod2, short g_nNeighborNum2, short g_nRadius2, int g_iFrameCount2
			bgs = new SJN_MultiCueBGS(20, 1, 10, 0.05f, 15, 20, 200, 160, 120, 300, 30, 6, 2, 0);
			methodName = "MultiCue BGS";
			methodType = "Multiple features";
			methodAuthor = "SeungJong Noh and Moongu Jeon (2012)";
			break;
		case 29:
			// float fRelLBSPThreshold, size_t nDescDistThresholdOffset, size_t nMinColorDistThreshold,
			// size_t nBGSamples, size_t nRequiredBGSamples, size_t nSamplesForMovingAvgs;			
			bgs = new SuBSENSEBGS(0.333f, 3, 30, 50, 2, 100);
			methodName = "SuBSENSE";
			methodType = "Multiple features";
			methodAuthor = "Pierre-Luc et al. (2014)";
			break;
		case 30:
			// bool showOutput, float fRelLBSPThreshold, size_t nLBSPThresholdOffset, size_t nDescDistThreshold
			// , size_t nColorDistThreshold, size_t nBGSamples, size_t nRequiredBGSamples			
			bgs = new LOBSTERBGS(0.365f, 0, 4, 30, 35, 2);
			methodName = "LOBSTER";
			methodType = "Multiple features";
			methodAuthor = "Pierre-Luc and Guillaume-Alexandre (2014)";
			break;
		case 31:
			// bool enableInputBlur,bool enableOutputBlur, float alpha, float beta, int N, int Raute_min,
			// float R_incdec, int R_lower, int R_scale, float T_dec, int T_inc, int T_init, int T_lower,
			// int T_upper
			bgs = new PixelBasedAdaptiveSegmenterBGS(1, 1, 7.0f, 1.0f, 20, 2, 0.05f, 18, 5, 0.05f, 1, 18, 2, 200);
			methodName = "PBAS";
			methodType = "Multiple features";
			methodAuthor = "Pierre-Luc and Guillaume-Alexandre (2014)";
			break;
		case 32:
			// float fRelLBSPThreshold ,size_t nDescDistThresholdOffset ,
			// size_t nMinColorDistThreshold, size_t nMaxNbWords, size_t nSamplesForMovingAvgs			
			bgs = new PAWCSBGS(0.333f, 2, 20, 50, 100);
			methodName = "PAWCS";
			methodType = "Multi features";
			methodAuthor = "St-Charles et al. (2015)";
			break;
			// Non-parametric Type
		case 33:
			// int initializationFrames, double decisionThreshold
			bgs = new GMG(20, 0.7);
			methodName = "GMG";
			methodType = "Non-parametric";
			methodAuthor = "Godbehere et al. (2012)";
			break;
		case 34:
			// bool enableFilter, int binSize, double alpha, double threshold
			bgs = new VuMeter(1, 8, 0.995, 0.03);
			methodName = "VuMeter";
			methodType = "Non-parametric";
			methodAuthor = "Goyat et al. (2006)";
			break;
		case 35:
			// int SequenceLength,int TimeWindowSize = 100,int SDEstimationFlag = 1,int lUseColorRatiosFlag,double th = 10e-8,
			// double alpha,int framesToLearn = 10,int frameNumber;
			bgs = new KDE(50, 100, 1, 1, 10e-8, 0.3, 10, 0);
			methodName = "KDE";
			methodType = "Non-parametric";
			methodAuthor = "Elgammal et al. (2000)";
			break;
		case 36:
			// int fps
			bgs = new IndependentMultimodalBGS(10);
			methodName = "IMBS";
			methodType = "Non-parametric";
			methodAuthor = "Domenico Bloisi and Luca Iocchi (2012)";
			break;
			//	// Subspace Type
		case 37:
			// long frameNumber,int threshold,int historySize,int embeddedDim
			bgs = new DPEigenbackgroundBGS(0, 225, 20, 10);
			methodName = "Eigenbackground / SL-PCA";
			methodType = "Subspace";
			methodAuthor = "Oliver et al. (2000)";
			break;
			// Neural and Neuro-Fuzzy Type
		case 38:
			// int sensitivity, int trainingSensitivity, int learningRate, int trainingLearningRate, int trainingSteps
			bgs = new LBAdaptiveSOM(75, 245, 62, 255, 55);
			methodName = "Adaptive SOM";
			methodType = "Neural and neuro-fuzzy";
			methodAuthor = "Maddalena and Petrosino (2008)";
			break;
		case 39:
			// int sensitivity, int trainingSensitivity, int learningRate, int trainingLearningRate, int trainingSteps
			bgs = new LBFuzzyAdaptiveSOM(90, 240, 38, 255, 81);
			methodName = "Fuzzy Adaptive SOM";
			methodType = "Neural and neuro-fuzzy";
			methodAuthor = "Maddalena and Petrosino (2010)";
			break;
		default:
			std::cout << "Wrong method id. Please input method ID again." << std::endl;
			success = false;
			break;
		}
	}
	return bgs;
}