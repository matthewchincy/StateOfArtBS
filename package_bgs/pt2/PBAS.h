#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
#include "../IBGS.h"

class PBAS : public IBGS
{
private:
	void saveConfig();
	void loadConfig();
	void checkValid(int *x, int *y);
	void createRandomNumberArray();
	void checkXY(cv::Point2i*);
	void getFeatures(std::vector<cv::Mat>* descriptor, cv::Mat* intImg);

	bool isMovement;
	double beta, alpha;

	//####################################################################################
	//N - Number: Defining the size of the background-history-model
	// number of samples per pixel
	int N;
	// background model
	std::vector<cv::Mat> backgroundModel;
	//####################################################################################
	//####################################################################################
	//R-Threshhold - Variables
	// radius of the sphere -> lower border boundary
	double R;
	// scale for the sphere threshhold to define pixel-based Thresholds
	double rThreshScale;
	// increasing/decreasing factor for the r-Threshold based on the result of rTreshScale * meanMinDistBackground
	double rIncDecFac;
	cv::Mat sumThreshBack, rThresh; //arrays
	float *sumArrayDistBack; //sum of minDistBackground ->pointer to arrays
	float *rData; //new pixel-based r-threshhold -> pointer to arrays
				  //#####################################################################################
				  //####################################################################################
				  // Defining the number of background-model-images, which have a lowerDIstance to the current Image than defined by the R-Thresholds, that are necessary
				  // to decide that this pixel is background
	int parts;
	//#####################################################################################
	//####################################################################################
	// Initialize the background-model update rate 
	int nrSubsampling;

	// scale that defines the increasing of the update rate of the background model, if the current pixel is background 
	//--> more frequently updates if pixel is background because, there shouln't be any change
	double increasingRateScale;
	// defining an upper value, that nrSubsampling can achieve, thus it doesn't reach to an infinite value, where actually no update is possible 
	// at all
	int upperTimeUpdateRateBoundary;
	//holds update rate of current pixel
	cv::Mat tempCoeff;
	float *tCoeff;
	// opposite scale to increasingRateScale, for decreasing the update rate of the background model, if the current pixel is foreground
	//--> Thesis: Our Foreground is a real moving object -> thus the background-model is good, so don't update it
	double decreasingRateScale;
	// defining a minimum value for nrSubsampling --> base value 2.0
	int lowerTimeUpdateRateBoundary;
	//#####################################################################################




	// background/foreground segmentation map -> result Map
	cv::Mat* segMap;

	// background and foreground identifiers
	int foreground, background;

	int height, width;

	//random number generator
	cv::RNG randomGenerator;

	//pre - initialize the randomNumbers for better performance
	std::vector<int> randomSubSampling, randomN, randomX, randomY, randomDist;
	int runs, countOfRandomNumb;

	uchar* data, *segData;
	float* dataBriefNorm, *dataBriefDir;
	uchar* dataBriefCol;
	uchar* dataStats;

	std::vector<uchar*> backgroundPt;
	std::vector<uchar*>backgroundPtBriefCol;
	std::vector<float*> backgroundPtBriefNorm, backgroundPtBriefDir;

	std::vector<std::vector<cv::Mat>> backGroundFeatures;

	std::vector<cv::Mat> temp, imgFeatures;

	//cv::ORB orb;
	//std::vector<cv::KeyPoint> keypoints1;
	//cv::Mat descr1;
	int xNeigh, yNeigh;
	float meanDistBack;
	double formerMaxNorm, formerMaxPixVal, formerMaxDir;

	//new background distance model
	std::vector<float*> distanceStatPtBack;/*,distanceStatPtFore;*/
	std::vector<cv::Mat*> distanceStatisticBack; /*,distanceStatisticFore;*/

	float formerDistanceFore, formerDistanceBack;
	cv::Mat* tempDist, *tempDistB;
	double setR;
	cv::Mat sobelX, sobelY;
	bool showOutput;
public:
	PBAS();
	PBAS(int N, double R, double rThreshScale, double rIncDecFac, int parts, int nrSubsampling,
		double increasingRateScale, double decreasingRateScale
		, int lowerTimeUpdateRateBoundary, int upperTimeUpdateRateBoundar
		, double formerMaxNorm, double formerMaxDir, double formerMaxPixVal, int height, int width
		, int runs, int countOfRandomNumb, bool isMovement, double beta, double alpha, bool showOutput);
	~PBAS();
	void process(const cv::Mat &input, cv::Mat &img_output, cv::Mat &img_bgmodel);
	//void process(cv::Mat *input, cv::Mat*);
	void initialization(int newN, double newR, int newParts, int newNrSubSampling, double a, double b, double rThrSc, double rIndDec, double incrTR, double decrTR, int lowerTB, int upperTB);
	void setConstForeground(double constF);
	void setConstBackground(double constB);
	void setAlpha(double alph);
	void setBeta(double bet);
	bool isMoving();
	double getR();
	cv::Mat* getTImage();
	cv::Mat* getRImage();
};
