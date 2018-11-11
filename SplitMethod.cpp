#include <opencv2/opencv.hpp>
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/features2d.hpp"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;


void alignImages(Mat &im1, Mat &im2, Mat &im1Reg, Mat &h, std::string &leftName, std::string &rightName)

{

	// Convert images to grayscale
	Mat im1Gray, im2Gray;
	cvtColor(im1, im1Gray, COLOR_BGR2GRAY);
	cvtColor(im2, im2Gray, COLOR_BGR2GRAY);
	imwrite("leftGray.jpg", im1Gray);
	imwrite("rightGray.jpg", im2Gray);

	//smallheight = im1.size().height/2
	int smallWidth = im1.size().width / 4, smallheight = im1.size().height, splitSize = smallWidth * smallheight;

	std::cout << "\tSpliting image..." << std::endl;
	std::vector<cv::Mat> splitImagesLeft, splitImagesRight;
	for (int y = 0; y < im1.size().height; y += smallheight) {
		for (int x = 0; x < im1.size().width; x += smallWidth) {
			cv::Rect rect = cv::Rect(x, y, smallWidth, smallheight);
			splitImagesLeft.push_back(cv::Mat(im1Gray, rect).clone());
			splitImagesRight.push_back(cv::Mat(im2Gray, rect).clone());
		}
	}
	std::cout << "\tprint split image test:.... " << std::endl;
	imwrite("splitTestL.jpg", splitImagesLeft[0]);
	imwrite("splitTestR.jpg", splitImagesRight[0]);


	std::cout << "\tDetecting left keypoints/descriptors..." << std::endl;
	// Variables to store keypoints and descriptors
	std::vector<std::vector<cv::KeyPoint>> leftKeypoints(4), rightKeypoints(4),
		LeftGoodKeypoints(4), RightGoodKeypoints(4);
	std::vector<cv::KeyPoint> correctLeftKey, correctRightKey;
	std::vector<Mat> leftDescriptors(4), rightDescriptors(4);
	std::vector<std::vector<DMatch>> goodSplitMatches(4), goodMatches(4);
	int skips = 0;



	Ptr<Feature2D> sift = SIFT::create(10000);
	std::vector<std::vector<DMatch>> splitMatches(8);
	Ptr<DescriptorMatcher> matcher = FlannBasedMatcher::create();
	Mat splitMatch, goodSplitMatch;
	for (int i = 0; i < 4; i++) {
		std::cout << "\tDetecting split image: " << i << endl;
		sift->detectAndCompute(splitImagesLeft[i], Mat(), leftKeypoints[i], leftDescriptors[i]);
		sift->detectAndCompute(splitImagesRight[i], Mat(), rightKeypoints[i], rightDescriptors[i]);
		if (leftDescriptors[i].empty() || rightDescriptors[i].empty()) {
			std::cout << "\tno keypoints skiping split " << i << endl;
			skips++;
			continue;

		}
		leftDescriptors[i].convertTo(leftDescriptors[i], CV_32F);
		rightDescriptors[i].convertTo(rightDescriptors[i], CV_32F);
		std::cout << "\tmatching split " << i << "..." << endl;
		matcher->match(leftDescriptors[i], rightDescriptors[i], splitMatches[i]);
		double min = 50;
		std::cout << "geting min distance for split: " << i << " ... " << endl;
		drawMatches(splitImagesLeft[i], leftKeypoints[i], splitImagesRight[i], rightKeypoints[i], splitMatches[i], splitMatch);
		imwrite("splitMatch.jpg", splitMatch);
		imwrite("splitTestL.jpg", splitImagesLeft[i]);
		imwrite("splitTestR.jpg", splitImagesRight[i]);
		std::sort(splitMatches[i].begin(), splitMatches[i].end(), [](DMatch a, DMatch b)
		{return (std::abs(a.distance) < std::abs(b.distance)); });
		if (splitMatches[i][0].distance < min) {
			min = splitMatches[i][0].distance;
		}
		for (int j = 0; j < leftDescriptors[i].rows; j++) {

			if (splitMatches[i][j].distance <= std::max(2 * min, 50.0)) {
				goodSplitMatches[i].push_back(splitMatches[i][j]);
			}
		}
		drawMatches(splitImagesLeft[i], leftKeypoints[i], splitImagesRight[i], rightKeypoints[i], goodSplitMatches[i], goodSplitMatch);
		imwrite("goodSplitMatch.jpg", goodSplitMatch);
		std::cout << "\tSorting Matches of split:" << i << " ..." << std::endl;
		std::sort(goodSplitMatches[i].begin(), goodSplitMatches[i].end(), [](DMatch a, DMatch b)
		{return (std::abs(a.distance) < std::abs(b.distance)); });
		for (int j = 0; j < 2; j++) {
			std::cout << "\tgeting top matches" << endl;
			goodMatches[i].push_back(goodSplitMatches[i][j]);
		}

		for (int j = 0; j < goodMatches[i].size(); j++) {
			int i1 = goodMatches[i][j].queryIdx;
			int i2 = goodMatches[i][j].trainIdx;
			CV_Assert(i1 > 0 && i1 < static_cast<int>(leftKeypoints[i].size()));
			CV_Assert(i2 > 0 && i2 < static_cast<int>(rightKeypoints[i].size()));
			LeftGoodKeypoints[i].push_back(leftKeypoints[i][i1]);
			RightGoodKeypoints[i].push_back(rightKeypoints[i][i2]);
		}
		std::cout << "test" << endl;
		for (int j = 0; j < LeftGoodKeypoints[i].size(); j++) {
			std::cout << "test 2" << endl;

			switch (i) {
			case 0: //LeftGoodKeypoints[i][j].pt.y = LeftGoodKeypoints[i][j].pt.y + smallheight;
				break;
			case 1: //LeftGoodKeypoints[i][j].pt.y = LeftGoodKeypoints[i][j].pt.y + smallheight;
				LeftGoodKeypoints[i][j].pt.x = LeftGoodKeypoints[i][j].pt.x + smallWidth;
				break;
			case 2: //LeftGoodKeypoints[i][j].pt.y = LeftGoodKeypoints[i][j].pt.y + smallheight;
				LeftGoodKeypoints[i][j].pt.x = LeftGoodKeypoints[i][j].pt.x + smallWidth * 2;
				break;
			case 3: //LeftGoodKeypoints[i][j].pt.y = LeftGoodKeypoints[i][j].pt.y + smallheight;
				LeftGoodKeypoints[i][j].pt.x = LeftGoodKeypoints[i][j].pt.x + smallWidth * 3;
				break;
				/*case 5:	LeftGoodKeypoints[i][j].pt.x = LeftGoodKeypoints[i][j].pt.x + smallWidth;
				cout << "\tx keypoint after change : " << std::to_string(LeftGoodKeypoints[i][j].pt.x) << endl;
				break;
				case 6: LeftGoodKeypoints[i][j].pt.x = LeftGoodKeypoints[i][j].pt.x + smallWidth * 2;
				break;
				case 7:cout << "\tx keypoint before change : " << std::to_string(LeftGoodKeypoints[i][j].pt.x) << endl;
				LeftGoodKeypoints[i][j].pt.x = LeftGoodKeypoints[i][j].pt.x + smallWidth * 3;
				cout << "\tx keypoint after change : " << std::to_string(LeftGoodKeypoints[i][j].pt.x) << endl;
				break;*/
			}

			correctLeftKey.push_back(LeftGoodKeypoints[i][j]);
		}
		std::cout << "end test" << endl;
		for (int j = 0; j < RightGoodKeypoints[i].size(); j++) {
			switch (i) {
			case 0: //RightGoodKeypoints[i][j].pt.y = RightGoodKeypoints[i][j].pt.y + smallheight;
				break;
			case 1: //RightGoodKeypoints[i][j].pt.y = RightGoodKeypoints[i][j].pt.y + smallheight;
				RightGoodKeypoints[i][j].pt.x = RightGoodKeypoints[i][j].pt.x + smallWidth;
				break;
			case 2: //RightGoodKeypoints[i][j].pt.y = RightGoodKeypoints[i][j].pt.y + smallheight;
				RightGoodKeypoints[i][j].pt.x = RightGoodKeypoints[i][j].pt.x + smallWidth * 2;
				break;
			case 3:// RightGoodKeypoints[i][j].pt.y = RightGoodKeypoints[i][j].pt.y + smallheight;
				RightGoodKeypoints[i][j].pt.x = RightGoodKeypoints[i][j].pt.x + smallWidth * 3;
				break;
				/*case 5: RightGoodKeypoints[i][j].pt.x = RightGoodKeypoints[i][j].pt.x + smallWidth;
				break;
				case 6: RightGoodKeypoints[i][j].pt.x = RightGoodKeypoints[i][j].pt.x + smallWidth * 2;
				break;
				case 7: RightGoodKeypoints[i][j].pt.x = RightGoodKeypoints[i][j].pt.x + smallWidth * 3;*/
			}

			correctRightKey.push_back(RightGoodKeypoints[i][j]);
		}
	}



	std::cout << "\tCalculating Vectors..." << std::endl;
	vector<vector<vector<float>>> LeftVectors(10, vector<vector<float>>(10, vector<float>(2))),
		RightVectors(10, vector<vector<float>>(10, vector<float>(2))),
		EVectors(10, vector<vector<float>>(10, vector<float>(2)));
	double l2 = 0.0;
	Mat LeftVectorImage = im1Gray.clone(), RightVectorImage = im2Gray.clone();
	vector<Point> leftPoints, rightPoints;

	for (int i = 0; i < correctLeftKey.size(); i++) {
		for (int j = 0; j < correctRightKey.size(); j++) {
			vector<float> tmpL;
			tmpL.push_back(correctLeftKey[j].pt.x - correctLeftKey[i].pt.x);
			tmpL.push_back(correctLeftKey[j].pt.y - correctLeftKey[i].pt.y);
			LeftVectors[i][j] = tmpL;

			vector<float> tmpR;
			tmpR.push_back(correctRightKey[j].pt.x - correctRightKey[i].pt.x);
			tmpR.push_back(correctRightKey[j].pt.y - correctRightKey[i].pt.y);
			RightVectors[i][j] = tmpR;

			vector<float> tmpE;
			tmpE.push_back(tmpR[0] - tmpL[0]);
			tmpE.push_back(tmpR[1] - tmpL[1]);
			EVectors[i][j] = tmpE;

			l2 += sqrt(pow(tmpE[0], 2) + pow(tmpE[1], 2));

			if (tmpE[0] == 0 && tmpE[1] == 0) {
				continue;
			}
			else {
				Point Left1(correctLeftKey[i].pt.x, correctLeftKey[i].pt.y), Left2(correctLeftKey[i].pt.x + tmpE[0], correctLeftKey[i].pt.y + tmpE[1]),
					Right1(correctRightKey[i].pt.x, correctRightKey[i].pt.y), Right2(correctRightKey[i].pt.x + tmpE[0], correctRightKey[i].pt.y + tmpE[1]);
				arrowedLine(LeftVectorImage, Left1, Left2, COLORMAP_RAINBOW, 15);
				arrowedLine(RightVectorImage, Right1, Right2, COLORMAP_RAINBOW, 15);
			}
		}
	}
	imwrite("LeftVectorDraw.jpg", LeftVectorImage);
	imwrite("RightVectorDraw.jpg", RightVectorImage);

	ofstream fs18("splitLeftVectors.csv", std::ofstream::out);
	ofstream fs19("splitRightVectors.csv", std::ofstream::out);
	ofstream fs20("splitEVectors.csv", std::ofstream::out);


	for (int i = 0; i < 10; i++) {
		for (int j = 0; j < 10; j++) {

			fs18 << LeftVectors[i][j][0] << "," << LeftVectors[i][j][1] << ",";
			fs19 << RightVectors[i][j][0] << "," << RightVectors[i][j][1] << ",";
			fs20 << EVectors[i][j][0] << "," << EVectors[i][j][1] << ",";
		}
		fs18 << "\n";
		fs19 << "\n";
		fs20 << "\n";
	}

	fs18.close();
	fs19.close();
	fs20.close();
	fstream fs13("Vsplit_L2_results.txt", ios::in | ios::out | ios::app);
	fs13 << leftName << ", " << rightName << ": " << l2 << ",   skips: " << skips << endl;


	FileStorage fs6("correctLeftKey.txt", FileStorage::WRITE);
	write(fs6, "correctLeftKey", correctLeftKey);
	fs6.release();
	Mat keyLeft;
	drawKeypoints(im1, correctLeftKey, keyLeft);
	imwrite("keyLeft.jpg", keyLeft);


	FileStorage fs("Descriptor.txt", FileStorage::WRITE);
	write(fs, "descritor of split 0", leftDescriptors[0]);
	fs.release();
	Mat tmp;
	drawKeypoints(splitImagesLeft[0], leftKeypoints[0], tmp);
	imwrite("keyTest.jpg", tmp);
}


int main(int argc, char **argv)
{
	// Read reference image
	string refFilename(argv[1]);
	cout << "Reading reference image : " << refFilename << endl;
	Mat imReference = imread(refFilename);


	// Read image to be aligned
	string imFilename(argv[2]);
	cout << "Reading image to align : " << imFilename << endl;
	Mat im = imread(imFilename);


	// Registered image will be resotred in imReg. 
	// The estimated homography will be stored in h. 
	Mat imReg, h;

	// Align images
	cout << "Aligning images ..." << endl;
	alignImages(imReference, im, imReg, h, refFilename, imFilename);



}