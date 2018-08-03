#include <opencv2/opencv.hpp>
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/features2d.hpp"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

const int MAX_FEATURES = 2000;
const float GOOD_MATCH_PERCENT = 0.15f;

void alignImages(Mat &im1, Mat &im2, Mat &im1Reg, Mat &h, std::string &leftName, std::string &rightName)

{
	

	// Convert images to grayscale
	Mat im1Gray, im2Gray;
	//cvtColor(LeftProjection, im1Gray, COLOR_BGR2GRAY);
	//cvtColor(RightProjection, im2Gray, COLOR_BGR2GRAY);
	cvtColor(im1, im1Gray, COLOR_BGR2GRAY);
	cvtColor(im2, im2Gray, COLOR_BGR2GRAY);




	//single image detect
	std::vector<KeyPoint> keypoints1, keypoints2;
	Mat descriptors1, descriptors2;
	Ptr<Feature2D> sift = SIFT::create();


	sift->detectAndCompute(im1Gray, Mat(), keypoints1, descriptors1);
	std::cout << "\tDetecting right keypoints/descriptors..." << std::endl;
	sift->detectAndCompute(im2Gray, Mat(), keypoints2, descriptors2);
	descriptors1.convertTo(descriptors1, CV_32F);
	descriptors2.convertTo(descriptors2, CV_32F);

	//single image matching

	// CHANGED
	// Match features.
	std::cout << "\tMatching Descriptors..." << std::endl;
	std::vector<DMatch> matches;
	Ptr<DescriptorMatcher> matcher = FlannBasedMatcher::create();
	matcher->match(descriptors1, descriptors2, matches);





	//single image good image matching

	std::cout << "\tSelecting Good Matches..." << std::endl;

	double max_dist = 0; double min_dist = 100;
	for (int i = 0; i < descriptors1.rows; i++)
	{
	double dist = matches[i].distance;
	if (dist < min_dist) min_dist = dist;
	if (dist > max_dist) max_dist = dist;
	}

	std::vector< DMatch > good_matches;
	for (int i = 0; i < descriptors1.rows; i++)
	{
	if (matches[i].distance <= max(2 * min_dist, 0.02))
	{
	good_matches.push_back(matches[i]);
	}
	}
	Mat tmp;
	drawMatches(im1, keypoints1, im2, keypoints2, good_matches, tmp);
	imwrite("goodMatches.jpg", tmp);

	// Sort matches by score
	std::cout << "\tSorting Matches..." << std::endl;
	std::sort(good_matches.begin(), good_matches.end(), [](DMatch a, DMatch b)
	{return (std::abs(a.distance) < std::abs(b.distance));} );

	// Remove not so good matches
	std::vector<KeyPoint> LeftGoodKeypoints, RightGoodKeypoints;
	std::vector< DMatch > TopTenMatches;
	for (int i = 0; i < 10; i++) {
	int i1 = good_matches[i].queryIdx;
	int i2 = good_matches[i].trainIdx;
	CV_Assert(i1 >= 0 && i1 < static_cast<int>(keypoints1.size()));
	CV_Assert(i2 >= 0 && i2 < static_cast<int>(keypoints2.size()));
	LeftGoodKeypoints.push_back(keypoints1[i1]);
	RightGoodKeypoints.push_back(keypoints2[i2]);
	TopTenMatches.push_back(good_matches[i]);
	}





	//single image distance matching

	std::cout << "\tCalculating Distance..." << std::endl;
	double LeftDistance[10][10], RightDistance[10][10], GLDistance[10][10], GRDistance[10][10];
	double Ldistance, Rdistance, l2 = 0.0;
	for (int i = 0; i < 10; i++) {
		for (int j = 0; j < 10; j++) {
			Ldistance = sqrt(pow((LeftGoodKeypoints[j].pt.x - LeftGoodKeypoints[i].pt.x),2)
				+ pow((LeftGoodKeypoints[j].pt.y - LeftGoodKeypoints[i].pt.y),2));
			Rdistance = sqrt(pow((RightGoodKeypoints[j].pt.x - RightGoodKeypoints[i].pt.x), 2)
				+ pow((RightGoodKeypoints[j].pt.y - RightGoodKeypoints[i].pt.y), 2));
			GLDistance[i][j] = min(Ldistance, std::abs(double(im1.size().width) - Ldistance));
			GRDistance[i][j] = min(Rdistance, std::abs(double(im2.size().width) - Rdistance));
			LeftDistance[i][j] = Ldistance;
			RightDistance[i][j] = Rdistance;
			l2 += sqrt(pow(Ldistance - Rdistance, 2));
		}
	}

	


	//image detail prints


	FileStorage fs4("GoodKeypoints.txt", FileStorage::WRITE);
	write(fs4, "good_keypoints", good_matches);
	fs4.release();
	FileStorage fs5("leftGoodKeypoints.txt", FileStorage::WRITE);
	write(fs5, "left_good_keypoints", LeftGoodKeypoints);
	fs5.release();
	FileStorage fs6("rightGoodKeypoints.txt", FileStorage::WRITE);
	write(fs6, "right_good_keypoints", RightGoodKeypoints);
	fs6.release();
	FileStorage fs7("TopTenKeypoints.txt", FileStorage::WRITE);
	write(fs7, "top_tent_keypoints", TopTenMatches);
	fs7.release();

	ofstream fs8("LeftVectors.csv",std::ofstream::out);
	ofstream fs9("RightVectors.csv", std::ofstream::out);
	ofstream fs10("EVectors.csv", std::ofstream::out);
	fstream fs12("L2_results.txt", ios::in | ios::out | ios::app);
	fs12 << leftName << ", " << rightName << ": " << l2 << endl;

	for (int i = 0; i < 10; i++) {
	for (int j = 0; j < 10; j++) {
	//		fs8 << std::to_string(LeftDistance[i][j]) <<",";
	//		fs9 << std::to_string(RightDistance[i][j]) <<",";
	//		fs10 << std::to_string(GLDistance[i][j]) << ",";
	//		fs11 << std::to_string(GRDistance[i][j]) << ",";

	fs8 << GLDistance[i][j] << "," ;
	fs9 << RightDistance[i][j] << "," ;
	}
	fs8 << "\n";
	fs9 << "\n";
	fs10 << "\n";
	//	fs11 << "\n";
	}

	//
	fs8.close();
	fs9.close();
	fs10.close();
	//fs11.close();
	fs12.close();





	//single image draws

	// Draw top matches
	Mat imMatches;
	Mat TenMatches;
	drawMatches(im1, keypoints1, im2, keypoints2, good_matches, imMatches);
	drawMatches(im1, keypoints1, im2, keypoints2, TopTenMatches, TenMatches);
	// CHANGED
	imwrite("matches.jpg", imMatches);
	imwrite("top_ten_matches.jpg", TenMatches);
	




	//imshow("Display Left Projection", im1);
	//imshow("Display Right Projection", im2);

	//imwrite("matches-brisk-flann.jpg", imMatches);
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
	alignImages(im, imReference, imReg, h, refFilename, imFilename);



}