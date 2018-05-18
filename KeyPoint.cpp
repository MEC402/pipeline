#include <opencv2/opencv.hpp>
//#include "opencv2/xfeatures2d.hpp"
#include "opencv2/features2d.hpp"

using namespace std;
using namespace cv;
//using namespace cv::xfeatures2d;

const int MAX_FEATURES = 2000;
const float GOOD_MATCH_PERCENT = 0.15f;


void alignImages(Mat &im1, Mat &im2, Mat &im1Reg, Mat &h)

{
	// Convert images to grayscale
	Mat im1Gray, im2Gray;
	cvtColor(im1, im1Gray, CV_BGR2GRAY);
	cvtColor(im2, im2Gray, CV_BGR2GRAY);

	// Variables to store keypoints and descriptors
	std::vector<KeyPoint> keypoints1, keypoints2;
	Mat descriptors1, descriptors2;

	// CHANGED
	 //Detect ORB features and compute descriptors.
	 Ptr<Feature2D> orb = ORB::create(MAX_FEATURES);
	 orb->detectAndCompute(im1Gray, Mat(), keypoints1, descriptors1);
	 orb->detectAndCompute(im2Gray, Mat(), keypoints2, descriptors2);
	std::cout << "\tDetecting keypoints/descriptors..." << std::endl;
	/*Ptr<Feature2D> brisk = BRISK::create();
	brisk->detectAndCompute(im1Gray, Mat(), keypoints1, descriptors1);
	brisk->detectAndCompute(im2Gray, Mat(), keypoints2, descriptors2);*/
	descriptors1.convertTo(descriptors1, CV_32F);
	descriptors2.convertTo(descriptors2, CV_32F);

	// CHANGED
	// Match features.
	std::cout << "\tMatching Descriptors..." << std::endl;
	std::vector<DMatch> matches;
	// Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
	Ptr<DescriptorMatcher> matcher = FlannBasedMatcher::create();
	matcher->match(descriptors1, descriptors2, matches, Mat());

	FileStorage fs("LeftKeypoints.txt", FileStorage::WRITE);
	write(fs, "left_keypoints", keypoints1);
	fs.release();
	FileStorage fs2("RightKeypoints.txt", FileStorage::WRITE);
	write(fs2, "right_keypoints", keypoints2);
	fs2.release();
	FileStorage fs3("MatchKeypoints.txt", FileStorage::WRITE);
	write(fs3, "match_keypoints", matches);
	fs3.release();

	//double max_dist = 0; double min_dist = 100;
	////-- Quick calculation of max and min distances between keypoints
	//for (int i = 0; i < descriptors1.rows; i++)
	//{
	//	double dist = matches[i].distance;
	//	if (dist < min_dist) min_dist = dist;
	//	if (dist > max_dist) max_dist = dist;
	//}

	//std::vector< DMatch > good_matches;
	//for (int i = 0; i < descriptors1.rows; i++)
	//{
	//	if (matches[i].distance <= max(2 * min_dist, 0.02))
	//	{
	//		good_matches.push_back(matches[i]);
	//	}
	//}

	// Sort matches by score
	std::cout << "\tSorting Matches..." << std::endl;
	std::sort(matches.begin(), matches.end());
	// std::reverse(matches.begin(), matches.end());

	// Remove not so good matches

	const int numGoodMatches = matches.size() * GOOD_MATCH_PERCENT;
	matches.erase(matches.begin() + numGoodMatches, matches.end());
	FileStorage fs4("GoodKeypoints.txt", FileStorage::WRITE);
	write(fs4, "good_keypoints", matches);
	fs4.release();

	// Draw top matches
	Mat imMatches;
	drawMatches(im1, keypoints1, im2, keypoints2, matches, imMatches);
	// CHANGED
	 imwrite("matches.jpg", imMatches);
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
	alignImages(im, imReference, imReg, h);



}