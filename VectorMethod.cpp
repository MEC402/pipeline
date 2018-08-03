#include <opencv2/opencv.hpp>
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/features2d.hpp"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

const int MAX_FEATURES = 2000;
const float GOOD_MATCH_PERCENT = 0.15f;
Mat eulerAnglesToRotationMatrix(Vec3f &theta)
{
	// Calculate rotation about x axis
	Mat R_x = (Mat_<float>(3, 3) <<
		1, 0, 0,
		0, cosf(theta[0]), -sinf(theta[0]),
		0, sinf(theta[0]), cosf(theta[0])
		);
	// Calculate rotation about y axis
	Mat R_y = (Mat_<float>(3, 3) <<
		cosf(theta[1]), 0, sinf(theta[1]),
		0, 1, 0,
		-sinf(theta[1]), 0, cosf(theta[1])
		);
	// Calculate rotation about z axis
	Mat R_z = (Mat_<float>(3, 3) <<
		cosf(theta[2]), -sinf(theta[2]), 0,
		sinf(theta[2]), cosf(theta[2]), 0,
		0, 0, 1);

	// Combined rotation matrix
	Mat R = R_z * R_y * R_x;
	return R;
}

void alignImages(Mat &im1, Mat &im2, Mat &im1Reg, Mat &h, std::string &leftName, std::string &rightName)

{
//	Mat LeftProjection, RightProjection;
//	float fx = (20 * im1.size().width) / 17.3;
//	Mat L = (Mat_<float>(3, 3) <<
//		fx, 0, im1.size().width / 2,
//		0, fx, im1.size().height / 2,
//		0, 0, 1);
//	Vec3f rot = Vec3f{ 0, 0.5, 0 };
//	Mat R = eulerAnglesToRotationMatrix(rot);
//	// Mat R = Mat::eye(3, 3, CV_32F);
//	std::cout << "\tWarping Images..." << std::endl;
//	//detail::SphericalWarper warper = detail::SphericalWarper( (4* 3.141592653*2));
//	detail::SphericalWarper warper = detail::SphericalWarper((fx / 20) * 2 * 3.141592653);
//	// detail::SphericalWarper warper = detail::SphericalWarper();
//	warper.warp(im1, L, R, INTER_LINEAR, BORDER_CONSTANT, LeftProjection);
//	//warper = detail::SphericalWarper((RightImage.size().width / (2 * 3.141592653)) / 8);
//	warper.warp(im2, L, R, INTER_LINEAR, BORDER_CONSTANT, RightProjection);

	// Convert images to grayscale
	Mat im1Gray, im2Gray;
	//cvtColor(LeftProjection, im1Gray, COLOR_BGR2GRAY);
	//cvtColor(RightProjection, im2Gray, COLOR_BGR2GRAY);
	cvtColor(im1, im1Gray, COLOR_BGR2GRAY);
	cvtColor(im2, im2Gray, COLOR_BGR2GRAY);
	imwrite("leftGray.jpg", im1Gray);
	imwrite("rightGray.jpg", im2Gray);

	//smallheight = im1.size().height/2
	int smallWidth = im1.size().width / 4, smallheight = im1.size().height, splitSize = smallWidth*smallheight;

	std::cout << "\tSpliting image..." << std::endl;
	std::vector<cv::Mat> splitImagesLeft,splitImagesRight;
	for (int y = 0; y < im1.size().height; y += smallheight){
		for (int x = 0; x < im1.size().width; x += smallWidth) {
			
			cv::Rect rect = cv::Rect(x, y, smallWidth, smallheight);
			//cv::Mat tmp = cv::Mat(im1Gray, rect).clone();
		
			//splitImagesLeft.push_back(im1Gray(rect).clone());
			//splitImagesRight.push_back(im2Gray(rect));
			splitImagesLeft.push_back(cv::Mat(im1Gray, rect).clone());
			splitImagesRight.push_back(cv::Mat(im2Gray, rect).clone());
		}
	}
	std::cout << "\tprint split image test:.... " << std::endl;
	//imshow("sfd", splitImagesLeft[0]);
	//imshow("sfd", im1Gray(cv::Rect(0, 0, im1.size().width / 4, im1.size().height / 2)));
	imwrite("splitTestL.jpg", splitImagesLeft[0]);
	imwrite("splitTestR.jpg", splitImagesRight[0]);


	//imshow("sfd", cv::Mat(im1, cv::Rect(0, 0, im1.size().width / 4, im1.size().height / 2)));
	// CHANGED
	//Detect ORB features and compute descriptors.
	//Ptr<Feature2D> orb = ORB::create(50000);
	//orb->detectAndCompute(im1Gray, Mat(), keypoints1, descriptors1);
	//orb->detectAndCompute(im2Gray, Mat(), keypoints2, descriptors2);
	std::cout << "\tDetecting left keypoints/descriptors..." << std::endl;
	// Variables to store keypoints and descriptors
	std::vector<std::vector<cv::KeyPoint>> leftKeypoints(4), rightKeypoints(4),
		LeftGoodKeypoints(4), RightGoodKeypoints(4);
	std::vector<cv::KeyPoint> correctLeftKey, correctRightKey;
	std::vector<Mat> leftDescriptors(4), rightDescriptors(4);
	std::vector<std::vector<DMatch>> goodSplitMatches(4),goodMatches(4);
	int skips = 0;

	
	
	//std::vector<KeyPoint> keypoints1, keypoints2;
	//Mat descriptors1, descriptors2;

	//Ptr<Feature2D> brisk = BRISK::create();
	//brisk->detectAndCompute(im1Gray, Mat(), keypoints1, descriptors1);
	//brisk->detectAndCompute(im2Gray, Mat(), keypoints2, descriptors2);
	Ptr<Feature2D> sift = SIFT::create(10000);
	std::vector<std::vector<DMatch>> splitMatches(8);
	Ptr<DescriptorMatcher> matcher = FlannBasedMatcher::create();
	Mat splitMatch,goodSplitMatch;
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
					LeftGoodKeypoints[i][j].pt.x = LeftGoodKeypoints[i][j].pt.x + smallWidth*2;
					break;
				case 3: //LeftGoodKeypoints[i][j].pt.y = LeftGoodKeypoints[i][j].pt.y + smallheight;
					LeftGoodKeypoints[i][j].pt.x = LeftGoodKeypoints[i][j].pt.x + smallWidth*3;
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


	//std::cout << "\tCalculating Distance..." << std::endl;
	//vector<vector<double>> LeftDistance(16), RightDistance(16), GLDistance(16), GRDistance(16);
	//double Ldistance, Rdistance, l2 = 0.0;
	//for (int i = 0; i < correctLeftKey.size(); i++) {
	//	for (int j = 0; j < correctRightKey.size(); j++) {
	//		Ldistance = sqrt(pow((correctLeftKey[j].pt.x - correctLeftKey[i].pt.x),2)
	//		+ pow((correctLeftKey[j].pt.y - correctLeftKey[i].pt.y),2));
	//		Rdistance = sqrt(pow((correctRightKey[j].pt.x - correctRightKey[i].pt.x), 2)
	//		+ pow((correctRightKey[j].pt.y - correctRightKey[i].pt.y), 2));
	//		GLDistance[i].push_back(min(Ldistance, std::abs(double(im1.size().width) - Ldistance)));
	//		GRDistance[i].push_back(min(Rdistance, std::abs(double(im2.size().width) - Rdistance)));
	//		LeftDistance[i].push_back(Ldistance);
	//		RightDistance[i].push_back(Rdistance);
	//		l2 += sqrt(pow(GLDistance[i][j] - GRDistance[i][j], 2)) ;
	//	}
	//}

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
			//		fs8 << std::to_string(LeftDistance[i][j]) <<",";
			//		fs9 << std::to_string(RightDistance[i][j]) <<",";
			//		fs10 << std::to_string(GLDistance[i][j]) << ",";
			//		fs11 << std::to_string(GRDistance[i][j]) << ",";

			fs18 << LeftVectors[i][j][0] << "," << LeftVectors[i][j][1] << ",";
			fs19 << RightVectors[i][j][0] << "," << RightVectors[i][j][1] << ",";
			fs20 << EVectors[i][j][0] << "," << EVectors[i][j][1] << ",";
		}
		fs18 << "\n";
		fs19 << "\n";
		fs20 << "\n";
		//	fs11 << "\n";
	}

	//
	fs18.close();
	fs19.close();
	fs20.close();
	fstream fs13("Vsplit_L2_results.txt", ios::in | ios::out | ios::app);
	fs13 << leftName << ", " << rightName << ": " << l2 << ",   skips: " << skips << endl;

	/*l2 += 1000 * (8-skips);

	std::ofstream fs10("splitGoodLeftDistances.csv");
	std::ofstream fs11("splitGoodRightDistances.csv");
	fstream fs12("split_L2_results.txt", ios::in | ios::out | ios::app);
	fs12 << leftName << ", " << rightName << ": " << l2 << ",   skips: " << skips<< endl;
	

	for (int i = 0; i < GLDistance.size(); i++) {
		for (int j = 0; j < GLDistance[i].size(); j++) {
			fs10 << std::to_string(GLDistance[i][j]) << ",";
			fs11 << std::to_string(GRDistance[i][j]) << ",";
		}
		fs10 << "\n";
		fs11 << "\n";
	}*/

	FileStorage fs6("correctLeftKey.txt", FileStorage::WRITE);
	write(fs6, "correctLeftKey", correctLeftKey);
	fs6.release();
	Mat keyLeft;
	drawKeypoints(im1, correctLeftKey, keyLeft);
	imwrite("keyLeft.jpg", keyLeft);

	//std::ofstream fs8("LeftSplitkeypoints.txt");
	//for (int i = 0; i < 8; i++) {
	//	if (LeftGoodKeypoints[i].empty()) {
	//		continue;
	//	}
	//	for (int j = 0; j < 2; j++) {
	//		fs8 << std::to_string(LeftGoodKeypoints[i][j].pt.x) << ",";
	//		fs8 << std::to_string(LeftGoodKeypoints[i][j].pt.y) << endl;
	//	}
	//}
	

	



	FileStorage fs("Descriptor.txt", FileStorage::WRITE);
	write(fs, "descritor of split 0", leftDescriptors[0]);
	fs.release();
	Mat tmp;
	drawKeypoints(splitImagesLeft[0], leftKeypoints[0], tmp);
	imwrite("keyTest.jpg", tmp);

	

	//Mat splitMatch;
	//drawMatches(splitImagesLeft[0], leftKeypoints[0], 
	//	splitImagesRight[0], rightKeypoints[0], goodSplitMatches[0], splitMatch);
	//imwrite("splitMatch.jpg", splitMatch);


	//single image detect
	/*std::vector<KeyPoint> keypoints1, keypoints2;
	Mat descriptors1, descriptors2;
	Ptr<Feature2D> sift = SIFT::create();


	sift->detectAndCompute(im1Gray, Mat(), keypoints1, descriptors1);
	std::cout << "\tDetecting right keypoints/descriptors..." << std::endl;
	sift->detectAndCompute(im2Gray, Mat(), keypoints2, descriptors2);
	descriptors1.convertTo(descriptors1, CV_32F);
	descriptors2.convertTo(descriptors2, CV_32F);
	//std::cout << "\tDrawing Keypoints..." << std::endl;
	//Mat leftKeypoints;
	//Mat rightKeypoints;
	//drawKeypoints(im1, keypoints1, leftKeypoints );
	//drawKeypoints(im2, keypoints2, rightKeypoints);
	//imwrite("left_keypoints.png", leftKeypoints);
	//imwrite("right_keypoints.png", rightKeypoints);
	




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
	//std::sort(matches.begin(), matches.end());
	// std::reverse(matches.begin(), matches.end());

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

	//std::cout << "\tCalculating Distance..." << std::endl;
	//double LeftDistance[10][10], RightDistance[10][10], GLDistance[10][10], GRDistance[10][10];
	//double Ldistance, Rdistance, l2 = 0.0;
	//for (int i = 0; i < 10; i++) {
	//	for (int j = 0; j < 10; j++) {
	//		Ldistance = sqrt(pow((LeftGoodKeypoints[j].pt.x - LeftGoodKeypoints[i].pt.x),2)
	//			+ pow((LeftGoodKeypoints[j].pt.y - LeftGoodKeypoints[i].pt.y),2));
	//		Rdistance = sqrt(pow((RightGoodKeypoints[j].pt.x - RightGoodKeypoints[i].pt.x), 2)
	//			+ pow((RightGoodKeypoints[j].pt.y - RightGoodKeypoints[i].pt.y), 2));
	//		GLDistance[i][j] = min(Ldistance, std::abs(double(im1.size().width) - Ldistance));
	//		GRDistance[i][j] = min(Rdistance, std::abs(double(im2.size().width) - Rdistance));
	//		LeftDistance[i][j] = Ldistance;
	//		RightDistance[i][j] = Rdistance;
	//		l2 += sqrt(pow(Ldistance - Rdistance, 2));
	//	}
	//}

	std::cout << "\tCalculating vectors... " << endl;
	vector<vector<vector<float>>> LeftVectors(10,vector<vector<float>>(10,vector<float>(2))), 
		RightVectors(10, vector<vector<float>>(10, vector<float>(2))), 
		EVectors(10, vector<vector<float>>(10, vector<float>(2)));
	double l2 = 0.0;
	Mat LeftVectorImage = im1.clone(), RightVectorImage = im2.clone();
	vector<Point> leftPoints, rightPoints;
	for (int i = 0; i < 10; i++) {
		for (int j = 0; j < 10; j++) {
			vector<float> tmpL;
			tmpL.push_back(LeftGoodKeypoints[j].pt.x - LeftGoodKeypoints[i].pt.x);
			tmpL.push_back(LeftGoodKeypoints[j].pt.y - LeftGoodKeypoints[i].pt.y);
			LeftVectors[i][j] = tmpL;

			vector<float> tmpR;
			tmpR.push_back(RightGoodKeypoints[j].pt.x - RightGoodKeypoints[i].pt.x);
			tmpR.push_back(RightGoodKeypoints[j].pt.y - RightGoodKeypoints[i].pt.y);
			RightVectors[i][j] = tmpR;

			vector<float> tmpE;
			tmpE.push_back(tmpR[0]-tmpL[0]);
			tmpE.push_back(tmpR[1] - tmpL[1]);
			EVectors[i][j] = tmpE;

			l2 += sqrt(pow(tmpE[0], 2) + pow(tmpE[1], 2));

			if (tmpE[0] == 0 && tmpE[1] == 0) {
				continue;
			}
			else {
			Point Left1(LeftGoodKeypoints[i].pt.x, LeftGoodKeypoints[i].pt.y), Left2(LeftGoodKeypoints[i].pt.x +tmpE[0], LeftGoodKeypoints[i].pt.y+tmpE[1]),
				Right1(RightGoodKeypoints[i].pt.x, RightGoodKeypoints[i].pt.y), Right2(RightGoodKeypoints[i].pt.x + tmpE[0], RightGoodKeypoints[i].pt.y + tmpE[1]);
			arrowedLine(LeftVectorImage, Left1, Left2, cv::COLORMAP_RAINBOW, 8);
			arrowedLine(RightVectorImage, Right1, Right2, cv::COLORMAP_RAINBOW, 8);
			}
		}
	}

	//Mat LeftVectorImage = im1.clone(), RightVectorImage = im2.clone();
	//vector<Point> leftPoints, rightPoints;

	//for (int i = 0; i < 10; i++) {
	//	for (int j = 0; j < 10; j++) {
	//		//Point LeftPoint(LeftVectors[i][j][0], LeftVectors[i][j][1]), RightPoint(RightVectors[i][j][0], RightVectors[i][j][1]);

	//		if (i == 9 && j == 9) {
	//			continue;
	//		}else if (j == 9) {
	//			Point Left1(LeftVectors[i][j][0], LeftVectors[i][j][1]), Left2(LeftVectors[i+1][0][0], LeftVectors[i+1][0][1]),
	//				Right1(RightVectors[i][j][0], RightVectors[i][j][1]), Right2(RightVectors[i + 1][0][0], RightVectors[i + 1][0][1]);
	//			arrowedLine(LeftVectorImage, Left1, Left2, cv::COLORMAP_RAINBOW,5);
	//			arrowedLine(RightVectorImage, Right1, Right2, cv::COLORMAP_RAINBOW, 5);
	//		}
	//		else {
	//		Point Left1(LeftVectors[i][j][0], LeftVectors[i][j][1]), Left2(LeftVectors[i][j+1][0], LeftVectors[i][j+1][1]),
	//			Right1(RightVectors[i][j][0], RightVectors[i][j][1]), Right2(RightVectors[i][j+1][0], RightVectors[i][j+1][1]);

	//		arrowedLine(LeftVectorImage,Left1,Left2, cv::COLORMAP_RAINBOW, 5);
	//		arrowedLine(RightVectorImage, Right1, Right2, cv::COLORMAP_RAINBOW, 5);
	//		}
	//	}
	//}

	
	imwrite("LeftVectorDraw.jpg", LeftVectorImage);
	imwrite("RightVectorDraw.jpg", RightVectorImage);

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
	fstream fs12("Vec_L2_results.txt", ios::in | ios::out | ios::app);
	fs12 << leftName << ", " << rightName << ": " << l2 << endl;
	
	for (int i = 0; i < 10; i++) {
		for (int j = 0; j < 10; j++) {
	//		fs8 << std::to_string(LeftDistance[i][j]) <<",";
	//		fs9 << std::to_string(RightDistance[i][j]) <<",";
	//		fs10 << std::to_string(GLDistance[i][j]) << ",";
	//		fs11 << std::to_string(GRDistance[i][j]) << ",";
		
			fs8 << LeftVectors[i][j][0] << "," << LeftVectors[i][j][1] << ",";
			fs9 << RightVectors[i][j][0] << "," << RightVectors[i][j][1] << ",";
			fs10 << EVectors[i][j][0] << "," << EVectors[i][j][1] << ",";
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
	drawMatches(LeftVectorImage, keypoints1, RightVectorImage, keypoints2, good_matches, imMatches);
	drawMatches(LeftVectorImage, keypoints1, RightVectorImage, keypoints2, TopTenMatches, TenMatches);
	// CHANGED
	 imwrite("matches.jpg", imMatches);
	 imwrite("top_ten_matches.jpg", TenMatches);
	 */
	 


	 
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