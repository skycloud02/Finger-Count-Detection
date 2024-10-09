#include <iostream>
#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/video/background_segm.hpp>
#include <string>
#include <cmath>
#include <algorithm>

using namespace std;
using namespace cv;

Mat cameraFeed;
Mat cameraMask;
Mat flippedImage;
Mat transformROI;
Mat grayImage;
Mat cannyImage;
int thresh = 140;
int maxVal = 255;
int type = 1;
int deger = 8;

const string trackbarWindow = "Trackbars";
const int THRESH = 250;
const int MAX_VAL = 255;
const int TYPE = 4;
const int EDGES = 100;

void gaussianFilterImage(const Mat& src, Mat& dst) {
	int w = 3;
	int height = src.rows;
	int width = src.cols;

	// Prepare the output matrix with the same size and type as src
	dst.create(height, width, CV_8UC1);

	float sigma = w / 0.6f;
	float matrix[3][3]; // Fixed-size array for simplicity
	float sum = 0.0;

	// Calculate the Gaussian kernel
	for (int k = 0; k < w; k++) {
		for (int l = 0; l < w; l++) {
			float x = k - w / 2;
			float y = l - w / 2;
			float exponent = -(x * x + y * y) / (2 * sigma * sigma);
			matrix[k][l] = exp(exponent) / (2 * CV_PI * sigma * sigma);
			sum += matrix[k][l];
		}
	}

	// Normalize the kernel
	for (int k = 0; k < w; k++) {
		for (int l = 0; l < w; l++) {
			matrix[k][l] /= sum;
		}
	}

	// Apply the Gaussian filter
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			float weightedSum = 0.0;
			for (int m = -w / 2; m <= w / 2; m++) {
				for (int n = -w / 2; n <= w / 2; n++) {
					int x = std::max(0, std::min(i + m, height - 1));
					int y = std::max(0, std::min(j + n, width - 1));
					weightedSum += matrix[m + w / 2][n + w / 2] * src.at<uchar>(x, y);
				}
			}
			dst.at<uchar>(i, j) = static_cast<uchar>(weightedSum);
		}
	}
}

void track(int, void*) {
	int count = 0;
	char a[40];
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;

	//GaussianBlur(cameraMask, cameraMask, Size(27, 27), 3.5, 3.5);
	gaussianFilterImage(cameraMask, cameraMask);
	threshold(cameraMask, cameraMask, thresh, maxVal, type);
	Canny(cameraMask, cannyImage, deger, deger * 2, 3);
	findContours(cameraMask, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));

	Mat cannyTransform = Mat::zeros(cannyImage.size(), CV_8UC3);
	if (contours.size() > 0) {
		size_t indexOfBiggestContour = -1;
		size_t sizeOfBiggestContour = 0;

		for (size_t i = 0; i < contours.size(); i++) {
			if (contours[i].size() > sizeOfBiggestContour) {
				sizeOfBiggestContour = contours[i].size();
				indexOfBiggestContour = i;
			}
		}

		vector<vector<int>> hull(contours.size());
		vector<vector<Point>> hullPoint(contours.size());
		vector<vector<Vec4i>> defects(contours.size());
		vector<vector<Point>> defectPoint(contours.size());
		vector<vector<Point>> contours_poly(contours.size());

		Point2f rect_point[4];
		vector<RotatedRect> minRect(contours.size());
		vector<Rect> boundRect(contours.size());

		for (size_t i = 0; i < contours.size(); i++) {
			if (contourArea(contours[i]) > 5000) {
				convexHull(contours[i], hull[i], true);
				convexityDefects(contours[i], hull[i], defects[i]);

				if (indexOfBiggestContour == i) {
					minRect[i] = minAreaRect(contours[i]);

					for (size_t k = 0; k < hull[i].size(); k++) {
						int ind = hull[i][k];
						hullPoint[i].push_back(contours[i][ind]);
					}

					count = 0;

					for (size_t k = 0; k < defects[i].size(); k++) {
						if (defects[i][k][3] > 13 * 256) {
							int p_start = defects[i][k][0];
							int p_end = defects[i][k][1];
							int p_far = defects[i][k][2];
							defectPoint[i].push_back(contours[i][p_far]);
							circle(grayImage, contours[i][p_end], 3, Scalar(0, 255, 0), 2);
							count++;
						}
					}

					if (count == 1) strcpy_s(a, "1");
					else if (count == 2) strcpy_s(a, "2");
					else if (count == 3) strcpy_s(a, "3");
					else if (count == 4) strcpy_s(a, "4");
					else if (count == 5 || count == 6) strcpy_s(a, "5");
					else strcpy_s(a, "No number");

					putText(flippedImage, a, Point(75, 450), FONT_HERSHEY_SIMPLEX, 3, Scalar(0, 255, 0), 3, 8, false);
					drawContours(cannyTransform, contours, i, Scalar(255, 255, 0), 2, 8, vector<Vec4i>(), 0, Point());
					drawContours(cannyTransform, hullPoint, i, Scalar(255, 255, 0), 1, 8, vector<Vec4i>(), 0, Point());
					drawContours(grayImage, hullPoint, i, Scalar(0, 0, 255), 2, 8, vector<Vec4i>(), 0, Point());

					approxPolyDP(contours[i], contours_poly[i], 3, false);
					boundRect[i] = boundingRect(contours_poly[i]);
					rectangle(grayImage, boundRect[i].tl(), boundRect[i].br(), Scalar(255, 0, 0), 2, 8, 0);
					minRect[i].points(rect_point);

					for (size_t k = 0; k < 4; k++) {
						line(grayImage, rect_point[k], rect_point[(k + 1) % 4], Scalar(0, 255, 0), 2, 8);
					}
				}
			}
		}

	}
	imshow("Final Image", cannyTransform);
}

void on_trackbar(int, void*) {}

void createTrackbars() {
	namedWindow(trackbarWindow, 0);
	char TrackbarName[50];
	sprintf_s(TrackbarName, "Threshold", THRESH);
	sprintf_s(TrackbarName, "Maximum", MAX_VAL);
	sprintf_s(TrackbarName, "Threshold Type", TYPE);
	sprintf_s(TrackbarName, "Edges", EDGES);

	createTrackbar("Threshold", trackbarWindow, &thresh, THRESH, on_trackbar);
	createTrackbar("Maximum", trackbarWindow, &maxVal, MAX_VAL, on_trackbar);
	createTrackbar("Threshold Type", trackbarWindow, &type, TYPE, on_trackbar);
	createTrackbar("Edges", trackbarWindow, &deger, EDGES, on_trackbar);
}

Mat colorToGrayScaleInDst(Mat& src) 
{
	int height = src.rows;
	int width = src.cols;
	Mat dst = Mat(height, width, CV_8UC1);
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			Vec3b v3 = src.at<Vec3b>(i, j);
			uchar b = v3[0];
			uchar g = v3[1];
			uchar r = v3[2];
			dst.at<uchar>(i, j) = (r + g + b) / 3;
		}
	}

	return dst;
}

int main()
{
	VideoCapture capture;
	capture.open(0);
	
	Ptr<BackgroundSubtractor> pMOG2;
	pMOG2 = createBackgroundSubtractorMOG2();

	cv::Rect imageROI(288, 12, 288, 288);
	Mat element = getStructuringElement(MORPH_RECT, Size(3, 3), Point(1, 1));
	Mat frame;
	Mat resizeFrame;

	createTrackbars();

	if (!capture.isOpened()) {
		cerr << "ERROR: COuld not open the camera" << endl;
		return -1;
	}
	while (true) {
		capture >> cameraFeed;
		if (cameraFeed.empty()) {
			cerr << "ERROR: Could not capture frame";
			break;
		}

		flip(cameraFeed, flippedImage, 1);
		rectangle(flippedImage, imageROI, Scalar(0, 0, 255));
		transformROI = flippedImage(imageROI);
		//cvtColor(transformROI, grayImage, COLOR_RGB2GRAY);
		grayImage = colorToGrayScaleInDst(transformROI);

		//GaussianBlur(grayImage, grayImage, Size(23, 23), 0);
		gaussianFilterImage(grayImage, grayImage);

		imshow("gaussian blur", grayImage);
		pMOG2->apply(transformROI, cameraMask);
		track(0, 0);

		imshow("Original image", cameraFeed);
		imshow("Flipped image", flippedImage);
		imshow("Gray image", grayImage);
		waitKey(30);
	}

	return 0;
}
