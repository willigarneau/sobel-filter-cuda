// CudaLab1.cpp : définit le point d'entrée pour l'application console.
//
#include "stdafx.h"
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <opencv2/core/core.hpp> 
#include <opencv2/highgui/highgui.hpp>

const int PIXEL_INCREMENTATION = 100;

// méthode pour connaôtre la valeur minimale et maximale d'uen image : cv::minmaxloc

using namespace cv;

extern "C" void ParallelBlackAndWhite(uchar *inputMatrixPointer, int pixelIncrementation, uchar *outputMatrixPointer, dim3 matrixDimension);
extern "C" void ParallelSobelFilter(uchar *inputMatrixPointer, uchar *outputMatrixPointer, dim3 matrixDimension);

int maximumGradient = 0;

int Gx[3][3] = { // valeur approximative de la dérivée horizontale
	{-1, 0, 1},
	{-2, 0, 2},
	{-1, 0, 1}, 
};

int Gy[3][3] = { // valeur approximative de la dérivée verticale
	{-1, -2, -1},
	{0, 0, 0},
	{1, 2, 1}, 
};

int xGradient(Mat frame, Point position) {
	int tGradient = 0;
	for (int x = 0; x < 3; x++) {
		for (int y = 0; y < 3; y++) {
			int rows = position.y + y;
			int cols = position.x + x;
			int currentPixelValue = (int)frame.at<uchar>(rows, cols);
			int currentGradient = currentPixelValue * Gx[x][y];
			tGradient += currentGradient;
			currentGradient > maximumGradient ? maximumGradient = currentGradient: maximumGradient;
		}
	}
	return tGradient;
}

int yGradient(Mat frame, Point position) {
	int tGradient = 0;
	for (int x = 0; x < 3; x++) {
		for (int y = 0; y < 3; y++) {
			int rows = position.y + y;
			int cols = position.x + x;
			int currentPixelValue = (int)frame.at<uchar>(rows, cols);
			int currentGradient = currentPixelValue * Gy[x][y];
			tGradient += currentGradient;
			currentGradient > maximumGradient ? maximumGradient = currentGradient : maximumGradient;
		}
	}
	return tGradient;
}

Mat SerialSobel(Mat imgToConvert) {
	for (int rows = 0; rows < imgToConvert.rows - 2; rows++) {
		for (int cols = 0; cols < imgToConvert.cols - 2; cols++) {
			int currentPixel = imgToConvert.at<uchar>(rows, cols);
			Point currentPosition = Point(cols, rows);
			int gradientX = xGradient(imgToConvert, currentPosition);
			int gradientY = yGradient(imgToConvert, currentPosition);
			int approxGradient = (abs(gradientX) + abs(gradientY) * 255) / maximumGradient;
			imgToConvert.at<uchar>(rows, cols) = approxGradient;

		}
	}
	return imgToConvert;
}


Mat SerialBlackAndWhite(Mat imgToConvert) {
	int rows = imgToConvert.rows;
	int cols = imgToConvert.cols;
	Mat gray(rows, cols, CV_8UC1);
	for (int r = 0; r < rows; r++) {
		for (int c = 0; c < cols; c++) {
			double gray_val = (int)imgToConvert.at<uchar>(r, c) + PIXEL_INCREMENTATION;
			gray.at<uchar>(r, c) = (uchar)gray_val;
		}
	}
	return gray;
}


int main()
{
	Mat originalframe = imread("lena.png");

	// Serial conversion to black and white (CPU)
	Mat serialFrameToConvert = imread("lena.png", 0);
	Mat serialConvertedFrame = SerialBlackAndWhite(serialFrameToConvert);

	// Serial conversion to sobel filter (CPU)
	Mat serialFrameToFilter = imread("lena.png", 0);
	Mat serialFilteredFrame = SerialSobel(serialFrameToFilter);

	// Parallel conversion (GPU)
	Mat inputParallelConvertedFrame = imread("lena.png", 0);
	Mat outputParallelConvertedFrame = imread("lena.png", 0);
	ParallelBlackAndWhite(inputParallelConvertedFrame.data,
		PIXEL_INCREMENTATION,
		outputParallelConvertedFrame.data, 
		dim3(inputParallelConvertedFrame.rows, inputParallelConvertedFrame.cols));


	imshow("original frame", originalframe);
	imshow("serial sobel filtered frame", serialFilteredFrame);
	imshow("serial converted frame", serialConvertedFrame);
	imshow("parallel converted frame", outputParallelConvertedFrame);

	waitKey(0);
	return 0;
}
