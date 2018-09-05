// CudaLab1.cpp : définit le point d'entrée pour l'application console.
//
#include "stdafx.h"
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <opencv2/core/core.hpp> 
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/types_c.h> 
#include <opencv2/imgproc/imgproc.hpp> 

const int PIXEL_INCREMENTATION = 100;

using namespace cv;

extern "C" void ParallelBlackAndWhite(uchar *pMatA, int pixelIncrementation, uchar *pMatR, dim3 matrixDimension);

Mat SerialBlackAndWhite(Mat imgToConvert) {
	int rows = imgToConvert.rows;
	int cols = imgToConvert.cols;
	Mat gray(rows, cols, CV_8UC1);
	for (int r = 0; r<rows; r++) {
		for (int c = 0; c<cols; c++) {
			double gray_val = (int)imgToConvert.at<uchar>(r, c) + PIXEL_INCREMENTATION;
			gray.at<uchar>(r, c) = (uchar)gray_val;
		}
	}
	return gray;
}

int main()
{
	Mat originalframe = imread("lena.png");

	// Serial conversion (CPU)
	Mat serialFrameToConvert = imread("lena.png", 0);
	Mat serialConvertedFrame = SerialBlackAndWhite(serialFrameToConvert);

	// Parallel conversion (GPU)
	Mat inputParallelConvertedFrame = imread("lena.png", 0);
	Mat outputParallelConvertedFrame = imread("lena.png", 0);
	ParallelBlackAndWhite(inputParallelConvertedFrame.data,
		PIXEL_INCREMENTATION,
		outputParallelConvertedFrame.data, 
		dim3(inputParallelConvertedFrame.rows, inputParallelConvertedFrame.cols));


	imshow("original frame", originalframe);
	imshow("serial converted frame", serialConvertedFrame);
	imshow("parallel converted frame", outputParallelConvertedFrame);

	waitKey(0);
	return 0;
}
