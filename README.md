# Laboratory 1 - Industrial Intelligent System

> 🖼️ Laboratory 1 in Intelligent Industrial System at Cégep Lévis-Lauzon. Learning Cuda and OpenCV by creating a sobel filter. 💻

## Part 1 :
> Creating a program in cuda and C++ to apply a constant increment to any image, with OpenCV and Cuda.

#### Code example :
> This is the general function which will add the constant to each pixels.
```c++
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
```

# Sobel Edge Detector

A Sobel Edge Detection Filter written in OpenCV, Cuda and C++. Made with no external library

#### Calculating vertical gradient :
```c++

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

```

#### Calculating horizontal gradient :
```c++
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
```

#### And applying them to the actual image
```c++
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
```

## Example

Here's what this program does:

![original](https://raw.githubusercontent.com/bamcis-io/SobelFilter/master/SobelFilter/images/lena.bmp)
![filtered](https://raw.githubusercontent.com/bamcis-io/SobelFilter/master/SobelFilter/images/sobel_lena.bmp)

![original](https://raw.githubusercontent.com/bamcis-io/SobelFilter/master/SobelFilter/images/valve.png)
![filtered](https://raw.githubusercontent.com/bamcis-io/SobelFilter/master/SobelFilter/images/sobel_valve.bmp)
