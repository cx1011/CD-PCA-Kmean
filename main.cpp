#include <iostream>
#include <fstream>
#include <cassert>
#include <string>
#include <sstream>
#include "cv.h"
#include <highgui\highgui.hpp>
#include "math.h"
using namespace std;
using namespace cv;

int main() {
	Mat img1 = imread("D:/data/cv/CD/im3.png",0);
	Mat img2 = imread("D:/data/cv/CD/im4.png",0);
	vector<double> v1;
	vector<vector<double>> dif_vec;
	double minv = 0.0, maxv = 0.0, sum1 = 0, sum2 = 0;
	int sumpix = img1.cols * img1.rows;
	int count = 0;
	bool flag = true;
	Mat dif, dif_pad;
	absdiff(img1, img2, dif);
	copyMakeBorder(dif, dif_pad, 2, 2, 2, 2, BORDER_CONSTANT, 0);

	for (int i = 2; i < dif.rows+2; i++)
		for (int j = 2; j < dif.cols+2; j++)
		{	
			for (int m = i - 2; m <= i + 2; m++) {
				for (int n = j - 2; n <= j + 2; n++) {
					v1.push_back((double)dif_pad.at<uchar>(m, n));
				}
			}
			dif_vec.push_back(v1);
			v1.clear();
		}

	Mat dif_vecmat = Mat(dif.cols * dif.rows, 25, CV_32FC1);

	for (int i = 0; i < dif_vec.size(); i++) {
		for (int j = 0; j < 25; j++) {
			dif_vecmat.at<float>(i, j) = dif_vec[i][j];

		}
	}

	PCA pca(dif_vecmat, Mat(), CV_PCA_DATA_AS_ROW,5);

	Mat dif_vec5 = pca.project(dif_vecmat);

	Mat res = Mat(sumpix, 1, CV_32FC1);

	kmeans(dif_vec5, 2, res, cvTermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 10, 1.0),30,0);

	Mat res1 = res.reshape(0, img1.rows);
	double* minp = &minv;
	double* maxp = &maxv;
	minMaxIdx(res1, minp, maxp);

	for (int i = 0; i < dif.rows; i++) {
		for (int j = 0; j < dif.cols; j++) {
			if ((float)res1.at<float>(i, j) != 0) {
				sum1 += (double)dif.at<uchar>(i, j);
				count++;
			}
			else {
				sum2 += (double)dif.at<uchar>(i, j);
			}
		}
	}

	if (sum1 / count > sum2 / (dif.rows * dif.cols - count))
		flag = true;
	else
		flag = false;

	for (int i = 0; i < dif.rows; i++) {
		for (int j = 0; j < dif.cols; j++) {
			if ((float)res1.at<float>(i, j) != 0) {
				if(flag)
					res1.at<float>(i, j) = 255;
				else
					res1.at<float>(i, j) = 0;
			}
			else {
				if(flag)
					res1.at<float>(i, j) = 0;
				else
					res1.at<float>(i, j) = 255;
			}
		}
	}


	imwrite("D:/data/cv/CD/res.png", res1);
	cvNamedWindow("Í¼Ïñ1-SIFTÌØÕ÷", 1);
	imshow("Í¼Ïñ1-SIFTÌØÕ÷", res1);
	cvWaitKey(0);



	system("pause");

	return 0;
}