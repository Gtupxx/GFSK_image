/*
 * @Author: 柚岩龙蜥
 * @Date: 2024-07-17 10:07:35
 * @LastEditors: 柚岩龙蜥
 * @LastEditTime: 2024-07-17 11:12:30
 * @FilePath: \GFSK\GFSK.cpp
 * @Description:  
 * 
 */
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>

void applyHighPassFilter(const cv::Mat& src, cv::Mat& dst, double D0) {
    int rows = src.rows;
    int cols = src.cols;
    cv::Mat padded;
    int m = cv::getOptimalDFTSize(rows);
    int n = cv::getOptimalDFTSize(cols);
    cv::copyMakeBorder(src, padded, 0, m - rows, 0, n - cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));

    cv::Mat planes[] = { cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F) };
    cv::Mat complexI;
    cv::merge(planes, 2, complexI);
    
    cv::dft(complexI, complexI);

    cv::Mat magI;
    cv::magnitude(planes[0], planes[1], magI);

    cv::Mat filter = cv::Mat::zeros(complexI.size(), CV_32F);
    for (int u = 0; u < complexI.rows; ++u) {
        for (int v = 0; v < complexI.cols; ++v) {
            double D_square = pow((u - m / 2), 2) + pow((v - n / 2), 2);
            filter.at<float>(u, v) = 1 - exp(-D_square / (2 * D0 * D0));
        }
    }

    cv::Mat planesH[] = { filter, filter };
    cv::Mat complexFilter;
    cv::merge(planesH, 2, complexFilter);

    cv::mulSpectrums(complexI, complexFilter, complexI, 0);

    cv::idft(complexI, complexI);
    cv::split(complexI, planes);
    cv::magnitude(planes[0], planes[1], dst);
    dst = dst(cv::Rect(0, 0, src.cols, src.rows));
    cv::normalize(dst, dst, 0, 1, cv::NORM_MINMAX);
}

int main() {
    cv::Mat A = cv::imread("Fig. 4.41(a).tif", cv::IMREAD_GRAYSCALE);
    if (A.empty()) {
        std::cerr << "Error opening image!" << std::endl;
        return -1;
    }

    A.convertTo(A, CV_32F, 1.0 / 255);
    cv::Mat F1_30, F1_60, F1_120;

    applyHighPassFilter(A, F1_30, 30);
    applyHighPassFilter(A, F1_60, 60);
    applyHighPassFilter(A, F1_120, 120);

    cv::imshow("Original Image", A);
    cv::imshow("High-Pass Filter D0=30", F1_30);
    cv::imshow("High-Pass Filter D0=60", F1_60);
    cv::imshow("High-Pass Filter D0=120", F1_120);

    cv::waitKey(0);

    return 0;
}
