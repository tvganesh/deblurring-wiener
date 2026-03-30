/*
============================================================================
Name : deblurring.cpp
Author : Tinniam V Ganesh & Egli Simon
Description : Implementation of Wiener filter in OpenCV (modern C++ API, OpenCV 4.x)
============================================================================
*/

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

#define kappa 10.0
#define rad   8

// Compute DFT of an image padded to (dft_M x dft_N), display magnitude spectrum,
// and return the full complex DFT matrix.
cv::Mat computeDFT(const cv::Mat& im, int dft_M, int dft_N, const std::string& winName)
{
    // Convert to 64-bit float and build a two-channel (real + zero imaginary) matrix
    cv::Mat realPart, imagPart;
    im.convertTo(realPart, CV_64F);
    imagPart = cv::Mat::zeros(realPart.size(), CV_64F);

    std::vector<cv::Mat> planes = { realPart, imagPart };
    cv::Mat complexInput;
    cv::merge(planes, complexInput);

    // Pad with zeros to the optimal DFT size
    cv::Mat dft_A = cv::Mat::zeros(dft_M, dft_N, CV_64FC2);
    cv::Mat roi = dft_A(cv::Rect(0, 0, im.cols, im.rows));
    complexInput.copyTo(roi);

    cv::dft(dft_A, dft_A, cv::DFT_COMPLEX_OUTPUT);

    // --- Display log-magnitude spectrum ---
    std::vector<cv::Mat> dftPlanes(2);
    cv::split(dft_A, dftPlanes);

    cv::Mat mag;
    cv::pow(dftPlanes[0], 2.0, dftPlanes[0]);
    cv::pow(dftPlanes[1], 2.0, dftPlanes[1]);
    cv::add(dftPlanes[0], dftPlanes[1], mag);
    cv::pow(mag, 0.5, mag);

    cv::add(mag, cv::Scalar::all(1.0), mag);   // log(1 + magnitude)
    cv::log(mag, mag);

    double m, M;
    cv::minMaxLoc(mag, &m, &M);
    mag = (mag - m) / (M - m);

    std::string title = "DFT - " + winName;
    cv::namedWindow(title, cv::WINDOW_NORMAL);
    cv::imshow(title, mag);

    return dft_A;
}

// Compute inverse DFT, extract the real part, normalise to [0,1], and display.
void showInvDFT(const cv::Mat& /*im*/, const cv::Mat& dft_A,
                int /*dft_M*/, int /*dft_N*/, const std::string& winName)
{
    cv::Mat result;
    cv::dft(dft_A, result, cv::DFT_INVERSE | cv::DFT_SCALE | cv::DFT_COMPLEX_OUTPUT);

    std::vector<cv::Mat> planes(2);
    cv::split(result, planes);
    cv::Mat image_Re = planes[0];   // deblurred image is in the real part

    double m, M;
    cv::minMaxLoc(image_Re, &m, &M);
    image_Re = (image_Re - m) / (M - m);

    std::string title = "DFT INVERSE - " + winName;
    cv::namedWindow(title, cv::WINDOW_NORMAL);
    cv::imshow(title, image_Re);
}

int main(int argc, char** argv)
{
    // ---- Load image --------------------------------------------------------
    cv::Mat im1 = cv::imread("kutty-1.jpg", cv::IMREAD_COLOR);
    if (im1.empty()) {
        printf("Error: could not load 'kutty-1.jpg'\n");
        return -1;
    }
    cv::namedWindow("Original-color", cv::WINDOW_NORMAL);
    cv::imshow("Original-color", im1);

    cv::Mat im = cv::imread("kutty-1.jpg", cv::IMREAD_GRAYSCALE);
    cv::namedWindow("Original-gray", cv::WINDOW_NORMAL);
    cv::imshow("Original-gray", im);

    // ---- Add random noise --------------------------------------------------
    cv::Mat noise(im.rows, im.cols, CV_8UC1);
    cv::randu(noise, cv::Scalar(0), cv::Scalar(128));
    cv::add(im, noise, im);

    cv::namedWindow("Original + Noise", cv::WINDOW_NORMAL);
    cv::imshow("Original + Noise", im);

    // ---- Gaussian smoothing ------------------------------------------------
    cv::GaussianBlur(im, im, cv::Size(7, 7), 0.5, 0.5);
    cv::namedWindow("Gaussian Smooth", cv::WINDOW_NORMAL);
    cv::imshow("Gaussian Smooth", im);

    // ---- Build circular blur kernel ----------------------------------------
    float r      = rad;
    float radius = ((int)(r) * 2 + 1) / 2.0f;
    int rowLength = (int)(2 * radius);
    printf("rowLength = %d\n", rowLength);

    std::vector<float> kernelData(rowLength * rowLength, 0.0f);
    int norm = 0;
    for (int x = 0; x < rowLength; x++)
        for (int y = 0; y < rowLength; y++)
            if (std::sqrt((x - (int)radius) * (x - (int)radius) +
                          (y - (int)radius) * (y - (int)radius)) <= (int)radius)
                norm++;

    for (int y = 0; y < rowLength; y++) {
        for (int x = 0; x < rowLength; x++) {
            if (std::sqrt((x - (int)radius) * (x - (int)radius) +
                          (y - (int)radius) * (y - (int)radius)) <= (int)radius) {
                kernelData[y * rowLength + x] = 1.0f / norm;
                printf("%f ", 1.0f / norm);
            }
        }
    }
    printf("\n");

    cv::Mat k_image(rowLength, rowLength, CV_32FC1, kernelData.data());
    cv::namedWindow("blur kernel", cv::WINDOW_NORMAL);
    cv::imshow("blur kernel", k_image);

    // ---- Optimal DFT sizes -------------------------------------------------
    int dft_M1 = cv::getOptimalDFTSize(im.rows + rowLength - 1);
    int dft_N1 = cv::getOptimalDFTSize(im.cols + rowLength - 1);
    printf("dft_N1=%d, dft_M1=%d\n", dft_N1, dft_M1);

    // ---- DFT of image and kernel -------------------------------------------
    cv::Mat dft_A = computeDFT(im,      dft_M1, dft_N1, "original");
    cv::Mat dft_B = computeDFT(k_image, dft_M1, dft_N1, "kernel");

    // ---- Wiener filter in frequency domain ---------------------------------
    // Numerator:   G(u,v) * H*(u,v)   (G = DFT of blurred image, H* = conjugate of kernel DFT)
    cv::Mat dft_C(dft_M1, dft_N1, CV_64FC2);
    cv::mulSpectrums(dft_A, dft_B, dft_C, 0, /*conjB=*/true);

    std::vector<cv::Mat> planesC(2);
    cv::split(dft_C, planesC);
    cv::Mat& image_ReC = planesC[0];
    cv::Mat& image_ImC = planesC[1];

    // Denominator: |H(u,v)|^2 + kappa
    std::vector<cv::Mat> planesB(2);
    cv::split(dft_B, planesB);
    cv::Mat image_ReB, image_ImB;
    cv::pow(planesB[0], 2.0, image_ReB);
    cv::pow(planesB[1], 2.0, image_ImB);
    cv::add(image_ReB, image_ImB, image_ReB);
    cv::add(image_ReB, cv::Scalar::all(kappa), image_ReB);

    // Divide numerator by denominator
    cv::divide(image_ReC, image_ReB, image_ReC);
    cv::divide(image_ImC, image_ReB, image_ImC);

    // Merge back into complex matrix
    cv::Mat complex_ImC;
    cv::merge(planesC, complex_ImC);

    // ---- Inverse DFT → deblurred result ------------------------------------
    char str[80];
    snprintf(str, sizeof(str), "O/P Wiener - K=%6.4f rad=%d", kappa, rad);
    showInvDFT(im, complex_ImC, dft_M1, dft_N1, str);

    cv::waitKey(0);
    return 0;
}
