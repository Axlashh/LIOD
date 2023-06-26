#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

std::vector<cv::Rect> Detect(cv::Mat image_mat, cv::Mat depth_mat, int image_width, int image_height, bool useDepth, bool interFrame);