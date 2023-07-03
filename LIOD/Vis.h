#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

void cal_mdepth(cv::Rect bb, cv::Mat depth_mat, cv::Mat image_mat);

cv::Mat  time_targeted(int bbnum, double elapsed_secs, cv::Mat image_mat, bool hits = true);