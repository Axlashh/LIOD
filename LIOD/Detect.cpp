#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>
#include "Detect.h"

bool cmp(cv::Rect rect1, cv::Rect rect2)
{
    return rect1.area() > rect2.area();
}

double calcIOU(cv::Rect rect1, cv::Rect rect2) {
    cv::Rect intersection = rect1 & rect2;
    double IOU = (double)intersection.area() / (rect1.area() + rect2.area() - intersection.area());
    return IOU;
}

std::vector<cv::Rect> last_frame;

std::vector<cv::Rect> Detect(cv::Mat image_mat, cv::Mat depth_mat, int image_width, int image_height, bool useDepth, bool interFrame)
{
    // 设置不输出日志
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);

    std::vector<cv::Rect> BBVect;
    std::vector<cv::Rect> BBVect2;
    std::vector<cv::Rect> BBVect3;
    
    // 双边滤波，转换为灰度图像
    cv::Mat gray;
    bilateralFilter(image_mat, gray, 13, 300, 200);
    cvtColor(gray, gray, cv::COLOR_BGR2GRAY);

    // 对比度拉伸
    //normalize(gray, gray, 0, 255, cv::NORM_MINMAX);

    // 边缘检测
    Canny(gray, gray, 100, 200);

    // 查找轮廓
    std::vector<std::vector<cv::Point>> contours;
    findContours(gray, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // 遍历每个轮廓
    for (size_t i = 0; i < contours.size(); i++)
    {
        // 找到外接矩形
        cv::Rect rect = boundingRect(contours[i]);

        // 检查矩形面积是否大于一定值
        if (rect.area() > 50 && ((double)rect.width / rect.height) > 0.3333 && ((double)rect.width / rect.height) < 3)
        {
            // 画出矩形边框
            //rectangle(image, rect, cv::Scalar(0, 0, 255));
            BBVect.push_back(rect);
        }
    }

    // 消除重框
    sort(BBVect.begin(), BBVect.end(), cmp);
    for (std::vector<cv::Rect>::iterator i = BBVect.begin(); i != BBVect.end(); i++)
    {
        if (i == BBVect.begin())
            BBVect2.push_back(*i);
        else
        {
            int flag = 0;
            for (std::vector<cv::Rect>::iterator j = BBVect.begin(); j != i; j++)
            {
                if (calcIOU(*i, *j) > 0)
                {
                    flag = 1;
                    break;
                }
            }
            if (flag == 0)
                BBVect2.push_back(*i);
        }
    }

    // 使用深度信息
    if(!useDepth)
        return BBVect2;
    else
    {
        int sum;
        double mean;
        for (std::vector<cv::Rect>::iterator i = BBVect2.begin(); i != BBVect2.end(); i++)
        {
            sum = 0;
            for (int row = i->y; row < i->y + i->height; row++)
                for (int col = i->x; col < i->x + i->width; col++)
                    sum += depth_mat.at<int>(row, col);
            mean = (double)sum / (i->height * i->width);// 深度平均值
            if (mean > 0 && mean < 500)
                BBVect3.push_back(*i);
            else if (interFrame)// 使用帧间信息
            {
                for (std::vector<cv::Rect>::iterator j = last_frame.begin(); j != last_frame.end(); j++)
                {
                    if (calcIOU(*i, *j) > 0.1)
                    {
                        BBVect3.push_back(*i);
                        break;
                    }
                }
            }
        }
        if (interFrame)
            last_frame = BBVect3;
        return BBVect3;
    }
}