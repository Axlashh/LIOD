#pragma once
#ifndef __LIOD__
#define __LIOD__

#include <iostream>
#include <fstream>
#include <vector>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include "Detect.h"
#include "ctime"
#include <sstream>
#include <string>
#include <cmath>
#include <TYCoordinateMapper.h>
#include <algorithm>
#include <Shlwapi.h>
#pragma comment(lib,"Shlwapi.lib")


class Box {
public:
    int class_id;
    double x_center, y_center, width, height;
};

struct YOLO_RECT {
    float x_center;
    float y_center;
    float width;
    float height;
};

typedef struct vec_3d {
    float x;
    float y;
    float z;
}vec_3d;

//用来存放点云的类
class pt3d {
public:    
    //构造函数
    pt3d();
    pt3d(cv::Mat mat);
    pt3d(const pt3d& other);
    ~pt3d();

    //=运算符重载，傻逼
    pt3d& operator=(const pt3d& other);

    //返回点云图中点的数量
    int size();

    //返回一个迭代器
    std::vector<vec_3d>::iterator iterator();

    //返回对应位置的点
    vec_3d at(int position);

    //返回平均x, y, 深度
    double avex();
    double avey();
    double aved();

    //将图像转换为点云图
    void depth2PointCloud(cv::Mat mat);

    //写出点云信息
    void writePointCloud(const char* path, int mod = 0);

    //返回是否为空
    bool is_empty();

private:
    //点云结构体
    std::vector<vec_3d>* pc;
};


YOLO_RECT COCO2YOLO(cv::Rect coco_rect, int width, int height);

cv::Mat get_mat_fromfile(std::string fullfilename, int rows, int cols);

// 读取文件中的边界框信息
std::vector<Box> read_boxes_from_file(std::string filepath);

// 计算两个边界框的 IoU 值
double cal_iou(Box box1, Box box2);

// 计算 IoU 阈值下的正检个数
int count_tp_by_iou_thresh(double iou_thresh, std::string gt_filepath, std::string pred_filepath);

//筛选bb框,基础版本
void fliterBB_BL(std::vector<cv::Rect> &BBVector, cv::Mat depth);

//读取calib
TY_CAMERA_CALIB_INFO* read_calib(std::string path);

//在文件第一行写东西
void writeFirstLine(const char* path, std::string content);

//偷的
static void writePointCloud(const cv::Point3f* pnts, const cv::Vec3b* color, size_t n, const char* file, int format);
static void writePC_XYZ(const cv::Point3f* pnts, const cv::Vec3b* color, size_t n, FILE* fp);
int WriteData(std::string fileName, cv::Mat& matData);

#endif
