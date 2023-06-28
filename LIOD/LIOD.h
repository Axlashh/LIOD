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

//������ŵ��Ƶ���
class pt3d {
public:    
    //���캯��
    pt3d();
    pt3d(cv::Mat mat);
    pt3d(const pt3d& other);
    ~pt3d();

    //=��������أ�ɵ��
    pt3d& operator=(const pt3d& other);

    //���ص���ͼ�е������
    int size();

    //����һ��������
    std::vector<vec_3d>::iterator iterator();

    //���ض�Ӧλ�õĵ�
    vec_3d at(int position);

    //����ƽ��x, y, ���
    double avex();
    double avey();
    double aved();

    //��ͼ��ת��Ϊ����ͼ
    void depth2PointCloud(cv::Mat mat);

    //д��������Ϣ
    void writePointCloud(const char* path, int mod = 0);

    //�����Ƿ�Ϊ��
    bool is_empty();

private:
    //���ƽṹ��
    std::vector<vec_3d>* pc;
};


YOLO_RECT COCO2YOLO(cv::Rect coco_rect, int width, int height);

cv::Mat get_mat_fromfile(std::string fullfilename, int rows, int cols);

// ��ȡ�ļ��еı߽����Ϣ
std::vector<Box> read_boxes_from_file(std::string filepath);

// ���������߽��� IoU ֵ
double cal_iou(Box box1, Box box2);

// ���� IoU ��ֵ�µ��������
int count_tp_by_iou_thresh(double iou_thresh, std::string gt_filepath, std::string pred_filepath);

//ɸѡbb��,�����汾
void fliterBB_BL(std::vector<cv::Rect> &BBVector, cv::Mat depth);

//��ȡcalib
TY_CAMERA_CALIB_INFO* read_calib(std::string path);

//���ļ���һ��д����
void writeFirstLine(const char* path, std::string content);

//͵��
static void writePointCloud(const cv::Point3f* pnts, const cv::Vec3b* color, size_t n, const char* file, int format);
static void writePC_XYZ(const cv::Point3f* pnts, const cv::Vec3b* color, size_t n, FILE* fp);
int WriteData(std::string fileName, cv::Mat& matData);

#endif
