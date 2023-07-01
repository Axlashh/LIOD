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
    pt3d(cv::Mat mat, int fx = 100, int fy = 100);
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
    void depth2PointCloud(cv::Mat mat, int fx = 100, int fy = 100);

    //д��������Ϣ
    void writePointCloud(const char* path, int mod = 0);

    //�����Ƿ�Ϊ��
    bool is_empty();

private:
    //���ƽṹ��
    std::vector<vec_3d>* pc;
};

//  init_seq:   the initial seqence
//  seq_num:    the number of seqences to be detected
//  pic_num:    the number of pictures to be detected in every seqence
//  det_seq:    if not empty, the programe will detect the seqences in this vector
//  only_move:  whether to skip the still seqence
//  show_pic:   whether to show the pictures
//  fx, fy:     the conversion factor of point cloud
int LIOD(std::string input_path, std::string output_path, double iou_thresh,
    int init_seq, int seq_num, int pic_num = 300, std::vector<int> det_seq = {}, bool only_move = true, bool show_pic = false,
    int fx = 100, int fy = 100);

YOLO_RECT COCO2YOLO(cv::Rect coco_rect, int width, int height);

cv::Mat get_mat_fromfile(std::string fullfilename, int rows, int cols);

// ��ȡ�ļ��еı߽����Ϣ
std::vector<Box> read_boxes_from_file(std::string filepath);

// ���������߽��� IoU ֵ
double cal_iou(Box box1, Box box2);

// ���� IoU ��ֵ�µ��������
int count_tp_by_iou_thresh(double iou_thresh, std::string gt_filepath, std::string pred_filepath);

//ɸѡbb��,�����汾
void fliterBB_BL(std::vector<cv::Rect> &BBVector, cv::Mat depth, int group,
    std::string point_cloud_path, int fx = 100, int fy = 100);

//��ȡcalib
TY_CAMERA_CALIB_INFO* read_calib(std::string path);

//���ļ���һ��д����
void writeFirstLine(const char* path, std::string content);

//͵��
static void writePointCloud(const cv::Point3f* pnts, const cv::Vec3b* color, size_t n, const char* file, int format);
static void writePC_XYZ(const cv::Point3f* pnts, const cv::Vec3b* color, size_t n, FILE* fp);
int WriteData(std::string fileName, cv::Mat& matData);

#endif
