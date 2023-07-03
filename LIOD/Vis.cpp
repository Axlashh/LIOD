#include <iostream>
#include <string>
#include <sstream>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <vector>
#include <filesystem>


using namespace std;
namespace fs = std::filesystem;
using namespace cv;



void cal_mdepth(cv::Rect bb, cv::Mat depth_mat,cv::Mat image_mat)
{
    //�����ƽ��    
    int startRow = bb.y;  // ָ������ƽ��ֵ��ʼ���к�
    int endRow = bb.y + bb.height;  // ָ������ƽ��ֵ�������к�
    int startCol = bb.x;  // ָ������ƽ��ֵ��ʼ���к�
    int endCol = bb.x + bb.width;  // ָ������ƽ��ֵ�������к�

    double sum = 0.0;  // ָ�������������Ϣ���ܺ�
    double averageValue = 0.0;  // ָ�������������Ϣ��ƽ��ֵ
    int dnum = 0;
    for (int i = startRow; i < endRow; ++i) {
        for (int j = startCol; j < endCol; ++j) {
            sum += depth_mat.at<int>(i, j);  // ����ָ�������������Ϣ���ܺ�
            if (depth_mat.at<int>(i, j) != 0) {
                dnum++;
            }

        }
    }
    if (dnum == 0) averageValue = 0;
    else averageValue = sum / dnum;
   

     //����Ҫ��ӵ��������ݺ�λ��
    std::ostringstream ss1, ss2;
    ss1 << "A:" << (int)bb.area();
    ss2 << "D:" << (int)averageValue;
    string text1 = ss1.str();
    string text2 = ss2.str();
    Point pos1(bb.x, bb.y - 5);
    Point pos2(bb.x, bb.y - 18);

     //�����������ͺʹ�С
    int fontface = FONT_HERSHEY_SIMPLEX;
    double fontsize = 0.4;

     //�������ֵ���ɫ�ͺ��
    Scalar t_color(255, 192, 203);
    int t_thickness = 0.6;

    //��ͼƬ���������
    putText(image_mat, text1, pos1, fontface, fontsize, t_color, t_thickness);
    putText(image_mat, text2, pos2, fontface, fontsize, t_color, t_thickness);
    return;
}



cv::Mat  time_targeted(int bbnum, double elapsed_secs,cv::Mat image_mat,bool hits)
{
  
    std::ostringstream ss3;
    if (hits) ss3 << "hits = " << bbnum << " ";
    ss3 << "time=" << elapsed_secs * 1000 << "ms";
    string text3 = ss3.str();
    Point pos3(10, 30);

    int fontface = FONT_HERSHEY_SIMPLEX;
    double fontsize = 0.8;

    // �������ֵ���ɫ�ͺ��
    Scalar t_color(255, 192, 203);
    int t_thickness = 2;

    putText(image_mat, text3, pos3, fontface, fontsize, t_color, t_thickness);
    return image_mat;

}