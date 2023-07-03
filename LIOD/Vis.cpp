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
    //求深度平均    
    int startRow = bb.y;  // 指定计算平均值开始的行号
    int endRow = bb.y + bb.height;  // 指定计算平均值结束的行号
    int startCol = bb.x;  // 指定计算平均值开始的列号
    int endCol = bb.x + bb.width;  // 指定计算平均值结束的列号

    double sum = 0.0;  // 指定区域内深度信息的总和
    double averageValue = 0.0;  // 指定区域内深度信息的平均值
    int dnum = 0;
    for (int i = startRow; i < endRow; ++i) {
        for (int j = startCol; j < endCol; ++j) {
            sum += depth_mat.at<int>(i, j);  // 计算指定区域内深度信息的总和
            if (depth_mat.at<int>(i, j) != 0) {
                dnum++;
            }

        }
    }
    if (dnum == 0) averageValue = 0;
    else averageValue = sum / dnum;
   

     //定义要添加的文字内容和位置
    std::ostringstream ss1, ss2;
    ss1 << "A:" << (int)bb.area();
    ss2 << "D:" << (int)averageValue;
    string text1 = ss1.str();
    string text2 = ss2.str();
    Point pos1(bb.x, bb.y - 5);
    Point pos2(bb.x, bb.y - 18);

     //定义字体类型和大小
    int fontface = FONT_HERSHEY_SIMPLEX;
    double fontsize = 0.4;

     //定义文字的颜色和厚度
    Scalar t_color(255, 192, 203);
    int t_thickness = 0.6;

    //在图片上添加文字
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

    // 定义文字的颜色和厚度
    Scalar t_color(255, 192, 203);
    int t_thickness = 2;

    putText(image_mat, text3, pos3, fontface, fontsize, t_color, t_thickness);
    return image_mat;

}