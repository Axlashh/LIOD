// LIOD.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

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

using namespace std;
namespace fs = std::filesystem;
using namespace cv;

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

YOLO_RECT COCO2YOLO(cv::Rect coco_rect, int width, int height);

cv::Mat get_mat_fromfile(std::string fullfilename, int rows, int cols);

// 读取文件中的边界框信息
vector<Box> read_boxes_from_file(string filepath);

// 计算两个边界框的 IoU 值
double cal_iou(Box box1, Box box2);

// 计算 IoU 阈值下的正检个数
int count_tp_by_iou_thresh(double iou_thresh, string gt_filepath, string pred_filepath);

//判断是否为正检框
bool isPostiveBB();


const char* out = "D:\\AAAAAAA\\documents\\scitri\\data\\output\\%d_%d_%d_%d_%02d\\";

int main()
{
    int seq_num = 1;
    int init_seq = 3;
    bool only_move = true;


	char tout[200];
    time_t timep;
    time(&timep);
	struct tm* t = new struct tm; 
	localtime_s(t, &timep);
    sprintf_s(tout, sizeof(tout), out, 1900 + t->tm_year,1 + t->tm_mon, t->tm_wday, t->tm_hour, t->tm_min);
    std::string output_path = tout;
    std::string input_path = "D:\\AAAAAAA\\documents\\scitri\\data\\input\\";
	if (!fs::exists(output_path)) {
		fs::create_directory(output_path);
	}

    // input info
    std::ifstream input_imageList("C:\\Users\\XJH\\Desktop\\pipe\\Data\\input\\seq06\\images\\filename_list.txt");
    std::string input_depthPath = "C:\\Users\\XJH\\Desktop\\pipe\\Data\\input\\seq06\\depth";
    std::string input_gt_bbPath = "C:\\Users\\XJH\\Desktop\\pipe\\Data\\input\\seq06\\gt_bb";
	std::string input_imagePath;
    // output info
    std::string output_imagePath = "C:\\Users\\XJH\\Desktop\\pipe\\Data\\output\\pred\\seq06\\images";
    std::string output_bboxPath = "C:\\Users\\XJH\\Desktop\\pipe\\Data\\output\\pred\\seq06\\pred_bb";
    double iou_thresh = 0.3f;
    // Open the list file.
    //if (!input_imageList) {
    //    std::cerr << "Error: could not open list file.\n";
    //    return 1;
    //}

    // Read each line of the list file and load the corresponding image file.
    std::string image_fullfilename;
    std::vector<cv::Rect> BBVector;
	while (seq_num--) {
		std::ostringstream tseq;
		tseq << "seq" << std::setfill('0') << std::setw(2) << std::to_string(init_seq);
		std::string seq = tseq.str();
		input_depthPath = input_path + seq + "\\depth\\";
		input_gt_bbPath = input_path + seq + "\\gt__bb\\";
		input_imagePath = input_path + seq + "\\images\\";
		output_imagePath = output_path + seq + "\\images\\";
		output_bboxPath = output_path + seq + "\\pred__bb\\";
		fs::create_directories(output_imagePath);
		fs::create_directories(output_bboxPath);

		std::vector<std::string> files;
		for (const auto& f : fs::directory_iterator(input_imagePath)) {
			if (f.path().extension() == ".jpg")
				files.push_back(f.path().string());
		}

		for (auto& img : files) {
			fs::path p(image_fullfilename);
			std::string fileName = p.filename().string();
			size_t dotPos = fileName.find_last_of(".");
			std::string depth_fullfilename = input_depthPath + fileName.substr(0, dotPos) + ".xls";
			std::cout << image_fullfilename << std::endl;

			// Read image from a file


			cv::Mat image_mat = cv::imread(image_fullfilename);
			if (image_mat.empty()) {
				std::cerr << "Error: could not load image file " << image_fullfilename << ".\n";
				return 1;
			}
			int width = image_mat.cols;
			int height = image_mat.rows;

			// Read depth matrix from file
			cv::Mat depth_mat = get_mat_fromfile(depth_fullfilename, height, width);

			// Procecessing
			// Detection
			clock_t start = clock();
			BBVector = Detect(image_mat, depth_mat, width, height, true, true);
			clock_t end = clock();
			double elapsed_secs = double(end - start) / CLOCKS_PER_SEC;

			// Draw the boundingbox
			cv::Scalar color(0, 0, 255);
			int thickness = 3; int bbnum = 0;
			for (auto it = BBVector.begin(); it != BBVector.end(); it++) {
				auto bb = *it;

				if (!isPostiveBB()) {
					continue;
				}

				rectangle(image_mat, bb.tl(), bb.br(), color, thickness);

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
				//averageValue = sum / ((endRow - startRow + 1) * (endCol - startCol + 1));  // 除以总个数，得到平均值

				// 定义要添加的文字内容和位置
				std::ostringstream ss1, ss2;
				ss1 << "A:" << (int)bb.area();
				ss2 << "D:" << (int)averageValue;
				String text1 = ss1.str();
				String text2 = ss2.str();
				Point pos1(bb.x, bb.y - 5);
				Point pos2(bb.x, bb.y - 18);

				// 定义字体类型和大小
				int fontface = FONT_HERSHEY_SIMPLEX;
				double fontsize = 0.4;

				// 定义文字的颜色和厚度
				Scalar t_color(255, 192, 203);
				int t_thickness = 0.6;

				// 在图片上添加文字
				cv::putText(image_mat, text1, pos1, fontface, fontsize, t_color, t_thickness);
				cv::putText(image_mat, text2, pos2, fontface, fontsize, t_color, t_thickness);

			}
			// Write image and bbox to files

			std::string output_bboxFileName = output_bboxPath + fileName.substr(0, dotPos) + ".txt";
			std::ofstream bboxFile(output_bboxFileName);
			for (auto& bb : BBVector) {
				YOLO_RECT yolo_rect = COCO2YOLO(bb, width, height);
				bboxFile << 0 << ' ' << yolo_rect.x_center << ' ' << yolo_rect.y_center << ' ' << yolo_rect.width << ' ' << yolo_rect.height << "\n";
			}
			bboxFile.close();

			bbnum = count_tp_by_iou_thresh(iou_thresh, input_gt_bbPath + fileName.substr(0, dotPos) + ".txt", output_bboxFileName);
			// 击中数和处理时间
			std::ostringstream ss3;
			ss3 << "hits = " << bbnum << " " << "time=" << elapsed_secs * 1000 << "ms";
			String text3 = ss3.str();
			Point pos3(10, 30);

			// 定义字体类型和大小
			int fontface = FONT_HERSHEY_SIMPLEX;
			double fontsize = 0.8;

			// 定义文字的颜色和厚度
			Scalar t_color(255, 192, 203);
			int t_thickness = 2;

			cv::putText(image_mat, text3, pos3, fontface, fontsize, t_color, t_thickness);

			// Write image and bbox to files
			std::string output_imageFileName = output_imagePath + fileName.substr(0, dotPos) + ".jpg";
			cv::imwrite(output_imageFileName, image_mat);

			// Show image
			cv::imshow("current image", image_mat);
			cv::waitKey(40);

		}
	}
	delete t;
	return 0;
}

int count_tp_by_iou_thresh(double iou_thresh, string gt_filepath, string pred_filepath){

	std::vector<Box> true_boxes = read_boxes_from_file(gt_filepath);
    std::vector<Box> pred_boxes = read_boxes_from_file(pred_filepath);
    int num_tp = 0;

    for (Box pred_box : pred_boxes) {
        bool is_matched = false;
        double max_iou = 0;
        for (Box true_box : true_boxes) {
            double iou = cal_iou(true_box, pred_box);
            if (iou >= iou_thresh && iou > max_iou) {
                is_matched = true;
                max_iou = iou;
            }
        }
        if (is_matched) {
            num_tp++;
        }
    }

    return num_tp;
}

double cal_iou(Box box1, Box box2) {
    double left = max(box1.x_center - box1.width / 2., box2.x_center - box2.width / 2.);
    double right = min(box1.x_center + box1.width / 2., box2.x_center + box2.width / 2.);
    double top = max(box1.y_center - box1.height / 2., box2.y_center - box2.height / 2.);
    double bottom = min(box1.y_center + box1.height / 2., box2.y_center + box2.height / 2.);

    double intersection_area = 0;
    if (left < right && top < bottom) {
        intersection_area = (right - left) * (bottom - top);
    }

    double union_area = box1.width * box1.height + box2.width * box2.height - intersection_area;

    return intersection_area / union_area;
}

vector<Box> read_boxes_from_file(string filepath) {
    vector<Box> boxes;
    ifstream infile(filepath);

    string line;
    while (getline(infile, line)) {
        istringstream iss(line);
        Box box;
        iss >> box.class_id >> box.x_center >> box.y_center >> box.width >> box.height;
        boxes.push_back(box);
    }

    return boxes;
}

cv::Mat get_mat_fromfile(std::string fullfilename, int rows, int cols)
{
    cv::Mat matrix(rows, cols, CV_32SC1);
    std::ifstream depth_file(fullfilename);
    // read the matrix elements
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            depth_file >> matrix.at<int>(i, j);
        }
    }
    return matrix;
}

YOLO_RECT COCO2YOLO(cv::Rect coco_rect, int width, int height)
{
    float x_center = (1.0 * coco_rect.x + coco_rect.width / 2.0) / width;
    float y_center = (1.0 * coco_rect.y + coco_rect.height / 2.0) / height;
    float bb_width = 1.0 * coco_rect.width / width;
    float bb_height = 1.0 * coco_rect.height / height;

    YOLO_RECT rect = { x_center, y_center, bb_width, bb_height };
    return rect;
}

bool isPostiveBB() {
    return true;
}
// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
