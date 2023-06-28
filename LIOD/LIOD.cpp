// LIOD.cpp : This file contains the 'main' function. Program execution begins and ends there.
#include "LIOD.h"
#define _CRT_SECURE_NO_WARNINGS

namespace fs = std::filesystem;
using namespace std;
using cv::Scalar;
using cv::Point;

int group = 0;
int nummm = 0;
time_t timep;
struct tm* t;

// input info
std::string input_depthPath;
std::string input_gt_bbPath;
std::string input_imagePath;
// output info
std::string output_imagePath;
std::string output_bboxPath;
double iou_thresh = 0.3f;

std::string output_path;
std::string input_path = "D:\\AAAAAAA\\documents\\scitri\\data\\input\\";
std::string calib_path = "D:\\AAAAAAA\\documents\\scitri\\data\\calib\\1.txt";
std::string point_cloud_path = "D:\\AAAAAAA\\documents\\scitri\\data\\point_cloud\\";

char tout[200];
const char* out = "D:\\AAAAAAA\\documents\\scitri\\data\\output\\%d_%d_%d_%d_%02d\\";

//the number of seqence to be detected
int seq_num = 1;
//the initial seqence
int init_seq = 33;
//only detect the movement(skip 2 seqences each time)
bool only_move = true;
//the pause time between two pictures
int pause_time = 1;
//camera static
int fx = 100;
int fy = 100;

int main()
{
	//create the output directory with time information
	time(&timep);
	t = new struct tm; 
	localtime_s(t, &timep);
    sprintf_s(tout, sizeof(tout), out, 1900 + t->tm_year,1 + t->tm_mon, t->tm_wday, t->tm_hour, t->tm_min);
    output_path = tout;
 	if (!fs::exists(output_path)) {
		fs::create_directory(output_path);
	}

    // Read each line of the list file and load the corresponding image file.
    std::string image_fullfilename;
    std::vector<cv::Rect> BBVector;
	while (seq_num--) {
		//get the corresponding seqence
		std::ostringstream tseq;
		tseq << "seq" << std::setfill('0') << std::setw(2) << std::to_string(init_seq);
		std::string seq = tseq.str();
		input_depthPath = input_path + seq + "\\depth\\";
		input_gt_bbPath = input_path + seq + "\\gt__bb\\";
		input_imagePath = input_path + seq + "\\images\\";
		output_imagePath = output_path + seq + "\\images\\";
		output_bboxPath = output_path + seq + "\\pred__bb\\";
		point_cloud_path = point_cloud_path + seq + "21\\";
		fs::create_directories(output_imagePath);
		fs::create_directories(output_bboxPath);
		fs::create_directories(point_cloud_path);

		//add all jpg files' path to the vector
		std::vector<std::string> files;
		for (const auto& f : fs::directory_iterator(input_imagePath)) {
			if (f.path().extension() == ".jpg")
				files.push_back(f.path().string());
		}

		//iterate all the files
		for (auto& image_fullfilename : files) {
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
			nummm = 0;
			group++;
			for (auto it = BBVector.begin(); it != BBVector.end();) {
				cv::Rect bb = *it;

				if (!isPostiveBB_BL(bb, depth_mat)) {
					it = BBVector.erase(it);
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
				string text1 = ss1.str();
				string text2 = ss2.str();
				Point pos1(bb.x, bb.y - 5);
				Point pos2(bb.x, bb.y - 18);

				// 定义字体类型和大小
				int fontface = cv::FONT_HERSHEY_SIMPLEX;
				double fontsize = 0.4;

				// 定义文字的颜色和厚度
				Scalar t_color(255, 192, 203);
				int t_thickness = 0.6;

				// 在图片上添加文字
				cv::putText(image_mat, text1, pos1, fontface, fontsize, t_color, t_thickness);
				cv::putText(image_mat, text2, pos2, fontface, fontsize, t_color, t_thickness);

				it++;
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
			string text3 = ss3.str();
			Point pos3(10, 30);

			// 定义字体类型和大小
			int fontface = cv::FONT_HERSHEY_SIMPLEX;
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
			cv::waitKey(pause_time);

		}
		init_seq++;
		if (only_move) init_seq += 2;
	}
	delete t;
	return 0;
}

int count_tp_by_iou_thresh(double iou_thresh, std::string gt_filepath, std::string pred_filepath){

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

std::vector<Box> read_boxes_from_file(std::string filepath) {
	std::vector<Box> boxes;
	std::ifstream infile(filepath);

	std::string line;
    while (std::getline(infile, line)) {
		std::istringstream iss(line);
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

TY_CAMERA_CALIB_INFO* read_calib(string path) {
	TY_CAMERA_CALIB_INFO* ret = new TY_CAMERA_CALIB_INFO();
	std::ifstream inputFile(path);
	if (inputFile.is_open()) {
		inputFile >> ret->intrinsicWidth;
		inputFile >> ret->intrinsicHeight;
		for (int i = 0; i < 9; i++) 
			inputFile >> ret->intrinsic.data[i];
		for (int i = 0; i < 16; i++) 
			inputFile >> ret->extrinsic.data[i];
		for (int i = 0; i < 12; i++)
			inputFile >> ret->distortion.data[i];
	}
	inputFile.close();
	return ret;
}

bool isPostiveBB_BL(cv::Rect rec, cv::Mat depth_mat) {
	cv::Mat rigion = depth_mat(rec).clone();
	cv::Mat bb_depth = cv::Mat::zeros(depth_mat.rows, depth_mat.cols, depth_mat.type());
	for (int i = 0; i < rigion.cols; i++) {
		for (int j = 0; j < rigion.rows; j++) {
			bb_depth.at<int32_t>(rec.y + j, rec.x + i) = rigion.at<int32_t>(j, i);
		}
	}
	/* 图漾的转换点云
	std::vector<TY_VECT_3F> pc;
	pc.resize(bb_depth.size().area());
	TYMapDepthImageToPoint3d(read_calib(calib_path), bb_depth.cols, bb_depth.rows
					, (uint16_t*)bb_depth.data, &pc[0]);*/
	pt3d pc(bb_depth);
	//std::string pa = point_cloud_path + std::to_string(group) + "//";
	//fs::create_directory(pa);
	//std::string f_path = pa + std::to_string(nummm++) + ".txt";
	//writePointCloud(point3d, f_path.c_str());
	if (pc.size() == 0) return true;
	long long y = pc.at(0).y;
	cout << y << endl;
	if (y < -500) return false;
	return true;
}

static void writePointCloud(const cv::Point3f* pnts, const cv::Vec3b* color, size_t n, const char* file, int format)
{
	FILE* fp = nullptr;
	errno_t err = fopen_s(&fp, file, "w");
	if (err) {
		return;
	}

	switch (format) {
	case 0:
		writePC_XYZ(pnts, color, n, fp);
		break;
	default:
		break;
	}

	fclose(fp);
}

static void writePC_XYZ(const cv::Point3f* pnts, const cv::Vec3b* color, size_t n, FILE* fp)
{
	if (color) {
		for (size_t i = 0; i < n; i++) {
			if (!std::isnan(pnts[i].x)) {
				fprintf(fp, "%f %f %f %d %d %d\n", pnts[i].x, pnts[i].y, pnts[i].z, color[i][0], color[i][1], color[i][2]);
			}
		}
	}
	else {
		for (size_t i = 0; i < n; i++) {
			if (!std::isnan(pnts[i].x)) {
				fprintf(fp, "%f %f %f 0 0 0\n", pnts[i].x, pnts[i].y, pnts[i].z);
			}
		}
	}
}

int WriteData(std::string fileName, cv::Mat& matData)
{
	int retVal = 0;

	// 打开文件  
	ofstream outFile(fileName.c_str(), ios_base::out);  //按新建或覆盖方式写入  
	if (!outFile.is_open())
	{
		cout << "打开文件失败" << endl;
		retVal = -1;
		return (retVal);
	}

	// 检查矩阵是否为空  
	if (matData.empty())
	{
		cout << "矩阵为空" << endl;
		retVal = 1;
		return (retVal);
	}

	// 写入数据  
	for (int r = 0; r < matData.rows; r++)
	{
		for (int c = 0; c < matData.cols; c++)
		{
			int32_t data = matData.at<int32_t>(r, c);  //读取数据，at<type> - type 是矩阵元素的具体数据格式  
			outFile << data << "\t";   //每列数据用 tab 隔开  
		}
		outFile << endl;  //换行  
	}
	return (retVal);
}

pt3d::pt3d() {
	pc = new std::vector<vec_3d>();
}

pt3d::pt3d(cv::Mat mat) : pt3d() {
	this->depth2PointCloud(mat);
}

pt3d::~pt3d() {
	delete pc;
}

int pt3d::size() {
	return pc->size();
}

std::vector<vec_3d>::iterator pt3d::iterator() {
	return pc->begin();
}

vec_3d pt3d::at(int position) {
	return pc->at(position);
}

void pt3d::depth2PointCloud(cv::Mat mat) {
	int x_cen = mat.cols / 2;
	int y_cen = mat.rows / 2;
	for (int i = 0; i < mat.rows; i++) {
		for (int j = 0; j < mat.cols; j++) {
			float dep = mat.at<int32_t>(i, j);
			if (!dep) continue;
			float tx = (j - x_cen) * dep / fx;
			float ty = (i - y_cen) * dep / fy;
			pc->push_back({ tx, ty, dep });
		}
	}
}

void pt3d::writePointCloud(const char* path) {
	FILE* fp = nullptr;
	errno_t err = fopen_s(&fp, path, "w");
	if (err) return;

	for (auto& p : *pc) {
		fprintf(fp, "%f %f %f\n", p.x, p.y, p.z);
	}
	fclose(fp);
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
