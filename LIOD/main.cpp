#include "LIOD.h"
#include "TOD.h"
#include "Detect.h"
#include "DetEval.h"

namespace fs = std::filesystem;

int main() {
	std::vector<int>* det_seq = new std::vector<int>{45,57};
	double iou_thresh = 0.3f;

	std::string output_path = "D:\\AAAAAAA\\documents\\scitri\\data\\output\\";
	std::string input_path = "D:\\AAAAAAA\\documents\\scitri\\data\\input\\";
	output_path = output_path + "%d_%d_%d_%d_%02d\\";
	time_t timep;
	struct tm* t;
	char tout[100];
	char out[90];
	strcpy_s(out, output_path.c_str());
	//create the output directory with time information
	time(&timep);
	t = new struct tm;
	localtime_s(t, &timep);
	sprintf_s(tout, 90, out, 1900 + t->tm_year, 1 + t->tm_mon, t->tm_mday, t->tm_hour, t->tm_min);
	output_path = tout;
	if (!fs::exists(output_path)) {
		fs::create_directory(output_path);
	}
	LIOD(input_path, output_path, iou_thresh, 3, 20, 300);

	DetEval(input_path, output_path, 3, 20);
	delete det_seq;
	delete t;
}
