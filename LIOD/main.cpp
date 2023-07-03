#include "LIOD.h"
#include "TOD.h"
#include "Detect.h"

int main() {
	std::vector<int>* det_seq = new std::vector<int>{45,57};
	double iou_thresh = 0.3f;

	std::string output_path = "D:\\AAAAAAA\\documents\\scitri\\data\\output\\";
	std::string input_path = "D:\\AAAAAAA\\documents\\scitri\\data\\input\\";
	LIOD(input_path, output_path, iou_thresh, 3, 20, 1);
	delete det_seq;
}
