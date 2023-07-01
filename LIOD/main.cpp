#include "LIOD.h"
#include "TOD.h"
#include "Detect.h"

int main() {
	std::vector<int> det_seq = {3, 6, 9};
	double iou_thresh = 0.3f;

	std::string output_path = "D:\\AAAAAAA\\documents\\scitri\\data\\output\\";
	std::string input_path = "D:\\AAAAAAA\\documents\\scitri\\data\\input\\";
	LIOD(input_path, output_path, iou_thresh, 45, 1, 5, det_seq);
}
