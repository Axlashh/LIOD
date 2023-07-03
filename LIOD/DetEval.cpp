#include "DetEval.h"

using namespace std;
namespace fs = std::filesystem;


float overlap(float x1, float w1, float x2, float w2) {
    float l1 = x1 - w1 / 2;
    float l2 = x2 - w2 / 2;
    float left = l1 > l2 ? l1 : l2;
    float r1 = x1 + w1 / 2;
    float r2 = x2 + w2 / 2;
    float right = r1 < r2 ? r1 : r2;
    return right - left;
}
float box_intersection(Object a, Object b) {
    float w = overlap(a.x, a.w, b.x, b.w);
    float h = overlap(a.y, a.h, b.y, b.h);
    if (w < 0 || h < 0) return 0;
    float area = w * h;
    return area;
}
float box_union(Object a, Object b) {
    float i = box_intersection(a, b);
    float u = a.w * a.h + b.w * b.h - i;
    return u;
}
float box_iou(Object a, Object b) {
    return box_intersection(a, b) / box_union(a, b);
}
bool cmp(Object a, Object b) {
    return a.confidence > b.confidence;
}


vector<Object> read_objects(string filename) {
    ifstream ifs(filename.c_str());
    vector<Object> objects;
    string line;
    while (getline(ifs, line)) {
        Object obj;
        stringstream ss(line);
        // 读取Yolo格式的真实框
        ss >> obj.class_id >> obj.x >> obj.y >> obj.w >> obj.h;
        obj.confidence = 1.0;  // 将所有检测框的置信度设置为1
        // 将Yolo格式的真实框转换为普通格式的真实框
        obj.x = obj.x - obj.w / 2;
        obj.y = obj.y - obj.h / 2;
        objects.push_back(obj);
    }
    return objects;
}

float compute_ap(vector<Object> objects, vector<Object> truth_objects, float overlap_threshold) {
    sort(objects.begin(), objects.end(), cmp);
    int tp = 0;
    int fp = 0;
    vector<bool> detected(truth_objects.size(), false);
    for (int i = 0; i < objects.size(); ++i) {
        Object& obj = objects[i];
        int best_match_idx = -1;
        float best_match_iou = 0;
        for (int j = 0; j < truth_objects.size(); ++j) {
            if (obj.class_id != truth_objects[j].class_id) continue;
            float iou = box_iou(obj, truth_objects[j]);
            if (iou > overlap_threshold && iou > best_match_iou) {
                best_match_idx = j;
                best_match_iou = iou;
            }
        }
        if (best_match_idx != -1) {
            if (!detected[best_match_idx]) {
                tp++;
                detected[best_match_idx] = true;
            }
            else {
                fp++;
            }
        }
        else {
            fp++;
        }
    }
    int fn = 0;
    for (int i = 0; i < truth_objects.size(); ++i) {
        if (!detected[i]) {
            fn++;
        }
    }
    float precision = (float)tp / (tp + fp);
    float recall = (float)tp / (tp + fn);
    float ap = precision * recall;
    return ap;
}

void DetEval(std::string input_path, std::string output_path, int init_seq, int seq_num,
    bool only_move, float overlap_threshold0, float overlap_thresholdm, float gap) {
    std::string out_file_path = output_path + "resAP.txt";
    std::string gt_path;
    std::string bb_path;
    ofstream out_file(out_file_path);

    while (seq_num--) {
        std::ostringstream tseq;
        tseq << "seq" << std::setfill('0') << std::setw(2) << std::to_string(init_seq);
        std::string seq = tseq.str();
        gt_path = input_path + seq + "\\gt_bb\\";
        bb_path = output_path + seq + "\\pred_bb\\";
        init_seq++;
        if (only_move) init_seq += 2;

        vector<string> file_path1;  // 保存第一个文件夹下的所有txt文件的路径
        vector<string> file_path2;  // 保存第二个文件夹下的所有txt文件的路径
        for (const auto& entry1 : fs::directory_iterator(gt_path)) {  // 遍历第一个文件夹下的所有文件和文件夹
            if (entry1.is_regular_file() && entry1.path().extension() == ".txt") {  // 如果是txt文件
                file_path1.push_back(entry1.path().string());  // 将文件路径添加到第一个向量中
            }
        }
        for (const auto& entry2 : fs::directory_iterator(bb_path)) {  // 遍历第二个文件夹下的所有文件和文件夹
            if (entry2.is_regular_file() && entry2.path().extension() == ".txt") {  // 如果是txt文件
                file_path2.push_back(entry2.path().string());  // 将文件路径添加到第二个向量中
            }
        }
        float overlap_threshold0 = 0.5f;
        float overlap_thresholdm = 0.95f;
        float gap = 0.05f;
        float overlap_threshold = overlap_threshold0;
        float avrsum = 0;
        int apnum = 0;
        for (int i = 0; overlap_threshold <= overlap_thresholdm + gap; i++)
        {
            double totalnum = 0; double sumAP = 0;
            for (size_t i = 0; i < file_path1.size(); i++) {  // 遍历第一个向量中的所有文件
                for (size_t j = 0; j < file_path2.size(); j++) {  // 遍历第二个向量中的所有文件
                    if (fs::path(file_path1[i]).filename() == fs::path(file_path2[j]).filename()) {  // 如果文件名相同
                        //操作               
                        string truth_filename = file_path1[i];
                        string detection_filename = file_path2[j];
                        vector<Object> truth_objects = read_objects(truth_filename);
                        vector<Object> detection_objects = read_objects(detection_filename);
                        float ap = compute_ap(detection_objects, truth_objects, overlap_threshold);
                        if (isfinite(ap))//如果ap为数字，省去空的情况
                        {
                            //cout <<ap<<endl;
                            sumAP = sumAP + ap;
                        }
                    }
                }
                totalnum++;
            }
            avrsum += sumAP / totalnum;
            if (out_file.is_open()) {
                out_file << "IOU@" << overlap_threshold << "mAP = " << sumAP / totalnum << endl;
            }
            cout << "IOU@" << overlap_threshold << "mAP = " << sumAP / totalnum << endl;
            overlap_threshold += gap;
            apnum = i + 1;
        }
        cout << "IOU@" << overlap_threshold0 << "~" << overlap_threshold - gap << "mAP=" << avrsum / apnum << endl << endl;

        out_file << "IOU@" << overlap_threshold0 << "~" << overlap_threshold - gap << "mAP=" << avrsum / apnum << endl << endl;
    }
}