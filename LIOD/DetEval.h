#pragma once
#include <string>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <sstream>
#include <cmath>

struct Object {
    int class_id;
    float confidence;
    float x, y, w, h;
};

float overlap(float x1, float w1, float x2, float w2);

float box_intersection(Object a, Object b);

float box_union(Object a, Object b);

float box_iou(Object a, Object b);

bool cmp(Object a, Object b);

std::vector<Object> read_objects(std::string filename);

float compute_ap(std::vector<Object> objects, std::vector<Object> truth_objects, float overlap_threshold);

void DetEval(std::string input_path, std::string output_path, int init_seq, int seq_num, 
    bool only_move = true, float overlap_threshold0 = 0.5f, float overlap_thresholdm = 0.95f, float gap = 0.05f);


