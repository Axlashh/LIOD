#include <opencv2/opencv.hpp>
#include <iostream>
#define MAX_NUM 32

class TOD
{
public:
	//构造/析构
	TOD();
	~TOD();
	//初始化
	bool Initialize(int width, int height, int estW, int estH, int blockNumH, int blockNumW);
	//执行
	bool DoVer(cv::Mat* pGryImg, cv::Mat* depthImg, int frameID);
	bool DoHor(cv::Mat* pGryImg, cv::Mat* depthImg, int frameID);
	//得到水平波峰的起点和终点
	int GetHorRange(int* pX1, int* pX2, int N);
	//得到垂直波峰的起点和终点
	int GetVerRange(int* pX1, int* pX2, int N);
	//得到水平坐标范围
	int GetHorCandidate(int candidate[][2], int N);
	//得到垂直坐标范围
	int GetVerCandidate(int candidate[][2], int N);
	//得到水平坐标范围分块
	int GetHorFrameID(int* candidateFrameID, int N);
	//返回垂直投影数据
	int* GetPrjVer();
	//返回水平投影数据
	int* GetPrjHor();
	//调试用，打印参数
	void print();
private:
	//内存释放
	void Dump();
	//垂直投影分割
	void PrjAndSegmentVer(cv::Mat* pGryImg, cv::Mat* depthImg, int frameID);
	//水平投影分割
	void PrjAndSegmentHor(cv::Mat* pGryImg, cv::Mat* depthImg, int frameID);
	//计算垂直投影
	void GetProjectVer(cv::Mat* pGryImg, int width, int height1, int height2, int* prjVer);
	//计算水平投影
	void GetProjectHor(cv::Mat* pGryImg, int width1, int width2, int height, int* prjHor);
	//计算深度垂直投影
	void GetDepthProjectVer(cv::Mat* depthImg, int width, int height1, int height2, int* prjVer);
	//计算深度水平投影
	void GetDepthProjectHor(cv::Mat* depthImg, int width1, int width2, int height, int* prjHor);
	//一维数据均值滤波
	void AverageFilter1D(int* pData, int n, int filterSize, int* pResData);
	//计算自适应阈值
	int GetDifAdaptiveThreshold(int* prjOrg, int* prjAvr, int width);
	//Otsu阈值求取
	int GetOtsuThreshold(int* histogram, int nSize);
private:
	//初始化成功的标志
	bool m_isInitedOK;
	//图像属性
	int m_width;
	int m_height;
	//估计的方块宽度/高度
	int m_estW;
	int m_estH;
	//内存
	int* m_prjVerOrg;
	int* m_prjVerAvr;
	int* m_prjHorOrg;
	int* m_prjHorAvr;
	int* m_prjVerDepth;
	int* m_prjHorDepth;
	int m_memImgWSize;
	int m_memImgHSize;
	//结果
	int m_threshold;
	int m_pX1[MAX_NUM];
	int m_pX2[MAX_NUM];
	int m_pXFrameID[MAX_NUM];
	int m_num;
	int m_candidate[MAX_NUM][2];
	int m_candidateFrameID[MAX_NUM];
	int m_candidateNum;
	//图像帧号
	int m_frameID;
	int m_blockNumH;
	int m_blockNumW;
	int m_blockHeight;
	int m_blockWidth;
};