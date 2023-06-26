#define _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "TOD.h"

//构造函数
TOD::TOD()
{
	m_isInitedOK = false;
	m_width = 0;
	m_height = 0;
	m_estH = 0;
	m_estW = 0;
	m_prjVerOrg = NULL;
	m_prjVerAvr = NULL;
	m_prjHorOrg = NULL;
	m_prjHorAvr = NULL;
	m_prjVerDepth = NULL;
	m_prjHorDepth = NULL;
	m_memImgWSize = 0;
	m_memImgHSize = 0;
	m_threshold = 0;
	m_num = 0;
	m_candidateNum = 0;
	m_frameID = 0;
	m_blockNumH = 0;
	m_blockNumW = 0;
	m_blockHeight = 0;
	m_blockWidth = 0;
}

//析构函数
TOD::~TOD()
{
	Dump();
}

//内存释放
void TOD::Dump()
{
	if (m_prjVerOrg)
	{
		delete m_prjVerOrg;
		m_prjVerOrg = NULL;
	}
	if (m_prjVerAvr)
	{
		delete m_prjVerAvr;
		m_prjVerAvr = NULL;
	}
	if (m_prjVerDepth)
	{
		delete m_prjVerDepth;
		m_prjVerDepth = NULL;
	}
	if (m_prjHorOrg)
	{
		delete m_prjHorOrg;
		m_prjHorOrg = NULL;
	}
	if (m_prjHorAvr)
	{
		delete m_prjHorAvr;
		m_prjHorAvr = NULL;
	}
	if (m_prjHorDepth)
	{
		delete m_prjHorDepth;
		m_prjHorDepth = NULL;
	}
}

//初始化
bool TOD::Initialize(int width, int height, int estW, int estH, int blockNumH, int blockNumW)
{
	m_width = width;
	m_height = height;
	m_estW = estW;
	m_estH = estH;
	m_blockNumH = blockNumH;
	m_blockNumW = blockNumW;
	m_blockHeight = height / blockNumH;
	m_blockWidth = width / blockNumW;
	//内存申请
	if (m_width > m_memImgWSize || m_height > m_memImgHSize)
	{
		Dump();
		m_prjVerOrg = new int[m_width];
		m_prjVerAvr = new int[m_width];
		m_prjVerDepth = new int[m_width];
		m_prjHorOrg = new int[m_height];
		m_prjHorAvr = new int[m_height];
		m_prjHorDepth = new int[m_height];
		m_memImgWSize = m_width;
		m_memImgHSize = m_height;
	}
	m_isInitedOK = m_prjVerOrg && m_prjVerAvr && m_prjHorOrg && m_prjHorAvr && m_prjVerDepth && m_prjHorDepth;
	return m_isInitedOK;
}

//执行
bool TOD::DoVer(cv::Mat* pGryImg, cv::Mat* depthImg, int frameID)
{
	if (!m_isInitedOK)
		return false;
	if (frameID == 0)
		m_candidateNum = 0;
	m_frameID = frameID;
	PrjAndSegmentVer(pGryImg, depthImg, frameID);
	double IOU;
	int flag;
	for (int i = 0; i < m_num; i++)
	{
		flag = 0;
		for (int j = 0; j < m_candidateNum; j++)
		{
			if (m_pX1[i] >= m_candidate[j][0] && m_pX2[i] <= m_candidate[j][1])
			{
				flag = 1;
				break;
			}
			if (m_pX1[i] <= m_candidate[j][0] && m_pX2[i] >= m_candidate[j][1])
			{
				m_candidate[j][0] = m_pX1[i];
				m_candidate[j][1] = m_pX2[i];
				m_candidateFrameID[j] = m_pXFrameID[i];
				flag = 1;
				break;
			}
			if (m_pX1[i] <= m_candidate[j][0] && m_pX2[i] >= m_candidate[j][0] && m_candidate[j][1] >= m_pX2[i])
			{
				IOU = (m_pX2[i] - m_candidate[j][0]) / (double)(m_candidate[j][1] - m_pX1[i]);
				if (IOU >= 0.3)
				{
					flag = 1;
					break;
				}	
			}
			if (m_candidate[j][0] <= m_pX1[i] && m_candidate[j][1] >= m_pX1[i] && m_pX2[i] >= m_candidate[j][1])
			{
				IOU = (m_candidate[j][1] - m_pX1[i]) / (double)(m_pX2[i] - m_candidate[j][0]);
				if (IOU >= 0.3)
				{
					flag = 1;
					break;
				}
			}
		}
		if (flag == 0 && m_candidateNum <= MAX_NUM)
		{
			m_candidate[m_candidateNum][0] = m_pX1[i];
			m_candidate[m_candidateNum][1] = m_pX2[i];
			m_candidateFrameID[m_candidateNum] = m_pXFrameID[i];
			m_candidateNum++;
		}
	}
	return true;
}

bool TOD::DoHor(cv::Mat* pGryImg, cv::Mat* depthImg, int frameID)
{
	if (!m_isInitedOK)
		return false;
	if (frameID == 0)
		m_candidateNum = 0;
	m_frameID = frameID;
	PrjAndSegmentHor(pGryImg, depthImg, frameID);
	double IOU;
	int flag;
	for (int i = 0; i < m_num; i++)
	{
		flag = 0;
		for (int j = 0; j < m_candidateNum; j++)
		{
			if (m_pX1[i] >= m_candidate[j][0] && m_pX2[i] <= m_candidate[j][1])
			{
				flag = 1;
				break;
			}
			if (m_pX1[i] <= m_candidate[j][0] && m_pX2[i] >= m_candidate[j][1])
			{
				m_candidate[j][0] = m_pX1[i];
				m_candidate[j][1] = m_pX2[i];
				flag = 1;
				break;
			}
			if (m_pX1[i] <= m_candidate[j][0] && m_pX2[i] >= m_candidate[j][0] && m_candidate[j][1] >= m_pX2[i])
			{
				IOU = (m_pX2[i] - m_candidate[j][0]) / (double)(m_candidate[j][1] - m_pX1[i]);
				if (IOU >= 0.3)
				{
					flag = 1;
					break;
				}
			}
			if (m_candidate[j][0] <= m_pX1[i] && m_candidate[j][1] >= m_pX1[i] && m_pX2[i] >= m_candidate[j][1])
			{
				IOU = (m_candidate[j][1] - m_pX1[i]) / (double)(m_pX2[i] - m_candidate[j][0]);
				if (IOU >= 0.3)
				{
					flag = 1;
					break;
				}
			}
		}
		if (flag == 0 && m_candidateNum <= MAX_NUM)
		{
			m_candidate[m_candidateNum][0] = m_pX1[i];
			m_candidate[m_candidateNum][1] = m_pX2[i];
			m_candidateNum++;
		}
	}
	return true;
}

//得到水平波峰的起点和终点
int TOD::GetHorRange(int* pX1, int* pX2, int N)
{
	for (int i = 0; i < std::min(N, m_num); i++)
	{
		pX1[i] = m_pX1[i];
		pX2[i] = m_pX2[i];
	}
	return std::min(N, m_num);
}

//得到垂直波峰的起点和终点
int TOD::GetVerRange(int* pX1, int* pX2, int N)
{
	for (int i = 0; i < std::min(N, m_num); i++)
	{
		pX1[i] = m_pX1[i];
		pX2[i] = m_pX2[i];
	}
	return std::min(N, m_num);
}

//得到水平坐标范围
int TOD::GetHorCandidate(int candidate[][2], int N)
{
	for (int i = 0; i < std::min(N, m_candidateNum); i++)
	{
		candidate[i][0] = m_candidate[i][0];
		candidate[i][1] = m_candidate[i][1];
	}
	return std::min(N, m_candidateNum);
}

//得到垂直坐标范围
int TOD::GetVerCandidate(int candidate[][2], int N)
{
	for (int i = 0; i < std::min(N, m_candidateNum); i++)
	{
		candidate[i][0] = m_candidate[i][0];
		candidate[i][1] = m_candidate[i][1];
	}
	return std::min(N, m_candidateNum);
}

//得到水平坐标范围分块
int TOD::GetHorFrameID(int* candidateFrameID, int N)
{
	for (int i = 0; i < std::min(N, m_candidateNum); i++)
	{
		candidateFrameID[i] = m_candidateFrameID[i];
		candidateFrameID[i] = m_candidateFrameID[i];
	}
	return std::min(N, m_candidateNum);
}

//返回垂直投影数据
int* TOD::GetPrjVer()
{
	return m_prjVerOrg;
}

//返回水平投影数据
int* TOD::GetPrjHor()
{
	return m_prjHorOrg;
}

//计算垂直投影
void TOD::GetProjectVer(cv::Mat* pGryImg, int width, int height1, int height2, int* prjVer)
{
	int x, y;
	memset(prjVer, 0, sizeof(int) * width);
	for (y = height1; y < height2; y++)
	{
		for (x = 0; x < width; x++)
			prjVer[x] += (int)pGryImg->at<unsigned char>(y, x);
	}
	return;
}

//计算水平投影
void TOD::GetProjectHor(cv::Mat* pGryImg, int width1, int width2, int height, int* prjHor)
{
	int x, y;
	memset(prjHor, 0, sizeof(int) * height);
	for (y = width1; y < width2; y++)
	{
		for (x = 0; x < height; x++)
			prjHor[x] += (int)pGryImg->at<unsigned char>(x, y);
	}
	return;
}

//计算深度垂直投影
void TOD::GetDepthProjectVer(cv::Mat* depthImg, int width, int height1, int height2, int* prjVer)
{
	int x, y;
	memset(prjVer, 0, sizeof(int) * width);
	for (y = height1; y < height2; y++)
	{
		for (x = 0; x < width; x++)
			prjVer[x] += (int)depthImg->at<int>(y, x);
	}
	return;
}

//计算深度水平投影
void TOD::GetDepthProjectHor(cv::Mat* depthImg, int width1, int width2, int height, int* prjHor)
{
	int x, y;
	memset(prjHor, 0, sizeof(int) * height);
	for (y = width1; y < width2; y++)
	{
		for (x = 0; x < height; x++)
			prjHor[x] += (int)depthImg->at<int>(x, y);
	}
	return;
}

//一维数据均值滤波
void TOD::AverageFilter1D(int* pData, int n, int filterSize, int* pResData)
{
	int sum, wSize, halfw;
	int x, j;
	halfw = filterSize / 2;
	wSize = 1;
	for (sum = pData[0], x = 0, j = 1; x < halfw; x++, j += 2)
	{
		pResData[x] = sum / wSize;
		sum += pData[j] + pData[j + 1];
		wSize += 2;
	}
	wSize = (2 * halfw + 1);
	for (x = halfw; x < n - halfw - 1; x++)
	{
		pResData[x] = sum / wSize;
		sum = sum - pData[x - halfw] + pData[x + halfw + 1];
	}
	for (x = n - halfw - 1, j = x - halfw; x < n; x++, j += 2)
	{
		pResData[x] = sum / wSize;
		sum -= (pData[j] + pData[j + 1]);
		wSize -= 2;
	}
	return;
}

//Otsu阈值求取
int TOD::GetOtsuThreshold(int* histogram, int nSize)
{
	int thre;
	int i, gmin, gmax;
	double dist, f, max;
	int s1, s2, n1, n2, n;
	gmin = 0;
	while (histogram[gmin] == 0)
		++gmin;
	gmax = nSize - 1;
	while (histogram[gmax] == 0)
		--gmax;
	if (gmin == gmax)
		return gmin;
	max = 0;
	thre = 0;
	s1 = n1 = 0;
	for (s2 = n2 = 0, i = gmin; i <= gmax; i++)
	{
		s2 += histogram[i] * i;
		n2 += histogram[i];
	}
	for (i = gmin, n = n2; i < gmax; i++)
	{
		if (!histogram[i])
			continue;
		s1 += histogram[i] * i;
		s2 -= histogram[i] * i;
		n1 += histogram[i];
		n2 -= histogram[i];
		dist = (s1 * 1.0 / n1 - s2 * 1.0 / n2);
		f = dist * dist * (n1 * 1.0 / n) * (n2 * 1.0 / n);
		if (f > max)
		{
			max = f;
			thre = i;
		}
	}
	return thre + 1;
}

//计算自适应阈值
int TOD::GetDifAdaptiveThreshold(int* prjOrg, int* prjAvr, int width)
{
	int histogram[256];
	int x, dif;
	memset(histogram, 0, sizeof(int) * 256);
	for (x = 0; x < width; x++)
	{
		dif = abs(prjOrg[x] - prjAvr[x]);
		histogram[dif]++;
	}
	return GetOtsuThreshold(histogram, 256);
}

//垂直投影分割
void TOD::PrjAndSegmentVer(cv::Mat* pGryImg, cv::Mat* depthImg, int frameID)
{
	int x, dif;
	bool isBegin;
	double length;
	int haveDepth;
	GetProjectVer(pGryImg, m_width, frameID * m_blockHeight, (frameID + 1) * m_blockHeight, m_prjVerOrg);
	GetDepthProjectVer(depthImg, m_width, frameID * m_blockHeight, (frameID + 1) * m_blockHeight, m_prjVerDepth);
	//归一化
	for (x = 0; x < m_width; x++)
	{
		m_prjVerOrg[x] /= m_blockHeight;
		m_prjVerDepth[x] /= m_blockHeight;
	}
	//均值滤波
	AverageFilter1D(m_prjVerOrg, m_width, m_estW, m_prjVerAvr);
	//计算自适应阈值
	m_threshold = GetDifAdaptiveThreshold(m_prjVerOrg, m_prjVerAvr, m_width);
	m_threshold = std::max(4, m_threshold);
	m_num = 0;
	for (isBegin = true, x = 0; x < m_width; x++)
	{
		dif = m_prjVerOrg[x] - m_prjVerAvr[x];
		if (dif < m_threshold)
		{
			if (!isBegin)
			{
				//记录终点
				if (m_num < MAX_NUM)
				{
					if (x - m_pX1[m_num] > 10)
					{
						m_pX2[m_num] = x - 1;
						m_pXFrameID[m_num] = frameID;
						length = (double)(m_pX2[m_num] - m_pX1[m_num] + 1);
						haveDepth = 0;
						for (int i = m_pX1[m_num]; i <= m_pX2[m_num]; i++)
						{
							if (m_prjVerDepth[i] > 0)
								haveDepth++;
						}
						if((haveDepth/length)>=0.3)
							m_num++;
					}
				}
			}
			isBegin = true;
		}
		else
		{
			//记录起点
			if (isBegin)
			{
				m_pX1[m_num] = x;
				isBegin = false;
			}
		}
	}
	return;
}

//水平投影分割
void TOD::PrjAndSegmentHor(cv::Mat* pGryImg, cv::Mat* depthImg, int frameID)
{
	int x, dif;
	bool isBegin;
	double length;
	int haveDepth;
	GetProjectHor(pGryImg, frameID * m_blockWidth, (frameID + 1) * m_blockWidth, m_height, m_prjHorOrg);
	GetDepthProjectHor(depthImg, frameID * m_blockWidth, (frameID + 1) * m_blockWidth, m_height, m_prjHorDepth);
	//归一化
	for (x = 0; x < m_height; x++)
	{
		m_prjHorOrg[x] /= m_blockWidth;
		m_prjHorDepth[x] /= m_blockWidth;
	}
	//均值滤波
	AverageFilter1D(m_prjHorOrg, m_height, m_estH, m_prjHorAvr);
	//计算自适应阈值
	m_threshold = GetDifAdaptiveThreshold(m_prjHorOrg, m_prjHorAvr, m_height);
	m_threshold = std::max(4, m_threshold);
	m_num = 0;
	for (isBegin = true, x = 0; x < m_height; x++)
	{
		dif = m_prjHorOrg[x] - m_prjHorAvr[x];
		if (dif < m_threshold)
		{
			if (!isBegin)
			{
				//记录终点
				if (m_num < MAX_NUM)
				{
					if (x - m_pX1[m_num] > 10)
					{
						m_pX2[m_num] = x - 1;
						length = (double)(m_pX2[m_num] - m_pX1[m_num] + 1);
						haveDepth = 0;
						for (int i = m_pX1[m_num]; i <= m_pX2[m_num]; i++)
						{
							if (m_prjHorDepth[i] > 0)
								haveDepth++;
						}
						if ((haveDepth / length) >= 0.3)
							m_num++;
					}
				}
			}
			isBegin = true;
		}
		else
		{
			//记录起点
			if (isBegin)
			{
				m_pX1[m_num] = x;
				isBegin = false;
			}
		}
	}
	return;
}

//调试用，打印参数
void TOD::print()
{
	std::cout << "width:" << m_width << std::endl;
	std::cout << "height:" << m_height << std::endl;
	std::cout << "estW:" << m_estW << std::endl;
	std::cout << "estH:" << m_estH << std::endl;
	std::cout << "blockNumW:" << m_blockNumW << std::endl;
	std::cout << "blockNumH:" << m_blockNumH << std::endl;
	std::cout << "m_blockHeight:" << m_blockHeight << std::endl;
	std::cout << "m_blockWidth:" << m_blockWidth << std::endl;
	return;
}
