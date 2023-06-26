#include <opencv2/opencv.hpp>
#include <iostream>
#define MAX_NUM 32

class TOD
{
public:
	//����/����
	TOD();
	~TOD();
	//��ʼ��
	bool Initialize(int width, int height, int estW, int estH, int blockNumH, int blockNumW);
	//ִ��
	bool DoVer(cv::Mat* pGryImg, cv::Mat* depthImg, int frameID);
	bool DoHor(cv::Mat* pGryImg, cv::Mat* depthImg, int frameID);
	//�õ�ˮƽ����������յ�
	int GetHorRange(int* pX1, int* pX2, int N);
	//�õ���ֱ����������յ�
	int GetVerRange(int* pX1, int* pX2, int N);
	//�õ�ˮƽ���귶Χ
	int GetHorCandidate(int candidate[][2], int N);
	//�õ���ֱ���귶Χ
	int GetVerCandidate(int candidate[][2], int N);
	//�õ�ˮƽ���귶Χ�ֿ�
	int GetHorFrameID(int* candidateFrameID, int N);
	//���ش�ֱͶӰ����
	int* GetPrjVer();
	//����ˮƽͶӰ����
	int* GetPrjHor();
	//�����ã���ӡ����
	void print();
private:
	//�ڴ��ͷ�
	void Dump();
	//��ֱͶӰ�ָ�
	void PrjAndSegmentVer(cv::Mat* pGryImg, cv::Mat* depthImg, int frameID);
	//ˮƽͶӰ�ָ�
	void PrjAndSegmentHor(cv::Mat* pGryImg, cv::Mat* depthImg, int frameID);
	//���㴹ֱͶӰ
	void GetProjectVer(cv::Mat* pGryImg, int width, int height1, int height2, int* prjVer);
	//����ˮƽͶӰ
	void GetProjectHor(cv::Mat* pGryImg, int width1, int width2, int height, int* prjHor);
	//������ȴ�ֱͶӰ
	void GetDepthProjectVer(cv::Mat* depthImg, int width, int height1, int height2, int* prjVer);
	//�������ˮƽͶӰ
	void GetDepthProjectHor(cv::Mat* depthImg, int width1, int width2, int height, int* prjHor);
	//һά���ݾ�ֵ�˲�
	void AverageFilter1D(int* pData, int n, int filterSize, int* pResData);
	//��������Ӧ��ֵ
	int GetDifAdaptiveThreshold(int* prjOrg, int* prjAvr, int width);
	//Otsu��ֵ��ȡ
	int GetOtsuThreshold(int* histogram, int nSize);
private:
	//��ʼ���ɹ��ı�־
	bool m_isInitedOK;
	//ͼ������
	int m_width;
	int m_height;
	//���Ƶķ�����/�߶�
	int m_estW;
	int m_estH;
	//�ڴ�
	int* m_prjVerOrg;
	int* m_prjVerAvr;
	int* m_prjHorOrg;
	int* m_prjHorAvr;
	int* m_prjVerDepth;
	int* m_prjHorDepth;
	int m_memImgWSize;
	int m_memImgHSize;
	//���
	int m_threshold;
	int m_pX1[MAX_NUM];
	int m_pX2[MAX_NUM];
	int m_pXFrameID[MAX_NUM];
	int m_num;
	int m_candidate[MAX_NUM][2];
	int m_candidateFrameID[MAX_NUM];
	int m_candidateNum;
	//ͼ��֡��
	int m_frameID;
	int m_blockNumH;
	int m_blockNumW;
	int m_blockHeight;
	int m_blockWidth;
};