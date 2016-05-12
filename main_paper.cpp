#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <ctime>
#include <cstdio>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/core/core_c.h>
#include <opencv2/ml/ml.hpp>

using namespace std;
using namespace cv;

#define GLCM_DIS 3
#define GLCM_CLASS 64
#define GLCM_ANGLE_HORIZATION 0
#define GLCM_ANGLE_VERTICAL 1
#define GLCM_ANGLE_DIGONAL 2

#define VAR_COUNT 42

vector<double> all_feature;

int ImageAdjust(IplImage* src, IplImage* dst,
				double low, double high,   // X方向：low and high are the intensities of src
				double bottom, double top, // Y方向：mapped to bottom and top of dst
				double gamma )
{
	double low2 = low * 255;
	double high2 = high * 255;
	double bottom2 = bottom * 255;
	double top2 = top * 255;
	double err_in = high2 - low2;
	double err_out = top2 - bottom2;
	int x,y;
	double val;
	if(low<0 && low>1 && high <0 && high>1 && 
		bottom<0 && bottom>1 && top<0 && top>1 && low>high)
		return -1;
	// intensity transform
	CvScalar avg = cvAvg(src);
	double avg_gray = avg.val[0];

	for(y = 0; y < src->height; y++)
	{
		for (x = 0; x < src->width; x++)
		{
			val = ((uchar*)(src->imageData + src->widthStep*y))[x];
			val = pow((val - low2)/err_in, avg_gray/120)*err_out;
			if(val>255)
				val=255;
			if(val<0)
				val=0; // Make sure src is in the range [low,high]
			((uchar*)(dst->imageData + dst->widthStep*y))[x] = (uchar) val;
		}
	}

	return 0;
}

void dis_hsv(double* hsv_feature, IplImage* img)
{
	IplImage* hsv = cvCreateImage(cvGetSize(img), 8, 3);
	cvCvtColor(img, hsv, CV_BGR2HSV);
	
	unsigned int h_temp, s_temp, v_temp, l;
	int i, j;
	unsigned char *ptr = (unsigned char *)hsv->imageData;

	for(i = 0; i < 20; i++)
		hsv_feature[i] = 0;

	for(i = 0; i < img->height; i++)
		for(j = 0; j < img->width; j++)
		{
			h_temp = ptr[i*img->widthStep + j*img->nChannels];
			s_temp = ptr[i*img->widthStep + j*img->nChannels + 1];
			v_temp = ptr[i*img->widthStep + j*img->nChannels + 2];
			double s, v;
			h_temp = h_temp * 2;
			s = s_temp / 255.0;
			v = v_temp / 255.0;
			if(h_temp <= 25)
				h_temp = 0;
			else if(h_temp <= 41)
				h_temp = 1;
			else if(h_temp <= 75)
				h_temp = 2;
			else if(h_temp <= 156)
				h_temp = 3;
			else if(h_temp <= 201)
				h_temp = 4;
			else if(h_temp <= 272)
				h_temp = 5;
			else if(h_temp <= 285)
				h_temp = 6;
			else if(h_temp <= 330)
				h_temp = 7;
			else 
				h_temp = 0;

			if(v <= 0.15)
				l = 16;
			else if(s > 0.1 && v > 0.15)
			{
				if(s < 0.65)
					l = 2*h_temp;
				else
					l = 2*h_temp + 1;
			}
			else if(v <= 0.65)
				l = 17;
			else if(v <= 0.9)
				l = 18;
			else
				l = 19;

			hsv_feature[l]++;
		}
	for(i = 0; i < 20; i++)
		hsv_feature[i] /= (img->width * img->height * 1.0);
}

void cvSkinHSV(IplImage* src,IplImage* dst)
{
	IplImage* hsv=cvCreateImage(cvGetSize(src),8,3);   
    cvCvtColor(src,hsv,CV_BGR2HSV);  
  
    static const int V=2;  
    static const int S=1;  
    static const int H=0;  
  
    cvZero(dst);  
  
    for (int h=0;h<src->height;h++) {  
        unsigned char* phsv=(unsigned char*)hsv->imageData+h*hsv->widthStep;  
        unsigned char* pdst=(unsigned char*)dst->imageData+h*dst->widthStep;  
        for (int w=0;w<src->width;w++) {  
            if (phsv[H]>=7&&phsv[H]<=29)  
            {  
				*pdst=255;
            }  
            phsv+=3;  
            pdst++;  
        }  
    }
}

void cvTongueRGB(IplImage* src,IplImage* dst)
{
	static const int R=2;
    static const int G=1;
    static const int B=0;
  
    cvZero(dst);

	double gate;
    for (int h=0;h<src->height;h++) {  
        unsigned char* prgb=(unsigned char*)src->imageData+h*src->widthStep;  
        unsigned char* pdst=(unsigned char*)dst->imageData+h*dst->widthStep;  
        for (int w=0;w<src->width;w++) {  
			gate=(prgb[R]-prgb[G])/255.0 + (prgb[B]-prgb[G])*6/255.0 + (prgb[R]+prgb[G]+prgb[B])/3/255.0;
			if(gate < 0.527){
				*pdst=255;
			}
            prgb+=3;
            pdst++;
        }
    }
}

void cvThresholdOtsu(IplImage* src, IplImage* dst)  
{  
    int height=src->height;  
    int width=src->width;  
  
    //histogram  
    float histogram[256]={0};  
    for(int i=0;i<height;i++) {  
        unsigned char* p=(unsigned char*)src->imageData+src->widthStep*i;  
        for(int j=0;j<width;j++) {  
            histogram[*p++]++;  
        }  
    }  
    //normalize histogram  
    int size=height*width;  
    for(int i=0;i<256;i++) {  
        histogram[i]=histogram[i]/size;  
    }  
  
    //average pixel value  
    float avgValue=0;  
    for(int i=0;i<256;i++) {  
        avgValue+=i*histogram[i];  
    }  
  
    int threshold;    
    float maxVariance=0;  
    float w=0,u=0;  
    for(int i=0;i<256;i++) {  
        w+=histogram[i];  
        u+=i*histogram[i];  
  
        float t=avgValue*w-u;  
        float variance=t*t/(w*(1-w));  
        if(variance>maxVariance) {  
            maxVariance=variance;  
            threshold=i;  
        }  
    }  
  
    cvThreshold(src,dst,threshold,255,CV_THRESH_BINARY);  
}  

void cvSkinOtsu(IplImage* src, IplImage* dst)  
{  
    assert(dst->nChannels==1&& src->nChannels==3);  
  
    IplImage* ycrcb=cvCreateImage(cvGetSize(src),8,3);  
    IplImage* cr=cvCreateImage(cvGetSize(src),8,1);  
    cvCvtColor(src,ycrcb,CV_BGR2YCrCb);  
    cvSplit(ycrcb,0,cr,0,0);  
  
    cvThresholdOtsu(cr,cr);  
    cvCopyImage(cr,dst);  
    cvReleaseImage(&cr);  
    cvReleaseImage(&ycrcb);  
}

void cvSkinYUV(IplImage* src,IplImage* dst)  
{  
    IplImage* ycrcb=cvCreateImage(cvGetSize(src),8,3);  
    //IplImage* cr=cvCreateImage(cvGetSize(src),8,1);  
    //IplImage* cb=cvCreateImage(cvGetSize(src),8,1);  
    cvCvtColor(src,ycrcb,CV_BGR2YCrCb);  
    //cvSplit(ycrcb,0,cr,cb,0);  
  
    static const int Cb=2;  
    static const int Cr=1;  
    static const int Y=0;  
  
    //IplImage* dst=cvCreateImage(cvGetSize(_dst),8,3);  
    cvZero(dst);  
  
    for (int h=0;h<src->height;h++) {  
        unsigned char* pycrcb=(unsigned char*)ycrcb->imageData+h*ycrcb->widthStep;  
        unsigned char* psrc=(unsigned char*)src->imageData+h*src->widthStep;  
        unsigned char* pdst=(unsigned char*)dst->imageData+h*dst->widthStep;  
        for (int w=0;w<src->width;w++) {  
            if (pycrcb[Cr]>=133&&pycrcb[Cr]<=173&&pycrcb[Cb]>=77&&pycrcb[Cb]<=127)  
            {  
				*pdst=255;
            }  
            pycrcb+=3;  
            psrc+=3;  
            pdst++;  
        }  
    }  
    //cvCopyImage(dst,_dst);  
    //cvReleaseImage(&dst);  
}

int cvJudgeRect(IplImage* img, CvRect rect)
{
	IplImage* hsv = cvCreateImage(cvGetSize(img), 8, 3);
	cvCvtColor(img, hsv, CV_BGR2HSV);
	int width = img->width;
	int height = img->height;
	int r_width = rect.width;
	int r_height = rect.height;

	CvPoint p_o = cvPoint(width/2, height/2);
	CvPoint p_rect = cvPoint(rect.x+r_width/2, rect.y+r_height/2);
	double d_rect = (p_o.x-p_rect.x)*(p_o.x-p_rect.x)+(p_o.y-p_rect.y)*(p_o.y-p_rect.y);
	d_rect = sqrt(d_rect);

	double scale = r_width*r_height*1.0 / (width*height);

	double r_scale = 0;
	if(r_width < r_height)
		r_scale = r_width*1.0 / r_height;
	else 
		r_scale = r_height*1.0 / r_width;

	unsigned int avg_h = 0, sum_h = 0, count = 0;
	for(int h=rect.y;h<rect.y+rect.height;h++){  
		unsigned char* phsv=(unsigned char*)hsv->imageData+h*hsv->widthStep;  
		for(int w=rect.x;w<rect.x+rect.width;w++){
			sum_h += phsv[3*w];
			count++;
		}
	}
	avg_h = sum_h / count;
	cvReleaseImage(&hsv);

	int score = 0;
	
	if(scale < 0.1){
		if(r_scale < 0.3)
			score = 1;
		else if(r_scale < 0.6)
			score = 1;
		else if(r_scale < 0.8)
			score = 2;
		else
			score = 3;
	}
	else if(scale < 0.3){
		if(r_scale < 0.3)
			score = 1;
		else if(r_scale < 0.6)
			score = 2;
		else if(r_scale < 0.8){
			if(avg_h < 75)
				score = 3;
			else if(avg_h < 120)
				score = 4;
			else if(avg_h < 150)
				score = 5;
			else
				score = 4;
		}
		else{
			if(avg_h < 75)
				score = 4;
			else if(avg_h < 120)
				score = 5;
			else if(avg_h < 150)
				score = 6;
			else
				score = 4;
		}
	}
	else if(scale < 0.5){
		if(r_scale < 0.3)
			score = 1;
		else if(r_scale < 0.6)
			score = 2;
		else if(r_scale < 0.8){
			if(avg_h < 75)
				score = 3;
			else if(avg_h < 120)
				score = 4;
			else if(avg_h < 150)
				score = 5;
			else
				score = 4;
		}
		else{
			if(avg_h < 75)
				score = 3;
			else if(avg_h < 120)
				score = 4;
			else if(avg_h < 150)
				score = 5;
			else
				score = 4;
		}
	}
	else{
		if(r_scale < 0.3)
			score = 1;
		else if(r_scale < 0.6)
			score = 1;
		else if(r_scale < 0.8)
			score = 2;
		else
			score = 2;
	}

	if(d_rect > 200)
		score -= 2;
	else if(d_rect > 150)
		score -= 1;
	else if(d_rect >60)
		score += 0;
	else if(d_rect > 20)
		score += 1;
	else
		score += 2;
	return score;
}

IplImage* cvPretreatment(double* hu_feature, IplImage* img)
{
	IplImage* smooth = cvCreateImage(cvGetSize(img), 8, 3);
	cvSmooth(img, smooth, CV_GAUSSIAN, 5, 5);

	IplImage* temp = cvCreateImage(cvGetSize(img), 8, 1);
	IplConvKernel* kernel = cvCreateStructuringElementEx(11, 11, 5, 5, CV_SHAPE_ELLIPSE);

	IplImage* skin = cvCreateImage(cvGetSize(img), 8, 1);
	cvSkinHSV(smooth, skin);
	IplImage* open_skin = cvCreateImage(cvGetSize(img), 8, 1);
	cvMorphologyEx(skin, open_skin, temp, kernel, CV_MOP_CLOSE, 1);
	cvNot(open_skin, open_skin);

	IplImage* t_rgb = cvCreateImage(cvGetSize(img), 8, 1);
	cvTongueRGB(smooth, t_rgb);
	IplImage* open_rgb = cvCreateImage(cvGetSize(img), 8, 1);
	cvMorphologyEx(t_rgb, open_rgb, temp, kernel, CV_MOP_CLOSE, 1);
	cvNot(open_rgb, open_rgb);

	IplImage* otsu = cvCreateImage(cvGetSize(img), 8, 1);
	cvSkinOtsu(smooth, otsu);
	IplImage* open_otsu = cvCreateImage(cvGetSize(img), 8, 1);
	cvMorphologyEx(otsu, open_otsu, temp, kernel, CV_MOP_CLOSE, 1);
	
	IplImage* a_dst = cvCreateImage(cvGetSize(img), 8, 1);
	cvAnd(open_skin, open_otsu, a_dst);
	cvAnd(a_dst, open_rgb, a_dst);
	IplImage* erode_dst = cvCreateImage(cvGetSize(img), 8, 1);
	IplImage* dilate_dst = cvCreateImage(cvGetSize(img), 8, 1);
	cvErode(a_dst, erode_dst, kernel, 2);
	cvDilate(erode_dst, dilate_dst, kernel, 2);

	cvReleaseImage(&smooth);
	cvReleaseImage(&temp);
	cvReleaseStructuringElement(&kernel);
	cvReleaseImage(&skin);
	cvReleaseImage(&open_skin);
	cvReleaseImage(&t_rgb);
	cvReleaseImage(&open_rgb);
	cvReleaseImage(&otsu);
	cvReleaseImage(&open_otsu);
	cvReleaseImage(&a_dst);
	cvReleaseImage(&erode_dst);

	CvMemStorage * storage = cvCreateMemStorage(0);
	CvSeq* contour = 0;
	CvSeq* max = 0;
	int Nc = cvFindContours(dilate_dst, storage, &contour, sizeof(CvContour), CV_RETR_EXTERNAL);
	int score = 0;
	for(CvSeq* c=contour; c!=NULL; c=c->h_next){
		CvRect temp = ((CvContour*)c)->rect;
		if(1.0*temp.height*temp.width/(img->width*img->height) > 0.02){
			int temp_score = cvJudgeRect(img, temp);
			if(temp_score > score){
				score = temp_score;
				max = c;
			}
			if(temp_score == score && max){
				CvRect temp_max = ((CvContour*)max)->rect;
				int temp_1 = (img->width-temp.x-temp.width/2) + (img->height-temp.y-temp.height/2);
				int temp_2 = (img->width-temp_max.x-temp_max.width/2) + (img->height-temp_max.y-temp_max.height/2);
				if(temp_1 < temp_2)
					max = c;
			}
		}
	}

	cvZero(dilate_dst);
	cvDrawContours(dilate_dst, max, cvScalarAll(255), cvScalarAll(255), -1);

	CvMoments moments;
	CvHuMoments huMoments;
	cvMoments(dilate_dst, &moments, 1);
	cvGetHuMoments(&moments, &huMoments);

	int i;
	for(i = 0; i < 7; i++)
	{
		hu_feature[i] = ((double*)&huMoments)[i];
	}
	cvReleaseImage(&dilate_dst);

	CvRect rect;
	IplImage* dst;
	if(max)
	{
		rect = ((CvContour*)max)->rect;
		cvSetImageROI(img, rect);
		dst = cvCreateImage(cvSize(rect.width, rect.height), 8, 3);
		cvCopy(img, dst, 0);
		cvResetImageROI(img);
	}
	else{
		dst = cvCreateImage(cvGetSize(img), 8, 3);
		cvCopy(img, dst, 0);
	}
	return dst;
}

/*
获取hu矩。
void get_hu(double* hu_feature, IplImage* img)
{
	IplImage* dst_crotsu = cvCreateImage(cvGetSize(img), 8, 1);
	cvSkinOtsu(img,dst_crotsu);

	IplImage* hsv = cvCreateImage(cvGetSize(img), 8, 3);
	cvCvtColor(img, hsv, CV_BGR2HSV);

	IplImage* g_hsv = cvCreateImage(cvGetSize(img), 8, 1);
	cvCvtColor(hsv, g_hsv, CV_BGR2GRAY);

	IplImage* smooth_g_hsv = cvCreateImage(cvGetSize(img), 8, 1);
	cvSmooth(g_hsv, smooth_g_hsv, CV_MEDIAN, 5);

	cvAdaptiveThreshold(smooth_g_hsv, smooth_g_hsv, 255);
	cvNot(smooth_g_hsv, smooth_g_hsv);

	IplImage* dst = cvCreateImage(cvGetSize(img), 8, 1);
	cvAnd(smooth_g_hsv, dst_crotsu, dst);

	CvMoments moments;
	CvHuMoments huMoments;
	cvMoments(dst, &moments, 1);
	cvGetHuMoments(&moments, &huMoments);

	int i;
	for(i = 0; i < 7; i++)
	{
		hu_feature[i] = ((double*)&huMoments)[i];
	}
	
	cvReleaseImage(&hsv);
	cvReleaseImage(&g_hsv);
	cvReleaseImage(&smooth_g_hsv);
	cvReleaseImage(&dst_crotsu);
	cvReleaseImage(&dst);
}

直方图均衡
void cvEqual(IplImage* src, IplImage* dst)
{
	int i;
	IplImage* pImageChannel[4] = {0,0,0,0};
	if(src){
		for(i = 0; i < src->nChannels; i++){
			pImageChannel[i] = cvCreateImage(cvGetSize(src), 8, 1);
		}
		cvSplit(src, pImageChannel[0], pImageChannel[1], pImageChannel[2], pImageChannel[3]);
		for(i = 0; i < src->nChannels; i++){
			cvEqualizeHist(pImageChannel[i], pImageChannel[i]);
		}
		cvMerge(pImageChannel[0], pImageChannel[1], pImageChannel[2], pImageChannel[3], dst);
		for(i = 0; i < src->nChannels; i++){
			if(pImageChannel[i]){
				cvReleaseImage(&pImageChannel[i]);
				pImageChannel[i] = 0;
			}
		}
	}
}
*/

int calGLCM(IplImage* bWavelet, int angleDirection, double* feature)
{
	int width, height;
	int i, j;
	if(bWavelet == NULL)
		return -1;
	width = bWavelet->width;
	height = bWavelet->height;

	int* glcm = new int[GLCM_CLASS * GLCM_CLASS];
	int* histImage = new int[width * height];

	if(glcm == NULL || histImage == NULL)
		return -2;

	//灰度等级化
	uchar* data = (uchar*)bWavelet->imageData;
	for(i = 0; i < height; i++)
		for(j = 0; j < width; j++)
			histImage[i*width+j] = (int)(data[bWavelet->widthStep*i+j] * GLCM_CLASS / 256);

	for(i = 0; i < GLCM_CLASS; i++)
		for(int j = 0; j < GLCM_CLASS; j++)
			glcm[i*GLCM_CLASS+j] = 0;

	int k, l, h, w;
	int all_sum = 0;
	if(angleDirection == GLCM_ANGLE_HORIZATION)
	{
		for(h = 0; h < height; h++)
			for (w = 0; w < width; w++)
			{
				l = histImage[h*width+w];
				if(w + GLCM_DIS >= 0 && w + GLCM_DIS < width)
				{
					k = histImage[h*width+w+GLCM_DIS];
					glcm[l*GLCM_CLASS+k]++;
					all_sum++;
				}
				if(w - GLCM_DIS >= 0 && w - GLCM_DIS < width)
				{
					k = histImage[h*width+w-GLCM_DIS];
					glcm[l*GLCM_CLASS+k]++;
					all_sum++;
				}
			}
	}
	
	else if(angleDirection == GLCM_ANGLE_VERTICAL)
	{
		for(h = 0; h < height; h++)
			for (w = 0; w < width; w++)
			{
				l = histImage[h*width+w];
				if(h + GLCM_DIS >= 0 && h + GLCM_DIS < height)
				{
					k = histImage[(h+GLCM_DIS)*width+w];
					glcm[l*GLCM_CLASS+k]++;
					all_sum++;
				}
				if(h - GLCM_DIS >= 0 && h - GLCM_DIS < height)
				{
					k = histImage[(h-GLCM_DIS)*width+w];
					glcm[l*GLCM_CLASS+k]++;
					all_sum++;
				}
			}
	}
	else if(angleDirection == GLCM_ANGLE_DIGONAL)
	{
		for(h = 0; h < height; h++)
			for (w = 0; w < width; w++)
			{
				l = histImage[h*width+w];
				if(w + GLCM_DIS >= 0 && w + GLCM_DIS < width && h + GLCM_DIS >= 0 && h + GLCM_DIS < height)
				{
					k = histImage[(h+GLCM_DIS)*width+w+GLCM_DIS];
					glcm[l*GLCM_CLASS+k]++;
					all_sum++;
				}
				if(w - GLCM_DIS >= 0 && w - GLCM_DIS < width && h - GLCM_DIS >= 0 && h - GLCM_DIS < height)
				{
					k = histImage[(h-GLCM_DIS)*width+w-GLCM_DIS];
					glcm[l*GLCM_CLASS+k]++;
					all_sum++;
				}
			}
	}
	else
		return -3;

	double entropy = 0, energy = 0, contrast = 0, homogenity = 0, correlation = 0;
	double dSum[65], dMean = 0, dStdDev = 0;
	memset(dSum, 0, sizeof(dSum));
	for(i = 0; i < GLCM_CLASS; i++)
		for(j = 0; j < GLCM_CLASS; j++)
		{
			if(glcm[i*GLCM_CLASS+j] != 0)
			{
				double dValue = (double)glcm[i*GLCM_CLASS+j] / (double)all_sum;
				entropy -= (dValue * log(dValue));

				energy += (dValue * dValue);
	
				contrast += ((i-j) * (i-j) * dValue);

				homogenity += (dValue / (1 + (i-j) * (i-j)));

				correlation += (i * j * dValue);
				dSum[i] += dValue;
			}
		}
	for(i = 0; i < GLCM_CLASS; i++)
		dMean += (i * dSum[i]);
	for(i = 0; i < GLCM_CLASS; i++)
		dStdDev += ((i-dMean) * (i-dMean) * dSum[i]);
	if(abs(dStdDev) > 1e-15)
		correlation = (correlation - dMean*dMean) / dStdDev;
	else
		correlation = 0;

	i = 0;
	feature[i++] = entropy;
	feature[i++] = energy;
	feature[i++] = contrast;
	feature[i++] = homogenity;
	feature[i++] = correlation;

	//cout << energy << " " << contrast << " " << homogenity  << " " << entropy  << endl;

	delete[] glcm;
	delete[] histImage;
	return 0;
}

void img_resize(IplImage* img, string full_path)
{
	int width, height;
	double zoom_pro;
	CvSize cvSize;
	width = img->width;
	height = img->height;
	if(height > width && width > 480)
	{
		zoom_pro = 480.0 / width;
		cvSize.width = width * zoom_pro;
		cvSize.height = height * zoom_pro;
	}
	else if(height <= width && height > 480)
	{
		zoom_pro = 480.0 / height;
		cvSize.width = width * zoom_pro;
		cvSize.height = height * zoom_pro;
	}
	else
		return;
	IplImage* r_img = cvCreateImage(cvSize, 8, 3);
	cvResize(img, r_img, CV_INTER_LINEAR);

	cvSaveImage(full_path.c_str(), r_img);
	cvReleaseImage(&r_img);
}

void img_gamma(IplImage* img, IplImage* adj_img)
{
	IplImage* img_r = cvCreateImage(cvGetSize(img), 8, 1);
	IplImage* img_g = cvCreateImage(cvGetSize(img), 8, 1);
	IplImage* img_b = cvCreateImage(cvGetSize(img), 8, 1);
	cvSplit(img, img_b, img_g, img_r, NULL);
	
	IplImage* adj_r = cvCreateImage(cvGetSize(img), 8, 1);
	IplImage* adj_g = cvCreateImage(cvGetSize(img), 8, 1);
	IplImage* adj_b = cvCreateImage(cvGetSize(img), 8, 1);
	ImageAdjust(img_r, adj_r, 0, 1, 0, 1, 1);
	ImageAdjust(img_g, adj_g, 0, 1, 0, 1, 1);
	ImageAdjust(img_b, adj_b, 0, 1, 0, 1, 1);

	cvMerge(adj_b, adj_g, adj_r, NULL, adj_img);
	cvReleaseImage(&img_r);
	cvReleaseImage(&img_g);
	cvReleaseImage(&img_b);
	cvReleaseImage(&adj_r);
	cvReleaseImage(&adj_g);
	cvReleaseImage(&adj_b);
}

void get_rand(int n, int arr[])
{
	srand((unsigned)time(NULL));
	int i, p, tmp;
	for(i = 0; i < n; i++)
		arr[i] = i;
	for(i = n-1; i > 0; i--)
	{
		p = rand() % i + 1;
		tmp = arr[p];
		arr[p] = arr[i];
		arr[i] = tmp;
	}
}

int get_data_tmp()
{
	char buf[2048][1024];
	ifstream in_pos, in_neg;
	ofstream outfile;
	in_pos.open("pos.data");
	in_neg.open("neg.data");
	outfile.open("tongue_train_tmp.data");

	if(!in_neg || !in_pos || !outfile)
	{
		cout << "error in open file" << endl;
		return 1;
	}
	
	int num = 0;
	if (in_neg.is_open() && in_pos.is_open())
	{
		while(in_pos.good() && !in_pos.eof())
		{
			memset(buf[num], 0, 1024);
			in_pos.getline(buf[num], 1024);
			num ++;
		}
		num --;

		while(in_neg.good() && !in_neg.eof())
		{
			memset(buf[num], 0, 1024);
			in_neg.getline(buf[num], 1024);
			num ++;
		}
	}
	
	if (outfile.is_open())
	{
		int i;
		for(i = 0; i < num - 2; i++)
			outfile << buf[i] << endl;
		outfile << buf[i];
	}

	in_pos.close();
	in_neg.close();
	outfile.close();

	cout << "done" << endl;

	getchar();
	return 0;
}

int disorder()
{
	char buf[2048][1024];
	ifstream infile;
	ofstream outfile;
	infile.open("tongue_train_tmp.data");
	outfile.open("tongue_train.data");

	if(!infile || !outfile)
	{
		cout << "error in open file" << endl;
		return 1;
	}
	int num = 0;
	if (infile.is_open())
	{
		while(infile.good() && !infile.eof())
		{
			memset(buf[num], 0, 1024);
			infile.getline(buf[num], 1024);
			num ++;
		}
	}

	int *a = new int[num];
	get_rand(num, a);
	
	if (outfile.is_open())
	{
		for(int i = 0; i < num; i++)
			outfile << buf[a[i]] << endl;
	}

	infile.close();
	outfile.close();

	cout << "done" << endl;
	getchar();
	return 0;
}

static int
read_num_class_data( const char* filename, int var_count,
                     CvMat** data, CvMat** responses )
{
    const int M = 1024;
    FILE* f = fopen( filename, "rt" );
    CvMemStorage* storage;
    CvSeq* seq;
    char buf[M+2];
    float* el_ptr;
    CvSeqReader reader;
    int i, j;

    if( !f )
        return 0;

    el_ptr = new float[var_count+1];
    storage = cvCreateMemStorage();
    seq = cvCreateSeq( 0, sizeof(*seq), (var_count+1)*sizeof(float), storage );

    for(;;)
    {
        char* ptr;
        if( !fgets( buf, M, f ) || !strchr( buf, ',' ) )
            break;
        el_ptr[0] = buf[0];
        ptr = buf+2;
        for( i = 1; i <= var_count; i++ )
        {
            int n = 0;
            sscanf( ptr, "%f%n", el_ptr + i, &n );
            ptr += n + 1;
        }
        if( i <= var_count )
            break;
        cvSeqPush( seq, el_ptr );
    }
    fclose(f);

    *data = cvCreateMat( seq->total, var_count, CV_32F );
    *responses = cvCreateMat( seq->total, 1, CV_32F );

    cvStartReadSeq( seq, &reader );

    for( i = 0; i < seq->total; i++ )
    {
        const float* sdata = (float*)reader.ptr + 1;
        float* ddata = data[0]->data.fl + var_count*i;
        float* dr = responses[0]->data.fl + i;

        for( j = 0; j < var_count; j++ )
            ddata[j] = sdata[j];
        *dr = sdata[-1];
        CV_NEXT_SEQ_ELEM( seq->elem_size, reader );
    }

    cvReleaseMemStorage( &storage );
    delete el_ptr;
    return 1;
}

static
int build_rtrees_classifier( char* data_filename,
    char* filename_to_save, char* filename_to_load )
{
    CvMat* data = 0;
    CvMat* responses = 0;
    CvMat* var_type = 0;
    CvMat* sample_idx = 0;

    int ok = read_num_class_data( data_filename, VAR_COUNT, &data, &responses );
    int nsamples_all = 0, ntrain_samples = 0;
    int i = 0;
    double train_hr = 0, test_hr = 0;
    CvRTrees forest;
    CvMat* var_importance = 0;

    if( !ok )
    {
        printf( "Could not read the database %s\n", data_filename );
        return -1;
    }

    printf( "The database %s is loaded.\n", data_filename );
    nsamples_all = data->rows;
    ntrain_samples = (int)(nsamples_all*0.8);

    // Create or load Random Trees classifier
    if( filename_to_load )
    {
        // load classifier from the specified file
        forest.load( filename_to_load );
        ntrain_samples = 0;
        if( forest.get_tree_count() == 0 )
        {
            printf( "Could not read the classifier %s\n", filename_to_load );
            return -1;
        }
        printf( "The classifier %s is loaded.\n", data_filename );
    }
    else
    {
        // create classifier by using <data> and <responses>
        printf( "Training the classifier ...\n");

        // 1. create type mask
        var_type = cvCreateMat( data->cols + 1, 1, CV_8U );
        cvSet( var_type, cvScalarAll(CV_VAR_ORDERED) );
        cvSetReal1D( var_type, data->cols, CV_VAR_CATEGORICAL );

        // 2. create sample_idx
        sample_idx = cvCreateMat( 1, nsamples_all, CV_8UC1 );
        {
            CvMat mat;
            cvGetCols( sample_idx, &mat, 0, ntrain_samples );
            cvSet( &mat, cvRealScalar(1) );

            cvGetCols( sample_idx, &mat, ntrain_samples, nsamples_all );
            cvSetZero( &mat );
        }

        // 3. train classifier
        forest.train( data, CV_ROW_SAMPLE, responses, 0, sample_idx, var_type, 0,
            CvRTParams(25,5,0,false,15,0,true,20,100,0.01f,CV_TERMCRIT_ITER));
        printf( "\n");
    }

    // compute prediction error on train and test data
    for( i = 0; i < nsamples_all; i++ )
    {
        double r;
        CvMat sample;
        cvGetRow( data, &sample, i );

        r = forest.predict( &sample );
        r = fabs((double)r - responses->data.fl[i]) <= FLT_EPSILON ? 1 : 0;

        if( i < ntrain_samples )
            train_hr += r;
        else
            test_hr += r;
    }

    test_hr /= (double)(nsamples_all-ntrain_samples);
    train_hr /= (double)ntrain_samples;
    printf( "Recognition rate: train = %.1f%%, test = %.1f%%\n",
            train_hr*100., test_hr*100. );

    printf( "Number of trees: %d\n", forest.get_tree_count() );

    // Print variable importance
    var_importance = (CvMat*)forest.get_var_importance();
    if( var_importance )
    {
        double rt_imp_sum = cvSum( var_importance ).val[0];
        printf("var#\timportance (in %%):\n");
        for( i = 0; i < var_importance->cols; i++ )
            printf( "%-2d\t%-4.1f\n", i,
            100.f*var_importance->data.fl[i]/rt_imp_sum);
    }

    //Print some proximitites
    /*printf( "Proximities between some samples corresponding to the letter 'T':\n" );
    {
        CvMat sample1, sample2;
        const int pairs[][2] = {{0,103}, {0,106}, {106,103}, {-1,-1}};

        for( i = 0; pairs[i][0] >= 0; i++ )
        {
            cvGetRow( data, &sample1, pairs[i][0] );
            cvGetRow( data, &sample2, pairs[i][1] );
            printf( "proximity(%d,%d) = %.1f%%\n", pairs[i][0], pairs[i][1],
                forest.get_proximity( &sample1, &sample2 )*100. );
        }
    }*/

    // Save Random Trees classifier to file if needed
    if( filename_to_save )
        forest.save( filename_to_save );

    cvReleaseMat( &sample_idx );
    cvReleaseMat( &var_type );
    cvReleaseMat( &data );
    cvReleaseMat( &responses );

    return 0;
}

int CvRtree_Train()
{
	char* filename_to_save = "tongue_recog.xml";
    char* filename_to_load = 0;
    char default_data_filename[] = "./tongue_train.data";
	char* data_filename = default_data_filename;

	build_rtrees_classifier(data_filename, filename_to_save, filename_to_load);

	return 0;
}

int CvRtree_Predict()
{
	char* filename_load = "tongue_recog.xml";
	char* data_file = "test.data";
	CvRTrees forest;
	forest.load(filename_load);
	if(forest.get_tree_count() == 0)
	{
		printf("error!\n");
		return -1;
	}
	printf("The classifier %s is loaded. \n", filename_load);

	CvMat* data = 0;
	CvMat* response = 0;
	int	nsamples_all = 0;
	
	int ok = read_num_class_data(data_file, VAR_COUNT, &data, &response);
	if(!ok)
	{
		printf("error in read data %s\n", data_file);
		return -1;
	}
	printf("The data %s is loaded. \n", data_file);
	nsamples_all = data->rows;

	int error_sum = 0;
	for(int i = 0; i < nsamples_all; i++)
	{
		double r;
		CvMat sample;
		cvGetRow(data, &sample, i);

		r = forest.predict(&sample);
		printf("%d: r = %c, response = %c\n", i+1, (int)r, (int)response->data.fl[i]);
		if(fabs(r - response->data.fl[i]) > 0.00001)
			error_sum++;
	}

	printf("The error detection of tongue_image：%d\n", error_sum);

	return 0;
}

int get_feature(int method, bool flag)  //flag判断是否要调整图片size
{
	if(method == -1)
		return 0;
	string p_path = "./POS/";
	char* infile = "pos.txt";
	char* outfile = "pos.data";
	if(method == 1)
	{
		p_path = "./NEG/";
		infile = "neg.txt";
		outfile = "neg.data";
	}
	if(method == 2)
	{
		p_path = "./TEST/";
		infile = "test.txt";
		outfile = "test.data";
	}
	ifstream in(infile);
	ofstream testout(outfile);

	int num = 0;
	for(string file_name; getline(in, file_name); )
	{
		string full_path = p_path + file_name;
		cout << full_path << endl;
		
		IplImage* img = cvLoadImage(full_path.c_str(), 1);
		if(!img)
		{
			cout << "error in LoadImage: " << full_path << endl;
			return -1;
		}

		if(img->width > 480 && flag)
		{
			img_resize(img, full_path);
			img = cvLoadImage(full_path.c_str(), 1);
		}

		IplImage* adj_img = cvCreateImage(cvGetSize(img), 8, 3);
		img_gamma(img, adj_img);
		cvSaveImage("test.jpg", adj_img);

		double* hu_feature = new double[7];
		IplImage* n_img = cvPretreatment(hu_feature, adj_img);
		
		double* hsv_feature = new double[20];
		dis_hsv(hsv_feature, n_img);

		IplImage* img_gray = cvCreateImage(cvGetSize(n_img), 8, 1);
		cvCvtColor(n_img, img_gray, CV_BGR2GRAY);
		double* horization_feature = new double[5];
		double* vertical_feature = new double[5];
		double* digonal_feature = new double[5];
		calGLCM(img_gray, GLCM_ANGLE_HORIZATION, horization_feature);
		calGLCM(img_gray, GLCM_ANGLE_VERTICAL, vertical_feature);
		calGLCM(img_gray, GLCM_ANGLE_DIGONAL, digonal_feature);

		cvShowImage("test", img_gray);
		waitKey(0);

		cvReleaseImage(&img);
		cvReleaseImage(&adj_img);
		cvReleaseImage(&img_gray);
		cvReleaseImage(&n_img);

		all_feature.clear();
		if(method == 1)
			all_feature.push_back(0);
		else
			all_feature.push_back(1);

		int i;
		for(i = 0; i < 20; i++)
			all_feature.push_back(hsv_feature[i]);
		for(i = 0; i < 7; i++)
			all_feature.push_back(hu_feature[i]);
		for(i = 0; i < 5; i++)
			all_feature.push_back(horization_feature[i]);
		for(i = 0; i < 5; i++)
			all_feature.push_back(vertical_feature[i]);
		for(i = 0; i < 5; i++)
			all_feature.push_back(digonal_feature[i]);

		vector<double>::iterator it;
		for(it=all_feature.begin(); it != all_feature.end()-1; ++it)
		{
			testout << *it << ",";
		}
		testout << *it << endl;
		num++;
	}
	cout << num << endl;

	in.close();
	testout.close();

	return 0;
}

int main(int argc, char *argv[])
{
	int i;
	int method = -1;
	bool flag_resize = false;
	bool flag_data = false;
	bool flag_train = false;
	bool flag_predict = false;

	for(i = 1; i < argc; i++)
	{
		if(strcmp(argv[i], "-method") == 0)
		{
			i++;
			method = atoi(argv[i]);
		}
		else if(strcmp(argv[i], "-resize") == 0)
		{
			flag_resize = true;
		}
		else if(strcmp(argv[i], "-getdata") == 0)
		{
			flag_data = true;
		}
		else if(strcmp(argv[i], "-train") == 0)
		{
			flag_train = true;
		}
		else if(strcmp(argv[i], "-predict") == 0)
		{
			flag_predict = true;
		}
	}

	//get_feature(method, flag_resize);
	get_feature(2, false);
	if(flag_data)
	{
		get_data_tmp();
		disorder();
	}
	if(flag_train)
		CvRtree_Train();
	if(flag_predict)
		CvRtree_Predict();

	getchar();
	return 0;
}