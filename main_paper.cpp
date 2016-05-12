#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <ctime>
#include <cstdio>
#include <cmath>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/core/core_c.h>
#include <opencv2/ml/ml.hpp>
#include <opencv2/legacy/legacy.hpp>

using namespace std;
using namespace cv;

#define VAR_COUNT 45
#define INF 2147483647

vector<double> all_feature;
vector<Point> point_poly;
vector<Point> cv_polyCorrect();
int cv_logPolar(Mat& img, Mat& dst, int method);
int cv_colorClear(Mat& img, Mat& dst);

#pragma region gamma_v
int cv_gamma(Mat& img, Mat& dst){  //v通道gamma变换 8UC3 to 8UC3
	Mat hsv(img.size(), CV_8UC3);
	cvtColor(img, hsv, CV_BGR2HSV);
	vector<Mat> channels;
	
	split(hsv, channels);
	Mat hsv_v = channels.at(2);

	Scalar avg = mean(img);
	double avg_gray = avg.val[0];
	double gamma = avg_gray / 120.0;

	unsigned char lut[256];
	for(int i = 0; i < 256; i++){
		lut[i] = saturate_cast<uchar>(pow((double)(i/255.0), gamma) * 255.0f);
	}

	int nc;
	if(hsv_v.isContinuous()){
		nc = hsv_v.rows * hsv_v.cols * hsv_v.channels();
	}
	uchar* data = hsv_v.ptr<uchar>(0);
	for(int i = 0; i < nc; i++){
		data[i] = lut[data[i]];
	}
	
	merge(channels, hsv);
	cvtColor(hsv, dst, CV_HSV2BGR);

	return 0;
}
#pragma endregion

#pragma region GVF
int cv_GVF(Mat& img, Mat& dst, double alpha, double mu, int iter){
	//梯度矢量流 8UC3 to 8UC1  0.5, 0.5, 20
	if(alpha <= 0 || mu <= 0){
		cout << "error in para" << endl;
		return -1;
	}
	Mat gray(img.size(), CV_8UC1);
	cvtColor(img, gray, CV_BGR2GRAY);
	gray.convertTo(gray, CV_32FC1);

	//vector<Mat> channels;
	//split(img, channels);
	//Mat gray = channels.at(2);

	double minVal = 0, maxVal = 0;
	minMaxIdx(gray, &minVal, &maxVal);
	gray = gray - minVal;
	//gray = (gray - minVal) / (maxVal - minVal) * 255.0;

	Mat ones = Mat::ones(img.size(), CV_32F);
	multiply(gray, ones, gray, 1.0f/(maxVal-minVal));
	multiply(gray, ones, gray, 255.0);

	Mat img_x(img.size(), CV_32FC1);
	Mat img_y(img.size(), CV_32FC1);
	Sobel(gray, img_x, -1, 1, 0);
	Sobel(gray, img_y, -1, 0, 1);
	
	//img_x /= 255.0;
	//img_y /= 255.0;
	multiply(img_x, ones, img_x, 1.0/255.0);
	multiply(img_y, ones, img_y, 1.0/255.0);

	Mat SqrMat = Mat::zeros(img.size(), CV_32F);
	accumulateSquare(SqrMat, img_x);
	accumulateSquare(SqrMat, img_y);

	Mat u1,v1;
	img_x.copyTo(u1);
	img_y.copyTo(v1);

	Mat del2u1 = Mat::zeros(u1.size(), CV_32F);
	Mat del2v1 = Mat::zeros(v1.size(), CV_32F);

	Mat temp = Mat::zeros(v1.size(), CV_32F);
	Mat temp_2 = Mat::zeros(v1.size(), CV_32F);

	for(int i = 0; i < iter; i++){
		Laplacian(u1, del2u1, -1);
		Laplacian(v1, del2v1, -1);

		subtract(u1, img_x, temp);
		multiply(temp, SqrMat, temp);
		temp_2 = mu * del2u1;
		subtract(temp_2, temp, temp);
		scaleAdd(temp, alpha, u1, u1);

		subtract(v1, img_y, temp);
		multiply(temp, SqrMat, temp);
		temp_2 = mu * del2v1;
		subtract(temp_2, temp, temp);
		scaleAdd(temp, alpha, v1, v1);
	}
	
	addWeighted(u1, 0.5, v1, 0.5, 0, temp);
	//temp *= 255.0;
	multiply(temp, ones, temp, 255.0);
	temp.convertTo(dst, CV_8UC1);

	return 0;
}
#pragma endregion

#pragma region watershed
int cv_watershed(Mat& img, Mat& gvf, Mat& dst){
	//原图和梯度矢量图的分水岭变换 dst:8UC1
	Mat gvf_c;
	gvf.copyTo(gvf_c);
	int compCount = 0;
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(gvf_c, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

	if(contours.empty()) return -1;
	Mat markers = Mat::zeros(img.size(), CV_32S);

	int idx = 0;
	for(; idx >= 0; idx = hierarchy[idx][0], compCount++ )
		drawContours(markers, contours, idx, Scalar::all(compCount+1), -1, 8, hierarchy, INT_MAX);

	if(compCount == 0) return -1;

	Mat hsv;
	cvtColor(img, hsv, CV_BGR2HSV);
	watershed(hsv, markers);

	Mat wshed(img.size(), CV_8UC3);
	vector<Vec3b> colorTab;
    for(int i = 0; i < compCount; i++ )
    {
        int b = theRNG().uniform(0, 255);
        int g = theRNG().uniform(0, 255);
        int r = theRNG().uniform(0, 255);

        colorTab.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
    }
	for(int i = 0; i < markers.rows; i++ )
	{
		for(int j = 0; j < markers.cols; j++ )
		{
			int index = markers.at<int>(i,j);
			if( index == -1 ){
				img.at<Vec3b>(i,j) = Vec3b(255,255,255);
				wshed.at<Vec3b>(i,j) = Vec3b(255,255,255);
			}
			else if( index <= 0 || index > compCount )
				wshed.at<Vec3b>(i,j) = Vec3b(0,0,0);
			else
				//wshed.at<Vec3b>(i,j) = img.at<Vec3b>(i,j);
				wshed.at<Vec3b>(i,j) = colorTab[index - 1]; 
		}
	}

	imshow("wshed", wshed);
	return 0;
}

int cv_resize(Mat& img, Mat& dst){
	//尺寸调整
	int size = img.cols > img.rows ? img.cols : img.rows; 
	double scale = 1.0;
	if(size > 640){
		scale = 640.0 / size;
	}
	resize(img, dst, Size(), scale, scale);

	return 0;
}
#pragma endregion

#pragma region tonthresh
int cv_skin(Mat& img, Mat& dst){
	//h通道肤色提取
	Mat hsv;
	cvtColor(img, hsv, CV_BGR2HSV);

	vector<Mat> channels;
	split(hsv, channels);
	Mat hsv_h = channels.at(0);
	
	int nc;
	if(hsv_h.isContinuous()){
		nc = hsv_h.rows * hsv_h.cols * hsv_h.channels();
	}
	uchar* data_h = hsv_h.ptr<uchar>(0);
	uchar* data_dst = dst.ptr<uchar>(0);
	for(int i = 0; i < nc; i++){
		if(data_h[i] >= 7 && data_h[i] <= 29)
			data_dst[i] = 255;
	}

	return 0;
}

int cv_tonRGB(Mat& img, Mat& dst){
	//RGB三色差值
	int nr = img.rows;
	int nc = img.cols * img.channels();
	
	if(img.isContinuous()){
		nr = 1;
		nc = nc * img.rows;
	}
	
	for(int i = 0; i < nr; i++){
		uchar* data = img.ptr<uchar>(i);
		uchar* data_dst = dst.ptr<uchar>(i);
		double gate;

		for(int j = 0; j < nc; j += 3){
			gate = (data[j+2]-data[j+1])/255.0 + (data[j]-data[j+1])*6/255.0 + (data[j+2]+data[j+1]+data[j])/255.0/3.0;
			if(gate < 0.627)
				*data_dst = 255;
			data_dst++;
		}
	}

	return 0;
}

int cv_rgb2gray(Mat& img, Mat& dst){
	int nr = img.rows;
	int nc = img.cols * img.channels();
	
	if(img.isContinuous()){
		nr = 1;
		nc = nc * img.rows;
	}
	
	for(int i = 0; i < nr; i++){
		uchar* data = img.ptr<uchar>(i);
		uchar* data_dst = dst.ptr<uchar>(i);
		double gray;

		for(int j = 0; j < nc; j += 3){
			gray = 0.5 * data[j + 2] + 0.3 * data[j + 1] + 0.2 * data[j];
			*data_dst = (unsigned int)gray;
			data_dst++;
		}
	}

	return 0;
}

int cv_otsu(Mat& img, Mat& dst){
	//最大类间差
	float histogram[256] = {0};
	int nc;
	if(img.isContinuous()){
		nc = img.rows * img.cols * img.channels();
	}
	uchar* data = img.ptr<uchar>(0);
	for(int i = 0; i < nc; i++){
		histogram[data[i]]++;
	}

    for(int i = 0; i < 256; i++){
        histogram[i] = histogram[i] / nc;
    }

	float avgValue = 0;
	for(int i = 0; i < 256; i++){
		avgValue += i * histogram[i];
	}

	int thresh = 0;
	float maxVariance = 0;
	float w = 0, u = 0;
	for(int i = 0; i < 256; i++){
		w += histogram[i];
		u += i * histogram[i];

		float t = avgValue * w - u;
		float variance = t * t / (w * (1 - w));
		if(variance > maxVariance){
			maxVariance = variance;
			thresh = i;
		}
	}

	threshold(img, dst, thresh, 255, CV_THRESH_BINARY);

	return 0;
}

int cv_MaxEntropy(Mat& img, Mat& dst){
	//最大熵, 没卵用
	float histogram[256] = {0};
	int nc;
	if(img.isContinuous()){
		nc = img.rows * img.cols * img.channels();
	}
	uchar* data = img.ptr<uchar>(0);
	for(int i = 0; i < nc; i++){
		histogram[data[i]]++;
	}

	float total[256] = {0};
	float tmp = 0;
	for(int i = 0; i < 256; i++){
		tmp += histogram[i];
		total[i] = tmp;
	}

	int total_back = 0, total_obj = 0;
	float entropy_back = 0, entropy_obj = 0;

	int thresh = 0;
	float maxVariance = 0;
	for(int i = 0; i < 256; i++){
		float percentage = histogram[i] / total[i];
		if(percentage - 0.0 > 0.000001)
			entropy_back += -percentage * log(percentage);

		entropy_obj = 0;
		for(int j = i + 1; j < 256; j++){
			percentage = histogram[j] / (nc - total[i]);
			if(percentage - 0.0 > 0.000001)
				entropy_obj += -percentage * log(percentage);
		}

		if((entropy_back + entropy_obj) > maxVariance){
			maxVariance = entropy_back + entropy_obj;
			thresh = i;
		}
	}

	threshold(img, dst, thresh, 255, CV_THRESH_BINARY);

	return 0;
}

int cv_minErr(Mat& img, Mat& dst){
	//最小误差法，不适用，适用于车牌等分界明显的提取
	float histogram[256] = {0};
	int nc;
	if(img.isContinuous()){
		nc = img.rows * img.cols * img.channels();
	}
	uchar* data = img.ptr<uchar>(0);
	for(int i = 0; i < nc; i++){
		histogram[data[i]]++;
	}

    for(int i = 0; i < 256; i++){
        histogram[i] = histogram[i] / nc;
    }

	float avgValue = 0;
	for(int i = 0; i < 256; i++){
		avgValue += i * histogram[i];
	}

	int thresh = 0;
	float minErr = 0x7FFFFFFF;
	float p = 0, u = 0;
	float variance_0, variance_1;
	for(int i = 0; i < 256; i++){
		p += histogram[i];
		u += i * histogram[i];
		if(p - 0 < 0.000001) continue;

		variance_0 = 0;
		for(int k = 0; k < i; k++){
			float t = k - u / p;
			variance_0 += t * t * histogram[k];
		}
		variance_0 /= p;

		variance_1 = 0;
		for(int j = i + 1; j < 256; j++){
			float t = j - (avgValue - u) / (1 - p);
			variance_1 += t * t * histogram[j];
		}
		variance_1 /= (1 - p);

		if(variance_0 - 0 < 0.000001 || variance_1 - 0 < 0.000001) continue;
		float curErr = 1 + 2 * (p * log(sqrt(variance_0)) + (1 - p) * log(sqrt(variance_1))) - 2 * (p * log(p) + (1 - p) * log(1 - p));
		if(curErr < minErr)
		{
			minErr = curErr;
			thresh = i;
		}
	}

	cout << thresh << endl;
	threshold(img, dst, thresh, 255, CV_THRESH_BINARY);

	return 0;
}

int cv_adaptive(Mat& img, Mat& dst){
	adaptiveThreshold(img, dst, 255, CV_ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 51, 9);
	return 0;
}
#pragma endregion

#pragma region formal_tonRegion
int cv_tonRegion(Mat& img, Mat& dst){
	//舌象提取
	Mat median_blur;
	medianBlur(img, median_blur, 5);

	Mat kernel = getStructuringElement(CV_SHAPE_ELLIPSE, Size(11, 11));

	Mat skin = Mat::zeros(img.size(), CV_8UC1);
	cv_skin(median_blur, skin);
	Mat mor_skin;
	morphologyEx(skin, mor_skin, MORPH_CLOSE, kernel);
	bitwise_not(mor_skin, mor_skin);
	//imshow("mor_skin", mor_skin);

	Mat tonRGB = Mat::zeros(img.size(), CV_8UC1);
	cv_tonRGB(median_blur, tonRGB);
	Mat mor_tonRGB;
	morphologyEx(tonRGB, mor_tonRGB, MORPH_CLOSE, kernel);
	bitwise_not(mor_tonRGB, mor_tonRGB);
	//imshow("mor_tonRGB", mor_tonRGB);

	Mat img_gray_t = Mat::zeros(img.size(), CV_8UC1);
	cv_rgb2gray(median_blur, img_gray_t);
	Mat entropy = Mat::zeros(img.size(), CV_8UC1);
	cv_adaptive(img_gray_t, entropy);
	//Mat mor_entropy;
	//morphologyEx(entropy, mor_entropy, MORPH_CLOSE, kernel);
	//imshow("mor_entropy", entropy);

	Mat otsu_polar = Mat::zeros(img.size(), CV_8UC1);
	Mat polar;
	cv_logPolar(median_blur, polar, 1);
	Mat img_gray = Mat::zeros(img.size(), CV_8UC1);
	cv_rgb2gray(polar, img_gray);
	cv_otsu(img_gray, otsu_polar);

	cv_logPolar(otsu_polar, otsu_polar, 2);
	threshold(otsu_polar, otsu_polar, 155, 255, CV_THRESH_BINARY);
	Mat mor_otsu_polar;
	morphologyEx(otsu_polar, mor_otsu_polar, MORPH_CLOSE, kernel);
	//imshow("mor_otsu_polar", mor_otsu_polar);

	Mat hsv;
	cvtColor(median_blur, hsv, CV_BGR2HSV);
	vector<Mat> channels;
	split(hsv, channels);
	Mat hsv_h = channels.at(0);
	Mat hsv_v = channels.at(2);

	Mat otsu_h = Mat::zeros(img.size(), CV_8UC1);
	cv_otsu(hsv_h, otsu_h);
	Mat mor_otsu_h;
	morphologyEx(otsu_h, mor_otsu_h, MORPH_CLOSE, kernel);
	//imshow("mor_otsu_h", mor_otsu_h);

	Mat otsu_v = Mat::zeros(img.size(), CV_8UC1);
	cv_otsu(hsv_v, otsu_v);
	Mat mor_otsu_v;
	morphologyEx(otsu_v, mor_otsu_v, MORPH_CLOSE, kernel);
	//imshow("mor_otsu_v", mor_otsu_v);

	Mat mor_otsu_hv;
	bitwise_and(mor_otsu_v, mor_otsu_h, mor_otsu_hv);
	//imshow("mor_otsu_hv", mor_otsu_hv);
	
	//bitwise_and(mor_skin, mor_tonRGB, dst);
	//bitwise_and(dst, mor_otsu_polar, dst);
	Mat dst_tmp;
	dst_tmp = 0.4 * mor_tonRGB + 0.2 * mor_skin + 0.2 * mor_otsu_polar + 0.2 * mor_otsu_hv;
	bitwise_and(dst_tmp, entropy, dst_tmp);
	//imshow("dst_tmp", dst_tmp);
	threshold(dst_tmp, dst, 0.7*255, 255, CV_THRESH_BINARY);

	erode(dst, dst, kernel, Point(-1, -1), 2);
	dilate(dst, dst, kernel, Point(-1, -1), 2);
	//imshow("dst", dst);
	return 0;
}
#pragma endregion

#pragma region polar_tonRegion
//int cv_tonRegion(Mat& img, Mat& dst){
//	//舌象提取
//	Mat median_blur;
//	medianBlur(img, median_blur, 5);
//
//	Mat polar;
//	cv_logPolar(median_blur, polar, 1);
//	//imshow("polar", polar);
//
//	Mat kernel = getStructuringElement(CV_SHAPE_ELLIPSE, Size(11, 11));
//
//	Mat skin = Mat::zeros(img.size(), CV_8UC1);
//	cv_skin(polar, skin);
//	Mat mor_skin;
//	morphologyEx(skin, mor_skin, MORPH_CLOSE, kernel);
//	bitwise_not(mor_skin, mor_skin);
//	//imshow("mor_skin", mor_skin);
//
//	Mat tonRGB = Mat::zeros(img.size(), CV_8UC1);
//	cv_tonRGB(polar, tonRGB);
//	Mat mor_tonRGB;
//	morphologyEx(tonRGB, mor_tonRGB, MORPH_CLOSE, kernel);
//	bitwise_not(mor_tonRGB, mor_tonRGB);
//	//imshow("mor_tonRGB", mor_tonRGB);
//
//	Mat otsu_polar = Mat::zeros(img.size(), CV_8UC1);
//	cv_otsu(polar, otsu_polar);
//	Mat mor_otsu_polar;
//	morphologyEx(otsu_polar, mor_otsu_polar, MORPH_CLOSE, kernel);
//	//imshow("mor_otsu_polar", mor_otsu_polar);
//
//	bitwise_and(mor_skin, mor_tonRGB, dst);
//	bitwise_and(dst, mor_otsu_polar, dst);
//	//imshow("dst1", dst);
//
//	erode(dst, dst, kernel, Point(-1, -1), 2);
//	dilate(dst, dst, kernel, Point(-1, -1), 2);
//	cv_logPolar(dst, dst, 2);
//	threshold(dst, dst, 155, 255, CV_THRESH_BINARY);
//	//imshow("dst", dst);
//	return 0;
//}
#pragma endregion

#pragma region judge
int cv_judge(double area, double radio, double dis, double match){  
	//判断舌象的连通域
	int score = 0;

	//return (1/(abs(0.5-area) * (1-radio) * log10(dis) * match));

	if(area < 0.02) return -1;
	
	if(area < 0.1){
		if(radio < 0.6) score = 1;
		else if(radio < 0.8) score = 2;
		else score = 3;
	}
	else if(area < 0.3){
		if(radio < 0.3) score = 1;
		else if(radio < 0.6) score = 2;
		else score = 3;
	}
	else if(area < 0.5){
		if(radio < 0.3) score = 1;
		else if(radio < 0.6) score = 2;
		else score = 4;
	}
	else{
		if(radio < 0.6) score = 1;
		else score = 2;
	}

	if(dis < 20) score += 2;
	else if(dis < 60) score += 1;
	else if(dis > 200) score -= 2;
	else if(dis > 150) score -= 1;

	if(match < 0.1) score += 3;
	else if(match < 0.5) score += 2;
	else if(match > 20) score -= 2;
	else if(match > 10) score -= 1;

	return score;
}
#pragma endregion

int ialpha = 10;
int ibeta = 20;
int igamma = 20;

#pragma region snake
void cv_snake(int pos, void* usrdata){//Mat& img, Mat& hull){
	vector<Mat> param = *(vector<Mat> *)usrdata;
	Mat img = param[0].clone();
	Mat hull = param[1];
	vector<vector<Point>> pp;
	pp.push_back(hull);
	drawContours(img, pp, -1, Scalar(0, 0, 255), 2);

	vector<Point> temp = pp.at(0);
	int length = temp.size();
	CvPoint* point = new CvPoint[length];
	for(int i = 0; i < length; i++)
	{
		point[i] = temp.at(i);
	}
	float alpha = ialpha / 100.0;
	float beta = ibeta / 100.0;
	float gamma = igamma / 100.0;
	CvSize size;
	size.width = 3;
	size.height = 3;
	CvTermCriteria criteria;
	criteria.type = CV_TERMCRIT_ITER;
	criteria.max_iter = 1000; 
	criteria.epsilon = 0.1;

	IplImage* I_img = new IplImage(img);
	IplImage* gray = cvCreateImage(cvGetSize(I_img), 8, 1);
	cvCvtColor(I_img, gray, CV_BGR2GRAY);
	cvSnakeImage(gray, point, length, &alpha, &beta, &gamma, CV_VALUE, size, criteria, 0);
	
	for(int i=0;i<length;i++)
	{
		int j = (i+1)%length;
		line( img, point[i],point[j], Scalar(0, 255,0), 2 ); 
	}
	imshow("snake", img);
	
	return;
}
#pragma endregion

#pragma region dealwithregion
int cv_findCon(Mat& img, Mat& img_feature, Mat& img_mask){
	Mat dst = Mat::zeros(img.size(), CV_8UC1);
	Mat old_dst = Mat::zeros(img.size(), CV_8UC1);
	cv_tonRegion(img, dst);
	imshow("dst", dst);
	imwrite("dst.jpg", dst);

	Mat mode = imread("mode.jpg", 0);
	threshold(mode, mode, 127, 255, CV_THRESH_BINARY);

	vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
	vector<Point> hull, hull_mode;
	Point img_cen = Point(img.cols / 2, img.rows / 2);

	findContours(mode, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
	if(contours.empty()) return -1;
	//approxPolyDP(Mat(contours[0]), poly_mode, 5, true);
	convexHull(Mat(contours[0]), hull_mode);

	contours.clear();
	hierarchy.clear();
	int idx = 0, compCount = 0;

	findContours(dst, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
	if(contours.size() == 0){
		img.copyTo(img_feature);
		return -1;
	}
	
	Mat temp = Mat::zeros(img.size(), CV_8UC1);
	int max_score = -1, comp_record = -1;
	double min_dis = 1024;

	for( ; idx >= 0; idx = hierarchy[idx][0], compCount++ ){
		//approxPolyDP(Mat(contours[compCount]), poly, 5, true);
		convexHull(Mat(contours[compCount]), hull);

		double area = contourArea(hull) / (img.rows * img.cols);

		Rect rect = boundingRect(Mat(contours[compCount]));
		double radio = rect.width*1.0 / rect.height;
		radio = radio < 1 ? radio : (1.0 / radio);

		Moments mom = moments(Mat(contours[compCount]));
		Point centroid = Point(mom.m10 / mom.m00, mom.m01 / mom.m00);
		double dis = (img_cen.x-centroid.x)*(img_cen.x-centroid.x) + (img_cen.y-centroid.y)*(img_cen.y-centroid.y);
		dis = sqrt(dis);

		double match = matchShapes(hull, hull_mode, 1, 0);

		int score = cv_judge(area, radio, dis, match);

		if(score > 0 && score > max_score){
			max_score = score;
			min_dis = dis;
			comp_record = compCount;
		}
		else if(score > 0 && score == max_score){
			if(dis < min_dis){
				min_dis = dis;
				comp_record = compCount;
			}
		}
	}

	Rect roi;
	Moments moment;
	double *hu = new double[7];
	if(comp_record != -1){
		Mat hull_mom;
		convexHull(Mat(contours[comp_record]), hull_mom);
		roi = boundingRect(Mat(contours[comp_record]));
		//moment = moments(hull_mom);
		//HuMoments(moment, hu);

		//for(int i = 0; i < 7; i++){
		//	all_feature.push_back(hu[i]);
		//}

		//Mat poly;
		//approxPolyDP(Mat(contours[comp_record]), point_poly, 2, true);

		/*Point centroid = Point(moment.m10 / moment.m00, moment.m01 / moment.m00);
		Mat polar;
		cv_logPolar(img, polar, centroid, 1);
		imshow("polar", polar);*/

		/*for(int i = 0; i < point_poly.size(); i++){
			char buf[10];
			sprintf(buf, "%d", i);
			string b = buf;
			putText(img, b, point_poly[i], FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 255));
		}*/

		//vector<Point> point_dst = cv_polyCorrect();

		vector<vector<Point>> pp;
		//pp.push_back(point_poly);
		pp.push_back(hull_mom);
		
		/*pp.clear();
		pp.push_back(point_dst);
		drawContours(img, pp, -1, Scalar(255,0,0), 2);*/
		vector<Mat> param;
		param.push_back(img);
		param.push_back(hull_mom);
		namedWindow("snake", 0);
		createTrackbar("alpha", "snake", &ialpha, 100, cv_snake, &param);
		createTrackbar("beta", "snake", &ibeta, 100, cv_snake, &param);
		createTrackbar("gamma", "snake", &igamma, 100, cv_snake, &param);
		waitKey();
		//cv_snake(img, hull_mom);
		
		//rectangle(img, rr, Scalar::all(255), 2);
		//drawContours(img, pp, -1, Scalar(0,0,255), 2);
		drawContours(old_dst, pp, 0, Scalar(255), CV_FILLED);
		imshow("img", img);
		//imwrite("img.jpg", img);
	}
	else{
		roi.x = 0;
		roi.y = 0;
		roi.width = img.cols;
		roi.height = img.rows;
		//7个0
		//for(int i = 0; i < 7; i++){
		//	all_feature.push_back(0);
		//}
	}
	Mat tmp_roi = img(roi);
	tmp_roi.copyTo(img_feature);

	tmp_roi = old_dst(roi);
	tmp_roi.copyTo(img_mask);

	delete[] hu;
	return 0;
}
#pragma endregion

#pragma region tonfeature
int cv_calcGLCM(Mat& img, Mat& gl, vector<double>& feature){
	double energy = 0, contrast = 0, correlation = 0, IDM = 0, entropy = 0, median = 0;

	Scalar mean, stddev;
	double meanValue, stddevValue;
	meanStdDev(gl, mean, stddev);
	meanValue = mean.val[0];
	stddevValue = stddev.val[0];

	gl = gl + gl.t();
	gl = gl / sum(gl)[0];

	for(int i = 0; i < 256; i++){
		for(int j = 0; j < 256; j++){
			energy += gl.at<float>(i, j) * gl.at<float>(i, j);
			contrast += (i - j) * (i - j) * gl.at<float>(i, j);
			IDM += gl.at<float>(i, j) / ((i - j) * (i - j) + 1);
			if(gl.at<float>(i, j) != 0) entropy -= gl.at<float>(i, j) * log10(gl.at<float>(i, j));
			median += 0.5 * (i * gl.at<float>(i, j) + j * gl.at<float>(i, j));
			correlation += gl.at<float>(i, j) * (i - meanValue) * (j - meanValue) / stddevValue / stddevValue;
		}
	}

	feature.push_back(energy);
	feature.push_back(contrast);
	feature.push_back(IDM);
	feature.push_back(entropy);
	feature.push_back(median);
	feature.push_back(correlation);

	return 0;
}

int cv_glcm(Mat& img, vector<double>& feature, Mat& mask){
	Mat gray;
	cvtColor(img, gray, CV_BGR2GRAY);
	
	int row = gray.rows;
	int col = gray.cols;
	Mat gl = Mat::zeros(256, 256, CV_32FC1);

	//水平
	for(int i = 0; i < row; i++){
		for(int j = 0; j < col-1; j++){
			if(mask.at<uchar>(i, j) < 255 || mask.at<uchar>(i, j+1) < 255) continue;
			gl.at<float>(gray.at<uchar>(i, j), gray.at<uchar>(i, j+1)) += 1;
		}
	}
	cv_calcGLCM(gray, gl, feature);

	//45°
	gl.zeros(256, 256, CV_32FC1);
	for(int i = 0; i < row-1; i++){
		for(int j = 0; j < col-1; j++){
			if(mask.at<uchar>(i, j) < 255 || mask.at<uchar>(i+1, j+1) < 255) continue;
			gl.at<float>(gray.at<uchar>(i, j), gray.at<uchar>(i+1, j+1)) += 1;
		}
	}
	cv_calcGLCM(gray, gl, feature);

	//垂直
	gl.zeros(256, 256, CV_32FC1);
	for(int i = 0; i < row-1; i++){
		for(int j = 0; j < col; j++){
			if(mask.at<uchar>(i, j) < 255 || mask.at<uchar>(i+1, j) < 255) continue;
			gl.at<float>(gray.at<uchar>(i, j), gray.at<uchar>(i+1, j)) += 1;
		}
	}
	cv_calcGLCM(gray, gl, feature);
	
	return 0;
}

int cv_colorFeature(Mat& img, vector<double> &feature, Mat& img_mask){
	Mat hsv;
	cvtColor(img, hsv, CV_BGR2HSV);

	int nc, l;
	if(hsv.isContinuous()){
		nc = hsv.rows * hsv.cols * hsv.channels();
	}
	uchar* data = hsv.ptr<uchar>(0);
	uchar* data_mask = img_mask.ptr<uchar>(0);

	int *hsv_feature = new int[20];
	for(int i = 0; i < 20; i++)
		hsv_feature[i] = 0;

	for(int i = 0; i < nc; i += 3){
		if(data_mask[i/3] < 255) continue;
		int hsv_h = data[i] * 2;
		double hsv_s = data[i+1] / 255.0;
		double hsv_v = data[i+2] / 255.0;

		if(hsv_h <= 25) hsv_h = 0;
		else if(hsv_h <= 41) hsv_h = 1;
		else if(hsv_h <= 75) hsv_h = 2;
		else if(hsv_h <= 156) hsv_h = 3;
		else if(hsv_h <= 201) hsv_h = 4;
		else if(hsv_h <= 272) hsv_h = 5;
		else if(hsv_h <= 285) hsv_h = 6;
		else if(hsv_h <= 330) hsv_h = 7;
		else hsv_h = 0;

		if(hsv_v <= 0.15) l = 16;
		else if(hsv_s > 0.1 && hsv_v > 0.15){
			if(hsv_s < 0.65) l = 2 * hsv_h;
			else l = 2 * hsv_h + 1;
		}
		else if(hsv_v <= 0.65) l = 17;
		else if(hsv_v <= 0.9) l = 18;
		else l = 19;
		
		hsv_feature[l]++;
	}

	int sum = 0;
	for(int i = 0; i < 20; i++)
		sum += hsv_feature[i];
	if(sum == 0){
		for(int i = 0; i < 20; i++)
			feature.push_back(0);
	}
	else{
		for(int i = 0; i < 20; i++)
			feature.push_back(1.0 * hsv_feature[i] / sum);
	}

	delete[] hsv_feature;
	return 0;
}
#pragma endregion

#pragma region randomforest
static int read_num_class_data( const char* filename, int var_count, CvMat** data, CvMat** responses ){
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

static int build_rtrees_classifier( char* data_filename, char* filename_to_save, char* filename_to_load ){
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

int cv_rand(int n, int arr[])
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
	return 0;
}

int cv_dataTmp()
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

int cv_getData()
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
	cv_rand(num, a);
	
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
#pragma endregion

#pragma region gamma_grayworld
int cv_gammaCor(Mat& img, double avg_I, double beta_min, double beta_max, int beta_n){
	unsigned char lut[256];
	unsigned char abs_min;
	Mat dst;

	double* beta = new double[beta_n];
	uchar* data;
	int nc;
	if(img.isContinuous()){
		nc = img.rows * img.cols * img.channels();
	}
	do{
		beta[0] = beta_min;
		beta[beta_n - 1] = beta_max;
		double step = (beta_max-beta_min)/(beta_n-1);
		for(int i = 1; i < beta_n-1; i++){
			beta[i] = beta_min + i * step;
		}

		Mat temp;
		int record_i;
		abs_min = 255;
		for(int i = 0; i < beta_n; i++){
			for(int j = 0; j < 256; j++){
				lut[j] = saturate_cast<uchar>(pow((double)(j/255.0), beta[i]) * 255.0f);
			}

			img.copyTo(temp);
			int avg_channel = 0;

			data = temp.ptr<uchar>(0);
			for(int k = 0; k < nc; k++){
				data[k] = lut[data[k]];
				avg_channel += data[k];
			}
			avg_channel /= nc;

			if(abs(avg_channel - avg_I) < abs_min){
				abs_min = abs(avg_channel - avg_I);
				record_i = i;
			}
		}

		if(abs_min > 1){
			if(record_i == 0) beta_min /= 2;
			else beta_min = beta[record_i-1];
			if(record_i == beta_n - 1) beta_max *= 2;
			else beta_max = beta[record_i+1];
		}
		else{
			for(int j = 0; j < 256; j++){
				lut[j] = saturate_cast<uchar>(pow((double)(j/255.0), beta[record_i]) * 255.0f);
			}
			data = img.ptr<uchar>(0);
			for(int k = 0; k < nc; k++){
				data[k] = lut[data[k]];
			}
			break;
		}
	}while(abs_min > 1);

	delete[] beta;
	return 0;
}

int cv_grayWorld(Mat& img, Mat& dst, int method){ //0：green 1：min 2：max 3：mid 4：改进gamma
	vector<Mat> channels;
	
	split(img, channels);
	Mat rgb_b = channels.at(0);
	Mat rgb_g = channels.at(1);
	Mat rgb_r = channels.at(2);

	Scalar avg = mean(img);
	double avg_b = avg.val[0];
	double avg_g = avg.val[1];
	double avg_r = avg.val[2];

	double alpha_r, alpha_b, alpha_g, avg_min, avg_max, avg_mid, avg_I;

	switch(method){
	case 0:
		alpha_r = avg_g / avg_r;
		alpha_b = avg_g / avg_b;

		rgb_b *= alpha_b;
		rgb_r *= alpha_r;

		merge(channels, dst);
		break;
	case 1:
		avg_min = std::min(std::min(avg_b, avg_g), avg_r);
		alpha_r = avg_min / avg_r;
		alpha_b = avg_min / avg_b;
		alpha_g = avg_min / avg_g;

		rgb_b *= alpha_b;
		rgb_r *= alpha_r;
		rgb_g *= alpha_g;
		merge(channels, dst);
		break;
	case 2:
		avg_max = std::max(std::max(avg_b, avg_g), avg_r);
		alpha_r = avg_max / avg_r;
		alpha_b = avg_max / avg_b;
		alpha_g = avg_max / avg_g;

		rgb_b *= alpha_b;
		rgb_r *= alpha_r;
		rgb_g *= alpha_g;
		merge(channels, dst);
		break;
	case 3:
		avg_mid = (avg_r + avg_b + avg_g) / 3;//std::min(std::max(avg_b, avg_g), avg_r);
		alpha_r = avg_mid / avg_r;
		alpha_b = avg_mid / avg_b;
		alpha_g = avg_mid / avg_g;

		rgb_b *= alpha_b;
		rgb_r *= alpha_r;
		rgb_g *= alpha_g;
		merge(channels, dst);
		break;
	case 4:
		avg_I = (avg_r + avg_b + avg_g) / 3;
		//cout << avg_I << endl;
		//all_feature.push_back(avg_I);
		cv_gammaCor(rgb_b, 0.94*avg_I, 0.5, 2, 5);
		cv_gammaCor(rgb_g, 0.956*avg_I, 0.5, 2, 5);
		cv_gammaCor(rgb_r, 1.09*avg_I, 0.5, 2, 5);
		merge(channels, dst);
		break;
	default:
		break;
	}
	
	return 0;
}
#pragma endregion

#pragma region poly_Correct
double cv_disPoint(Point p1, Point p2){
	double dis = (p2.x-p1.x)*(p2.x-p1.x) + (p2.y-p1.y)*(p2.y-p1.y);
	return std::sqrt(dis);
}

bool cv_signCross(Point p1, Point p2, Point p3){
	double cross = (p2.x - p1.x) * (p3.y - p1.y) - (p2.y - p1.y) * (p3.x - p1.x);
	if(cross > 0) return false;
	else return true;
}

vector<Point> cv_polyCorrect(){
	if(point_poly.empty())
		return point_poly;
	int poly_size = point_poly.size();

	double **lens;
	lens = new double*[poly_size];
	for(int i = 0; i < poly_size; i++){
		lens[i] = new double[poly_size];
	}

	for(int i = 0; i < poly_size; i++){
		int j = (i+1)%poly_size;
		lens[i][j] = 0.0;
		double l = cv_disPoint(point_poly[i], point_poly[j]); 
		for(int k = (j+1)%poly_size; k != i; k = (j+1)%poly_size){
			l += cv_disPoint(point_poly[j], point_poly[k]);
			double t = cv_disPoint(point_poly[i], point_poly[k]);
			lens[i][k] = (l*t);
			j = k;
		}
	}

	double minWaste = INF;
	double initArea = contourArea(point_poly);
	vector<Point> dst_poly;

	for(int i = 0; i < poly_size; i++){
		double **dp;
		dp = new double*[poly_size];
		for(int init = 0; init < poly_size; init++){
			dp[init] = new double[poly_size];
		}
		for(int init = 0; init < poly_size; init++){
			for(int initj = 0; initj < poly_size; initj++){
				dp[init][initj] = INF;
			}
		}

		int **s;
		s = new int*[poly_size];
		for(int init = 0; init < poly_size; init++){
			s[init] = new int[poly_size];
		}

		vector<Point> ps;

		for(int t = 0; t < poly_size; t++){
			dp[i][t] = lens[i][t];
			s[i][t] = i;
		}
		int j = (i+1)%poly_size;
		int s1, s2;
		double min = INF;

		for(j = (j+1)%poly_size; j != i; j = (j+1)%poly_size){
			if(point_poly[j].y < point_poly[i].y) continue;
			for(int k = (i+1)%poly_size; k != j; k = (k+1)%poly_size){
				if(point_poly[k].y < point_poly[i].y) continue;
				if(lens[k][j] >= minWaste) continue;
				if(!cv_signCross(point_poly[i], point_poly[k], point_poly[j])) continue;
				for(int l = i; l != k; l = (l+1)%poly_size){
					if(point_poly[l].y < point_poly[i].y) continue;
					if(dp[l][k] == INF) continue;
					if(cv_signCross(point_poly[l], point_poly[k], point_poly[j])){
						if(dp[l][k] + lens[k][j] < dp[k][j]){
							dp[k][j] = dp[l][k] + lens[k][j];
							s[k][j] = l;
						}
					}
				}
				if(dp[k][j] != INF){
					if(cv_signCross(point_poly[k], point_poly[j], point_poly[i])){
						if(dp[k][j] + lens[j][i] < min){
							s1 = k;
							s2 = j;
							min = dp[k][j] + lens[j][i];
						}
					}
				}
			}// for k
		}// for j

		if(min == INF) continue;

		while(s2 != i){
			ps.push_back(point_poly[s2]);
			int t = s1;
			s1 = s[s1][s2];
			s2 = t;
		}

		double area = contourArea(ps) / initArea;
		if(area < 0.3) continue;

		if(min >= minWaste || ps.size() <= 4) continue;

		//ps.push_back(point_poly[i]);
		dst_poly = ps;
		minWaste = min;

		for(int init = 0; init < poly_size; init++){
			delete[] dp[init];
		}
		delete[] dp;

		for(int init = 0; init < poly_size; init++){
			delete[] s[init];
		}
		delete[] s;
	}


	for(int i = 0; i < poly_size; i++){
		delete[] lens[i];
	}
	delete[] lens;
	return dst_poly;
}
#pragma endregion

#pragma region usingotherspaper
int cv_hisbalance(Mat& img, Mat& dst){	//学姐论文
	vector<Mat> channels(img.channels());
	split(img, channels);

	for(int i = 0; i < img.channels(); i++)
		equalizeHist(channels[i], channels[i]);

	merge(channels, dst);
	return 0;
}

int cv_otsu2(Mat& img, Mat& dst){		//学姐论文
	vector<Mat> channels(img.channels());
	split(img, channels);

	float histogram[256] = {0};
	int nc;
	if(img.isContinuous()){
		nc = img.rows * img.cols * img.channels();
	}

	uchar* data = img.ptr<uchar>(0);
	for(int i = 0; i < nc; i++){
		histogram[data[i]]++;
	}

    for(int i = 0; i < 256; i++){
        histogram[i] = histogram[i] / nc;
    }

	float avgValue = 0;
	for(int i = 0; i < 256; i++){
		avgValue += i * histogram[i];
	}

	int thresh = 0;
	float maxVariance = 0;
	float w = 0, u = 0;
	for(int i = 0; i < 256; i++){
		w += histogram[i];
		u += i * histogram[i];

		float t = avgValue * w - u;
		float variance = t * t / (w * (1 - w));
		if(variance > maxVariance){
			maxVariance = variance;
			thresh = i;
		}
	}

	uchar* data_img = img.ptr<uchar>(0);
	uchar* data_dst = dst.ptr<uchar>(0);
	for(int i = 0; i < nc; i += 3){
		if((data_img[i]+data_img[i+1]+data_img[i+2]) / 3 < (thresh-18) && data_img[i+2]<=140 || data_img[i+2]>=160)
			*data_dst = 0;
		else
			*data_dst = 255;
		data_dst++;
	}

	return 0;
}

int cv_huecut(Mat& img, Mat& dst){
	Mat hsv;
	cvtColor(img, hsv, CV_BGR2HSV);

	vector<Mat> channels;
	split(hsv, channels);
	Mat h = channels.at(0);

	int nc;
	if(h.isContinuous()){
		nc = h.rows * h.cols * h.channels();
	}
	uchar* data = h.ptr<uchar>(0);
	uchar* data_dst = dst.ptr<uchar>(0);
	for(int i = 0; i < nc; i++){
		if(data[i] < 95)
			*data_dst = 0;
		else
			*data_dst = 255;
		data_dst++;
	}
	
	return 0;
}

int cv_tonRegion2(Mat& img, Mat& dst){	//学姐论文
	Mat hisbalance;
	cv_hisbalance(img, hisbalance);

	Mat kernel = getStructuringElement(CV_SHAPE_ELLIPSE, Size(11, 11));

	Mat hue = Mat::zeros(img.size(), CV_8UC1);
	cv_huecut(hisbalance, hue);
	Mat mor_hue;
	morphologyEx(hue, mor_hue, MORPH_OPEN, kernel);
	imshow("hue", mor_hue);

	Mat tonRGB = Mat::zeros(img.size(), CV_8UC1);
	cv_tonRGB(hisbalance, tonRGB);
	Mat mor_tonRGB;
	morphologyEx(tonRGB, mor_tonRGB, MORPH_CLOSE, kernel);
	bitwise_not(mor_tonRGB, mor_tonRGB);
	imshow("mor_tonRGB2", mor_tonRGB);

	Mat otsu2 = Mat::zeros(img.size(), CV_8UC1);
	cv_otsu(img, otsu2);
	Mat mor_otsu;
	morphologyEx(otsu2, mor_otsu, MORPH_OPEN, kernel);
	imshow("mor_otsu2", mor_otsu);

	int nc;
	if(img.isContinuous()){
		nc = img.rows * img.cols;
	}
	uchar* data_hue = mor_hue.ptr<uchar>(0);
	uchar* data_rgb = mor_tonRGB.ptr<uchar>(0);
	uchar* data_otsu = mor_otsu.ptr<uchar>(0);
	uchar* data_dst = dst.ptr<uchar>(0);
	for(int i = 0; i < nc; i++){
		if(data_hue[i] == 0 || data_rgb[i] == 0 || data_otsu[i] == 0)
			*data_dst = 255;
		else 
			*data_dst = 0;
		*data_dst++;
	}

	return 0;
}
#pragma endregion

#pragma region formal2polar
int cv_logPolar(Mat& img, Mat& dst, int method){
	IplImage* I_img = new IplImage(img);
	IplImage* I_dst = cvCreateImage(cvGetSize(I_img), 8, img.channels());

	// method: 1 cartToPolar  2 polarToCart
	switch(method){
	case 1:
		cvLogPolar(I_img, I_dst, cvPoint2D32f(I_img->width/2, I_img->height/2), I_img->width/6, CV_INTER_LINEAR | CV_WARP_FILL_OUTLIERS);
		break;
	case 2:
		cvLogPolar(I_img, I_dst, cvPoint2D32f(I_img->width/2, I_img->height/2), I_img->width/6, CV_INTER_LINEAR | CV_WARP_FILL_OUTLIERS | CV_WARP_INVERSE_MAP);
		break;
	default:
		return 1;
	}
	
	dst = Mat(I_dst);

	/*Mat gray;
	cvtColor(dst, gray, CV_BGR2HSV);
	vector<Mat> channels;
	split(gray, channels);
	Mat img_x(dst.size(), CV_8UC1);
	Sobel(channels[0], img_x, -1, 1, 0);
	imshow("img_x", img_x);*/

	return 0;
}
#pragma endregion

double cv_calcCos(vector<double>& feature,vector<double>& stdfeature)
{
	double dis = 0;
	/*for(int i = 0; i < feature.size(); i++)
	{
		feature[i] = log10(feature[i]);
		len += feature[i] * feature[i];
	}
	len = sqrt(len);
	for(int i = 0; i < feature.size(); i++)
	{
		dis += stdfeature[i] * feature[i] / len;
	}*/
	for(int i = 0; i < feature.size(); i++)
	{
		if(feature[i] == 0 || stdfeature[i] == 0) dis += abs(feature[i]) + abs(stdfeature[i]);
		else dis += abs(feature[i] - stdfeature[i]) / (abs(feature[i]) + abs(stdfeature[i]));
	}
	return dis;
}

#pragma region LQ_color_correct
double cv_tcdDeltaE(Mat &img1, Mat &img2)
{
	Mat lab1, lab2;
	cvtColor(img1, lab1, CV_BGR2Lab);
	cvtColor(img2, lab2, CV_BGR2Lab);

	vector<Mat> channels1, channels2;
	split(lab1, channels1);
	split(lab2, channels2);
	Mat l1, a1, b1, l2, a2, b2, tmp_l, tmp_a, tmp_b;
	l1 = channels1.at(0);
	a1 = channels1.at(1);
	b1 = channels1.at(2);
	l2 = channels2.at(0);
	a2 = channels2.at(1);
	b2 = channels2.at(2);

	subtract(l1, l2, tmp_l);
	subtract(a1, a2, tmp_a);
	subtract(b1, b2, tmp_b);

	Mat mull = Mat::ones(tmp_l.size(), CV_32F);
	Mat mula = Mat::ones(tmp_a.size(), CV_32F);
	Mat mulb = Mat::ones(tmp_b.size(), CV_32F);
	Mat addall = Mat::ones(tmp_l.size(), CV_32F);
	tmp_l.convertTo(tmp_l, CV_32FC1);
	tmp_a.convertTo(tmp_a, CV_32FC1);
	tmp_b.convertTo(tmp_b, CV_32FC1);
	multiply(tmp_l, tmp_l, mull);
	multiply(tmp_a, tmp_a, mula);
	multiply(tmp_b, tmp_b, mulb);

	addall = mull + mula + mulb;
	Mat delta_e = Mat::ones(addall.size(), CV_32F);
	sqrt(addall, delta_e);
	Scalar result = mean(delta_e);
	cout << result.val[0] << endl;
	return 0;
}
#pragma endregion

int cv_dcstretch(Mat& src, Mat& dst)
{
	vector<Mat> channels;
	split(src, channels);
	/*Mat rgb_b = channels.at(0).t();
	Mat rgb_g = channels.at(1).t();
	Mat rgb_r = channels.at(2).t();
	Mat rgb_all = rgb_r.clone();
	rgb_all.push_back(rgb_g);
	rgb_all.push_back(rgb_b);*/

	int row_rgb = src.rows, col_rgb = src.cols;
	Mat matrixA(3, row_rgb*col_rgb, CV_8UC1);
	int col_a = matrixA.cols, row_a = matrixA.rows;
	for(int i = 0; i < row_a; i++)
	{
		uchar* Ma = matrixA.ptr<uchar>(i);
		for(int j = 0; j < col_a; j++)
		{
			//Ma[j] = rgb_all.at<uchar>(pos/col_rgb, pos%col_rgb);
			Ma[j] = channels.at(i).at<uchar>(j % row_rgb, j / row_rgb);
		}
	}

	Mat covar, mean;
	calcCovarMatrix(matrixA.t(), covar, mean, CV_COVAR_NORMAL | CV_COVAR_ROWS);
	Mat evalue, evector;
	eigen(covar, evalue, evector);

	Mat evector_tmp = evector.clone();
	Mat matrixT;
	pow(evalue, -1/2, evalue);
	for(int i = 0; i < evector.rows; i++){
		evector.row(i) *= evalue.at<double>(i);
	}

	matrixT = evector.t() * evector_tmp;

	matrixA.convertTo(matrixA, CV_64FC1);
	matrixA = matrixA.t();
	Mat result;

	result = matrixA * matrixT;
	result = result.t();

	result.convertTo(result, CV_8UC1);
	for(int i = 0; i < row_a; i++)
	{
		uchar* Ma = result.ptr<uchar>(i);
		for(int j = 0; j < col_a; j++)
		{
			channels.at(i).at<uchar>(j % row_rgb, j / row_rgb) = Ma[j];	
		}
	}
	merge(channels, dst);
	return 0;
}

Mat dstretch(Mat& input, Mat& targetMean = Mat(), Mat& targetSigma = Mat())
{
   CV_Assert(input.channels() > 1);
 
   Mat dataMu, dataSigma, eigDataSigma, scale, stretch;

   Mat data = input.reshape(1, input.rows*input.cols);
   cout << data.rows << " " << data.cols << endl;
   /*
   data stored as rows. 
   if x(i) = [xi1 xi2 .. xik]' is vector representing an input point, 
   data is now an N x k matrix:
   data = [x(1)' ; x(2)' ; .. ; x(N)']
   */

   // take the mean and standard deviation of input data
   meanStdDev(input, dataMu, dataSigma);

   /*
   perform PCA that gives us an eigenspace.
   eigenvectors matrix (R) lets us project input data into the new eigenspace.
   square root of eigenvalues gives us the standard deviation of projected data.
   */
   PCA pca(data, Mat(), CV_PCA_DATA_AS_ROW);

   /*
   prepare scaling (Sc) and strecthing (St) matrices.
   we use the relation var(a.X) = a^2.var(X) for a random variable X and 
   set
   scaling factor a = 1/(sigma of X) for diagonal entries of scaling matrix.
   stretching factor a = desired_sigma for diagonal entries of stretching matrix.
   */

   // scaling matrix (Sc)
   sqrt(pca.eigenvalues, eigDataSigma);
   scale = Mat::diag(1/eigDataSigma);

   // stretching matrix (St)
   // if targetSigma is empty, set sigma of transformed data equal to that of original data
   if (targetSigma.empty())
   {
      stretch = Mat::diag(dataSigma);
   }
   else
   {
      CV_Assert((1 == targetSigma.cols) &&  (1 == targetSigma.channels()) && 
         (input.channels() == targetSigma.rows));

      stretch = Mat::diag(targetSigma);
   }
   // convert to 32F
   stretch.convertTo(stretch, CV_32F);
 
   // subtract the mean from input data
   Mat zmudata;
   Mat repMu = repeat(dataMu.t(), data.rows, 1);
   subtract(data, repMu, zmudata, Mat(), CV_32F);

   // if targetMean is empty, set mean of transformed data equal to that of original data
   if (!targetMean.empty())
   {
      CV_Assert((1 == targetMean.cols) && (1 == targetMean.channels()) && 
         (input.channels() == targetMean.rows));

      repMu = repeat(targetMean.t(), data.rows, 1);
   }

   /*
   project zero mean data to the eigenspace, normalize the variance and reproject,
   then stretch it so that is has the desired sigma: StR'ScR(x(i) - mu), R'R = I.
   since the x(i)s are organized as rows in data, take the transpose of the above
   expression: (x(i)' - mu')R'Sc'(R')'St' = (x(i)' - mu')R'ScRSt,
   then add the desired mean:
   (x(i)' - mu')R'ScRSt + mu_desired
   */
   Mat transformed = zmudata*(pca.eigenvectors.t()*scale*pca.eigenvectors*stretch);
   add(transformed, repMu, transformed, Mat(), CV_32F);

   // reshape transformed data
   Mat dstr32f = transformed.reshape(input.channels(), input.rows);

   return dstr32f;
}

int cv_HVProjection(Mat& img, Mat& src){
	Mat dst_h = Mat::zeros(src.size(), CV_8UC1);
	dst_h += 255;
	Mat dst_v = Mat::zeros(src.size(), CV_8UC1);
	dst_v += 255;
	Rect r;
	
	int* v = new int[src.cols];
	int* h = new int[src.rows];
	memset(v,0,src.cols * 4);
	memset(h,0,src.rows * 4);

	int avg_v = 0, max_v = 0, min_v = src.rows;
	for(int i = 0; i < src.cols; i++){
		for(int j = 0; j < src.rows; j++){
			if(src.at<uchar>(j, i) == 255) v[i]++;
		}
		avg_v += v[i];
		if(v[i] > max_v) max_v = v[i];
		if(v[i] < min_v) min_v = v[i];
	}
	avg_v /= src.cols;
	//avg_v = (max_v - min_v)/2;
	avg_v /= 2;

	int l_pos = -1, r_pos = -1, record_l = -1, record_r = -1;
	for(int i = 1; i < src.cols-1; i++){
		if(l_pos == -1 && r_pos == -1 && v[i] > avg_v && v[i+1] > avg_v && v[i-1] < avg_v)
			l_pos = i;
		if(l_pos != -1 && r_pos == -1 && v[i] < avg_v && v[i+1] < avg_v && v[i-1] > avg_v)
			r_pos = i;
		if(l_pos != -1 && r_pos != -1){
			if((r_pos - l_pos) > (record_r - record_l)){
				record_l = l_pos;
				record_r = r_pos;
			}
			l_pos = -1;
			r_pos = -1;
		}
	}
	if(record_l == -1) r.x = 0;
	else r.x = record_l;
	if(record_r == -1) r.width = src.cols;
	else r.width = record_r - r.x;

	for(int i = 0; i < src.cols; i++){
		for(int j = 0; j < src.rows-v[i]; j++){
			dst_v.at<uchar>(j, i) = 0;
		}
	}

	int avg_h = 0, max_h = 0, min_h = src.cols;
	l_pos = -1;
	r_pos = -1;
	record_l = -1;
	record_r = -1;
	for(int j = 0; j < src.rows; j++){
		for(int i = 0; i < src.cols; i++){
			if(src.at<uchar>(j, i) == 255) h[j]++;
		}
		avg_h += h[j];
		if(h[j] > max_h) max_h = h[j];
		if(h[j] < min_h) min_h = h[j];
	}
	avg_h /= src.rows;
	//avg_h = (max_h - min_h) / 2;
	avg_h /= 2;

	for(int i = 1; i < src.rows-1; i++){
		if(l_pos == -1 && r_pos == -1 && h[i] > avg_h && h[i+1] > avg_v && h[i-1] < avg_v)
			l_pos = i;
		if(l_pos != -1 && r_pos == -1 && h[i] < avg_h && h[i+1] < avg_v && h[i-1] > avg_v)
			r_pos = i;
		if(l_pos != -1 && r_pos != -1){
			if((r_pos - l_pos) > (record_r - record_l)){
				record_l = l_pos;
				record_r = r_pos;
			}
			l_pos = -1;
			r_pos = -1;
		}
	}
	if(record_l == -1) r.y = 0;
	else r.y = record_l;
	if(record_r == -1) r.height = src.rows;
	else r.height = record_r - r.y;

	for(int j = 0; j < src.rows; j++){
		for(int i = 0; i < src.cols-h[j]; i++){
			dst_h.at<uchar>(j, i) = 0;
		}
	}

	imshow("垂直", dst_v);
	imshow("水平", dst_h);

	rectangle(img, r, Scalar(0, 0, 255), 2);
	imshow("src", img);
	return 0;
}

int main()
{
	string p_path = "./pos/";  //./Tongue_Image/20131019/
	char* infile = "./pos.txt";
	char* outfile = "test.data";
	ifstream in(infile);
	ofstream out(outfile);

	//double a[18] = {0.00280036,59.4779,0.428998,2.9444,136.397,939.903,0.00197886,68.4362,0.348336,3.0673,136.616,1336.41,0.00292386,38.6503,0.442026,2.9221,136.406,901.906};
	double a[20] = {0.951918,0.0455224,0,0,0,0,0,0,0,0,0,0,0,0,0.000938607,0,0,0,0.000255984,0.00136525};
	vector<double> stdfeature;
	/*double tmp_dis = 0;
	for(int i = 0; i < 18; i++){
		a[i] = log10(a[i]);
		tmp_dis += a[i] * a[i];
	}*/
	for(int i = 0; i < 20; i++){
		stdfeature.push_back(a[i]);
	}
	
	all_feature.clear();
	int count = 0;
	vector<double> feature;
	for(string file_name; getline(in, file_name); )
	{
		string full_path = p_path + file_name;
		cout << full_path << endl;
		
		/*string p1 = p_path + "pos_258_ps.jpg";
		Mat img1 = imread(p1.c_str());
		string p2 = p_path + "pos_258.jpg";
		Mat img2 = imread(p2.c_str());
		cv_tcdDeltaE(img1, img2);
		imshow("img1", img2);*/
		Mat img = imread(full_path.c_str());
		//Mat img = imread("./pos/pos_140.jpg");

		assert(img.data != NULL); 
		//Mat gvf(img.size(), CV_8UC1);
		//Mat wshed(img.size(), CV_8UC1);
		//cv_GVF(img, gvf, 0.5, 0.5, 20);
		//cv_watershed(img, gvf, wshed);

		//all_feature.push_back(0);

		cv_resize(img, img);
		//imshow("1", img);
		//Mat mean;
		//pyrMeanShiftFiltering(img, mean, 10, 10);

		Mat gw(img.size(), CV_8UC3);
		//Mat huidu(img.size(), CV_8UC3);
		//cv_grayWorld(img, huidu, 3);
		//imwrite("huidu.jpg", huidu);
		
		cv_grayWorld(img, gw, 4);
		//imshow("2", gw);
		
		/*Mat test, test_hsv;
		cv_dcstretch(img, test);
		
		cvtColor(test, test_hsv, CV_BGR2HSV);
		imshow("test", test_hsv);
		vector<Mat> channels;
		split(test_hsv, channels);
		Mat hsv_h = channels.at(0);
		Mat otsu_h = Mat::zeros(img.size(), CV_8UC1);
		cv_otsu(hsv_h, otsu_h);

		imshow("hsv", test_hsv);
		imshow("otsu", otsu_h);

		cv_HVProjection(img, otsu_h);*/
		
		/*Mat mean = Mat::ones(3, 1, CV_32F) * 120;
		Mat sigma = Mat::ones(3, 1, CV_32F) * 50;
		Mat dstrbgr32f = dstretch(img, mean, sigma);
		Mat dstrbgr8u;
		dstrbgr32f.convertTo(dstrbgr8u, CV_8UC3);
		Mat dstrhsv8u;
		cvtColor(dstrbgr8u, dstrhsv8u, CV_BGR2HSV);
		imshow("dstrhsv8u", dstrhsv8u);*/

		//imshow("gw", gw);
		//imwrite("gw.jpg", gw);

		//Mat gamma(img.size(), CV_8UC3);
		//cv_gamma(gw, gamma);

		Mat img_feature, img_mask;
		cv_findCon(gw, img_feature, img_mask);
		
		//Scalar avg = mean(img_mask);
		//
		//if((int)avg.val[0] == 0) cout << -1 << endl;
		//else{
		//	feature.clear();
		//	cv_colorFeature(img_feature, feature, img_mask);
		//
		//	double score = cv_calcCos(feature, stdfeature);
		//	//cout << score << "\t" << count << endl;

		//	/*vector<double>::iterator it;
		//	for(it=feature.begin(); it != feature.end()-1; ++it)
		//	{
		//		out << *it << ",";
		//	}*/
		//	out << score << endl;
		//}

		//Mat dst = Mat::zeros(img.size(), CV_8UC1);
		//cv_tonRegion2(img, dst);
		//imshow("img_feature", img_feature);

		//waitKey(0);
		/*cv_glcm(img_feature);
		cv_colorFeature(img_feature);*/

		count++;
	}
	//for(vector<double>::iterator it = all_feature.begin(); it != all_feature.end(); it++)
	//	out << *it << "\n";
	in.close();
	out.close();
	cout << count << endl;

	//cv_dataTmp();
	//cv_getData();

	//CvRtree_Train();
	
	getchar();
	return 0;
}