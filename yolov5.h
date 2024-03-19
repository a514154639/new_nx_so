#ifndef __YOLOV5_H__
#define __YOLOV5_H__

// #include "utils.h"
#include <opencv2/opencv.hpp>
#include "NvInfer.h"
#ifdef __cplusplus
	#define EXTERN_C extern "C" 
#else
	#define EXTERN_C
#endif

#define DETECORS_MAX_NUM 64

struct ModelParam{
    char modelPath[256]; //.engine file path
    int classesNum;
    // char labels[DETECORS_MAX_NUM][128]; // labels
    float iouThres;
    float confThres;
    int inputWidth;
    int inputHeight;
};

struct Rect{
    int left;
    int top;
    int width;
    int height;
};

struct DetectRes{
    int labelId;
    float confidence;
    Rect rect;
	
	DetectRes(){};
	DetectRes(int labelid_, float confidence_, Rect rect_)
	:labelId(labelid_), confidence(confidence_), rect(rect_){}
};

//typedef void (*CallBackFunc)(char* data, size_t frameh, size_t framew, size_t fstep, DetectRes* res, size_t resLen);

EXTERN_C nvinfer1::IExecutionContext* infer_init(const ModelParam& model_param);
//EXTERN_C nvinfer1::IExecutionContext* infer_init_640(const ModelParam& model_param);
//int infer_set_callback(CallBackFunc func);
//EXTERN_C int infer_run_one(cv::Mat mat, DetectRes* res, size_t *resNum, size_t *waste);
EXTERN_C int infer_run_one(unsigned char *image1,int row, int clomn, DetectRes* res, int *resNum, float *waste, nvinfer1::IExecutionContext *context);
//EXTERN_C int infer_run_one_640(unsigned char *image1,int row, int clomn, DetectRes* res, int *resNum, float *waste, nvinfer1::IExecutionContext *context);
EXTERN_C int infer_run_one_index(int index, DetectRes* res, int *resNum, float *waste, nvinfer1::IExecutionContext *context, int x);
EXTERN_C void infer_release(nvinfer1::IExecutionContext *context);

EXTERN_C void Init_uri(const char * uri,int i);
EXTERN_C void getmat(int i);
EXTERN_C bool isconnected(int i);
EXTERN_C void Getbyte(unsigned char *image, bool flag ,int i, bool rotate);
EXTERN_C void reconnect(int i);


#endif
