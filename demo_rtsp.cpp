#include <string>
#include <chrono>
#include <thread>
#include "yolov5.h"
#include "NvInfer.h"
#include <mutex>
#include <chrono>
#include <dlfcn.h>
#include <vector>
#include <dirent.h>
#include <sys/stat.h>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <time.h>

typedef unsigned char byte;

void *handle = dlopen("libyolov5_infer_rtsp.so", RTLD_LAZY);

// if(!handle)
// {
//     printf("open lib error\n");
//     std::cout<<dlerror()<<std::endl;
//     return -1;
// }

typedef cv::Mat (*_mat)(int a);
typedef bool (*_connect)(int a);
typedef void (*_init)(char *a,int i);
//typedef void (*_init)(std::string const &a,int i);
typedef void (*_getbyte)(unsigned char* a,bool flag,int i,bool b);
typedef void (*_recon)(int i);
typedef int (*_infer)(int index, DetectRes* res, int *resNum, float *waste, nvinfer1::IExecutionContext *context);
typedef nvinfer1::IExecutionContext* (*_inimodel)(const ModelParam& model_param);

_mat gmat = (_mat) dlsym(handle, "getmat");
_connect Conn = (_connect) dlsym(handle, "isconnected");
_init ini = (_init) dlsym(handle, "Init_uri");
_getbyte gbyte = (_getbyte) dlsym(handle, "Getbyte");
_recon reConn = (_recon) dlsym(handle, "reconnect");
_infer infer = (_infer) dlsym(handle, "infer_run_one_index");
_inimodel init = (_inimodel) dlsym(handle, "infer_init");


static void sleep_ms(unsigned int secs)
{
    struct timeval tval;
    tval.tv_sec=secs/1000;
    tval.tv_usec=(secs*1000)%1000000;
    select(0,NULL,NULL,NULL,&tval);
}

void test_cam(int i){

    if(Conn(i)){
        byte * bytes1 = new byte[6220800];
        //string filename =  ctime(&timep);
        auto start = std::chrono::high_resolution_clock::now();
        gbyte(bytes1,true,i,false);
        cv::Mat img1(1920,1080, CV_8UC3, bytes1);
        //cv::imwrite("out.jpg", img1);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        std::cout << "Time taken by getmat" << i << ": " << elapsed.count() << " ms" << std::endl;
        delete[] bytes1;
        img1.release();

    }
    else{
        reConn(i);
    }

}

int main(int argc, char* argv[])
{

    std::string path1 = "rtspsrc location=rtsp://admin:21232f297a57a5a743894a0e4a801fc3@10.100.8.42";
    std::string path = "rtsp://admin:wanji168@10.100.8.64";
    //ini(path1,0);
    ini("rtspsrc location=rtsp://admin:wanji168@10.100.8.64",0);
    ini("rtspsrc location=rtsp://admin:wanji168@10.100.8.64",1);
    ini("rtspsrc location=rtsp://admin:wanji168@10.100.8.64",2);
//    ini("rtspsrc location=rtsp://admin:wanji168@10.100.8.64",3);
//    ini("rtspsrc location=rtsp://admin:wanji168@10.100.8.64",4);
//    ini("rtspsrc location=rtsp://admin:wanji168@10.100.8.64",5);
//    
    //cv::VideoCapture capture(path);


     do{
        test_cam(0);
        test_cam(1);
        test_cam(2);      
        
        //openc软解码
//        if (capture.isOpened())  {
//            
//            cv::Mat frame; 
//            auto start = std::chrono::high_resolution_clock::now();       
//            if (!capture.read(frame))  
//            {  
//                std::cerr << "Error reading frame from video stream" << std::endl;  
//                break;  
//            }  
//    
//            // 显示帧  
//            cv::imwrite("opencv.jpg", frame); 
//            auto end = std::chrono::high_resolution_clock::now();
//            std::chrono::duration<double, std::milli> elapsed = end - start;
//            std::cout << "Time taken by opencv: " << elapsed.count() << " ms" << std::endl;
//            frame.release();
//        }
        
           
     }
     while(1);


    return 1;
}
