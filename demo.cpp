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

//std::vector<std::string> getFilesInFolder(const std::string& folderPath) {
//    std::vector<std::string> files;
//
//    DIR* dir;
//    struct dirent* entry;
//    struct stat fileStat;
//
//    if ((dir = opendir(folderPath.c_str())) != nullptr) {
//        while ((entry = readdir(dir)) != nullptr) {
//            std::string filePath = folderPath + "/" + entry->d_name;
//
//            if (stat(filePath.c_str(), &fileStat) == 0 && S_ISREG(fileStat.st_mode)) {
//                files.push_back(filePath);
//            }
//        }
//        closedir(dir);
//    }
//
//    return files;
//}
//
//int main(int argc, char* argv[]) {
//    if (argc < 2) {
//        std::cout << "Usage:\n  ./demo <input_folder> <output_folder>" << std::endl;
//        return 0;
//    }
//
//    std::string inputPath = argv[1];
//    std::string outputFolder;
//    if (argc >= 3) {
//        outputFolder = argv[2];
//    } else {
//        outputFolder = ".";  // 默认为当前目录
//    }
//
//    std::vector<std::string> inputFiles;
//
//    // 检查输入路径是文件还是文件夹
//    struct stat pathStat;
//    if (stat(inputPath.c_str(), &pathStat) == 0) {
//        if (S_ISREG(pathStat.st_mode)) {
//            // 输入路径是文件
//            inputFiles.push_back(inputPath);
//        } else if (S_ISDIR(pathStat.st_mode)) {
//            // 输入路径是文件夹
//            inputFiles = getFilesInFolder(inputPath);
//        } else {
//            std::cout << "Invalid input path: " << inputPath << std::endl;
//            return -1;
//        }
//    } else {
//        std::cout << "Invalid input path: " << inputPath << std::endl;
//        return -1;
//    }
//
//    // 遍历输入文件夹中的图像文件
//    for (const std::string& filePath : inputFiles) {
//        std::string outputFilePath;
//        if(inputFiles.size() == 1){
//            outputFilePath = "out.jpg";
//        } else{
//            outputFilePath = outputFolder + "/" + filePath.substr(filePath.find_last_of('/') + 1);
//        }
//
//        ModelParam model1 = {"../models/s_car_nx.engine", 2, 0.4f, 0.4f, 640, 640};
//        //ModelParam model2 = {"../models/zhou.engine", 1, 0.4f, 0.4f, 640, 640};
//
//        nvinfer1::IExecutionContext* con1 = infer_init(model1);
//        //nvinfer1::IExecutionContext* con2 = infer_init(model2);
//
//        cv::Mat img = cv::imread(filePath);
//        if (img.empty()) {
//            std::cout << "Failed to read image: " << filePath << std::endl;
//            return -1;  // 或者执行其他适当的错误处理操作
//        }
//
//        std::cout << "Processing: " << filePath << std::endl;
//
//        int cols = img.cols;
//        int rows = img.rows;
//        int size = img.total() * img.elemSize();
//        byte* bytes = new byte[size];
//        std::memcpy(bytes, img.data, size * sizeof(byte));
//
//        float waste1 = 0.0;
//        float waste2 = 0.0;
//        int resNum1 = 0;
//        int resNum2 = 0;
//        auto res1 = new DetectRes[128];
//        auto res2 = new DetectRes[128];
//
//        infer_run_one(bytes, rows, cols, res1, &resNum1, &waste1, con1);
//        //infer_run_one(bytes, rows, cols, res2, &resNum2, &waste2, con2);
//
//        std::memcpy(img.data, bytes, size * sizeof(byte));
//        cv::imwrite(outputFilePath, img);
//
//        delete[] bytes;
//        delete[] res1;
//        delete[] res2;
//
//        std::cout << "Output saved: " << outputFilePath << std::endl;
//    }
//
//    return 0;
//}

static void sleep_ms(unsigned int secs)
{
    struct timeval tval;
    tval.tv_sec=secs/1000;
    tval.tv_usec=(secs*1000)%1000000;
    select(0,NULL,NULL,NULL,&tval);
}

int main(int argc, char* argv[])
{

     void *handle = dlopen("libyolov5_infer_rtsp.so", RTLD_LAZY);

     if(!handle)
     {
         printf("open lib error\n");
         std::cout<<dlerror()<<std::endl;
         return -1;
     }

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

//    if(argc < 1)
//    {
//        std::cout << "example:\n  ./demo ../sample/bus.jpg" << std::endl;
//        return 0;
//    }
//
//    ModelParam model1 = {"../models/s_car.engine",2, 0.4f, 0.4f, 640,640};
//    ModelParam model2 = {"../models/zhou.engine",1,0.4f, 0.4f, 640,640};//0是货车 1是客车
//
//    nvinfer1::IExecutionContext* con1 = infer_init(model1);
//    nvinfer1::IExecutionContext* con2 = infer_init(model2);
//    cv::Mat img = cv::imread(argv[1]);
//
//    std::cout<< "before input:" << img.size() <<std::endl;
//    //std::cout<< "input:" << img.size() <<std::endl;
//    int cols = img.cols;
//    int rows = img.rows;
//    int size = img.total() * img.elemSize();
//    byte * bytes = new byte[size];
//    std::memcpy(bytes,img.data,size * sizeof(byte));
//    float waste1 = 0.0;
//    float waste2 = 0.0;
//    int resNum1 = 0;
//    int resNum2 = 0;
//    auto res1 = new DetectRes[128];
//    auto res2 = new DetectRes[128];
//    infer_run_one(bytes,rows,cols, res1, &resNum1, &waste1,con1);
//    infer_run_one(bytes,rows,cols, res2, &resNum2, &waste2,con2);
//
//    std::memcpy(img.data,bytes,size * sizeof(byte));
//    cv::imwrite("./out.jpg", img);

    // infer_release(&con);
    std::string path1 = "rtspsrc location=rtsp://admin:21232f297a57a5a743894a0e4a801fc3@10.100.8.42";
    std::string path = "rtspsrc location=rtsp://admin:wanji168@10.100.8.64";
    //ini(path1,0);
    ini("rtspsrc location=rtsp://admin:wanji168@192.168.0.152",0);


     do{

//         if(Conn(0)){
//             byte * bytes = new byte[6220800];
//             auto start = std::chrono::high_resolution_clock::now();
//             gbyte(bytes,true,0,true);
//             auto end = std::chrono::high_resolution_clock::now();
//             std::chrono::duration<double, std::milli> elapsed = end - start;
//             std::cout << "Time taken by getmat(0): " << elapsed.count() << " ms" << std::endl;
//             delete[] bytes;
//
//         }
//         else{
//             reConn(0);
//         }

        if(Conn(0)){
            byte * bytes1 = new byte[6220800];
            //string filename =  ctime(&timep);
            auto start = std::chrono::high_resolution_clock::now();
            gbyte(bytes1,true,0,false);
            cv::Mat img1(1080,1920, CV_8UC3, bytes1);
             //cv::Mat img1 = gmat(1);
            // cv::imwrite("./out/" + filename1, img);
            cv::imwrite("out1.jpg", img1);
            sleep_ms(500);
            //cv::Mat img1 = cam_map[1];
            //infer(1, res2, &resNum2, &waste2,con2);
            //std::cout << "waste2:" << waste2 << std::endl;
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> elapsed = end - start;
            std::cout << "Time taken by getmat(1): " << elapsed.count() << " ms" << std::endl;
            delete[] bytes1;
            img1.release();

        }
        else{
            reConn(0);
        }
     }
     while(1);


    return 1;
}
