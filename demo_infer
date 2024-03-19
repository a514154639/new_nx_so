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

//std::mutex mtx;
//static uint8_t* img_host = nullptr;
//void thread_func(ThreadData* data) {
//    std::lock_guard<std::mutex> lock(mtx);
//    int ret = infer_run_one(data->image_data, data->row, data->clomn, data->res, data->res_num, data->waste, data->context);   
//    std::cout << "Thread finished with result " << ret << std::endl;
//}

//int main(int argc, char* argv[])
//{
//
//
//    if(argc < 1)
//    {
//       std::cout << "example:\n  ./demo ../sample/bus.jpg" << std::endl;
//       return 0;
//    }
//
//    //ModelParam model2 = {"../models/s_car_nx.engine",2, 0.1f, 0.1f, 640, 640};
//    ModelParam model2 = {"../models/zhou.engine",1,0.1f, 0.6f, 640,640};
//
//    //nvinfer1::IExecutionContext* con1 = init(model1);
//    //std::cout<< "con1:" << con1 <<std::endl;
//    nvinfer1::IExecutionContext* con2 = infer_init(model2);
//    //std::cout<< "con2:" << con2 <<std::endl;
//    cv::Mat img = cv::imread(argv[1]);
//    std::cout<< "before input:" << img.size() <<std::endl;
//    //cv::Mat img = cv::imread("../image/huoche.jpg");
//    //int ymin = img.rows > line.yMax - line.yMin ? img.rows - (line.yMax - line.yMin) : 0;
//    int ymin = img.rows > 1.3 * (1555 - 1067) ? (int)(img.rows - 1.3 * (1555 - 1067)) : 0;
//    img = img(cv::Range(ymin, img.rows), cv::Range(0, img.cols));
//    std::cout<< "input:" << img.size() <<std::endl;
//    int cols = img.cols;
//    int rows = img.rows;
//    int size = img.total() * img.elemSize();
//    byte * bytes = new byte[size];
//    //std::cout<< "size:" << size <<std::endl;
//    std::memcpy(bytes,img.data,size * sizeof(byte));
//    //std::cout<< "picnum:" << i<<std::endl;
//    float waste1 = 0.0;
//    float waste2 = 0.0;
//    int resNum1 = 0;
//    int resNum2 = 0;
//    auto res1 = new DetectRes[128];
//    auto res2 = new DetectRes[128];
//    //infer_run_one(bytes,rows,cols, res1, &resNum1, &waste1,con1);
//    infer_run_one(bytes,rows,cols, res2, &resNum2, &waste2,con2);
//
//    // std::thread task01(infer_run_one, bytes,rows,cols, res1, &resNum1, &waste1, con1);
//    // std::thread task02(infer_run_one, bytes,rows,cols, res2, &resNum2, &waste2, con2);
//    // task01.join();
//    // task02.join();
//
//    // ThreadData data1 = {bytes, rows, cols, res1, &resNum1, &waste1, con1};
//    // ThreadData data2 = {bytes, rows, cols, res2, &resNum2, &waste2, con2};
//
//    // // create threads and start running infer_run_one
//    // std::thread t1(thread_func, &data1);
//    // std::thread t2(thread_func, &data2);
//
//    // // wait for threads to finish
//    // t1.join();
//    // t2.join();
//    
//
//    // // infer_run_one(bytes, res, &resNum, &waste);
//    
//    std::memcpy(img.data,bytes,size * sizeof(byte));
//    cv::imwrite("./out.jpg", img);
//      
//	
//    
//    return 1;


//}

std::vector<std::string> getFilesInFolder(const std::string& folderPath) {
    std::vector<std::string> files;

    DIR* dir;
    struct dirent* entry;
    struct stat fileStat;

    if ((dir = opendir(folderPath.c_str())) != nullptr) {
        while ((entry = readdir(dir)) != nullptr) {
            std::string filePath = folderPath + "/" + entry->d_name;

            if (stat(filePath.c_str(), &fileStat) == 0 && S_ISREG(fileStat.st_mode)) {
                files.push_back(filePath);
            }
        }
        closedir(dir);
    }

    return files;
}

int main(int argc, char* argv[]) {
    // if (argc < 2) {
    //     std::cout << "Usage:\n  ./demo <input_folder> <output_folder>" << std::endl;
    //     return 0;
    // }

    std::string inputPath = argv[1];
    //std::string inputPath = "../images/huoche.jpg";
    //std::string inputPath1 = "../images/test1.jpg";
    std::string outputFolder;
    if (argc >= 3) {
        outputFolder = argv[2];
    } else {
        outputFolder = "."; 
    }

    std::vector<std::string> inputFiles;

    struct stat pathStat;
    if (stat(inputPath.c_str(), &pathStat) == 0) {
        if (S_ISREG(pathStat.st_mode)) {
            inputFiles.push_back(inputPath);
        } else if (S_ISDIR(pathStat.st_mode)) {
            inputFiles = getFilesInFolder(inputPath);
        } else {
            std::cout << "Invalid input path: " << inputPath << std::endl;
            return -1;
        }
    } else {
        std::cout << "Invalid input path: " << inputPath << std::endl;
        return -1;
    }

    for (const std::string& filePath : inputFiles) {
        std::string outputFilePath;
        if(inputFiles.size() == 1){
            outputFilePath = "out.jpg";
        } else{
            outputFilePath = outputFolder + "/" + filePath.substr(filePath.find_last_of('/') + 1);
        }

        //ModelParam model1 = {"../models/tmp.engine", 2, 0.4f, 0.4f, 416, 416};
        ModelParam model1 = {"../models/yolov5n.engine", 80, 0.4f, 0.2f, 640, 640};
//        ModelParam model2 = {"../models/s_car_nx.engine", 2, 0.4f, 0.4f, 640, 640};       
//        ModelParam model3 = {"../models/zhou.engine", 1, 0.4f, 0.4f, 640, 640};

        nvinfer1::IExecutionContext* con1 = infer_init(model1);
//        nvinfer1::IExecutionContext* con2 = infer_init(model2);
//        nvinfer1::IExecutionContext* con3 = infer_init(model3);

        cv::Mat img = cv::imread(filePath);
        //cv::Mat img1 = cv::imread(inputPath1);
        if (img.empty()) {
            std::cout << "Failed to read image: " << filePath << std::endl;
            return -1; 
        }

        std::cout << "Processing: " << filePath << std::endl;

        int cols = img.cols;
        int rows = img.rows;
        int size = img.total() * img.elemSize();
        byte* bytes = new byte[size];
        std::memcpy(bytes, img.data, size * sizeof(byte));

        float waste1 = 0.0;
        float waste2 = 0.0;
        float waste3 = 0.0;
        int resNum1 = 0;
        int resNum2 = 0;
        int resNum3 = 0;
        auto res1 = new DetectRes[128];
        auto res2 = new DetectRes[128];
        auto res3 = new DetectRes[128];

        infer_run_one_index(img, res1, &resNum1, &waste1, con1);
//        infer_run_one_index(img, res2, &resNum2, &waste2, con2);
//        infer_run_one_index(img, res3, &resNum3, &waste3, con3);

        // infer_run_one_640(bytes, rows, cols, res1, &resNum1, &waste1, con1);
        // infer_run_one_640(bytes, rows, cols, res2, &resNum2, &waste2, con2);
        // infer_run_one(bytes, rows, cols, res3, &resNum3, &waste3, con3);

        std::memcpy(img.data, bytes, size * sizeof(byte));
        cv::imwrite(outputFilePath, img);

        delete[] bytes;
        delete[] res1;
        delete[] res2;

        std::cout << "Output saved: " << outputFilePath << std::endl;
    }

    return 0;
}
