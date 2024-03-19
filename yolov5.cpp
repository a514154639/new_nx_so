#include <iostream>
#include <chrono>
#include <cmath>
#include <vector>
#include <time.h>
#include <sys/select.h>
#include "cuda_utils.h"
#include "common.hpp"
#include "preprocess.h"
#include "yolov5.h"
// #include <time.h>

#define USE_INT8  // set USE_INT8 or USE_FP16 or USE_FP32
#define DEVICE 0  // GPU id
#define BATCH_SIZE 1
#define MAX_IMAGE_INPUT_SIZE_THRESH 2000 * 30000 // ensure it exceed the maximum size in the input images !

// stuff we know about the network and the input/output blobs
static const int OUTPUT_SIZE = Yolo::MAX_OUTPUT_BBOX_COUNT * sizeof(Yolo::Detection) / sizeof(float) + 1;  // we assume the yololayer outputs no more than MAX_OUTPUT_BBOX_COUNT boxes that conf >= 0.1
const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";
// static Logger gLogger;

static uint8_t* img_host = nullptr;
static uint8_t* img_device = nullptr;
static cudaStream_t stream;
// IExecutionContext* context = nullptr;
IRuntime* runtime = nullptr;
ICudaEngine* engine = nullptr;
static float* buffers[2];
static int inputIndex = 0;
static int outputIndex = 0;

static float NMS_THRESH = 0.4f;
static float CONF_THRESH = 0.5f;

const int NUM_VIDEOS = 5;
cv::VideoCapture videos[NUM_VIDEOS]; 
std::map<int,std::string> url_map;
std::vector<int> emptyFrameCount(NUM_VIDEOS,0);
std::vector<cv::Mat> cam_map(NUM_VIDEOS);

class Logger : public nvinfer1::ILogger
{
    void log(Severity severity, const char* msg) noexcept override
    {
        // suppress info-level messages
        if (severity != Severity::kINFO){
            // std::cout << msg << std::endl;
        }
    }
} gLogger;

static int get_width(int x, float gw, int divisor = 8) {
    return int(ceil((x * gw) / divisor)) * divisor;
}

static int get_depth(int x, float gd) {
    if (x == 1) return 1;
    int r = round(x * gd);
    if (x * gd - int(x * gd) == 0.5 && (int(x * gd) % 2) == 0) {
        --r;
    }
    return std::max<int>(r, 1);
}

void doInference(IExecutionContext& context, cudaStream_t& stream, void **buffers, float* output, int batchSize) {
    // infer on the batch asynchronously, and DMA output back to host
    context.enqueue(batchSize, buffers, stream, nullptr);
    CUDA_CHECK(cudaMemcpyAsync(output, buffers[1], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
}

IExecutionContext* infer_init(const ModelParam& model_param)
{
    IExecutionContext* context = nullptr;
    // if(strlen(model_param.modelPath) == 0)
    // {
    //     std::cerr << "param value error!" << std::endl;
    //     return 0;
    // }
    
	Yolo::CLASS_NUM = model_param.classesNum;
	Yolo::INPUT_H = model_param.inputHeight;
	Yolo::INPUT_W = model_param.inputWidth;
	NMS_THRESH = model_param.iouThres;
	CONF_THRESH = model_param.confThres;
	
    // deserialize the .engine and run inference
    std::ifstream file(model_param.modelPath, std::ios::binary);
    // if (!file.good()) {
    //     std::cerr << "read " << model_param.modelPath << " error!" << std::endl;
    //     return 0;
    // }
    char *trtModelStream = nullptr;
    size_t size = 0;
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    trtModelStream = new char[size];
    assert(trtModelStream);
    file.read(trtModelStream, size);
    file.close();
    
	cudaSetDevice(DEVICE);
    runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;
    assert(engine->getNbBindings() == 2);
    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
    outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
    assert(inputIndex == 0);
    assert(outputIndex == 1);
    // Create GPU buffers on device
    CUDA_CHECK(cudaMalloc((void**)&buffers[inputIndex], BATCH_SIZE * 3 * Yolo::INPUT_H * Yolo::INPUT_W * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&buffers[outputIndex], BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));

    // Create stream
    CUDA_CHECK(cudaStreamCreate(&stream));

    // prepare input data cache in pinned memory 
    CUDA_CHECK(cudaMallocHost((void**)&img_host, MAX_IMAGE_INPUT_SIZE_THRESH * 3));
    // prepare input data cache in device memory
    CUDA_CHECK(cudaMalloc((void**)&img_device, MAX_IMAGE_INPUT_SIZE_THRESH * 3));
    
    return context;
    // return 1;
}

IExecutionContext* infer_init_640(const ModelParam& model_param)
{
    IExecutionContext* context = nullptr;
    // if(strlen(model_param.modelPath) == 0)
    // {
    //     std::cerr << "param value error!" << std::endl;
    //     return 0;
    // }
    
	Yolo::CLASS_NUM = model_param.classesNum;
	Yolo::INPUT_H = model_param.inputHeight;
	Yolo::INPUT_W = model_param.inputWidth;
	NMS_THRESH = model_param.iouThres;
	CONF_THRESH = model_param.confThres;

	//std::cout<< "deserialize the .engine..." <<std::endl;
    // deserialize the .engine and run inference
    std::ifstream file(model_param.modelPath, std::ios::binary);
    // if (!file.good()) {
    //     std::cerr << "read " << model_param.modelPath << " error!" << std::endl;
    //     return 0;
    // }
    char *trtModelStream = nullptr;
    size_t size = 0;
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    trtModelStream = new char[size];
    assert(trtModelStream);
    file.read(trtModelStream, size);
    file.close();

	cudaSetDevice(DEVICE);
    runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;
    assert(engine->getNbBindings() == 2);
    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
    outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
    assert(inputIndex == 0);
    assert(outputIndex == 1);
    // Create GPU buffers on device
    CUDA_CHECK(cudaMalloc((void**)&buffers[inputIndex], BATCH_SIZE * 3 * Yolo::INPUT_H * Yolo::INPUT_W * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&buffers[outputIndex], BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));

    // Create stream
    CUDA_CHECK(cudaStreamCreate(&stream));

    //std::cout<< "prepare input data cache in pinned memory .." <<std::endl;
    // prepare input data cache in pinned memory 
    CUDA_CHECK(cudaMallocHost((void**)&img_host, MAX_IMAGE_INPUT_SIZE_THRESH * 3));

    //std::cout<< "prepare input data cache in device memory .." <<std::endl;
    // prepare input data cache in device memory
    CUDA_CHECK(cudaMalloc((void**)&img_device, MAX_IMAGE_INPUT_SIZE_THRESH * 3));
    //std::cout<< "init engine finish " <<std::endl;
    return context;
    // return 1;
}

int infer_run_one_640(unsigned char *image1, int row, int clomn, DetectRes* res, int *resNum, float *waste, IExecutionContext *context)
{
    //std::cout<< "runing infer..." <<std::endl;
    cv::Mat img=cv::Mat::zeros(row, clomn, CV_8UC3);
    //std::cout<< "index:" << (int)img.cols * img.rows * 3 <<std::endl;
    std::memcpy(img.data, image1, (int)img.cols * img.rows * 3);
    //std::cout<< "img:" << img.size() <<std::endl;
    // cv::Mat img=cv::Mat(1920,1080,CV_8UC3,image1);
    if (img.empty())
    {
        std::cerr << "input img is empty." << std::endl;
        return 0;
    }
    size_t size_image = img.cols * img.rows * 3;
    //size_t  size_image_dst = Yolo::INPUT_H * Yolo::INPUT_W * 3;
    float* buffer_idx = (float*)buffers[inputIndex];

    // allocate memory for res, resNum and waste
    std::vector<DetectRes> res_vec(10);
    int res_num = 0;
    float waste_time = 0.0f;

    //std::cout<< "copy data to pinned memory..." <<std::endl;
    //copy data to pinned memory
    memcpy(img_host, img.data, size_image);

    //std::cout<< "copy data to device memory..." <<std::endl;
    //copy data to device memory
    CUDA_CHECK(cudaMemcpyAsync(img_device, img_host, size_image, cudaMemcpyHostToDevice, stream));
    preprocess_kernel_img(img_device, img.cols, img.rows, buffer_idx, Yolo::INPUT_W, Yolo::INPUT_H, stream);

    // Run inference
    auto start = std::chrono::system_clock::now();
    static float prob[BATCH_SIZE * OUTPUT_SIZE];
    //std::cout<< "doInference..." <<std::endl;
    doInference(*context, stream, (void**)buffers, prob, BATCH_SIZE);
    auto end = std::chrono::system_clock::now();
    waste_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // std::cout << "inference time: " << waste_time << "ms" << std::endl;
    std::vector<Yolo::Detection> result;
    result.reserve(10);
    nms(result, prob, CONF_THRESH, NMS_THRESH);
    std::cout << result.size() << std::endl;
    int j = 0;
    for (; j < result.size(); j++) {
        cv::Rect r = get_rect(img, result[j].bbox);
        // cv::rectangle(img, r, cv::Scalar(0x27, 0xC1, 0x36), 5);
        // cv::putText(img, std::to_string((int)result[j].class_id), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
        DetectRes det(result[j].class_id, result[j].conf, Rect{r.x, r.y, r.width, r.height});
        //std::cout <<"label: "<< result[j].class_id << " confidence: " << result[j].conf << " location:" << (r.x + r.width)/2 << ","<< (r.y + r.height)/2 <<" top:"<< r.y <<" bottom:" << r.y+r.height <<std::endl;
        res_vec[j] = det;
    }
    res_num = j;
    //std::cout<< "copy result..." <<std::endl;
    // assign the results to the output variables
    std::copy(res_vec.begin(), res_vec.end(), res);

    //*res = res_vec.data();
    *resNum = res_num;
    *waste = waste_time;

    return 1;
}


int infer_run_one(unsigned char *image1,int row,int clomn, DetectRes* res, int *resNum, float *waste, IExecutionContext *context)
{
    //std::cout<< "runing infer..." <<std::endl;
    cv::Mat img=cv::Mat::zeros(row, clomn, CV_8UC3);
    //cv::Mat img=cv::Mat::zeros(2560, 1440, CV_8UC3);
    //std::cout<< "index:" << (int)img.cols * img.rows * 3 <<std::endl;
    std::memcpy(img.data, image1, (int)img.cols * img.rows * 3);
    //std::cout<< "img:" << img.size() <<std::endl;
    // cv::Mat img=cv::Mat(1920,1080,CV_8UC3,image1);
    //img = img(cv::Range(0, img.rows), cv::Range(100, img.cols));
    if (img.empty())
    {
        std::cerr << "input img is empty." << std::endl;
        return 0;
    }
    size_t  size_image = img.cols * img.rows * 3;
    //size_t  size_image_dst = Yolo::INPUT_H * Yolo::INPUT_W * 3;
  	float* buffer_idx = (float*)buffers[inputIndex];
    //copy data to pinned memory
    memcpy(img_host,img.data,size_image);
    //copy data to device memory
    CUDA_CHECK(cudaMemcpyAsync(img_device,img_host,size_image,cudaMemcpyHostToDevice,stream));
    //img preprocess
    preprocess_kernel_img(img_device, img.cols, img.rows, buffer_idx, Yolo::INPUT_W, Yolo::INPUT_H, stream);
    
  	// Run inference
    
  	auto start = std::chrono::system_clock::now();
  	static float prob[BATCH_SIZE * OUTPUT_SIZE];
  	doInference(*context, stream, (void**)buffers, prob, BATCH_SIZE);
  	auto end = std::chrono::system_clock::now();
  	*waste = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    // std::cout << "inference time: " << *waste << "ms" << std::endl;
    std::vector<Yolo::Detection> result;
    result.reserve(10);
    nms(result, prob, CONF_THRESH, NMS_THRESH);
    std::cout << result.size() << std::endl;
    int j = 0;
    for (; j < result.size(); j++) {
        cv::Rect r = get_rect(img, result[j].bbox);
        cv::rectangle(img, r, cv::Scalar(0x27, 0xC1, 0x36), 5);
        cv::putText(img, std::to_string((int)result[j].class_id), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
        DetectRes det(result[j].class_id, result[j].conf, Rect{r.x, r.y, r.width, r.height});
        std::cout <<"label: "<< result[j].class_id << " confidence: " << result[j].conf << " mid:" << r.x + r.width / 2 << " width:"<< r.width << std::endl;     
        res[j] = det;
	}
	*resNum = j;
    std::memcpy(image1, img.data, (int)img.cols * img.rows * 3);
    // if (result.size()> 0){
    //     char name[256] = {0};
    //     time_t timep;
    //     struct tm *p;
    //     time(&timep);
    //     p = localtime(&timep);
    //     sprintf(name, "%d_%d_%d_%d_%02d.jpg",1900+p->tm_year,1+p->tm_mon,p->tm_mday,p->tm_hour,p->tm_min);
    //     cv::imwrite(name, img);
    // }
	return 1;
}


int infer_run_one_index(int index, DetectRes* res, int *resNum, float *waste, IExecutionContext *context, int startpix)
{   
    //std::cout<< "runing infer..." <<std::endl;
    //cv::Mat img=cv::Mat::zeros(row, clomn, CV_8UC3);
    //std::memcpy(img.data, image1, (int)img.cols * img.rows * 3);
    cv::Mat img = cam_map[index];
    if(startpix != 0){
        //img = img(cv::Range(0, img.rows), cv::Range(startpix, img.cols));
        cv::Rect roi(startpix, 0, img.cols - startpix, img.rows);
        img(roi).copyTo(img);
        
    }    
    if (img.empty())
    {
        std::cerr << "input img is empty." << std::endl;
        return 0;
    }
    //cv::imwrite("./cut.jpg", img);
    size_t  size_image = img.cols * img.rows * 3;
    //size_t  size_image_dst = Yolo::INPUT_H * Yolo::INPUT_W * 3;
  	float* buffer_idx = (float*)buffers[inputIndex];
    //copy data to pinned memory
    memcpy(img_host,img.data,size_image);
    //copy data to device memory
    CUDA_CHECK(cudaMemcpyAsync(img_device,img_host,size_image,cudaMemcpyHostToDevice,stream));
    preprocess_kernel_img(img_device, img.cols, img.rows, buffer_idx, Yolo::INPUT_W, Yolo::INPUT_H, stream);  
  	// Run inference  
  	auto start = std::chrono::system_clock::now();
  	static float prob[BATCH_SIZE * OUTPUT_SIZE];
  	doInference(*context, stream, (void**)buffers, prob, BATCH_SIZE);
  	auto end = std::chrono::system_clock::now();
  	*waste = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::vector<Yolo::Detection> result;
    result.reserve(10);
    nms(result, prob, CONF_THRESH, NMS_THRESH);
    //std::cout << result.size() << std::endl;    
    int j = 0;
    for (; j < result.size(); j++) {
        cv::Rect r = get_rect(img, result[j].bbox);
        DetectRes det(result[j].class_id, result[j].conf, Rect{r.x, r.y, r.width, r.height});
        //std::cout <<"label: "<< result[j].class_id << " confidence: " << result[j].conf << " location:" << (r.x + r.width)/2 << ","<< (r.y + r.height)/2 <<" top:"<< r.y <<" bottom:" << r.y+r.height <<" zhou:" << r.height << std::endl;       
        //std::lock_guard<std::mutex> lock(mtx);
        res[j] = det;
	}
	*resNum = j;
	return 1;
}



void infer_release(IExecutionContext *context)
{
	// Release stream and buffers
    cudaStreamDestroy(stream);
    CUDA_CHECK(cudaFree(img_device));
    CUDA_CHECK(cudaFreeHost(img_host));
    CUDA_CHECK(cudaFree(buffers[inputIndex]));
    CUDA_CHECK(cudaFree(buffers[outputIndex]));
    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();
}

// sleep for ms 
static void sleep_ms(unsigned int secs)
{
    struct timeval tval;
    tval.tv_sec=secs/1000;
    tval.tv_usec=(secs*1000)%1000000;
    select(0,NULL,NULL,NULL,&tval);
}

void Init_uri(const char * uri,int i){       
    const char* path = ":554 latency = 0 ! rtph265depay ! h265parse ! nvv4l2decoder ! nvvidconv ! video/x-raw,format=(string)BGRx ! videoconvert ! appsink sync = false\0";
    char res[200];
    sprintf(res, "%s%s", uri, path);
    std::cout <<res<< std::endl;
    url_map[i] = res;
    videos[i].open(res);
    if(videos[i].isOpened()){
        std::cout<<"Init success"<<std::endl;
    }
    else{
        std::cout<<"Init fail"<< std::endl;
    }
    
    //delete[] res;   
}

void getmat(int i){  
    //emptyFrameCount[i] = 0;  // counter for consecutive empty frames
    cv::Mat frame;
    if (videos[i].isOpened()) {
        videos[i].read(frame);
        
    } 
    else {
        std::cout << "Cam " << i << " link fail." <<  std::endl;
        reconnect(i);
    }
    //return frame;
    frame.release();
}

bool isconnected(int i){
    if(videos[i].isOpened()){
        return true;
    }
    else{
         std::cout<< "cam: "<< i <<" link fail..."<< std::endl;
        return false;        
    } 
}

void reconnect(int i){
    std::cout<< "cam: "<< i <<" reconnecting..."<< std::endl;
    sleep_ms(5000);
    videos[i].open(url_map[i]);
    if(videos[i].isOpened()){
         std::cout<< "cam: "<< i <<" reconnected"<< std::endl;
         videos[i].release();//刚链接上时相机的像素还未发生操作变化，需要等待一段时间完成配置文件读取（旋转等变化）
         sleep_ms(5000);
         videos[i].open(url_map[i]);
    }
    else{
          std::cout<< "cam: "<< i <<" reconnected fail"<< std::endl;
          videos[i].release();
          reconnect(i);
    }
}




void Getbyte1(unsigned char *image, bool flag, int i, bool rotate)
{
    //static int emptyFrameCount = 0;  // counter for consecutive empty frames
    cv::Mat frame;
    if (videos[i].isOpened()) {
        videos[i].read(frame);
        if (frame.empty()) {
            std::cout << "Input frame is empty." << std::endl;
            emptyFrameCount[i]++;
            if (emptyFrameCount[i] >= 3) {
                std::cout << "5 consecutive empty frames detected. Reconnecting to cam " << i << "." <<  std::endl;
                videos[i].release();
                reconnect(i);
                emptyFrameCount[i] = 0;  // reset the counter after reconnecting
            }
        } 
        else {
            emptyFrameCount[i] = 0;  // reset the counter if a non-empty frame is received  
            cam_map[i] = frame;       
            if(flag){              
                if(rotate){
                  transpose(frame,frame);
                  //cam_map[i] = frame;
                }
                memcpy(image, frame.data, (int)frame.cols * frame.rows * 3);
            }
            else{                
                return;
            }
            
        }
    } else {
        std::cout << "Cam " << i << " link fail." <<  std::endl;
        videos[i].release();
        reconnect(i);
    }

    //frame.release();
}

void Getbyte(unsigned char *image, bool flag, int i, bool rotate)
{
    cv::Mat frame;
    if (videos[i].isOpened()) {
        videos[i].read(frame);
        if (frame.empty()) {
            std::cout << "Input frame is empty" << std::endl;
            if (++emptyFrameCount[i] >= 3) {
                std::cout << "5 consecutive empty frames detected. Reconnecting to cam" << i << "。" <<  std::endl;
                videos[i].release();
                reconnect(i);
                emptyFrameCount[i] = 0;  // 重连后重置计数器
            }
        } 
        else {
            emptyFrameCount[i] = 0;  // 收到非空帧时重置计数器
            cam_map[i] = frame;       
            if(flag){              
                if(rotate){
                  transpose(frame,frame);
                  cam_map[i] = frame;
                }
                memcpy(image, frame.data, (int)frame.cols * frame.rows * 3);
            }
            return;   // 如果flag为false，直接返回
        }
    } else {
        std::cout << "cam " << i << " link fail" <<  std::endl;
        videos[i].release();
        reconnect(i);
    }
}

void Getbyte_new(unsigned char *image, bool flag, int i, bool rotate)
{
    cv::Mat frame;
    if (videos[i].isOpened()) {
        videos[i].read(frame);
        if (frame.empty()) {
            std::cout << "Input frame is empty" << std::endl;
            if (++emptyFrameCount[i] >= 3) {
                std::cout << "3 consecutive empty frames detected. Reconnecting to cam" << i << "," <<  std::endl;
                videos[i].release();
                reconnect(i);
                emptyFrameCount[i] = 0;  // 重连后重置计数器
            }
        } 
        else {
            emptyFrameCount[i] = 0;  // 收到非空帧时重置计数器
            if(flag){              
                if(rotate){
                  transpose(frame,frame);
                }
                cam_map[i] = frame.clone(); // 使用clone方法复制图像

                // 将图像数据复制到image中
                std::memcpy(image, cam_map[i].data, cam_map[i].total() * cam_map[i].elemSize());
            }
            return;   // 如果flag为false，直接返回
        }
    } else {
        std::cout << "cam " << i << " link fail" <<  std::endl;
        videos[i].release();
        reconnect(i);
    }
}



'
