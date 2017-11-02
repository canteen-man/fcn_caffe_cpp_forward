#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include<string.h>
#include <caffe/caffe.hpp>
#include <string>
#include <vector>
using namespace cv;
using namespace std;
using namespace caffe;
class Detector{//检测类
  public:
    Detector(const string& model_file,
             const string& weights_file );
    Mat Predict(const cv::Mat& img);//分类成员函数
  private:
    void WrapInputLayer(std::vector<cv::Mat>* input_channels);//私有函数，在各个检测成员函数中调用，用来输入图像
    void caffePreprocess(const cv::Mat& img,
                      std::vector<cv::Mat>* input_channels);//用来准备输入图像

 private:
   shared_ptr<caffe::Net <float> > net_;//网络指针
   cv::Size input_geometry_;//输入图像size
   int num_channels_;//输入通道
   cv::Mat mean_;//均值图像

};
Detector::Detector(const string& model_file,
                   const string& weights_file ){
         Caffe::set_mode(Caffe::GPU);
         net_.reset(new caffe::Net<float>(model_file, TEST));
         net_ ->CopyTrainedLayersFrom(weights_file);

}//构造函数，设定GPU模式，载入网络权重
void Detector::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
  caffe::Blob<float>* input_layer = net_->input_blobs()[0];

  int width = input_layer->width();
  int height = input_layer->height();
  float* input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels(); ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
}//用来输入图像进入网络，各个成员函数均调用，固定格式
void Detector::caffePreprocess(const cv::Mat& img,
                            std::vector<cv::Mat>* input_channels)
{
    //cv::Mat sample_resized=img;
     cv::Mat sample_float;
    img.convertTo(sample_float, CV_32FC3);
    cv::split(sample_float,* input_channels);
}
Mat getMean(const size_t& imageHeight, const size_t& imageWidth)
{
    Mat mean;

    const int meanValues[3] = {104, 117, 123};
    vector<Mat> meanChannels;
    for(size_t i = 0; i < 3; i++)
    {
        Mat channel(imageHeight, imageWidth, CV_32F, Scalar(meanValues[i]));
        meanChannels.push_back(channel);
    }
    cv::merge(meanChannels, mean);
    return mean;
}
Mat fcnpreprocess(const Mat& crop)
{
    Mat fcnpreprocessed;
    crop.convertTo(fcnpreprocessed, CV_32FC3);
    resize(fcnpreprocessed, fcnpreprocessed, Size(256, 256));

    Mat mean = getMean(256, 256);
    cv::subtract(fcnpreprocessed, mean, fcnpreprocessed);

    return fcnpreprocessed;
}
/**********************************************************/
/**************************Predict**************************/
/**********************对fcn语义分割安全带进行前向测试**************************************/
 Mat Detector::Predict(const cv::Mat& img) {
    caffe::Blob<float>* input_layer = net_->input_blobs()[0];//拿输入层数据
    input_layer->Reshape(1, 3,256, 256);//reshape层blob格式
    /* Forward dimension change to all layers. */
    net_->Reshape();//reshape

    std::vector<cv::Mat> input_channels;
    WrapInputLayer(&input_channels);

    caffePreprocess(img, &input_channels);//准备输入数据

    net_->Forward();//前向
    caffe::Blob<float>* output_layer = net_->output_blobs()[0];
    float* results = output_layer->mutable_cpu_data();
    Mat src =Mat::zeros(cv::Size(256,256),CV_32FC1);
   //caffe::Datum;
    for(int j=0;j<256;j++){
       float* data =src.ptr<float>(j);
          for(int i=0;i<256;i++){
             if(results[i+j*256+256*256]>results[i+j*256])
              {
                 data[i]=255;
               }
    }
  }
 return src;
}

int main(int argc, char *argv[])
{
                static const string fcnType = "fcn32s";
                String fcnmodelTxt = fcnType + "-heavy-pascal.prototxt";
                String fcnmodelBin = fcnType + "-heavy-pascal.caffemodel";
                Detector fcndetector(fcnmodelTxt,fcnmodelBin);
                Mat frame=imread("test.jpg");
                resize(frame,frame,Size(256,256));
                 clock_t doublestart,doubleend;
                 doublestart = clock();
                Mat seg=fcndetector.Predict(frame);
  		doubleend = clock();
  	        double dur = (double)(doubleend - doublestart);
               cout<<"                     fcn use time:"<<dur/CLOCKS_PER_SEC<<endl;
               imwrite("fcn result.jpg",seg);


    return 0;
}
