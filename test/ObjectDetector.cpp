#include "ObjectDetector.hpp"  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/imgproc.hpp>  
#include <vector>  
#include <fstream>  
  
using std::string;  
using std::vector;  
using namespace caffe;  
using  std::max;  
using std::min;  
  
  
ObjectDetector::ObjectDetector(const std::string &model_file,const std::string &weights_file){  
#ifdef CPU_ONLY  
    Caffe::set_mode(Caffe::CPU);  
#else  
    Caffe::set_mode(Caffe::GPU);  
#endif   
    net_.reset(new Net<float>(model_file, TEST));  
    net_->CopyTrainedLayersFrom(weights_file);  
    this->class_num_ = net_->blob_by_name("cls_prob")->channels();  //求得类别数+1  
}  
  
//对一张图片，进行检测，将结果保存进map数据结构里,分别表示每个类别对应的目标框，如果需要分数信息，则计算分数  
map<int,vector<cv::Rect> > ObjectDetector::detect(const cv::Mat& image,map<int,vector<float> >* objectScore){  
  
    if(objectScore!=NULL)   //如果需要保存置信度  
        objectScore->clear();  
  
    float CONF_THRESH = 0.8;  //置信度阈值  
    float NMS_THRESH = 0.3;   //非极大值抑制阈值  
    int max_side = max(image.rows, image.cols);   //分别求出图片宽和高的较大者  
    int min_side = min(image.rows, image.cols);  
    float max_side_scale = float(max_side) / float(INPUT_SIZE_LONG);    //分别求出缩放因子  
    float min_side_scale = float(min_side) / float(INPUT_SIZE_NARROW);  
    float max_scale = max(max_side_scale, min_side_scale);  
  
    float img_scale = float(1) / max_scale;  
    int height = int(image.rows * img_scale);  
    int width = int(image.cols * img_scale);  
  
    int num_out;  
    cv::Mat cv_resized;  
    image.convertTo(cv_resized, CV_32FC3);  
    cv::resize(cv_resized, cv_resized, cv::Size(width, height));   
    cv::Mat mean(height, width, cv_resized.type(), cv::Scalar(102.9801, 115.9465, 122.7717));  
    cv::Mat normalized;  
    subtract(cv_resized, mean, normalized);  
  
    float im_info[3];  
    im_info[0] = height;  
    im_info[1] = width;  
    im_info[2] = img_scale;  
    shared_ptr<Blob<float> > input_layer = net_->blob_by_name("data");  
    input_layer->Reshape(1, normalized.channels(), height, width);  
    net_->Reshape();  
    float* input_data = input_layer->mutable_cpu_data();  
    vector<cv::Mat> input_channels;  
    for (int i = 0; i < input_layer->channels(); ++i) {  
        cv::Mat channel(height, width, CV_32FC1, input_data);  
        input_channels.push_back(channel);  
        input_data += height * width;  
    }  
    cv::split(normalized, input_channels);  
    net_->blob_by_name("im_info")->set_cpu_data(im_info);  
    net_->Forward();                                       //进行网络前向传播  
  
  
    int num = net_->blob_by_name("rois")->num();    //产生的 ROI 个数,比如为 13949个ROI  
    const float *rois_data = net_->blob_by_name("rois")->cpu_data();    //维度比如为：13949*5*1*1  
    int num1 = net_->blob_by_name("bbox_pred")->num();   //预测的矩形框 维度为 13949*84  
    cv::Mat rois_box(num, 4, CV_32FC1);  
    for (int i = 0; i < num; ++i)  
    {  
        rois_box.at<float>(i, 0) = rois_data[i * 5 + 1] / img_scale;  
        rois_box.at<float>(i, 1) = rois_data[i * 5 + 2] / img_scale;  
        rois_box.at<float>(i, 2) = rois_data[i * 5 + 3] / img_scale;  
        rois_box.at<float>(i, 3) = rois_data[i * 5 + 4] / img_scale;  
    }  
  
    shared_ptr<Blob<float> > bbox_delt_data = net_->blob_by_name("bbox_pred");   // 13949*84  
    shared_ptr<Blob<float> > score = net_->blob_by_name("cls_prob");             // 3949*21  
  
    map<int,vector<cv::Rect> > label_objs;    //每个类别，对应的检测目标框  
    for (int i = 1; i < class_num_; ++i){     //对每个类，进行遍历  
        cv::Mat bbox_delt(num, 4, CV_32FC1);  
        for (int j = 0; j < num; ++j){  
            bbox_delt.at<float>(j, 0) = bbox_delt_data->data_at(j, i * 4 + 0, 0, 0);  
            bbox_delt.at<float>(j, 1) = bbox_delt_data->data_at(j, i * 4 + 1, 0, 0);  
            bbox_delt.at<float>(j, 2) = bbox_delt_data->data_at(j, i * 4 + 2, 0, 0);  
            bbox_delt.at<float>(j, 3) = bbox_delt_data->data_at(j, i * 4 + 3, 0, 0);  
        }  
        cv::Mat box_class = RPN::bbox_tranform_inv(rois_box, bbox_delt);  
  
        vector<RPN::abox> aboxes;   //对于 类别i，检测出的矩形框保存在这  
        for (int j = 0; j < box_class.rows; ++j){  
            if (box_class.at<float>(j, 0) < 0)  box_class.at<float>(j, 0) = 0;  
            if (box_class.at<float>(j, 0) > (image.cols - 1))   box_class.at<float>(j, 0) = image.cols - 1;  
            if (box_class.at<float>(j, 2) < 0)  box_class.at<float>(j, 2) = 0;  
            if (box_class.at<float>(j, 2) > (image.cols - 1))   box_class.at<float>(j, 2) = image.cols - 1;  
  
            if (box_class.at<float>(j, 1) < 0)  box_class.at<float>(j, 1) = 0;  
            if (box_class.at<float>(j, 1) > (image.rows - 1))   box_class.at<float>(j, 1) = image.rows - 1;  
            if (box_class.at<float>(j, 3) < 0)  box_class.at<float>(j, 3) = 0;  
            if (box_class.at<float>(j, 3) > (image.rows - 1))   box_class.at<float>(j, 3) = image.rows - 1;  
            RPN::abox tmp;  
            tmp.x1 = box_class.at<float>(j, 0);  
            tmp.y1 = box_class.at<float>(j, 1);  
            tmp.x2 = box_class.at<float>(j, 2);  
            tmp.y2 = box_class.at<float>(j, 3);  
            tmp.score = score->data_at(j, i, 0, 0);  
            aboxes.push_back(tmp);  
        }  
        std::sort(aboxes.rbegin(), aboxes.rend());  
        RPN::nms(aboxes, NMS_THRESH);  //与非极大值抑制消除对于的矩形框  
        for (int k = 0; k < aboxes.size();){  
            if (aboxes[k].score < CONF_THRESH)  
                aboxes.erase(aboxes.begin() + k);  
            else  
                k++;  
        }  
        //################ 将类别i的所有检测框，保存  
        vector<cv::Rect> rect(aboxes.size());    //对于类别i，检测出的矩形框  
        for(int ii=0;ii<aboxes.size();++ii)  
            rect[ii]=cv::Rect(cv::Point(aboxes[ii].x1,aboxes[ii].y1),cv::Point(aboxes[ii].x2,aboxes[ii].y2));  
        label_objs[i]=rect;     
        //################ 将类别i的所有检测框的打分，保存  
        if(objectScore!=NULL){           //################ 将类别i的所有检测框的打分，保存  
            vector<float> tmp(aboxes.size());       //对于 类别i，检测出的矩形框的得分  
            for(int ii=0;ii<aboxes.size();++ii)  
                tmp[ii]=aboxes[ii].score;  
            objectScore->insert(pair<int,vector<float> >(i,tmp));  
        }  
    }  
    return label_objs;  
}  