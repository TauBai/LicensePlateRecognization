#define BOOST_FILESYSTEM_VERSION 3

//  As an example program, we don't want to use any deprecated features
#ifndef BOOST_FILESYSTEM_NO_DEPRECATED
#  define BOOST_FILESYSTEM_NO_DEPRECATED
#endif
#ifndef BOOST_SYSTEM_NO_DEPRECATED
#  define BOOST_SYSTEM_NO_DEPRECATED
#endif

#include "boost/filesystem/operations.hpp"
#include "boost/filesystem/path.hpp"
#include "boost/progress.hpp"
#include <iostream>

#include <opencv2/opencv.hpp>
namespace fs = boost::filesystem;
using namespace cv;

const char* chars_folder_ = "/home/tau/Documents/SVM/Input";
//const char* SVM_xml_ = "/home/tau/Documents/SVM/SVM_xml_.xml";
const char* SVM_xml_ = "/home/tau/Documents/SVM/0.xml";

bool getSubDIRfilenames(const char* chars_folder_, std::vector<std::string>& filename,
                                                   std::vector<int>& labels,
                                                   std::vector<std::string>& corfoldername);

Mat ProjectedHistogram(Mat img, int t);
Mat features(Mat in, int sizeData);
int main(int argc, char** argv)
{
    assert(chars_folder_);
    cv::Mat samples;
    std::vector<int> labels;
    std::cout << "Collecting chars in " << chars_folder_ << std::endl;
    std::vector<std::string> filename;    //all the files correspondent foldername
    std::vector<std::string> foldername;
    getSubDIRfilenames(chars_folder_,filename,labels,foldername);


    for (auto s : filename){
        auto img = cv::imread(s, 0);  // a grayscale image
        if(img.size() != cv::Size(20,20))
            cv::resize(img,img,cv::Size(20,20));
        cv::Mat f = features(img, 5);
        f.resize(1);
        //samples.push_back(img);
        samples.push_back(f);
    }

    cv::Mat samples_;
    samples.convertTo(samples_, CV_32F);
    //cv::Mat train_classes(labels.size(),foldername.size(), CV_32F,cv::Scalar(0));
/*    cv::Mat train_classes(samples_.rows,foldername.size(),CV_32F,cv::Scalar(0));
    for (int i = 0; i < train_classes.rows; ++i) {
      train_classes.at<float>(i, labels[i])= 1.f;
    }*/
    cv::Mat train_classes;
    for(auto a : labels)
        train_classes.push_back(a);
    train_classes.resize(labels.size(),0);

    cv::Ptr<cv::ml::TrainData> trainData =
            cv::ml::TrainData::create(samples_, cv::ml::SampleTypes::ROW_SAMPLE,
                                      labels);


    /**************************************************************/
    cv::Ptr<cv::ml::SVM> svm_ = cv::ml::SVM::create();
    svm_->setType(cv::ml::SVM::C_SVC);
    svm_->setC(0.1);
    svm_->setKernel(cv::ml::SVM::RBF);
//    svm_->setTermCriteria(cv::TermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER,
//                                           (int)1e7, 1e-6 ));
    std::cout << "Training SVM , please wait..." << std::endl;

    //svm_->trainAuto(trainData);
    svm_ ->train(trainData);
    //svm_->train(samples_,cv::ml::ROW_SAMPLE,labels);
    //svm_->train(samples_,cv::ml::ROW_SAMPLE,train_classes);
    std::cout << "training process finished " << std::endl;
    svm_->save(SVM_xml_);
    std::cout << "SVM data saved" << SVM_xml_ << std::endl;
    /****************************************************************/
}


bool getSubDIRfilenames(const char* chars_folder_, std::vector<std::string>& filename,
                                                   std::vector<int>& labels,
                                                   std::vector<std::string>& foldername){
    std::cout << "getSubDIR .. " << std::endl;
    fs::path p(fs::current_path());
    if (chars_folder_)
        p = fs::system_complete(chars_folder_);
    else
        std::cout << "\nusage:   simple_ls [path]" << std::endl;

    unsigned long file_count = 0;
    unsigned long dir_count = 0;
    unsigned long other_count = 0;
    unsigned long err_count = 0;

    //if (!fs::exists(p)){
    //    std::cout << "\nNot found: " << p << std::endl;
    //}else
    //    return 0;

    if (fs::is_directory(p)){
        std::cout << "\nIn directory: " << p << "\n\n";
        fs::directory_iterator end_iter;
        for (fs::directory_iterator dir_itr(p);dir_itr != end_iter;++dir_itr){
            if (fs::is_directory(dir_itr->status())){
                std::string s = dir_itr->path().string();

                size_t found = s.find_first_not_of(chars_folder_);
                if (found != std::string::npos)
                   s.erase(s.begin(),s.begin() + found);
                 else
                   s.erase(s.begin(),s.begin()+ sizeof(chars_folder_));
                foldername.push_back(s);                       //get foldername


                std::cout << s << " [directory]\n";
                fs::path subP = fs::system_complete(*dir_itr);
                if(fs::is_directory(subP)){
                    fs::directory_iterator end_iter;
                    for(fs::directory_iterator dir_itr(subP);
                                    dir_itr != end_iter;++dir_itr){
                        if(fs::is_regular_file(dir_itr->status())){
                            ++file_count;
                            std::string sFileName = dir_itr->path().string();
                            std::cout << sFileName << "\n";
                            filename.push_back(sFileName);                //get filename
                            labels.push_back(dir_count);                  //get label
                        }
                    }
                }else if(fs::is_regular_file(dir_itr->status())){
                    ++file_count;
                    std::cout << dir_itr->path().filename() << "\n";
                }else{
                    ++other_count;
                    std::cout << dir_itr->path().filename() << " [other]\n";
                }

                ++dir_count;
            }
        }

        std::cout << "\n" << file_count << " files\n"
              << dir_count << " directories\n"
              << other_count << " others\n"
              << err_count << " errors\n";
        return 1;
    }else{
        std::cout << "\nFound: " << p << "\n";
        return 0;
    }
}


Mat ProjectedHistogram(Mat img, int t)
{
    int sz=(t)?img.rows:img.cols;
    Mat mhist=Mat::zeros(1,sz,CV_32F);

    for(int j=0; j<sz; j++){
        Mat data=(t)?img.row(j):img.col(j);
        mhist.at<float>(j)=countNonZero(data);
    }

    //Normalize histogram
    double min, max;
    minMaxLoc(mhist, &min, &max);

    if(max>0)
        mhist.convertTo(mhist,-1 , 1.0f/max, 0);

    return mhist;
}

Mat features(Mat in, int sizeData){
    //Histogram features
    Mat vhist=ProjectedHistogram(in,1);
    Mat hhist=ProjectedHistogram(in,0);

    //Low data feature
    Mat lowData;
    resize(in, lowData, Size(sizeData, sizeData) );


    //Last 10 is the number of moments components
    int numCols=vhist.cols+hhist.cols+lowData.cols*lowData.cols;

    Mat out=Mat::zeros(1,numCols,CV_32F);
    //Asign values to feature
    int j=0;
    for(int i=0; i<vhist.cols; i++)
    {
        out.at<float>(j)=vhist.at<float>(i);
        j++;
    }
    for(int i=0; i<hhist.cols; i++)
    {
        out.at<float>(j)=hhist.at<float>(i);
        j++;
    }
    for(int x=0; x<lowData.cols; x++)
    {
        for(int y=0; y<lowData.rows; y++){
            out.at<float>(j)=(float)lowData.at<uchar>(x,y);
            j++;
        }
    }
    return out;
}
