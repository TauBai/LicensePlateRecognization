#ifndef PLATE_H
#define PLATE_H

#include <opencv2/opencv.hpp>
#include <vector>
class plate
{
public:
    plate() = default;
    plate(cv::Mat img) : src(img){}
    plate(cv::Mat img, std::vector<cv::Rect> roi) : src(img),roi(roi){}

    void setCharsDataPath(const char* a){if(a) svm_xml_path = a;}
    cv::Mat plates_found() const { return result;}   //return img Mat
    std::string plates_char() const { return charactors;} //return the charactors
    bool set_debugmode(bool flag) {if(flag) debugflag = 1; else debugflag = 0; return flag;}
    bool set_savemode (bool flag) {if(flag) saveflag = 1; else saveflag = 0; return flag;}
    bool set_roughmode (bool flag) {if(flag) roughmode = 1; else roughmode = 0; return flag;}
    bool set_colorcombinedmode (bool flag) {if(flag) colorcombinedmode = 1; else colorcombinedmode = 0; return flag;}
    cv::Mat img_preprocessing();
    void colorExtractor(const cv::Mat& image,cv::Mat& dst,cv::InputArray lowerb,cv::InputArray upperb);
    enum colors {BLUE=1,YELLOW,WHITE,BLACK};
    //fun colorExractor return the mask;
    int color = BLUE;

    cv::Mat colorExtractor(cv::Mat& image, cv::Mat& dst,int c = BLUE);
    cv::Mat plates_roughlocate();
    std::vector<cv::Mat> plates_locate();
    cv::Mat floodFillSeg(cv::Mat img);
    void plates_recognize();
    cv::Mat platesLocateByColor();
    cv::Mat platesPerspectiveTrans(cv::Rect rect);
    std::vector<char32_t> charRecognization(cv::Mat img);
    cv::Mat charSegmentPreprocessing(cv::Mat &img);
    cv::Mat charSegmentVProjection(cv::Mat img);

    cv::Mat ProjectedHistogram(cv::Mat img, int t);
    cv::Mat createCharFeatures(cv::Mat in, int sizeData);

    std::vector<int> charSegmentHProjection(cv::Mat img);
    std::vector<std::vector<cv::Mat> > vProjectionBasedDissection(cv::Mat hist,
                                                                  cv::Mat img);
    void plates_save(cv::Mat img, std::string s);   //save the string of plate char with its affined picture, better in excel , 添加图片批注, 添加密码
private:

    const char* svm_xml_path = "/home/tau/Documents/SVM/SVM_xml_.xml";
    mutable cv::Ptr<cv::ml::SVM> svm_ ;
    cv::Mat src;
    cv::Rect src_rect_roi;
    cv::Mat result;
    cv::Mat plate_morpholo;
    std::string charactors;
    std::vector<cv::Rect> roi;
    cv::Mat sobelThreshold;
    std::vector<cv::Rect> rois;
    mutable bool saveflag = 1;    //save the plates' information
    mutable bool debugflag = 1;   //
    mutable bool roughmode = 0;   // rough dectection mode for efficience
    mutable bool colorcombinedmode =1; //combine the color information
    int sampleSize =20;
    const std::vector<char32_t> charBox = {
                             '藏','4','3','贵','川','T','黑','Y','苏','辽','粤',
                             '豫','渝','闽','鄂','吉','晋','1','新','赣','6','桂',
                             'Q','鲁','甘','琼','S','Z','N','云','京',
                             'K','湘','A','M','蒙','C','D','5',
                             '陕','P','G','冀','津','宁','E','R','皖','青','F','8','0','H','L',
                            'W','B','V','7','浙','X','湖','J','2','9','U'};

};

bool roughSizes(cv::Rect rect);
struct LinePolar
{
        float rho;
        float angle;
};

void HoughLinesPeak(LinePolar& linepolar,
                     std::vector<cv::Point> linepoint, cv::Size size,
                    float rho, float theta,
                    double min_theta, double max_theta );
void ransacLines(std::vector<cv::Point>& input,std::vector<cv::Vec4d>& lines,
            double distance = 3,  unsigned int ngon = 4,unsigned int itmax = 500);

#endif // PLATE_H
