
#include "plate.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <random>

cv::Mat plate::img_preprocessing(){
    svm_=cv::ml::SVM::load<cv::ml::SVM>(svm_xml_path);
    auto var_cout = svm_ ->getVarCount();
    assert(var_cout);//if 特征向量数为0 ,  可能要么没训练好要么没读好xml
    cv::Mat image(src.size(),CV_8UC3,cv::Scalar(0,0,0)) ;
    //only defined regions to be processed, 80% of the src
    auto x = 0.1 * src.size().width;
    auto y = 0.1 * src.size().height;
    cv::Rect roi(x,y,src.cols - 2*x, src.rows - 2*y);
    src_rect_roi = roi;
    cv::Mat src_roi(src,src_rect_roi);
    cv::Mat image_roi(image,cv::Rect(x,y,src.cols - 2*x, src.rows - 2*y));
    src_roi.copyTo(image_roi);
    cv::GaussianBlur(image, image, cv::Size(3,3), 1.6 , 1.6);// the sigma value?
    if(debugflag)
        cv::imshow("blurring", image);
    return image;
}
void plate::colorExtractor(const cv::Mat& image,cv::Mat& dst,cv::InputArray lowerb,cv::InputArray upperb){
    if(image.channels() == 3){
        cv::Mat img;
        cv::cvtColor(image,img,CV_BGR2HSV);
        cv::Mat mask;
        cv::inRange(img,lowerb,upperb,mask);
        if(debugflag)
            cv::imshow("mask",mask);
        std::vector<cv::Mat> channels(3,cv::Mat());
        std::vector<cv::Mat> dischannels(3,cv::Mat());
        cv::split(image,channels);
        channels[0].copyTo(dischannels[0],mask);
        channels[1].copyTo(dischannels[1],mask);
        channels[2].copyTo(dischannels[2],mask);
        cv::merge(dischannels,dst);
        if(debugflag)
            cv::imshow("extract the color",dst);
    }else std::cout << "Channels error of the input image in colorExtractor" << std::endl;

}
cv::Mat plate::colorExtractor(cv::Mat& image, cv::Mat& dst, int c){
    if(image.empty())
        image = src;
    cv::Mat mask;
    if(image.channels() == 3){
        cv::Mat img;
        cv::cvtColor(image,img,CV_BGR2HSV);

        cv::Scalar lowerb,upperb;
        if(c == BLUE){
            lowerb = cv::Scalar(90,0,25);
            upperb = cv::Scalar(120,255,255);
        }
        if(c == YELLOW){
            lowerb = cv::Scalar(20,25,25);
            upperb = cv::Scalar(40,255,255);
        }
        if(c == WHITE){
            lowerb = cv::Scalar(0,0,220);
            upperb = cv::Scalar(255,60,255);
        }
        if(c == BLACK){
            lowerb = cv::Scalar(0,150,0);
            upperb = cv::Scalar(255,255,25);
        }


        cv::inRange(img,lowerb,upperb,mask);
        if(debugflag)
            cv::imshow("mask",mask);
        std::vector<cv::Mat> channels(3,cv::Mat());
        std::vector<cv::Mat> dischannels(3,cv::Mat());
        cv::split(image,channels);
        channels[0].copyTo(dischannels[0],mask);
        channels[1].copyTo(dischannels[1],mask);
        channels[2].copyTo(dischannels[2],mask);
        cv::merge(dischannels,dst);
        if(debugflag)
            cv::imshow("extract the color",dst);
    }else std::cout << "Channels error of the input image in colorExtractor" << std::endl;
    return mask;
}
cv::Mat plate::plates_roughlocate(){
    //std::vector<cv::RotatedRect> plate_candi;
    cv::Mat img = plate::img_preprocessing();

    if(img.channels() == 3)
        cv::cvtColor(img,img,CV_BGR2GRAY);
    if(img.channels() != 1)
        std::cout << "something wrong with the img channels" << std::endl;

    cv::Mat G,Gx,Gy,G_otsu, G_edge;
    cv::Sobel(img,G,CV_8U,1,0,3);
    if(debugflag)
        cv::imshow("Sobel x",G);

    cv::threshold(G,G_otsu,100,255,cv::THRESH_BINARY+cv::THRESH_OTSU);
    sobelThreshold = G_otsu;

    cv::imshow("threshold of G_otsu",G_otsu);
    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(22, 7) );
    cv::Mat img_close;
    cv::Mat img_morpholo;
    cv::morphologyEx(G_otsu, img_close, CV_MOP_CLOSE, element);
    cv::morphologyEx(img_close, img_morpholo, CV_MOP_OPEN, element);
    plate_morpholo = img_morpholo.clone();
    if(debugflag)
        cv::imshow("morphology 22,7 in roughmode", img_morpholo);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(img_morpholo,contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

    std::vector<cv::Rect> regions;
    auto itc= contours.begin();
    while(itc != contours.end()) {
        //Create bounding rect of object
        cv::Rect rect= cv::boundingRect(cv::Mat(*itc) );

        //int addedOffset =0.4*rect.height;
        int addedOffset = 0;
        rect += cv::Size(addedOffset,addedOffset); // x , y are fixed
        rect.x -= 0.5 * addedOffset; // x, y  are signed value
        rect.y -= 0.5 * addedOffset;
        //check if the rect overlap the boundry
        if( rect.x >= 0 &&
            rect.y >= 0 &&
            rect.x + rect.width <= src.cols &&
            rect.y + rect.height <= src.rows){
                if( !roughSizes(rect)){
                    itc = contours.erase(itc);
                }else{
                    regions.push_back(rect);
                    ++itc;
                }
            }
    }
    //return regions;
    roi = regions;
    //if there is less than one region, it's no need to run plate_locate()
////////////////完全调试好后要把下面的注释删掉//////////////////
/*    if(regions.size() > 1)
        set_roughmode(0);
    else
        set_roughmode(1);
*/
    if(debugflag)
        if(!regions.size())
            std::cout << "no region detected" << std::endl;

    cv::Mat result;
    src.copyTo(result);
    cv::drawContours(result, contours,-1, CV_RGB(0,255,0), 1,
                     cv::LINE_8,cv::noArray(),contours.size(),cvPoint(0,0));
    if(debugflag)
        cv::imshow("drawresults",result);
    return result;
}
std::vector<cv::Mat> plate::plates_locate(){
    cv::Mat img = plates_roughlocate();
    if(roughmode)
        return img;
    if(colorcombinedmode){
        cv::Mat dstcolor;
        cv::Mat mask = colorExtractor(src,dstcolor,plate::color);
        cv::Mat color_vedge;
        cv::bitwise_and(plate_morpholo,plate_morpholo,color_vedge,mask);
        cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15,7));
        cv::morphologyEx(color_vedge,color_vedge,cv::MORPH_CLOSE,element);
        if(debugflag)
            cv::imshow("AND operation to the plate_morpho",color_vedge);
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(color_vedge,contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);

        std::vector<cv::Rect> regions;
        auto itc= contours.begin();
        while(itc != contours.end()) {
            //Create bounding rect of object
            cv::Rect rect= cv::boundingRect(cv::Mat(*itc) );
            int addedOffset = 0;
            //int addedOffset =0.4*rect.height;
            rect += cv::Size(addedOffset,addedOffset); // x , y are fixed
            rect.x -= 0.5 * addedOffset; // x, y  are signed value
            rect.y -= 0.5 * addedOffset;
            //check if the rect overlap the boundry
            if( rect.x >= 0 &&
                rect.y >= 0 &&
                rect.x + rect.width <= src.cols &&
                rect.y + rect.height <= src.rows){
                    if( !roughSizes(rect)){
                        itc = contours.erase(itc);
                    }else{
                        regions.push_back(rect);
                        ++itc;
                    }
                }
        }
        roi = regions;

    }
    std::vector<cv::Mat> plate_candi;

    if(roi.size())
        for(auto iter = roi.begin();iter != roi.end(); ++iter){
            cv::Mat img;
            img = platesPerspectiveTrans(*iter);
            plate_candi.push_back(img);
        }
    return plate_candi;
}
cv::Mat plate::platesLocateByColor(){
    cv::Mat img =  plate::img_preprocessing();
    cv::Mat dst;
    cv::Mat mask = plate::colorExtractor(img,dst,BLUE);
    cv::Mat morpho = mask.clone();
    if(!morpho.empty()){
        cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7,7));
        cv::morphologyEx(morpho,morpho,cv::MORPH_OPEN,element);
        cv::morphologyEx(morpho,morpho,cv::MORPH_CLOSE,element);
        if(debugflag)
            cv::imshow("color morpho",morpho);
    }else std::cout << " morpho empty" <<std::endl;
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(morpho,contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);
    std::vector<cv::Point> contour;
    for(auto a : contours)
        contour.insert(contour.end(),a.begin(), a.end());
    if(debugflag){
        cv::Mat contourMat(morpho.size(),CV_8UC1,cv::Scalar(0));
        for(auto a: contour)
            contourMat.at<uchar>(a) = 255;
        cv::imshow("contour",contourMat);
    }
    return morpho;
}
cv::Mat plate::platesPerspectiveTrans(cv::Rect rect) {

    cv::Mat rectified_dst;
    cv::Mat regionsOfinterest(src,rect);                        //return it

    cv::Mat img = regionsOfinterest.clone();
    cv::GaussianBlur(img,img,cv::Size(5,5),1.6,1.6);
//    cv::Mat mask = floodFillSeg(img); // 注意mask比img宽与高均多2 roi大小需与mask一致
    cv::Mat dst_enlarged(rect.height*2, rect.width *2,CV_8UC3,cv::Scalar(0,0,0));
    cv::Mat dst_enlarged_temp(dst_enlarged,
                              cv::Rect(rect.width - rect.width/2 ,
                                       rect.height - rect.height/2 ,
                                       rect.width,  rect.height));
    img.copyTo(dst_enlarged_temp);
    if(debugflag)
        cv::imshow("dst_enlarged",dst_enlarged);
    cv::Mat tempdst;
    cv::Mat mask = colorExtractor(img,tempdst,BLUE);
    if(debugflag)
        cv::imshow("color mask for perspectivetrans",mask);
    cv::Mat roi_enlarged(rect.height*2, rect.width *2,CV_8UC1,cv::Scalar(0));
    cv::Mat roi_enlarged_temp(roi_enlarged,
                              cv::Rect(rect.width - rect.width/2 ,
                                       rect.height - rect.height/2 ,
                                       rect.width,  rect.height));
    mask.copyTo(roi_enlarged_temp);

//    cv::Mat regionsOfinterestPlusOne(src,rect);

    cv::Mat element1 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5) );
    cv::Mat element2 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(10, 10) );
    cv::morphologyEx(roi_enlarged, roi_enlarged, CV_MOP_OPEN, element1);
    cv::morphologyEx(roi_enlarged, roi_enlarged, CV_MOP_CLOSE, element2);
//    cv::Mat temp = regionsOfinterestPlusOne.clone();
//    cv::bitwise_and(temp,temp,rectified_dst,mask);
    if(debugflag)
        cv::imshow("roi_enlarged",roi_enlarged);

    std::vector<std::vector<cv::Point>> roi_point;
    cv::findContours(roi_enlarged,roi_point,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);

    auto itc= roi_point.begin();
    while(itc != roi_point.end()) {
        //Create bounding rect of object
        if((*itc).size() > 1000){
            itc = roi_point.erase(itc);
            continue;
        }
        cv::Rect rect= cv::boundingRect(*itc );
        //check if the rect overlap the boundry
        if( rect.x >= 0 &&
            rect.y >= 0 &&
            rect.x + rect.width <= src.cols &&
            rect.y + rect.height <= src.rows){
                if( !roughSizes(rect)){
                    itc = roi_point.erase(itc);
                }else{
                    ++itc;
                }
            }
    }
    std::vector<cv::Point> edgepoints;
    for(auto a :roi_point)
        edgepoints.insert(edgepoints.end(),a.begin(), a.end());

    std::vector<cv::Point2f> vertexes;
    double distance = 3;

    cv::Mat dst(roi_enlarged.size(),CV_8UC3,cv::Scalar(0,0,0));

//    std::vector<cv::Vec4d> lines;
//    ransacLines(edgepoints,lines);


    //houghNgon
    LinePolar linepolar;
    std::vector<LinePolar> lines;
    for(int i = 0;i< 4;++i){
        HoughLinesPeak( linepolar,edgepoints,roi_enlarged.size(), 0.5, 0.001, 0., CV_PI );
        lines.push_back(linepolar);
        float rho = lines[i].rho;
        float theta = lines[i].angle;
        double a = cos(theta), b = sin(theta);
        double x0 = a*rho, y0 = b*rho;  // the foot of the line's normal through (0,0)
        cv::Point imax(cvRound(x0 + 1000*(-b)),  cvRound(y0 + 1000*(a)));
        cv::Point jmax(cvRound(x0 - 1000*(-b)),  cvRound(y0 - 1000*(a)));
        cv::line( dst, imax, jmax, cv::Scalar(0,0,255), 1, 8 );
        auto iter = edgepoints.begin();
        while(iter != edgepoints.end()){
            double dis = fabs((jmax.x - imax.x)*((*iter).y - imax.y) -
                                    (jmax.y - imax.y)*((*iter).x - imax.x))
                         / sqrt((jmax.x - imax.x)*(jmax.x - imax.x)
                                 + (jmax.y - imax.y)*(jmax.y - imax.y));
            if(dis < distance)
                iter = edgepoints.erase(iter);  //erase the dis within , then point to
                                           //   the next element
            else ++iter;
        }
    }

    for(size_t i = 0;i < lines.size(); ++i){
        for(size_t j = i+1; j < lines.size();++j){
            cv::Matx22f A(std::cos(lines[i].angle), std::sin(lines[i].angle),
                          std::cos(lines[j].angle), std::sin(lines[j].angle));
            cv::Matx21f b(lines[i].rho,lines[j].rho),x;
            if( std::fabs(lines[i].angle- lines[j].angle) > CV_PI/4){
                cv::solve(A,b,x,cv::DECOMP_LU);         //方程组Ax = b 求x
                cv::Point vertex;
                vertex.x = cvRound(x(0,0));
                vertex.y = cvRound(x(0,1));
                vertexes.push_back(vertex);
            }
        }
    }
    bool outRangeFlag = 0;
    //confirm those points whether all is in the range of image
    for(auto a :vertexes)
        if(a.x < 0 ||
                a.y < 0 ||
                a.x > roi_enlarged.cols ||
                a.y > roi_enlarged.rows)
            outRangeFlag = 1;

    if(vertexes.size() != 4 || outRangeFlag)
        return rectified_dst;
    else{
        // sort vertex
        cv::Point central = cv::Point(0,0);
        for(auto a : vertexes){
            central.x += a.x;
            central.y += a.y;
        }
        central = cv::Point(central.x /4.0 , central.y/4.0);
        std::vector<cv::Point2f> vertexes_sorted(4);

        for(auto a :vertexes){
            if (a.x < central.x && a.y < central.y)
                vertexes_sorted[0] = a;
            if (a.x > central.x && a.y < central.y)
                vertexes_sorted[1] = a;
            if (a.x > central.x && a.y > central.y)
                vertexes_sorted[2] = a;
            if (a.x < central.x && a.y > central.y)
                vertexes_sorted[3] = a;
        }
        for(auto a :vertexes_sorted)  std::cout << a <<std::endl;
        for(auto i : vertexes_sorted)
            cv::circle(dst,i,5,cv::Scalar(0,255,0));
        cv::imshow("houghNgon",dst);
        std::vector<cv::Point2f> warp_dst;
        float standard_ratio = 140./440.;
        float p_width  = vertexes_sorted[1].x - vertexes_sorted[0].x;
        float p_height = p_width * standard_ratio;
        auto i = vertexes_sorted.begin();
/*
        warp_dst.push_back(cv::Point(0,0));
        warp_dst.push_back(cv::Point(p_width , 0));
        warp_dst.push_back(cv::Point(p_width ,p_height));
        warp_dst.push_back(cv::Point(0, p_height));
*/
        warp_dst.push_back(*i);
        warp_dst.push_back(cv::Point((*i).x + p_width , (*i).y));
        warp_dst.push_back(cv::Point((*i).x + p_width ,(*i).y+p_height));
        warp_dst.push_back(cv::Point((*i).x,(*i).y+p_height));

        for(auto a :warp_dst)  std::cout << a <<std::endl;

        cv::Matx33d warp_mat = cv::getPerspectiveTransform(vertexes_sorted,warp_dst);
        //cv::Matx23d warp_mat = cv::getAffineTransform(vertexes_sorted,warp_dst);
        std::cout << warp_mat <<std::endl;


        cv::warpPerspective(dst_enlarged,rectified_dst, warp_mat ,roi_enlarged.size());
        cv::warpPerspective( dst,dst, warp_mat ,roi_enlarged.size());
        //cv::warpAffine( roi_enlarged,rectified_dst, warp_mat ,rectified_dst.size());
        //cv::warpAffine( dst,dst, warp_mat ,roi_enlarged.size());
        if(debugflag)
            cv::imshow("rectified houghlines" ,dst);
        if(debugflag)
            cv::imshow("rectified" ,rectified_dst);

        // Calculates a perspective transform from four pairs of the corresponding points.

        /*
    std::vector<std::vector<cv::Point>> roi_point;
    cv::findContours(roi_enlarged,roi_point,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);
    cv::Mat dst(roi_enlarged.size(),CV_8UC3,cv::Scalar(0,0,0));

    std::vector<cv::Vec4d> lines;
    ransacLines(roi_point[0],lines);
    for( size_t i = 0; i < lines.size(); i++ ){
        cv::line( dst, cv::Point(lines[i][0], lines[i][1]),
        cv::Point(lines[i][2], lines[i][3]), cv::Scalar(0,0,255), 3, 8 );
    }
*/
        /*
    cv::Mat dst(roi_enlarged.size(),CV_8UC3,cv::Scalar(0,0,0));
    std::vector<cv::Vec4f> lines;
    cv::HoughLinesP( roi_enlarged, lines, 1, CV_PI/180, 20, 20, 20);
    std::cout << lines.size() << std::endl;
    for( size_t i = 0; i < lines.size(); i++ ){
        cv::line( dst, cv::Point(lines[i][0], lines[i][1]),
        cv::Point(lines[i][2], lines[i][3]), cv::Scalar(0,0,255), 3, 8 );
    }
    struct thetaLines{
        cv::Vec4f endpoints;
        double theta;
        double length;
    };
    std::vector<thetaLines> lines_theta;
    for(auto a :lines){
        double length =sqrt((a[0]-a[2])*(a[0]-a[2]) + (a[1]-a[3])*(a[1]-a[3]));
        if(length> 20){
            double theta = atan(double(a[1]-a[3])/(a[0]-a[2]));
            thetaLines line;
            line.endpoints = a;
            line.theta =theta;
            line.length = length;
            lines_theta.push_back(line);
        }
    }
    std::vector<thetaLines> quadrangle;
    //找出最符合的四条边
    for(auto it =  lines_theta.begin();it != lines_theta.end(); ++it){
        for(auto ita = it +1; ita != lines_theta.end();++itc ){
            if((*it).theta - (*ita).theta < CV_PI/9){
                double distance = ((*it).endpoints)
            }
        }
    }
*/
        /*
    std::vector<cv::Point> roi_point_approx;
    cv::Mat roi_approx(roi_enlarged.size(),CV_8UC3,cv::Scalar(0,0,0));
    approxPolyDP( roi_enlarged.begin(), roi_point_approx, 7, 1 );

    for(auto a : roi_point_approx)
        cv::circle(roi_approx,a,2,cv::Scalar(0,0,255));
    if(debugflag)
        cv::imshow("roi_approx",roi_approx);
*/
        /*
    cv::Mat dst(roi_enlarged.size(),CV_8UC3,cv::Scalar(0,0,0));
    std::vector<std::vector<cv::Point>> linespoint;
    std::vector<LinePolar> lines;
    std::vector<cv::Point2f> vertexes;

        LinePolar linepolar;
            HoughLinesPeak( linepolar,roi_point,roi_enlarged.size(), 1, CV_PI/500, 0., CV_PI );
            lines.push_back(linepolar);

        //draw all lines detected by hough
        for( size_t i = 0; i < lines.size(); i++ )
        {
            float rho = lines[i].rho;
            float theta = lines[i].angle;
            double a = cos(theta), b = sin(theta);
            double x0 = a*rho, y0 = b*rho;
            cv::Point pt1(cvRound(x0 + 1000*(-b)),  cvRound(y0 + 1000*(a)));
            cv::Point pt2(cvRound(x0 - 1000*(-b)),  cvRound(y0 - 1000*(a)));
            cv::line( dst, pt1, pt2, cv::Scalar(0,0,255), 1, 8 );
        }
        cv::imshow("houghline",dst);
*/

        /*
    cv::HoughLinesP( roi_enlarged, lines, 1, CV_PI/180, 20, 20, 20);
    std::cout << lines.size() << std::endl;
    for( size_t i = 0; i < lines.size(); i++ ){
        cv::line( dst, cv::Point(lines[i][0], lines[i][1]),
        cv::Point(lines[i][2], lines[i][3]), cv::Scalar(0,0,255), 3, 8 );
    }
*/
        /*
    std::vector<cv::Vec2f> lines;
    cv::HoughLines(G_otsu, lines,1, CV_PI/180, 30,0,0);
    for( size_t i = 0; i < lines.size(); i++ ){
        float rho = lines[i][0];
        float theta = lines[i][1];
        double a = cos(theta), b = sin(theta);
        double x0 = a*rho, y0 = b*rho;
        cv::Point pt1(cvRound(x0 + 1000*(-b)), cvRound(y0 + 1000*(a)));
        cv::Point pt2(cvRound(x0 - 1000*(-b)), cvRound(y0 - 1000*(a)));
        cv::line( dst, pt1, pt2, cv::Scalar(0,0,255), 3, 8 );

    }
*/
        return rectified_dst;
    }
}

void plate::plates_recognize(){
    std::vector<cv::Mat> plates_candi = plates_locate();
    if(!plates_candi.empty()){
        for(auto a : plates_candi){
            std::vector<char32_t> s = plate::charRecognization(a);
            if(s.size() == 7){
                //plates_save(a, s);
                std::cout << "plate number is ";
                for(auto a : s)
                    std::cout << a ;
                std::cout << std::endl;
            }
        }
    }
    cv::waitKey();
}
std::vector<char32_t> plate::charRecognization(cv::Mat img){
    std::vector<char32_t> charsDetected;
    if(!img.empty()){
        img = plate::charSegmentPreprocessing(img);
//        cv::Canny(img,img,80,200);
        //cv::GaussianBlur(img,img,cv::Size(7,7),1.);
        cv::adaptiveThreshold(img,img,255,CV_ADAPTIVE_THRESH_GAUSSIAN_C,
                              CV_THRESH_BINARY,7,0);
        if(debugflag)
            cv::imshow("img for char dissection",img);

        //delete long straight lines
        std::vector<cv::Vec4i> lines4del;
        cv::HoughLinesP(img,lines4del,0.5,CV_PI/180,img.rows,img.rows,1.5);
        for(auto a : lines4del)
            cv::line(img,cv::Point(a[0],a[1]),cv::Point(a[2],a[3]),cv::Scalar(0),2);
        if(debugflag)
            cv::imshow("del lines by hough using cv::Line",img);

        plate::charSegmentHProjection(img);
        cv::Mat vHist = plate::charSegmentVProjection(img);
        std::vector<std::vector<cv::Mat>> chars =
                plate::vProjectionBasedDissection(vHist,img);

        for(auto itc=chars.begin(); itc != chars.end(); ++itc){
            cv::Mat chs;
            cv::Mat label;
            std::vector<int> labels;
            for(auto ch : *itc){
                cv::resize(ch,ch,cv::Size(sampleSize,sampleSize));
                ch = plate::createCharFeatures(ch,sampleSize);
                chs.push_back(ch);
            }            
            chs.convertTo(chs,CV_32F);
            svm_ ->predict(chs,label,cv::ml::StatModel::RAW_OUTPUT);
            if(label.rows == 7){
                for(auto it = label.begin<float>();it !=label.end<float>();++it){
                    charsDetected.push_back(charBox[static_cast<int>(*it)]);
                    break;
                }
            }

        }
    }
    return charsDetected;
}
cv::Mat plate::charSegmentVProjection(cv::Mat img){
    std::vector<int> data (img.cols, 0);
    //memcpy(data,0,img.cols);
    for(auto it = img.begin<uchar>();it != img.end<uchar>();++it){
        auto pos = (it-img.begin<uchar>())%img.step;
        if(*it && ( pos < img.cols) )
            ++data[pos];
    }
    cv::Mat hist(img.size(),CV_8UC1,cv::Scalar(0));
    int col = 0;
    for(auto a : data){
        if(a){
            int top2ceil = img.rows - a;
            for(int i = top2ceil ; i < img.rows; ++i )
                hist.at<uchar>(i,col) = 255;
        }
        ++col;
    }

    if(debugflag)
        cv::imshow("Vhist",hist);
    return hist;
}
std::vector<int> plate::charSegmentHProjection(cv::Mat img){
    std::vector<int> data (img.rows, 0);
    for(auto it = img.begin<uchar>();it != img.end<uchar>();++it){
        auto pos = (it-img.begin<uchar>())/img.step;
        if(*it && ( pos < img.rows) )
            ++data[pos];
    }
    cv::Mat hist(img.size(),CV_8UC1,cv::Scalar(0));
    int row = 0;
    for(auto a : data){
        if(a){
            for(int i = 0 ; i < a; ++i )
                hist.at<uchar>(row,i) = 255;
        }
        ++row;
    }
    if(debugflag)
        cv::imshow("Hhist",hist);
    return data;
}
std::vector<std::vector<cv::Mat>> plate::vProjectionBasedDissection(
                                      cv::Mat hist,cv::Mat img){
    std::vector<std::vector<cv::Mat>> detectedchar;
    cv::Mat vHist= plate::charSegmentVProjection(hist);
    struct slice{
        std::vector<cv::Vec2i> dissectPos;     //the starting/end point of char
        std::vector<int> ScanLine ;          //
        unsigned int charNum = 0;
    };
    std::vector<slice> vSlice;
    int maxWidth = 0 ;
    for(int row = vHist.rows -1; row >0 ; --row){
        slice scan;
        for(int col=0 ; col < vHist.cols; ++col){
            uchar t = vHist.at<uchar>(row,col);
            if(t != 0)
                scan.ScanLine.push_back(col);
        }
        if(!scan.ScanLine.empty()){
            auto pstart = scan.ScanLine.begin();
            auto pend = scan.ScanLine.rbegin();
            int width = *pend - *pstart ;
            if( width > vHist.cols /5 )
                vSlice.push_back(scan);
            if(width > maxWidth)
                maxWidth = width;
        }
    }
    std::vector<slice> candi;
    auto it = vSlice.begin();
    while(it != vSlice.end()){
        slice sli;
        //auto pstart = (*it).ScanLine.begin();
        //auto pend = (*it).ScanLine.rbegin();
        //int width = *pend - *pstart ;
        //if(width>)
        {               //筛选出符合宽度的slice
            int lastPos = *(*it).ScanLine.begin();
            int start = *(*it).ScanLine.begin();
            for(auto a : (*it).ScanLine){
                if(lastPos != a -1 ){                 //如果出现不连续
                    //如果像素为0连续超过2格,认为是空白部分
                    if( 2 < a - lastPos){
                        int wid =lastPos - start;     //字符宽度
                        if( wid > maxWidth/20 &&    //字符宽度范围
                                wid < vHist.cols/7.0){
                            sli.dissectPos.push_back(cv::Vec2i(start,lastPos));
                            ++sli.charNum;
                            //debug
                            //std::cout << start <<','<< lastPos<<
                              //        '.'<<wid<<std::endl;
                        }
                    }
                    start = a;                  //更新字符开始位置
                }
                lastPos = a;
            }
            if(sli.charNum >=7 && sli.charNum <=10)
                candi.push_back(sli);
            //debug
            //for(auto a : (*it).ScanLine)   std::cout << a;
            //std::cout <<std::endl;
        }
            ++it;
    }

    std::string s = "slice ";
    for(auto a : candi){
        std::string ss;
        std::vector<cv::Mat> pchar;
        for(auto b:a.dissectPos){
            char count = '0';
            int col1 = b[0];
            int col2 = b[1];
            if(count > '9')
                ss = s + '1' +count;
            cv::Mat dissect = img.colRange(col1,col2);
            if(!img.empty()){
                if(debugflag)
                    cv::imshow(ss,dissect);
                cv::waitKey();
                pchar.push_back(dissect);
            }
            ++count;
        }
        detectedchar.push_back(pchar);
    }
    return detectedchar;
}

cv::Mat plate::charSegmentPreprocessing(cv::Mat& img){
    if(img.channels() == 3)
        cv::cvtColor(img,img,CV_BGR2GRAY);
    cv::medianBlur(img,img,3);
    if(debugflag)
        cv::imshow("medianBlur for charSegment", img);

    return img;
}


cv::Mat plate::ProjectedHistogram(cv::Mat img, int t){
    ///本函数来源   mastering opencv with practical projects
    int sz=(t)?img.rows:img.cols;
    cv::Mat mhist=cv::Mat::zeros(1,sz,CV_32F);

    for(int j=0; j<sz; j++){
        cv::Mat data=(t)?img.row(j):img.col(j);
        mhist.at<float>(j)=countNonZero(data);
    }

    //Normalize histogram
    double min, max;
    cv::minMaxLoc(mhist, &min, &max);

    if(max>0)
        mhist.convertTo(mhist,-1 , 1.0f/max, 0);

    return mhist;
}

cv::Mat plate::createCharFeatures(cv::Mat in, int sizeData){
    ///本函数来源   mastering opencv with practical projects
    //Histogram features
    cv::Mat vhist=ProjectedHistogram(in,1);
    cv::Mat hhist=ProjectedHistogram(in,0);

    //Low data feature
    cv::Mat lowData;
    cv::resize(in, lowData, cv::Size(sizeData, sizeData) );


    //Last 10 is the number of moments components
    int numCols=vhist.cols+hhist.cols+lowData.cols*lowData.cols;

    cv::Mat out=cv::Mat::zeros(1,numCols,CV_32F);
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
            out.at<float>(j)=(float)lowData.at<unsigned char>(x,y);
            j++;
        }
    }
    return out;
}


void plate::plates_save(cv::Mat img, std::string s){
////保存车牌, 整车图片, 保存车牌号码
}


bool roughSizes(cv::Rect rect){
    float tolerance = 0.6;
    // car plate size: 44x14 aspect 3.14286
    float ratio = 3.14286;
    //Set a min and max area. All other patchs are discarded
    float min = 20 * ratio * 20; // minimum area
    float max = 80 * ratio * 80; // maximum area
    float rmin = ratio - ratio * tolerance;
    float rmax = ratio + ratio * tolerance;

    float area = rect.area();
    float r = (float)rect.width / (float)rect.height;// ratio of width/height

    if(( area < min || area > max ) || ( r < rmin || r > rmax )){
        return false;
    }else{
        return true;
    }

}


void HoughLinesPeak( LinePolar& linepolar,
                     std::vector<cv::Point> linepoint,cv::Size size,
                    float rho, float theta,
                    double min_theta, double max_theta )
{
    float irho = 1 / rho;

    int width = size.width;
    int height = size.height;


    if (max_theta < min_theta ) {
        CV_Error( CV_StsBadArg, "max_theta must be greater than min_theta" );
    }
    int numangle = cvRound((max_theta - min_theta) / theta);
    int numrho = cvRound(((width + height) * 2 + 1) / rho);

    cv::AutoBuffer<int> _accum((numangle+2) * (numrho+2));
    cv::AutoBuffer<float> _tabSin(numangle);
    cv::AutoBuffer<float> _tabCos(numangle);
    int *accum = _accum;
    float *tabSin = _tabSin, *tabCos = _tabCos;


    memset( accum, 0, sizeof(accum[0]) * (numangle+2) * (numrho+2) );


    float ang = static_cast<float>(min_theta);
    for(int n = 0; n < numangle; ang += theta, n++ )
    {
        tabSin[n] = (float)(sin((double)ang) * irho);
        tabCos[n] = (float)(cos((double)ang) * irho);
    }


    // stage 1. fill accumulator
    for(auto i = linepoint.begin(); i !=  linepoint.end(); ++i){
        for(int n = 0; n < numangle; n++ ){
            int r = cvRound( (*i).x * tabCos[n] + (*i).y * tabSin[n] );              //  ρ = x cos θ + y sin θ
            r += (numrho - 1) / 2;
            accum[(n+1) * (numrho+2) + r+1]++;
        }
    }

    // stage 2. finding peak
    int peak = 0 , rpeak = 0, npeak = 0;

    for(int r = 0; r < numrho; r++ )
        for(int n = 0; n < numangle; n++ ){
            int base = (n+1) * (numrho+2) + r+1;

            if( accum[base] > peak ){
                peak = accum[base];
                rpeak = r;
                npeak = n;
            }
        }
    linepolar.rho =   (rpeak - (numrho - 1)*0.5f) * rho;
    linepolar.angle = static_cast<float>(min_theta) + npeak * theta;
}
void ransacLines(std::vector<cv::Point>& input,std::vector<cv::Vec4d>& lines,
                    double distance ,  unsigned int ngon,unsigned int itmax ){

    if(!input.empty())
    for(unsigned i = 0; i < ngon; ++i){
        srand((unsigned) time(NULL));
        unsigned int Mmax = 0;
        cv::Point imax;
        cv::Point jmax;
        cv::Vec4d line;
        size_t t1 , t2;
        std::random_device rd; // only used once to initialise (seed) engine
        std::mt19937 rng(rd());// random-number engine used (Mersenne-Twister in this case)
        std::uniform_int_distribution<int> uni(0,input.size()-1); // guaranteed unbiased

        unsigned int it = itmax;
        while(--it){
            t1 = uni(rng);
            t2 = uni(rng);
            t2 = (t1 == t2 ? uni(rng): t2);
            unsigned int M = 0;
            cv::Point i = input[t1];
            cv::Point j = input[t2];
            for(auto a : input){
                double dis = fabs((j.x - i.x)*(a.y - i.y) - (j.y - i.y)*(a.x - i.x))
                             /sqrt((j.x - i.x)*(j.x - i.x) + (j.y - i.y)*(j.y - i.y));

                if( dis < distance)
                    ++M;
            }
            if(M > Mmax ){
                Mmax = M;
                imax = i;
                jmax = j;
            }
        }
        line[0] = imax.x;
        line[1] = imax.y;
        line[2] = jmax.x;
        line[3] = jmax.y;
        lines.push_back(line);
        auto iter = input.begin();
        while(iter != input.end()){
            double dis = fabs((jmax.x - imax.x)*((*iter).y - imax.y) -
                                    (jmax.y - imax.y)*((*iter).x - imax.x))
                         / sqrt((jmax.x - imax.x)*(jmax.x - imax.x)
                                 + (jmax.y - imax.y)*(jmax.y - imax.y));
            if(dis < distance)
                iter = input.erase(iter);  //
            // Draw blue contours on a white imageerase the dis within , then point to
                                           //   the next element
            else ++iter;
        }
    }
    else std::cout << "no input to ransac" << std::endl;
}
cv::Mat plate::floodFillSeg(cv::Mat img){
//来源 Mastering Opencv , modified by tau
// only for specific condition.
    cv::Point imgcenter = cv::Point(img.cols/2,img.rows/2);
    float minhei = img.rows/2;
    float minwid = img.cols/2;
    minhei -= 0.3*minhei;
    minwid -= 0.3*minwid;
    std::random_device rd; // only used once to initialise (seed) engine
    std::mt19937 rng(rd());// random-number engine used (Mersenne-Twister in this case)
    std::uniform_int_distribution<int> nhei(-int(0.6*minhei),-int(minhei*0.2));
    std::uniform_int_distribution<int> nwid(-int(0.6*minwid),-int(minwid*0.2));
    std::uniform_int_distribution<int> phei(int(0.2*minhei),int(0.6*minhei));
    std::uniform_int_distribution<int> pwid(int(0.2*minwid),int(minwid*0.6));
    cv::Mat mask(img.rows + 2, img.cols + 2, CV_8UC1,cv::Scalar(0));
    int upDiff = 100;
    int loDiff = 1000;
    int connectivity = 4;
    int newMaskVal = 255;
    int NumSeeds = 40;
    cv::Rect ccomp;
    int flags = connectivity + (newMaskVal << 8 ) + CV_FLOODFILL_FIXED_RANGE + CV_FLOODFILL_MASK_ONLY;
    for(int j=0; j<NumSeeds; j++){
        cv::Point seed;
        seed.x = imgcenter.x + nwid(rng);
        seed.y = imgcenter.y + nhei(rng);
        cv::floodFill(img, mask, seed, cv::Scalar(0,0,255), &ccomp,
                      cv::Scalar(loDiff, loDiff, loDiff),
                      cv::Scalar(upDiff, upDiff, upDiff), flags);
        seed.x = imgcenter.x + pwid(rng);
        seed.y = imgcenter.y + phei(rng);
        cv::floodFill(img, mask, seed, cv::Scalar(0,0,255), &ccomp,
                      cv::Scalar(loDiff, loDiff, loDiff),
                      cv::Scalar(upDiff, upDiff, upDiff), flags);
    }
    return mask;

}
