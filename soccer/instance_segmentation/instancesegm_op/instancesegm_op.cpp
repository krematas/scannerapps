#include "instancesegm.pb.h"            // for ResizeArgs (generated file)
#include "scanner/api/kernel.h"   // for VideoKernel and REGISTER_KERNEL
#include "scanner/api/op.h"       // for REGISTER_OP
#include "scanner/util/memory.h"  // for device-independent memory management
#include "scanner/util/opencv.h"  // for using OpenCV
#include "scanner/util/serialize.h"

#include <opencv2/ximgproc.hpp>

#include <iostream>               // for std::cout

#include <Eigen/Sparse>
typedef float var_t;
typedef Eigen::SparseMatrix<var_t> SpMat;
typedef Eigen::Triplet<var_t> T;
typedef char byte;

#include <ctime>
#include "opencv2/highgui.hpp"
#include "opencv2/core/utility.hpp"

void getPixelNeighbors(int height, int width, std::vector<std::vector<int>>& neighborId){

    for(int i=0; i<height; i++){
        for(int j=0; j<width; j++){

            if(i == 0){
                neighborId[i*width+j].push_back((i+1)*width+j);
            }else if(i == height-1){
                neighborId[i*width+j].push_back((i-1)*width+j);
            }else{
                neighborId[i*width+j].push_back((i+1)*width+j);
                neighborId[i*width+j].push_back((i-1)*width+j);
            }

            if(j == 0){
                neighborId[i*width+j].push_back(i*width+j+1);
            }else if(j == width-1){
                neighborId[i*width+j].push_back(i*width+j-1);
            }else{
                neighborId[i*width+j].push_back(i*width+j+1);
                neighborId[i*width+j].push_back(i*width+j-1);
            }

        }
    }

}


void getLabelPosition(var_t *img, int h, int w, std::map<int, std::vector<int>>& ht){
    for(int i=0; i<h; i++) {
        for (int j = 0; j < w; j++) {

            if(img[i*w+j] >= 1.0){
                int lbl = int(img[i*w+j]-1);
                ht[lbl].push_back(i*w+j);
            }

        }
    }
}


SpMat setU(int N, std::map<int, std::vector<int>>& ht, Eigen::VectorXf& y){
    std::vector<T> tripletList;

    for(std::map<int,std::vector<int>>::iterator it = ht.begin(); it != ht.end(); ++it) {
        std::vector<int> pixLocation = it->second;
        for(int i=0; i<pixLocation.size();i++){
            tripletList.push_back(T(pixLocation[i], pixLocation[i], 1.));
            y[pixLocation[i]] = float(it->first);
        }
    }

    SpMat U(N,N);
    U.setFromTriplets(tripletList.begin(), tripletList.end());
    return U;
}

void setDW(var_t* image, var_t* edges, int h, int w, std::vector<SpMat>& out, float sigma1, float sigma2){

    int N = h * w;
    std::vector<std::vector<int>> neighborId(N);
    getPixelNeighbors(h, w, neighborId);

    std::vector<T> tripletListD;
    std::vector<T> tripletListW;
    int M = 0;

    for(int i=0; i<neighborId.size(); i++){
        int x, y;
        y = i/w;
        x = i%w;
        var_t r1 = image[y*w*3+x*3+0];
        var_t g1 = image[y*w*3+x*3+1];
        var_t b1 = image[y*w*3+x*3+2];
        var_t e1 = edges[y*w+x];

        for(int j=0; j<neighborId[i].size(); j++){

            y = neighborId[i][j]/w;
            x = neighborId[i][j]%w;
            var_t r2 = image[y*w*3+x*3+0];
            var_t g2 = image[y*w*3+x*3+1];
            var_t b2 = image[y*w*3+x*3+2];

            var_t weight0 = exp(-((r1-r2)*(r1-r2) + (g1-g2)*(g1-g2)+ (b1-b2)*(b1-b2))/sigma1);
            var_t weight1 = exp(-(e1*e1)/sigma2);

            // std::cout<<weight0<<" "<<weight1<<std::endl;

            tripletListD.push_back(T(M, i, 1.));
            tripletListD.push_back(T(M, neighborId[i][j], -1.));
            tripletListW.push_back(T(M, M, weight0*weight1));

            M++;

        }
    }

    SpMat D(M, N);
    D.setFromTriplets(tripletListD.begin(), tripletListD.end());

    SpMat W(M, M);
    W.setFromTriplets(tripletListW.begin(), tripletListW.end());
    out.push_back(D);
    out.push_back(W);
}


var_t* segmentFromPoses(var_t *img, var_t *edges, var_t *poseData, int height, int width, float sigma1, float sigma2){
    std::map<int, std::vector<int>> ht;
    getLabelPosition(poseData, height, width, ht);
    Eigen::VectorXf y(height*width);
    SpMat U = setU(height*width, ht, y);
    std::vector<SpMat> DW;
    setDW(img, edges, height, width, DW, sigma1, sigma2);
    SpMat D = DW[0];
    SpMat W = DW[1];

    Eigen::VectorXf b = U*y;

    SpMat A = U + D.transpose()*W*D;

    Eigen::SimplicialCholesky <SpMat> solver(A);
    Eigen::VectorXf x = solver.solve(b);

    var_t *output = new var_t[height*width];
    for(int i=0; i<height*width; i++)
        output[i] = x[i];

    return output;
}



class InstanceSegmentKernel : public scanner::Kernel, public scanner::VideoKernel {
 public:
  InstanceSegmentKernel(const scanner::KernelConfig& config)
      : scanner::Kernel(config) {
    auto start = scanner::now();
    MySegmentArgs args;
    args.ParseFromArray(config.args.data(), config.args.size());
    sigma1 = args.sigma1();
    sigma2 = args.sigma2();
    if (profiler_) {
      profiler_->add_interval("constructor", start, scanner::now());
    }
  }

  void execute(const scanner::Elements& input_columns,
               scanner::Elements& output_columns) override {

    auto start = scanner::now();

    auto& img_col = input_columns[0];
    auto& poseimg_col = input_columns[1];
    auto& edge_col = input_columns[2];

    check_frame(scanner::CPU_DEVICE, img_col);
    check_frame(scanner::CPU_DEVICE, poseimg_col);
    check_frame(scanner::CPU_DEVICE, edge_col);


    // Image
    const scanner::Frame* frame = img_col.as_const_frame();
    cv::Mat image = scanner::frame_to_mat(frame);

    image.convertTo(image, cv::DataType<var_t>::type, 1.0/255.0);
    var_t *imgData = (var_t*)(image.data);

    // Pose image
    const scanner::Frame* pose_img_frame = poseimg_col.as_const_frame();
    cv::Mat poseImage = scanner::frame_to_mat(pose_img_frame);

    poseImage.convertTo(poseImage, cv::DataType<var_t>::type);
    var_t *poseData = (var_t*)(poseImage.data);


    // Edges image
    const scanner::Frame* edge_frame = edge_col.as_const_frame();
    cv::Mat edges = scanner::frame_to_mat(edge_frame);
//    cv::Mat edges;
//    cv::cvtColor(_edges, edges, CV_BGR2GRAY);

    edges.convertTo(edges, cv::DataType<var_t>::type, 1.0/255.0);
    var_t *edgesData = (var_t*)(edges.data);



//std::cout<<image.channels()<<std::endl;
//std::cout<<poseImage.channels()<<std::endl;
//std::cout<<edges.channels()<<std::endl;

    // Segmentation part
    start = scanner::now();

    int height = image.rows;
    int width = image.cols;

//    for(int i=0; i<height; i++) {
//        for (int j = 0; j < width; j++) {
//        if(i == 10){
//        std::cout<<edgesData[i*width+j]<<std::endl;
//        }
//        }
//        }

//cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE );// Create a window for display.
//    cv::imshow( "Display window", edges );                   // Show our image inside it.
//    cv::waitKey(0);

    var_t* segm_output = segmentFromPoses(imgData, edgesData, poseData, height, width, sigma1, sigma2);
     if (profiler_) {
      profiler_->add_interval("segmentFromPoses", start, scanner::now());
    }

    start = scanner::now();
    cv::Mat new_mask(height, width, CV_8U);
    for(int i=0; i<height; i++) {
        for (int j = 0; j < width; j++) {
            if(segm_output[i*width+j] > 1.5)
                new_mask.at<uchar>(i,j) = 255;
            else
                new_mask.at<uchar>(i,j) = 0;
        }
    }

    ProtoImage proto_image;
    int size = new_mask.total() * new_mask.elemSize();
    byte * bytes = new byte[size];  // you will have to delete[] that later
    std::memcpy(bytes, new_mask.data, size * sizeof(byte));
    proto_image.set_image_data(bytes, size * sizeof(byte));
    proto_image.set_h(height);
    proto_image.set_w(width);
    delete []bytes;


    size_t size2 = proto_image.ByteSize();
    scanner::u8* buffer = scanner::new_buffer(scanner::CPU_DEVICE, size2);
    proto_image.SerializeToArray(buffer, size2);

    scanner::insert_element(output_columns[0], buffer, size2);
    if (profiler_) {
      profiler_->add_interval("final part", start, scanner::now());
    }
  }

 private:
  cv::Ptr<cv::ximgproc::StructuredEdgeDetection> pDollar_;
  float sigma1;
  float sigma2;
};

// These functions run statically when the shared library is loaded to tell the
// Scanner runtime about your custom op.

REGISTER_OP(InstanceSegment).frame_input("frame").frame_input("poseimg").frame_input("edges").output("frame").protobuf_name("MySegmentArgs");

REGISTER_KERNEL(InstanceSegment, InstanceSegmentKernel)
    .device(scanner::DeviceType::CPU)
    .num_devices(1);
