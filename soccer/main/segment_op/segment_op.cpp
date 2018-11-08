#include "segment.pb.h"            // for ResizeArgs (generated file)
#include "scanner/api/kernel.h"   // for VideoKernel and REGISTER_KERNEL
#include "scanner/api/op.h"       // for REGISTER_OP
#include "scanner/util/memory.h"  // for device-independent memory management
#include "scanner/util/opencv.h"  // for using OpenCV

#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/core/utility.hpp"

#include <iostream>               // for std::cout
//#include "build/segment.pb.h"

#include <Eigen/Sparse>
typedef float var_t;
typedef Eigen::SparseMatrix<var_t> SpMat;
typedef Eigen::Triplet<var_t> T;

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


/*
 * Ops in Scanner are abstract units of computation that are implemented by
 * kernels. Kernels are pinned to a specific device (CPU or GPU). Here, we
 * implement a custom op to resize an image. After reading this file, look
 * at CMakeLists.txt for how to build the op.
 */

// Custom kernels must inherit the Kernel class or any subclass thereof,
// e.g. the VideoKernel which provides support for processing video frames.
class MySegmentKernel : public scanner::Kernel, public scanner::VideoKernel {
 public:
  // To allow ops to be customized by users at a runtime, e.g. to define the
  // target width and height of the MyResizeKernel, Scanner uses Google's Protocol
  // Buffers, or protobufs, to define serialzable types usable in C++ and
  // Python (see resize_op/args.proto). By convention, ops that take
  // arguments must define a protobuf called <OpName>Args, e.g. ResizeArgs,
  // In Python, users will provide the argument fields to the op constructor,
  // and these will get serialized into a string. This string is part of the
  // general configuration each kernel receives from the runtime, config.args.
  MySegmentKernel(const scanner::KernelConfig& config)
      : scanner::Kernel(config) {
    // The protobuf arguments must be decoded from the input string.
    MySegmentArgs args;
    args.ParseFromArray(config.args.data(), config.args.size());
    width_ = args.w();
    height_ = args.h();
    sigma1 = args.sigma1();
    sigma2 = args.sigma2();
    pDollar_ = cv::ximgproc::createStructuredEdgeDetection(
        args.model_path());
  }

  // Execute is the core computation of the kernel. It maps a batch of rows
  // from an input table to a batch of rows of the output table. Here, we map
  // from one input column from the video, "frame", and return
  // a single column, "frame".
  void execute(const scanner::Elements& input_columns,
               scanner::Elements& output_columns) override {
    auto& frame_col = input_columns[0];
    auto& mask_col = input_columns[1];

    // This must be called at the top of the execute method in any VideoKernel.
    // See the VideoKernel for the implementation check_frame_info.
    check_frame(scanner::CPU_DEVICE, frame_col);
    check_frame(scanner::CPU_DEVICE, mask_col);

    MyImage proto_image;
    proto_image.ParseFromArray(frame_col.buffer, frame_col.size);

    MyImage proto_mask;
    proto_mask.ParseFromArray(mask_col.buffer, mask_col.size);
//    std::cout<<frame_col.size << " - "<< mask_col.size<<std::endl;

    auto& resized_frame_col = output_columns[0];
    scanner::FrameInfo output_frame_info(height_, width_, 3, scanner::FrameType::U8);


    std::vector<uint8_t> bytes_img(proto_image.image_data().begin(), proto_image.image_data().end());
    cv::Mat image = cv::imdecode(bytes_img, 1);

    std::vector<uint8_t> bytes_pose(proto_mask.image_data().begin(), proto_mask.image_data().end());
    cv::Mat poseImage = cv::imdecode(bytes_pose, 0);



//    const scanner::Frame* mask = mask_col.as_const_frame();
//    cv::Mat poseImage = scanner::frame_to_mat(mask);
//
    image.convertTo(image, cv::DataType<var_t>::type, 1.0/255.0);
    var_t *imgData = (var_t*)(image.data);

//    std::cout<<image.channels()<< " - "<<image.rows<<std::endl;
//    cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE );// Create a window for display.
//    cv::imshow( "Display window", image );                   // Show our image inside it.
//    cv::waitKey(0);

    poseImage.convertTo(poseImage, cv::DataType<var_t>::type);
    var_t *poseData = (var_t*)(poseImage.data);
//    std::cout<<poseImage.channels()<<std::endl;
    //
    cv::Mat img2;
    image.copyTo(img2);
    img2.convertTo(img2, cv::DataType<var_t>::type);
    cv::Mat edges(img2.size(), img2.type());

    pDollar_->detectEdges(img2, edges);
//    std::cout<<edges.channels()<<std::endl;

//    std::cout<<" -------------------------- "<<std::endl<<std::endl;
    int height = image.rows;
    int width = image.cols;

    var_t *edgesData = (var_t*)(edges.data);

    var_t* segm_output = segmentFromPoses(imgData, edgesData, poseData, height, width, sigma1, sigma2);

    // cv::Mat new_mask(height, width, cv::DataType<var_t>::type, segm_output);

    //copy vector to mat
    // std::cout<<edges.size()<<edges.type()<<std::endl;

    cv::Mat new_mask(height, width, CV_8U);
    for(int i=0; i<height; i++) {
        for (int j = 0; j < width; j++) {
          // std::cout<<segm_output[i*width+j]<<std::endl;
            if(segm_output[i*width+j] > 1.5)
                new_mask.at<uchar>(i,j) = 255;
            else
                new_mask.at<uchar>(i,j) = 0;
        }
    }


    // new_mask.convertTo(new_mask, CV_8UC3, 255.0);
    // cv::Mat output_img;
    // edges.convertTo(output_img, cv::DataType<uint8>::type);
    cv::cvtColor(new_mask, new_mask, cv::COLOR_GRAY2BGR);

    // Allocate a frame for the resized output frame
    scanner::Frame* resized_frame = scanner::new_frame(scanner::CPU_DEVICE, output_frame_info);
    cv::Mat output = scanner::frame_to_mat(resized_frame);

    cv::resize(new_mask, output, cv::Size(width_, height_));

    image.convertTo(image, CV_8UC3);

//    std::cout<<image.rows<< " - "<<output.rows<<std::endl;
//    cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE );// Create a window for display.
//    cv::imshow( "Display window", output );                   // Show our image inside it.
//    cv::waitKey(0);


    scanner::insert_frame(resized_frame_col, resized_frame);
  }

 private:
  cv::Ptr<cv::ximgproc::StructuredEdgeDetection> pDollar_;
  int width_;
  int height_;
  float sigma1;
  float sigma2;
};

// These functions run statically when the shared library is loaded to tell the
// Scanner runtime about your custom op.

REGISTER_OP(MySegment).frame_input("frame").frame_input("mask").frame_output("frame").protobuf_name("MySegmentArgs");

REGISTER_KERNEL(MySegment, MySegmentKernel)
    .device(scanner::DeviceType::CPU)
    .num_devices(1);
