#include "distance.pb.h"            // for ResizeArgs (generated file)
#include "scanner/api/kernel.h"   // for VideoKernel and REGISTER_KERNEL
#include "scanner/api/op.h"       // for REGISTER_OP
#include "scanner/util/memory.h"  // for device-independent memory management
#include "scanner/util/opencv.h"  // for using OpenCV

#include <iostream>               // for std::cout
#include <math.h>       /* sqrt */


#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/core/utility.hpp"

/*
 * Ops in Scanner are abstract units of computation that are implemented by
 * kernels. Kernels are pinned to a specific device (CPU or GPU). Here, we
 * implement a custom op to resize an image. After reading this file, look
 * at CMakeLists.txt for how to build the op.
 */

// Custom kernels must inherit the Kernel class or any subclass thereof,
// e.g. the VideoKernel which provides support for processing video frames.
class DistanceTransformKernel : public scanner::Kernel, public scanner::VideoKernel {
 public:
  // To allow ops to be customized by users at a runtime, e.g. to define the
  // target width and height of the MyResizeKernel, Scanner uses Google's Protocol
  // Buffers, or protobufs, to define serialzable types usable in C++ and
  // Python (see resize_op/args.proto). By convention, ops that take
  // arguments must define a protobuf called <OpName>Args, e.g. ResizeArgs,
  // In Python, users will provide the argument fields to the op constructor,
  // and these will get serialized into a string. This string is part of the
  // general configuration each kernel receives from the runtime, config.args.
  DistanceTransformKernel(const scanner::KernelConfig& config)
      : scanner::Kernel(config) {
    // The protobuf arguments must be decoded from the input string.
    DistanceTransformArgs args;
    args.ParseFromArray(config.args.data(), config.args.size());
    width_ = args.w();
    height_ = args.h();
    edge_sfactor = args.edge_sfactor();

    lsd = cv::createLineSegmentDetector(0);
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

    auto& resized_frame_col = output_columns[0];
    scanner::FrameInfo output_frame_info(height_, width_, 1, scanner::FrameType::F32);

    const scanner::Frame* image_frame = frame_col.as_const_frame();
    cv::Mat image = scanner::frame_to_mat(image_frame);

    const scanner::Frame* mask_frame = mask_col.as_const_frame();
    cv::Mat mask = scanner::frame_to_mat(mask_frame);

    // =================================================================================================================
    cv::Mat grayimage, blur_gray, edges;
    cv::cvtColor(image, grayimage, cv::COLOR_RGB2GRAY);
    cv::GaussianBlur( grayimage, blur_gray, cv::Size( 5, 5 ), 0);
    cv::Canny(blur_gray, edges, 10, 200, 5);

    std::vector<cv::Vec4f> lines_std;
    lsd->detect(edges, lines_std);

    std::vector<cv::Vec4f> long_lines;
    for(int i = 0; i < lines_std.size(); i++){
        float norm = sqrt((lines_std[i][0]-lines_std[i][2])*(lines_std[i][0]-lines_std[i][2]) + (lines_std[i][1]-lines_std[i][3])*(lines_std[i][1]-lines_std[i][3]));
        if(norm > 50.){
            long_lines.push_back(lines_std[i]);
        }
    }

    cv::Mat edge_lines(height_, width_, CV_64FC3);
    lsd->drawSegments(edge_lines, long_lines);


    std::vector<cv::Mat> channels(3);
    cv:split(edge_lines, channels);

    cv::Mat element = cv::getStructuringElement( 0, cv::Size( 7, 7 ));
    cv::morphologyEx( channels[2], channels[2], 3, element );
    element = cv::getStructuringElement( 0, cv::Size( 3, 3 ));
    cv::morphologyEx( channels[2], channels[2], 2, element );

    channels[2].convertTo(channels[2], CV_8U);
    cv::ximgproc::thinning(channels[2] ,  channels[2]);

    cv::Mat dist_transf;


//    cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE );// Create a window for display.
//    cv::imshow( "Display window", dist_transf );                   // Show our image inside it.
//    cv::waitKey(0);


    // Allocate a frame for the resized output frame
    scanner::Frame* resized_frame = scanner::new_frame(scanner::CPU_DEVICE, output_frame_info);
    cv::Mat output = scanner::frame_to_mat(resized_frame);

    cv::distanceTransform(255-channels[2], output, CV_DIST_L2, 0);
//    cv::resize(dist_transf, output, cv::Size(width_, height_));

    scanner::insert_frame(resized_frame_col, resized_frame);
  }

 private:
  int width_;
  int height_;
  float edge_sfactor;

  cv::Ptr<cv::LineSegmentDetector> lsd;
};

// These functions run statically when the shared library is loaded to tell the
// Scanner runtime about your custom op.

REGISTER_OP(DistanceTransform).frame_input("frame").frame_input("mask").frame_output("frame").protobuf_name("DistanceTransformArgs");

REGISTER_KERNEL(DistanceTransform, DistanceTransformKernel)
    .device(scanner::DeviceType::CPU)
    .num_devices(1);