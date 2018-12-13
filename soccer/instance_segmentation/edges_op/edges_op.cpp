#include "edges.pb.h"            // for ResizeArgs (generated file)
#include "scanner/api/kernel.h"   // for VideoKernel and REGISTER_KERNEL
#include "scanner/api/op.h"       // for REGISTER_OP
#include "scanner/util/memory.h"  // for device-independent memory management
#include "scanner/util/opencv.h"  // for using OpenCV
#include "scanner/util/serialize.h"

#include <opencv2/ximgproc.hpp>

#include <iostream>

typedef float var_t;
typedef char byte;


class EdgeDetectionKernel : public scanner::Kernel, public scanner::VideoKernel {
 public:

  EdgeDetectionKernel(const scanner::KernelConfig& config)
      : scanner::Kernel(config) {

    MySegmentArgs args;
    args.ParseFromArray(config.args.data(), config.args.size());
    auto start = scanner::now();
    pDollar_ = cv::ximgproc::createStructuredEdgeDetection(args.model_path());
    if (profiler_) {
      profiler_->add_interval("constructor", start, scanner::now());
    }
  }


  void execute(const scanner::Elements& input_columns,
               scanner::Elements& output_columns) override {

    auto start = scanner::now();
    auto& frame_col = input_columns[0];

    check_frame(scanner::CPU_DEVICE, frame_col);

    const scanner::Frame* frame = frame_col.as_const_frame();
    cv::Mat image = scanner::frame_to_mat(frame);


    image.convertTo(image, cv::DataType<var_t>::type, 1.0/255.0);
    var_t *imgData = (var_t*)(image.data);

    cv::Mat img2;
    image.copyTo(img2);
    img2.convertTo(img2, cv::DataType<var_t>::type);
    cv::Mat edges(img2.size(), img2.type());

    if (profiler_) {
      profiler_->add_interval("edge detect Initialization", start, scanner::now());
    }


    start = scanner::now();
    pDollar_->detectEdges(img2, edges);
    if (profiler_) {
      profiler_->add_interval("edge detect main", start, scanner::now());
    }

    start = scanner::now();

    int height = image.rows;
    int width = image.cols;


    ProtoImage proto_image;
    int size = edges.total() * edges.elemSize();
    byte * bytes = new byte[size];  // you will have to delete[] that later
    std::memcpy(bytes, edges.data, size * sizeof(byte));
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
};

REGISTER_OP(EdgeDetection).frame_input("frame").output("frame").protobuf_name("MySegmentArgs");

REGISTER_KERNEL(EdgeDetection, EdgeDetectionKernel)
    .device(scanner::DeviceType::CPU)
    .num_devices(1);
