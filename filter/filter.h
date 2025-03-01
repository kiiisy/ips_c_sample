#pragma once

#include <cstdint>
#include <opencv2/opencv.hpp>

using namespace cv;

namespace filter {

#define FILTER_SIZE 3

enum class IpsType
{
    EqualizationFilter  = 0,
    WeightedAverage     = 1,
    SharpeningFilter    = 2,
    EdgeDetectionFilter = 3,
    SobelFilter         = 4,
    PrewittFilter       = 5,
    RobertsFilter       = 6,
    EmbossingFilter     = 7,
    None                = 99
};

class ImageProcessor
{
public:
    void equalizationFilter(Mat inImg, int32_t height, int32_t width, int32_t filterCoeff, Mat outImg);
    void weightedAverageFilter(Mat inImg, int32_t height, int32_t width, Mat outImg);
    void sharpeningFilter(Mat inImg, int32_t height, int32_t width, Mat outImg);
    void edgeDetectionFilter(Mat inImg, int32_t height, int32_t width, Mat outImg);
    void sobelFilter(Mat inImg, int32_t height, int32_t width, Mat outImg);
    void prewittFilter(Mat inImg, int32_t height, int32_t width, Mat outImg);
    void robertsFilter(Mat inImg, int32_t height, int32_t width, Mat outImg);
    void embossingFilter(Mat inImg, int32_t height, int32_t width, Mat outImg);
};

}  // namespace filter
