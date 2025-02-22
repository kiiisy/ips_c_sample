#pragma once

#include <opencv2/opencv.hpp>

using namespace cv;

namespace pixelwise {

enum class IpsType
{
    ToneCurve        = 0,
    Linear           = 1,
    Nega             = 2,
    Gamma            = 3,
    Sigmoid          = 4,
    HistEqualization = 5,
    None             = 99
};

class ImageProcessor
{
public:
    void toneCurve(Mat inImg, int32_t height, int32_t width, double coeff, Mat outImg);
    void effectLinear(Mat inImg, int32_t height, int32_t width, double a, double b, Mat outImg);
    void effectNega(Mat inImg, int32_t height, int32_t width, Mat outImg);
    void effectGamma(Mat inImg, int32_t height, int32_t width, double gammaVal, Mat outImg);
    void effectSigmoid(Mat inImg, int32_t height, int32_t width, double k, double x0, Mat outImg);
    void calcNormHist(Mat inImg, int32_t height, int32_t width, float *hist);
    void histEqualization(Mat inImg, int32_t height, int32_t width, Mat outImg);
};

}  // namespace pixelwise