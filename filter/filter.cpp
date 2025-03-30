#include "filter.h"
#include "../param.h"
#include <algorithm>
#include <array>
#include <iostream>
#include <numeric>

namespace filter {
/*************************************************
 * void equalizationFilter(Mat inImg, int32_t height, int32_t width, uint16_t filterCoeff, Mat outImg)
 * Mat inImg : 入力画像
 * int32_t height : 高さ
 * int32_t width : 横幅
 * uint16_t filterCoeff : フィルタ係数
 * Mat outImg : 出力画像
 *
 * 機能 : 平滑化フィルタ処理
 *
 * return : void
 *************************************************/
void ImageProcessor::equalizationFilter(Mat inImg, int32_t height, int32_t width, int32_t filterCoeff, Mat outImg)
{
    int32_t filterSize = (2 * filterCoeff + 1) * (2 * filterCoeff + 1);
    int32_t redSum, greenSum, blueSum;

    for (int32_t y = 0; y < height; y++) {
        for (int32_t x = 0; x < width; x++) {
            redSum = greenSum = blueSum = 0;

            // fiterCoeff x filterCoeffのフィルタ処理
            for (int32_t yy = -filterCoeff; yy <= filterCoeff; yy++) {
                for (int32_t xx = -filterCoeff; xx <= filterCoeff; xx++) {
                    // 画像の端の処理 (リピート)
                    int32_t replicateY = std::clamp(y + yy, 0, height - 1);
                    int32_t replicateX = std::clamp(x + xx, 0, width - 1);

                    // 画素取得
                    Vec3b &pix = inImg.at<Vec3b>(replicateY, replicateX);

                    // 画素値の総和
                    redSum += pix[RED];
                    greenSum += pix[GREEN];
                    blueSum += pix[BLUE];
                }
            }
            // 画素値の平均化
            uint8_t red   = static_cast<uint8_t>(std::clamp(redSum / filterSize, 0, 255));
            uint8_t green = static_cast<uint8_t>(std::clamp(greenSum / filterSize, 0, 255));
            uint8_t blue  = static_cast<uint8_t>(std::clamp(blueSum / filterSize, 0, 255));

            // 画素の書き込み
            outImg.at<Vec3b>(y, x) = Vec3b(blue, green, red);
        }
    }
}

/*************************************************
 * void weightedAverageFilter(Mat inImg, int32_t height, int32_t width, Mat outImg)
 * Mat inImg : 入力画像
 * int32_t height : 高さ
 * int32_t width : 横幅
 * Mat outImg : 出力画像
 *
 * 機能 : 加重平均フィルタ処理
 *
 * return : void
 *************************************************/
void ImageProcessor::weightedAverageFilter(Mat inImg, int32_t height, int32_t width, Mat outImg)
{
    int32_t filter[3][3] = {
        {1, 2, 1},
        {2, 8, 2},
        {1, 2, 1}
    };  // 加重平均フィルタ
    int32_t filterSum = std::reduce(&filter[0][0], &filter[0][0] + 3 * 3);
    int32_t redSum, greenSum, blueSum;

    for (int32_t y = 0; y < height; y++) {
        for (int32_t x = 0; x < width; x++) {
            redSum = greenSum = blueSum = 0;

            // 3x3のフィルタ処理(-1から1までの範囲)
            for (int32_t yy = -1; yy <= 1; yy++) {
                for (int32_t xx = -1; xx <= 1; xx++) {
                    // 画像の端の処理 (リピート)
                    int32_t replicateY = std::clamp(y + yy, 0, height - 1);
                    int32_t replicateX = std::clamp(x + xx, 0, width - 1);

                    // 画素取得
                    Vec3b &pix = inImg.at<Vec3b>(replicateY, replicateX);

                    // 画素値とフィルタの積和
                    redSum += filter[yy + 1][xx + 1] * pix[RED];
                    greenSum += filter[yy + 1][xx + 1] * pix[GREEN];
                    blueSum += filter[yy + 1][xx + 1] * pix[BLUE];
                }
            }
            // 画素値の正規化、0~255に収める
            uint8_t red   = static_cast<uint8_t>(std::clamp(redSum / filterSum, 0, 255));
            uint8_t green = static_cast<uint8_t>(std::clamp(greenSum / filterSum, 0, 255));
            uint8_t blue  = static_cast<uint8_t>(std::clamp(blueSum / filterSum, 0, 255));

            // 画素の書き込み
            outImg.at<Vec3b>(y, x) = Vec3b(blue, green, red);
        }
    }
}

/*************************************************
 * void sharpeningFilter(Mat inImg, int32_t height, int32_t width, Mat outImg)
 * Mat inImg : 入力画像
 * int32_t height : 高さ
 * int32_t width : 横幅
 * Mat outImg : 出力画像
 *
 * 機能 : 先鋭化フィルタ処理
 *
 * return : void
 *************************************************/
void ImageProcessor::sharpeningFilter(Mat inImg, int32_t height, int32_t width, Mat outImg)
{
    int32_t filter[3][3] = {
        {0,  -1, 0 },
        {-1, 5,  -1},
        {0,  -1, 0 }
    };  // 先鋭化フィルタ(4近傍)
    int32_t redSum, greenSum, blueSum;

    for (int32_t y = 0; y < height; y++) {
        for (int32_t x = 0; x < width; x++) {
            redSum = greenSum = blueSum = 0;

            // 3x3のフィルタ処理(-1から1までの範囲)
            for (int32_t yy = -1; yy <= 1; yy++) {
                for (int32_t xx = -1; xx <= 1; xx++) {
                    // 画像の端の処理 (リピート)
                    int32_t replicateY = std::clamp(y + yy, 0, height - 1);
                    int32_t replicateX = std::clamp(x + xx, 0, width - 1);

                    // 画素取得
                    Vec3b &pix = inImg.at<Vec3b>(replicateY, replicateX);

                    // 画素値とフィルタの積和
                    redSum += filter[yy + 1][xx + 1] * pix[RED];
                    greenSum += filter[yy + 1][xx + 1] * pix[GREEN];
                    blueSum += filter[yy + 1][xx + 1] * pix[BLUE];
                }
            }
            // 画像の輝度値を0~255に収める
            uint8_t red   = static_cast<uint8_t>(std::clamp(redSum, 0, 255));
            uint8_t green = static_cast<uint8_t>(std::clamp(greenSum, 0, 255));
            uint8_t blue  = static_cast<uint8_t>(std::clamp(blueSum, 0, 255));

            // 画素の書き込み
            outImg.at<Vec3b>(y, x) = Vec3b(blue, green, red);
        }
    }
}

/*************************************************
 * void edgeDetectionFilter(Mat inImg, int32_t height, int32_t width, Mat outImg)
 * Mat inImg : 入力画像
 * int height : 高さ
 * int width : 横幅
 * Mat outImg : 出力画像
 *
 * 機能 : エッジ検出フィルタ処理
 *
 * return : void
 *************************************************/
void ImageProcessor::edgeDetectionFilter(Mat inImg, int32_t height, int32_t width, Mat outImg)
{
    int32_t filter1[3][3] = {
        {0, 0, 0 },
        {1, 0, -1},
        {0, 0, 0 }
    };  // 横フィルタ
    int32_t filter2[3][3] = {
        {0, 1,  0},
        {0, 0,  0},
        {0, -1, 0}
    };  // 縦フィルタ
    int32_t rrx, ggx, bbx;
    int32_t rry, ggy, bby;

    for (int32_t y = 0; y < height; y++) {
        for (int32_t x = 0; x < width; x++) {
            rrx = ggx = bbx = 0;
            rry = ggy = bby = 0;

            // 3x3のフィルタ処理
            for (int32_t yy = 0; yy < FILTER_SIZE; yy++) {
                for (int32_t xx = 0; xx < FILTER_SIZE; xx++) {
                    // 画像の端の処理 (リピート)
                    int32_t replicateY = std::clamp(y + yy - 1, 0, height - 1);
                    int32_t replicateX = std::clamp(x + xx - 1, 0, width - 1);

                    // 画素取得
                    Vec3b &pix = inImg.at<Vec3b>(replicateY, replicateX);

                    // 画素値と横フィルタの積和
                    rrx += pix[RED] * filter1[yy][xx];
                    ggx += pix[GREEN] * filter1[yy][xx];
                    bbx += pix[BLUE] * filter1[yy][xx];

                    // 画素値と縦フィルタの積和
                    rry += pix[RED] * filter2[yy][xx];
                    ggy += pix[GREEN] * filter2[yy][xx];
                    bby += pix[BLUE] * filter2[yy][xx];
                }
            }

            // 画素値の平方根
            double red   = sqrt(static_cast<double>(rrx * rrx + rry * rry));
            double green = sqrt(static_cast<double>(ggx * ggx + ggy * ggy));
            double blue  = sqrt(static_cast<double>(bbx * bbx + bby * bby));

            // 画像の輝度値を0~255に収める
            uint8_t red8   = static_cast<uint8_t>(std::clamp(red, 0., 255.));
            uint8_t green8 = static_cast<uint8_t>(std::clamp(green, 0., 255.));
            uint8_t blue8  = static_cast<uint8_t>(std::clamp(blue, 0., 255.));

            // 画素の書き込み
            outImg.at<Vec3b>(y, x) = Vec3b(blue8, green8, red8);
        }
    }
}

/*************************************************
 * void sobelFilter(Mat inImg, int32_t height, int32_t width, Mat outImg)
 * Mat inImg : 入力画像
 * int height : 高さ
 * int width : 横幅
 * Mat outImg : 出力画像
 *
 * 機能 : Sobelフィルタを用いてエッジを抽出する
 *
 * return : void
 *************************************************/
void ImageProcessor::sobelFilter(Mat inImg, int32_t height, int32_t width, Mat outImg)
{
    int32_t filter1[3][3] = {
        {1,  2,  1 },
        {0,  0,  0 },
        {-1, -2, -1}
    };  // 横方向のSobelフィルタ
    int32_t filter2[3][3] = {
        {1, 0, -1},
        {2, 0, -2},
        {1, 0, -1}
    };  // 縦方向のSobelフィルタ
    int32_t rrx, ggx, bbx;
    int32_t rry, ggy, bby;

    for (int32_t y = 0; y < height; y++) {
        for (int32_t x = 0; x < width; x++) {
            rrx = ggx = bbx = 0;
            rry = ggy = bby = 0;

            // 3x3のフィルタ処理
            for (int32_t yy = 0; yy < FILTER_SIZE; yy++) {
                for (int32_t xx = 0; xx < FILTER_SIZE; xx++) {
                    // 画像の端の処理 (リピート)
                    int32_t replicateY = std::clamp(y + yy - 1, 0, height - 1);
                    int32_t replicateX = std::clamp(x + xx - 1, 0, width - 1);

                    // 画素取得
                    Vec3b &pix = inImg.at<Vec3b>(replicateY, replicateX);

                    // 画素値と横フィルタの積和
                    rrx += pix[RED] * filter1[yy][xx];
                    ggx += pix[GREEN] * filter1[yy][xx];
                    bbx += pix[BLUE] * filter1[yy][xx];

                    // 画素値と縦フィルタの積和
                    rry += pix[RED] * filter2[yy][xx];
                    ggy += pix[GREEN] * filter2[yy][xx];
                    bby += pix[BLUE] * filter2[yy][xx];
                }
            }

            // 画素値の平方根
            double red   = sqrt(static_cast<double>(rrx * rrx + rry * rry));
            double green = sqrt(static_cast<double>(ggx * ggx + ggy * ggy));
            double blue  = sqrt(static_cast<double>(bbx * bbx + bby * bby));

            // 画像の輝度値を0~255に収める
            uint8_t red8   = static_cast<uint8_t>(std::clamp(red, 0., 255.));
            uint8_t green8 = static_cast<uint8_t>(std::clamp(green, 0., 255.));
            uint8_t blue8  = static_cast<uint8_t>(std::clamp(blue, 0., 255.));

            // 画素の書き込み
            outImg.at<Vec3b>(y, x) = Vec3b(blue8, green8, red8);
        }
    }
}

/*************************************************
 * void prewittFilter(Mat inImg, int32_t height, int32_t width, Mat outImg)
 * Mat inImg : 入力画像
 * int height : 高さ
 * int width : 横幅
 * Mat outImg : 出力画像
 *
 * 機能 : Prewittフィルタを用いてエッジを抽出する
 *
 * return : void
 *************************************************/
void ImageProcessor::prewittFilter(Mat inImg, int32_t height, int32_t width, Mat outImg)
{
    int32_t filter1[3][3] = {
        {1,  1,  1 },
        {0,  0,  0 },
        {-1, -1, -1}
    };  // 横方向のPrewittフィルタ
    int32_t filter2[3][3] = {
        {1, 0, -1},
        {1, 0, -1},
        {1, 0, -1}
    };  // 縦方向のPrewittフィルタ
    int32_t rrx, ggx, bbx;
    int32_t rry, ggy, bby;

    for (int32_t y = 0; y < height; y++) {
        for (int32_t x = 0; x < width; x++) {
            rrx = ggx = bbx = 0;
            rry = ggy = bby = 0;

            // 3x3のフィルタ処理
            for (int32_t yy = 0; yy < FILTER_SIZE; yy++) {
                for (int32_t xx = 0; xx < FILTER_SIZE; xx++) {
                    // 画像の端の処理 (リピート)
                    int32_t replicateY = std::clamp(y + yy - 1, 0, height - 1);
                    int32_t replicateX = std::clamp(x + xx - 1, 0, width - 1);

                    // 画素取得
                    Vec3b &pix = inImg.at<Vec3b>(replicateY, replicateX);

                    // 画素値と横フィルタの積和
                    rrx += pix[RED] * filter1[yy][xx];
                    ggx += pix[GREEN] * filter1[yy][xx];
                    bbx += pix[BLUE] * filter1[yy][xx];

                    // 画素値と縦フィルタの積和
                    rry += pix[RED] * filter2[yy][xx];
                    ggy += pix[GREEN] * filter2[yy][xx];
                    bby += pix[BLUE] * filter2[yy][xx];
                }
            }

            // 画素値の平方根
            double red   = sqrt(static_cast<double>(rrx * rrx + rry * rry));
            double green = sqrt(static_cast<double>(ggx * ggx + ggy * ggy));
            double blue  = sqrt(static_cast<double>(bbx * bbx + bby * bby));

            // 画像の輝度値を0~255に収める
            uint8_t red8   = static_cast<uint8_t>(std::clamp(red, 0., 255.));
            uint8_t green8 = static_cast<uint8_t>(std::clamp(green, 0., 255.));
            uint8_t blue8  = static_cast<uint8_t>(std::clamp(blue, 0., 255.));

            // 画素の書き込み
            outImg.at<Vec3b>(y, x) = Vec3b(blue8, green8, red8);
        }
    }
}

/*************************************************
 * void robertsFilter(Mat inImg, int32_t height, int32_t width, Mat outImg)
 * Mat inImg : 入力画像
 * int height : 高さ
 * int width : 横幅
 * Mat outImg : 出力画像
 *
 * 機能 : Robertsフィルタを用いてエッジを抽出する
 *
 * return : void
 *************************************************/
void ImageProcessor::robertsFilter(Mat inImg, int32_t height, int32_t width, Mat outImg)
{
    int32_t filter1[3][3] = {
        {0, 0, 0 },
        {0, 1, 0 },
        {0, 0, -1}
    };  // 横方向のRobertsフィルタ
    int32_t filter2[3][3] = {
        {0, 0,  0},
        {0, 0,  1},
        {0, -1, 0}
    };  // 縦方向のRobertsフィルタ
    int32_t rrx, ggx, bbx;
    int32_t rry, ggy, bby;

    for (int32_t y = 0; y < height; y++) {
        for (int32_t x = 0; x < width; x++) {
            rrx = ggx = bbx = 0;
            rry = ggy = bby = 0;

            // 3x3のフィルタ処理
            for (int32_t yy = 0; yy < FILTER_SIZE; yy++) {
                for (int32_t xx = 0; xx < FILTER_SIZE; xx++) {
                    // 画像の端の処理 (リピート)
                    int32_t replicateY = std::clamp(y + yy - 1, 0, height - 1);
                    int32_t replicateX = std::clamp(x + xx - 1, 0, width - 1);

                    // 画素取得
                    Vec3b &pix = inImg.at<Vec3b>(replicateY, replicateX);

                    // 画素値と横フィルタの積和
                    rrx += pix[RED] * filter1[yy][xx];
                    ggx += pix[GREEN] * filter1[yy][xx];
                    bbx += pix[BLUE] * filter1[yy][xx];

                    // 画素値と縦フィルタの積和
                    rry += pix[RED] * filter2[yy][xx];
                    ggy += pix[GREEN] * filter2[yy][xx];
                    bby += pix[BLUE] * filter2[yy][xx];
                }
            }

            // 画素値の平方根
            double red   = sqrt(static_cast<double>(rrx * rrx + rry * rry));
            double green = sqrt(static_cast<double>(ggx * ggx + ggy * ggy));
            double blue  = sqrt(static_cast<double>(bbx * bbx + bby * bby));

            // 画像の輝度値を0~255に収める
            uint8_t red8   = static_cast<uint8_t>(std::clamp(red, 0., 255.));
            uint8_t green8 = static_cast<uint8_t>(std::clamp(green, 0., 255.));
            uint8_t blue8  = static_cast<uint8_t>(std::clamp(blue, 0., 255.));

            // 画素の書き込み
            outImg.at<Vec3b>(y, x) = Vec3b(blue8, green8, red8);
        }
    }
}

/*************************************************
 * void embossingFilter(Mat inImg, int32_t height, int32_t width, Mat outImg)
 * Mat inImg : 入力画像
 * int32_t height : 高さ
 * int32_t width : 横幅
 * Mat outImg : 出力画像
 *
 * 機能 : エンボスフィルタ処理
 *
 * return : void
 *************************************************/
void ImageProcessor::embossingFilter(Mat inImg, int32_t height, int32_t width, Mat outImg)
{
    int32_t filter[3][3] = {
        {0,  0, 0},
        {-3, 0, 3},
        {0,  0, 0}
    };  // エンボスフィルタ (数値を大きくするとエンボスの強さが増す)
    int32_t redSum, greenSum, blueSum;

    for (int32_t y = 0; y < height; y++) {
        for (int32_t x = 0; x < width; x++) {
            redSum = greenSum = blueSum = 0;

            // 3x3のフィルタ処理(-1から1までの範囲)
            for (int32_t yy = -1; yy <= 1; yy++) {
                for (int32_t xx = -1; xx <= 1; xx++) {
                    // 画像の端の処理 (リピート)
                    int32_t replicateY = std::clamp(y + yy, 0, height - 1);
                    int32_t replicateX = std::clamp(x + xx, 0, width - 1);

                    // 画素取得
                    Vec3b &pix = inImg.at<Vec3b>(replicateY, replicateX);

                    // 画素値とフィルタの積和
                    redSum += filter[yy + 1][xx + 1] * pix[RED];
                    greenSum += filter[yy + 1][xx + 1] * pix[GREEN];
                    blueSum += filter[yy + 1][xx + 1] * pix[BLUE];
                }
            }
            // 画像の輝度値を0~255に収める
            // 128は中間の明るさを示し、6はフィルタの係数の絶対値、それで割ることで画像のコントラストを調整
            uint8_t red   = static_cast<uint8_t>(std::clamp(redSum / 6 + 128, 0, 255));
            uint8_t green = static_cast<uint8_t>(std::clamp(greenSum / 6 + 128, 0, 255));
            uint8_t blue  = static_cast<uint8_t>(std::clamp(blueSum / 6 + 128, 0, 255));

            // 画素の書き込み
            outImg.at<Vec3b>(y, x) = Vec3b(blue, green, red);
        }
    }
}

/*************************************************
 * void medianFilter(Mat inImg, int32_t height, int32_t width, Mat outImg)
 * Mat inImg : 入力画像
 * int32_t height : 高さ
 * int32_t width : 横幅
 * Mat outImg : 出力画像
 *
 * 機能 : メディアンフィルタ処理
 *
 * return : void
 *************************************************/
void ImageProcessor::medianFilter(Mat inImg, int32_t height, int32_t width, Mat outImg)
{
    int32_t filterSize = 3;
    int32_t red[9], green[9], blue[9];

    for (int32_t y = 0; y < height; y++) {
        for (int32_t x = 0; x < width; x++) {
            // 3x3のフィルタ処理(-1から1までの範囲)
            int32_t idx = 0;
            for (int32_t yy = -1; yy <= 1; yy++) {
                for (int32_t xx = -1; xx <= 1; xx++) {
                    // 画像の端の処理 (リピート)
                    int32_t replicateY = std::clamp(y + yy, 0, height - 1);
                    int32_t replicateX = std::clamp(x + xx, 0, width - 1);

                    // 画素取得
                    Vec3b &pix = inImg.at<Vec3b>(replicateY, replicateX);

                    red[idx]   = pix[RED];
                    green[idx] = pix[GREEN];
                    blue[idx]  = pix[BLUE];
                    idx++;
                }
            }
            // 画素値のソート
            std::sort(red, red + filterSize * filterSize);
            std::sort(green, green + filterSize * filterSize);
            std::sort(blue, blue + filterSize * filterSize);

            // 画素の書き込み
            outImg.at<Vec3b>(y, x) = Vec3b(blue[4], green[4], red[4]);
        }
    }
}

}  // namespace filter
