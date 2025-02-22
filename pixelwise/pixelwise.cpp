#include "pixelwise.h"
#include "../param.h"

namespace pixelwise {
/*************************************************
 * void toneCurve(Mat inImg, int32_t height, int32_t width, double coeff, Mat outImg)
 * Mat inImg : 入力画像
 * int32_t height : 高さ
 * int32_t width : 横幅
 * double coeff : 係数
 * Mat outImg : 出力画像
 *
 * 機能 : トーンカーブ処理
 *
 * return : void
 *************************************************/
void ImageProcessor::toneCurve(Mat inImg, int32_t height, int32_t width, double coeff, Mat outImg)
{
    for (int32_t y = 0; y < height; y++) {
        for (int32_t x = 0; x < width; x++) {
            // 画素値を取得
            Vec3b &pix = inImg.at<Vec3b>(y, x);

            // 画素値を係数倍して範囲を0-255に収める
            uint8_t blue  = static_cast<uint8_t>(std::min(255.0, std::max(0.0, pix[BLUE] * coeff)));
            uint8_t green = static_cast<uint8_t>(std::min(255.0, std::max(0.0, pix[GREEN] * coeff)));
            uint8_t red   = static_cast<uint8_t>(std::min(255.0, std::max(0.0, pix[RED] * coeff)));

            // 画素値を設定
            outImg.at<Vec3b>(y, x) = Vec3b(blue, green, red);
        }
    }
}

/*************************************************
 * void effectLinear(Mat inImg, int32_t height, int32_t width, double a, double b, Mat outImg)
 * Mat inImg : 入力画像
 * int32_t height : 高さ
 * int32_t width : 横幅
 * double a : 係数a(コントラスト)
 * double b : 係数b(明るさ)
 * Mat outImg : 出力画像
 *
 * 機能 : 線形変換
 *
 * return : void
 *************************************************/
void ImageProcessor::effectLinear(Mat inImg, int32_t height, int32_t width, double a, double b, Mat outImg)
{
    for (int32_t y = 0; y < height; y++) {
        for (int32_t x = 0; x < width; x++) {
            // 画素値を取得
            Vec3b &pix = inImg.at<Vec3b>(y, x);

            // 画素値を線形変換して範囲を0-255に収める
            uint8_t blue  = static_cast<uint8_t>(std::min(255.0, std::max(0.0, a * pix[BLUE] + b)));
            uint8_t green = static_cast<uint8_t>(std::min(255.0, std::max(0.0, a * pix[GREEN] + b)));
            uint8_t red   = static_cast<uint8_t>(std::min(255.0, std::max(0.0, a * pix[RED] + b)));

            // 画素値を設定
            outImg.at<Vec3b>(y, x) = Vec3b(blue, green, red);
        }
    }
}

/*************************************************
 * void effectNega(Mat img, int32_t height, int32_t width, Mat outImg)
 * Mat inImg : 入力画像
 * int32_t height : 高さ
 * int32_t width : 横幅
 * Mat outImg : 出力画像
 *
 * 機能 : ネガ処理
 *
 * return : void
 *************************************************/
void ImageProcessor::effectNega(Mat inImg, int32_t height, int32_t width, Mat outImg)
{
    for (int32_t y = 0; y < height; y++) {
        for (int32_t x = 0; x < width; x++) {
            // 画素値を取得
            Vec3b &pix = inImg.at<Vec3b>(y, x);

            // 画素値を反転して範囲を0-255に収める
            uint8_t blue  = 255 - pix[BLUE];
            uint8_t green = 255 - pix[GREEN];
            uint8_t red   = 255 - pix[RED];

            // 画素値を設定
            outImg.at<Vec3b>(y, x) = Vec3b(blue, green, red);
        }
    }
}

/*************************************************
 * void effectGamma(Mat inImg, int32_t height, int32_t width, double gammaVal, Mat outImg)
 * Mat inImg : 入力画像
 * int32_t height : 高さ
 * int32_t width : 横幅
 * double gammaVal : ガンマ値
 * Mat outImg : 出力画像
 *
 * 機能 : ガンマ変換
 *
 * return : void
 *************************************************/
void ImageProcessor::effectGamma(Mat inImg, int32_t height, int32_t width, double gammaVal, Mat outImg)
{
    uint8_t LUT[256];  // Look Up Table

    // LUTを作成
    for (int32_t i = 0; i < 256; i++) {
        // 0~1に正規化
        double tmp = i / 255.0;
        // ガンマ変換
        LUT[i] = static_cast<uint8_t>(std::pow(tmp, gammaVal) * 255.0);
    }

    for (int32_t y = 0; y < height; y++) {
        for (int32_t x = 0; x < width; x++) {
            // 画素値を取得
            Vec3b &pix = inImg.at<Vec3b>(y, x);

            // 画素値をガンマ変換して範囲を0-255に収める
            uint8_t blue  = LUT[pix[BLUE]];
            uint8_t green = LUT[pix[GREEN]];
            uint8_t red   = LUT[pix[RED]];

            // 画素値を設定
            outImg.at<Vec3b>(y, x) = Vec3b(blue, green, red);
        }
    }
}

/*************************************************
 * void effectSigmoid(Mat inImg, int32_t height, int32_t width, double k, double x0, Mat outImg)
 * Mat inImg : 入力画像
 * int32_t height : 高さ
 * int32_t width : 横幅
 * double k : 傾き
 * double x0 : 変化の中心点
 * Mat outImg : 出力画像
 *
 * 機能 : シグモイド関数
 *
 * return : void
 *************************************************/
void ImageProcessor::effectSigmoid(Mat inImg, int32_t height, int32_t width, double k, double x0, Mat outImg)
{
    uint8_t LUT[256];  // Look Up Table

    // LUTを作成
    for (int32_t i = 0; i < 256; i++) {
        // 0~1に正規化
        double norm = i / 255.0;
        // シグモイド関数を適用
        LUT[i] = static_cast<uint8_t>((1.0 / (1.0 + std::exp(-k * (norm - x0)))) * 255.0);
    }

    for (int32_t y = 0; y < height; y++) {
        for (int32_t x = 0; x < width; x++) {
            // 画素値を取得
            Vec3b &pix = inImg.at<Vec3b>(y, x);

            // 画素値をシグモイド関数で変換して範囲を0-255に収める
            uint8_t blue  = LUT[pix[BLUE]];
            uint8_t green = LUT[pix[GREEN]];
            uint8_t red   = LUT[pix[RED]];

            // 画素値を設定
            outImg.at<Vec3b>(y, x) = Vec3b(blue, green, red);
        }
    }
}

/*************************************************
 * void calcNormHist(Mat inImg, int32_t height, int32_t width, float *hist)
 * Mat inImg : 入力画像
 * int32_t height : 高さ
 * int32_t width : 横幅
 * flaot hist : ヒストグラム
 *
 * 機能 : 正規化ヒストグラムを計算
 *
 * return : void
 *************************************************/
void ImageProcessor::calcNormHist(Mat inImg, int32_t height, int32_t width, float *hist)
{
    int64_t histTmp[256] = {0};
    int64_t sum          = 0;

    // ヒストグラムの作成(画像の全画素を走査)
    for (int32_t y = 0; y < height; y++) {
        for (int32_t x = 0; x < width; x++) {
            // 画素値を取得
            Vec3b &pix = inImg.at<Vec3b>(y, x);

            // グレースケールなためどれか一つの値を取得
            int32_t pixVal  = pix[BLUE];
            histTmp[pixVal] = histTmp[pixVal] + 1;
            sum += 1;
        }
    }
    // ヒストグラムの正規化
    for (int32_t i = 0; i < 256; i++) {
        hist[i] = static_cast<float>(histTmp[i]) / sum;
    }
}

/*************************************************
 * void histEqualization(Mat inImg, int32_t height, int32_t width, Mat outImg)
 * Mat inImg : 入力画像
 * int32_t height : 高さ
 * int32_t width : 横幅
 * Mat outImg : 出力画像
 *
 * 機能 : ヒストグラム均等化
 *
 * return : void
 *************************************************/
void ImageProcessor::histEqualization(Mat inImg, int32_t height, int32_t width, Mat outImg)
{
    float   hist[256];
    uint8_t histEq[256];
    float   sum;

    // ヒストグラムの正規化
    calcNormHist(inImg, height, width, hist);

    // iの画素値までの累積分布関数を計算
    for (int32_t i = 0; i < 256; i++) {
        sum = 0.0;

        // 0~iまでのヒストグラムの和を計算
        for (int32_t j = 0; j <= i; j++) {
            sum = sum + hist[j];
        }

        // ヒストグラムの累積分布関数を計算(0~255の範囲に正規化)
        histEq[i] = static_cast<uint8_t>(255 * sum + 0.5);
    }

    // ヒストグラム均等化
    for (int32_t y = 0; y < height; y++) {
        for (int32_t x = 0; x < width; x++) {
            // 画素値を取得
            Vec3b &pix = inImg.at<Vec3b>(y, x);

            // 画素値をヒストグラム均等化して範囲を0-255に収める
            uint8_t blue  = histEq[pix[BLUE]];
            uint8_t green = histEq[pix[GREEN]];
            uint8_t red   = histEq[pix[RED]];

            // 画素値を設定
            outImg.at<Vec3b>(y, x) = Vec3b(blue, green, red);
        }
    }
}
}  // namespace pixelwise
