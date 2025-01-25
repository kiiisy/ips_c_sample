#include <math.h>

#include <iostream>
#include <opencv2/opencv.hpp>

#define BLUE  0
#define GREEN 1
#define RED   2

using namespace std;
using namespace cv;

void createHist(Mat img, Mat imgHist, const double fixedHistMax);
void toneCurve(Mat inImg, int height, int width, double coeff, Mat outImg);
void effectLinear(Mat inImg, int height, int width, double a, double b, Mat outImg);
void effectNega(Mat inImg, int height, int width, Mat outImg);
void effectGamma(Mat inImg, int height, int width, double gammaVal, Mat outImg);
void effectSigmoid(Mat inImg, int height, int width, double k, double x0, Mat outImg);
void histEqualization(Mat inImg, int height, int width, Mat outImg);

int main()
{
    Mat img = imread("Girl.bmp");

    int height = img.rows;
    int width  = img.cols;
    Mat outImg = Mat{height, width, CV_8UC3, Scalar(0, 0, 0)};

    // 折れ線トーンカーブ
    double coeff = 2;
    //toneCurve(img, height, width, coeff, outImg);

    // 線形変換
    double a = 1.;  // コントラストが変わる
    double b = 50;  // 明るさが変わる
    effectLinear(img, height, width, a, b, outImg);

    // ガンマ補正
    double gammaVal = 0.7;
    //effectGamma(img, height, width, gammaVal, outImg);

    // シグモイド関数
    double k  = 1;
    double x0 = 0.5;
    //effectSigmoid(img, height, width, k, x0, outImg);

    // ヒストグラム均等化
    //histEqualization(img, height, width, outImg);

    // ネガポジ反転
    //effectNega(img, height, width, outImg);

    // ヒストグラム作成
    Mat    imgHist      = Mat{512, 1024, CV_8UC3, Scalar(0, 0, 0)};
    double fixedHistMax = 3000;
    createHist(outImg, imgHist, fixedHistMax);

    // 画像表示処理
    namedWindow("img", WINDOW_AUTOSIZE);
    imshow("out", outImg);
    imshow("histgram", imgHist);
    imwrite("./outimg.bmp", outImg);

    waitKey(0);
    destroyWindow("img");
    return 0;
}

/*************************************************
 * void createHist(Mat img, Mat imgHist))
 * Mat img : 入力画像
 * Mat imgHist : ヒストグラム画像
 * double fixedHistMax : ヒストグラムの最大値 (デフォルト: 512*512=262144)
 *                       画像によっては262144だと見づらいので適宜変更
 * 機能 : ヒストグラムを作成
 *
 * return : void
 *************************************************/
void createHist(Mat img, Mat imgHist, const double fixedHistMax = 262144)
{
    // グレースケールに変換
    Mat grayImg;
    cvtColor(img, grayImg, COLOR_BGR2GRAY);

    const int    hdims[]   = {256};
    const float  hRanges[] = {0, 256};
    const float *ranges[]  = {hRanges};

    // 度数分布を計算
    Mat hist;
    calcHist(&grayImg, 1, 0, Mat(), hist, 1, hdims, ranges);

    // 背景を白に設定
    imgHist.setTo(Scalar(255, 255, 255));

    // 軸とメモリのマージン
    int margin = 50;

    // ヒストグラムの縦軸メモリを描画
    for (int y = 0; y <= 5; y++) {
        int posY  = imgHist.rows - margin - y * (imgHist.rows - 2 * margin) / 5;
        int value = static_cast<int>(fixedHistMax * y / 5);  // 固定された最大値を使用
        line(imgHist, Point(margin, posY), Point(margin - 5, posY), Scalar(0, 0, 0));  // 縦軸目盛り線
        putText(imgHist, std::to_string(value), Point(5, posY + 5), FONT_HERSHEY_SIMPLEX, 0.4,
                Scalar(0, 0, 0));  // 値を描画
    }

    // ヒストグラムの横軸メモリを描画
    int xTickCount = 4;  // 横軸の目盛り数（例: 0, 64, 128, 192, 256）
    for (int x = 0; x <= xTickCount; x++) {
        int posX  = margin + x * (imgHist.cols - 2 * margin) / xTickCount;
        int value = x * 256 / xTickCount;  // 目盛りの値（0, 64, ... 256）
        line(imgHist, Point(posX, imgHist.rows - margin), Point(posX, imgHist.rows - margin + 5),
             Scalar(0, 0, 0));  // 横軸目盛り線
        putText(imgHist, std::to_string(value), Point(posX - 10, imgHist.rows - margin + 20), FONT_HERSHEY_SIMPLEX, 0.4,
                Scalar(0, 0, 0));  // 値を描画
    }

    //// ヒストグラムを描画
    for (int i = 0; i < 256; i++) {
        int v         = saturate_cast<int>(hist.at<float>(i));
        int binHeight = (imgHist.rows - 2 * margin) * v / fixedHistMax;  // 固定された最大値を使用
        binHeight     = std::min(binHeight, imgHist.rows - 2 * margin);  // オーバーフロー防止
        line(imgHist, Point(margin + i * (imgHist.cols - 2 * margin) / 256, imgHist.rows - margin),
             Point(margin + i * (imgHist.cols - 2 * margin) / 256, imgHist.rows - margin - binHeight), Scalar(0, 0, 0));
    }

    // X軸とY軸を描画
    line(imgHist, Point(margin, margin), Point(margin, imgHist.rows - margin), Scalar(0, 0, 0));  // Y軸
    line(imgHist, Point(margin, imgHist.rows - margin), Point(imgHist.cols - margin, imgHist.rows - margin),
         Scalar(0, 0, 0));  // X軸
}

/*************************************************
 * void toneCurve(Mat inImg, int height, int width, double coeff, Mat outImg)
 * Mat inImg : 入力画像
 * int height : 高さ
 * int width : 横幅
 * double coeff : 係数
 * Mat outImg : 出力画像
 *
 * 機能 : トーンカーブ処理
 *
 * return : void
 *************************************************/
void toneCurve(Mat inImg, int height, int width, double coeff, Mat outImg)
{
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // 画素取得
            Vec3b &pix = inImg.at<Vec3b>(y, x);

            // 画像処理
            unsigned char blue     = static_cast<unsigned char>(std::min(255.0, std::max(0.0, pix[BLUE] * coeff)));
            unsigned char green    = static_cast<unsigned char>(std::min(255.0, std::max(0.0, pix[GREEN] * coeff)));
            unsigned char red      = static_cast<unsigned char>(std::min(255.0, std::max(0.0, pix[RED] * coeff)));
            outImg.at<Vec3b>(y, x) = Vec3b(blue, green, red);
        }
    }
}

/*************************************************
 * void effectLinear(Mat inImg, int height, int width, double a, double b, Mat outImg)
 * Mat inImg : 入力画像
 * int height : 高さ
 * int width : 横幅
 * double a : 係数a(コントラスト)
 * double b : 係数b(明るさ)
 * Mat outImg : 出力画像
 *
 * 機能 : 線形変換
 *
 * return : void
 *************************************************/
void effectLinear(Mat inImg, int height, int width, double a, double b, Mat outImg)
{
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // 画素取得
            Vec3b &pix = inImg.at<Vec3b>(y, x);

            // 画像処理
            unsigned char blue  = static_cast<unsigned char>(std::min(255.0, std::max(0.0, a * pix[BLUE] + b)));
            unsigned char green = static_cast<unsigned char>(std::min(255.0, std::max(0.0, a * pix[GREEN] + b)));
            unsigned char red   = static_cast<unsigned char>(std::min(255.0, std::max(0.0, a * pix[RED] + b)));

            outImg.at<Vec3b>(y, x) = Vec3b(blue, green, red);
        }
    }
}

/*************************************************
 * void effectNega(Mat img, int height, int width, Mat outImg)
 * Mat inImg : 入力画像
 * int height : 高さ
 * int width : 横幅
 * Mat outImg : 出力画像
 *
 * 機能 : ネガ処理
 *
 * return : void
 *************************************************/
void effectNega(Mat inImg, int height, int width, Mat outImg)
{
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // 画素取得
            Vec3b &pix = inImg.at<Vec3b>(y, x);

            // 画像処理
            unsigned char blue  = 255 - pix[BLUE];
            unsigned char green = 255 - pix[GREEN];
            unsigned char red   = 255 - pix[RED];

            outImg.at<Vec3b>(y, x) = Vec3b(blue, green, red);
        }
    }
}

/*************************************************
 * void effectGamma(Mat inImg, int height, int width, double gammaVal, Mat outImg)
 * Mat inImg : 入力画像
 * int height : 高さ
 * int width : 横幅
 * double gammaVal : ガンマ値
 * Mat outImg : 出力画像
 *
 * 機能 : ガンマ変換
 *
 * return : void
 *************************************************/
void effectGamma(Mat inImg, int height, int width, double gammaVal, Mat outImg)
{
    unsigned char LUT[256];

    // LUTを作成
    for (int i = 0; i < 256; i++) {
        // 0~1に正規化
        double tmp = i / 255.0;
        LUT[i]     = static_cast<unsigned char>(std::pow(tmp, gammaVal) * 255.0);
    }

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // 画素取得
            Vec3b &pix = inImg.at<Vec3b>(y, x);

            // 画像処理
            unsigned char blue  = LUT[pix[BLUE]];
            unsigned char green = LUT[pix[GREEN]];
            unsigned char red   = LUT[pix[RED]];

            outImg.at<Vec3b>(y, x) = Vec3b(blue, green, red);
        }
    }
}

/*************************************************
 * void effectSigmoid(Mat inImg, int height, int width, double k, double x0, Mat outImg)
 * Mat inImg : 入力画像
 * int height : 高さ
 * int width : 横幅
 * double k : 傾き
 * double x0 : 変化の中心点
 * Mat outImg : 出力画像
 *
 * 機能 : シグモイド関数
 *
 * return : void
 *************************************************/
void effectSigmoid(Mat inImg, int height, int width, double k, double x0, Mat outImg)
{
    unsigned char LUT[256];

    // LUTを作成
    for (int i = 0; i < 256; i++) {
        // 0~1に正規化
        double norm = i / 255.0;
        LUT[i]      = static_cast<unsigned char>((1.0 / (1.0 + std::exp(-k * (norm - x0)))) * 255.0);
    }

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // 画素取得
            Vec3b &pix = inImg.at<Vec3b>(y, x);

            // 画像処理
            unsigned char blue  = LUT[pix[BLUE]];
            unsigned char green = LUT[pix[GREEN]];
            unsigned char red   = LUT[pix[RED]];

            outImg.at<Vec3b>(y, x) = Vec3b(blue, green, red);
        }
    }
}

/*************************************************
 * void calcNormHist(Mat inImg, int height, int width, float *hist)
 * Mat inImg : 入力画像
 * int height : 高さ
 * int width : 横幅
 * flaot hist : ヒストグラム
 *
 * 機能 : 正規化ヒストグラムを計算
 *
 * return : void
 *************************************************/
void calcNormHist(Mat inImg, int height, int width, float *hist)
{
    long histTmp[256] = {0};
    long sum          = 0;

    // ヒストグラム作成
    // 画像の全画素を走査
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            Vec3b &pix = inImg.at<Vec3b>(y, x);
            // グレースケールなためどれか一つの値を取得
            int pixVal      = pix[BLUE];
            histTmp[pixVal] = histTmp[pixVal] + 1;
            sum += 1;
        }
    }

    // ヒストグラムの正規化
    for (int i = 0; i < 256; i++) {
        hist[i] = static_cast<float>(histTmp[i]) / sum;
    }
}

/*************************************************
 * void histEqualization(Mat inImg, int height, int width, Mat outImg)
 * Mat inImg : 入力画像
 * int height : 高さ
 * int width : 横幅
 * Mat outImg : 出力画像
 *
 * 機能 : ヒストグラム均等化
 *
 * return : void
 *************************************************/
void histEqualization(Mat inImg, int height, int width, Mat outImg)
{
    float         hist[256];
    unsigned char histEq[256];
    float         sum;

    // ヒストグラムの正規化
    calcNormHist(inImg, height, width, hist);

    // iの画素値までの累積分布関数を計算
    for (int i = 0; i < 256; i++) {
        sum = 0.0;
        // 0~iまでのヒストグラムの和を計算
        for (int j = 0; j <= i; j++) {
            sum = sum + hist[j];
        }

        // ヒストグラムの累積分布関数を計算
        histEq[i] = static_cast<unsigned char>(255 * sum + 0.5);
    }

    // ヒストグラム均等化
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // 画素取得
            Vec3b &pix = inImg.at<Vec3b>(y, x);

            // 画像処理
            unsigned char blue  = histEq[pix[BLUE]];
            unsigned char green = histEq[pix[GREEN]];
            unsigned char red   = histEq[pix[RED]];

            outImg.at<Vec3b>(y, x) = Vec3b(blue, green, red);
        }
    }
}