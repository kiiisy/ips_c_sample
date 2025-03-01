#include "filter/filter.h"
#include "pixelwise/pixelwise.h"
#include <cstdint>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

using namespace cv;

void createHist(Mat img, Mat imgHist, const double fixedHistMax);

int32_t main()
{
    Mat img = imread("./data/Girl.bmp");

    int32_t height = img.rows;
    int32_t width = img.cols;
    Mat outImg = Mat{height, width, CV_8UC3, Scalar(0, 0, 0)};

    std::string outName = "./output/outimg_";
    std::string ipsName = "None";
    std::string extName = ".bmp";

    // 濃淡処理
    pixelwise::ImageProcessor ips;
    pixelwise::IpsType ipsType = pixelwise::IpsType::None;

    // フィルタ処理
    filter::ImageProcessor ips2;
    filter::IpsType ipsType2 = filter::IpsType::EmbossingFilter;

    double coeff, a, b, gammaVal, k, x0;
    int32_t filterCoeff;

    switch (ipsType)
    {
    case pixelwise::IpsType::ToneCurve:
        coeff = 2;
        ips.toneCurve(img, height, width, coeff, outImg);
        ipsName = "ToneCurve";
        break;
    case pixelwise::IpsType::Linear:
        a = 1.; // コントラストが変わる
        b = 50; // 明るさが変わる
        ips.effectLinear(img, height, width, a, b, outImg);
        ipsName = "Linear";
        break;
    case pixelwise::IpsType::Nega:
        ips.effectNega(img, height, width, outImg);
        ipsName = "Nega";
        break;
    case pixelwise::IpsType::Gamma:
        gammaVal = 0.7;
        ips.effectGamma(img, height, width, gammaVal, outImg);
        ipsName = "Gamma";
        break;
    case pixelwise::IpsType::Sigmoid:
        k = 1;
        x0 = 0.5;
        ips.effectSigmoid(img, height, width, k, x0, outImg);
        ipsName = "Sigmoid";
        break;
    case pixelwise::IpsType::HistEqualization:
        ips.histEqualization(img, height, width, outImg);
        ipsName = "HistEqualization";
        break;
    default:
        // 何もしない
        break;
    }

    switch (ipsType2)
    {
    case filter::IpsType::EqualizationFilter:
        filterCoeff = 2;
        ips2.equalizationFilter(img, height, width, filterCoeff, outImg);
        ipsName = "EqualizationFilter";
        break;
    case filter::IpsType::WeightedAverage:
        ips2.weightedAverageFilter(img, height, width, outImg);
        ipsName = "WeightedAverage";
        break;
    case filter::IpsType::SharpeningFilter:
        ips2.sharpeningFilter(img, height, width, outImg);
        ipsName = "SharpeningFilter";
        break;
    case filter::IpsType::EdgeDetectionFilter:
        ips2.edgeDetectionFilter(img, height, width, outImg);
        ipsName = "EdgeDetectionFilter";
        break;
    case filter::IpsType::SobelFilter:
        ips2.sobelFilter(img, height, width, outImg);
        ipsName = "SobelFilter";
        break;
    case filter::IpsType::PrewittFilter:
        ips2.prewittFilter(img, height, width, outImg);
        ipsName = "PrewittFilter";
        break;
    case filter::IpsType::RobertsFilter:
        ips2.robertsFilter(img, height, width, outImg);
        ipsName = "RobertsFilter";
        break;
    case filter::IpsType::EmbossingFilter:
        ips2.embossingFilter(img, height, width, outImg);
        ipsName = "EmbossingFilter";
        break;
    default:
        // 何もしない
        break;
    }

    // どちらも処理がない場合入力画像をそのまま出力
    if (ipsType == pixelwise::IpsType::None && ipsType2 == filter::IpsType::None)
    {
        outImg = img;
    }

    // ヒストグラム作成
    Mat imgHist = Mat{512, 1024, CV_8UC3, Scalar(0, 0, 0)};
    double fixedHistMax = 17000; // 20000
    createHist(outImg, imgHist, fixedHistMax);

    // 画像表示処理
    namedWindow("img", WINDOW_AUTOSIZE);
    imshow("out", outImg);
    imshow("histgram", imgHist);
    std::string outImgName = outName + ipsName + extName;
    imwrite(outImgName, outImg);

    waitKey(0);
    destroyWindow("img");
    return 0;
}

/*************************************************
 * void createHist(Mat img, Mat imgHist)
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

    const int32_t hdims[] = {256};
    const float hRanges[] = {0, 256};
    const float *ranges[] = {hRanges};

    // 度数分布を計算
    Mat hist;
    calcHist(&grayImg, 1, 0, Mat(), hist, 1, hdims, ranges);

    // 背景を白に設定
    imgHist.setTo(Scalar(255, 255, 255));

    // 軸とメモリのマージン
    int32_t margin = 50;

    // ヒストグラムの縦軸メモリを描画
    for (int32_t y = 0; y <= 5; y++)
    {
        int32_t posY = imgHist.rows - margin - y * (imgHist.rows - 2 * margin) / 5;
        int32_t value = static_cast<int32_t>(fixedHistMax * y / 5);                   // 固定された最大値を使用
        line(imgHist, Point(margin, posY), Point(margin - 5, posY), Scalar(0, 0, 0)); // 縦軸目盛り線
        putText(imgHist, std::to_string(value), Point(5, posY + 5), FONT_HERSHEY_SIMPLEX, 0.4,
                Scalar(0, 0, 0)); // 値を描画
    }

    // ヒストグラムの横軸メモリを描画
    int32_t xTickCount = 4; // 横軸の目盛り数（例: 0, 64, 128, 192, 256）
    for (int32_t x = 0; x <= xTickCount; x++)
    {
        int32_t posX = margin + x * (imgHist.cols - 2 * margin) / xTickCount;
        int32_t value = x * 256 / xTickCount; // 目盛りの値（0, 64, ... 256）
        line(imgHist, Point(posX, imgHist.rows - margin), Point(posX, imgHist.rows - margin + 5),
             Scalar(0, 0, 0)); // 横軸目盛り線
        putText(imgHist, std::to_string(value), Point(posX - 10, imgHist.rows - margin + 20), FONT_HERSHEY_SIMPLEX, 0.4,
                Scalar(0, 0, 0)); // 値を描画
    }

    //// ヒストグラムを描画
    for (int32_t i = 0; i < 256; i++)
    {
        int32_t v = saturate_cast<int32_t>(hist.at<float>(i));
        int32_t binHeight = (imgHist.rows - 2 * margin) * v / fixedHistMax; // 固定された最大値を使用
        binHeight = std::min(binHeight, imgHist.rows - 2 * margin);         // オーバーフロー防止
        line(imgHist, Point(margin + i * (imgHist.cols - 2 * margin) / 256, imgHist.rows - margin),
             Point(margin + i * (imgHist.cols - 2 * margin) / 256, imgHist.rows - margin - binHeight), Scalar(0, 0, 0));
    }

    // X軸とY軸を描画
    line(imgHist, Point(margin, margin), Point(margin, imgHist.rows - margin), Scalar(0, 0, 0)); // Y軸
    line(imgHist, Point(margin, imgHist.rows - margin), Point(imgHist.cols - margin, imgHist.rows - margin),
         Scalar(0, 0, 0)); // X軸
}
