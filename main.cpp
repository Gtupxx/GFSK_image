#include <iostream>
#include <vector>
#include <cmath>

#include "matplotlibcpp.h"

namespace plt = matplotlibcpp; // 命名空间简化

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// FIR低通滤波器系数（示例值，可以根据实际需求调整）
const std::vector<double> firCoefficients = {
    -0.001, -0.002, 0.0, 0.006, 0.015, 0.026, 0.037, 0.045, 0.046, 0.037,
    0.015, -0.016, -0.054, -0.092, -0.119, -0.124, -0.097, -0.033, 0.058, 0.166,
    0.273, 0.360, 0.407, 0.407, 0.360, 0.273, 0.166, 0.058, -0.033, -0.097,
    -0.124, -0.119, -0.092, -0.054, -0.016, 0.015, 0.037, 0.046, 0.045, 0.037,
    0.026, 0.015, 0.006, 0.0, -0.002, -0.001};

// 生成示例信号
std::vector<double> generateSignal(int length, double lowFreq, double highFreq, double sampleRate)
{
    std::vector<double> signal(length);
    for (int i = 0; i < length; ++i)
    {
        double t = i / sampleRate;
        signal[i] = sin(2 * M_PI * lowFreq * t) + 0.5 * sin(2 * M_PI * highFreq * t);
    }
    return signal;
}

// 应用FIR滤波器
std::vector<double> applyFIRFilter(const std::vector<double> &signal, const std::vector<double> &coefficients)
{
    int signalLength = signal.size();
    int filterLength = coefficients.size();
    std::vector<double> filteredSignal(signalLength, 0.0);

    for (int i = 0; i < signalLength; ++i)
    {
        for (int j = 0; j < filterLength; ++j)
        {
            if (i - j >= 0)
            {
                filteredSignal[i] += coefficients[j] * signal[i - j];
            }
        }
    }
    return filteredSignal;
}

// 主函数
int main()
{
    // 示例参数
    int length = 1000;
    double lowFreq = 10.0;      // 低频成分
    double highFreq = 50.0;     // 高频成分
    double sampleRate = 1000.0; // 采样率

    // 生成示例信号
    std::vector<double> signal = generateSignal(length, lowFreq, highFreq, sampleRate);

    // 应用FIR滤波器
    std::vector<double> filteredSignal = applyFIRFilter(signal, firCoefficients);

    // 准备绘图数据
    std::vector<double> time(length);
    for (int i = 0; i < length; ++i)
    {
        time[i] = i / sampleRate;
    }

    // 绘制原始信号和滤波后信号
    try
    {
        plt::figure_size(1200, 600);
        plt::subplot(2, 1, 1);
        plt::plot(time, signal);
        plt::title("Original Signal");
        plt::subplot(2, 1, 2);
        plt::plot(time, filteredSignal);
        plt::title("Filtered Signal");
        plt::show();
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
