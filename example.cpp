#include "dsp_lib.h"
#include <iostream>
#include <fstream>
#include <cmath>

int main() {
    using namespace dsp;
    
    // 生成测试信号
    std::vector<double> signal(1024);
    double pi = std::acos(-1.0);
    double fs = 1000; // 采样率 1kHz
    
    for (int i = 0; i < 1024; i++) {
        double t = i / fs;
        // 50Hz + 120Hz + 白噪声
        signal[i] = std::sin(2 * pi * 50 * t) + 
                    0.5 * std::sin(2 * pi * 120 * t) +
                    0.1 * (rand() / (double)RAND_MAX - 0.5);
    }
    
    // 1. 计算FFT并显示频谱
    auto spectrum = DSPLib<double>::rfft(signal);
    std::cout << "Spectrum magnitude at key frequencies:\n";
    for (int i = 0; i < 150; i++) {
        double freq = i * fs / signal.size();
        double mag = std::abs(spectrum[i]);
        std::cout << "  " << freq << " Hz: " << mag << "\n";
    }
    
    // 2. 设计并应用滤波器
    auto fir_coeff = DSPLib<double>::design_lowpass_fir(0.1, 51); // 100Hz低通
    DSPLib<double>::FIRFilter lowpass(fir_coeff);
    
    auto filtered = lowpass.process(signal);
    
    // 3. STFT分析
    auto window = DSPLib<double>::window_hann(256);
    auto stft_result = DSPLib<double>::stft(signal, 256, 64, window);
    auto mag_spec = DSPLib<double>::magnitude_spectrogram(stft_result);
    
    std::cout << "\nSTFT result dimensions: "
              << mag_spec.size() << " x " 
              << mag_spec[0].size() << std::endl;
    
    // 保存结果到文件（用于绘图）
    std::ofstream out("signal_data.txt");
    out<<"time(s)\torigin signal\tfiltered signal\n";
    for (size_t i = 0; i < signal.size(); i++) {
        out << i/fs << "\t" << signal[i] << "\t\t" << filtered[i] << "\n";
    }
    out.close();
    
    return 0;
}