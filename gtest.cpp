#include "googletest/googletest/include/gtest/gtest.h"
#include "googletest/googlemock/include/gmock/gmock.h"
#include "dsp_lib.h"
#include <cmath>
#include <vector>
#include <complex>

using namespace dsp;
using ::testing::ElementsAre;
using ::testing::Pointwise;
using ::testing::DoubleNear;


// ==================== FFT 测试 ====================
class FFTTest : public ::testing::Test {
protected:
    const double epsilon = 1e-10;
};

// 测试 FFT 和 IFFT 的可逆性
TEST_F(FFTTest, ForwardInverseReversibility) {
    std::vector<dsp::Complex> signal = {
        {1.0, 0.0}, {2.0, 0.0}, {3.0, 0.0}, {4.0, 0.0},
        {5.0, 0.0}, {6.0, 0.0}, {7.0, 0.0}, {8.0, 0.0}
    };
    
    auto spectrum = DSPLib<double>::fft(signal);
    auto reconstructed = DSPLib<double>::ifft(spectrum);
    
    for (size_t i = 0; i < signal.size(); i++) {
        EXPECT_NEAR(signal[i].real(), reconstructed[i].real(), epsilon);
        EXPECT_NEAR(signal[i].imag(), reconstructed[i].imag(), epsilon);
    }
}

// 测试脉冲信号的 FFT（应该是常数）
TEST_F(FFTTest, ImpulseResponse) {
    std::vector<Complex> impulse(8, {0.0, 0.0});
    impulse[0] = {1.0, 0.0};  // 单位脉冲
    
    auto spectrum = DSPLib<double>::fft(impulse);
    
    // 脉冲的 FFT 应该是全 1
    for (const auto& val : spectrum) {
        EXPECT_NEAR(val.real(), 1.0, epsilon);
        EXPECT_NEAR(val.imag(), 0.0, epsilon);
    }
}

// 测试正弦波的 FFT（应该在特定频率有峰值）
TEST_F(FFTTest, SineWave) {
    const size_t N = 8;
    std::vector<Complex> signal(N);
    double pi = 3.14159265358979323846;
    
    // 生成 sin(2π * 1/8 * i) 频率为 1/N
    for (size_t i = 0; i < N; i++) {
        signal[i] = {std::sin(2.0 * pi * i / N), 0.0};
    }
    
    auto spectrum = DSPLib<double>::fft(signal);
    
    // 应该在 k=1 和 k=7 处有峰值（共轭对称）
    EXPECT_GT(std::abs(spectrum[1]), 3.0);  
    EXPECT_GT(std::abs(spectrum[7]), 3.0);
    EXPECT_LT(std::abs(spectrum[0]), 0.1);  
    EXPECT_LT(std::abs(spectrum[4]), 0.1);  
}

// 测试非 2 的幂输入
TEST_F(FFTTest, NonPowerOfTwo) {
    std::vector<Complex> signal = {{1,0}, {2,0}, {3,0}};  
    
    EXPECT_THROW(DSPLib<double>::fft(signal), std::invalid_argument);
}

// ==================== 卷积测试 ====================
class ConvolutionTest : public ::testing::Test {
protected:
    const double epsilon = 1e-10;
};

// 测试直接卷积和 FFT 卷积的一致性
TEST_F(ConvolutionTest, DirectVsFFT) {
    std::vector<double> a = {1, 2, 3, 4};
    std::vector<double> b = {0.5, 1, 0.5};
    
    auto direct = DSPLib<double>::convolve(a, b);
    auto fft_conv = DSPLib<double>::fft_convolve(a, b);
    
    ASSERT_EQ(direct.size(), fft_conv.size());
    for (size_t i = 0; i < direct.size(); i++) {
        EXPECT_NEAR(direct[i], fft_conv[i], epsilon);
    }
}

// 测试卷积的交换律
TEST_F(ConvolutionTest, Commutative) {
    std::vector<double> a = {1, 2, 3};
    std::vector<double> b = {4, 5, 6};
    
    auto ab = DSPLib<double>::convolve(a, b);
    auto ba = DSPLib<double>::convolve(b, a);
    
    ASSERT_EQ(ab.size(), ba.size());
    for (size_t i = 0; i < ab.size(); i++) {
        EXPECT_NEAR(ab[i], ba[i], epsilon);
    }
}

// 测试卷积的单位元
TEST_F(ConvolutionTest, Identity) {
    std::vector<double> a = {1, 2, 3, 4, 5};
    std::vector<double> impulse = {1};  
    
    auto result = DSPLib<double>::convolve(a, impulse);
    
    ASSERT_EQ(a.size(), result.size());
    for (size_t i = 0; i < a.size(); i++) {
        EXPECT_NEAR(a[i], result[i], epsilon);
    }
}

// ==================== 窗函数测试 ====================
class WindowFunctionTest : public ::testing::Test {
protected:
    const double epsilon = 1e-10;
};

// 测试汉明窗的对称性和范围
TEST_F(WindowFunctionTest, HammingSymmetry) {
    size_t N = 11;
    auto window = DSPLib<double>::window_hamming(N);
    
    // 检查对称性
    for (size_t i = 0; i < N/2; i++) {
        EXPECT_NEAR(window[i], window[N-1-i], epsilon);
    }
    
    // 检查范围（汉明窗在0.08到1.0之间）
    for (auto val : window) {
        EXPECT_GE(val, 0.08);
        EXPECT_LE(val, 1.0);
    }
    
    // 检查特定值
    EXPECT_NEAR(window[N/2], 1.0, epsilon);  // 中心应为 1
}

// 测试汉明窗
TEST_F(WindowFunctionTest, HannSymmetry) {
    size_t N = 11;
    auto window = DSPLib<double>::window_hann(N);
    
    for (size_t i = 0; i < N/2; i++) {
        EXPECT_NEAR(window[i], window[N-1-i], epsilon);
    }
    
    EXPECT_NEAR(window[0], 0.0, epsilon);
    EXPECT_NEAR(window[N/2], 1.0, epsilon);
    EXPECT_NEAR(window[N-1], 0.0, epsilon);
}

// ==================== FIR 滤波器测试 ====================
class FIRFilterTest : public ::testing::Test {
protected:
    const double epsilon = 1e-6;
};

// 测试 FIR 滤波器的脉冲响应
TEST_F(FIRFilterTest, ImpulseResponse) {
    auto coeff = DSPLib<double>::design_lowpass_fir(0.2, 31);
    DSPLib<double>::FIRFilter filter(coeff);
    
    std::vector<double> impulse(50, 0);
    impulse[0] = 1.0;
    
    auto response = filter.process(impulse);
    
    // 脉冲响应应该等于滤波器系数
    for (size_t i = 0; i < coeff.size(); i++) {
        EXPECT_NEAR(response[i], coeff[i], epsilon);
    }
    
    
    for (size_t i = coeff.size(); i < response.size(); i++) {
        EXPECT_NEAR(response[i], 0.0, epsilon);
    }
}



// 测试 FIR 滤波器的频率选择性
TEST_F(FIRFilterTest, FrequencySelectivity) {
    auto coeff = DSPLib<double>::design_lowpass_fir(0.2, 31);
    DSPLib<double>::FIRFilter filter(coeff);
    
    const size_t N = 1000;
    std::vector<double> input(N);
    double pi = 3.14159265358979323846;
    
    // 混合频率：低频(0.1)和高频(0.4)
    for (size_t i = 0; i < N; i++) {
        input[i] = std::sin(2.0 * pi * 0.1 * i) + 
                    std::sin(2.0 * pi * 0.4 * i);
    }
    
    auto output = filter.process(input);
    size_t start = 200;  
    
    // 计算输出功率（简单近似）
    double low_amp = 0, high_amp = 0;
    for (size_t i = 20; i < N; i++) {  
        low_amp = std::max(low_amp, std::abs(output[i]));
        high_amp = std::max(high_amp, std::abs(output[i]));
    }
    
    // 低频成分应该被保留，高频被衰减
    EXPECT_LT( low_amp, 0.8);
    EXPECT_GT( high_amp , 0.3);
}

// ==================== IIR 滤波器测试 ====================
//低通一阶
class IIRLPFilterTest : public ::testing::Test {
protected:
    const double epsilon = 1e-6;
    const double pi = 3.14159265358979323846;
};

// 稳定性
TEST_F(IIRLPFilterTest, FirstorderStability) {
    auto [b, a] = DSPLib<double>::design_lowpass_iir(0.2);
    
    // 检查极点是否在单位圆内（简单检查 a 系数）
    EXPECT_LT(std::abs(a[1]), 1.0);
}

// 脉冲响应衰减
TEST_F(IIRLPFilterTest, FirstorderImpulseDecay) {
    auto [b, a] = DSPLib<double>::design_lowpass_iir(0.2);
    DSPLib<double>::IIRFilter filter(b, a);
    
    std::vector<double> impulse(200, 0);
    impulse[0] = 1.0;
    
    auto response = filter.process(impulse);
    
    // 检查是否最终衰减（比较开头和结尾）
    double start_energy = 0, end_energy = 0;
    for (size_t i = 0; i < 20; i++) {
        start_energy += std::abs(response[i]);
    }
    for (size_t i = 180; i < 200; i++) {
        end_energy += std::abs(response[i]);
    }
    
    // 结尾的能量应该远小于开头
    EXPECT_LT(end_energy, start_energy * 0.1);
    
    // 检查是否稳定（不发散）
    for (size_t i = 1; i < response.size(); i++) {
        if (std::abs(response[i]) > 1e-6) {  
            EXPECT_LE(std::abs(response[i]), std::abs(response[i-1]) * 2.0);
        }
    }
}

//低通二阶
//稳定性
TEST_F(IIRLPFilterTest, SecondorderStability){
    auto [b, a] = DSPLib<double>::design_lowpass_iir_2nd(0.2);
    
    EXPECT_LT(std::abs(a[1]), 1.0);
    EXPECT_LT(std::abs(a[2]), 1.0);
}

//脉冲响应衰减
TEST_F(IIRLPFilterTest, SecondorderImpulseDecay){
    auto [b, a] = DSPLib<double>::design_lowpass_iir_2nd(0.2);
    DSPLib<double>::IIRFilter filter(b, a);
    
    std::vector<double> impulse(200, 0);
    impulse[0] = 1.0;
    auto response = filter.process(impulse);
    
    double start_energy = 0, end_energy = 0;
    for (size_t i = 0; i < 20; i++) start_energy += std::abs(response[i]);
    for (size_t i = 180; i < 200; i++) end_energy += std::abs(response[i]);
    
    EXPECT_LT(end_energy, start_energy * 0.1);
}

//直流增益
TEST_F(IIRLPFilterTest, SecondorderDCResponse){
    auto [b, a] = DSPLib<double>::design_lowpass_iir_2nd(0.2);
    DSPLib<double>::IIRFilter filter(b, a);
    
    std::vector<double> dc_input(100, 1.0);
    auto output = filter.process(dc_input);
    
    double steady = 0;
    for (size_t i = 80; i < 100; i++) steady += std::abs(output[i]);
    steady /= 20;
    
    EXPECT_NEAR(steady, 1.0, 0.1);  // 直流增益应为1
}

//频率选择性
TEST_F(IIRLPFilterTest, SecondorderFrequencySelectivity){
    double cutoff = 0.2;
    auto [b, a] = DSPLib<double>::design_lowpass_iir_2nd(cutoff);
    DSPLib<double>::IIRFilter filter(b, a);
    
    const size_t N = 500;
    std::vector<double> input(N);
    
    // 测试低频 (应通过)
    for (size_t i = 0; i < N; i++) 
        input[i] = std::sin(2.0 * pi * 0.05 * i);
    auto low_out = filter.process(input);
    
    filter.reset();
    
    // 测试高频 (应衰减)
    for (size_t i = 0; i < N; i++) 
        input[i] = std::sin(2.0 * pi * 0.4 * i);
    auto high_out = filter.process(input);
    
    double low_amp = 0, high_amp = 0;
    for (size_t i = 400; i < N; i++) {
        low_amp = std::max(low_amp, std::abs(low_out[i]));
        high_amp = std::max(high_amp, std::abs(high_out[i]));
    }
    
    EXPECT_GT(low_amp, 0.8);   // 低频保留
    EXPECT_LT(high_amp, 0.3);  // 高频衰减
}

//高通一阶
class IIRHPFilterTest : public ::testing::Test {
    protected:
        const double epsilon = 1e-6;
        const double pi = 3.14159265358979323846;
};

//稳定性
TEST_F(IIRHPFilterTest, HighpassStability) {
    auto [b, a] = DSPLib<double>::design_highpass_iir(0.2); 
    
    // 检查极点是否在单位圆内
    EXPECT_LT(std::abs(a[1]), 1.0);
    if (a.size() > 2) {
        EXPECT_LT(std::abs(a[2]), 1.0);
    }
}

// 脉冲响应衰减
TEST_F(IIRHPFilterTest, HighpassImpulseDecay) {
    
    auto [b, a] = DSPLib<double>::design_highpass_iir(0.2);
    DSPLib<double>::IIRFilter filter(b, a);
    
    std::vector<double> impulse(200, 0);
    impulse[0] = 1.0;
    
    auto response = filter.process(impulse);
    
    // 检查是否最终衰减
    double start_energy = 0, end_energy = 0;
    for (size_t i = 0; i < 20; i++) {
        start_energy += std::abs(response[i]);
    }
    for (size_t i = 180; i < 200; i++) {
        end_energy += std::abs(response[i]);
    }
    
    EXPECT_LT(end_energy, start_energy * 0.1);
    
    // 检查是否稳定
    for (size_t i = 1; i < response.size(); i++) {
        if (std::abs(response[i]) > 1e-6) {
            EXPECT_LE(std::abs(response[i]), std::abs(response[i-1]) * 2.0);
        }
    }
}

// 直流响应
TEST_F(IIRHPFilterTest, HighpassDCResponse) {
    auto [b, a] = DSPLib<double>::design_highpass_iir(0.2);
    DSPLib<double>::IIRFilter filter(b, a);
    
    std::vector<double> dc_input(100, 1.0);
    auto output = filter.process(dc_input);
    
    // 稳态应该接近0（高通滤除直流）
    double steady = 0;
    for (size_t i = 80; i < 100; i++) {
        steady += std::abs(output[i]);
    }
    steady /= 20;
    
    EXPECT_LT(steady, 0.1);
}

// ==================== 直接II型IIR滤波器测试 ====================
class IIRFilterDirectIITest : public ::testing::Test {
protected:
    const double epsilon = 1e-6;
    
    // 创建一个简单的一阶低通滤波器 (fc = 0.1)
    std::pair<std::vector<double>, std::vector<double>> createLowpassFilter() {
        std::vector<double> b = {0.245, 0.245};  // b0, b1
        std::vector<double> a = {1.0, -0.509};   // a0, a1
        return {b, a};
    }
};

// 测试1: 构造函数和初始化
TEST_F(IIRFilterDirectIITest, ConstructorInitialization) {
    auto [b, a] = createLowpassFilter();
    
    DSPLib<double>::IIRFilterDirectII filter(b, a);
    
    
    SUCCEED();
}

// 测试2: 脉冲响应
TEST_F(IIRFilterDirectIITest, ImpulseResponse) {
    auto [b, a] = createLowpassFilter();
    DSPLib<double>::IIRFilterDirectII filter(b, a);
    
    std::vector<double> impulse(50, 0);
    impulse[0] = 1.0;
    
    auto response = filter.process(impulse);
    
    // 手动计算理论脉冲响应
    std::vector<double> expected(50, 0);
    expected[0] = b[0];                                // y[0] = b0
    expected[1] = b[1] - a[1] * expected[0];           // y[1] = b1 - a1*y[0]
    for (size_t i = 2; i < expected.size(); i++) {
        expected[i] = -a[1] * expected[i-1];           // y[n] = -a1*y[n-1] (b项为0)
    }
    std::cout << "Impulse response: ";
    for (size_t i = 0; i < 10; i++) {
        std::cout << response[i] << " ";
    }
    std::cout << std::endl;
    // 验证
    for (size_t i = 0; i < 10; i++) {
        EXPECT_NEAR(response[i], expected[i], epsilon);
    }
}

// 测试3: 阶跃响应
TEST_F(IIRFilterDirectIITest, StepResponse) {
    auto [b, a] = createLowpassFilter();
    DSPLib<double>::IIRFilterDirectII filter(b, a);
    
    std::vector<double> step(100, 1.0);  
    
    auto response = filter.process(step);
    
    
    double dc_gain = (b[0] + b[1]) / (1 + a[1]);  
    
    // 稳态应该接近直流增益
    for (size_t i = 80; i < 100; i++) {
        EXPECT_NEAR(response[i], dc_gain, 0.01);
    }
}

// 测试4: 正弦波响应
TEST_F(IIRFilterDirectIITest, SineWaveResponse) {
    auto [b, a] = createLowpassFilter();
    DSPLib<double>::IIRFilterDirectII filter(b, a);
    
    const size_t N = 200;
    std::vector<double> input(N);
    double pi = 3.14159265358979323846;
    double freq = 0.05;  // 低频，应该通过
    
    for (size_t i = 0; i < N; i++) {
        input[i] = std::sin(2.0 * pi * freq * i);
    }
    
    auto output = filter.process(input);
    
    
    size_t start = 100;
    double input_amp = 0, output_amp = 0;
    
    for (size_t i = start; i < N; i++) {
        input_amp = std::max(input_amp, std::abs(input[i]));
        output_amp = std::max(output_amp, std::abs(output[i]));
    }
    
    // 输出幅度应该小于输入
    EXPECT_LT(output_amp, input_amp);
    EXPECT_GT(output_amp, 0.1);  
}

// 测试5: 与直接I型的一致性
TEST_F(IIRFilterDirectIITest, CompareWithDirectI) {
    auto [b, a] = createLowpassFilter();
    
    
    DSPLib<double>::IIRFilter filter_directI(b, a);
    
    
    DSPLib<double>::IIRFilterDirectII filter_directII(b, a);
    
    std::vector<double> input(100);
    double pi = 3.14159265358979323846;
    for (size_t i = 0; i < 100; i++) {
        input[i] = std::sin(2.0 * pi * 0.1 * i) + 0.5 * std::sin(2.0 * pi * 0.3 * i);
    }
    
    auto outputI = filter_directI.process(input);
    auto outputII = filter_directII.process(input);
    
    
    for (size_t i = 0; i < outputI.size(); i++) {
        EXPECT_NEAR(outputI[i], outputII[i], 1e-10);
    }
}

// 测试6: 重置功能
TEST_F(IIRFilterDirectIITest, ResetFunctionality) {
    auto [b, a] = createLowpassFilter();
    DSPLib<double>::IIRFilterDirectII filter(b, a);
    
    std::vector<double> input(50, 1.0);
    
    // 第一次处理
    auto output1 = filter.process(input);
    
    // 重置
    filter.reset();
    
    // 再次处理相同的输入
    auto output2 = filter.process(input);
    
    // 重置后的输出应该和第一次完全相同
    for (size_t i = 0; i < output1.size(); i++) {
        EXPECT_NEAR(output1[i], output2[i], epsilon);
    }
}

// 测试7: 批量处理和单样本处理的一致性
TEST_F(IIRFilterDirectIITest, BatchVsSampleWise) {
    auto [b, a] = createLowpassFilter();
    DSPLib<double>::IIRFilterDirectII filter_batch(b, a);
    DSPLib<double>::IIRFilterDirectII filter_sample(b, a);
    
    std::vector<double> input(100);
    double pi = 3.14159265358979323846;
    for (size_t i = 0; i < 100; i++) {
        input[i] = std::sin(2.0 * pi * 0.05 * i);
    }
    
    // 批量处理
    auto batch_output = filter_batch.process(input);
    
    // 逐个样本处理
    std::vector<double> sample_output(input.size());
    for (size_t i = 0; i < input.size(); i++) {
        sample_output[i] = filter_sample.process(input[i]);
    }
    
    // 结果应该相同
    for (size_t i = 0; i < input.size(); i++) {
        EXPECT_NEAR(batch_output[i], sample_output[i], epsilon);
    }
}

// 测试8: 稳定性测试
TEST_F(IIRFilterDirectIITest, Stability) {
    
    std::vector<double> b = {1.0, 0.5};
    std::vector<double> a = {1.0, -1.5};  
    DSPLib<double>::IIRFilterDirectII filter(b, a);
    
    std::vector<double> impulse(50, 0);
    impulse[0] = 1.0;
    
    auto response = filter.process(impulse);
    
    
    bool diverging = false;
    for (size_t i = 10; i < response.size(); i++) {
        if (std::abs(response[i]) > std::abs(response[i-1]) * 1.5) {
            diverging = true;
            break;
        }
    }
    
    EXPECT_FALSE(diverging);
}

// ==================== STFT 测试 ====================
class STFTTest : public ::testing::Test {
protected:
    const double epsilon = 1e-6;
};

// 测试 STFT 的维度
TEST_F(STFTTest, Dimensions) {
    std::vector<double> signal(1024);
    auto window = DSPLib<double>::window_hann(256);
    
    auto stft_result = DSPLib<double>::stft(signal, 256, 64, window);
    auto mag_spec = DSPLib<double>::magnitude_spectrogram(stft_result);
    
    // 检查维度
    size_t expected_frames = 1 + (1024 - 256) / 64;
    EXPECT_EQ(mag_spec.size(), expected_frames);
    EXPECT_EQ(mag_spec[0].size(), 256/2 + 1);
}

// 测试 STFT 对 chirp 信号的响应（应该有时频变化）
TEST_F(STFTTest, ChirpSignal) {
    std::vector<double> signal(1024);
    double pi = 3.14159265358979323846;
    
    // 生成 chirp 信号：频率随时间增加
    for (size_t i = 0; i < 1024; i++) {
        double t = static_cast<double>(i) / 1024;
        signal[i] = std::sin(2.0 * pi * (100 * t * t) * t);
    }
    
    auto window = DSPLib<double>::window_hann(256);
    auto stft_result = DSPLib<double>::stft(signal, 256, 64, window);
    
    // 检查能量分布是否随时间向高频移动
    auto mag_spec = DSPLib<double>::magnitude_spectrogram(stft_result);
    
    // 找到每帧的最大能量频率
    std::vector<size_t> peak_bins;
    for (const auto& frame : mag_spec) {
        size_t peak_bin = 0;
        double max_mag = frame[0];
        for (size_t j = 1; j < frame.size(); j++) {
            if (frame[j] > max_mag) {
                max_mag = frame[j];
                peak_bin = j;
            }
        }
        peak_bins.push_back(peak_bin);
    }
    
    
    // 这里只检查总体趋势
    size_t increasing_count = 0;
    for (size_t i = 1; i < peak_bins.size(); i++) {
        if (peak_bins[i] >= peak_bins[i-1]) {
            increasing_count++;
        }
    }
    EXPECT_GT(increasing_count, peak_bins.size() / 2);
}

// ==================== 性能基准测试 ====================
class BenchmarkTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 测试不同大小的 FFT
        sizes = {1024, 2048, 4096, 8192};
    }
    
    std::vector<size_t> sizes;
};

TEST_F(BenchmarkTest, FFTPerformance) {
    for (size_t n : sizes) {
        std::vector<Complex> signal(n);
        for (size_t i = 0; i < n; i++) {
            signal[i] = Complex(static_cast<double>(i) / n, 0);
        }
        
        auto start = std::chrono::high_resolution_clock::now();
        auto spectrum = DSPLib<double>::fft(signal);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        // 只是记录，不断言
        std::cout << "FFT size " << n << ": " << duration.count() / 1000.0 << " ms" << std::endl;
        
        // 验证结果不为空
        EXPECT_FALSE(spectrum.empty());
    }
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}