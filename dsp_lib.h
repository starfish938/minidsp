// dsp_lib.h
#ifndef DSP_LIB_H
#define DSP_LIB_H

#include <vector>
#include <complex>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <numbers>

namespace dsp {
    
using Complex = std::complex<double>;  //类型别名

template<typename T>
class DSPLib {
static_assert(std::is_floating_point_v<T>, "T must be floating point type");  //断言，T必须是一个浮点型

public:
    // ==================== FFT ====================
    static std::vector<Complex> fft(const std::vector<Complex>& input) {
        size_t n = input.size();
        if (n == 1) return input;  //长度为1时返回自身
        
        // 检查是否为2的幂
        if ((n & (n - 1)) != 0) {  //位与
            throw std::invalid_argument("Input size must be power of 2");
        }
        
        // Cooley-Tukey FFT
        if (n % 2 != 0) return input;  //保证2的倍数
        
        std::vector<Complex> even(n/2);  
        std::vector<Complex> odd(n/2);
        
        for (size_t i = 0; i < n/2; i++) {
            even[i] = input[2 * i];
            odd[i] = input[2 * i + 1];
        }
        
        even = fft(even);
        odd = fft(odd);
        
        std::vector<Complex> result(n);
        T pi = 3.14159265358979323846;
        
        for (size_t k = 0; k < n/2; k++) {
            Complex t = std::polar(1.0, -2.0 * pi * k / n) * odd[k];  //旋转因子
            result[k] = even[k] + t;
            result[k + n/2] = even[k] - t;
        }
        
        return result;



    }
    
    // 逆FFT
    static std::vector<Complex> ifft(const std::vector<Complex>& input) {
        size_t n = input.size();
        std::vector<Complex> conj_input(n);
        
        for (size_t i = 0; i < n; i++) {
            conj_input[i] = std::conj(input[i]);
        }
        
        auto result = fft(conj_input);
        
        for (size_t i = 0; i < n; i++) {
            result[i] = std::conj(result[i]) /  static_cast<double>(n);
        }
        //IFFT（X) = 1/N FFT（X的共轭）
        return result;
    }
    
    // 实数输入的FFT转化复数
    static std::vector<Complex> rfft(const std::vector<T>& input) {
        std::vector<Complex> complex_input(input.size());
        for (size_t i = 0; i < input.size(); i++) {
            complex_input[i] = Complex(input[i], 0);
        }
        return fft(complex_input);
    }
    
    // ==================== 卷积 ====================
    static std::vector<T> convolve(const std::vector<T>& a, const std::vector<T>& b) {
        size_t n = a.size();
        size_t m = b.size();
        size_t result_size = n + m - 1;
        
        std::vector<T> result(result_size, 0);
        
        // 直接卷积 O(n*m)
        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < m; j++) {
                result[i + j] += a[i] * b[j];
            }
        }
        
        return result;
    }
    
    // 基于FFT的快速卷积 O(n log n)
    static std::vector<T> fft_convolve(const std::vector<T>& a, const std::vector<T>& b) {
        size_t n = a.size();
        size_t m = b.size();
        size_t result_size = n + m - 1;
        
        // 找到下一个2的幂
        size_t fft_size = 1;
        while (fft_size < result_size) fft_size <<= 1;
        
        // 填充到FFT大小
        std::vector<Complex> fa(fft_size), fb(fft_size);
        for (size_t i = 0; i < n; i++) fa[i] = Complex(a[i], 0);
        for (size_t i = 0; i < m; i++) fb[i] = Complex(b[i], 0);
        
        // FFT
        fa = fft(fa);
        fb = fft(fb);
        
        // 频域相乘
        for (size_t i = 0; i < fft_size; i++) {
            fa[i] *= fb[i];
        }
        
        // 逆FFT
        auto ifft_result = ifft(fa);
        
        // 取实部
        std::vector<T> result(result_size);
        for (size_t i = 0; i < result_size; i++) {
            result[i] = ifft_result[i].real();
        }
        
        return result;
    }
    
    // ==================== 窗函数 ====================
    static std::vector<T> window_hamming(size_t n) {
        std::vector<T> window(n);
        T pi = 3.14159265358979323846;
        
        for (size_t i = 0; i < n; i++) {
            window[i] = 0.54 - 0.46 * std::cos(2.0 * pi * i / (n - 1));
        }
        
        return window;
    }
    
    static std::vector<T> window_hann(size_t n) {
        std::vector<T> window(n);
        T pi = 3.14159265358979323846;
        
        for (size_t i = 0; i < n; i++) {
            window[i] = 0.5 * (1.0 - std::cos(2.0 * pi * i / (n - 1)));
        }
        
        return window;
    }
    
    static std::vector<T> window_blackman(size_t n) {
        std::vector<T> window(n);
        T pi = 3.14159265358979323846;
        T a0 = 0.42, a1 = 0.5, a2 = 0.08;
        
        for (size_t i = 0; i < n; i++) {
            window[i] = a0 - a1 * std::cos(2.0 * pi * i / (n - 1)) 
                            + a2 * std::cos(4.0 * pi * i / (n - 1));
        }
        
        return window;
    }
    
    // ==================== FIR滤波器 ====================
    class FIRFilter {
    private:
        std::vector<T> coefficients;
        std::vector<T> buffer;
        size_t index;
        
    public:
        FIRFilter(const std::vector<T>& coeff) 
            : coefficients(coeff), buffer(coeff.size(), 0), index(0) {}
        
        T process(T input) {
            buffer[index] = input;
            
            T output = 0;
            size_t tap = index;
            
            for (size_t i = 0; i < coefficients.size(); i++) {
                output += coefficients[i] * buffer[tap];
                tap = (tap == 0) ? coefficients.size() - 1 : tap - 1;
            }
            
            index = (index + 1) % coefficients.size();
            return output;
        }
        
        std::vector<T> process(const std::vector<T>& input) {
            std::vector<T> output(input.size());
            for (size_t i = 0; i < input.size(); i++) {
                output[i] = process(input[i]);
            }
            return output;
        }
        
        void reset() {
            std::fill(buffer.begin(), buffer.end(), 0);
            index = 0;
        }
    };
    
    // 设计低通FIR滤波器（窗口法）
    static std::vector<T> design_lowpass_fir(T cutoff, size_t num_taps) {
        cutoff *= 0.5;   // ★ 修正：将 0~1 (Nyquist) 转为 0~0.5 (fs)
        if (num_taps % 2 == 0) num_taps++; // 确保奇数
        
        std::vector<T> coeff(num_taps);
        T pi = 3.14159265358979323846;
        int mid = num_taps / 2;
        
        for (int i = 0; i < static_cast<int>(num_taps); i++) {
            int n = i - mid;
            if (n == 0) {
                coeff[i] = 2.0 * cutoff;
            } else {
                coeff[i] = std::sin(2.0 * pi * cutoff * n) / (pi * n);
            }
            
            // 应用汉明窗
            coeff[i] *= (0.54 - 0.46 * std::cos(2.0 * pi * i / (num_taps - 1)));
        }
        
        return coeff;
    }
    
    // ==================== IIR滤波器 ====================
    //直接I型
    class IIRFilter {
    private:
        std::vector<T> b; // 前向系数
        std::vector<T> a; // 反馈系数
        std::vector<T> x_buffer;
        std::vector<T> y_buffer;
        
    public:
        IIRFilter(const std::vector<T>& numerator, const std::vector<T>& denominator) 
            : b(numerator), a(denominator) {
            if (a.empty() || a[0] == 0) {
                throw std::invalid_argument("Invalid denominator coefficients");
            }
            
            // 归一化
            T a0 = a[0];
            for (auto& coeff : b) coeff /= a0;
            for (auto& coeff : a) coeff /= a0;
            
            x_buffer.resize(b.size(), 0);
            y_buffer.resize(a.size() - 1, 0);
        }
        
        T process(T input) {
            // 移动输入缓冲区
            for (size_t i = x_buffer.size() - 1; i > 0; i--) {
                x_buffer[i] = x_buffer[i - 1];
            }
            x_buffer[0] = input;
            
            // 计算输出
            T output = 0;
            
            // 前向部分
            for (size_t i = 0; i < b.size(); i++) {
                output += b[i] * x_buffer[i];
            }
            
            // 反馈部分
            for (size_t i = 1; i < a.size(); i++) {
                if (i - 1 < y_buffer.size()) {
                    output -= a[i] * y_buffer[i - 1];
                }
            }
            
            // 移动输出缓冲区
            for (size_t i = y_buffer.size() - 1; i > 0; i--) {
                y_buffer[i] = y_buffer[i - 1];
            }
            y_buffer[0] = output;
            
            return output;
        }
        
        std::vector<T> process(const std::vector<T>& input) {
            std::vector<T> output(input.size());
            for (size_t i = 0; i < input.size(); i++) {
                output[i] = process(input[i]);
            }
            return output;
        }
        
        void reset() {
            std::fill(x_buffer.begin(), x_buffer.end(), 0);
            std::fill(y_buffer.begin(), y_buffer.end(), 0);
        }
    };
    //直接II型
    class IIRFilterDirectII {
    private:
        std::vector<T> b;           // 前向系数 [b0, b1, b2, ...]
        std::vector<T> a;           // 反馈系数 [a0, a1, a2, ...] (a0=1)
        std::vector<T> w;           // 状态变量（延迟线）
    
    public:
        IIRFilterDirectII(const std::vector<T>& numerator, 
                        const std::vector<T>& denominator) 
            : b(numerator), a(denominator) {
            
            // 归一化使 a0 = 1
            T a0 = a[0];
            for (auto& coeff : b) coeff /= a0;
            for (auto& coeff : a) coeff /= a0;
            
            // 状态变量数量 = max(b.size(), a.size()) - 1
            size_t order = std::max(b.size(), a.size()) - 1;
            w.resize(order, 0);
        }
    
        T process(T input) {
            // 计算新的状态变量 w[0]
            T w0 = input;
            for (size_t i = 1; i < a.size(); i++) {
                w0 -= a[i] * w[i - 1];  // w0 = x[n] - a1·w1 - a2·w2 - ...
            }
            
            // 计算输出 y[n]
            T output = b[0] * w0;
            for (size_t i = 1; i < b.size(); i++) {
                output += b[i] * w[i - 1];  // + b1·w1 + b2·w2 + ...
            }
            
            // 更新状态变量（移位）
            for (size_t i = w.size() - 1; i > 0; i--) {
                w[i] = w[i - 1];
            }
            w[0] = w0;
            
            return output;
        }
        
        std::vector<T> process(const std::vector<T>& input) {
            std::vector<T> output(input.size());
            for (size_t i = 0; i < input.size(); i++) {
                output[i] = process(input[i]);
            }
            return output;
        }
        
        void reset() {
            std::fill(w.begin(), w.end(), 0);
        }
    };
    
    // 低通IIR滤波器（一阶双线性）
    static std::pair<std::vector<T>, std::vector<T>> design_lowpass_iir(T cutoff) {
        T pi = 3.14159265358979323846;
        T w0 = 2 * pi * cutoff;
        T alpha = 2 * std::tan(w0 / 2);

        std::vector<T> b(2);
        std::vector<T> a(2);

        T norm = alpha + 2;

        b[0] = alpha / norm ;
        b[1] = alpha / norm ;
        

        a[0] = 1.0;
        a[1] = (alpha - 2) / norm;
        

        return {b, a};
    }
    
    //高通IIR滤波器（一阶双线性）
    static std::pair<std::vector<T>, std::vector<T>> design_highpass_iir(T cutoff) {
        T pi = 3.14159265358979323846;
        T w0 = 2 * pi * cutoff;
        T alpha = 2 * std::tan(w0 / 2);

        std::vector<T> b(2);
        std::vector<T> a(2);

        T norm = alpha + 2;

        b[0] = 2 / norm ;
        b[1] = -2 / norm ;
        

        a[0] = 1.0;
        a[1] = (alpha - 2) / norm;

        return {b, a};
    }
    
    
    // ==================== 短时傅里叶变换 ====================
    struct STFTResult {
        std::vector<std::vector<Complex>> spectrogram;
        std::vector<T> frequencies;
        std::vector<T> times;
    };
    
    static STFTResult stft(const std::vector<T>& signal, 
                           size_t window_size, 
                           size_t hop_size,
                           const std::vector<T>& window) {
        if (window.size() != window_size) {
            throw std::invalid_argument("Window size mismatch");
        }
        
        size_t n_frames = 1 + (signal.size() - window_size) / hop_size;
        
        STFTResult result;
        result.times.resize(n_frames);
        result.spectrogram.resize(n_frames);
        
        // 准备FFT输入
        for (size_t frame = 0; frame < n_frames; frame++) {
            size_t start = frame * hop_size;
            
            // 加窗
            std::vector<Complex> frame_data(window_size);
            for (size_t i = 0; i < window_size; i++) {
                frame_data[i] = Complex(signal[start + i] * window[i], 0);
            }
            
            // FFT
            auto spectrum = fft(frame_data);
            
            // 只保留正频率部分（对于实数输入）
            size_t n_bins = window_size / 2 + 1;
            result.spectrogram[frame].resize(n_bins);
            for (size_t i = 0; i < n_bins; i++) {
                result.spectrogram[frame][i] = spectrum[i];
            }
            
            result.times[frame] = static_cast<T>(start) / static_cast<T>(hop_size);
        }
        
        // 计算频率轴
        result.frequencies.resize(window_size / 2 + 1);
        for (size_t i = 0; i < result.frequencies.size(); i++) {
            result.frequencies[i] = static_cast<T>(i) / window_size;
        }
        
        return result;
    }
    
    // 计算幅度谱（dB）
    static std::vector<std::vector<T>> magnitude_spectrogram(const STFTResult& stft_result) {
        std::vector<std::vector<T>> mag_spec(stft_result.spectrogram.size());
        
        for (size_t i = 0; i < stft_result.spectrogram.size(); i++) {
            mag_spec[i].resize(stft_result.spectrogram[i].size());
            for (size_t j = 0; j < stft_result.spectrogram[i].size(); j++) {
                // 转换为dB：20 * log10(|x| + epsilon)
                T mag = std::abs(stft_result.spectrogram[i][j]);
                mag_spec[i][j] = 20.0 * std::log10(mag + 1e-10);
            }
        }
        
        return mag_spec;
    }
};

} // namespace dsp

#endif 