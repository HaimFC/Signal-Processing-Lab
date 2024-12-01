
# Lab Assignment - Signal Processing

### General Information
- **Course**: Introduction to Signal Processing
---

## Task Overview
This lab assignment focuses on signal sampling, reconstruction, and frequency-domain analysis.

---

## Task Details

### Question 1: Signal Sampling and Reconstruction

#### Problem Description
1. **Continuous Signal**: Analyze the signal \( x(t) = \frac{4}{\omega_m \pi t^2} \sin^2\left(\frac{\omega_m t}{2}\right) \cos\left(\frac{\omega_m t}{2}\right) \sin(\omega_m t) \).
2. **Frequency Analysis**: Compute and plot the Fourier Transform \( |X(\omega)| \).
3. **Sampling**: Sample the signal and create a Zero Order Hold (ZOH) reconstructed version.
4. **Frequency Domain ZOH**: Plot \( |X_{ZOH}(\omega)| \) for ZOH reconstruction.
5. **Reconstruction**: Use ideal reconstruction and plot the restored signal \( x_{rec}(t) \).
6. **Under-sampling**: Analyze reconstruction when sampling frequency is insufficient.

#### MATLAB Implementation
The script:
1. **Plots** the absolute value of \( x(t) \) and \( |X(\omega)| \).
2. **Samples** the signal and performs ZOH reconstruction.
3. **Reconstructs** the signal using ideal reconstruction with sufficient and insufficient sampling rates.

### Question 2: Non-Uniform Sampling

#### Problem Description
1. **Signal**: \( x(t) = 5\cos(\omega_A t) - 3\sin(\omega_B t) \), where \( \omega_A = 5\pi, \omega_B = 2\pi \).
2. **Uniform Sampling**: Sample the signal uniformly over one period and restore it using Fourier coefficients.
3. **Non-Uniform Sampling**: Sample the signal randomly over one period and restore it.
4. **Condition Number Analysis**: Compute and compare the condition number of the Fourier matrix for uniform and non-uniform sampling.

#### MATLAB Implementation
The script:
1. **Samples** the signal uniformly and non-uniformly.
2. **Restores** the signal using Fourier coefficients.
3. **Plots** the original and restored signals.

---

### Question 3: Functional Basis Analysis

#### Problem Description
1. Analyze periodic signals \( f(t) = 4\cos\left(\frac{4\pi}{T}t\right) + \sin\left(\frac{10\pi}{T}t\right) \) and \( g(t) = 2	ext{sign}\left(\sin\left(\frac{6\pi}{T}t\right)\right) - 4	ext{sign}\left(\sin\left(\frac{4\pi}{T}t\right)\right) \).
2. Use Fourier basis and rectangular pulse basis for reconstruction.
3. Compute Fourier coefficients and reconstruct the signals.

#### MATLAB Implementation
The script:
1. **Computes** Fourier coefficients using the trapezoidal method.
2. **Reconstructs** signals \( f(t) \) and \( g(t) \) using both bases.
