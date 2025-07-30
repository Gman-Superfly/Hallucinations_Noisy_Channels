import numpy as np
from scipy import signal

# Parameters
f_max = 5  # Max frequency in Hz
fs_nyquist = 2 * f_max  # Nyquist rate
T_nyquist = 1 / fs_nyquist

fs_under = 1.5 * f_max  # Undersampling
T_under = 1 / fs_under

t = np.linspace(0, 1, 1000)  # Continuous time

# Bandlimited signal
freq1 = 1  # Below f_max
freq2 = 3  # Below f_max
original_signal = np.sin(2 * np.pi * freq1 * t) + 0.5 * np.sin(2 * np.pi * freq2 * t)

# Function for sinc reconstruction
def reconstruct_signal(t, t_samples, samples, T):
    recon = np.zeros_like(t)
    for i, ti in enumerate(t):
        x = (ti - t_samples) / T
        recon[i] = np.sum(samples * np.sinc(x))  # np.sinc is sin(pi x)/(pi x)
    return recon

# Nyquist sampling and reconstruction
t_samples_ny = np.arange(0, 1 + T_nyquist, T_nyquist)[:len(np.arange(0, 1, T_nyquist)) + 1]  # Adjust to cover [0,1]
samples_ny = np.sin(2 * np.pi * freq1 * t_samples_ny) + 0.5 * np.sin(2 * np.pi * freq2 * t_samples_ny)
recon_ny = reconstruct_signal(t, t_samples_ny, samples_ny, T_nyquist)

# Undersampling and reconstruction
t_samples_under = np.arange(0, 1 + T_under, T_under)[:len(np.arange(0, 1, T_under)) + 1]
samples_under = np.sin(2 * np.pi * freq1 * t_samples_under) + 0.5 * np.sin(2 * np.pi * freq2 * t_samples_under)
recon_under = reconstruct_signal(t, t_samples_under, samples_under, T_under)

# Compute MSE to test reconstruction accuracy
mse_ny = np.mean((original_signal - recon_ny)**2)
mse_under = np.mean((original_signal - recon_under)**2)

print("MSE at Nyquist rate (should be low, near zero):", mse_ny)
print("MSE at undersampling rate (should be higher, showing distortion):", mse_under)

# To 'test phase alignment', check reconstruction at a non-sample point
test_point = 0.1  # Arbitrary point not on sample grid
idx = np.argmin(np.abs(t - test_point))
true_val = original_signal[idx]

# Sum of sinc contributions at test_point for Nyquist
x_ny = (test_point - t_samples_ny) / T_nyquist
contrib_ny = samples_ny * np.sinc(x_ny)
recon_val_ny = np.sum(contrib_ny)
print("\nAt t=0.1 (Nyquist):")
print("True value:", true_val)
print("Reconstructed value:", recon_val_ny)
print("Individual sinc contributions (first 5):", contrib_ny[:5])

# For undersampling
x_under = (test_point - t_samples_under) / T_under
contrib_under = samples_under * np.sinc(x_under)
recon_val_under = np.sum(contrib_under)
print("\nAt t=0.1 (Undersampling):")
print("True value:", true_val)
print("Reconstructed value:", recon_val_under)
print("Individual sinc contributions (first 5):", contrib_under[:5])