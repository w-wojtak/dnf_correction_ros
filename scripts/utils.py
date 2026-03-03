import numpy as np


def kernel_gauss(x, a, b, c):
    """Gaussian kernel: a * exp(-b * x^2) - c"""
    return a * np.exp(-b * x ** 2) - c


def kernel_osc(x, a, b, alpha):
    """Oscillatory kernel function."""
    return a * (np.exp(-b * np.abs(x)) *
                (b * np.sin(np.abs(alpha * x)) + np.cos(alpha * x)))


def compute_convolution(u, theta, w_hat, dx):
    """Compute FFT-based convolution for field dynamics."""
    f     = np.heaviside(u - theta, 1)
    f_hat = np.fft.fft(f)
    return dx * np.fft.ifftshift(np.real(np.fft.ifft(f_hat * w_hat)))


def compute_action_bounds(x, positions):
    """Compute spatial index ranges for each action bucket."""
    positions  = np.array(positions)
    midpoints  = (positions[:-1] + positions[1:]) / 2
    boundaries = np.concatenate(([x[0]], midpoints, [x[-1]]))
    return [np.where((x >= boundaries[i]) & (x < boundaries[i + 1]))[0]
            for i in range(len(positions))]


def get_inputs(x, t, dt, input_pars, input_flag):
    if not input_flag:
        return np.zeros((len(t), len(x)))

    [input_shape, input_positions, input_onsets, input_durations] = input_pars
    amplitude, sigma = input_shape
    inputs = np.zeros((len(t), len(x)))

    for pos, onset, dur in zip(input_positions, input_onsets, input_durations):
        time_on = int(onset / dt)
        time_off = int((onset + dur) / dt)
        gaussian = amplitude * np.exp(-((x - pos) ** 2) / (2 * sigma ** 2))
        inputs[time_on:time_off, :] += gaussian

    return inputs
