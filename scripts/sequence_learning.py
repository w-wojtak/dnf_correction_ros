# pylint: disable=C0200
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from utils import *
from pathlib import Path

# ====================================
# -------- Project Setup -------------
# ====================================

PROJECT_ROOT = Path.cwd()
results_dir = PROJECT_ROOT / "data_learning"
results_dir.mkdir(exist_ok=True)

# ====================================
# -------- Parameters ----------------
# ====================================

# Kernel and spatial/temporal parameters
kernel_pars = [1, 0.7, 0.9]
x_lim, t_lim = 80, 60
dx, dt = 0.2, 0.1
theta = 1

# Field parameters
tau_h = 20
h_0 = 0
h_0_d = 0
tau_h_d = 20

# Input configuration
input_flag = True
input_shape = [3, 1.5]
input_duration = [1, 1, 1, 1]
input_position_1 = [-60, -20, 20, 40]
input_onset_time_1 = [9, 20, 28, 35]

input_pars_1 = [input_shape, input_position_1, input_onset_time_1, input_duration]

# Plotting parameters
plot_fields = True
plot_every = 5
plot_delay = 0.05

# ====================================
# -------- Initialization ------------
# ====================================

x = np.arange(-x_lim, x_lim + dx, dx)
t = np.arange(0, t_lim + dt, dt)

input_indices_1 = [np.argmin(np.abs(x - pos)) for pos in input_position_1]
inputs_1 = get_inputs(x, t, dt, input_pars_1, input_flag)

# Initialize fields
u_field_1 = h_0 * np.ones_like(x)
h_u_1 = h_0 * np.ones_like(x)
u_d = h_0_d * np.ones_like(x)
h_u_d = h_0_d * np.ones_like(x)

# Precompute kernel FFT
w_hat = np.fft.fft(kernel_osc(x, *kernel_pars))

# History storage
u_1_tc = []
u_d_tc = []

# ====================================
# -------- Plot Setup ----------------
# ====================================

fig = axs = line1_field = line1_input = line2_field = None

if plot_fields:
    plt.ion()
    fig, axs = plt.subplots(2, 1, figsize=(6, 6), sharex=True)

    # Sequence memory field
    line1_field, = axs[0].plot(x, u_field_1, label='Field activity u_sm(x)')
    line1_input, = axs[0].plot(x, inputs_1[0, :], label='Input 1')
    axs[0].set_ylim(-2, 10)
    axs[0].set_ylabel("Activity")
    axs[0].legend()
    axs[0].set_title("Sequence Memory Field - Time = 0")

    # Task duration field
    line2_field, = axs[1].plot(x, u_d, label='Field activity u_d(x)')
    axs[1].set_ylim(-2, 10)
    axs[1].set_xlabel("x")
    axs[1].set_ylabel("Activity")
    axs[1].legend()
    axs[1].set_title("Task Duration Field - Time = 0")

# ====================================
# -------- Helper Function -----------
# ====================================

def compute_field_update(u, theta, w_hat, h_u, tau_h, input_signal):
    """Compute field dynamics update."""
    f = np.heaviside(u - theta, 1)
    f_hat = np.fft.fft(f)
    conv = dx * np.fft.ifftshift(np.real(np.fft.ifft(f_hat * w_hat)))
    
    h_u_new = h_u + dt / tau_h * f
    u_new = u + dt * (-u + conv + input_signal + h_u_new)
    
    return u_new, h_u_new

# ====================================
# -------- Simulation Loop -----------
# ====================================

for i in range(len(t)):
    # Task duration field input (only at t=0)
    input_d = 3.0 * np.exp(-((x - 0) ** 2) / (2 * 1.5 ** 2)) if i < 1/dt else 0.0

    # Update sequence memory field
    u_field_1, h_u_1 = compute_field_update(u_field_1, theta, w_hat, h_u_1, tau_h, inputs_1[i, :])
    
    # Update task duration field
    u_d, h_u_d = compute_field_update(u_d, theta, w_hat, h_u_d, tau_h_d, input_d)

    # Store time course data
    u_1_tc.append([u_field_1[idx] for idx in input_indices_1])
    u_d_tc.append(u_d[int(len(x) / 2)])

    # Update plots
    if plot_fields and (i % plot_every == 0 or i == len(t) - 1):
        line1_field.set_ydata(u_field_1)
        line1_input.set_ydata(inputs_1[i, :])
        line2_field.set_ydata(u_d)
        
        axs[0].set_title(f"Sequence Memory Field - Time = {t[i]:.2f}")
        axs[1].set_title(f"Task Duration Field - Time = {t[i]:.2f}")
        
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(plot_delay)

print(f"Max of u_sm1: {max(u_field_1)}")
print(f"Max of u_d: {max(u_d)}")

# ====================================
# -------- Save Results --------------
# ====================================

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
file_path_1 = results_dir / f"u_field_1_{timestamp}.npy"
file_path_2 = results_dir / f"u_d_{timestamp}.npy"

np.save(file_path_1, u_field_1)
np.save(file_path_2, u_d)

print(f"Saved u_field_1 to {file_path_1}")
print(f"Saved u_d to {file_path_2}")

plt.ioff()
plt.show()

# ====================================
# -------- Analysis ------------------
# ====================================

# Convert to arrays
u_f1_history = np.array(u_1_tc)
u_d_history = np.array(u_d_tc)
timesteps = np.arange(len(u_f1_history))

# Find threshold crossings
print("\nThreshold crossings:")
for i, pos in enumerate(input_position_1):
    crossing_idx = np.argmax(u_f1_history[:, i] >= theta)
    print(f"u_field_1 at x={pos} crosses theta at time {crossing_idx*dt}")

crossing_idx = np.argmax(u_d_history >= theta)
print(f"u_d at x=0 crosses theta at time {crossing_idx*dt}")

# ====================================
# -------- Visualization -------------
# ====================================

fig, axs = plt.subplots(2, 1, figsize=(9, 5), sharex=False)

# Plot u_field_1 time courses
for i, pos in enumerate(input_position_1):
    axs[0].plot(timesteps, u_f1_history[:, i], label=f'x = {pos}')
axs[0].axhline(theta, color='r', linestyle='--', label='theta = 1')
axs[0].set_ylabel('u_field_1')
axs[0].set_ylim(-1, 5)
axs[0].legend()
axs[0].grid(True)

# Plot u_d time course
axs[1].plot(timesteps, u_d_history, label='x = 0')
axs[1].axhline(theta, color='r', linestyle='--', label='theta = 1')
axs[1].set_ylabel('u_d')
axs[1].set_xlabel('Timestep')
axs[1].set_ylim(-1, 3)
axs[1].set_xlim(0, 100)
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()