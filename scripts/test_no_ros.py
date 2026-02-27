"""
test_no_ros.py

Runs CorrectionExperiment without ROS — use this to validate
field dynamics on Windows before deploying to the robot PC.

Usage:
    python test_no_ros.py
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# make sure scripts/ is on the path when running from repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from correction_experiment import CorrectionExperiment, FeedbackType

# ====================================
# -------- Config --------------------
# ====================================

DATA_PATH    = os.path.join(os.path.dirname(__file__), "..", "data")
N_ITERATIONS = 2
T_LIM        = 30.0
T_FEEDBACK   = 30.0
TRIGGER_STEP = 100

# One correction per run — change to test different feedback types
# format: (FeedbackType, action_name)           — SKIP, LOCK, EARLY, LATE
# format: (FeedbackType, action_name, target)   — SWAP
HUMAN_FEEDBACK = (FeedbackType.SKIP, "grasp")

ACTION_COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]


# ====================================
# -------- Run -----------------------
# ====================================

def run():
    exp = CorrectionExperiment(
        data_path      = DATA_PATH,
        n_iterations   = N_ITERATIONS,
        t_lim          = T_LIM,
        t_feedback_lim = T_FEEDBACK,
        trigger_step   = TRIGGER_STEP,
    )

    for iteration in range(N_ITERATIONS):

        # ── Loop 1: Execution ─────────────────────────────────────────
        exp.start_iteration()
        u_act_memory_before = exp.u_act_memory.copy()

        while True:
            result = exp.execution_step()
            if result is None:
                print("Loop 1 complete.")
                break

        # ── Loop 2: Feedback window ───────────────────────────────────
        exp.set_feedback(HUMAN_FEEDBACK)

        while True:
            still_running = exp.feedback_step()
            if not still_running:
                print("Loop 2 complete.")
                break

        # ── Apply correction ──────────────────────────────────────────
        exp.apply_correction()

        # ── Plot this iteration ───────────────────────────────────────
        _plot_iteration(exp, iteration, u_act_memory_before, HUMAN_FEEDBACK)

    print("\nDone. All iterations complete.")


# ====================================
# -------- Plot ----------------------
# ====================================

def _plot_iteration(exp, iteration, u_act_memory_before, human_feedback):
    feedback_type = human_feedback[0]
    action_name   = human_feedback[1]
    target_name   = human_feedback[2] if feedback_type == FeedbackType.SWAP else None

    action_center = exp._resolve_action(action_name)
    target_idx    = np.argmin(np.abs(exp.x - action_center))
    ff            = exp.feedback_fields[feedback_type]

    title_str = f"{feedback_type.value.upper()}: {action_name}"
    if target_name:
        title_str += f" ↔ {target_name}"

    fig, axs = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle(f"Iteration {iteration + 1} — {title_str}")

    # memory before/after
    for bucket_idx, color, name in zip(exp.action_buckets, ACTION_COLORS, exp.action_names):
        axs[0].plot(exp.x[bucket_idx], u_act_memory_before[bucket_idx],
                    '--', linewidth=2, color=color)
        axs[0].plot(exp.x[bucket_idx], exp.u_act_memory[bucket_idx],
                    '-',  linewidth=2, color=color, label=name)
    axs[0].set_title("Memory: before (--) vs after (-)")
    axs[0].legend(loc="upper right")
    axs[0].grid(True)

    # feedback field spatial profile at end of window
    axs[1].plot(exp.x, ff.u,        label="u at end")
    axs[1].plot(exp.x, ff.output(), linestyle='--', label="output")
    axs[1].axhline(ff.params.theta, color='k',    linestyle='--', label='theta')
    axs[1].axvline(action_center,   color='gray', linestyle=':',  label=action_name)
    if target_name:
        target_center = exp._resolve_action(target_name)
        axs[1].axvline(target_center, color='orange', linestyle=':', label=target_name)
    axs[1].set_title(f"{feedback_type.value.upper()} field spatial profile")
    axs[1].set_xlabel("x")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run()