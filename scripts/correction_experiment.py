import numpy as np
import os
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from utils import kernel_gauss, kernel_osc, compute_convolution, compute_action_bounds


# ====================================
# -------- Feedback types ------------
# ====================================

class FeedbackType(Enum):
    SKIP  = "skip"
    LOCK  = "lock"
    EARLY = "early"
    LATE  = "late"
    SWAP  = "swap"
    # TODO: ADD = "add"


# ====================================
# -------- Feedback field ------------
# ====================================

@dataclass
class FeedbackFieldParams:
    kernel_type: str   # "gauss" or "osc"
    kernel_pars: list
    h_0: float         # resting level
    theta: float       # threshold
    tau: float         # time constant
    transient: bool    # True = decaying, False = sustained


class FeedbackField:
    def __init__(self, x, dx, params: FeedbackFieldParams, input_duration):
        self.x  = x
        self.dx = dx
        self.params = params
        self.u = params.h_0 * np.ones(len(x))
        self.h = params.h_0 * np.ones(len(x))
        self.s = np.zeros(len(x))
        self.input_duration = input_duration
        self._inject_step   = None

        if params.kernel_type == "gauss":
            self.kernel = kernel_gauss(x, *params.kernel_pars)
        elif params.kernel_type == "osc":
            self.kernel = kernel_osc(x, *params.kernel_pars)
        self.w_hat = np.fft.fft(self.kernel)

    def inject(self, center, amplitude, width, current_step):
        """Set Gaussian external input — applied every step until cleared."""
        self.s = amplitude * np.exp(-0.5 * ((self.x - center) / width) ** 2)
        self._inject_step = current_step

    def inject_add(self, center, amplitude, width):
        """Add a second Gaussian to existing input — used for SWAP."""
        self.s += amplitude * np.exp(-0.5 * ((self.x - center) / width) ** 2)

    def clear_input(self):
        self.s = np.zeros(len(self.x))
        self._inject_step = None

    def output(self):
        return np.heaviside(self.u - self.params.theta, 1.0)

    def step(self, dt, current_step):
        """One Euler step of field dynamics."""
        if (self.params.transient and
                self._inject_step is not None and
                current_step >= self._inject_step + self.input_duration):
            self.clear_input()

        conv    = compute_convolution(self.u, self.params.theta, self.w_hat, self.dx)
        self.u += (dt / self.params.tau) * (-self.u + conv + self.h + self.s)

    def reset(self):
        self.u = self.params.h_0 * np.ones(len(self.x))
        self.clear_input()


# ====================================
# -------- Coupling functions --------
# ====================================

EARLY_LATE_COUPLING = 0.5


def apply_skip_from_field(u_act_memory, skip_field):
    u_new = u_act_memory.copy()
    u_new[skip_field.output() > 0] = u_act_memory[0]
    return u_new


def apply_lock_from_field(h_amem_mask, lock_field):
    mask_new = h_amem_mask.copy()
    mask_new[lock_field.output() > 0] = 0.0
    return mask_new


def apply_early_from_field(u_act_memory, early_field, theta_act):
    u_new  = u_act_memory.copy()
    u_new -= EARLY_LATE_COUPLING * early_field.output() * np.heaviside(u_act_memory - theta_act, 1.0)
    return u_new


def apply_late_from_field(u_act_memory, late_field, theta_act):
    u_new  = u_act_memory.copy()
    u_new += EARLY_LATE_COUPLING * late_field.output() * np.heaviside(u_act_memory - theta_act, 1.0)
    return u_new


def apply_swap_from_field(u_act_memory, swap_field):
    u_new  = u_act_memory.copy()
    active = swap_field.output() > 0

    regions   = []
    in_region = False
    start     = 0
    for i in range(len(active)):
        if active[i] and not in_region:
            start     = i
            in_region = True
        elif not active[i] and in_region:
            regions.append(np.arange(start, i))
            in_region = False
    if in_region:
        regions.append(np.arange(start, len(active)))

    if len(regions) != 2:
        print(f"[WARNING] SWAP expected 2 active regions, found {len(regions)}. Skipping.")
        return u_act_memory

    idx_a, idx_b = regions[0], regions[1]
    u_new[idx_a], u_new[idx_b] = u_new[idx_b].copy(), u_new[idx_a].copy()
    return u_new


# ====================================
# -------- Main experiment class -----
# ====================================

class CorrectionExperiment:
    """
    Full DNF correction experiment — pure Python, no ROS dependencies.
    ROS node wraps this class and calls its methods via timer callbacks.
    """

    def __init__(self, data_path, n_iterations=2, t_lim=30.0,
                 t_feedback_lim=30.0, trigger_step=100):

        # ── experiment params ─────────────────────────────────────────
        self.data_path       = data_path
        self.n_iterations    = n_iterations
        self.t_lim           = t_lim
        self.t_feedback_lim  = t_feedback_lim
        self.trigger_step    = trigger_step

        # ── spatial / temporal grid ───────────────────────────────────
        self.x_lim = 80
        self.dx    = 0.05
        self.dt    = 0.05

        self.x = np.arange(-self.x_lim, self.x_lim + self.dx, self.dx)
        self.t = np.arange(0, self.t_lim + self.dt, self.dt)
        self.t_feedback = np.arange(0, self.t_feedback_lim + self.dt, self.dt)

        # ── action layout ─────────────────────────────────────────────
        self.input_positions = [-60, -30, 0, 30, 60]
        self.action_names    = ["reach", "grasp", "lift", "transport", "place"]
        self.input_indices   = [np.argmin(np.abs(self.x - p)) for p in self.input_positions]
        self.action_buckets  = compute_action_bounds(self.x, self.input_positions)

        # ── main field params ─────────────────────────────────────────
        self.kernel_pars_act = [1.5, 0.8, 0.1]
        self.kernel_pars_wm  = [1.75, 0.5, 0.8]
        self.h_0_wm          = -1.0
        self.theta_wm        = 0.8
        self.tau_h_act       = 20
        self.theta_act       = 1.5

        kernel_act    = kernel_gauss(self.x, *self.kernel_pars_act)
        kernel_wm     = kernel_osc(self.x,   *self.kernel_pars_wm)
        self.w_hat_act = np.fft.fft(kernel_act)
        self.w_hat_wm  = np.fft.fft(kernel_wm)

        # ── feedback field setup ──────────────────────────────────────
        self.input_duration = 50

        ff_params = FeedbackFieldParams(
            kernel_type="gauss",
            kernel_pars=[2.0, 0.8, 0.05],
            h_0=-1.0, theta=0.5, tau=1.0, transient=False
        )

        self.feedback_fields = {
            FeedbackType.SKIP:  FeedbackField(self.x, self.dx, ff_params, self.input_duration),
            FeedbackType.LOCK:  FeedbackField(self.x, self.dx, ff_params, self.input_duration),
            FeedbackType.EARLY: FeedbackField(self.x, self.dx, ff_params, self.input_duration),
            FeedbackType.LATE:  FeedbackField(self.x, self.dx, ff_params, self.input_duration),
            FeedbackType.SWAP:  FeedbackField(self.x, self.dx, ff_params, self.input_duration),
        }

        # ── load memory ───────────────────────────────────────────────
        self.u_act_memory, self.h_d_initial, self.input_action_onset = self._load_memory()
        self.h_amem_mask = np.ones(len(self.x))

        # ── runtime state (reset each iteration) ─────────────────────
        self.u_act   = None
        self.u_wm    = None
        self.h_u_act = None
        self.h_u_wm  = None

        self.u_act_history = []
        self.u_wm_history  = []

        # ── feedback state ────────────────────────────────────────────
        # format: (FeedbackType, action_name) or (FeedbackType, action_name, target_name)
        self.pending_feedback = None
        self.feedback_injected = False

        # ── iteration tracking ────────────────────────────────────────
        self.iteration        = 0
        self.exec_step        = 0
        self.feedback_step_i  = 0
        self.phase            = "execution"   # "execution" | "feedback" | "done"

        # ── callbacks (set by ROS node or test runner) ────────────────
        self.on_execution_complete = None   # called when Loop 1 finishes
        self.on_experiment_done    = None   # called when all iterations finish
        self.on_field_update       = None   # called each exec step with field values

        print(f"[CorrectionExperiment] Initialized. "
              f"{n_iterations} iterations, t_lim={t_lim}s, t_feedback={t_feedback_lim}s")

    # ====================================
    # -------- Memory I/O ----------------
    # ====================================

    def _load_memory(self):
        """Load u_act_memory and u_d from data_path."""
        try:
            u_field_files = sorted([f for f in os.listdir(self.data_path)
                                    if f.startswith("u_field_1_")])
            u_d_files     = sorted([f for f in os.listdir(self.data_path)
                                    if f.startswith("u_d_")])

            if not u_field_files or not u_d_files:
                raise FileNotFoundError("Memory files not found.")

            u_field_1 = np.load(os.path.join(self.data_path, u_field_files[-1]))
            u_d       = np.load(os.path.join(self.data_path, u_d_files[-1]))

            u_d         = u_d.flatten()
            h_d_initial = float(np.max(u_d))

            input_action_onset = u_field_1.flatten()
            u_act_memory       = u_field_1.copy()

            print(f"[CorrectionExperiment] Memory loaded from {self.data_path}")
            return u_act_memory, h_d_initial, input_action_onset

        except (FileNotFoundError, IndexError) as e:
            print(f"[CorrectionExperiment] No memory found ({e}), using defaults.")
            h_d_initial        = 3.2
            input_action_onset = np.zeros(len(self.x))
            u_act_memory       = np.zeros(len(self.x))
            return u_act_memory, h_d_initial, input_action_onset

    def save_memory(self):
        """Save updated u_act_memory and h_amem_mask to data_path."""
        timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
        memory_path = os.path.join(self.data_path,
                                   f"u_act_memory_iter{self.iteration}_{timestamp}.npy")
        mask_path   = os.path.join(self.data_path,
                                   f"h_amem_mask_iter{self.iteration}_{timestamp}.npy")
        np.save(memory_path, self.u_act_memory)
        np.save(mask_path,   self.h_amem_mask)
        print(f"[CorrectionExperiment] Memory saved: {memory_path}")

    # ====================================
    # -------- Iteration reset -----------
    # ====================================

    def start_iteration(self):
        """Reset all fields and state for a new execution iteration."""
        self.iteration       += 1
        self.exec_step        = 0
        self.feedback_step_i  = 0
        self.feedback_injected = False
        self.pending_feedback  = None
        self.phase             = "execution"

        # reset execution fields from current memory
        self.u_act   = self.u_act_memory.copy() - self.h_d_initial + 1.5
        self.u_wm    = self.h_0_wm * np.ones(len(self.x))
        self.h_u_act = -self.h_d_initial * np.ones(len(self.x)) + 1.5
        self.h_u_wm  = self.h_0_wm * np.ones(len(self.x))

        # reset feedback fields
        for ff in self.feedback_fields.values():
            ff.reset()

        self.u_act_history = []
        self.u_wm_history  = []

        print(f"\n{'='*50}")
        print(f"Iteration {self.iteration}")
        print(f"{'='*50}")
        print("Loop 1: Executing sequence...")

    # ====================================
    # -------- Execution step ------------
    # ====================================

    def execution_step(self):
        """
        One Euler step of execution fields (u_act, u_wm).
        Returns field values at input_indices for publishing.
        Call repeatedly until returns None (execution complete).
        """
        if self.exec_step >= len(self.t):
            return None  # signal: execution complete

        conv_act = compute_convolution(self.u_act, self.theta_act,
                                       self.w_hat_act, self.dx)
        conv_wm  = compute_convolution(self.u_wm,  self.theta_wm,
                                       self.w_hat_wm,  self.dx)
        f_act = np.heaviside(self.u_act - self.theta_act, 1)
        f_wm  = np.heaviside(self.u_wm  - self.theta_wm,  1)

        self.h_u_act += self.dt / self.tau_h_act

        self.u_act += self.dt * (-self.u_act + conv_act + self.input_action_onset
                                 + self.h_u_act - 6.0 * f_wm * conv_wm)
        self.u_wm  += (self.dt / 1.25) * (-self.u_wm + conv_wm
                                           + 8 * (f_act * self.u_act) + self.h_u_wm)

        self.u_act_history.append([self.u_act[idx] for idx in self.input_indices])
        self.u_wm_history.append([self.u_wm[idx]   for idx in self.input_indices])

        self.exec_step += 1

        # return current field values at action positions for publishing
        return [self.u_act[idx] for idx in self.input_indices]

    # ====================================
    # -------- Feedback step -------------
    # ====================================

    def set_feedback(self, feedback_tuple):
        """
        Receive feedback from human/speech recognition.
        feedback_tuple: (FeedbackType, action_name) or (FeedbackType, action_name, target_name)
        """
        self.pending_feedback = feedback_tuple
        print(f"\nLoop 2: Feedback window open...")

    def feedback_step(self):
        """
        One Euler step of feedback fields.
        Call repeatedly until returns False (feedback window complete).
        """
        if self.feedback_step_i >= len(self.t_feedback):
            return False  # signal: feedback window complete

        # inject at trigger_step
        if self.feedback_step_i == self.trigger_step and not self.feedback_injected:
            self._inject_feedback()
            self.feedback_injected = True

        # update all feedback fields
        for ff in self.feedback_fields.values():
            ff.step(self.dt, self.feedback_step_i)

        self.feedback_step_i += 1
        return True

    def _inject_feedback(self):
        """Inject Gaussian input into the appropriate feedback field."""
        if self.pending_feedback is None:
            return

        feedback_type = self.pending_feedback[0]
        action_name   = self.pending_feedback[1]
        action_center = self._resolve_action(action_name)
        ff = self.feedback_fields[feedback_type]

        print(f"  [{action_name} {feedback_type.value}] triggered at step {self.feedback_step_i}")
        ff.inject(center=action_center, amplitude=3.0, width=5.0,
                  current_step=self.feedback_step_i)

        if feedback_type == FeedbackType.SWAP:
            target_name   = self.pending_feedback[2]
            target_center = self._resolve_action(target_name)
            print(f"  [{target_name} {feedback_type.value}] triggered at step {self.feedback_step_i}")
            ff.inject_add(center=target_center, amplitude=3.0, width=5.0)

    # ====================================
    # -------- Apply correction ----------
    # ====================================

    def apply_correction(self):
        """Apply feedback field output to u_act_memory."""
        if self.pending_feedback is None:
            print("[CorrectionExperiment] No feedback to apply.")
            return

        feedback_type = self.pending_feedback[0]
        action_name   = self.pending_feedback[1]
        print(f"\nApplying correction: {feedback_type.value} {action_name}")

        if feedback_type == FeedbackType.SKIP:
            self.u_act_memory = apply_skip_from_field(
                self.u_act_memory, self.feedback_fields[FeedbackType.SKIP])

        elif feedback_type == FeedbackType.LOCK:
            self.h_amem_mask = apply_lock_from_field(
                self.h_amem_mask, self.feedback_fields[FeedbackType.LOCK])

        elif feedback_type == FeedbackType.EARLY:
            self.u_act_memory = apply_early_from_field(
                self.u_act_memory, self.feedback_fields[FeedbackType.EARLY], self.theta_act)

        elif feedback_type == FeedbackType.LATE:
            self.u_act_memory = apply_late_from_field(
                self.u_act_memory, self.feedback_fields[FeedbackType.LATE], self.theta_act)

        elif feedback_type == FeedbackType.SWAP:
            self.u_act_memory = apply_swap_from_field(
                self.u_act_memory, self.feedback_fields[FeedbackType.SWAP])

        self.save_memory()

    # ====================================
    # -------- Helpers -------------------
    # ====================================

    def _resolve_action(self, action_name):
        """Map action name to spatial center."""
        idx = [n.lower() for n in self.action_names].index(action_name.lower())
        return self.input_positions[idx]

    def is_done(self):
        return self.iteration >= self.n_iterations and self.phase == "done"