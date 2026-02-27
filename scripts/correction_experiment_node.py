#!/usr/bin/env python
"""
correction_experiment_node.py

Thin ROS wrapper around CorrectionExperiment.
All field dynamics live in correction_experiment.py.
This file only handles ROS communication and timer callbacks.
"""

import rospy
import threading
import numpy as np
from std_msgs.msg import Float32MultiArray, String, Bool

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from correction_experiment import CorrectionExperiment, FeedbackType


class CorrectionExperimentNode:

    def __init__(self):
        rospy.init_node('correction_experiment', anonymous=False)

        # ── load ROS params (set in launch file) ──────────────────────
        data_path      = rospy.get_param('~data_path',      '../data')
        n_iterations   = rospy.get_param('~n_iterations',   2)
        t_lim          = rospy.get_param('~t_lim',          30.0)
        t_feedback_lim = rospy.get_param('~t_feedback_lim', 30.0)
        trigger_step   = rospy.get_param('~trigger_step',   100)

        # ── core experiment (pure Python) ─────────────────────────────
        self.exp  = CorrectionExperiment(
            data_path      = data_path,
            n_iterations   = n_iterations,
            t_lim          = t_lim,
            t_feedback_lim = t_feedback_lim,
            trigger_step   = trigger_step,
        )

        self._lock = threading.Lock()

        # ── publishers ────────────────────────────────────────────────
        # field activity at action positions — robot uses this to know which action is active
        self.pub_field   = rospy.Publisher('field_activity',       Float32MultiArray, queue_size=10)
        # signals execution phase is complete
        self.pub_exec    = rospy.Publisher('execution_complete',   Bool,              queue_size=1)
        # signals memory has been updated after correction
        self.pub_updated = rospy.Publisher('memory_updated',       Bool,              queue_size=1)

        # ── subscribers ───────────────────────────────────────────────
        # human feedback string, e.g. "skip grasp" or "swap grasp lift"
        self.sub_feedback = rospy.Subscriber(
            'human_feedback', String, self._feedback_callback, queue_size=1)

        # ── execution timer (starts first iteration immediately) ──────
        self.exec_timer     = None
        self.feedback_timer = None

        rospy.loginfo("[CorrectionExperimentNode] Ready. Starting first iteration...")
        self._start_execution()

    # ====================================
    # -------- Execution phase -----------
    # ====================================

    def _start_execution(self):
        """Start Loop 1 timer."""
        with self._lock:
            self.exp.start_iteration()

        self.exec_timer = rospy.Timer(
            rospy.Duration(self.exp.dt), self._execution_callback)

    def _execution_callback(self, event):
        with self._lock:
            result = self.exp.execution_step()

        if result is None:
            # Loop 1 complete
            self.exec_timer.shutdown()
            rospy.loginfo("[CorrectionExperimentNode] Execution complete. Waiting for feedback...")
            self.pub_exec.publish(Bool(data=True))
            self._start_feedback_window()
        else:
            # publish field activity
            msg      = Float32MultiArray()
            msg.data = result
            self.pub_field.publish(msg)

    # ====================================
    # -------- Feedback phase ------------
    # ====================================

    def _start_feedback_window(self):
        """Start Loop 2 timer."""
        self.feedback_timer = rospy.Timer(
            rospy.Duration(self.exp.dt), self._feedback_step_callback)

    def _feedback_callback(self, msg):
        """Parse incoming human feedback string and pass to experiment."""
        try:
            parts = msg.data.strip().lower().split()

            # expected formats:
            #   "skip grasp"
            #   "lock lift"
            #   "early grasp"
            #   "late transport"
            #   "swap grasp lift"

            if len(parts) < 2:
                rospy.logwarn(f"[CorrectionExperimentNode] Invalid feedback: '{msg.data}'")
                return

            feedback_type = FeedbackType(parts[0])

            if feedback_type == FeedbackType.SWAP:
                if len(parts) < 3:
                    rospy.logwarn("[CorrectionExperimentNode] SWAP needs two action names.")
                    return
                feedback_tuple = (feedback_type, parts[1], parts[2])
            else:
                feedback_tuple = (feedback_type, parts[1])

            with self._lock:
                self.exp.set_feedback(feedback_tuple)

            rospy.loginfo(f"[CorrectionExperimentNode] Feedback received: {feedback_tuple}")

        except ValueError as e:
            rospy.logwarn(f"[CorrectionExperimentNode] Unknown feedback type: {e}")

    def _feedback_step_callback(self, event):
        with self._lock:
            still_running = self.exp.feedback_step()

        if not still_running:
            # Loop 2 complete
            self.feedback_timer.shutdown()
            rospy.loginfo("[CorrectionExperimentNode] Feedback window closed. Applying correction...")

            with self._lock:
                self.exp.apply_correction()

            self.pub_updated.publish(Bool(data=True))

            # start next iteration or finish
            if self.exp.iteration < self.exp.n_iterations:
                self._start_execution()
            else:
                rospy.loginfo("[CorrectionExperimentNode] All iterations complete.")
                rospy.signal_shutdown("Experiment finished.")


# ====================================
# -------- Entry point ---------------
# ====================================

if __name__ == "__main__":
    try:
        node = CorrectionExperimentNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("[CorrectionExperimentNode] Interrupted.")