#!/usr/bin/env python3
"""
safety_monitor.py  —  Монітор безпеки Puma560 у реальному часі (100 Гц).

Постійно перевіряє поточний стан і при порушенні:
  - публікує /puma/emergency_stop = True
  - публікує детальне попередження
  - логує причину

Також публікує діагностику: маніпулятивність, відстань до перешкод, стан.
"""

from __future__ import annotations

import time
import math
import numpy as np
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, String, Float64MultiArray

from puma560_safe_planner.puma560_kinematics import (
    DOF, JOINT_NAMES, VEL_MAX, OBSTACLES,
    ee_position, manipulability, full_safety_check,
    min_obstacle_distance, check_singularity,
    link_transforms,
)


class SafetyMonitor(Node):

    def __init__(self):
        super().__init__('safety_monitor')

        self.declare_parameter('check_rate_hz',    100.0)
        self.declare_parameter('sing_threshold',   0.01)
        self.declare_parameter('warn_obs_dist',    0.08)  # м
        self.declare_parameter('estop_on_violation', True)

        rate      = self.get_parameter('check_rate_hz').value
        self._st  = self.get_parameter('sing_threshold').value
        self._wd  = self.get_parameter('warn_obs_dist').value
        self._do_estop = self.get_parameter('estop_on_violation').value

        # Стан
        self._q         = np.zeros(DOF)
        self._prev_q    = np.zeros(DOF)
        self._vel       = np.zeros(DOF)
        self._has_state = False
        self._prev_time: float | None = None
        self._alpha     = 0.3   # EMA для швидкості

        self._estop_active = False

        # Статистика
        self._checks    = 0
        self._violations: dict[str, int] = {
            'WORKSPACE': 0, 'JOINT_LIMIT': 0,
            'COLLISION': 0, 'VELOCITY': 0, 'SINGULARITY': 0,
        }

        # ── Підписки ────────────────────────────────────────────────
        self.create_subscription(
            JointState, '/joint_states', self._js_cb, 10)
        self.create_subscription(
            Bool, '/puma/emergency_stop', self._estop_cb, 10)

        # ── Публікатори ─────────────────────────────────────────────
        self._estop_pub  = self.create_publisher(Bool,   '/puma/emergency_stop', 10)
        self._warn_pub   = self.create_publisher(String, '/puma/safety_warning', 10)
        self._diag_pub   = self.create_publisher(String, '/puma/diagnostics',    10)
        self._safe_pub   = self.create_publisher(Bool,   '/puma/is_safe',        10)
        self._manip_pub  = self.create_publisher(
            Float64MultiArray, '/puma/manipulability', 10)
        self._dist_pub   = self.create_publisher(
            Float64MultiArray, '/puma/obstacle_distances', 10)

        # Таймери
        period = 1.0 / rate
        self.create_timer(period,   self._check_cb)
        self.create_timer(0.5,      self._diag_cb)

        self.get_logger().info(
            f"Монітор безпеки запущено. Частота: {rate:.0f} Гц")

    # ── Зворотні виклики ─────────────────────────────────────────────
    def _js_cb(self, msg: JointState):
        now = time.monotonic()
        name_to_val = dict(zip(msg.name, msg.position))

        new_q = np.array([
            name_to_val.get(n, self._q[i])
            for i, n in enumerate(JOINT_NAMES)
        ])

        if self._prev_time is not None:
            dt = now - self._prev_time
            if 1e-6 < dt < 1.0:
                raw_vel = (new_q - self._prev_q) / dt
                self._vel = self._alpha * raw_vel + (1 - self._alpha) * self._vel

        self._prev_q    = self._q.copy()
        self._q         = new_q
        self._prev_time = now
        self._has_state = True

    def _estop_cb(self, msg: Bool):
        self._estop_active = msg.data

    # ── Основна перевірка ─────────────────────────────────────────────
    def _check_cb(self):
        if not self._has_state:
            return
        self._checks += 1

        q  = self._q.copy()
        dq = self._vel.copy()

        res = full_safety_check(q, dq)

        # Публікуємо загальний стан
        safe_msg = Bool(); safe_msg.data = res.safe
        self._safe_pub.publish(safe_msg)

        if not res.safe:
            self._violations[res.code] = self._violations.get(res.code, 0) + 1

            warn = String()
            warn.data = f"[{res.code}] {res.reason}"
            self._warn_pub.publish(warn)
            self.get_logger().warn(f"ПОРУШЕННЯ: {res.reason}")

            # Аварійна зупинка
            if self._do_estop and not self._estop_active:
                self._estop_active = True
                estop = Bool(); estop.data = True
                self._estop_pub.publish(estop)
                self.get_logger().error(f"!!! АВАРІЙНА ЗУПИНКА: {res.reason}")

        elif self._estop_active and res.safe:
            # Скинути ESTOP якщо стан нормалізувався
            self._estop_active = False
            estop = Bool(); estop.data = False
            self._estop_pub.publish(estop)
            self.get_logger().info("Аварійна зупинка знята — стан безпечний")

        # Маніпулятивність
        w = manipulability(q)
        m_msg = Float64MultiArray()
        m_msg.data = [w]
        self._manip_pub.publish(m_msg)

        # Відстані до кожної перешкоди
        transforms = link_transforms(q)
        dists = []
        for obs in OBSTACLES:
            d_min = min(
                float(np.linalg.norm(T[:3, 3] - obs[:3])) - obs[3]
                for T in transforms
            )
            dists.append(d_min)
        dist_msg = Float64MultiArray()
        dist_msg.data = dists
        self._dist_pub.publish(dist_msg)

        # Попередження якщо близько до перешкоди
        for i, d in enumerate(dists):
            if 0 < d < self._wd:
                lbl = chr(65 + i)
                warn = String()
                warn.data = f"[WARN] Близько до перешкоди {lbl}: {d:.3f} м"
                self._warn_pub.publish(warn)

    # ── Діагностика ───────────────────────────────────────────────────
    def _diag_cb(self):
        if not self._has_state:
            return

        q   = self._q.copy()
        pos = ee_position(q)
        w   = manipulability(q)
        od  = min_obstacle_distance(q)
        total_v = int(sum(self._violations.values()))

        lines = [
            "╔══ Puma560 Safety Monitor ═══════════════════════════",
            f"║  ЕЕ позиція : [{pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f}] м",
            f"║  Суглоби    : {[f'{v:.2f}' for v in q]}",
            f"║  Швидкості  : {[f'{v:.2f}' for v in self._vel]}",
            f"║  Маніпулятивність: {w:.4f}  {'⚠' if w < self._st*2 else '✓'}",
            f"║  Відстань до перешкод: {od:.3f} м  {'⚠' if od < self._wd else '✓'}",
            f"║  ESTOP: {'АКТИВНИЙ !!!' if self._estop_active else 'неактивний'}",
            f"║  Перевірок: {self._checks} | Порушень: {total_v}",
            "║  Порушення: " + " | ".join(
                f"{k}:{v}" for k, v in self._violations.items() if v > 0
            ),
            "╚═════════════════════════════════════════════════════",
        ]

        msg = String(); msg.data = '\n'.join(lines)
        self._diag_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = SafetyMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
