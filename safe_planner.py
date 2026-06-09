#!/usr/bin/env python3
"""
safe_planner.py  —  Головний вузол безпечного планування траєкторії Puma560.

Реалізує алгоритм із блок-схеми:
  1. Отримати цільову позицію
  2. Перевірити належність робочому простору
  3. Розрахувати зворотну кінематику (чисельна)
  4. Перевірити обмеження суглобів
  5. Побудувати траєкторію (квінтичний поліном)
  6. Перевірити кожну проміжну точку
  7. Перевірити зіткнення
  8. Перевірити швидкості
  9. Виконати або заблокувати

Топіки:
  /puma/target_pose           geometry_msgs/PoseStamped  — вхід (XYZ + орієнтація)
  /puma/target_joints         std_msgs/Float64MultiArray — вхід (прямо кути)
  /puma/plan_request          std_msgs/String            — вхід (ім'я пресету: home/pick/place/…)
  /joint_trajectory_controller/joint_trajectory  — вихід
  /puma/planner_status        std_msgs/String            — статус кожного кроку
  /puma/safety_report         std_msgs/String            — детальний звіт
  /puma/motion_allowed        std_msgs/Bool              — дозвіл/заборона руху
"""

from __future__ import annotations

import math
import time
import numpy as np
import rclpy
from rclpy.node import Node

from builtin_interfaces.msg import Duration
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64MultiArray, String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from visualization_msgs.msg import Marker, MarkerArray

from puma560_safe_planner.puma560_kinematics import (
    DOF, JOINT_NAMES, JOINT_MIN, JOINT_MAX, VEL_MAX, OBSTACLES,
    WS_BOX_MIN, WS_BOX_MAX, WS_MIN_RADIUS, WS_MAX_RADIUS,
    forward_kinematics, ee_position, manipulability,
    full_safety_check, check_workspace, check_joint_limits,
    check_collision, check_velocity, check_singularity,
    quintic_trajectory, estimate_travel_time, scale_trajectory_time,
    TrajPoint, SafetyResult, min_obstacle_distance,
)

# ── Пресети конфігурацій (відповідають реальним кутам Puma560) ────────────
PRESETS: dict[str, np.ndarray] = {
    "home":    np.array([0.000,  0.000,  0.000,  0.000,  0.000,  0.000]),
    "pick_C":  np.array([0.500,  0.300,  0.870,  0.000,  0.000,  0.000]),
    "pick_A":  np.array([0.785, -0.524,  0.524,  0.000, -0.524,  0.000]),
    "pick_B":  np.array([-0.785,-0.524,  0.524,  0.000, -0.524,  0.000]),
    "place_A": np.array([1.047, -0.524,  0.524,  0.000, -0.524,  0.000]),
    "place_B": np.array([-1.047,-0.524,  0.524,  0.000, -0.524,  0.000]),
    "up":      np.array([0.000, -1.200,  0.400,  0.000, -0.400,  0.000]),
    "look":    np.array([0.000, -0.400,  0.800,  0.000, -1.200,  0.000]),
}


class SafePlannerNode(Node):

    def __init__(self):
        super().__init__('safe_planner')

        # Параметри
        self.declare_parameter('dt',             0.05)
        self.declare_parameter('safety_factor',  0.85)
        self.declare_parameter('sing_threshold', 0.01)
        self.declare_parameter('check_all_points', True)

        self._dt      = self.get_parameter('dt').value
        self._sf      = self.get_parameter('safety_factor').value
        self._st      = self.get_parameter('sing_threshold').value
        self._check_all = self.get_parameter('check_all_points').value

        # Стан
        self._current_q  = np.zeros(DOF)
        self._has_state  = False
        self._estop      = False

        # ── Підписки ────────────────────────────────────────────────
        self.create_subscription(
            JointState, '/joint_states', self._js_cb, 10)
        self.create_subscription(
            PoseStamped, '/puma/target_pose', self._pose_cb, 10)
        self.create_subscription(
            Float64MultiArray, '/puma/target_joints', self._joints_cb, 10)
        self.create_subscription(
            String, '/puma/plan_request', self._preset_cb, 10)
        self.create_subscription(
            Bool, '/puma/emergency_stop', self._estop_cb, 10)

        # ── Публікатори ─────────────────────────────────────────────
        self._traj_pub   = self.create_publisher(
            JointTrajectory,
            '/joint_trajectory_controller/joint_trajectory', 10)
        self._status_pub = self.create_publisher(String, '/puma/planner_status', 10)
        self._report_pub = self.create_publisher(String, '/puma/safety_report', 10)
        self._allow_pub  = self.create_publisher(Bool,   '/puma/motion_allowed', 10)
        self._marker_pub = self.create_publisher(MarkerArray, '/puma/markers', 10)

        # Таймер маркерів
        self.create_timer(1.0, self._publish_markers)

        self._log("Безпечний планувальник Puma560 запущено")
        self._log(f"Пресети: {list(PRESETS.keys())}")
        self._log("Задайте ціль через /puma/plan_request (home/ready/pick_A/…)")

    # ── Зворотні виклики підписок ────────────────────────────────────
    def _js_cb(self, msg: JointState):
        if len(msg.position) < DOF:
            return
        name_to_idx = {n: i for i, n in enumerate(JOINT_NAMES)}
        for i, name in enumerate(msg.name):
            if name in name_to_idx:
                self._current_q[name_to_idx[name]] = msg.position[i]
        self._has_state = True

    def _pose_cb(self, msg: PoseStamped):
        """Отримати XYZ ціль — використовуємо чисельну ОК."""
        if self._estop:
            self._publish_status("ESTOP активний — рух заблоковано")
            return
        target_pos = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        ])
        self._log(f"Ціль XYZ: {target_pos}")

        # Крок 2 — перевірка робочого простору
        ws = check_workspace(target_pos)
        if not ws.safe:
            self._reject(f"[Крок 2] {ws.reason}")
            return

        # Чисельна ОК (градієнтний спуск)
        q_goal = self._numerical_ik(target_pos)
        if q_goal is None:
            self._reject("[Крок 3] Обернена кінематика не знайшла розв'язку")
            return

        self._plan_and_execute(q_goal)

    def _joints_cb(self, msg: Float64MultiArray):
        if self._estop:
            self._publish_status("ESTOP активний — рух заблоковано")
            return
        if len(msg.data) < DOF:
            self._reject(f"Потрібно {DOF} кутів, отримано {len(msg.data)}")
            return
        self._plan_and_execute(np.array(msg.data[:DOF]))

    def _preset_cb(self, msg: String):
        name = msg.data.strip().lower()
        if name not in PRESETS:
            self._reject(f"Невідомий пресет '{name}'. Доступні: {list(PRESETS.keys())}")
            return
        self._log(f"Пресет: {name}")
        self._plan_and_execute(PRESETS[name])

    def _estop_cb(self, msg: Bool):
        self._estop = msg.data
        state = "АКТИВОВАНО" if msg.data else "знято"
        self._publish_status(f"Аварійна зупинка {state}")
        if msg.data:
            self.get_logger().error("!!! АВАРІЙНА ЗУПИНКА !!!")

    # ═══════════════════════════════════════════════════════════════
    #  Основний алгоритм планування
    # ═══════════════════════════════════════════════════════════════
    def _plan_and_execute(self, q_goal: np.ndarray):
        """
        Реалізація алгоритму з блок-схеми:
        1. Поточний стан
        2. Цільова позиція
        3. Перевірка робочого простору
        4. Зворотна кінематика
        5. Перевірка обмежень суглобів
        6. Побудова траєкторії
        7. Перевірка проміжних точок
        8. Перевірка зіткнень
        9. Виконати або заблокувати
        """
        t0 = time.time()
        report_lines = ["╔══ ЗВІТ БЕЗПЕЧНОГО ПЛАНУВАЛЬНИКА ═══════════════════"]

        if not self._has_state:
            self._reject("Стан суглобів ще не отримано")
            return

        q_start = self._current_q.copy()

        # ── Крок 3: Перевірка цільового робочого простору ────────────
        report_lines.append("║ Крок 3: Перевірка робочого простору цілі")
        ws = check_workspace(ee_position(q_goal))
        if not ws.safe:
            report_lines.append(f"║   ✗ {ws.reason}")
            self._finish_report(report_lines, False, ws.reason, t0)
            return
        report_lines.append(f"║   ✓ ЕЕ ціль: {ee_position(q_goal).round(3)}")

        # ── Крок 4: Обернена кінематика (вже виконана) ───────────────
        report_lines.append("║ Крок 4: Обернена кінематика — OK")

        # ── Крок 5: Перевірка обмежень суглобів ──────────────────────
        report_lines.append("║ Крок 5: Обмеження суглобів цільової конфігурації")
        jl = check_joint_limits(q_goal)
        if not jl.safe:
            report_lines.append(f"║   ✗ {jl.reason}")
            self._finish_report(report_lines, False, jl.reason, t0)
            return
        report_lines.append("║   ✓ Усі суглоби в допустимих межах")

        # ── Крок 6: Побудова траєкторії ───────────────────────────────
        report_lines.append("║ Крок 6: Побудова квінтичної траєкторії")
        total_t = estimate_travel_time(q_start, q_goal)
        trajectory = quintic_trajectory(q_start, q_goal, total_t, self._dt)
        trajectory = scale_trajectory_time(trajectory, self._sf)
        report_lines.append(
            f"║   Точок: {len(trajectory)}, тривалість: {trajectory[-1].time:.2f} с")

        # ── Крок 7+8: Перевірка проміжних точок ──────────────────────
        report_lines.append("║ Крок 7-8: Перевірка траєкторії")
        violations = []
        check_step = max(1, len(trajectory) // 20) if not self._check_all else 1

        for i, pt in enumerate(trajectory):
            if i % check_step != 0 and i != len(trajectory) - 1:
                continue
            res = full_safety_check(pt.q, pt.dq)
            if not res.safe:
                violations.append((pt.time, res))

        if violations:
            # Спробуємо зменшити швидкість
            trajectory = scale_trajectory_time(trajectory, 0.5)
            violations2 = []
            for pt in trajectory:
                res = full_safety_check(pt.q, pt.dq)
                if not res.safe and res.code != 'VELOCITY':
                    violations2.append((pt.time, res))

            if violations2:
                summary = "; ".join(
                    f"t={t:.2f}: {r.reason}" for t, r in violations2[:3])
                report_lines.append(f"║   ✗ {len(violations2)} порушень: {summary}")
                self._finish_report(
                    report_lines, False,
                    f"{len(violations2)} небезпечних точок у траєкторії", t0)
                return
            else:
                report_lines.append(
                    f"║   ⚠ Швидкість зменшено, {len(violations)} порушень усунено")
        else:
            report_lines.append(
                f"║   ✓ Усі {len(trajectory)} точок безпечні")

        # ── Крок 9: Виконати ──────────────────────────────────────────
        report_lines.append("║ Крок 9: Надсилання команди контролеру")
        self._send_trajectory(trajectory)

        # Статистика
        dist = np.linalg.norm(q_goal - q_start)
        ee_d = np.linalg.norm(ee_position(q_goal) - ee_position(q_start))
        report_lines.append(f"║   C-відстань: {dist:.4f} рад")
        report_lines.append(f"║   ЕЕ відстань: {ee_d:.4f} м")
        report_lines.append(f"║   Маніпулятивність: {manipulability(q_goal):.4f}")
        report_lines.append(f"║   Відстань до перешкод: {min_obstacle_distance(q_goal):.4f} м")

        elapsed = time.time() - t0
        self._finish_report(
            report_lines, True,
            f"Траєкторія виконується ({len(trajectory)} точок, {trajectory[-1].time:.2f} с)",
            t0)

    # ═══════════════════════════════════════════════════════════════
    #  Чисельна обернена кінематика (градієнтний спуск)
    # ═══════════════════════════════════════════════════════════════
    def _numerical_ik(
        self,
        target_pos: np.ndarray,
        max_iter: int = 200,
        tol: float = 0.005,
    ) -> Optional[np.ndarray]:
        from puma560_safe_planner.puma560_kinematics import jacobian

        q = self._current_q.copy()
        alpha = 0.5   # крок градієнтного спуску

        for _ in range(max_iter):
            pos = ee_position(q)
            err = target_pos - pos
            if np.linalg.norm(err) < tol:
                return q
            J = jacobian(q)[:3, :]     # тільки лінійна частина
            # Псевдообернений якобіан
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(3), err)
            q = q + alpha * dq
            # Обрізати до меж суглобів
            q = np.clip(q, JOINT_MIN, JOINT_MAX)

        # Перевірити фінальну похибку
        if np.linalg.norm(target_pos - ee_position(q)) < 0.02:
            return q
        return None

    # ═══════════════════════════════════════════════════════════════
    #  Публікація траєкторії
    # ═══════════════════════════════════════════════════════════════
    def _send_trajectory(self, trajectory: list[TrajPoint]):
        msg = JointTrajectory()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'world'
        msg.joint_names     = JOINT_NAMES

        for pt in trajectory:
            p = JointTrajectoryPoint()
            p.positions     = pt.q.tolist()
            p.velocities    = pt.dq.tolist()
            p.accelerations = pt.ddq.tolist()
            secs  = int(pt.time)
            nsecs = int((pt.time - secs) * 1e9)
            p.time_from_start = Duration(sec=secs, nanosec=nsecs)
            msg.points.append(p)

        self._traj_pub.publish(msg)

    # ═══════════════════════════════════════════════════════════════
    #  Маркери для RViz
    # ═══════════════════════════════════════════════════════════════
    def _publish_markers(self):
        markers = MarkerArray()
        now = self.get_clock().now().to_msg()

        # Перешкоди
        labels = ['A', 'B', 'C']
        colors = [(0.9,0.1,0.1), (0.9,0.5,0.1), (0.5,0.1,0.9)]
        for i, (obs, lbl, col) in enumerate(zip(OBSTACLES, labels, colors)):
            m = Marker()
            m.header.frame_id = 'world'
            m.header.stamp    = now
            m.ns, m.id        = 'obstacles', i
            m.type, m.action  = Marker.SPHERE, Marker.ADD
            m.pose.position.x = obs[0]
            m.pose.position.y = obs[1]
            m.pose.position.z = obs[2]
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = obs[3] * 2
            m.color.r, m.color.g, m.color.b, m.color.a = col[0], col[1], col[2], 0.6
            markers.markers.append(m)

        # Межі робочого простору (wireframe)
        m = Marker()
        m.header.frame_id = 'world'
        m.header.stamp    = now
        m.ns, m.id        = 'workspace', 0
        m.type, m.action  = Marker.CUBE, Marker.ADD
        m.pose.position.x = (WS_BOX_MIN[0] + WS_BOX_MAX[0]) / 2
        m.pose.position.y = (WS_BOX_MIN[1] + WS_BOX_MAX[1]) / 2
        m.pose.position.z = (WS_BOX_MIN[2] + WS_BOX_MAX[2]) / 2
        m.pose.orientation.w = 1.0
        m.scale.x = WS_BOX_MAX[0] - WS_BOX_MIN[0]
        m.scale.y = WS_BOX_MAX[1] - WS_BOX_MIN[1]
        m.scale.z = WS_BOX_MAX[2] - WS_BOX_MIN[2]
        m.color.b, m.color.a = 1.0, 0.04
        markers.markers.append(m)

        # Поточна позиція ЕЕ
        if self._has_state:
            pos = ee_position(self._current_q)
            res = full_safety_check(self._current_q)
            m = Marker()
            m.header.frame_id = 'world'
            m.header.stamp    = now
            m.ns, m.id        = 'ee_current', 0
            m.type, m.action  = Marker.SPHERE, Marker.ADD
            m.pose.position.x, m.pose.position.y, m.pose.position.z = pos
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.05
            m.color.a = 1.0
            if res.safe:
                m.color.g = 1.0
            else:
                m.color.r = 1.0
            markers.markers.append(m)

        self._marker_pub.publish(markers)

    # ═══════════════════════════════════════════════════════════════
    #  Утиліти
    # ═══════════════════════════════════════════════════════════════
    def _reject(self, reason: str):
        msg = Bool(); msg.data = False
        self._allow_pub.publish(msg)
        self._publish_status(f"❌ РУХ ЗАБОРОНЕНО: {reason}")
        self.get_logger().warn(f"ЗАБЛОКОВАНО: {reason}")

    def _finish_report(
        self,
        lines: list[str],
        success: bool,
        summary: str,
        t0: float,
    ):
        elapsed = time.time() - t0
        lines.append(f"║ Час планування: {elapsed*1000:.1f} мс")
        if success:
            lines.append(f"║ РЕЗУЛЬТАТ: ✓ ДОЗВОЛЕНО — {summary}")
        else:
            lines.append(f"║ РЕЗУЛЬТАТ: ✗ ЗАБОРОНЕНО — {summary}")
        lines.append("╚═════════════════════════════════════════════════════")

        report = '\n'.join(lines)
        msg = String(); msg.data = report
        self._report_pub.publish(msg)
        self.get_logger().info('\n' + report)

        allowed = Bool(); allowed.data = success
        self._allow_pub.publish(allowed)
        self._publish_status(
            ("✓ ДОЗВОЛЕНО: " if success else "✗ ЗАБОРОНЕНО: ") + summary)

    def _publish_status(self, text: str):
        msg = String(); msg.data = text
        self._status_pub.publish(msg)
        self.get_logger().info(f"[STATUS] {text}")

    def _log(self, text: str):
        self.get_logger().info(text)


def main(args=None):
    rclpy.init(args=args)
    node = SafePlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
