#!/usr/bin/env python3
"""
trajectory_demo.py  —  Автоматична демонстрація всіх сценаріїв.

Сценарій 1: home → pick_A → place_A → home
Сценарій 2: Тест порушень (задаємо небезпечні цілі → система блокує)
Сценарій 3: Рух навколо перешкод (home → up → pick_B → place_B → home)
"""

from __future__ import annotations

import time
import numpy as np
import rclpy
from rclpy.node import Node

from std_msgs.msg import Bool, Float64MultiArray, String
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped


class TrajectoryDemo(Node):

    def __init__(self):
        super().__init__('trajectory_demo')

        self.declare_parameter('scenario',  1)
        self.declare_parameter('delay',     4.0)
        self.declare_parameter('auto_run',  True)

        self._scenario  = self.get_parameter('scenario').value
        self._delay     = self.get_parameter('delay').value
        self._auto_run  = self.get_parameter('auto_run').value

        self._q         = np.zeros(6)
        self._has_state = False
        self._is_safe   = True
        self._last_status = ""

        # Підписки
        self.create_subscription(JointState, '/joint_states', self._js_cb, 10)
        self.create_subscription(Bool,   '/puma/is_safe',       self._safe_cb, 10)
        self.create_subscription(Bool,   '/puma/motion_allowed',self._allow_cb, 10)
        self.create_subscription(String, '/puma/planner_status',self._status_cb, 10)

        # Публікатори
        self._preset_pub = self.create_publisher(String, '/puma/plan_request', 10)
        self._joints_pub = self.create_publisher(
            Float64MultiArray, '/puma/target_joints', 10)
        self._pose_pub   = self.create_publisher(
            PoseStamped, '/puma/target_pose', 10)
        self._estop_pub  = self.create_publisher(Bool, '/puma/emergency_stop', 10)

        self.get_logger().info(f"Демо-вузол запущено, сценарій {self._scenario}")

        if self._auto_run:
            self.create_timer(3.0, self._start_once)
        self._started = False

    def _js_cb(self, msg: JointState):
        if len(msg.position) >= 6:
            self._q = np.array(msg.position[:6])
            self._has_state = True

    def _safe_cb(self, msg: Bool):      self._is_safe = msg.data
    def _allow_cb(self, msg: Bool):     pass
    def _status_cb(self, msg: String):
        self._last_status = msg.data
        self.get_logger().info(f"Статус: {msg.data}")

    def _start_once(self):
        if self._started or not self._has_state:
            return
        self._started = True
        {
            1: self._scenario_1,
            2: self._scenario_2,
            3: self._scenario_3,
        }.get(self._scenario, self._scenario_1)()

    # ═══════════════════════════════════════════════════════════════
    #  Сценарій 1: Підбір і укладання (стандартний)
    # ═══════════════════════════════════════════════════════════════
    def _scenario_1(self):
        self.get_logger().info("=== Сценарій 1: Підбір і укладання ===")
        steps = [
            ("home",    "Початкова позиція"),
            ("ready",   "Готовність"),
            ("pick_A",  "Підбір деталі A"),
            ("ready",   "Підйом"),
            ("place_A", "Укладання"),
            ("ready",   "Відхід"),
            ("home",    "Повернення додому"),
        ]
        self._run_preset_sequence(steps)

    # ═══════════════════════════════════════════════════════════════
    #  Сценарій 2: Тест системи безпеки (навмисні порушення)
    # ═══════════════════════════════════════════════════════════════
    def _scenario_2(self):
        self.get_logger().info("=== Сценарій 2: Тест системи безпеки ===")

        self.get_logger().info("--- Крок 1: Нормальний рух (має бути ДОЗВОЛЕНО)")
        self._send_preset("home")
        self._wait(self._delay)

        self.get_logger().info("--- Крок 2: Ціль поза робочим простором (має бути ЗАБОРОНЕНО)")
        self._send_pose(2.0, 0.0, 0.5)   # далеко за межами
        self._wait(2.0)

        self.get_logger().info("--- Крок 3: Ціль занадто низько (має бути ЗАБОРОНЕНО)")
        self._send_pose(0.3, 0.0, -0.5)   # нижче підлоги
        self._wait(2.0)

        self.get_logger().info("--- Крок 4: Ціль всередині перешкоди A (має бути ЗАБОРОНЕНО)")
        self._send_pose(0.5, 0.3, 0.87)   # центр перешкоди A
        self._wait(2.0)

        self.get_logger().info("--- Крок 5: Порушення обмежень суглобів (має бути ЗАБОРОНЕНО)")
        bad_q = Float64MultiArray()
        bad_q.data = [0.0, -2.0, 2.0, 0.0, 0.0, 0.0]   # joint2,3 поза межами
        self._joints_pub.publish(bad_q)
        self._wait(2.0)

        self.get_logger().info("--- Крок 6: Повернення в безпечну позицію")
        self._send_preset("home")
        self._wait(self._delay)

        self.get_logger().info("=== Тест безпеки завершено ===")

    # ═══════════════════════════════════════════════════════════════
    #  Сценарій 3: Обхід перешкод
    # ═══════════════════════════════════════════════════════════════
    def _scenario_3(self):
        self.get_logger().info("=== Сценарій 3: Обхід перешкод ===")
        steps = [
            ("home",    "Старт"),
            ("up",      "Підйом над перешкодами"),
            ("pick_B",  "Підбір деталі B (лівий бік)"),
            ("up",      "Підйом"),
            ("place_B", "Укладання B"),
            ("up",      "Відхід"),
            ("home",    "Повернення"),
        ]
        self._run_preset_sequence(steps)

    # ═══════════════════════════════════════════════════════════════
    #  Утиліти
    # ═══════════════════════════════════════════════════════════════
    def _run_preset_sequence(self, steps: list):
        for i, (preset, desc) in enumerate(steps, 1):
            self.get_logger().info(f"[{i}/{len(steps)}] {desc} → {preset}")
            self._send_preset(preset)
            self._wait(self._delay)

        self.get_logger().info("✓ Сценарій завершено")

    def _send_preset(self, name: str):
        msg = String(); msg.data = name
        self._preset_pub.publish(msg)

    def _send_pose(self, x: float, y: float, z: float):
        msg = PoseStamped()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'world'
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = z
        msg.pose.orientation.w = 1.0
        self._pose_pub.publish(msg)
        self.get_logger().info(f"Надіслано ціль XYZ: ({x:.2f}, {y:.2f}, {z:.2f})")

    def _wait(self, sec: float):
        end = time.time() + sec
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.1)


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryDemo()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
