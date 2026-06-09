#!/usr/bin/env python3
"""
workspace_vis.py  —  Публікує маркери робочого простору для RViz.
Показує: робочу зону, перешкоди, поточну позицію ЕЕ, стан безпеки.
"""

from __future__ import annotations
import numpy as np
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, String, Float64MultiArray
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

from puma560_safe_planner.puma560_kinematics import (
    DOF, JOINT_NAMES, OBSTACLES,
    WS_BOX_MIN, WS_BOX_MAX, WS_MIN_RADIUS, WS_MAX_RADIUS,
    ee_position, link_transforms, full_safety_check,
    manipulability, min_obstacle_distance,
)


class WorkspaceVis(Node):

    def __init__(self):
        super().__init__('workspace_vis')

        self._q         = np.zeros(DOF)
        self._has_state = False
        self._is_safe   = True
        self._manip     = 1.0
        self._obs_dists = [1.0, 1.0, 1.0]

        self.create_subscription(JointState, '/joint_states', self._js_cb, 10)
        self.create_subscription(Bool,   '/puma/is_safe',       self._safe_cb, 10)
        self.create_subscription(Float64MultiArray, '/puma/manipulability', self._m_cb, 10)
        self.create_subscription(Float64MultiArray, '/puma/obstacle_distances', self._d_cb, 10)

        self._marker_pub = self.create_publisher(MarkerArray, '/puma/workspace_markers', 10)

        self.create_timer(0.1, self._publish)
        self.get_logger().info("Візуалізатор робочого простору запущено")

    def _js_cb(self, msg: JointState):
        d = dict(zip(msg.name, msg.position))
        for i, n in enumerate(JOINT_NAMES):
            self._q[i] = d.get(n, self._q[i])
        self._has_state = True

    def _safe_cb(self, msg: Bool):   self._is_safe = msg.data
    def _m_cb(self, msg: Float64MultiArray):
        if msg.data: self._manip = msg.data[0]
    def _d_cb(self, msg: Float64MultiArray):
        self._obs_dists = list(msg.data)

    def _publish(self):
        if not self._has_state:
            return

        markers = MarkerArray()
        now = self.get_clock().now().to_msg()

        # ── Робочий простір (прозорий ящик) ────────────────────────
        m = Marker()
        m.header.frame_id = 'world'; m.header.stamp = now
        m.ns = 'ws_box'; m.id = 0
        m.type = Marker.CUBE; m.action = Marker.ADD
        m.pose.position.x = float((WS_BOX_MIN[0] + WS_BOX_MAX[0]) / 2)
        m.pose.position.y = float((WS_BOX_MIN[1] + WS_BOX_MAX[1]) / 2)
        m.pose.position.z = float((WS_BOX_MIN[2] + WS_BOX_MAX[2]) / 2)
        m.pose.orientation.w = 1.0
        m.scale.x = float(WS_BOX_MAX[0] - WS_BOX_MIN[0])
        m.scale.y = float(WS_BOX_MAX[1] - WS_BOX_MIN[1])
        m.scale.z = float(WS_BOX_MAX[2] - WS_BOX_MIN[2])
        m.color.b = 0.8; m.color.a = 0.03
        markers.markers.append(m)

        # ── Перешкоди ───────────────────────────────────────────────
        obs_colors = [(0.9,0.1,0.1), (0.9,0.5,0.1), (0.5,0.1,0.9)]
        for i, (obs, col) in enumerate(zip(OBSTACLES, obs_colors)):
            # Тверда куля
            m = Marker()
            m.header.frame_id = 'world'; m.header.stamp = now
            m.ns = 'obstacles'; m.id = i
            m.type = Marker.SPHERE; m.action = Marker.ADD
            m.pose.position.x = float(obs[0])
            m.pose.position.y = float(obs[1])
            m.pose.position.z = float(obs[2])
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = float(obs[3] * 2)
            m.color.r, m.color.g, m.color.b = col
            m.color.a = 0.7
            markers.markers.append(m)

            # Зона небезпеки (більша, прозора)
            m2 = Marker()
            m2.header.frame_id = 'world'; m2.header.stamp = now
            m2.ns = 'danger_zones'; m2.id = i
            m2.type = Marker.SPHERE; m2.action = Marker.ADD
            m2.pose.position.x = float(obs[0])
            m2.pose.position.y = float(obs[1])
            m2.pose.position.z = float(obs[2])
            m2.pose.orientation.w = 1.0
            m2.scale.x = m2.scale.y = m2.scale.z = float(obs[3] * 2.5)
            m2.color.r, m2.color.g, m2.color.b = col
            m2.color.a = 0.12
            markers.markers.append(m2)

        # ── Ланцюг маніпулятора (лінія) ────────────────────────────
        transforms = link_transforms(self._q)
        line = Marker()
        line.header.frame_id = 'world'; line.header.stamp = now
        line.ns = 'robot_chain'; line.id = 0
        line.type = Marker.LINE_STRIP; line.action = Marker.ADD
        line.scale.x = 0.02
        line.color.r = 0.2; line.color.g = 0.6; line.color.b = 1.0; line.color.a = 0.8
        for T in transforms:
            p = Point()
            p.x, p.y, p.z = float(T[0,3]), float(T[1,3]), float(T[2,3])
            line.points.append(p)
        markers.markers.append(line)

        # ── Суглоби (сфери) ──────────────────────────────────────────
        for i, T in enumerate(transforms):
            jm = Marker()
            jm.header.frame_id = 'world'; jm.header.stamp = now
            jm.ns = 'joints'; jm.id = i
            jm.type = Marker.SPHERE; jm.action = Marker.ADD
            jm.pose.position.x = float(T[0,3])
            jm.pose.position.y = float(T[1,3])
            jm.pose.position.z = float(T[2,3])
            jm.pose.orientation.w = 1.0
            r = 0.045 if i == 0 else 0.025
            jm.scale.x = jm.scale.y = jm.scale.z = r
            jm.color.r = 0.3; jm.color.g = 0.7; jm.color.b = 1.0; jm.color.a = 1.0
            markers.markers.append(jm)

        # ── Кінцевий ефектор ────────────────────────────────────────
        pos = ee_position(self._q)
        ee = Marker()
        ee.header.frame_id = 'world'; ee.header.stamp = now
        ee.ns = 'end_effector'; ee.id = 0
        ee.type = Marker.SPHERE; ee.action = Marker.ADD
        ee.pose.position.x = float(pos[0])
        ee.pose.position.y = float(pos[1])
        ee.pose.position.z = float(pos[2])
        ee.pose.orientation.w = 1.0
        ee.scale.x = ee.scale.y = ee.scale.z = 0.055
        ee.color.a = 1.0
        if self._is_safe:
            ee.color.r = 0.1; ee.color.g = 1.0; ee.color.b = 0.1   # зелений
        else:
            ee.color.r = 1.0; ee.color.g = 0.1; ee.color.b = 0.1   # червоний
        markers.markers.append(ee)

        # ── Текстова мітка ЕЕ ────────────────────────────────────────
        txt = Marker()
        txt.header.frame_id = 'world'; txt.header.stamp = now
        txt.ns = 'ee_label'; txt.id = 0
        txt.type = Marker.TEXT_VIEW_FACING; txt.action = Marker.ADD
        txt.pose.position.x = float(pos[0])
        txt.pose.position.y = float(pos[1])
        txt.pose.position.z = float(pos[2]) + 0.08
        txt.pose.orientation.w = 1.0
        txt.scale.z = 0.04
        txt.color.r = txt.color.g = txt.color.b = 1.0; txt.color.a = 1.0
        status = "✓ SAFE" if self._is_safe else "✗ UNSAFE"
        txt.text = (
            f"{status}\n"
            f"EE: [{pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f}]\n"
            f"w={self._manip:.3f}"
        )
        markers.markers.append(txt)

        self._marker_pub.publish(markers)


def main(args=None):
    rclpy.init(args=args)
    node = WorkspaceVis()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
