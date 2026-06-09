"""
puma560_kinematics.py
Пряма/обернена кінематика Puma560, перевірки безпеки, генератор траєкторії.

Суглоби в цьому пакеті: joint1..joint6  (назви як у URDF puma560_description)
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np

# ═══════════════════════════════════════════════════════════
#  DH-параметри Puma560  [a, d, alpha]  (м / рад)
# ═══════════════════════════════════════════════════════════
DH = np.array([
    [0.0,     0.6718,   math.pi / 2],   # joint1
    [0.4318,  0.0,      0.0        ],   # joint2
    [-0.0203, 0.1503,   math.pi / 2],   # joint3
    [0.0,     0.4331,  -math.pi / 2],   # joint4
    [0.0,     0.0,      math.pi / 2],   # joint5
    [0.0,     0.0,      0.0        ],   # joint6
], dtype=float)

DOF = 6

# Обмеження суглобів (рад)
JOINT_MIN = np.array([-3.1416, -1.5708, -1.5708,
                      -3.1416, -1.5708, -3.1416], dtype=float)
JOINT_MAX = np.array([ 3.1416,  1.5708,  1.5708,
                       3.1416,  1.5708,  3.1416], dtype=float)

# Макс. швидкості (рад/с)
VEL_MAX = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0], dtype=float)

# Макс. прискорення (рад/с²)
ACC_MAX = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0], dtype=float)

JOINT_NAMES = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']

# ═══════════════════════════════════════════════════════════
#  Обмеження робочого простору
# ═══════════════════════════════════════════════════════════
WS_BOX_MIN = np.array([-0.9, -0.9,  0.0])
WS_BOX_MAX = np.array([ 0.9,  0.9,  1.5])
WS_MIN_RADIUS = 0.15   # м — занадто близько до осі
WS_MAX_RADIUS = 0.90   # м — за межею досяжності

# Сферичні перешкоди  [cx, cy, cz, r]
OBSTACLES: List[np.ndarray] = [
    np.array([ 0.50,  0.30, 0.87, 0.12]),   # перешкода A (червона)
    np.array([-0.40,  0.40, 0.75, 0.10]),   # перешкода B (помаранчева)
    np.array([ 0.00, -0.50, 1.05, 0.15]),   # перешкода C (фіолетова)
]


# ═══════════════════════════════════════════════════════════
#  Кінематика
# ═══════════════════════════════════════════════════════════
def _dh(a: float, d: float, alpha: float, theta: float) -> np.ndarray:
    """Матриця перетворення DH (4×4)."""
    ct, st = math.cos(theta), math.sin(theta)
    ca, sa = math.cos(alpha), math.sin(alpha)
    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0.0, sa,       ca,      d     ],
        [0.0, 0.0,      0.0,     1.0   ],
    ])


def forward_kinematics(q: np.ndarray) -> np.ndarray:
    """T_{0→6}  (4×4)."""
    T = np.eye(4)
    for i in range(DOF):
        T = T @ _dh(DH[i, 0], DH[i, 1], DH[i, 2], q[i])
    return T


def link_transforms(q: np.ndarray) -> List[np.ndarray]:
    """Матриці T_{0→i} для i = 0..6 (7 елементів)."""
    transforms = [np.eye(4)]
    T = np.eye(4)
    for i in range(DOF):
        T = T @ _dh(DH[i, 0], DH[i, 1], DH[i, 2], q[i])
        transforms.append(T.copy())
    return transforms


def ee_position(q: np.ndarray) -> np.ndarray:
    return forward_kinematics(q)[:3, 3]


def jacobian(q: np.ndarray) -> np.ndarray:
    """Геометричний якобіан 6×6."""
    J = np.zeros((6, DOF))
    transforms = link_transforms(q)
    p_e = transforms[-1][:3, 3]
    for i in range(DOF):
        z = transforms[i][:3, 2]
        p = transforms[i][:3, 3]
        J[:3, i] = np.cross(z, p_e - p)
        J[3:, i] = z
    return J


def manipulability(q: np.ndarray) -> float:
    J = jacobian(q)
    val = np.linalg.det(J @ J.T)
    return float(math.sqrt(max(0.0, val)))


# ═══════════════════════════════════════════════════════════
#  Перевірки безпеки
# ═══════════════════════════════════════════════════════════
@dataclass
class SafetyResult:
    safe:     bool = True
    reason:   str  = ""
    code:     str  = "OK"   # OK | WORKSPACE | JOINT_LIMIT | COLLISION | VELOCITY | SINGULARITY


def check_joint_limits(q: np.ndarray) -> SafetyResult:
    for i in range(DOF):
        if q[i] < JOINT_MIN[i] or q[i] > JOINT_MAX[i]:
            return SafetyResult(
                False,
                f"joint{i+1}={q[i]:.3f} поза [{JOINT_MIN[i]:.3f}, {JOINT_MAX[i]:.3f}]",
                "JOINT_LIMIT"
            )
    return SafetyResult()


def check_workspace(pos: np.ndarray) -> SafetyResult:
    # Прямокутна зона
    if not (np.all(pos >= WS_BOX_MIN) and np.all(pos <= WS_BOX_MAX)):
        return SafetyResult(
            False,
            f"ЕЕ [{pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f}] поза ящиком",
            "WORKSPACE"
        )
    # Радіальна зона
    r = math.hypot(pos[0], pos[1])
    if r < WS_MIN_RADIUS:
        return SafetyResult(False, f"Занадто близько до осі: r={r:.3f}", "WORKSPACE")
    if r > WS_MAX_RADIUS:
        return SafetyResult(False, f"Поза досяжністю: r={r:.3f}", "WORKSPACE")
    return SafetyResult()


def check_collision(q: np.ndarray) -> SafetyResult:
    transforms = link_transforms(q)
    for i, T in enumerate(transforms):
        p = T[:3, 3]
        for j, obs in enumerate(OBSTACLES):
            dist = np.linalg.norm(p - obs[:3]) - obs[3]
            if dist < 0:
                lbl = chr(65 + j)
                return SafetyResult(
                    False,
                    f"Ланка {i} зіткнення з перешкодою {lbl} (d={dist:.3f})",
                    "COLLISION"
                )
    return SafetyResult()


def check_velocity(dq: np.ndarray) -> SafetyResult:
    for i in range(DOF):
        if abs(dq[i]) > VEL_MAX[i]:
            return SafetyResult(
                False,
                f"joint{i+1} швидкість {dq[i]:.3f} > {VEL_MAX[i]:.3f}",
                "VELOCITY"
            )
    return SafetyResult()


def check_singularity(q: np.ndarray, threshold: float = 0.01) -> SafetyResult:
    w = manipulability(q)
    if w < threshold:
        return SafetyResult(False, f"Сингулярність w={w:.4f}", "SINGULARITY")
    return SafetyResult()


def full_safety_check(
    q: np.ndarray,
    dq: Optional[np.ndarray] = None,
) -> SafetyResult:
    """
    Повна перевірка за алгоритмом із блок-схеми:
    1. Робочий простір ЕЕ
    2. Обмеження суглобів
    3. Зіткнення
    4. Швидкість
    5. Сингулярність
    """
    # 1
    res = check_workspace(ee_position(q))
    if not res.safe:
        return res
    # 2
    res = check_joint_limits(q)
    if not res.safe:
        return res
    # 3
    res = check_collision(q)
    if not res.safe:
        return res
    # 4
    if dq is not None:
        res = check_velocity(dq)
        if not res.safe:
            return res
    # 5
    res = check_singularity(q)
    if not res.safe:
        return res
    return SafetyResult()


def min_obstacle_distance(q: np.ndarray) -> float:
    transforms = link_transforms(q)
    min_d = float('inf')
    for T in transforms:
        p = T[:3, 3]
        for obs in OBSTACLES:
            d = float(np.linalg.norm(p - obs[:3])) - obs[3]
            min_d = min(min_d, d)
    return min_d


# ═══════════════════════════════════════════════════════════
#  Генератор траєкторії — квінтичний поліном
# ═══════════════════════════════════════════════════════════
@dataclass
class TrajPoint:
    q:    np.ndarray
    dq:   np.ndarray
    ddq:  np.ndarray
    time: float


def quintic_trajectory(
    q_start: np.ndarray,
    q_goal:  np.ndarray,
    total_t: float,
    dt:      float = 0.05,
) -> List[TrajPoint]:
    """
    Поліном 5-го степеня:
    q(0)=qs, dq(0)=0, ddq(0)=0
    q(T)=qg, dq(T)=0, ddq(T)=0
    """
    T3, T4, T5 = total_t**3, total_t**4, total_t**5
    dq = q_goal - q_start
    a3 =  10.0 * dq / T3
    a4 = -15.0 * dq / T4
    a5 =   6.0 * dq / T5

    points: List[TrajPoint] = []
    t = 0.0
    while t <= total_t + 1e-9:
        t2 = t * t
        q_   = q_start + a3*t**3  + a4*t**4  + a5*t**5
        dq_  =           3*a3*t2  + 4*a4*t**3 + 5*a5*t**4
        ddq_ =           6*a3*t   +12*a4*t2   +20*a5*t**3
        points.append(TrajPoint(q_, dq_, ddq_, t))
        t += dt
    return points


def estimate_travel_time(q_start: np.ndarray, q_goal: np.ndarray) -> float:
    dq = np.abs(q_goal - q_start)
    t_min = float(np.max(dq / (VEL_MAX * 0.6)))
    return max(t_min, 1.0)


def scale_trajectory_time(
    points: List[TrajPoint],
    safety: float = 0.85,
) -> List[TrajPoint]:
    """Масштабування якщо швидкість перевищена."""
    scale = 1.0
    for pt in points:
        ratios = np.abs(pt.dq) / (VEL_MAX * safety)
        scale = max(scale, float(np.max(ratios)))
    if scale <= 1.0:
        return points
    return [
        TrajPoint(pt.q, pt.dq / scale, pt.ddq / scale**2, pt.time * scale)
        for pt in points
    ]
