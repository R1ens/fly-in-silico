"""Щелчок по сцене MuJoCo — в координаты модели, и обратно.

Песочница обещает, что источник можно взять в сцене и перенести, а его
координаты будут координатами модели. Обещание проверяемое, и проверять
его надо не на глаз: окно не знает ни охвата камеры, ни того, куда она
смотрит, — камера слежения считает это по габариту мухи уже в работнике.

Поэтому работник кладёт рядом с каждым кадром позу камеры, которой этот
кадр снят (``MjvGLCamera`` разрешённой сцены: положение, направление,
верх и усечённая пирамида), а здесь она превращается в две функции:

* :func:`project` — куда в кадре попадёт точка модели;
* :func:`unproject` — какой точке плоскости ``z = z0`` соответствует
  пиксель кадра.

Ни numpy сверх того, ни torch, ни mujoco здесь не нужны: окно не тянет
ни одного тяжёлого модуля ради этого.

Знак оси «вправо» выведен не из рассуждения о том, какой системой
координат пользуется OpenGL, а замером: ``probe/scene_pick.py`` ставит
маркер в известное место, снимает кадр и ищет его цветные пиксели. При
неверном знаке маркер оказывается зеркально отражённым относительно
середины кадра, и проба это видит.
"""

from __future__ import annotations

import numpy as np


def basis(cam):
    """Орты камеры: вперёд, вправо, вверх — нормированные и ортогональные."""
    f = np.asarray(cam["forward"], dtype=float)
    u = np.asarray(cam["up"], dtype=float)
    f = f / max(float(np.linalg.norm(f)), 1e-12)
    # Верх из сцены не обязан быть строго перпендикулярен направлению:
    # отнимаем проекцию, иначе база получается скошенной и ошибка растёт
    # к краям кадра.
    u = u - f * float(np.dot(u, f))
    u = u / max(float(np.linalg.norm(u)), 1e-12)
    r = np.cross(f, u)
    r = r / max(float(np.linalg.norm(r)), 1e-12)
    return f, r, u


def _frustum(cam):
    """Границы усечённой пирамиды на ближней плоскости: слева, справа, снизу, сверху."""
    near = float(cam["near"])
    bottom, top = float(cam["bottom"]), float(cam["top"])
    width = float(cam.get("width") or 0.0)
    if width <= 0.0:
        # Старые сборки MuJoCo ширину не отдают: берём её из высоты и
        # отношения сторон кадра — это то же самое для симметричной
        # пирамиды, которую строит `mjv_updateScene`.
        w, h = float(cam["w"]), float(cam["h"])
        width = (top - bottom) * (w / max(h, 1e-9))
    centre = float(cam.get("centre") or 0.0)
    return centre - width / 2.0, centre + width / 2.0, bottom, top, near


def ray(cam, px, py):
    """Луч из камеры через пиксель кадра: (начало, направление).

    Пиксель считается от левого верхнего угла — так лежит кадр, который
    отдаёт ``mujoco.Renderer``.
    """
    f, r, u = basis(cam)
    left, right, bottom, top, near = _frustum(cam)
    w, h = float(cam["w"]), float(cam["h"])
    sx = (float(px) + 0.5) / max(w, 1e-9)
    sy = (float(py) + 0.5) / max(h, 1e-9)
    xn = left + sx * (right - left)
    yn = top - sy * (top - bottom)
    pos = np.asarray(cam["pos"], dtype=float)
    if cam.get("ortho"):
        # Ортографической камеры у нас нет, но молча считать её
        # перспективной значит дать неверные координаты без всякого
        # признака. Луч параллельный, начало смещено.
        return pos + r * xn + u * yn, f
    d = f * near + r * xn + u * yn
    return pos, d / max(float(np.linalg.norm(d)), 1e-12)


def unproject(cam, px, py, z=0.0):
    """Пиксель кадра — в точку плоскости ``z``. ``None``, если луч не пересекает её.

    Возвращает ``(x, y, z)`` в координатах модели, то есть в тех же
    миллиметрах, в которых заданы положения источников.
    """
    pos, d = ray(cam, px, py)
    dz = float(d[2])
    if abs(dz) < 1e-9:
        return None                      # луч параллелен плоскости
    t = (float(z) - float(pos[2])) / dz
    if t <= 0.0:
        return None                      # плоскость за спиной камеры
    p = pos + d * t
    return float(p[0]), float(p[1]), float(z)


def project(cam, point):
    """Точка модели — в пиксель кадра. ``None``, если она за камерой.

    Возвращает ``(px, py)`` от левого верхнего угла, дробными долями
    пикселя; значения за пределами кадра не отбрасываются — вызывающий
    сам решает, рисовать ли метку у края.
    """
    f, r, u = basis(cam)
    left, right, bottom, top, near = _frustum(cam)
    w, h = float(cam["w"]), float(cam["h"])
    v = np.asarray(point, dtype=float) - np.asarray(cam["pos"], dtype=float)
    depth = float(np.dot(v, f))
    if depth <= 1e-9:
        # Точка за камерой. Для перспективы это было отсечено и раньше, а
        # для ортографической — нет: точка позади проецировалась как
        # видимая, и `pick_slot` мог «взять» слот, стоящий за спиной.
        return None
    if cam.get("ortho"):
        xn, yn = float(np.dot(v, r)), float(np.dot(v, u))
    else:
        k = near / depth
        xn, yn = float(np.dot(v, r)) * k, float(np.dot(v, u)) * k
    sx = (xn - left) / max(right - left, 1e-12)
    sy = (top - yn) / max(top - bottom, 1e-12)
    return sx * w - 0.5, sy * h - 0.5


def pick_slot(cam, slots, px, py, radius_px=28.0):
    """Какой слот источника взят щелчком. ``-1``, если ни один.

    Берётся ближайший в кадре, а не ближайший в мире: пользователь
    целится в картинку. Радиус в пикселях кадра работника, поэтому
    вызывающий пересчитывает щелчок во вьюпорте в пиксели кадра.
    """
    best, best_d = -1, float(radius_px)
    for i, p in enumerate(slots):
        at = project(cam, p)
        if at is None:
            continue
        d = ((at[0] - px) ** 2 + (at[1] - py) ** 2) ** 0.5
        if d < best_d:
            best, best_d = i, d
    return best
