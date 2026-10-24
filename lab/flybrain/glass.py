"""Шейдеры окна: живой фон и матовое стекло панелей.

Подход взят у AmbientGlass (D:\\VlcAmbient): фон целиком процедурный — ни
картинок, ни видео, шейдер считает его заново каждый кадр, — а панели не
закрашены цветом, а показывают размытый фон под собой с каймой по краю.
Там этим управляла музыка: бас, верха, удар. Здесь — мозг: цвет берётся у
самой активной области, вспышка идёт на всплеске спайков, темп задаёт общая
частота.

Цепочка кадра:
  1. сцена (фон + облако нейронов) рисуется в текстуру;
  2. она размывается в две проходки, по горизонтали и вертикали;
  3. композит выводит сцену на экран, а внутри прямоугольников панелей
     подмешивает размытую версию с подкраской и каймой.

Панели — это окна ImGui: их положение известно с прошлого кадра, и этого
достаточно, задержка в один кадр незаметна.
"""

from __future__ import annotations

MAX_PANELS = 8

QUAD_VS = """
#version 330
in vec2 in_pos;
out vec2 uv;
void main() {
    uv = in_pos * 0.5 + 0.5;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

# --------------------------------------------------------------- живой фон

BACKGROUND_FS = """
#version 330
in vec2 uv;
out vec4 frag;

uniform vec2  uSize;
uniform float uTime;
uniform vec3  uAccent;    // цвет самой активной области
uniform vec3  uAccent2;   // цвет второй по активности
uniform float uPulse;     // 0..1, всплеск спайков
uniform float uLevel;     // 0..1, общая активность
uniform vec2  uCentre;    // центр свечения в пикселях

float hash(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

float vnoise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    // квинтика: на границах клеток не остаётся изломов, видимых в тонких линиях
    vec2 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    float a = hash(i), b = hash(i + vec2(1, 0));
    float c = hash(i + vec2(0, 1)), d = hash(i + vec2(1, 1));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

float fbm(vec2 p) {
    float s = 0.0, a = 0.5;
    for (int k = 0; k < 4; k++) {
        s += a * vnoise(p);
        p = mat2(1.6, 1.2, -1.2, 1.6) * p + 17.3;
        a *= 0.5;
    }
    return s / 0.9375;
}

// 1 на середине шума, 0 на краях: светлая нить там, где шум переходит 0.5
float ridge(float v) { return clamp(1.0 - abs(2.0 * v - 1.0), 0.0, 1.0); }

// Нить не тоньше примерно полутора пикселей: более тонкая рассыпалась бы в
// мерцающее зерно. Расширенная настолько же тускнеет, так что общая яркость
// сохраняется, как у настоящей линии, накрывающей часть пикселя.
float line_(float v, float sharp) {
    float fw = max(fwidth(v), 1e-5);
    float s = min(sharp, 0.45 / fw);
    return pow(ridge(v), s) * (s / sharp);
}

// Снос области шума, который никуда не убегает: сначала по направлению, потом
// медленно загибается по окружности, так что и через часы координаты рядом с
// нулём и точность float не теряется.
vec2 drift(float t, vec2 v) {
    float r = 60.0;
    float w = length(v) / r;
    float a = w * t;
    vec2 n = normalize(v + 1e-6);
    vec2 perp = vec2(-n.y, n.x);
    return r * (n * sin(a) + perp * (1.0 - cos(a)));
}

void main() {
    vec2 px = uv * uSize;
    vec2 d = px - uCentre;
    float reach = length(uSize) * 0.65;
    float r = length(d) / reach;
    float ang = atan(d.y, d.x);

    // База почти чёрная. Весь свет должен быть собран у центра, иначе
    // облако нейронов, которое рисуется аддитивно, тонет в фоне.
    vec3 col = vec3(0.007, 0.009, 0.014);

    // Ленты обрамляют кадр, а не заливают его: у центра их нет совсем,
    // к краям они проявляются и остаются приглушёнными.
    vec2 p = px / uSize.y * 2.2;
    float t = uTime * 0.04;
    float rib = 0.0;
    rib += line_(fbm(p * 0.9 + drift(t, vec2(0.6, 0.2))), 11.0) * 0.6;
    rib += line_(fbm(p * 1.7 + drift(t * 1.3, vec2(-0.45, 0.35))), 17.0) * 0.3;
    rib *= smoothstep(0.42, 1.15, r) * smoothstep(1.9, 1.25, r);
    col += mix(uAccent, uAccent2, 0.5 + 0.5 * sin(ang * 2.0 + uTime * 0.15))
           * rib * (0.045 + 0.10 * uLevel);

    // Лучи идут из центра наружу и гаснут, не доходя до краёв.
    float rays = fbm(vec2(ang * 3.1, r * 1.6 - uTime * 0.06));
    rays = pow(clamp(rays, 0.0, 1.0), 5.0);
    rays *= smoothstep(0.95, 0.12, r) * smoothstep(0.05, 0.28, r);
    col += uAccent * rays * (0.07 + 0.42 * uLevel + 0.3 * uPulse);

    // Масса света за мозгом: тугая, именно она вспыхивает на всплеске.
    col += uAccent * exp(-r * r * 7.0) * (0.045 + 0.40 * uLevel + 0.75 * uPulse);
    col += uAccent2 * exp(-r * r * 20.0) * (0.03 + 0.30 * uPulse);

    // Виньетка: углы уходят в чёрное, чтобы панели лежали на спокойном фоне.
    col *= mix(0.25, 1.0, smoothstep(1.45, 0.3, r));

    // Дизер: без него на тёмных градиентах видны ступеньки 8-битного вывода.
    float dith = (hash(px + fract(uTime)) - 0.5) / 255.0;
    frag = vec4(col + dith, 1.0);
}
"""

# --------------------------------------------------------------- размытие

BLUR_FS = """
#version 330
in vec2 uv;
out vec4 frag;
uniform sampler2D uTex;
uniform vec2 uDir;       // (1/w, 0) или (0, 1/h), умноженное на радиус

void main() {
    // Девять отсчётов с весами Гаусса: достаточно для матового стекла и
    // заметно дешевле честного большого ядра.
    float w[5] = float[](0.2270270270, 0.1945945946, 0.1216216216,
                         0.0540540541, 0.0162162162);
    vec3 s = texture(uTex, uv).rgb * w[0];
    for (int i = 1; i < 5; i++) {
        s += texture(uTex, uv + uDir * float(i)).rgb * w[i];
        s += texture(uTex, uv - uDir * float(i)).rgb * w[i];
    }
    frag = vec4(s, 1.0);
}
"""

# --------------------------------------------------------------- композит

COMPOSITE_FS = """
#version 330
in vec2 uv;
out vec4 frag;

uniform sampler2D uScene;
uniform sampler2D uBlur;
uniform vec2  uSize;
uniform int   uPanelCount;
uniform vec4  uPanels[%d];   // l, t, r, b в пикселях, начало слева сверху
uniform float uRadius;
uniform float uTint;         // доля подмешанного белого
uniform float uFrost;        // 0 — прозрачное, 1 — матовое
uniform vec3  uAccent;

// Расстояние до прямоугольника со скруглёнными углами, отрицательное внутри.
float rrect(vec2 p, vec2 half_, float r) {
    vec2 q = abs(p) - half_ + r;
    return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - r;
}

void main() {
    vec2 px = vec2(uv.x, 1.0 - uv.y) * uSize;
    vec3 scene = texture(uScene, uv).rgb;
    vec3 blur  = texture(uBlur, uv).rgb;
    vec3 col = scene;

    for (int i = 0; i < uPanelCount; i++) {
        vec4 p = uPanels[i];
        vec2 c = vec2((p.x + p.z) * 0.5, (p.y + p.w) * 0.5);
        vec2 h = vec2((p.z - p.x) * 0.5, (p.w - p.y) * 0.5);
        if (h.x <= 1.0 || h.y <= 1.0) continue;
        float d = rrect(px - c, h, uRadius);

        float inside = 1.0 - smoothstep(-1.0, 1.0, d);
        if (inside <= 0.001) continue;

        // Стекло: размытый фон, подложка холодного тёмного тона, чтобы
        // текст читался, и немного белого для материальности.
        vec3 glass = mix(scene, blur, uFrost);
        glass = mix(glass, vec3(0.055, 0.070, 0.098), 0.62);
        glass = mix(glass, vec3(1.0), uTint);
        // Лёгкий градиент сверху вниз: панель перестаёт быть плоской.
        glass *= 1.0 - 0.18 * clamp((px.y - p.y) / max(h.y * 2.0, 1.0), 0.0, 1.0);
        col = mix(col, glass, inside);

        // Кайма: узкая светлая полоса по самому краю, с подмесом цвета мозга.
        float rim = smoothstep(2.5, 0.0, abs(d + 1.2)) * inside;
        col += (vec3(1.0) * 0.16 + uAccent * 0.30) * rim;

        // Блик сверху: тонкая дуга вдоль верхней кромки.
        float top = smoothstep(0.0, 18.0, px.y - p.y) * (1.0 - smoothstep(18.0, 70.0, px.y - p.y));
        col += vec3(1.0) * 0.045 * top * inside;
    }

    frag = vec4(col, 1.0);
}
""" % MAX_PANELS

# ------------------------------------------------------- облако нейронов

POINTS_VS = """
#version 330
in vec3 in_pos;
in float in_reg;
in float in_act;
uniform mat4 uMVP;
uniform float uPointScale;
uniform float uMode;      // 0 — активность, 1 — области
uniform float uFocus;
uniform vec3 uRegCol[64];
out vec4 vCol;

vec3 magma(float t) {
    t = clamp(t, 0.0, 1.0);
    const vec3 c0 = vec3(0.063, 0.071, 0.102), c1 = vec3(0.231, 0.059, 0.439);
    const vec3 c2 = vec3(0.549, 0.161, 0.506), c3 = vec3(0.867, 0.288, 0.408);
    const vec3 c4 = vec3(0.984, 0.533, 0.380), c5 = vec3(0.988, 0.992, 0.749);
    float s = t * 5.0; int i = int(floor(s)); float f = fract(s);
    if (i == 0) return mix(c0, c1, f);
    if (i == 1) return mix(c1, c2, f);
    if (i == 2) return mix(c2, c3, f);
    if (i == 3) return mix(c3, c4, f);
    return mix(c4, c5, f);
}

void main() {
    gl_Position = uMVP * vec4(in_pos, 1.0);
    float depth = clamp(1.0 - gl_Position.z / gl_Position.w * 0.5, 0.35, 1.0);
    int ri = int(in_reg + 0.5);
    vec3 rc = uRegCol[ri];
    float focus = (uFocus < -0.5 || abs(in_reg - uFocus) < 0.5) ? 1.0 : 0.07;

    if (in_act < 0.004) {
        // Фоновое облако даёт силуэт мозга: оно должно быть отчётливо
        // видно на тёмном фоне, иначе остаётся невнятное пятно.
        vec3 c = (uMode > 0.5) ? rc * 1.05 : vec3(0.46, 0.60, 0.84);
        float a = (uMode > 0.5) ? 0.42 : 0.20;
        vCol = vec4(c * depth, a * focus);
        gl_PointSize = uPointScale;
    } else {
        vec3 c = (uMode > 0.5) ? rc : magma(0.18 + 0.82 * in_act);
        float a = min(1.0, 0.16 + 1.3 * in_act);
        vCol = vec4(c, a * max(focus, 0.12));
        gl_PointSize = uPointScale * (1.4 + 6.5 * in_act);
    }
}
"""

POINTS_FS = """
#version 330
in vec4 vCol;
out vec4 frag;
void main() {
    vec2 d = gl_PointCoord - vec2(0.5);
    float r = dot(d, d);
    if (r > 0.25) discard;
    float soft = exp(-r * 7.0);
    frag = vec4(vCol.rgb * soft, vCol.a * soft);
}
"""
