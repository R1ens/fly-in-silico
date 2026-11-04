r"""Съёмка для поста: кадры и ролик из НАСТОЯЩЕГО окна.

Ничего не рисуется отдельно «для красоты». Всё, что попадает в кадр, —
то же приложение, тот же работник, тот же коннектом и то же тело
MuJoCo, что считают научные прогоны. Снимается чтением накопителя
кадра, а не захватом экрана: захват swapchain из-под автоматики отдаёт
чёрное (`desk/shell.py`, повадка 2).

Ролик идёт в НАСТОЯЩЕЙ скорости, и это главное его свойство. Полный
контур считается со скоростью около 0,075× реального времени, то есть
в живом окне муха движется в тринадцать раз медленнее настоящей. Если
записывать то, что видно, получится замедление; если писать каждый
кадр окна и играть подряд — ускорение, и непонятно во сколько раз.

Поэтому кадры привязаны к МОДЕЛЬНОМУ времени: один кадр видео — ровно
один шаг управления, 20 мс модельного. Рендер офлайновый, по стене он
идёт сколько нужно, а готовая запись играет со скоростью мухи.
50 кадров в секунду здесь не выбор, а следствие: столько состояний в
модельной секунде и есть.

    cd D:\fly-brain\lab
    set PYTHONIOENCODING=utf-8
    ..\.venv\Scripts\python.exe probe\promo.py --кадры
    ..\.venv\Scripts\python.exe probe\promo.py --видео
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("FLYBRAIN_OUT", str(Path(__file__).parent / "_promo_out"))

import probe._isolate  # noqa: E402,F401

import numpy as np  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "promo"
W, H = 1920, 1080
МОДЕЛЬ_НА_КАДР = 20.0        # мс модельного времени на кадр видео
ВИДЕО_FPS = 50               # = 1000 / МОДЕЛЬ_НА_КАДР
ШРИФТЫ = {
    True: ("C:/Windows/Fonts/corbelb.ttf", "C:/Windows/Fonts/segoeuib.ttf"),
    False: ("C:/Windows/Fonts/corbel.ttf", "C:/Windows/Fonts/segoeui.ttf"),
}
rep = {"кадры": [], "сцены": [], "замеры": {}}


def кадр_в_массив(sh):
    """Тот же путь, что у `Shell.shot`, но без записи на диск."""
    a = np.clip(sh.surf.acc.to_numpy(), 0.0, 1.0)
    return (a.transpose(1, 0, 2)[::-1] * 255).astype(np.uint8)


def шрифт(размер, жирный=False):
    from PIL import ImageFont

    for путь in ШРИФТЫ[bool(жирный)]:
        if Path(путь).exists():
            return ImageFont.truetype(путь, размер)
    return ImageFont.load_default()


def подписать(кадр, строка, сноска=""):
    """Подпись снизу, теми же гарнитурами, что у самого окна."""
    from PIL import Image, ImageDraw

    им = Image.fromarray(кадр)
    d = ImageDraw.Draw(им, "RGBA")
    низ = им.height
    # Полоса НЕПРОЗРАЧНАЯ, и это не вкусовщина. При 198 из 255 сквозь
    # неё просвечивала собственная строка состояния окна, а сноска
    # ложилась ровно на неё: две разные подписи мелким кеглем одна
    # поверх другой, и читать нельзя ни ту ни другую. Полоса закрывает
    # строку целиком, а то, что в ней написано, сказано в сноске.
    d.rectangle([0, низ - 152, им.width, низ], fill=(8, 10, 13, 255))
    d.line([0, низ - 152, им.width, низ - 152], fill=(38, 43, 48, 255))
    d.text((56, низ - 120), строка, font=шрифт(40, True),
           fill=(232, 164, 73, 255))
    if сноска:
        d.text((56, низ - 62), сноска, font=шрифт(25),
               fill=(155, 167, 180, 255))
    return np.asarray(им)


def ждать(app, условие, сек, sh, t0):
    край = time.perf_counter() + сек
    while time.perf_counter() < край:
        sh.t = time.perf_counter() - t0
        app.frame()
        if условие():
            return True
    return False


def поднять(размер=(W, H)):
    from flybrain.desk.app import App
    from flybrain.desk.shell import Offscreen
    from flybrain.ui.session import Session

    sh = Offscreen(размер)
    ses = Session()
    app = App(sh, ses)
    t0 = time.perf_counter()
    while not app.st.worker.ready and time.perf_counter() - t0 < 300:
        app.frame()
    if not app.st.worker.ready:
        print("работник не поднялся")
        return None
    print("  работник готов за %.1f с" % (time.perf_counter() - t0),
          flush=True)
    return sh, ses, app, t0


def для_активности(app):
    """Настройки показа, при которых волну видно.

    По умолчанию невыделенное гаснет до 0,07, а точка рисуется в
    натуральную величину — на снимке это почти чёрный кадр с редкими
    искрами. Для съёмки силуэт поднимается до 0,30, а точка до 1,6:
    тогда видно И волну, И мозг, по которому она идёт. Это настройки
    ПОКАЗА, те же ползунки, что в рейке слева; на счёт они не влияют
    никак.

    Решает же дело `only_active` — кнопка «только сработавшие» в той же
    рейке. Работает примерно каждый восьмой нейрон, и в облаке из ста
    тридцати восьми тысяч его не разглядеть: фон своей плотностью
    забивает всё. Убрать молчащих — значит показать ровно тех, кто
    сейчас считается, а не меньше и не больше.
    """
    app.st.brain.dim = 0.30
    app.st.brain.point = 1.6
    app.st.brain.only_active = True


def снять_фокус(app):
    """Убрать выделение области ПОСЛЕ запуска, иначе кадр почти чёрный.

    Выделение области в окне делает две разные вещи: задаёт, кого
    стимулировать, и гасит весь остальной мозг. Первое нужно, второе
    на снимке губительно — за пределами антеннальной дольки остаётся
    треть яркости, и волна, которая как раз туда и уходит, не видна.

    Запрос окно забирает один раз, в момент пуска (`app._stim_query`),
    поэтому снятое после пуска выделение на счёт не влияет: стимул
    остаётся на той же популяции.
    """
    app.st.brain.region = -1


def выбрать_область(app, имя):
    """Что стимулировать. Окно берёт запрос из ВЫДЕЛЕНИЯ на карте.

    `app._stim_query` читает `sel_label` только вместе с `sel_idx`, а
    индексы приходят от работника отдельным запросом. Для съёмки проще
    и честнее выбрать область: тогда запрос станет `region:<имя>`,
    ровно как по щелчку мышью.
    """
    имена = list(app.brain.region_names)
    app.st.brain.region = имена.index(имя) if имя in имена else 0
    return имя


def песочница(app, ses, sh, t0, со_зрением=True):
    """Запустить сеанс песочницы и дождаться первого кадра тела."""
    st = app.st
    st.ws.switch("sandbox")
    sb = st.sb
    sb.t_run = 0.0
    sb.keep_s = 10.0
    sb.vision = bool(со_зрением)
    app.start_run()
    ok = ждать(app, lambda: ses.body_frame is not None, 300, sh, t0)
    print("  кадр тела: %s" % ("есть" if ok else "НЕ ДОЖДАЛИСЬ"), flush=True)
    cam = st.camera("sandbox")
    cam.view, cam.dist = "side", 1.1
    return sb, ok


def _поставить_слева(app, ses):
    """Положить шар вперёд-влево ОТ САМОЙ МУХИ и отправить, не ожидая.

    Координаты считаются в осях мухи и только потом переводятся в
    мировые: муха всё время поворачивает, и если взять «слева» в осях
    арены, через пару секунд это окажется у неё за спиной. На этом уже
    сорвался опыт про причинность, и здесь ошибка та же.
    """
    from flybrain.desk import ws_sandbox as X

    sb = app.st.sb
    ex = dict(ses.extra or {})
    fx = float(ex.get("x", 0.0) or 0.0)
    fy = float(ex.get("y", 0.0) or 0.0)
    th = float(ex.get("theta", 0.0) or 0.0)
    o = sb.objects[0]
    # Вперёд 14 мм, влево 10. Шар радиусом 4 мм отсюда виден чётким
    # куполом на фоне неба; при радиусе 8 с девяти миллиметров он
    # закрывал почти всё поле левого глаза, и смотреть было не на что.
    o.x = round(fx + 14.0 * np.cos(th) - 10.0 * np.sin(th), 3)
    o.y = round(fy + 14.0 * np.sin(th) + 10.0 * np.cos(th), 3)
    o.z, o.radius, o.grey, o.on = 2.0, 4.0, 0.0, True
    return X._send_objects(app, "предмет слева")


def предмет_слева(app, ses, sh, t0):
    """Поставить тёмный шар слева от мухи и дождаться применения."""
    cid = _поставить_слева(app, ses)
    ждать(app, lambda: cid not in app.st.sb.pending, 60, sh, t0)
    return cid


def переставлять_слева(app, ses, каждые=25):
    """Шаг сцены: держать шар слева, переставляя его на ходу.

    За семь секунд модельного времени муха уходит от неподвижного шара
    настолько, что в кадре глаз остаётся один пол, и подпись про шар
    становится неправдой. Поэтому шар переставляется — той же живой
    правкой сеанса, какой это делается мышью, и ответ работника тут не
    ждут: кадры выпускаются по модельному времени, а подтверждение
    приходит своим чередом через почту сеанса.

    `каждые` в кадрах: 25 кадров — это полсекунды модельного времени.
    Чаще нет смысла, работник всё равно применяет правку на границе
    шага управления.
    """
    счёт = {"n": 0}

    def шаг():
        if счёт["n"] % каждые == 0:
            _поставить_слева(app, ses)
        счёт["n"] += 1

    return шаг


# ------------------------------------------------------------------ кадры


def кадры():
    из = поднять()
    if из is None:
        return
    sh, ses, app, t0 = из
    st = app.st
    OUT.mkdir(parents=True, exist_ok=True)

    def снять(имя, note=""):
        app.frame()
        app.frame()
        sh.shot(OUT / имя)
        rep["кадры"].append({"файл": имя, "что": note,
                             "размер": [sh.w, sh.h]})
        print("  снят %s — %s" % (имя, note), flush=True)

    # 1. Мозг: облако по нейропилям. Без поворота — вид по умолчанию,
    # две доли рядом, и есть самый красивый.
    st.ws.switch("brain")
    st.brain.mode = "anatomy"
    st.ws.dock().maximized = True
    for _ in range(12):
        app.frame()
    снять("01-мозг.png", "138 639 сом, цвет по 36 нейропилям")

    # 2. Мозг в работе. Ждать не кадров, а МОДЕЛЬНОГО времени: на 0,22 с
    # волна ещё не вышла из антеннальной дольки.
    st.ws.dock().maximized = False
    st.ws.switch("experiment")
    st.t_run = 3.0
    st.rate_hz = 200.0
    выбрать_область(app, "AL")
    st.ws.switch("brain")
    st.brain.mode = "activity"
    для_активности(app)
    st.ws.dock().maximized = True
    app.start_run()
    снять_фокус(app)
    ждать(app, lambda: ses.t_ms > 1800.0, 300, sh, t0)
    for _ in range(12):
        app.frame()
    # Чем кадр тёмен, видно только по числам: шкала magma строится по
    # ПИКУ, и если пик вчетверо выше, чем у работающего большинства,
    # большинство ложится в нижний, почти чёрный край палитры.
    a = ses.act
    живые = a[a > 0.5]
    print("  активность: шкала %.0f Гц, ненулевых %d, "
          "медиана %.1f, 90%% %.1f, 99%% %.1f, пик %.1f"
          % (ses.scale, живые.size,
             float(np.median(живые)) if живые.size else 0.0,
             float(np.percentile(живые, 90)) if живые.size else 0.0,
             float(np.percentile(живые, 99)) if живые.size else 0.0,
             float(a.max())), flush=True)
    снять("02-активность.png", "волна от антеннальной дольки, цвет по частоте")
    ждать(app, lambda: not ses.busy, 180, sh, t0)
    st.ws.dock().maximized = False

    # 3. Тело и 4. глаза — один сеанс песочницы.
    sb, ok = песочница(app, ses, sh, t0)
    if ok:
        ждать(app, lambda: False, 6.0, sh, t0)
        снять("03-тело.png", "муха в MuJoCo, справа живые показания")
        предмет_слева(app, ses, sh, t0)
        ждать(app, lambda: False, 3.0, sh, t0)
        sb.eye_view = True
        ждать(app, lambda: False, 2.0, sh, t0)
        снять("04-глаза.png", "кадр обоих глаз и показания 721 омматидия")
        app.cancel_run()
        ждать(app, lambda: not ses.busy, 180, sh, t0)
    ses.close()
    (OUT / "promo.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")


# ------------------------------------------------------------------ видео


def видео():
    import imageio.v2 as iio

    из = поднять()
    if из is None:
        return
    sh, ses, app, t0 = из
    st = app.st
    OUT.mkdir(parents=True, exist_ok=True)
    путь = OUT / "fly-in-silico.mp4"
    писарь = iio.get_writer(str(путь), fps=ВИДЕО_FPS, codec="libx264",
                            quality=9, macro_block_size=1,
                            ffmpeg_params=["-pix_fmt", "yuv420p"])
    счёт = {"кадров": 0}

    # Шаг сглаживания окна — МОДЕЛЬНЫЙ, а не настенный. След активности
    # гаснет с постоянной 50 мс, и если кормить его настенным временем,
    # он погаснет в тринадцать раз быстрее, чем увидел бы человек.
    было_begin = sh.begin

    def begin():
        ctx = было_begin()
        sh.dt = МОДЕЛЬ_НА_КАДР / 1000.0
        return ctx

    sh.begin = begin

    def писать(строка, сноска=""):
        app.frame()
        писарь.append_data(подписать(кадр_в_массив(sh), строка, сноска))
        счёт["кадров"] += 1

    def сцена(имя, кадров, строка, сноска="", шаг=None, по_модели=True):
        """Кадр выпускается, когда модель дошла до его времени."""
        было = счёт["кадров"]
        база = ses.t_ms
        опоздали = 0
        for k in range(кадров):
            if шаг is not None:
                шаг()
            if по_модели:
                цель = база + (k + 1) * МОДЕЛЬ_НА_КАДР
                край = time.perf_counter() + 30.0
                while ses.t_ms < цель - 1e-9:
                    if time.perf_counter() > край:
                        опоздали += 1
                        break
                    # Почта разбирается БЕЗ затухания: гасить след надо
                    # раз на кадр видео, а не раз на оборот ожидания.
                    ses.pump(0.0)
                    time.sleep(0.0005)
            писать(строка, сноска)
        rep["сцены"].append({
            "сцена": имя, "кадров": кадров,
            "с": round(кадров / ВИДЕО_FPS, 1), "от кадра": было,
            "модельного, с": (round(кадров * МОДЕЛЬ_НА_КАДР / 1000.0, 2)
                              if по_модели else None),
            "кадров не дождались": опоздали})
        print("  сцена «%s»: %d кадров%s"
              % (имя, кадров,
                 (", не дождались %d" % опоздали) if опоздали else ""),
              flush=True)

    try:
        bv = app.brain

        def крутить():
            bv.yaw += 0.0035
            bv._fit_key = None

        # --- 1. мозг: облёт камеры, модель не считается --------------
        st.ws.switch("brain")
        st.brain.mode = "anatomy"
        st.ws.dock().maximized = True
        for _ in range(12):
            app.frame()
        сцена("мозг", 5 * ВИДЕО_FPS, "Коннектом FlyWire v783",
              "138 639 нейронов, 15 091 983 связи — облёт камеры",
              шаг=крутить, по_модели=False)

        # --- 2. активность -------------------------------------------
        st.ws.dock().maximized = False
        st.ws.switch("experiment")
        st.t_run = 12.0
        st.rate_hz = 200.0
        выбрать_область(app, "AL")
        st.ws.switch("brain")
        st.brain.mode = "activity"
        для_активности(app)
        st.ws.dock().maximized = True
        app.start_run()
        снять_фокус(app)
        ждать(app, lambda: ses.t_ms > 100.0, 300, sh, t0)
        сцена("активность", 6 * ВИДЕО_FPS,
              "Запах: волна идёт по настоящим связям",
              "стимул на антеннальную дольку, цвет — частота; "
              "кадр = 20 мс модельного времени")
        app.cancel_run()
        ждать(app, lambda: not ses.busy, 180, sh, t0)
        st.ws.dock().maximized = False

        # --- 3. тело и 4. глаза --------------------------------------
        sb, ok = песочница(app, ses, sh, t0)
        if ok:
            ждать(app, lambda: False, 3.0, sh, t0)
            сцена("тело", 6 * ВИДЕО_FPS, "Тело NeuroMechFly v2 в MuJoCo",
                  "шесть ног, контакты, прилипание лапок — "
                  "настоящая скорость мухи")
            предмет_слева(app, ses, sh, t0)
            sb.eye_view = True
            ждать(app, lambda: False, 1.0, sh, t0)
            сцена("глаза", 7 * ВИДЕО_FPS,
                  "Глазами мухи: 721 омматидий на глаз",
                  "сверху кадр камеры, снизу показания омматидиев; "
                  "тёмный шар переставляется слева прямо на ходу",
                  шаг=переставлять_слева(app, ses))
            app.cancel_run()
            ждать(app, lambda: not ses.busy, 180, sh, t0)

        # --- 5. концовка ---------------------------------------------
        st.ws.switch("brain")
        st.brain.mode = "anatomy"
        st.ws.dock().maximized = True
        for _ in range(8):
            app.frame()
        сцена("концовка", 3 * ВИДЕО_FPS, "github.com/R1ens/fly-in-silico",
              "код, отчёты и все оговорки", шаг=крутить, по_модели=False)
    finally:
        писарь.close()
        sh.begin = было_begin
        ses.close()
    rep["замеры"]["кадров всего"] = счёт["кадров"]
    rep["замеры"]["кадров в секунду"] = ВИДЕО_FPS
    rep["замеры"]["длительность, с"] = round(счёт["кадров"] / ВИДЕО_FPS, 1)
    rep["замеры"]["модельного времени на кадр, мс"] = МОДЕЛЬ_НА_КАДР
    rep["замеры"]["файл"] = str(путь)
    rep["замеры"]["размер, МБ"] = round(путь.stat().st_size / (1 << 20), 1)
    (OUT / "promo_video.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n  %s — %.1f с, %.1f МБ"
          % (путь.name, счёт["кадров"] / ВИДЕО_FPS,
             путь.stat().st_size / (1 << 20)), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--кадры", action="store_true")
    ap.add_argument("--видео", action="store_true")
    args = ap.parse_args()
    if getattr(args, "кадры"):
        кадры()
    if getattr(args, "видео"):
        видео()


if __name__ == "__main__":
    main()
