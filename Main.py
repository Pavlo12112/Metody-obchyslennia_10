"""
Лабораторна робота №10. Методи Рунге-Кутта та Адамса розв'язання задачі Коші.
Методи та технології обчислень.

Задача Коші:  y' = f(x, y),  y(a) = y0  на відрізку [a, b].
"""
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

A, B, Y0 = 0.0, 10.0, 1.0


def f(x, y):
    """Права частина: y' = -y + sin(x)."""
    return -y + math.sin(x)


def exact(x):
    """Аналітичний розв'язок: y = (sin x - cos x)/2 + (y0 + 1/2) e^{-x}."""
    return (math.sin(x) - math.cos(x)) / 2 + (Y0 + 0.5) * math.exp(-x)
# =======================================================================


# ----------------------------- Рунге-Кутта 4 -----------------------------
def rk4_step(x, y, h):
    k1 = f(x, y)
    k2 = f(x + h / 2, y + h * k1 / 2)
    k3 = f(x + h / 2, y + h * k2 / 2)
    k4 = f(x + h, y + h * k3)
    return y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def rk4(a, b, y0, h):
    """Метод Рунге-Кутта 4-го порядку зі сталим кроком h на [a, b]."""
    n = int(round((b - a) / h))
    xs = a + h * np.arange(n + 1)
    ys = np.empty(n + 1)
    ys[0] = y0
    for i in range(n):
        ys[i + 1] = rk4_step(xs[i], ys[i], h)
    return xs, ys


def rk4_local_error_exact(xs, ys):
    """phi_n = y_n - y(x_n) (точний розв'язок відомий)."""
    return ys - np.array([exact(x) for x in xs])


def rk4_runge_error(a, b, y0, h):
    """phi_n = 16/15 (y^{h/2}_n - y^h_n) у вузлах сітки з кроком h."""
    xs, y_h = rk4(a, b, y0, h)
    _, y_h2 = rk4(a, b, y0, h / 2)
    return xs, 16 / 15 * (y_h2[::2] - y_h)


def rk4_required_step(a, b, y0, eps, h0=1.0):
    """Крок, потрібний для точності eps (оцінка Рунге): ділимо h навпіл."""
    h = h0
    while True:
        _, phi = rk4_runge_error(a, b, y0, h)
        if np.max(np.abs(phi)) <= eps:
            return h
        h /= 2


def rk4_auto(a, b, y0, eps, h0=0.5):
    """Автоматичний вибір кроку: k = 2^{s+1} = 32 для s = 4."""
    k = 2 ** 5
    x, y, h = a, y0, h0
    xs, ys, hs = [x], [y], []
    while x < b - 1e-14:
        h = min(h, b - x)
        y1 = rk4_step(x, y, h)
        y2 = rk4_step(x + h / 2, rk4_step(x, y, h / 2), h / 2)
        err = 16 / 15 * abs(y2 - y1)
        if err > eps:
            h /= 2                      # крок завеликий
            continue
        x += h
        y = y2
        xs.append(x); ys.append(y); hs.append(h)
        if err <= eps / k:
            h *= 2                      # крок можна збільшити
    return np.array(xs), np.array(ys), np.array(hs)


# --------------------- Адамс: прогноз і корекція (2 порядок) ---------------------
def adams2(a, b, y0, h, n_iter=2, modify=True):
    """
    Прогноз:      y^пр  = y_n + h/2 (3 f_n - f_{n-1})
    Модифікація:  y^мод = y^пр + 5/6 (y^кор_{n-1} - y^пр_{n-1})   (різниця з попереднього кроку)
    Корекція:     y^(i+1) = y_n + h/2 (f(x_{n+1}, y^(i)) + f_n),  2 ітерації
    Кінцеве:      y_{n+1} = y^кор - 1/6 (y^кор - y^пр)
    Повертає вузли, значення, y^пр та y^кор (для оцінки похибки).
    """
    n = int(round((b - a) / h))
    xs = a + h * np.arange(n + 1)
    y = np.empty(n + 1)
    y_pr = np.full(n + 1, np.nan)
    y_kor = np.full(n + 1, np.nan)
    y[0] = y0
    y[1] = rk4_step(xs[0], y[0], h)     # стартове значення - методом Рунге-Кутта
    diff_prev = 0.0                     # (y^кор - y^пр) з попереднього кроку
    for i in range(1, n):
        fn, fn1 = f(xs[i], y[i]), f(xs[i - 1], y[i - 1])
        pr = y[i] + h / 2 * (3 * fn - fn1)
        cur = pr + 5 / 6 * diff_prev if modify else pr
        for _ in range(n_iter):
            cur = y[i] + h / 2 * (f(xs[i + 1], cur) + fn)
        y_pr[i + 1], y_kor[i + 1] = pr, cur
        diff_prev = cur - pr
        y[i + 1] = cur - 1 / 6 * (cur - pr)
    return xs, y, y_pr, y_kor


def adams2_auto(a, b, y0, eps, h0=0.2):
    """
    Автоматичний вибір кроку. Локальна похибка R = |y^кор - y^пр|/6.
    R > eps -> h/2 (повтор кроку);  R <= eps/k (k = 2^{s+1} = 8, s = 2) -> h*2.
    Після зміни кроку значення y_{n-1} (на сітці нового кроку) обчислюється
    кроком Рунге-Кутта назад.
    """
    k = 2 ** 3
    x, y, h = a, y0, h0
    y_prev = rk4_step(x, y, -h)
    xs, ys, hs = [x], [y], []
    while x < b - 1e-14:
        if x + h > b:
            h = b - x
            y_prev = rk4_step(x, y, -h)
        fn, fn1 = f(x, y), f(x - h, y_prev)
        pr = y + h / 2 * (3 * fn - fn1)
        cur = pr
        for _ in range(2):
            cur = y + h / 2 * (f(x + h, cur) + fn)
        err = abs(cur - pr) / 6
        if err > eps:
            h /= 2
            y_prev = rk4_step(x, y, -h)
            continue
        y_new = cur - (cur - pr) / 6
        y_prev, y, x = y, y_new, x + h
        xs.append(x); ys.append(y); hs.append(h)
        if err <= eps / k:
            h *= 2
            y_prev = rk4_step(x, y, -h)
    return np.array(xs), np.array(ys), np.array(hs)


# ------------------------------- Графіки -------------------------------
def savefig(name):
    plt.tight_layout()
    plt.savefig(name, dpi=150)
    plt.close()


def main():
    ex = np.vectorize(exact)
    res = {}

    # ===== Ч.1 =====
    h = 0.1
    xs, y, y_pr, y_kor = adams2(A, B, Y0, h)
    err_true = y - ex(xs)
    est = (y_kor - y_pr) / 6          # оцінка похибки на етапі корекції: -(kor-pr)/6
    res["adams_h"] = h
    res["adams_max_err"] = float(np.max(np.abs(err_true)))

    plt.figure(figsize=(7, 4))
    xx = np.linspace(A, B, 400)
    plt.plot(xx, ex(xx), "k-", label="точний розв'язок")
    plt.plot(xs, y, "r.", ms=3, label=f"Адамс 2-го порядку, h={h}")
    plt.xlabel("x"); plt.ylabel("y"); plt.grid(alpha=.3); plt.legend()
    savefig("fig1_adams_solution.png")

    plt.figure(figsize=(7, 4))
    plt.plot(xs, err_true, "b-", label=r"$\varphi_n=y_n-y(x_n)$")
    plt.xlabel("x"); plt.ylabel("локальна похибка"); plt.grid(alpha=.3); plt.legend()
    savefig("fig2_adams_error_exact.png")

    plt.figure(figsize=(7, 4))
    # істинна похибка одного кроку: беремо точні y_n, y_{n-1}
    loc = np.full(len(xs), np.nan)
    for i in range(1, len(xs) - 1):
        yn, yn1 = exact(xs[i]), exact(xs[i - 1])
        fn, fn1 = f(xs[i], yn), f(xs[i - 1], yn1)
        pr = yn + h / 2 * (3 * fn - fn1)
        cur = pr
        for _ in range(2):
            cur = yn + h / 2 * (f(xs[i + 1], cur) + fn)
        loc[i + 1] = (cur - (cur - pr) / 6) - exact(xs[i + 1])
    plt.plot(xs[2:], est[2:], "g-", label=r"оцінка $(y^{кор}_{n+1}-y^{пр}_{n+1})/6$")
    plt.plot(xs[2:], loc[2:], "b--", label="істинна похибка одного кроку")
    plt.xlabel("x"); plt.ylabel("похибка"); plt.grid(alpha=.3); plt.legend()
    savefig("fig3_adams_error_estimate.png")

    # перевірка оптимальності кроку: h, h/2
    table_adams = []
    for hh in (0.2, 0.1, 0.05, 0.025, 0.0125):
        xs_, y_, pr_, kr_ = adams2(A, B, Y0, hh)
        table_adams.append((hh, float(np.max(np.abs(y_ - ex(xs_)))),
                            float(np.nanmax(np.abs(kr_ - pr_)) / 6)))
    res["table_adams"] = table_adams

    eps_a = 1e-5
    xa, ya, ha = adams2_auto(A, B, Y0, eps_a)
    res["adams_auto"] = dict(eps=eps_a, steps=len(ha),
                             hmin=float(ha.min()), hmax=float(ha.max()),
                             max_err=float(np.max(np.abs(ya - ex(xa)))))
    plt.figure(figsize=(7, 4))
    plt.step(xa[:-1], ha, where="post", color="m")
    plt.xlabel("x"); plt.ylabel("h(x)"); plt.grid(alpha=.3)
    plt.title(f"Адамс 2-го порядку, автовибір кроку, eps={eps_a:g}")
    savefig("fig4_adams_auto_step.png")

    # ===== Ч.2 =====
    h2 = 1e-2
    xr, yr = rk4(A, B, Y0, h2)
    phi_exact = rk4_local_error_exact(xr, yr)
    _, phi_runge = rk4_runge_error(A, B, Y0, h2)
    res["rk4_h"] = h2
    res["rk4_max_err"] = float(np.max(np.abs(phi_exact)))
    res["rk4_max_runge"] = float(np.max(np.abs(phi_runge)))

    plt.figure(figsize=(7, 4))
    plt.plot(xr, phi_exact, "b-")
    plt.xlabel("x"); plt.ylabel(r"$\varphi_n=y_n-y(x_n)$"); plt.grid(alpha=.3)
    plt.title("РК4: похибка за точним розв'язком, h=0.01")
    savefig("fig5_rk4_error_exact.png")

    plt.figure(figsize=(7, 4))
    plt.plot(xr, phi_runge, "r-", label=r"оцінка Рунге $\frac{16}{15}(y^{h/2}-y^{h})$")
    plt.plot(xr, -phi_exact, "b--", label=r"$-\varphi_n$ (точна)")
    plt.xlabel("x"); plt.ylabel("похибка"); plt.grid(alpha=.3); plt.legend()
    savefig("fig6_rk4_error_runge.png")

    # залежність похибки від кроку
    hs_list = [0.4, 0.2, 0.1, 0.05, 0.025, 0.0125]
    e_rk, e_ad = [], []
    for hh in hs_list:
        x_, y_ = rk4(A, B, Y0, hh)
        e_rk.append(float(np.max(np.abs(y_ - ex(x_)))))
        x_, y_, _, _ = adams2(A, B, Y0, hh)
        e_ad.append(float(np.max(np.abs(y_ - ex(x_)))))
    res["h_study"] = list(zip(hs_list, e_rk, e_ad))
    slope_rk = np.polyfit(np.log(hs_list), np.log(e_rk), 1)[0]
    slope_ad = np.polyfit(np.log(hs_list), np.log(e_ad), 1)[0]
    res["slopes"] = (float(slope_rk), float(slope_ad))

    plt.figure(figsize=(7, 4))
    plt.loglog(hs_list, e_rk, "o-", label=f"РК4 (нахил ≈ {slope_rk:.2f})")
    plt.loglog(hs_list, e_ad, "s-", label=f"Адамс-2 (нахил ≈ {slope_ad:.2f})")
    plt.xlabel("h"); plt.ylabel("max |y_n - y(x_n)|"); plt.grid(alpha=.3, which="both"); plt.legend()
    savefig("fig7_error_vs_h.png")

    # потрібний крок для заданої точності
    eps = 1e-8
    h_req = rk4_required_step(A, B, Y0, eps)
    res["rk4_required"] = (eps, h_req)
    xq, yq, hq = rk4_auto(A, B, Y0, eps)
    res["rk4_auto"] = dict(eps=eps, steps=len(hq), hmin=float(hq.min()), hmax=float(hq.max()),
                           max_err=float(np.max(np.abs(yq - ex(xq)))))
    plt.figure(figsize=(7, 4))
    plt.step(xq[:-1], hq, where="post", color="m")
    plt.xlabel("x"); plt.ylabel("h(x)"); plt.grid(alpha=.3)
    plt.title(f"РК4, автовибір кроку, eps={eps:g}")
    savefig("fig8_rk4_auto_step.png")

    import json
    with open("results.json", "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    for k, v in res.items():
        print(k, v)


if __name__ == "__main__":
    main()