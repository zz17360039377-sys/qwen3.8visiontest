#!/usr/bin/env python3
"""render_stats_figures -- Nature-style statistics figures for the classification
experiments (recheck matrix + delegated-agent batch classification).

Figure contract (each figure defends one claim):
  stats_recheck_configs.png   claim: 提示词/参考图/预处理的选择决定复核一致率上限
  stats_recheck_perclass.png  claim: few-shot 参考图特异性修复图标类（0/6/7）
  stats_confusion_best.png    claim: 最优配置混淆矩阵近对角
  stats_cls_pattern_confusion.png  claim: 清晰图案的主要错误是 R/B 镜像混淆
  stats_cls_digit_confusion.png    claim: 暗光切片错误呈弥散分布（质量上限）
  stats_cls_perclass.png      claim: 单类准确率由图像清晰度决定

Style: Nature figure conventions (sans-serif, 7-8pt, no top/right spines,
restrained palette, direct labels, n reported). PNG 300 dpi + SVG (editable).

Run:  python3 tools/render_stats_figures.py
"""

from __future__ import annotations

import os
import re
import sys

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(TOOLS_DIR)
sys.path.insert(0, ROOT)

DS = os.path.join(ROOT, "testenv", "datasets")
RUN = os.path.join(ROOT, "testenv", "results", "vlm_probe")
ARCHIVE = os.path.join(ROOT, "docs", "test_assets")
OUT = os.path.join(ROOT, "docs", "pic")
os.makedirs(OUT, exist_ok=True)

import matplotlib as mpl
import matplotlib.font_manager as fm

_CJK_TTC = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if os.path.isfile(_CJK_TTC):
    fm.fontManager.addfont(_CJK_TTC)
    _cjk_name = fm.FontProperties(fname=_CJK_TTC).get_name()
else:
    _cjk_name = "Noto Sans CJK SC"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": [_cjk_name, "DejaVu Sans", "Arial", "sans-serif"],
    "font.size": 8,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
})

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

NEUTRAL = "#9aa3ad"   # 中性灰
SIGNAL = "#3b6fb5"    # 主信号蓝
ACCENT = "#c2504c"    # 强调红（仅用于最佳/关键项）
BEST = "#2f5f9e"


def _save(fig, name):
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(OUT, name + f".{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


# ------------------------------------------------------- recheck md 解析

def parse_recheck_md(path: str):
    """解析归档复核报告 -> (总体一致率, 每类一致率 dict, 混淆矩阵 (labels, M))。"""
    s = open(path, encoding="utf-8").read()
    acc = float(re.search(r"一致率[：:]\s*\**([\d.]+)%", s).group(1)) / 100
    per_cls = {}
    m = re.search(r"## 每类明细.*?\n(\|.*?\n)+", s, re.S)
    if m:
        for ln in m.group(0).splitlines()[2:]:
            cells = [c.strip().strip("*") for c in ln.split("|")[1:-1]]
            if len(cells) >= 5 and cells[0].isdigit():
                per_cls[cells[0]] = float(cells[4].rstrip("%")) / 100
    labels, mat = [], []
    m = re.search(r"## 混淆矩阵.*?\n(\|.*?\n)+", s, re.S)
    if m:
        rows = [ln for ln in m.group(0).splitlines() if ln.startswith("|")]
        head = [c.strip().strip("*") for c in rows[0].split("|")[1:-1]]
        labels = [c for c in head[1:] if c and c != "无解析"]  # 第一列是行名，跳过
        for ln in rows[2:]:
            cells = [c.strip().strip("*") for c in ln.split("|")[1:-1]]
            if not cells or not re.match(r"\d+", cells[0]):
                continue
            vals = []
            for c in cells[1:]:
                try:
                    vals.append(int(c))
                except ValueError:
                    vals.append(0)
            mat.append(vals[: len(labels)])
    return acc, per_cls, (labels, np.array(mat))


REPORTS = {
    "仅放大 ×3": "report_recheck_C_x3_norefs.md",
    "仅放大 ×6": "report_recheck_A_x6_norefs.md",
    "放大 ×3 + 自动对比度": "report_recheck_D_x3_ac.md",
    "放大 ×6 + few-shot 参考图": "report_recheck_B_x6_refs.md",
    "放大 ×6 + 参考图 + 自动对比度（最优）": "report_E.md",
    "最优配置换种子复测（每类 100 张）": "report_F.md",
}


# ------------------------------------------------------------- 图 1: 配置对比

def fig_recheck_configs():
    names, accs, ns = [], [], []
    for label, fn in REPORTS.items():
        p = os.path.join(ARCHIVE, fn)
        acc = float(re.search(r"一致率[：:]\s*\**([\d.]+)%", open(p, encoding="utf-8").read()).group(1)) / 100
        n = int(re.search(r"本次复核 (\d+) 张", open(p, encoding="utf-8").read()).group(1))
        names.append(label)
        accs.append(acc)
        ns.append(n)
    order = np.argsort(accs)
    names = [names[i] for i in order]
    accs = [accs[i] for i in order]
    ns = [ns[i] for i in order]

    fig, ax = plt.subplots(figsize=(5.4, 2.6))
    colors = [BEST if a == max(accs) else SIGNAL if "参考图" in n else NEUTRAL
              for n, a in zip(names, accs)]
    bars = ax.barh(names, [a * 100 for a in accs], color=colors, height=0.62)
    for b, a, n in zip(bars, accs, ns):
        ax.text(a * 100 + 0.15, b.get_y() + b.get_height() / 2,
                f"{a:.1%}  (n={n})", va="center", fontsize=7.5)
    ax.set_xlim(90, 101)
    ax.set_xlabel("与人工标注的一致率 (%)")
    ax.set_title("复核一致率由提示词配置决定\n(条形越靠上配置越优；n = 该组实际复核图片数)")
    ax.set_ylabel("提示词配置")
    _save(fig, "stats_recheck_configs")


# --------------------------------------------------- 图 2: 每类 base vs best

def fig_recheck_perclass():
    acc_a, per_a, _ = parse_recheck_md(os.path.join(ARCHIVE, REPORTS["仅放大 ×6"]))
    acc_e, per_e, _ = parse_recheck_md(os.path.join(ARCHIVE, REPORTS["放大 ×6 + 参考图 + 自动对比度（最优）"]))
    cls = sorted(per_a, key=int)
    x = np.arange(len(cls))
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    ax.bar(x - 0.2, [per_a[c] * 100 for c in cls], width=0.38,
           color=NEUTRAL, label="基线：仅放大 ×6")
    ax.bar(x + 0.2, [per_e[c] * 100 for c in cls], width=0.38,
           color=BEST, label="最优：+ few-shot 参考图 + 对比度")
    ax.set_xticks(x, cls)
    ax.set_ylim(60, 102)
    ax.set_xlabel("类别 (0=哨兵图标, 1–4=数字, 6=前哨, 7=基地)")
    ax.set_ylabel("与人工标注一致率 (%)")
    ax.set_title("few-shot 参考图特异性修复图标类")
    ax.legend(loc="lower right")
    for i, c in enumerate(cls):
        if per_a[c] < 99.9:
            ax.text(i - 0.2, per_a[c] * 100 + 0.5, f"{per_a[c]:.0%}",
                    ha="center", fontsize=6.5)
        if per_e[c] < 99.9:
            ax.text(i + 0.2, per_e[c] * 100 + 0.5, f"{per_e[c]:.0%}",
                    ha="center", fontsize=6.5)
    _save(fig, "stats_recheck_perclass")


# --------------------------------------------------- 图 3: 最优配置混淆矩阵

def fig_recheck_confusion_best():
    _, _, (labels, mat) = parse_recheck_md(os.path.join(ARCHIVE, REPORTS["放大 ×6 + 参考图 + 自动对比度（最优）"]))
    fig, ax = plt.subplots(figsize=(3.2, 2.9))
    im = ax.imshow(mat, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("VLM 预测")
    ax.set_ylabel("人工标注")
    ax.set_title("最优配置混淆矩阵（n=350）")
    for i in range(len(labels)):
        for j in range(len(labels)):
            if mat[i, j]:
                ax.text(j, i, str(mat[i, j]), ha="center", va="center", fontsize=7,
                        color="white" if mat[i, j] > mat.max() / 2 else "black")
    fig.colorbar(im, shrink=0.85)
    _save(fig, "stats_confusion_best")


# ------------------------------------- 图 4/5/6: 独立代理批量分类（vlm_probe）

def cls_confusion(ds_name):
    from tools.eval_testenv import cls_token, load_classes
    classes = load_classes(os.path.join(DS, ds_name))
    gt_dir = os.path.join(DS, ds_name, "labels")
    pred_dir = os.path.join(RUN, ds_name, "labels")
    pairs = []
    for f in sorted(os.listdir(gt_dir)):
        stem = os.path.splitext(f)[0]
        gt = open(os.path.join(gt_dir, f)).read().strip()
        pp = os.path.join(pred_dir, stem + ".txt")
        if not os.path.isfile(pp):
            continue
        pred = cls_token(open(pp).read().strip(), classes) or "?"
        pairs.append((gt, pred))
    labels = sorted({g for g, _ in pairs} | {p for _, p in pairs})
    idx = {l: i for i, l in enumerate(labels)}
    m = np.zeros((len(labels), len(labels)), dtype=int)
    for g, p in pairs:
        m[idx[g], idx[p]] += 1
    return labels, m, len(pairs)


def _confusion_fig(labels, m, title, name, n):
    fig, ax = plt.subplots(figsize=(max(3.4, 0.42 * len(labels) + 1),
                                    max(3.0, 0.42 * len(labels) + 0.6)))
    im = ax.imshow(m, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=45, fontsize=7)
    ax.set_yticks(range(len(labels)), labels, fontsize=7)
    ax.set_xlabel("VLM 预测")
    ax.set_ylabel("人工标注")
    ok = sum(m[i][i] for i in range(len(labels)))
    ax.set_title(f"{title}\naccuracy {ok}/{n} = {ok / max(1, n):.0%}")
    for i in range(len(labels)):
        for j in range(len(labels)):
            if m[i, j]:
                ax.text(j, i, str(m[i, j]), ha="center", va="center", fontsize=6.5,
                        color="white" if m[i, j] > m.max() / 2 else "black")
    _save(fig, name)


def fig_cls_confusions():
    labels, m, n = cls_confusion("cls_armor_digit_slices")
    _confusion_fig(labels, m, "灰度切片直接批量分类（同批数据）",
                   "stats_cls_digit_confusion", n)


def fig_recheck_confusion_baseline():
    _, _, (labels, mat) = parse_recheck_md(os.path.join(ARCHIVE, REPORTS["仅放大 ×6"]))
    fig, ax = plt.subplots(figsize=(3.2, 2.9))
    im = ax.imshow(mat, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("VLM 预测")
    ax.set_ylabel("人工标注")
    ax.set_title("基线配置混淆矩阵（n=350）")
    for i in range(len(labels)):
        for j in range(len(labels)):
            if mat[i, j]:
                ax.text(j, i, str(mat[i, j]), ha="center", va="center", fontsize=7,
                        color="white" if mat[i, j] > mat.max() / 2 else "black")
    fig.colorbar(im, shrink=0.85)
    _save(fig, "stats_confusion_baseline")


def fig_batch_perclass():
    """直接批量分类（无复核用法）在灰度切片上的逐类准确率。"""
    from tools.eval_testenv import load_classes
    ds_name = "cls_armor_digit_slices"
    classes = load_classes(os.path.join(DS, ds_name))
    gt_dir = os.path.join(DS, ds_name, "labels")
    pred_dir = os.path.join(RUN, ds_name, "labels")
    per = {c: [0, 0] for c in classes}
    for f in sorted(os.listdir(gt_dir)):
        stem = os.path.splitext(f)[0]
        gt = open(os.path.join(gt_dir, f)).read().strip()
        pp = os.path.join(pred_dir, stem + ".txt")
        if not os.path.isfile(pp):
            continue
        pred = open(pp).read().strip()
        per[gt][0] += 1
        if gt == pred:
            per[gt][1] += 1
    cls = sorted(per, key=int)
    vals = [per[c][1] / per[c][0] * 100 if per[c][0] else 0 for c in cls]
    ns = [per[c][0] for c in cls]
    fig, ax = plt.subplots(figsize=(4.2, 2.5))
    bars = ax.bar(cls, vals, color=NEUTRAL, width=0.55)
    for b, v, n in zip(bars, vals, ns):
        ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.0f}%\n(n={n})",
                ha="center", fontsize=6.5)
    ax.set_ylim(0, 108)
    ax.set_xlabel("类别 (0=哨兵图标, 1–4=数字, 6=前哨, 7=基地)")
    ax.set_ylabel("直接批量分类准确率 (%)")
    ax.set_title("同批灰度切片：直接批量分类的逐类准确率\n(提示词为测试代理自拟, n=40/类)")
    _save(fig, "stats_batch_perclass")


def main():
    fig_recheck_configs()
    fig_recheck_perclass()
    fig_recheck_confusion_best()
    fig_recheck_confusion_baseline()
    fig_cls_confusions()
    fig_batch_perclass()


if __name__ == "__main__":
    main()
