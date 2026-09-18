"""Synthetic holdout drawings of kinds the MCP was never tuned on (flowchart, table, engineering
drawing, scanned copy). Written to regress/holdout/g_*.png."""
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle, Polygon, Rectangle  # noqa: E402

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
OUT = Path(__file__).resolve().parent / 'holdout'
OUT.mkdir(exist_ok=True)


def save(fig, name, dpi=110):
    fig.savefig(OUT / f'{name}.png', dpi=dpi, facecolor='white')
    plt.close(fig)


def flowchart():
    fig, ax = plt.subplots(figsize=(8, 7)); ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')
    def box(x, y, t, fc='white'):
        ax.add_patch(FancyBboxPatch((x - 1.4, y - 0.45), 2.8, 0.9, boxstyle='round,pad=0.05', fc=fc, ec='black', lw=1.5))
        ax.text(x, y, t, ha='center', va='center', fontsize=12)
    def diamond(x, y, t):
        ax.add_patch(Polygon([[x, y + 0.8], [x + 1.6, y], [x, y - 0.8], [x - 1.6, y]], fc='#fff3c4', ec='black', lw=1.5))
        ax.text(x, y, t, ha='center', va='center', fontsize=11)
    def arrow(a, b, t=None):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=14, lw=1.4, color='black'))
        if t:
            ax.text((a[0] + b[0]) / 2 + 0.15, (a[1] + b[1]) / 2, t, fontsize=10)
    box(5, 9.2, '开始采集数据', '#dbeafe'); arrow((5, 8.75), (5, 8.05))
    box(5, 7.6, '预处理与去噪'); arrow((5, 7.15), (5, 6.45))
    diamond(5, 5.6, '质量合格？'); arrow((5, 4.8), (5, 4.05), '是'); arrow((6.6, 5.6), (8.2, 5.6), '否')
    box(8.4, 4.8, '重新采样', '#fee2e2'); arrow((8.4, 5.25), (8.4, 5.55))
    box(5, 3.6, '特征提取'); arrow((5, 3.15), (5, 2.45))
    box(5, 2.0, '模型训练', '#dcfce7'); arrow((5, 1.55), (5, 0.95))
    box(5, 0.5, '输出结果报告', '#dbeafe')
    save(fig, 'g_flowchart')


def table():
    fig, ax = plt.subplots(figsize=(8, 4)); ax.axis('off')
    rows = [['样品编号', '深度/m', '孔隙度/%', '渗透率/mD', '岩性'],
            ['A-01', '1250.5', '18.2', '35.6', '细砂岩'], ['A-02', '1262.0', '15.7', '12.3', '粉砂岩'],
            ['A-03', '1275.3', '21.4', '88.1', '中砂岩'], ['B-01', '1301.8', '9.6', '1.2', '泥岩'],
            ['B-02', '1318.4', '12.9', '6.8', '粉砂岩']]
    t = ax.table(cellText=rows, loc='center', cellLoc='center')
    t.auto_set_font_size(False); t.set_fontsize(12); t.scale(1, 1.8)
    for (r, c), cell in t.get_celld().items():
        cell.set_linewidth(1.0)
        if r == 0:
            cell.set_facecolor('#d9d9d9')
    save(fig, 'g_table')


def engineering():
    fig, ax = plt.subplots(figsize=(8, 6)); ax.set_xlim(0, 200); ax.set_ylim(0, 150); ax.set_aspect('equal'); ax.axis('off')
    ax.add_patch(Rectangle((30, 40), 120, 70, fill=False, lw=1.6))
    ax.add_patch(Circle((70, 75), 12, fill=False, lw=1.6)); ax.add_patch(Circle((115, 75), 8, fill=False, lw=1.6))
    for x0 in range(30, 60, 4):                                          # hatched section
        ax.plot([x0, x0 + 10], [40, 55], color='black', lw=0.8)
    ax.plot([70, 70], [55, 95], 'k-.', lw=0.8); ax.plot([50, 90], [75, 75], 'k-.', lw=0.8)
    def dim(x0, y0, x1, y1, t, off=(0, -8)):
        ax.annotate('', xy=(x0, y0), xytext=(x1, y1), arrowprops=dict(arrowstyle='<->', lw=0.9))
        ax.text((x0 + x1) / 2 + off[0], (y0 + y1) / 2 + off[1] + 3, t, ha='center', fontsize=11)
    dim(30, 28, 150, 28, '120'); dim(162, 40, 162, 110, '70', off=(8, 0))
    ax.text(70, 91, 'Φ24', ha='center', fontsize=11); ax.text(115, 86, 'Φ16', ha='center', fontsize=11)
    ax.text(100, 128, '法兰盘  主视图  1:2', ha='center', fontsize=14)
    ax.text(160, 10, '材料: Q235', fontsize=10)
    save(fig, 'g_engineering')


def scanned():
    from io import BytesIO
    from PIL import Image, ImageFilter
    img = Image.open(OUT / 'g_flowchart.png').convert('RGB')
    img = img.rotate(1.2, resample=Image.BICUBIC, fillcolor=(255, 255, 255))
    a = np.asarray(img).astype(np.float32) / 255 * np.array([245, 232, 205], np.float32)   # yellowish paper
    a = np.clip(a + np.random.default_rng(1).normal(0, 8, a.shape), 0, 255).astype(np.uint8)
    img = Image.fromarray(a).filter(ImageFilter.GaussianBlur(0.8))
    buf = BytesIO(); img.save(buf, 'JPEG', quality=60); buf.seek(0)
    Image.open(buf).convert('RGB').save(OUT / 'g_scanned.png')


if __name__ == '__main__':
    flowchart(); table(); engineering(); scanned()
    from PIL import Image
    for p in sorted(OUT.glob('g_*.png')):
        print(p.name, Image.open(p).size)
