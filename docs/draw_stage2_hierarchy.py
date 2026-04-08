"""
Stage 2 계층 트리 예시 + L_hier 작동 원리 다이어그램
교수님 면담 자료용
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# 한글 폰트
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11

fig, axes = plt.subplots(1, 2, figsize=(18, 9))

# =====================================================
# Left: 계층 트리 예시
# =====================================================
ax = axes[0]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')
ax.set_title('Stage 2: LLM Semantic Hierarchy Tree', fontsize=15, fontweight='bold', pad=15)

# Colors
c_super = '#2C3E7B'    # dark blue
c_cat = '#5B7EC7'      # medium blue
c_prim = '#8BAAE2'     # light blue
c_sub = '#B8D4F0'      # very light blue
c_rare = '#FFB74D'     # orange for rare

def draw_node(ax, x, y, text, color, w=1.4, h=0.55, fontsize=9, textcolor='white'):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle="round,pad=0.1",
                          facecolor=color, edgecolor='white', linewidth=1.5)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            color=textcolor, fontweight='bold')

def draw_edge(ax, x1, y1, x2, y2, style='-', color='#666666', lw=1.5):
    ax.plot([x1, x2], [y1, y2], style, color=color, lw=lw)

# Level 0: Super-category
draw_node(ax, 5, 9.2, 'physical_state', c_super, w=2.0, fontsize=10)

# Level 1: Categories
draw_node(ax, 2.5, 7.3, 'deteriorated', c_cat, w=1.8)
draw_node(ax, 7.5, 7.3, 'optical', c_cat, w=1.8)
draw_edge(ax, 5, 8.9, 2.5, 7.6)
draw_edge(ax, 5, 8.9, 7.5, 7.6)

# Level 2: Primitives (dataset attrs)
draw_node(ax, 1.2, 5.4, 'old', c_prim, w=1.2)
draw_node(ax, 3.8, 5.4, 'broken', c_prim, w=1.2)
draw_node(ax, 6.5, 5.4, 'translucent', c_rare, w=1.6, textcolor='black')
draw_node(ax, 8.8, 5.4, 'shiny', c_prim, w=1.2)
draw_edge(ax, 2.5, 7.0, 1.2, 5.7)
draw_edge(ax, 2.5, 7.0, 3.8, 5.7)
draw_edge(ax, 7.5, 7.0, 6.5, 5.7)
draw_edge(ax, 7.5, 7.0, 8.8, 5.7)

# Level 3: Sub-meanings (K prototypes)
# old → K=3
draw_node(ax, 0.3, 3.3, 'worn', c_sub, w=1.0, h=0.45, fontsize=8, textcolor='#333')
draw_node(ax, 1.2, 3.3, 'faded', c_sub, w=1.0, h=0.45, fontsize=8, textcolor='#333')
draw_node(ax, 2.1, 3.3, 'aged', c_sub, w=1.0, h=0.45, fontsize=8, textcolor='#333')
draw_edge(ax, 1.2, 5.1, 0.3, 3.55)
draw_edge(ax, 1.2, 5.1, 1.2, 3.55)
draw_edge(ax, 1.2, 5.1, 2.1, 3.55)

# broken → K=2
draw_node(ax, 3.3, 3.3, 'cracked', c_sub, w=1.1, h=0.45, fontsize=8, textcolor='#333')
draw_node(ax, 4.4, 3.3, 'shattered', c_sub, w=1.2, h=0.45, fontsize=8, textcolor='#333')
draw_edge(ax, 3.8, 5.1, 3.3, 3.55)
draw_edge(ax, 3.8, 5.1, 4.4, 3.55)

# translucent → K=1 (rare)
draw_node(ax, 6.5, 3.3, 'translucent', '#FFE0B2', w=1.5, h=0.45, fontsize=8, textcolor='#333')
draw_edge(ax, 6.5, 5.1, 6.5, 3.55)

# shiny → K=2
draw_node(ax, 8.3, 3.3, 'glossy', c_sub, w=1.0, h=0.45, fontsize=8, textcolor='#333')
draw_node(ax, 9.3, 3.3, 'reflective', c_sub, w=1.2, h=0.45, fontsize=8, textcolor='#333')
draw_edge(ax, 8.8, 5.1, 8.3, 3.55)
draw_edge(ax, 8.8, 5.1, 9.3, 3.55)

# K labels
ax.text(1.2, 2.6, 'K=3', ha='center', fontsize=9, color='#555', style='italic')
ax.text(3.85, 2.6, 'K=2', ha='center', fontsize=9, color='#555', style='italic')
ax.text(6.5, 2.6, 'K=1 (rare)', ha='center', fontsize=9, color='#D84315', style='italic', fontweight='bold')
ax.text(8.8, 2.6, 'K=2', ha='center', fontsize=9, color='#555', style='italic')

# Level labels
ax.text(-0.3, 9.2, 'Level 0\n(super)', fontsize=8, ha='center', va='center', color='#888')
ax.text(-0.3, 7.3, 'Level 1\n(category)', fontsize=8, ha='center', va='center', color='#888')
ax.text(-0.3, 5.4, 'Level 2\n(primitive)', fontsize=8, ha='center', va='center', color='#888')
ax.text(-0.3, 3.3, 'Level 3\n(sub-meaning)', fontsize=8, ha='center', va='center', color='#888')

# Source labels
ax.text(5, 1.8, 'Level 0-1: LLM generates grouping', fontsize=9, ha='center', color='#555')
ax.text(5, 1.3, 'Level 2: Dataset primitives (attrs/objs)', fontsize=9, ha='center', color='#555')
ax.text(5, 0.8, 'Level 3: LLM generates sub-meanings  \u2192  K per primitive', fontsize=9, ha='center', color='#555')

# =====================================================
# Right: L_hier 작동 원리
# =====================================================
ax2 = axes[1]
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.axis('off')
ax2.set_title('L_hier: Long-tail Compensation', fontsize=15, fontweight='bold', pad=15)

# --- Data-rich case (top) ---
ax2.text(5, 9.5, 'Case 1: Data-rich primitive ("old")', fontsize=12, ha='center',
         fontweight='bold', color='#2C3E7B')

# Parent prototype (CLIP-derived)
circle_parent = plt.Circle((2.5, 7.8), 0.9, fill=False, edgecolor=c_cat, linewidth=2.5, linestyle='--')
ax2.add_patch(circle_parent)
ax2.text(2.5, 7.8, '$p_{parent}$\n"deteriorated"', ha='center', va='center', fontsize=8, color=c_cat)

# Child prototypes (well-formed)
for i, (cx, cy, name) in enumerate([(4.5, 8.3, 'worn'), (5.2, 7.6, 'faded'), (4.8, 7.0, 'aged')]):
    c = plt.Circle((cx, cy), 0.25, facecolor=c_prim, edgecolor='white', linewidth=1.5)
    ax2.add_patch(c)
    ax2.text(cx, cy, f'$p_{{{name[:1]}}}$', ha='center', va='center', fontsize=7, color='white', fontweight='bold')

# Features (dots) - well clustered
np.random.seed(42)
for _ in range(12):
    fx = 4.5 + np.random.randn() * 0.5
    fy = 7.6 + np.random.randn() * 0.4
    ax2.plot(fx, fy, 'o', color='#4CAF50', markersize=3, alpha=0.6)

ax2.text(6.5, 8.3, 'features well-clustered', fontsize=9, color='#4CAF50', style='italic')
ax2.text(6.5, 7.8, '$L_{hier} \\approx 0$', fontsize=11, color='#4CAF50', fontweight='bold')
ax2.text(6.5, 7.3, 'sim(f, $p_{child}$) >> sim(f, $p_{parent}$)', fontsize=8, color='#666')

# Divider
ax2.plot([0.5, 9.5], [5.8, 5.8], '-', color='#DDD', lw=1.5)

# --- Long-tail case (bottom) ---
ax2.text(5, 5.3, 'Case 2: Long-tail primitive ("translucent")', fontsize=12, ha='center',
         fontweight='bold', color='#D84315')

# Parent prototype (CLIP-derived, stable)
circle_parent2 = plt.Circle((2.5, 3.5), 0.9, fill=False, edgecolor=c_cat, linewidth=2.5, linestyle='--')
ax2.add_patch(circle_parent2)
ax2.text(2.5, 3.5, '$p_{parent}$\n"optical"', ha='center', va='center', fontsize=8, color=c_cat)
ax2.annotate('CLIP text\n(stable anchor)', xy=(2.5, 2.5), fontsize=7, ha='center',
             color='#888', style='italic')

# Child prototype (poorly formed, far from parent)
c_child = plt.Circle((5.5, 3.0), 0.25, facecolor='#FFB74D', edgecolor='white', linewidth=1.5)
ax2.add_patch(c_child)
ax2.text(5.5, 3.0, '$p_{t}$', ha='center', va='center', fontsize=8, color='white', fontweight='bold')

# Few scattered features
for fx, fy in [(4.8, 3.8), (5.8, 2.3), (6.2, 3.5)]:
    ax2.plot(fx, fy, 'o', color='#FF7043', markersize=4, alpha=0.7)
ax2.text(6.8, 3.5, 'few samples\n(scattered)', fontsize=8, color='#FF7043', style='italic')

# Arrow: L_hier pulls child toward parent region
ax2.annotate('', xy=(4.0, 3.2), xytext=(5.2, 3.0),
            arrowprops=dict(arrowstyle='->', color='#D84315', lw=2.5))
ax2.text(4.6, 2.3, '$L_{hier}$ pulls\ntoward parent\nregion', fontsize=9, ha='center',
         color='#D84315', fontweight='bold')

# Formula
ax2.text(5, 1.2, '$L_{hier} = \\Sigma \\max(0,\\ sim(f, p_{parent}) - sim(f, p_{child}) + margin)$',
         fontsize=11, ha='center', color='#333',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='#F5F5F5', edgecolor='#CCC'))

ax2.text(5, 0.5, 'Parent provides stable semantic region for data-scarce primitives',
         fontsize=9, ha='center', color='#666', style='italic')

plt.tight_layout(pad=2)
plt.savefig('/home/dkkim/.gemini/antigravity/scratch/LHP-CZSL/docs/stage2_hierarchy.png',
            dpi=200, bbox_inches='tight', facecolor='white')
print("Saved: docs/stage2_hierarchy.png")
