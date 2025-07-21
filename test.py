import torch

# 检查 GPU 是否可用
if torch.cuda.is_available():
    print("GPU 可用，使用的 GPU 设备为:", torch.cuda.get_device_name(0))
    device = torch.device("cuda:0")
else:
    print("GPU 不可用，使用 CPU 进行计算。")
    device = torch.device("cpu")

# 创建张量并移动到指定设备
x = torch.randn(1000, 1000).to(device)
y = torch.randn(1000, 1000).to(device)

# 进行矩阵乘法运算
for i in range(100):
    result = torch.matmul(x, y)

# 检查结果是否在 GPU 上
print("计算结果是否在 GPU 上:", result.is_cuda)

# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# from matplotlib.patches import FancyArrowPatch
#
# fig, ax = plt.subplots(figsize=(8, 10))
# ax.set_xlim(0, 8)
# ax.set_ylim(0, 12)
# ax.axis('off')
#
# # Define boxes
# boxes = [
#     (4, 1, "Input\n\( \mathbf{X}^b \)", 2, 1),
#     (4, 2.5, "Linear\n\( S \\to d_m \)", 2, 1),
#     (6, 2.5, "Pos. Embed\n\( \mathbf{P}_{\\text{pos}}[0, :] \)", 2, 1),
#     (4, 4, "Encoder Layer 1\nSelf-Attn + FFN", 2, 1),
#     (4, 5.5, "Encoder Layer 2\nSelf-Attn + FFN", 2, 1),
#     (4, 7, "Memory\n\( \mathbf{M}^b \)", 2, 1),
#     (2, 7, "Zero Query\n\( \mathbf{0} \)", 2, 1),
#     (0, 7, "Query Pos.\n\( \mathbf{P}_{\\text{query}} \)", 2, 1),
#     (2, 8.5, "Decoder Layer 1\nSelf-Attn + Cross-Attn + FFN", 2, 1),
#     (2, 10, "Decoder Layer 2\nSelf-Attn + Cross-Attn + FFN", 2, 1),
#     (2, 11.5, "Linear\n\( d_m \\to S \)", 2, 1),
#     (2, 13, "Output\n\( \mathbf{y}^b \)", 2, 1)
# ]
#
# # Draw boxes
# for (x, y, label, w, h) in boxes:
#     ax.add_patch(patches.Rectangle((x-w/2, y-h/2), w, h, edgecolor='black', facecolor='lightblue'))
#     ax.text(x, y, label, ha='center', va='center', fontsize=10)
#
# # Draw arrows
# arrows = [
#     ((4, 1.5), (4, 2)),
#     ((4, 3), (4, 3.5)),
#     ((6, 2.5), (5, 2.5), (4.5, 3)),
#     ((4, 4.5), (4, 5)),
#     ((4, 6), (4, 6.5)),
#     ((0, 7), (1, 7), (1.5, 7.5)),
#     ((2, 7.5), (2, 8)),
#     ((2, 9), (2, 9.5)),
#     ((2, 10.5), (2, 11)),
#     ((4, 7), (4, 8), (3, 8.5)),
#     ((4, 7), (4, 9.5), (3, 10))
# ]
#
# for points in arrows:
#     if len(points) == 2:
#         (x1, y1), (x2, y2) = points
#         ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), mutation_scale=10))
#     else:
#         (x1, y1), (xm, ym), (x2, y2) = points
#         ax.add_patch(FancyArrowPatch((x1, y1), (xm, ym), mutation_scale=10))
#         ax.add_patch(FancyArrowPatch((xm, ym), (x2, y2), mutation_scale=10))
#
# plt.tight_layout()
# plt.show()