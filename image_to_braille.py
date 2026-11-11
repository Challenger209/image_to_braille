#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
依赖：Python 3.8+；安装依赖：python -m pip install -r requirements.txt

用法：
    image_to_braille.py input.png

    options:
    -h, --help            show this help message and exit
    -o OUTPUT, --output OUTPUT
                            输出文件
    -w WIDTH, --width WIDTH
                            输出字符宽度
    --height HEIGHT       输出字符高度
    --six                 使用6点盲文
    --invert              在模式处理前反色；对 edge 模式通常没有明显效果
    --enhance ENHANCE     对比度增强系数
    --mode {adaptive,global,edge}
                            处理模式 (未指定时默认 adaptive)
    --auto                自动分析并调整参数（未指定模式时选择模式，偏向 adaptive）
    --target-density TARGET_DENSITY
                            自动模式下期望的凸点密度 (建议 0.1~0.5, 默认 0.30)
    -t THRESHOLD, --threshold THRESHOLD
                            全局阈值(0-255)
    --edge-low EDGE_LOW   Canny低阈值
    --edge-high EDGE_HIGH
                            Canny高阈值
    --adaptive-block ADAPTIVE_BLOCK
                            自适应邻域大小(最小3，偶数自动加1)
    --adaptive-c ADAPTIVE_C
                            自适应惩罚常数
"""

import argparse
import os
import shlex
import subprocess
import sys
import numpy as np
import cv2
from PIL import Image, ImageEnhance

DOT_BITS_8 = ((0x01, 0x08), (0x02, 0x10), (0x04, 0x20), (0x40, 0x80))
DOT_BITS_6 = ((0x01, 0x08), (0x02, 0x10), (0x04, 0x20))

# 自动模式的轻微先验：优先保留局部明暗关系，避免 edge 因边缘噪声过早胜出。
AUTO_ADAPTIVE_BIAS = 0.10
AUTO_EDGE_WEIGHT = 0.85

def load_and_resize(path, char_width, char_height, enhance, invert, six_dot):
    """加载、透明底转白、灰度、对比度增强、缩放、反色"""
    img = Image.open(path)
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        bg.alpha_composite(img)
        img = bg
    img = img.convert("L")
    ow, oh = img.size

    dot_rows = 3 if six_dot else 4
    if char_height is None:
        char_height = round(char_width * oh * 2 / (ow * dot_rows))
        char_height = max(1, char_height)
    
    if enhance != 1.0:
        img = ImageEnhance.Contrast(img).enhance(enhance)

    # 缩放（统一在提取特征前缩放，避免细线丢失）
    img = img.resize((char_width * 2, char_height * dot_rows), Image.Resampling.BILINEAR)
    img_np = np.array(img)

    # 反色处理（放在最后，不影响原图对比度增强）。Canny 对整体反色通常不敏感。
    if invert:
        img_np = 255 - img_np

    return img_np, char_height

def generate_braille(img_np, char_height, char_width, threshold, six_dot):
    """根据给定阈值生成盲文字符串，并返回凸点密度"""
    dot_rows = 3 if six_dot else 4
    dot_bits = DOT_BITS_6 if six_dot else DOT_BITS_8
    lines = []
    total_dots = 0
    
    for cy in range(char_height):
        chars = []
        for cx in range(char_width):
            bits = 0
            base_x = cx * 2
            base_y = cy * dot_rows
            for ry in range(dot_rows):
                for rx in range(2):
                    v = img_np[base_y + ry, base_x + rx]
                    # 统一逻辑：小于阈值变凸点
                    if v < threshold:
                        bits |= dot_bits[ry][rx]
                        total_dots += 1
            chars.append(chr(0x2800 + bits))
        lines.append("".join(chars))
    
    density = total_dots / (char_height * char_width * dot_rows * 2)
    return "\n".join(lines), density

def find_best_threshold(img_np, target_density):
    """二分查找：根据目标密度，自动寻找最优全局阈值"""
    low, high = 0, 255
    best_th = 128
    img_flat = img_np.flatten()
    
    for _ in range(10):
        mid = (low + high) // 2
        density = np.sum(img_flat < mid) / img_flat.size
        
        if density > target_density:
            high = mid - 1
        else:
            low = mid + 1
        best_th = mid
    return best_th

def choose_auto_mode(img_np):
    """根据对比度和边缘复杂度评分，选择自动模式。"""
    std_dev = float(np.std(img_np))
    gx = cv2.Sobel(img_np, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img_np, cv2.CV_32F, 0, 1)
    edge_magnitude = cv2.magnitude(gx, gy)
    edge_complexity = float(np.mean(edge_magnitude))

    # 将图像统计量转换为 0~1 的可比较信号，保留原有的判定倾向。
    contrast_signal = float(np.clip((std_dev - 20.0) / 80.0, 0.0, 1.0))
    edge_signal = float(np.clip((edge_complexity - 30.0) / 120.0, 0.0, 1.0))
    scores = {
        "adaptive": (
            0.65 * (1.0 - contrast_signal)
            + 0.35 * (1.0 - edge_signal)
            + AUTO_ADAPTIVE_BIAS
        ),
        "global": 0.65 * contrast_signal + 0.35 * (1.0 - edge_signal),
        "edge": (
            0.70 * edge_signal + 0.30 * contrast_signal
        ) * AUTO_EDGE_WEIGHT,
    }
    mode = max(scores, key=scores.get)
    return mode, std_dev, edge_complexity, scores

def format_cli_value(value):
    """将参数值格式化为适合放进复现命令的文本。"""
    if isinstance(value, float):
        return format(value, "g")
    return str(value)

def format_reproducible_command(command):
    """按当前操作系统生成可直接复制执行的命令。"""
    if os.name == "nt":
        return subprocess.list2cmdline(command)
    return shlex.join(command)

def build_reproducible_command(args, mode, threshold, edge_low, edge_high,
                               block_size, adaptive_c):
    """生成关闭自动调参后，仍能得到相同结果的命令。"""
    # 日志包含 Unicode 图标，显式启用 UTF-8 以便在 Windows 控制台直接执行。
    command = ["python", "-X", "utf8", sys.argv[0], args.input]

    if args.output:
        command.extend(["-o", args.output])
    if args.width != 80:
        command.extend(["--width", format_cli_value(args.width)])
    if args.height is not None:
        command.extend(["--height", format_cli_value(args.height)])
    if args.six:
        command.append("--six")
    if args.invert:
        command.append("--invert")
    if args.enhance != 1.0:
        command.extend(["--enhance", format_cli_value(args.enhance)])

    # 自动选择出的模式必须显式写入，否则再次运行会回到默认 adaptive。
    command.extend(["--mode", mode])
    command.extend(["--target-density", format_cli_value(args.target_density)])

    if mode == "adaptive":
        command.extend(["--adaptive-block", format_cli_value(block_size)])
        command.extend(["--adaptive-c", format_cli_value(adaptive_c)])
    elif mode == "global":
        command.extend(["--threshold", format_cli_value(threshold)])
    elif mode == "edge":
        command.extend(["--edge-low", format_cli_value(edge_low)])
        command.extend(["--edge-high", format_cli_value(edge_high)])

    return format_reproducible_command(command)

def process_image(args):
    # 1. 预处理加载（包含所有基础功能）
    img_np, char_height = load_and_resize(
        args.input, args.width, args.height, args.enhance, args.invert, args.six
    )
    
    mode = args.mode
    threshold = args.threshold

    # ---------------- 自动智能判定 ----------------
    if args.auto:
        # 如果用户没有强制指定模式（默认None），则进行自动分析
        if mode is None:
            mode, std_dev, edge_complexity, mode_scores = choose_auto_mode(img_np)
            
            print(f"🤖 [自动分析] 图片对比度: {std_dev:.2f}, 边缘复杂度: {edge_complexity:.2f}")
            print(
                "🤖 [自动评分] "
                + ", ".join(f"{name}={score:.3f}" for name, score in mode_scores.items())
                + f"（adaptive 偏置 +{AUTO_ADAPTIVE_BIAS:.2f}，edge 权重 x{AUTO_EDGE_WEIGHT:.2f}）"
            )
            print(f"🤖 [自动选择] 最佳模式: {mode.upper()}")
        else:
            print(f"🤖 [自动分析] 用户强制指定模式: {mode.upper()}，将在该模式下自动优化参数。")
    else:
        # 不开启自动，且没指定模式，默认使用 adaptive
        if mode is None:
            mode = "adaptive"

    # ---------------- 执行对应模式 ----------------
    if mode == "adaptive":
        block_size = max(3, args.adaptive_block)
        if block_size % 2 == 0:
            block_size += 1
        adaptive_c = args.adaptive_c
        
        if args.auto:
            # 自适应模式参数比较难直接推导，这里跑一次看密度，如果太极端就稍微调一下 c 值
            temp_thresh = cv2.adaptiveThreshold(img_np, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, adaptive_c)
            temp_density = np.count_nonzero(temp_thresh == 0) / temp_thresh.size
            
            if temp_density > args.target_density * 0.8:
                adaptive_c += 2
                print(f"🤖 [自动调参] 局部噪点过多，增大惩罚常数 c 至 {adaptive_c}")
            elif temp_density < args.target_density * 0.3:
                adaptive_c = max(0, adaptive_c - 2)
                print(f"🤖 [自动调参] 局部细节过少，减小惩罚常数 c 至 {adaptive_c}")
        
        img_np = cv2.adaptiveThreshold(img_np, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, adaptive_c)
        text, final_density = generate_braille(img_np, char_height, args.width, 128, args.six)
        print(f"🔧 [自适应模式] Block: {block_size}, C: {adaptive_c}")

    elif mode == "global":
        if args.auto:
            threshold = find_best_threshold(img_np, args.target_density)
            print(f"🤖 [自动调参] 目标密度: {args.target_density*100}%, 自动计算最佳阈值: {threshold}")
        else:
            print(f"🔧 [手动模式] 使用全局阈值: {threshold}")
        
        text, final_density = generate_braille(img_np, char_height, args.width, threshold, args.six)

    elif mode == "edge":
        edge_low, edge_high = args.edge_low, args.edge_high
        
        if args.auto:
            edges = cv2.Canny(img_np, edge_low, edge_high)
            current_density = np.count_nonzero(edges) / edges.size
            
            # 自动反馈调整边缘密度
            if current_density > args.target_density * 0.8:
                edge_low = int(edge_low * 1.5)
                edge_high = int(edge_high * 1.5)
                print(f"🤖 [自动调参] 边缘过多 ({current_density*100:.1f}%)，提高 Canny 阈值...")
            elif current_density < args.target_density * 0.2:
                edge_low = max(10, int(edge_low * 0.6))
                edge_high = max(30, int(edge_high * 0.6))
                print(f"🤖 [自动调参] 边缘过少 ({current_density*100:.1f}%)，降低 Canny 阈值...")
            
            print(f"🤖 [自动调参] 最终 Canny 参数: low={edge_low}, high={edge_high}")
        
        # 执行 Canny
        edges = cv2.Canny(img_np, edge_low, edge_high)
        # Canny 边缘为 255(白)，背景为 0(黑)。我们需要背景白，边缘黑。
        img_np = cv2.bitwise_not(edges)
        
        text, final_density = generate_braille(img_np, char_height, args.width, 128, args.six)

    # ---------------- 输出 ----------------
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(f"✅ 已写入：{args.output} (最终凸点密度: {final_density*100:.1f}%)")
    else:
        print(text)

    if args.auto:
        replay_command = build_reproducible_command(
            args, mode, threshold, edge_low if mode == "edge" else args.edge_low,
            edge_high if mode == "edge" else args.edge_high,
            block_size if mode == "adaptive" else args.adaptive_block,
            adaptive_c if mode == "adaptive" else args.adaptive_c,
        )
        print(f"🔁 [可复现命令] {replay_command}")

def main():
    parser = argparse.ArgumentParser(description="终极全能盲文转换")
    parser.add_argument("input", help="输入图片路径")
    parser.add_argument("-o", "--output", help="输出文件")
    parser.add_argument("-w", "--width", type=int, default=80, help="输出字符宽度")
    parser.add_argument("--height", type=int, default=None, help="输出字符高度")
    parser.add_argument("--six", action="store_true", help="使用6点盲文")
    parser.add_argument("--invert", action="store_true", help="在模式处理前反色；对 edge 模式通常没有明显效果")
    parser.add_argument("--enhance", type=float, default=1.0, help="对比度增强系数")
    
    # 模式选择：默认 None，以便自动模式判断用户是否强制指定
    parser.add_argument("--mode", choices=["adaptive", "global", "edge"], default=None, help="处理模式 (未指定时默认 adaptive)")
    
    # 自动模式开关与目标密度
    parser.add_argument("--auto", action="store_true", help="自动分析并调整参数（未指定模式时选择模式，偏向 adaptive）")
    parser.add_argument("--target-density", type=float, default=0.30, 
                        help="自动模式下期望的凸点密度 (0.1~0.5, 默认 0.30)")
    
    # 手动调参参数
    parser.add_argument("-t", "--threshold", type=int, default=128, help="全局阈值(0-255)")
    parser.add_argument("--edge-low", type=int, default=50, help="Canny低阈值")
    parser.add_argument("--edge-high", type=int, default=150, help="Canny高阈值")
    parser.add_argument("--adaptive-block", type=int, default=3, help="自适应邻域大小(最小3，偶数自动加1)")
    parser.add_argument("--adaptive-c", type=int, default=6, help="自适应惩罚常数")

    args = parser.parse_args()
    process_image(args)

if __name__ == "__main__":
    main()
