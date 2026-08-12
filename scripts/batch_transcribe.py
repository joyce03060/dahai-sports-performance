#!/usr/bin/env python3
"""
batch_transcribe.py - 批量转录视频/音频为中文文字稿（faster-whisper，本地运行，零 token 消耗）

用法:
    python batch_transcribe.py <视频目录> [输出目录] [--model small|medium|large-v3]

默认:
    输入目录: sources/videos/
    输出目录: sources/transcripts/
    模型:     small（CPU 上速度/质量平衡；想更准可换 medium）

说明:
    - 支持 mp4/mov/webm/m4a/mp3/wav 等常见格式
    - 每个视频输出一个同名 .txt，带时间戳
    - 中文识别用 language=zh
    - 首次运行会自动下载模型（small 约 460MB，medium 约 1.5GB，只下一次）
"""
import argparse
import sys
import time
from pathlib import Path

SUFFIXES = {".mp4", ".mov", ".webm", ".m4a", ".mp3", ".wav", ".flac", ".mkv", ".avi"}


def main():
    parser = argparse.ArgumentParser(description="批量转录视频为中文文字稿")
    parser.add_argument("input_dir", nargs="?", default="sources/videos",
                        help="视频目录（默认 sources/videos/）")
    parser.add_argument("output_dir", nargs="?", default="sources/transcripts",
                        help="文字稿输出目录（默认 sources/transcripts/）")
    parser.add_argument("--model", default="small",
                        help="whisper 模型：tiny/base/small/medium/large-v3（默认 small）")
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    if not in_dir.exists():
        print(f"[ERROR] 目录不存在: {in_dir}")
        sys.exit(1)
    out_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(p for p in in_dir.iterdir() if p.suffix.lower() in SUFFIXES)
    if not videos:
        print(f"[WARN] {in_dir} 下没有找到视频/音频文件（支持: {sorted(SUFFIXES)}）")
        sys.exit(1)

    print(f"[INFO] 找到 {len(videos)} 个文件，模型={args.model}，设备=cpu")
    from faster_whisper import WhisperModel
    print("[INFO] 加载模型（首次会自动下载，请耐心等待）...")
    t0 = time.time()
    model = WhisperModel(args.model, device="cpu", compute_type="int8")
    print(f"[INFO] 模型就绪（{time.time()-t0:.0f}s）\n")

    for i, v in enumerate(videos, 1):
        out_txt = out_dir / (v.stem + ".txt")
        if out_txt.exists() and out_txt.stat().st_size > 0:
            print(f"[{i}/{len(videos)}] 跳过（已存在）: {v.name}")
            continue
        print(f"[{i}/{len(videos)}] 转录中: {v.name} ...", flush=True)
        t1 = time.time()
        segments, info = model.transcribe(str(v), language="zh", vad_filter=True)
        lines = [f"# {v.name}", f"# 时长: {info.duration:.0f}s | 语言: {info.language}",
                 ""]
        for seg in segments:
            ts = time.strftime("%H:%M:%S", time.gmtime(seg.start))
            lines.append(f"[{ts}] {seg.text.strip()}")
        out_txt.write_text("\n".join(lines), encoding="utf-8")
        print(f"    完成（{time.time()-t1:.0f}s）→ {out_txt.name}")

    print(f"\n[ALL DONE] 共 {len(videos)} 个文件，输出目录: {out_dir}")


if __name__ == "__main__":
    main()
