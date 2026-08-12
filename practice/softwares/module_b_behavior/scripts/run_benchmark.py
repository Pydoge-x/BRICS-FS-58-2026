"""Benchmark 入口（见 src/eval/benchmark.py）。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.benchmark import main

if __name__ == "__main__":
    main()
