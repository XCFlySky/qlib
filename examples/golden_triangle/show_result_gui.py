#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
用 tkinter 把选股结果 CSV 显示成图形界面表格。

Usage:
    .venv/Scripts/python examples/golden_triangle/show_result_gui.py result_20240620.csv
"""

import sys
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import pandas as pd


def show_csv(path: str):
    df = pd.read_csv(path)

    root = tk.Tk()
    root.title(f"Golden Triangle 选股结果 - {Path(path).name}")
    root.geometry("1400x600")

    # 框架 + 滚动条
    frame = ttk.Frame(root)
    frame.pack(fill="both", expand=True)

    scroll_y = ttk.Scrollbar(frame, orient="vertical")
    scroll_x = ttk.Scrollbar(frame, orient="horizontal")

    tree = ttk.Treeview(
        frame,
        columns=list(df.columns),
        show="headings",
        yscrollcommand=scroll_y.set,
        xscrollcommand=scroll_x.set,
    )

    scroll_y.config(command=tree.yview)
    scroll_x.config(command=tree.xview)

    scroll_y.pack(side="right", fill="y")
    scroll_x.pack(side="bottom", fill="x")
    tree.pack(side="left", fill="both", expand=True)

    # 表头
    for col in df.columns:
        tree.heading(col, text=col)
        # 根据列名设置列宽
        if col in ("instrument", "name", "signal_type", "industry"):
            tree.column(col, width=100, anchor="center")
        elif col == "cross_date":
            tree.column(col, width=120, anchor="center")
        else:
            tree.column(col, width=110, anchor="center")

    # 数据行，数值格式化
    for _, row in df.iterrows():
        values = []
        for col in df.columns:
            v = row[col]
            if col in ("ma5", "ma10", "ma20", "volume_ratio", "turnover", "ma10_20_ratio", "slope_diff"):
                values.append(f"{v:.4f}" if pd.notna(v) else "")
            elif col in ("volume", "avg_volume_5"):
                values.append(f"{v:,.0f}" if pd.notna(v) else "")
            else:
                values.append("" if pd.isna(v) else str(v))
        tree.insert("", "end", values=values)

    # 状态栏
    counts = df["signal_type"].value_counts().to_dict() if "signal_type" in df.columns else {}
    status = tk.Label(
        root,
        text=f"共 {len(df)} 行 | Confirmed={counts.get('confirmed', 0)} | "
             f"Predicting={counts.get('predicting', 0)} | Forming={counts.get('forming', 0)}",
        bd=1,
        relief=tk.SUNKEN,
        anchor=tk.W,
    )
    status.pack(side="bottom", fill="x")

    root.mainloop()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python show_result_gui.py <result_csv_path>")
        sys.exit(1)
    show_csv(sys.argv[1])
