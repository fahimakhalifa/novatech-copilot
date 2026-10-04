"""Convert guard_classifier.py (with "# %%" cell markers) into a Colab notebook.

The .py file is the source of truth: it can be run and smoke-tested locally, diffed and
reviewed. The .ipynb is generated from it for Colab. Run from the repo root:
  python training/make_notebook.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC, OUT = HERE / "guard_classifier.py", HERE / "guard_classifier.ipynb"


def cells(source: str) -> list[dict]:
    result, kind, lines = [], None, []

    def flush():
        text = "\n".join(lines).strip("\n")
        if kind and text:
            if kind == "markdown":
                text = "\n".join(line[2:] if line.startswith("# ") else line.lstrip("#")
                                 for line in text.splitlines())
            body = [line + "\n" for line in text.splitlines()]
            body[-1] = body[-1].rstrip("\n")
            cell = {"cell_type": kind, "metadata": {}, "source": body}
            if kind == "code":
                cell.update(execution_count=None, outputs=[])
            result.append(cell)

    for line in source.splitlines():
        if line.startswith("# %%"):
            flush()
            kind, lines = ("markdown" if "[markdown]" in line else "code"), []
        else:
            lines.append(line)
    flush()
    return result


notebook = {
    "cells": cells(SRC.read_text(encoding="utf-8")),
    "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                 "kernelspec": {"display_name": "Python 3", "name": "python3"}},
    "nbformat": 4, "nbformat_minor": 0,
}
OUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"wrote {OUT.name}: {len(notebook['cells'])} cells")
