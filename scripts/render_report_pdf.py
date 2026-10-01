"""Render reports/benchmark-report.md + plots into an A4 PDF."""

from __future__ import annotations

import re
from pathlib import Path

from fpdf import FPDF

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
MD = REPORTS / "benchmark-report.md"
OUT = REPORTS / "benchmark-report.pdf"


class ReportPDF(FPDF):
    def header(self) -> None:
        if self.page_no() == 1:
            self.set_y(10)
            return
        self.set_y(8)
        self.set_font("Helvetica", "I", 9)
        self.set_text_color(90, 90, 90)
        self.cell(0, 6, "SmartPark Benchmark Report  |  FIT3184 A1  |  16 Sep 2026", align="C")
        self.ln(6)
        self.set_text_color(20, 20, 20)

    def footer(self) -> None:
        self.set_y(-14)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(90, 90, 90)
        self.cell(0, 8, f"Page {self.page_no()}/{{nb}}", align="C")


def _latin(text: str) -> str:
    replacements = {
        "\u2014": "-",
        "\u2013": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2192": "->",
        "\u00d7": "x",
        "\u2248": "~",
        "`": "",
        "**": "",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text.encode("latin-1", "replace").decode("latin-1")


def _parse_table(block: str) -> list[list[str]]:
    rows = []
    for line in block.strip().splitlines():
        if re.match(r"^\|?\s*-+", line):
            continue
        cells = [c.strip().replace("**", "") for c in line.strip().strip("|").split("|")]
        if cells:
            rows.append(cells)
    return rows


def _word_count(text: str) -> int:
    cleaned = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)
    cleaned = re.sub(r"[#|*`>-]", " ", cleaned)
    return len(re.findall(r"[A-Za-z0-9][A-Za-z0-9$%./:_-]*", cleaned))


def _draw_table(pdf: ReportPDF, rows: list[list[str]]) -> None:
    cols = len(rows[0])
    usable = pdf.w - pdf.l_margin - pdf.r_margin
    col_w = usable / cols
    row_h = 5.6
    needed = row_h * len(rows) + 4
    if pdf.will_page_break(needed):
        pdf.add_page()
    for i, row in enumerate(rows):
        bold = i == 0 or any("Total" in c for c in row)
        pdf.set_font("Helvetica", "B" if bold else "", 8)
        if i == 0:
            pdf.set_fill_color(230, 230, 230)
        elif bold:
            pdf.set_fill_color(245, 245, 245)
        else:
            pdf.set_fill_color(255, 255, 255)
        for cell in row:
            pdf.cell(col_w, row_h, _latin(cell), border=1, fill=True)
        pdf.ln()
    pdf.ln(2)


def main() -> None:
    md = MD.read_text(encoding="utf-8")
    print("markdown word count:", _word_count(md))

    pdf = ReportPDF(format="A4", unit="mm")
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.add_page()
    pdf.set_text_color(20, 20, 20)

    paragraphs = md.split("\n\n")
    for block in paragraphs:
        block = block.strip()
        if not block:
            continue
        if block.startswith("# "):
            pdf.set_font("Helvetica", "B", 16)
            pdf.multi_cell(0, 8, _latin(block[2:]))
            pdf.ln(1)
            continue
        if block.startswith("## "):
            pdf.ln(1.5)
            pdf.set_font("Helvetica", "B", 12.5)
            pdf.multi_cell(0, 6.5, _latin(block[3:]))
            pdf.ln(0.5)
            continue
        if "![" in block:
            continue
        if block.startswith("|"):
            rows = _parse_table(block)
            if rows:
                _draw_table(pdf, rows)
            continue
        pdf.set_font("Helvetica", "", 9.5)
        text = _latin(block)
        lines = pdf.multi_cell(0, 4.6, text, dry_run=True, output="LINES")
        needed = 4.6 * max(len(lines), 1) + 1.0
        if pdf.will_page_break(needed):
            pdf.add_page()
        pdf.multi_cell(0, 4.6, text)
        pdf.ln(1.0)

    pdf.add_page()
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(20, 20, 20)
    pdf.cell(0, 7, "Figure 1. Latency vs concurrent users")
    pdf.ln(8)
    pdf.image(str(REPORTS / "latency_vs_users.png"), x=18, w=174)
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 7, "Figure 2. Throughput vs concurrent users")
    pdf.ln(8)
    pdf.image(str(REPORTS / "throughput_vs_users.png"), x=18, w=174)

    pdf.output(OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
