"""Export docs/SYSTEM_REPORT_EN.md to docs/SYSTEM_REPORT_EN.pdf (PyMuPDF, no LaTeX or browser needed).

    conda run -n hongyi python docs/export_pdf.py

Markdown -> HTML with markdown-it, laid out with PyMuPDF's Story on A4 pages; adds bookmarks and page numbers.
The ASCII-art pipeline block is left out because the figure right above it shows the same thing.
"""
import re
import sys
from pathlib import Path

import fitz
from markdown_it import MarkdownIt
from PIL import Image

DOCS = Path(__file__).resolve().parent
SRC = DOCS / "SYSTEM_REPORT_EN.md"
OUT = DOCS / "SYSTEM_REPORT_EN.pdf"
PAGE_W = 595.0
MARGIN_X, MARGIN_TOP, MARGIN_BOTTOM = 48, 52, 58
TEXT_W = PAGE_W - 2 * MARGIN_X

CSS = """
body { font-family: sans-serif; font-size: 9.2pt; line-height: 1.28; }
h1 { font-size: 17pt; margin-top: 0; margin-bottom: 6pt; }
h2 { font-size: 13.5pt; margin-top: 14pt; margin-bottom: 4pt; border-bottom: 0.6pt solid #888; }
h3 { font-size: 11pt; margin-top: 9pt; margin-bottom: 3pt; }
h4 { font-size: 10pt; margin-top: 9pt; margin-bottom: 3pt; page-break-before: always; }
p { margin-top: 3pt; margin-bottom: 4pt; }
li { margin-top: 1.5pt; margin-bottom: 1.5pt; }
blockquote { margin-left: 8pt; padding-left: 6pt; border-left: 2pt solid #bbb; color: #333; }
span.c { font-family: monospace; font-size: 8pt; color: #222; }
pre { font-family: monospace; font-size: 7pt; background-color: #f4f4f4; padding: 4pt; white-space: pre-wrap; }
table { border-collapse: collapse; width: 100%; margin-top: 4pt; margin-bottom: 6pt; }
th { font-size: 7.4pt; font-weight: bold; border: 0.5pt solid #999; padding: 2pt; text-align: left; }
td { font-size: 7.4pt; border: 0.5pt solid #bbb; padding: 2pt; }
td span.c, th span.c { font-size: 6.8pt; }
img { margin-top: 4pt; margin-bottom: 2pt; }
hr { border: 0.4pt solid #bbb; }
"""


def prepare_markdown(text):
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    text = re.sub(r"```\nRaw JSON threads.*?```\n", "", text, flags=re.S)  # ASCII art, see the figure above it
    text = text.replace(r"$1-\frac{\|a-b\|}{a+b+\epsilon}$", "1 - |a - b| / (a + b + ε)")
    return text


def sized_images(html):
    def repl(m):
        src = m.group(1)
        w, _ = Image.open(DOCS / src).size
        width = min(TEXT_W, w / 150 * 72)  # figures are saved at 150 dpi
        return f'<img src="{src}" width="{width:.0f}"'
    return re.sub(r'<img src="([^"]+)"', repl, html)


def shrink_wide_tables(html, min_columns=9, size="5.7pt"):
    """Tables with many columns (result tables) get a smaller font so that they fit the page width."""
    def repl(m):
        block = m.group(0)
        if block.count("<th") < min_columns:
            return block
        block = block.replace("<th>", f'<th style="font-size:{size}; padding:1pt">')
        block = block.replace("<td>", f'<td style="font-size:{size}; padding:1pt">')
        return block.replace('<span class="c">', f'<span class="c" style="font-size:{size}">')
    return re.sub(r"<table>.*?</table>", repl, html, flags=re.S)


def inline_code_as_span(html):
    """Inline code gets a plain span: the default <code> style paints grey streaks in PyMuPDF's layout."""
    html = html.replace("<pre><code", "<pre><@@code").replace("</code></pre>", "</@@code></pre>")
    html = html.replace("<code>", '<span class="c">').replace("</code>", "</span>")
    return html.replace("<@@code", "<code").replace("</@@code>", "</code>")


def main():
    md = MarkdownIt("commonmark").enable("table")
    html = shrink_wide_tables(inline_code_as_span(sized_images(md.render(prepare_markdown(SRC.read_text(encoding="utf-8"))))))
    story = fitz.Story(html=html, user_css=CSS, em=10, archive=fitz.Archive(str(DOCS)))
    mediabox = fitz.paper_rect("a4")
    where = fitz.Rect(MARGIN_X, MARGIN_TOP, PAGE_W - MARGIN_X, mediabox.height - MARGIN_BOTTOM)
    headings = []

    def rectfn(rect_num, filled):
        return mediabox, where, None

    def positionfn(position):
        if position.heading and position.open_close & 1 and position.heading <= 3:
            headings.append([position.heading, position.text or "", position.page_num])

    tmp = OUT.with_suffix(".tmp.pdf")
    writer = fitz.DocumentWriter(str(tmp))
    story.write(writer, rectfn, positionfn)
    writer.close()

    doc = fitz.open(str(tmp))
    toc, last = [], 0
    for level, title, page in headings:
        level = min(level, last + 1) if last else 1
        toc.append([level, title.strip() or "(untitled)", page])
        last = level
    if toc:
        doc.set_toc(toc)
    total = len(doc)
    for i, page in enumerate(doc):
        page.insert_text((PAGE_W / 2 - 14, page.rect.height - 28), f"{i + 1} / {total}", fontsize=8,
                         fontname="helv", color=(0.35, 0.35, 0.35))
    doc.set_metadata({"title": "Early Depression Detection from User Post Sequences (eRisk 2025 Task 2)",
                      "subject": "System design, experiments and findings"})
    doc.save(str(OUT), garbage=3, deflate=True)
    doc.close()
    tmp.unlink()
    print(f"saved {OUT} ({total} pages, {len(toc)} bookmarks)")


if __name__ == "__main__":
    sys.exit(main())
