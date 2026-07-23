"""Render the Markdown report to a self-contained HTML (base64 figures) and
then to PDF via the pre-installed Chromium (headless --print-to-pdf)."""
import os, re, base64, subprocess
import markdown
import geo_common as gc

REP = gc.p(gc.ROOT, "report")
md_path = gc.p(REP, "ЗВІТ_зонування_П1_Софіївка.md")
html_path = gc.p(REP, "report.html")
pdf_path = gc.p(REP, "ЗВІТ_зонування_П1_Софіївка.pdf")
FIGDIR = gc.p(gc.ROOT, "figures")

md = open(md_path, encoding="utf-8").read()

def embed(m):
    alt, path = m.group(1), m.group(2)
    fn = os.path.basename(path)
    fp = os.path.join(FIGDIR, fn)
    if os.path.exists(fp):
        b = base64.b64encode(open(fp, "rb").read()).decode()
        return f'<figure><img src="data:image/png;base64,{b}"/><figcaption>{alt}</figcaption></figure>'
    return m.group(0)

md = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", embed, md)
body = markdown.markdown(md, extensions=["tables", "fenced_code", "toc"])

html = f"""<!doctype html><html lang="uk"><head><meta charset="utf-8">
<style>
@page {{ size: A4; margin: 16mm 14mm; }}
body {{ font-family: 'DejaVu Sans', 'Liberation Sans', Arial, sans-serif;
        font-size: 10.5pt; line-height: 1.5; color: #1a1a1a; }}
h1 {{ font-size: 19pt; color: #14532d; border-bottom: 3px solid #14532d;
      padding-bottom: 6px; }}
h2 {{ font-size: 14pt; color: #166534; margin-top: 22px;
      border-bottom: 1px solid #ccc; padding-bottom: 3px; }}
h3 {{ font-size: 12pt; color: #333; }}
table {{ border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 9.5pt; }}
th, td {{ border: 1px solid #bbb; padding: 4px 7px; text-align: left; }}
th {{ background: #e8f3ea; }}
tr:nth-child(even) td {{ background: #f7faf7; }}
figure {{ margin: 12px 0; text-align: center; page-break-inside: avoid; }}
img {{ max-width: 100%; border: 1px solid #ddd; }}
figcaption {{ font-size: 8.5pt; color: #666; font-style: italic; margin-top: 3px; }}
code, pre {{ font-family: 'DejaVu Sans Mono', monospace; font-size: 8.5pt; }}
pre {{ background: #f4f4f4; padding: 10px; border-radius: 4px; overflow-x: auto;
       page-break-inside: avoid; }}
blockquote {{ border-left: 4px solid #86c28c; margin: 10px 0; padding: 4px 14px;
              background: #f2f8f3; color: #234; }}
</style></head><body>{body}</body></html>"""

open(html_path, "w", encoding="utf-8").write(html)
print("HTML:", html_path)

chrome = "/opt/pw-browsers/chromium-1194/chrome-linux/chrome"
subprocess.run([chrome, "--headless", "--no-sandbox", "--disable-gpu",
                "--no-pdf-header-footer",
                f"--print-to-pdf={pdf_path}", "file://" + html_path],
               check=True, capture_output=True)
print("PDF:", pdf_path, os.path.getsize(pdf_path), "bytes")
