"""Build the final submission PDFs from the Markdown reports (single source of truth).

    reports/business_insights_report.md  →  reports/business_insights_report.pdf   (Business Solution Document, ≥ 5 pages)
    reports/business_report.md           →  reports/business_report.pdf            (Business Report, 4–6 pages)

The Markdown files are only READ. In memory the text is converted to HTML with:
    * a cover page (team details) and a table of contents,
    * every chart the report references (`figures/NN_*.png`) inserted right after the block that first
      mentions it — the report text itself is not changed,
    * nested-list indentation normalised for the Markdown renderer (rendering only).
Headless Chrome (DevTools protocol) prints A4 PDFs with "Page X of Y" footers. Page counts and content
are verified with pypdf.

Run from the project root::

    .venv\\Scripts\\python.exe -m src.build_pdfs
"""

from __future__ import annotations

import asyncio
import base64
import json
import re
import shutil
import subprocess
import tempfile
import time
import urllib.request
from datetime import date
from pathlib import Path

import markdown
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "reports"
TEAM = {"name": "Genesis Grid", "group": "Group 4",
        "members": ["Manda Lohitha", "Pratik Jankiram Pattewar", "K Bonishira Akshaya", "Yateesh Chandra Maram", "Sanapalli Vikas"]}
DOCS = [
    {"md": "business_insights_report.md", "pdf": "business_insights_report.pdf",
     "title": "Business Insights Report", "subtitle": "Business Solution Document", "toc_levels": (2, 3),
     "toc_own_page": True, "img_height_mm": 105, "font_pt": 10, "pages": (5, None)},
    {"md": "business_report.md", "pdf": "business_report.pdf",
     "title": "Business Report", "subtitle": "Concise management report", "toc_levels": (2,),
     "toc_own_page": False, "img_height_mm": 56, "font_pt": 9.5, "pages": (4, 6)},
]
CHROME_CANDIDATES = [Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
                     Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")]

CSS = """
@page { size: A4; margin: 16mm 15mm 18mm 15mm; }
body { font-family: 'Segoe UI', Calibri, Arial, sans-serif; font-size: FONTpt; line-height: 1.42; color: #1d1d1b; }
h1 { font-size: 18pt; color: #123f78; border-bottom: 2px solid #2a78d6; padding-bottom: 4px; margin-top: 0; }
h2 { font-size: 14pt; color: #123f78; border-bottom: 1px solid #c9dcf3; padding-bottom: 3px; margin-top: 18px;
     page-break-after: avoid; break-after: avoid; }
h3 { font-size: 11.5pt; color: #1f5fae; margin-top: 14px; page-break-after: avoid; break-after: avoid; }
p, li { orphans: 3; widows: 3; }
table { border-collapse: collapse; width: 100%; margin: 8px 0 12px; font-size: 0.86em; }
th { background: #e8f0fb; color: #123f78; text-align: left; }
th, td { border: 1px solid #c8cdd3; padding: 3px 5px; vertical-align: top; overflow-wrap: anywhere; }
tr { page-break-inside: avoid; break-inside: avoid; }
code { font-family: Consolas, 'Courier New', monospace; font-size: 0.9em; background: #f2f4f7; padding: 0 2px; border-radius: 2px;
       overflow-wrap: anywhere; }
pre { background: #f2f4f7; padding: 8px; border-radius: 4px; white-space: pre-wrap; overflow-wrap: anywhere; font-size: 0.85em; }
pre code { background: none; }
blockquote { border-left: 4px solid #2a78d6; margin: 8px 0; padding: 4px 10px; background: #f3f7fd; }
figure { margin: 8px 0 14px; text-align: center; page-break-inside: avoid; break-inside: avoid; }
figure img { max-width: 100%; max-height: IMGmm; object-fit: contain; }
figcaption { font-size: 0.82em; color: #52514e; margin-top: 3px; }
.cover { height: 245mm; display: flex; flex-direction: column; justify-content: center; page-break-after: always; break-after: page; }
.cover.compact { height: auto; justify-content: flex-start; page-break-after: auto; break-after: auto; margin-bottom: 10px; }
.cover .brand { font-size: 12pt; letter-spacing: 2px; color: #2a78d6; font-weight: 600; }
.cover .t1 { font-size: 30pt; font-weight: 700; color: #123f78; margin: 6px 0; }
.cover .t2 { font-size: 16pt; color: #1f5fae; margin-bottom: 4px; }
.cover .t3 { font-size: 11pt; color: #52514e; margin-bottom: 18px; }
.cover table { width: 75%; font-size: 10pt; }
.toc { page-break-after: always; break-after: page; }
.toc ul { list-style: none; padding-left: 0; } .toc ul ul { padding-left: 18px; }
.toc li { margin: 2px 0; } .toc a { color: #1d1d1b; text-decoration: none; }
.cover .toc { page-break-after: auto; margin-top: 14px; }
"""


def chrome() -> Path:
    for c in CHROME_CANDIDATES:
        if c.exists():
            return c
    raise FileNotFoundError("Google Chrome or Microsoft Edge is required to print PDFs")


def caption(fig: str) -> str:
    stem = Path(fig).stem
    num, _, rest = stem.partition("_")
    return f"Figure {int(num)} — {rest.replace('_', ' ').capitalize()} (source: {fig})"


def prepare_markdown(text: str) -> tuple[str, list[str]]:
    """In-memory rendering copy: normalise nested lists and insert each referenced chart once."""
    lines = text.split("\n")
    # python-markdown needs 4-space nesting; the reports use 2 spaces (rendering-only change).
    lines = [re.sub(r"^( {2})(?=(?:[-*]|\d+\.) )", "    ", ln) for ln in lines]
    # A list item whose text starts with '>' (e.g. '- > 30 min …') would render as a quote: escape it.
    lines = [re.sub(r"^(\s*(?:[-*]|\d+\.) )>", r"\1&gt;", ln) for ln in lines]
    # python-markdown needs a blank line before a list that directly follows a paragraph line.
    is_item = lambda s: re.match(r"^\s*(?:[-*]|\d+\.) ", s) is not None
    fixed = []
    for ln in lines:
        prev = fixed[-1] if fixed else ""
        if is_item(ln) and prev.strip() and not is_item(prev) and not prev.lstrip().startswith(("|", "    ", "#")):
            fixed.append("")
        fixed.append(ln)
    lines = fixed
    out, pending, seen = [], [], []
    for ln in lines:
        if ln.strip() == "" and pending:
            for f in pending:
                out += ["", f"![{caption(f)}]({f})", ""]
            pending = []
        out.append(ln)
        for f in re.findall(r"figures/\d\d_[\w\-]+\.png", ln):
            if f not in seen and (ROOT / f).exists():
                seen.append(f)
                pending.append(f)
    for f in pending:
        out += ["", f"![{caption(f)}]({f})", ""]
    return "\n".join(out), seen


def build_html(doc: dict) -> tuple[str, list[str], list[str]]:
    src = (REP / doc["md"]).read_text(encoding="utf-8")
    md_text, figs = prepare_markdown(src)
    md = markdown.Markdown(extensions=["tables", "fenced_code", "sane_lists", "toc"],
                           extension_configs={"toc": {"toc_depth": "2-3"}})
    body = md.convert(md_text)
    # turn standalone images into captioned figures
    body = re.sub(r'<p><img alt="([^"]*)" src="([^"]+)" ?/?></p>',
                  r'<figure><img src="\2" alt="\1"/><figcaption>\1</figcaption></figure>', body)
    headings = []

    def toc_items(tokens):
        items = []
        for t in tokens:
            if t["level"] in doc["toc_levels"]:
                headings.append(t["name"])
                sub = toc_items(t["children"])
                items.append(f'<li><a href="#{t["id"]}">{t["name"]}</a>{"<ul>" + sub + "</ul>" if sub else ""}</li>')
            else:
                items.append(toc_items(t["children"]))
        return "".join(items)

    toc = f'<div class="toc"><h2>Contents</h2><ul>{toc_items(md.toc_tokens)}</ul></div>'
    members = "".join(f"<tr><td>{i}</td><td>{n}</td></tr>" for i, n in enumerate(TEAM["members"], 1))
    cover_cls = "cover" if doc["toc_own_page"] else "cover compact"
    cover = (f'<div class="{cover_cls}"><div class="brand">KLU + PATH3 · ML-01 · QUICK-COMMERCE DELIVERY INTELLIGENCE</div>'
             f'<div class="t1">DeliveryIQ</div><div class="t2">{doc["title"]} — {doc["subtitle"]}</div>'
             f'<div class="t3">AI-Powered Quick Commerce Delivery Intelligence System · {date.today():%d %B %Y}</div>'
             f'<table><tr><th>Team</th><td>{TEAM["name"]}</td></tr><tr><th>Group</th><td>{TEAM["group"]}</td></tr></table>'
             f'<table><tr><th>S. No.</th><th>Team member</th></tr>{members}</table>'
             f'<p style="font-size:9pt;color:#52514e">Source: <code>reports/{doc["md"]}</code> (unchanged). '
             f'Late = delivery &gt; 30 minutes is a project-defined benchmark, not an official SLA.</p>'
             f'{"" if doc["toc_own_page"] else toc}</div>')
    css = CSS.replace("FONT", str(doc["font_pt"])).replace("IMG", str(doc["img_height_mm"]))
    html = (f'<!doctype html><html><head><meta charset="utf-8"><base href="{ROOT.as_uri()}/">'
            f'<title>DeliveryIQ — {doc["title"]}</title><style>{css}</style></head><body>'
            f'{cover}{toc if doc["toc_own_page"] else ""}{body}</body></html>')
    return html, figs, headings


async def _print(ws_url: str, html_path: Path, pdf_path: Path, title: str) -> None:
    from tornado.websocket import websocket_connect
    ws = await websocket_connect(ws_url, max_message_size=500 * 1024 * 1024)
    counter = {"id": 0}

    async def cmd(method, **params):
        counter["id"] += 1
        mid = counter["id"]
        await ws.write_message(json.dumps({"id": mid, "method": method, "params": params}))
        while True:
            msg = json.loads(await ws.read_message())
            if msg.get("id") == mid:
                if "error" in msg:
                    raise RuntimeError(msg["error"])
                return msg.get("result", {})

    await cmd("Page.enable")
    await cmd("Page.navigate", url=html_path.as_uri())
    for _ in range(60):                                   # wait until every image has loaded
        r = await cmd("Runtime.evaluate", returnByValue=True,
                      expression="document.readyState==='complete' && [...document.images].every(i=>i.complete && i.naturalWidth>0)")
        if r.get("result", {}).get("value"):
            break
        await asyncio.sleep(0.5)
    else:
        raise RuntimeError("images did not load")
    foot = ('<div style="font-size:8px;width:100%;padding:0 15mm;color:#6b6b6b;display:flex;justify-content:space-between;">'
            f'<span>DeliveryIQ · {title} · Team Genesis Grid (Group 4)</span>'
            '<span>Page <span class="pageNumber"></span> of <span class="totalPages"></span></span></div>')
    r = await cmd("Page.printToPDF", printBackground=True, preferCSSPageSize=True, displayHeaderFooter=True,
                  headerTemplate="<div></div>", footerTemplate=foot)
    pdf_path.write_bytes(base64.b64decode(r["data"]))
    ws.close()


def print_pdf(html: str, pdf_path: Path, title: str) -> None:
    tmp = Path(tempfile.mkdtemp(prefix="deliveryiq_pdf_"))
    html_path = tmp / "report.html"
    html_path.write_text(html, encoding="utf-8")
    proc = subprocess.Popen([str(chrome()), "--headless=new", "--disable-gpu", "--allow-file-access-from-files",
                             "--remote-debugging-port=9335", f"--user-data-dir={tmp / 'profile'}", "about:blank"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(60):
            try:
                pages = json.load(urllib.request.urlopen("http://localhost:9335/json/list"))
                break
            except Exception:
                time.sleep(0.3)
        ws_url = [p for p in pages if p["type"] == "page"][0]["webSocketDebuggerUrl"]
        asyncio.run(_print(ws_url, html_path, pdf_path, title))
    finally:
        proc.terminate()
        proc.wait(timeout=20)
        shutil.rmtree(tmp, ignore_errors=True)


def verify(doc: dict, figs: list[str], headings: list[str]) -> dict:
    """Page count, embedded images and that every heading/key number survived into the PDF text."""
    reader = PdfReader(REP / doc["pdf"])
    n = len(reader.pages)
    text = re.sub(r"\s+", " ", " ".join(p.extract_text() or "" for p in reader.pages))
    images = sum(len(p.images) for p in reader.pages)
    norm = lambda s: re.sub(r"\s+", " ", re.sub(r"[*`]", "", s)).strip()
    missing_headings = [h for h in headings if norm(h)[:40] not in text]
    src = (REP / doc["md"]).read_text(encoding="utf-8")
    numbers = sorted(set(re.findall(r"\b\d{1,3}(?:,\d{3})*\.\d+\b", src)))
    missing_numbers = [x for x in numbers if x not in text]
    lo, hi = doc["pages"]
    ok_pages = n >= lo and (hi is None or n <= hi)
    return {"pdf": f"reports/{doc['pdf']}", "pages": n, "page_requirement": f"{lo}–{hi}" if hi else f"≥ {lo}",
            "pages_ok": ok_pages, "figures_inserted": len(figs), "images_in_pdf": images,
            "headings_checked": len(headings), "missing_headings": missing_headings,
            "decimal_numbers_checked": len(numbers), "missing_numbers": missing_numbers[:20],
            "ok": ok_pages and images >= len(figs) and not missing_headings and not missing_numbers}


def main() -> None:
    results = []
    for doc in DOCS:
        before = (REP / doc["md"]).read_bytes()
        html, figs, headings = build_html(doc)
        print_pdf(html, REP / doc["pdf"], doc["title"])
        assert (REP / doc["md"]).read_bytes() == before, "Markdown source must not change"
        res = verify(doc, figs, headings)
        results.append(res)
        print(json.dumps(res, ensure_ascii=False))
    (REP / "pdf_build_check.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    if not all(r["ok"] for r in results):
        raise SystemExit("PDF verification failed")


if __name__ == "__main__":
    main()
