"""Turn each lesson's docstring into docs/NN_name.md and build a self-contained gallery.html (theory + animations)."""
import ast, html, inspect, pathlib, re

ROOT = pathlib.Path(__file__).resolve().parent; OUT = ROOT / "outputs"; DOCS = ROOT / "docs"; NOTEBOOKS = ROOT / "notebooks"; DOCS.mkdir(exist_ok=True)
PARTS = [("Part 0 · Foundations", 0, 2), ("Part 1 · Classical supervised learning", 3, 14), ("Part 2 · Neural networks & deep learning", 15, 19),
         ("Part 3 · Generative models & representation learning", 20, 22), ("Part 4 · Unsupervised learning", 23, 31), ("Part 5 · Sequences, probability, recommenders, forecasting & RL", 32, 36)]


def inline(s):
    s = html.escape(s); s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s); return re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", s)


def md_to_html(text):
    out, para, lst, code = [], [], False, []
    def flush():
        nonlocal para
        if para: out.append("<p>" + inline(" ".join(para)) + "</p>"); para = []
    def endlist():
        nonlocal lst
        if lst: out.append("</ul>"); lst = False
    for line in text.split("\n"):
        if line.startswith("    ") or (code and not line.strip()):
            flush(); endlist(); code.append(line[4:] if line.startswith("    ") else ""); continue
        if code: out.append("<pre>" + html.escape("\n".join(code).rstrip()) + "</pre>"); code = []
        if line.startswith("## "): flush(); endlist(); out.append(f"<h3>{inline(line[3:])}</h3>")
        elif line.startswith("# "): continue
        elif line.startswith("- "):
            flush()
            if not lst: out.append("<ul>"); lst = True
            out.append(f"<li>{inline(line[2:])}</li>")
        elif not line.strip(): flush(); endlist()
        else: para.append(line.strip())
    if code: out.append("<pre>" + html.escape("\n".join(code).rstrip()) + "</pre>")
    flush(); endlist(); return "\n".join(out)


lessons = []
for p in sorted((ROOT / "lessons").glob("[0-9][0-9]_*.py")):
    doc = inspect.cleandoc(ast.get_docstring(ast.parse(p.read_text())) or ""); title = doc.split("\n")[0].lstrip("# ").strip(); fam = re.search(r"\*\*Family:\*\*\s*(.+)", doc); stem = p.stem
    media = sorted(f.name for f in OUT.glob(stem + "*") if f.suffix in (".gif", ".png")); lessons.append(dict(num=int(stem[:2]), stem=stem, title=title, fam=fam.group(1) if fam else "", doc=doc, media=media, file=p.name))
    md = f"{doc}\n\n## Formats\n- **Interactive Notebook:** [`notebooks/{stem}.ipynb`](../notebooks/{stem}.ipynb)\n- **Standalone Python Script:** [`lessons/{p.name}`](../lessons/{p.name})\n\n" + "".join(f"![{m}](../outputs/{m})\n\n" for m in media); (DOCS / f"{stem}.md").write_text(md)

idx = ["# ML Model Atlas · lesson index\n"]
for name, a, b in PARTS:
    idx.append(f"\n## {name}\n"); idx += [f"- [{l['title']}]({l['stem']}.md): {l['fam']} ([Notebook](../notebooks/{l['stem']}.ipynb))" for l in lessons if a <= l["num"] <= b]
(DOCS / "INDEX.md").write_text("\n".join(idx) + "\n")

nav, body = [], []
for name, a, b in PARTS:
    nav.append(f"<h4>{html.escape(name)}</h4>")
    for l in lessons:
        if a <= l["num"] <= b:
            nav.append(f"<a href='#{l['stem']}'>{html.escape(l['title'])}</a>")
            imgs = "".join(f"<img loading='lazy' src='outputs/{m}' alt='{m}'>" for m in l["media"]) or "<p><i>Run the lesson to generate its animation.</i></p>"
            links_html = f"<div class='fam'>{html.escape(l['fam'])} · <a href='notebooks/{l['stem']}.ipynb'><code>notebooks/{l['stem']}.ipynb</code></a> · <code>lessons/{l['file']}</code></div>"
            body.append(f"<section id='{l['stem']}'><h2>{html.escape(l['title'])}</h2>{links_html}<div class='media'>{imgs}</div>{md_to_html(l['doc'])}</section>")
CSS = """:root{--bg:#fff;--fg:#1d2330;--mut:#667;--card:#f5f6f9;--ac:#3b6fb6}@media(prefers-color-scheme:dark){:root{--bg:#14171d;--fg:#e6e8ee;--mut:#9aa3b2;--card:#1d222b;--ac:#7aa7e6}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.6 system-ui,sans-serif;display:flex}nav{width:300px;flex:none;height:100vh;overflow:auto;position:sticky;top:0;padding:16px;background:var(--card);font-size:13px}
nav h4{margin:16px 0 4px;color:var(--mut);text-transform:uppercase;font-size:11px;letter-spacing:.05em}nav a{display:block;padding:3px 0;color:var(--fg);text-decoration:none}nav a:hover{color:var(--ac)}
main{max-width:1000px;padding:24px 40px;min-width:0}section{padding:28px 0;border-bottom:1px solid var(--card)}h2{margin:0 0 4px}h3{margin:22px 0 6px;color:var(--ac)}.fam{color:var(--mut);font-size:13px;margin-bottom:12px}
.media img{max-width:100%;border-radius:8px;border:1px solid var(--card);margin:6px 0;display:block;background:#fff}pre{background:var(--card);padding:12px;border-radius:8px;overflow:auto;font-size:13px}code{background:var(--card);padding:1px 5px;border-radius:4px}pre code{padding:0}
@media(max-width:800px){body{display:block}nav{width:auto;height:auto;position:static}main{padding:16px}}"""
(ROOT / "gallery.html").write_text(f"<!doctype html><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>ML Model Atlas</title><style>{CSS}</style><nav><b style='font-size:16px'>ML Model Atlas</b><div style='color:var(--mut)'>Every model: theory · code · animation · notebook</div>{''.join(nav)}</nav><main><h1>ML Model Atlas</h1><p>Start anywhere: each lesson explains the model, shows how it learns as an animation, links to the interactive Jupyter notebook, and provides from-scratch NumPy code. Run <code>python run_all.py</code> to regenerate the animations.</p>{''.join(body)}</main>")
print(f"built docs/ ({len(lessons)} lessons) and gallery.html with notebook links")
