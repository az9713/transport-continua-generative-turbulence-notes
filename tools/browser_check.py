"""Headless browser check of every public page; writes reader-validation.json.

Opens each page at 1440 and 390 pixels with the installed Chrome (Playwright,
channel chrome) from the local files or from --base URL, waits for MathJax, and
records math errors, leftover TeX, console errors, failed requests, page-wide
horizontal overflow, broken same-page anchors and missing local link targets.
On the first chapter it also operates the reader controls.
Run: python tools/browser_check.py [--base https://...] [--output FILE]
"""
import argparse
import json
from pathlib import Path
from urllib.parse import urlsplit
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument('--base', help='site URL ending in /; default local files')
ap.add_argument('--output', default='reader-validation.json')
args = ap.parse_args()
manifest = json.loads((ROOT / 'source-manifest.json').read_text('utf-8'))
chapters = [c['file'] for c in manifest['chapters']]
pages = ['index.html', 'book-overview.html', 'source-map.html',
         'references.html', 'numerical-lab.html',
         'development-journey.html'] + chapters
SCAN = '''() => {
  const root = document.documentElement;
  const ids = new Set([...document.querySelectorAll('[id]')].map(e => e.id));
  const links = [...document.querySelectorAll('a[href]')]
    .map(a => a.getAttribute('href'));
  const article = document.getElementById('chapter-content');
  const text = article.cloneNode(true);
  text.querySelectorAll('pre,code,mjx-container,script,style')
    .forEach(e => e.remove());
  return {math: document.querySelectorAll('mjx-container').length,
    errors: document.querySelectorAll('mjx-merror').length,
    dollars: (text.textContent.match(/\\$/g) || []).length,
    overflow: root.scrollWidth - window.innerWidth,
    dead: links.filter(h => h.startsWith('#') && h.length > 1
                        && !ids.has(h.slice(1))),
    local: links.filter(h => !/^(#|https?:|mailto:)/.test(h))};
}'''
report, issues = [], []


def url_of(name):
    return args.base + name if args.base else (ROOT / name).as_uri()


with sync_playwright() as p:
    browser = p.chromium.launch(channel='chrome', headless=True)
    for width, height in ((1440, 900), (390, 844)):
        context = browser.new_context(viewport={'width': width,
                                                'height': height})
        for name in pages:
            page = context.new_page()
            noise = []
            page.on('console', lambda m: noise.append(m.text)
                    if m.type == 'error' else None)
            page.on('pageerror', lambda e: noise.append(str(e)))
            page.on('requestfailed', lambda r: noise.append(
                'failed ' + r.url))
            page.goto(url_of(name), wait_until='load')
            page.wait_for_function(
                "document.documentElement.dataset.mathReady==='true'",
                timeout=120000)
            got = page.evaluate(SCAN)
            missing = [h for h in got['local'] if not (
                ROOT / urlsplit(h).path).exists()]
            row = dict(page=name, width=width, rendered_math=got['math'],
                       math_errors=got['errors'], leftover_dollars=got[
                           'dollars'], overflow_px=got['overflow'],
                       dead_anchors=got['dead'], missing_local=missing,
                       errors=noise)
            bad = (got['errors'] or got['dollars'] or got['overflow'] > 1
                   or got['dead'] or missing or noise)
            if bad:
                issues.append(row)
            if width == 1440:
                row['body_sha256'] = page.evaluate(
                    "document.querySelector('meta[name=book-content-sha256]')"
                    ".content")
                report.append(row)
            page.close()
        context.close()
    page = browser.new_page(viewport={'width': 1440, 'height': 900})
    page.goto(url_of(chapters[0]), wait_until='load')
    page.wait_for_function(
        "document.documentElement.dataset.mathReady==='true'", timeout=120000)
    theme = page.evaluate("document.body.classList.contains('light')")
    page.click('#theme-toggle')
    flipped = page.evaluate("document.body.classList.contains('light')")
    page.click('#solutions-toggle')
    opened = page.evaluate(
        "[...document.querySelectorAll('details')].every(d => d.open)")
    toc = page.evaluate(
        "document.querySelectorAll('#chapter-toc a').length")
    controls = dict(theme_before=theme, theme_after=flipped,
                    solutions_open=opened, toc_links=toc)
    if theme == flipped or not opened or not toc:
        issues.append(dict(page=chapters[0], controls=controls))
    browser.close()
result = dict(scope='Headless Chrome DOM, rendering and controls; no human '
              'visual or scientific certification',
              source=args.base or 'local files', widths=[1440, 390],
              passed=not issues, controls=controls, issues=issues,
              pages=report)
out = ROOT / args.output
out.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8',
               newline='\n')
print(f'{len(pages)} pages at 2 widths: {len(issues)} issue(s); {out.name}')
for row in issues:
    print(row)
