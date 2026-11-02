"""Rebuild public reader chrome from root chapter bodies and public metadata."""
from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path, PurePosixPath
import re
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
START = '<!-- BOOK-CONTENT-START -->'
END = '<!-- BOOK-CONTENT-END -->'


def digest(body):
    return hashlib.sha256(body.encode('utf-8')).hexdigest()


def read(path):
    # Preserve raw authored-body line endings in every digest/parity consumer.
    return path.read_bytes().decode('utf-8-sig')


def write(path, value):
    path.write_text(value, encoding='utf-8', newline='\n')


def esc(value):
    return html.escape(str(value), quote=True)


def public_path(name, suffix=None):
    """Reject traversal, absolute paths, private roots and symbolic escapes."""
    if not isinstance(name, str) or not name or '\\' in name or ':' in name:
        raise ValueError('Invalid public relative path')
    posix=PurePosixPath(name)
    if posix.as_posix()!=name or name=='.':
        raise ValueError('Public paths must use canonical POSIX spelling')
    relative = Path(name)
    private={'work','.git','__pycache__'}
    if relative.is_absolute() or '..' in relative.parts or any(p.lower() in private for p in relative.parts):
        raise ValueError('Path traversal')
    cursor=ROOT
    for part in relative.parts:
        cursor=cursor/part
        if cursor.is_symlink():
            raise ValueError('Symbolic public paths are not allowed')
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT) or any(p.lower() in private for p in path.relative_to(ROOT).parts):
        raise ValueError('Private or outside path')
    if suffix and path.suffix != suffix:
        raise ValueError('Incorrect suffix')
    return path


def safe_url(value):
    parts = urlsplit(value)
    if parts.scheme != 'https' or not parts.netloc or parts.username is not None or parts.password is not None:
        raise ValueError('Expected public HTTPS source URL')
    return esc(value)


def body_of(document):
    if START not in document and END not in document:
        return document
    if document.count(START) != 1 or document.count(END) != 1:
        raise ValueError('Ambiguous content markers')
    body = document.split(START, 1)[1].split(END, 1)[0]
    prior = re.search(r'name="book-content-sha256" content="([a-f0-9]{64})"',
                      document)
    if prior and prior[1] != digest(body):
        raise ValueError('Authored body changed; update author evidence first')
    return body


def page(title, body, chapter=False, previous=None, following=None):
    links = ''.join(f'<a href="{name}">{label}</a>' for name, label in [
        ('index.html', 'Chapters'), ('book-overview.html', 'Study route'),
        ('source-map.html', 'Source map'),
        ('references.html', 'References'), ('numerical-lab.html', 'Numerical lab')])
    turns = ''
    if chapter:
        back = (f'<a rel="prev" href="{esc(previous)}">Previous chapter</a>'
                if previous else '<span>First available chapter</span>')
        ahead = (f'<a rel="next" href="{esc(following)}">Next chapter</a>'
                 if following else '<span>Last available chapter</span>')
        turns = f'<nav class="page-turn">{back}<a href="index.html">Index</a>{ahead}</nav>'
    toc = '<aside class="toc" id="chapter-toc"><h2>On this page</h2></aside>' if chapter else ''
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="dark light"><meta name="book-content-sha256" content="{digest(body)}">
<title>{esc(title)} · Transport, Continua and Generative Turbulence</title><link rel="stylesheet" href="assets/book.css">
<script>window.MathJax={{tex:{{inlineMath:[['$','$'],['\\\\(','\\\\)']],displayMath:[['$$','$$'],['\\\\[','\\\\]']],processEscapes:true}},svg:{{fontCache:'local'}},options:{{skipHtmlTags:['script','noscript','style','textarea','pre','code']}},startup:{{pageReady:()=>MathJax.startup.defaultPageReady().then(()=>{{document.documentElement.dataset.mathReady='true';}})}}}};</script>
<script defer src="assets/mathjax-tex-svg.js"></script><script defer src="assets/book.js"></script></head>
<body><a class="skip" href="#main">Skip to content</a><header class="site-header"><a class="brand" href="index.html">Transport, Continua and Generative Models of Turbulent Flow</a><nav class="course-nav" aria-label="Book">{links}</nav></header>
<div class="layout{'' if chapter else ' no-toc'}">{toc}<main id="main"><div class="reading-tools"><button id="theme-toggle" aria-pressed="false">Light theme</button><button id="text-toggle" aria-pressed="false">Larger text</button><button id="solutions-toggle" aria-pressed="false">Expand solutions</button><button id="print-page">Print with solutions</button></div>
<article id="chapter-content">{START}{body}{END}</article>{turns}</main></div><footer class="site-footer">Independent synthesis course built from the author's NPTEL and MIT study editions and primary research papers. Source and review limitations are recorded separately.</footer></body></html>'''


def build():
    manifest = json.loads(read(ROOT / 'source-manifest.json'))
    chapters = manifest['chapters']
    names = [public_path(c['file'], '.html').name for c in chapters]
    references = {}
    rows, source_rows = [], []
    for i, record in enumerate(chapters):
        path = public_path(record['file'], '.html')
        body = body_of(read(path))
        meta = json.loads(read(path.with_suffix('.meta.json')))
        if digest(body) != meta['authored_body_sha256']:
            raise ValueError('Sidecar body digest mismatch: ' + path.name)
        write(path, page(record['title'], body, True,
                         names[i-1] if i else None,
                         names[i+1] if i+1 < len(names) else None))
        rows.append(f'<li data-lecture><h2><a href="{esc(path.name)}">{esc(record["title"])}</a></h2><p>{esc(record["status"])}</p></li>')
        for reference in meta.get('references', []):
            safe_url(reference['url'])
            references[reference['url']] = reference
        source_rows.extend(dict(s, chapter=path.name) for s in meta['sections'])
    accepted=[]
    for record in chapters:
        meta=json.loads(read(public_path(record['file'],'.html').with_suffix('.meta.json')))
        review=meta.get('independent_review',{})
        if review.get('status')=='accepted' and review.get('accepted_body_sha256')==meta['authored_body_sha256']:
            accepted.append(record['file'])
    completion_path=ROOT/'completion.json'
    completion=json.loads(read(completion_path)) if completion_path.exists() else {}
    final=len(accepted)==len(chapters)
    if completion.get('complete') is True and (not final or completion.get('local_release_checks') is not True):
        raise ValueError('Final reader requires every exact-body acceptance and local release checks')
    notice=(f'All {len(chapters)} chapters have passed two independent scientific reviews. The public completion and publication records state the current release status, tested scope and limitations.'
            if final else
            f'All {len(chapters)} chapters are drafted. {len(accepted)} currently have two independent scientific acceptances. Final package and publication checks remain in progress.')
    index = '<h1>Transport, Continua and Generative Models of Turbulent Flow</h1><p class="lead">A synthesis course: continuum kinematics, flow matching and turbulence statistics, joined by new bridge chapters and laboratories.</p><p class="warning">'+esc(notice)+'</p>'
    index += '<p>Eleven of the fourteen course weeks link to existing chapters in five source editions. The chapters below are the new material. Chapter counts are not evidence of completion. <a href="book-overview.html">Read the fourteen-week study route.</a></p>'
    index += '<label class="search-label" for="lecture-search">Search available chapters</label><input id="lecture-search" type="search"><p id="search-status" role="status" aria-live="polite"></p><ol class="lecture-list">' + ''.join(rows) + '</ol>'
    write(ROOT / 'index.html', page('Chapter index', index))
    overview_path = ROOT / 'book-overview.html'
    if overview_path.exists():
        write(overview_path, page('Study route', body_of(read(overview_path))))
    inventory = ''
    for r in manifest['sources']:
        inventory += '<tr>' + ''.join(f'<td>{v}</td>' for v in [
            esc(r['edition']), f'<a href="{safe_url(r["url"])}">{esc(r["url"])}</a>',
            esc(r['used_for'])]) + '</tr>'
    source = '<h1>Source map</h1><p>The course links to chapters of five source editions and adds new chapters built from primary papers. Each section below states its source material, research extension and original construction.</p><table><thead><tr><th>Source edition</th><th>Live URL</th><th>Used for</th></tr></thead><tbody>' + inventory + '</tbody></table>'
    source += '<h2>Section provenance</h2>'
    overview_meta = ROOT / 'book-overview.meta.json'
    if overview_meta.exists():
        extra = json.loads(read(overview_meta))
        source_rows.extend(dict(s, chapter='book-overview.html') for s in extra['sections'])
    for s in source_rows:
        source += f'<h3>{esc(s["chapter"])} · {esc(s["anchor"])}</h3><p>{esc(s.get("lecture_material", ""))}</p><p>Extension: {esc(s.get("extension", ""))}. Original construction: {esc(s.get("construction", ""))}.</p>'
    write(ROOT / 'source-map.html', page('Source map', source))
    write(ROOT / 'source-map.json', json.dumps({'sections': source_rows}, indent=2) + '\n')
    reference_body = '<h1>Primary references</h1><ul>'
    for url, reference in references.items():
        reference_body += f'<li><a href="{safe_url(url)}">{esc(reference["title"])}</a><p>{esc(reference["supports"])}</p></li>'
    write(ROOT / 'references.html', page('References', reference_body + '</ul>'))
    lab_path = ROOT / 'numerical-lab.html'
    lab_body = (body_of(read(lab_path)) if lab_path.exists()
                else '<h1>Numerical laboratory</h1><p>Laboratory in preparation.</p>')
    write(lab_path, page('Numerical lab', lab_body))
    print(f'Built {len(names)} available chapters; independent review not inferred')


if __name__ == '__main__':
    build()
