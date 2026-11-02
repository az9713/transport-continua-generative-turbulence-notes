"""Run every chapter listing: its bytes and its printed output must match.

Each <pre><code class="language-python"> block of a chapter must equal one file
of code/, and the <div class="output"> that follows must equal what that file
prints under python -I. Also runs pycodestyle on code/ and labs/ if installed.
Run from anywhere: python tools/check_code.py [chapter-NN-...html ...]
Roundoff-level digits can differ on another machine or NumPy build.
"""
import html
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BLOCK = re.compile(r'<pre><code class="language-python">(.*?)</code></pre>'
                   r'\s*<div class="output"><pre>(.*?)</pre></div>', re.S)


def check(path):
    text = path.read_text(encoding='utf-8')
    programs = {p.read_text(encoding='utf-8').strip(): p
                for p in (ROOT / 'code').glob('*.py')}
    issues, count = [], 0
    for source, shown in BLOCK.findall(text):
        count += 1
        program = programs.get(html.unescape(source).strip())
        if program is None:
            issues.append(f'{path.name}: block {count} matches no code/ file')
            continue
        run = subprocess.run([sys.executable, '-I', str(program)], cwd=ROOT,
                             capture_output=True, text=True, encoding='utf-8')
        want = html.unescape(shown).strip()
        if run.returncode or run.stdout.strip() != want:
            diff = [(a, b) for a, b in zip(run.stdout.strip().split('\n'),
                                           want.split('\n')) if a != b][:3]
            issues.append(f'{path.name}: {program.name} output differs: '
                          f'{run.stderr[-200:]} {diff}')
    print(f'{path.name}: {count} listing(s), {len(issues)} issue(s)')
    return issues


manifest = json.loads((ROOT / 'source-manifest.json').read_text('utf-8'))
names = sys.argv[1:] or [c['file'] for c in manifest['chapters']]
problems = [i for n in names for i in check(ROOT / n)]
style = subprocess.run([sys.executable, '-m', 'pycodestyle',
                        '--max-line-length=79', 'code', 'labs'], cwd=ROOT,
                       capture_output=True, text=True)
print('pycodestyle:', style.stdout.strip() or style.stderr.strip() or 'clean')
if style.stdout.strip():
    problems.append('pycodestyle')
print('\n'.join(problems) if problems else 'all listings and outputs match')
sys.exit(1 if problems else 0)
