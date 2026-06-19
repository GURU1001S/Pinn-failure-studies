import os, re
labels = set()
refs = set()
cites = set()
bib_keys = set()
for root, _, files in os.walk('sections'):
    for f in files:
        if f.endswith('.tex'):
            text = open(os.path.join(root, f), encoding='utf-8').read()
            labels.update(re.findall(r'\\label\{([^}]+)\}', text))
            refs.update(re.findall(r'\\ref\{([^}]+)\}', text))
            refs.update(re.findall(r'\\pageref\{([^}]+)\}', text))
            for cite in re.findall(r'\\cite\{([^}]+)\}', text):
                cites.update([c.strip() for c in cite.split(',')])

main_text = open('main.tex', encoding='utf-8').read()
labels.update(re.findall(r'\\label\{([^}]+)\}', main_text))
refs.update(re.findall(r'\\ref\{([^}]+)\}', main_text))

bib_text = open('references.bib', encoding='utf-8').read()
bib_keys.update(re.findall(r'@\w+\{([^,]+),', bib_text))

print('Broken refs:')
broken_refs = [r for r in refs if r not in labels]
for r in sorted(broken_refs):
    print(f'✗ \\ref{{{r}}} NOT FOUND')
if not broken_refs:
    print('All refs OK')

print('\nBroken cites:')
broken_cites = [c for c in cites if c not in bib_keys]
for c in sorted(broken_cites):
    print(f'✗ \\cite{{{c}}} NOT FOUND')
if not broken_cites:
    print('All cites OK')
