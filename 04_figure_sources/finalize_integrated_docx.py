from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from lxml import etree
from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
import re, json, hashlib

SRC = Path('/mnt/data/paper1_V37_PLOS_integrated.docx')
OUT = Path('/mnt/data/paper1_V37_PLOS_integrated_FINAL.docx')

doc = Document(SRC)

# 1) Correct Fig. 4 caption and Results count wording.
for p in doc.paragraphs:
    if p.text.startswith('Fig 4.'):
        for r in p.runs:
            r.text = r.text.replace(
                'Seventeen of 18 complete hematopoietic prior runs contract.',
                'Seventeen of 18 complete hematopoietic prior/ablation runs contract.'
            )
    if p.text.startswith('We asked whether prior injection changes the latent space itself'):
        for r in p.runs:
            r.text = r.text.replace(
                '36 complete trajectories are shown in Fig 4d',
                '36 of 37 runs with complete iteration 0–4 trajectories are shown in Fig 4d'
            )

# Helper to replace a specific OMML node with an ordinary Word text run.
M_NS = 'http://schemas.openxmlformats.org/officeDocument/2006/math'
W_NS = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
XML_NS = 'http://www.w3.org/XML/1998/namespace'

def math_text(node):
    return ''.join((e.text or '') for e in node.iter() if e.tag == f'{{{M_NS}}}t')

def replace_math_in_para(para, old, new, count=1):
    done = 0
    for child in list(para._p):
        if child.tag in (qn('m:oMath'), qn('m:oMathPara')) and math_text(child) == old:
            r = OxmlElement('w:r')
            t = OxmlElement('w:t')
            t.set(f'{{{XML_NS}}}space', 'preserve')
            t.text = new
            r.append(t)
            para._p.replace(child, r)
            done += 1
            if done >= count:
                break
    return done

# 2) Replace three math fragments whose OMML glyphs render incorrectly in LibreOffice.
# Keep all numerical values and inferential content unchanged.
replacements = []
replacements.append(('Fig4 neural factorial t statistic', replace_math_in_para(doc.paragraphs[54], '|t|<0.3', '|t| < 0.3')))
replacements.append(('Fig4 neural factorial p value', replace_math_in_para(doc.paragraphs[54], 'p>0.39', 'p > 0.39')))
replacements.append(('Fig7 absolute delta kappa bound', replace_math_in_para(doc.paragraphs[133], '|Δκ|<0.015', '|Δκ| < 0.015')))
replacements.append(('Fig10 absolute delta A', replace_math_in_para(doc.paragraphs[165], '|ΔA|', '|ΔA|')))

# 3) Remove the unpublished-follow-up citation from the body.
# The citation was nonbreaking-space separated, so the earlier exact-run replacement missed it.
for p in doc.paragraphs:
    if 'a fuller benchmark is reported separately' in p.text:
        new_text = p.text.replace(
            'a fuller benchmark is reported separately\u00a0[42].',
            'a comprehensive perturbation-prediction benchmark is beyond the scope of the present study.'
        ).replace(
            'a fuller benchmark is reported separately [42].',
            'a comprehensive perturbation-prediction benchmark is beyond the scope of the present study.'
        )
        p.clear()
        p.add_run(new_text)

# Save.
doc.save(OUT)

# Structural validation and audit.
verify = Document(OUT)
with ZipFile(OUT, 'r') as z:
    names = set(z.namelist())
    root = etree.fromstring(z.read('word/document.xml'))
    ns = {'w': W_NS, 'm': M_NS}
    fulltext = ''.join(root.xpath('.//w:t/text() | .//m:t/text()', namespaces=ns))
    has_footnotes = 'word/footnotes.xml' in names
    footnote_refs = int(root.xpath('count(.//w:footnoteReference)', namespaces=ns))
    has_unpublished = ('Unpublished manuscript' in fulltext or
                       'A delivery-and-content audit finds gene-regulatory priors' in fulltext)
    old_phrase = 'a fuller benchmark is reported separately' in fulltext
    refs = []
    for p in verify.paragraphs:
        t = p.text.strip()
        if re.match(r'^\[\d+\]', t): refs.append(t)

fig_caps = {}
for n in range(1, 11):
    fig_caps[str(n)] = any(p.text.strip().startswith(f'Fig {n}.') for p in verify.paragraphs)

report = {
    'output_docx': str(OUT),
    'paragraph_count': len(verify.paragraphs),
    'inline_image_count': len(verify.inline_shapes),
    'footnotes_part_present': has_footnotes,
    'footnote_reference_count': footnote_refs,
    'unpublished_manuscript_text_present': has_unpublished,
    'old_unpublished_body_phrase_present': old_phrase,
    'figure_captions_1_to_10_present': fig_caps,
    'targeted_math_replacements': replacements,
    'references_count': len(refs),
    'last_reference_prefixes': refs[-5:],
}
Path('/mnt/data/V37_PLOS_integration_audit_FINAL.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
print(json.dumps(report, indent=2, ensure_ascii=False))
