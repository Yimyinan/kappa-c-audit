from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from lxml import etree
from PIL import Image
from docx import Document
from docx.shared import Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
import re, shutil, hashlib, json, os

BASE = Path('/mnt/data/paper1_V37_Figures_Aligned_37runs.docx')
OUT = Path('/mnt/data/paper1_V37_PLOS_integrated.docx')
PNG_DIR = Path('/mnt/data/v37_integration_work/png_for_word')
PNG_DIR.mkdir(parents=True, exist_ok=True)

# Authoritative figure assets supplied/revised in this conversation.
ASSETS = {
    2: Path('/mnt/data/fig2_revised_output/Fig2_dual_spectrum_workflow_600dpi.tif'),
    3: Path('/mnt/data/fig3_revised_output/Fig3_prior_injection_audit_600dpi.tif'),
    4: Path('/mnt/data/fig4_revised_output/Fig4_latent_spectrum_main_600dpi.tif'),
    5: Path('/mnt/data/fig5_revised_bundle/fig5_revised_output/Fig5_VEGA_masked_decoder_600dpi.tif'),
    6: Path('/mnt/data/integration_assets/fig6_10/V37_Figures_6-10/fig6_10_revised_output/Fig6_dose_time_plateau_600dpi.tif'),
    7: Path('/mnt/data/integration_assets/fig6_10/V37_Figures_6-10/fig6_10_revised_output/Fig7_support_resolution_bound_600dpi.tif'),
    8: Path('/mnt/data/integration_assets/fig6_10/V37_Figures_6-10/fig6_10_revised_output/Fig8_compression_band_calibration_600dpi.tif'),
    9: Path('/mnt/data/integration_assets/fig6_10/V37_Figures_6-10/fig6_10_revised_output/Fig9_prior_activity_cross_architecture_600dpi.tif'),
    10: Path('/mnt/data/integration_assets/fig6_10/V37_Figures_6-10/fig6_10_revised_output/Fig10_grn_basis_variance_600dpi.tif'),
}

CAPTIONS = {
2: Path('/mnt/data/fig2_revised_output/Fig2_caption.txt').read_text(encoding='utf-8').strip(),
3: ('Fig 3. No audited configuration reaches deep decoder-readout compression. '
    '(a) Range of global compression, C = (1 − κ) × 100%, across 33 prior-injection configurations: '
    'hematopoietic (n = 22), neural cortex (n = 10), and lung fibrosis (n = 1). Endpoints mark the minimum and maximum '
    'observed values; the lung marker represents its single configuration. (b) Aggregate predictability ratio, κ, for the '
    'same tissue groups. The shaded interval marks the cross-family G-VAE vanilla-equivalence band (0.937–1.064); '
    'the dashed line marks the deep-compression boundary (κ = 0.5). The 0/33 conclusion is unchanged under the '
    'family-matched recalibration; configuration-level values and calibration details are reported in S1 and S7 Tables.'),
4: Path('/mnt/data/fig4_revised_output/Fig4_caption.txt').read_text(encoding='utf-8').strip(),
5: Path('/mnt/data/fig5_revised_bundle/fig5_revised_output/Fig5_caption.txt').read_text(encoding='utf-8').strip(),
}
cap_text = Path('/mnt/data/integration_assets/fig6_10/V37_Figures_6-10/fig6_10_revised_output/Figure_captions_Fig6_Fig10.txt').read_text(encoding='utf-8')
for n in range(6, 11):
    m = re.search(rf'(?m)^Fig\s*{n}\.\s*(.+?)(?=\n\s*Fig\s*{n+1}\.\s*|\Z)', cap_text, flags=re.S)
    if not m:
        raise RuntimeError(f'Caption for Fig {n} not found')
    CAPTIONS[n] = f'Fig {n}. ' + re.sub(r'\s+', ' ', m.group(1)).strip()
# Normalize Fig 4 caption phrase against the user's authoritative run wording.
CAPTIONS[4] = CAPTIONS[4].replace('Seventeen of 18 complete hematopoietic prior runs contract.',
                                  'Seventeen of 18 complete hematopoietic prior runs contract.')
# Normalize spacing and capitalization for consistency.
for n, cap in CAPTIONS.items():
    cap = re.sub(rf'^Fig\s*{n}\.\s*', f'Fig {n}. ', cap)
    CAPTIONS[n] = cap

# Prepare 300-dpi RGB PNGs for Word embedding. TIFF masters are retained separately.
word_png = {}
for n, tif in ASSETS.items():
    if not tif.exists():
        raise FileNotFoundError(tif)
    im = Image.open(tif)
    # Downsample 600-dpi artwork to 300 dpi at its intended 170-mm placement width.
    nw = round(im.width / 2)
    nh = round(im.height / 2)
    im = im.resize((nw, nh), Image.Resampling.LANCZOS)
    if im.mode == 'RGBA':
        bg = Image.new('RGB', im.size, 'white')
        bg.paste(im, mask=im.getchannel('A'))
        im = bg
    elif im.mode != 'RGB':
        im = im.convert('RGB')
    dest = PNG_DIR / f'Fig{n}_300dpi_word.png'
    im.save(dest, format='PNG', dpi=(300, 300), optimize=True)
    word_png[n] = dest

# Load the current 37-run manuscript. It already contains the corrected 37-run counts.
doc = Document(BASE)

# Locate the 10 figure paragraphs by position in document order (Fig 1 is intentionally unchanged).
image_paras = [p for p in doc.paragraphs if p._p.xpath('.//w:drawing')]
if len(image_paras) != 10:
    raise RuntimeError(f'Expected 10 figure images, found {len(image_paras)}')

alt_text = {
2: 'Workflow of the dual-spectrum audit: parallel decoder-readout and latent-covariance analyses, followed by four prior-activity checks.',
3: 'Tissue-level observed ranges of global compression and aggregate predictability ratios for 33 audited configurations.',
4: 'Latent effective-rank trajectories across 37 runs in three tissues and run-level changes for 36 complete trajectories.',
5: 'VEGA masked-decoder positive control: paired aggregate readout ratios and five pathway-mask/control conditions.',
6: 'Matched per-gene readout effects across prior-weight dose and weighted training duration, with paired seed trajectories.',
7: 'Support-dependent resolution bound and decomposition of local named-gene and global other-gene contributions.',
8: 'Calibration of the 50-percent compression band across five training budgets and the relation to per-gene reconstruction quality.',
9: 'Prior-activity checks on human IPF and cross-architecture validation using eight CellOracle-style TF priors.',
10: 'Variance shares of GRN-derived, permuted, PCA-32, and NMF-32 bases in hematopoietic and melanoma datasets.',
}

# Replace Figures 2–10 in place; Figure 1 remains unchanged because no revised source was supplied.
for n in range(2, 11):
    p = image_paras[n-1]
    # Preserve paragraph properties, remove existing inline drawing/runs, then insert a new PNG at text width.
    for child in list(p._p):
        if child.tag != qn('w:pPr'):
            p._p.remove(child)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_together = True
    run = p.add_run()
    pic = run.add_picture(str(word_png[n]), width=Inches(6.69))
    pic._inline.docPr.set('descr', alt_text[n])
    pic._inline.docPr.set('title', f'Figure {n}')

# Update each caption paragraph directly following each figure image.
# The paragraphs remain in place; style/paragraph properties are retained.
for n in range(2, 11):
    p = image_paras[n-1]._p.getparent()
    # Find the next paragraph in the document body after this image paragraph.
    next_p = image_paras[n-1]._p.getnext()
    while next_p is not None and next_p.tag != qn('w:p'):
        next_p = next_p.getnext()
    if next_p is None:
        raise RuntimeError(f'Caption paragraph for Fig {n} not found')
    from docx.text.paragraph import Paragraph
    cap_p = Paragraph(next_p, doc._body)
    cap_p.clear()
    label = cap_p.add_run(f'Fig {n}. ')
    label.bold = True
    cap_p.add_run(CAPTIONS[n][len(f'Fig {n}. '):])
    cap_p.paragraph_format.keep_together = True
    cap_p.paragraph_format.keep_with_next = False

# Sentence-case title for PLOS style.
if doc.paragraphs:
    title_p = doc.paragraphs[0]
    title_txt = title_p.text.strip()
    if title_txt.lower().startswith('from unread gene-regulatory priors'):
        title_p.clear()
        title_p.add_run('From unread gene-regulatory priors to a support-dependent ceiling in single-cell VAEs')

# Soften the interpretation of dose/time agreement: plateau within tested ranges, not a universal ceiling.
for run in doc.paragraphs[106].runs:
    if 'Independent levers arriving at the same value is stronger evidence for a real ceiling' in run.text:
        run.text = run.text.replace(
            'Independent levers arriving at the same value is stronger evidence for a real ceiling than either ladder alone, because an artifact of one axis would not reproduce on the other.',
            'This agreement supports a local response plateau over the tested dose and training-duration ranges; it does not establish a universal ceiling.'
        )

# Update the Fig. 10 Results interpretation with dataset-specific ratios.
for run in doc.paragraphs[162].runs:
    if 'times a random basis of the same rank, and about one ninth of what the data’s own directions reach.' in run.text:
        run.text = run.text.replace(
            'times a random basis of the same rank, and about one ninth of what the data’s own directions reach.',
            'times a random basis of the same rank. Relative to PCA-32, the GRN spectral basis captured 8.7% of the variance share in hematopoiesis and 18.6% in melanoma; the corresponding ratios to NMF-32 were 9.2% and 21.6%, respectively.'
        )

# Make the 39-run inventory / 37-run latent-spectrum distinction auditable by naming the two exclusions.
for run in doc.paragraphs[229].runs:
    if 'The separate latent-spectrum analysis includes 37 runs after excluding two early neural runs without self-attested seed metadata;' in run.text:
        run.text = run.text.replace(
            'The separate latent-spectrum analysis includes 37 runs after excluding two early neural runs without self-attested seed metadata; one of these remains in the separate decoder audit under the manual seed mapping R1_s42, but is not included in the latent-spectrum analysis.',
            'The separate latent-spectrum analysis includes 37 runs after excluding two early neural run directories without self-attested seed metadata: R0_vanilla_unagi_20260530_223153 (max_iter = 8; no seed field) and R1_unagi_neuro_full_20260531_092524 (no seed field). The latter remains in the separate decoder audit under the manual seed mapping R1_s42 but is not included in the latent-spectrum analysis.'
        )

# Move the PANK4 overlap explanation from a true Word footnote into the Methods description of the delivered prior.
method_p = doc.paragraphs[285]
method_p.add_run(
    ' PANK4, one of the 203 curated targets, was also selected by the weight-matched permutation. '
    'It was excluded from both sets to preserve disjointness, leaving 202 real targets and 203 permuted-control genes in the contrast; '
    'the remaining 20,122 genes are outside both sets (20,527 − 202 − 203).'
)

# Remove the citation to the unpublished follow-up manuscript and replace it with a scope statement.
for para in doc.paragraphs:
    for run in para.runs:
        if 'a fuller benchmark is reported separately [42].' in run.text:
            run.text = run.text.replace(
                'a fuller benchmark is reported separately [42].',
                'a comprehensive perturbation-prediction benchmark is beyond the scope of the present study.'
            )

# Remove the actual footnote reference run (the explanatory text now lives in Methods).
footnote_refs_removed = 0
for para in doc.paragraphs:
    for child in list(para._p):
        if child.tag == qn('w:r') and any(el.tag == qn('w:footnoteReference') for el in child.iter()):
            para._p.remove(child)
            footnote_refs_removed += 1

# Delete unpublished reference [42], and decrement all subsequent numeric citations/reference labels by one.
ref_paras_removed = 0
for para in list(doc.paragraphs):
    if para.text.strip().startswith('[42] Yan Y, Wang L.'):
        para._p.getparent().remove(para._p)
        ref_paras_removed += 1

citation_re = re.compile(r'\[([0-9][0-9,\s–-]*)\]')
def shift_citation(match):
    inner = match.group(1)
    # Only decrement numeric tokens >= 43, preserving separators, spaces, and lower references.
    return '[' + re.sub(r'\d+', lambda m: str(int(m.group(0)) - 1 if int(m.group(0)) >= 43 else int(m.group(0))), inner) + ']'
for para in doc.paragraphs:
    for run in para.runs:
        if '[' in run.text and ']' in run.text:
            run.text = citation_re.sub(shift_citation, run.text)

# Save an intermediate docx. Footnote XML relationship/part are removed at package level below.
doc.save(OUT)

# Remove the now-empty Word footnotes part and its relationship/content-type declaration.
TMP = OUT.with_suffix('.tmp.docx')
with ZipFile(OUT, 'r') as zin, ZipFile(TMP, 'w', ZIP_DEFLATED) as zout:
    for item in zin.infolist():
        name = item.filename
        if name == 'word/footnotes.xml':
            continue
        data = zin.read(name)
        if name == 'word/_rels/document.xml.rels':
            root = etree.fromstring(data)
            for rel in list(root):
                if rel.get('Type', '').endswith('/footnotes'):
                    root.remove(rel)
            data = etree.tostring(root, xml_declaration=True, encoding='UTF-8', standalone=True)
        elif name == '[Content_Types].xml':
            root = etree.fromstring(data)
            for child in list(root):
                if child.get('PartName') == '/word/footnotes.xml':
                    root.remove(child)
            data = etree.tostring(root, xml_declaration=True, encoding='UTF-8', standalone=True)
        zout.writestr(item, data)
TMP.replace(OUT)

# Structural verification.
verify = Document(OUT)
# Note: python-docx's high-level parser should load the patched package.
shape_count = len(verify.inline_shapes)
caption_checks = {}
for n in range(1, 11):
    cap_para = None
    for i, p in enumerate(verify.paragraphs):
        if p.text.strip().startswith(f'Fig {n}.'):
            cap_para = p.text.strip()
            break
    caption_checks[str(n)] = cap_para

with ZipFile(OUT, 'r') as z:
    names = set(z.namelist())
    has_footnotes_part = 'word/footnotes.xml' in names
    root = etree.fromstring(z.read('word/document.xml'))
    ns = {'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
    footnote_ref_count = int(root.xpath('count(.//w:footnoteReference)', namespaces=ns))
    fulltext = ''.join(root.xpath('.//w:t/text()', namespaces=ns))
    remaining_42 = '[42]' in fulltext
    relroot = etree.fromstring(z.read('word/_rels/document.xml.rels'))
    footnote_rel_count = sum(1 for r in relroot if r.get('Type','').endswith('/footnotes'))

report = {
    'source_docx': str(BASE),
    'output_docx': str(OUT),
    'inline_shapes': shape_count,
    'figures_replaced': list(range(2,11)),
    'fig1_retained_original': True,
    'footnote_reference_runs_removed': footnote_refs_removed,
    'reference_42_paragraphs_removed': ref_paras_removed,
    'footnotes_part_remaining': has_footnotes_part,
    'footnote_reference_count_remaining': footnote_ref_count,
    'footnote_relationship_count_remaining': footnote_rel_count,
    'citation_42_remaining': remaining_42,
    'figure_caption_prefixes_found': {k: bool(v) for k,v in caption_checks.items()},
}
Path('/mnt/data/V37_PLOS_integration_audit.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
print(json.dumps(report, indent=2, ensure_ascii=False))
