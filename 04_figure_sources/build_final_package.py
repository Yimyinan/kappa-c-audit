from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from PIL import Image
from docx import Document
from lxml import etree
import shutil, json, re, hashlib, os

ROOT=Path('/mnt/data/V37_PLOS_Final_Integration_Package')
if ROOT.exists(): shutil.rmtree(ROOT)
for sub in ['Manuscript','Figures/Main','Figures/Supplementary_candidates','Code','Data_and_audits','Source_inputs','QA']:
    (ROOT/sub).mkdir(parents=True,exist_ok=True)

# Final manuscript and rendered review PDF.
shutil.copy2('/mnt/data/paper1_V37_PLOS_integrated_FINAL.docx', ROOT/'Manuscript/paper1_V37_PLOS_integrated_FINAL.docx')
shutil.copy2('/mnt/data/v37_plos_integrated_final_render/paper1_V37_PLOS_integrated_FINAL.pdf', ROOT/'Manuscript/paper1_V37_PLOS_integrated_FINAL.pdf')

# Extract original Fig 1 from the V37 source document; no revised source was provided.
with ZipFile('/mnt/data/paper1_V37_manuscript_submission.docx') as z:
    fig1=z.read('word/media/image1.png')
(ROOT/'Figures/Main/Fig1_original_embedded.png').write_bytes(fig1)

assets={
 2: Path('/mnt/data/fig2_revised_output'),
 3: Path('/mnt/data/fig3_revised_output'),
 4: Path('/mnt/data/fig4_revised_output'),
 5: Path('/mnt/data/fig5_revised_bundle/fig5_revised_output'),
 6: Path('/mnt/data/integration_assets/fig6_10/V37_Figures_6-10/fig6_10_revised_output'),
 7: Path('/mnt/data/integration_assets/fig6_10/V37_Figures_6-10/fig6_10_revised_output'),
 8: Path('/mnt/data/integration_assets/fig6_10/V37_Figures_6-10/fig6_10_revised_output'),
 9: Path('/mnt/data/integration_assets/fig6_10/V37_Figures_6-10/fig6_10_revised_output'),
 10:Path('/mnt/data/integration_assets/fig6_10/V37_Figures_6-10/fig6_10_revised_output'),
}
stems={
 2:'Fig2_dual_spectrum_workflow',
 3:'Fig3_prior_injection_audit',
 4:'Fig4_latent_spectrum_main',
 5:'Fig5_VEGA_masked_decoder',
 6:'Fig6_dose_time_plateau',
 7:'Fig7_support_resolution_bound',
 8:'Fig8_compression_band_calibration',
 9:'Fig9_prior_activity_cross_architecture',
 10:'Fig10_grn_basis_variance',
}
for n,folder in assets.items():
    stem=stems[n]
    for suffix in ['.pdf','.svg','_300dpi.png','_600dpi.tif']:
        src=folder/f'{stem}{suffix}'
        if not src.exists():
            raise FileNotFoundError(src)
        shutil.copy2(src,ROOT/f'Figures/Main/{stem}{suffix}')

# Supplementary candidate panels: deliberately left unnumbered until cross-checked against the existing SI inventory.
for fname in ['Fig4_SI_candidate_panels_EF.pdf','Fig4_SI_candidate_panels_EF_300dpi.png','Fig4_SI_candidate_panels_EF_600dpi.tif','Fig4_SI_candidate_caption.txt']:
    shutil.copy2(Path('/mnt/data/fig4_revised_output')/fname, ROOT/'Figures/Supplementary_candidates'/fname)

# Reproducible scripts and shared style.
code_files={
 'make_fig2_revised.py':'/mnt/data/fig2_revised_output/make_fig2_revised.py',
 'make_fig2_original.py':'/mnt/data/make_fig0_workflow.py',
 'make_fig3_revised.py':'/mnt/data/make_fig3_revised.py',
 'make_fig3_original.py':'/mnt/data/make_fig_3tissue.py',
 'make_fig4_revised.py':'/mnt/data/make_fig4_revised.py',
 'make_fig5_revised.py':'/mnt/data/fig5_revised_bundle/make_fig5_revised.py',
 'make_fig6_10_revised.py':'/mnt/data/fig_upgrade_work/make_fig6_10_revised.py',
 'scan_z_spectrum_3tissue.py':'/mnt/data/scan_z_spectrum_3tissue.py',
 'figure_palette.py':'/mnt/data/figure_palette.py',
 'integrate_v37.py':'/mnt/data/v37_integration_work/integrate_v37.py',
 'finalize_integrated_docx.py':'/mnt/data/v37_integration_work/finalize_integrated_docx.py',
 'finalize_text_and_figrefs.py':'/mnt/data/v37_integration_work/finalize_text_and_figrefs.py',
 'build_final_package.py':'/mnt/data/v37_integration_work/build_final_package.py',
}
for name,src in code_files.items(): shutil.copy2(src,ROOT/f'Code/{name}')

# Processed data, provenance/audit reports, and original input archives used for the revisions.
data_copies={
 'z_spectrum.csv':'/mnt/data/z_spectrum.csv',
 'z_spectrum_all.csv':'/mnt/data/z_spectrum_all.csv',
 'fig5_data_audit.json':'/mnt/data/fig5_revised_bundle/fig5_revised_output/Fig5_data_audit.json',
 'fig6_10_combined_data_audit.json':'/mnt/data/integration_assets/fig6_10/V37_Figures_6-10/fig6_10_revised_output/combined_data_audit.json',
 'fig4_data_audit.json':'/mnt/data/fig4_revised_output/figure4_data_audit.json',
 'integration_audit.json':'/mnt/data/V37_PLOS_integration_audit_FINAL.json',
 'wordcount_audit.json':'/mnt/data/V37_PLOS_text_wordcount_audit.json',
}
for name,src in data_copies.items(): shutil.copy2(src,ROOT/f'Data_and_audits/{name}')
for name,src in {
 'fig5_vega_original_inputs.zip':'/mnt/data/fig5_vega.zip',
 'fig6_8_10_prior_delivery_inputs.zip':'/mnt/data/fig6_8_10_prior_delivery.zip',
 'fig9_ipf_crossarch_inputs.zip':'/mnt/data/fig9_ipf_crossarch.zip',
}.items(): shutil.copy2(src,ROOT/f'Source_inputs/{name}')

# All revised figure captions in one file.
caption_files={
 2:'/mnt/data/fig2_revised_output/Fig2_caption.txt',
 4:'/mnt/data/fig4_revised_output/Fig4_caption.txt',
 5:'/mnt/data/fig5_revised_bundle/fig5_revised_output/Fig5_caption.txt',
}
lines=[]
lines.append('Fig 1. Original V37 Figure 1 retained without modification; no revised source was supplied.')
lines.append('')
for n in range(2,11):
    if n in caption_files:
        cap=Path(caption_files[n]).read_text(encoding='utf-8').strip()
    elif n==3:
        cap=('Fig 3. No audited configuration reaches deep decoder-readout compression. (a) Range of global compression, C = (1 − κ) × 100%, across 33 prior-injection configurations: hematopoietic (n = 22), neural cortex (n = 10), and lung fibrosis (n = 1). Endpoints mark the minimum and maximum observed values; the lung marker represents its single configuration. (b) Aggregate predictability ratio, κ, for the same tissue groups. The shaded interval marks the cross-family G-VAE vanilla-equivalence band (0.937–1.064); the dashed line marks the deep-compression boundary (κ = 0.5). The 0/33 conclusion is unchanged under the family-matched recalibration; configuration-level values and calibration details are reported in S1 and S7 Tables.')
    else:
        big=Path('/mnt/data/integration_assets/fig6_10/V37_Figures_6-10/fig6_10_revised_output/Figure_captions_Fig6_Fig10.txt').read_text(encoding='utf-8')
        m=re.search(rf'(?m)^Fig\s*{n}\.\s*(.+?)(?=\n\s*Fig\s*{n+1}\.\s*|\Z)',big,flags=re.S)
        if not m: raise RuntimeError(f'caption {n} missing')
        cap=f'Fig {n}. '+re.sub(r'\s+',' ',m.group(1)).strip()
    lines.extend([cap,''])
(ROOT/'Figures/Figures_2-10_captions.txt').write_text('\n'.join(lines),encoding='utf-8')

# QA outputs and final data integrity report.
for p in sorted(Path('/mnt/data/v37_plos_integrated_final_render').glob('contact_*.jpg')):
    shutil.copy2(p,ROOT/f'QA/{p.name}')
shutil.copy2('/mnt/data/v37_plos_integrated_final_render/page-1.png',ROOT/'QA/page-1-abstract.png')
shutil.copy2('/mnt/data/v37_plos_integrated_final_render/page-2.png',ROOT/'QA/page-2-author-summary.png')
shutil.copy2('/mnt/data/v37_plos_integrated_final_render/page-17.png',ROOT/'QA/page-17-fig4.png')
shutil.copy2('/mnt/data/v37_plos_integrated_final_render/page-29.png',ROOT/'QA/page-29-plateau-results.png')

# Verify figure embedding, footnotes, citations, title, key count and word counts.
doc=Document(ROOT/'Manuscript/paper1_V37_PLOS_integrated_FINAL.docx')
from zipfile import ZipFile
with ZipFile(ROOT/'Manuscript/paper1_V37_PLOS_integrated_FINAL.docx') as z:
    docxml=etree.fromstring(z.read('word/document.xml'))
    ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main','m':'http://schemas.openxmlformats.org/officeDocument/2006/math'}
    fulltext=''.join(docxml.xpath('.//w:t/text() | .//m:t/text()',namespaces=ns))
    footnotes_part=('word/footnotes.xml' in z.namelist())
    footnote_refs=int(docxml.xpath('count(.//w:footnoteReference)',namespaces=ns))
    unpublished=any(s in fulltext for s in ['Unpublished manuscript','A delivery-and-content audit finds gene-regulatory priors'])
    title=''.join(docxml.xpath('(.//w:body/w:p[1]//w:t/text())[1]',namespaces=ns))
image_count=len(doc.inline_shapes)
refs=[]
for p in doc.paragraphs:
    if re.match(r'^\[\d+\]',p.text.strip()): refs.append(int(re.match(r'^\[(\d+)\]',p.text.strip()).group(1)))
assert image_count==10,image_count
assert not footnotes_part and footnote_refs==0
assert not unpublished
assert refs==list(range(1,50)), (len(refs),refs[:4],refs[-4:])
assert any('37 latent-spectrum runs' in p.text for p in doc.paragraphs)
assert any('39-run on-disk PG-UNAGI inventory' in p.text for p in doc.paragraphs)
assert any('Seventeen of 18 complete hematopoietic prior/ablation runs contract' in p.text for p in doc.paragraphs)

# Check TIFF resolution, size, colour mode, etc.
img_audit={}
for n,stem in stems.items():
    tif=ROOT/f'Figures/Main/{stem}_600dpi.tif'
    im=Image.open(tif)
    dpi=im.info.get('dpi')
    img_audit[str(n)]={'dimensions_px':im.size,'dpi':[float(v) for v in dpi] if dpi else None,'mode':im.mode,'bytes':tif.stat().st_size}

report={
 'source':'V37 corrected 37-run manuscript; source file not overwritten',
 'final_manuscript':str(ROOT/'Manuscript/paper1_V37_PLOS_integrated_FINAL.docx'),
 'rendered_pdf_pages':len(list(Path('/mnt/data/v37_plos_integrated_final_render').glob('page-*.png'))),
 'embedded_inline_images':image_count,
 'figure2_to_10_replaced':True,
 'figure1_retained_original':True,
 'latent_spectrum_runs':37,
 'disk_inventory_runs':39,
 'latent_spectrum_observations':183,
 'complete_trajectory_runs':36,
 'footnotes_removed':True,
 'unpublished_followup_reference_removed':True,
 'reference_numbers':{'count':len(refs),'range':[min(refs),max(refs)]},
 'abstract_words':283,
 'author_summary_words':170,
 'TIFF_master_audit':img_audit,
}
(ROOT/'Data_and_audits/final_package_audit.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')

readme='''# V37 PLOS Computational Biology – integrated manuscript and figure package

This is the integrated working manuscript built from the user-confirmed V37 37-run correction. The original V37 source document was not overwritten.

## Main deliverables

- `Manuscript/paper1_V37_PLOS_integrated_FINAL.docx` – integrated DOCX with revised Figures 2–10 embedded; original Figure 1 retained.
- `Manuscript/paper1_V37_PLOS_integrated_FINAL.pdf` – PDF rendered from that exact DOCX (71 pages).
- `Figures/Main/` – original Figure 1 plus revised Figures 2–10 as 600-dpi TIFF, vector PDF, editable SVG, and 300-dpi PNG.
- `Figures/Figures_2-10_captions.txt` – consolidated figure captions.
- `Figures/Supplementary_candidates/` – revised Fig. 4E/F candidate panels; these are intentionally unnumbered pending reconciliation with the existing S1–S22 figure inventory.
- `Code/` – revised and original plotting scripts and shared styling module.
- `Data_and_audits/` – processed CSVs, per-figure audits, word-count report, and integration checks.
- `Source_inputs/` – uploaded original input bundles for selected figures.
- `QA/` – page contact sheets and selected full-size page previews from the final DOCX render.

## Confirmed analysis counts and scope

- The disk inventory contains 39 PG-UNAGI runs; this number remains where the manuscript refers to the on-disk inventory and the 33 decoder-audit configurations selected from it.
- The latent-spectrum analysis includes 37 runs: 24 hematopoietic, 11 neural-cortex, and 2 lung-fibrosis. These produce 183 observations across iterations 0–4.
- 36 of the 37 runs have complete iteration 0–4 trajectories and contribute to Fig. 4D; `R165_erythroid_myeloid_seed123` is present only for iterations 0–2.
- Two early neural runs were excluded from the latent-spectrum analysis due to missing self-attested seed metadata: `R0_vanilla_unagi_20260530_223153` and `R1_unagi_neuro_full_20260531_092524`. The latter remains in the separate decoder audit under the manual seed mapping `R1_s42`.

## Substantive integration changes

1. Replaced Figures 2–10 in the manuscript with their revised images; Figure 1 remains the original embedded image because no revised source was supplied.
2. Updated captions for Figures 2–10 to match the revised panel structure and statistics.
3. Updated Fig. 4 wording so A–C explicitly show all 37 runs and D shows the 36 complete trajectories; changed the contraction statement to refer to complete hematopoietic prior/ablation runs.
4. Corrected the Results heading to “A local response plateau” to avoid describing the dose/time result as a universal ceiling.
5. Updated the Fig. 10 Results ratios separately for hematopoiesis and melanoma: GRN spectral/PCA-32 = 8.7% and 18.6%; GRN spectral/NMF-32 = 9.2% and 21.6%.
6. Moved the PANK4 overlap explanation from a Word footnote into Methods; the final DOCX contains no footnote part or footnote references.
7. Removed the unavailable unpublished follow-up manuscript from the body and bibliography; subsequent references were renumbered. The final bibliography contains references [1]–[49]; current [42] is the renumbered published reference, not the removed unpublished item.
8. Replaced several OMML fragments that rendered as corrupted glyphs with equivalent plain-text mathematical expressions, without changing their numerical values.
9. Reduced the Abstract to 283 words and the Author Summary to 170 words.

## Items still requiring author confirmation before submission

1. **Fig. 5 random control provenance:** the figure uses the canonical `random_rows` condition because it matches the reported size-matched-random value (approximately 322). The uploaded source package also contains a `uniform_random` folder; it has not been included because its intended status (alternate/early run versus formal control) was not confirmed.
2. **Fig. 10 hematopoietic cell count:** the result JSON and `b1__basis_variance_audit.json` both record 3,000 cells. The V42 caption uses 3,000 cells; the old 12,000-cell plotting comment is not the source used for this panel.
3. **Supplementary Figure numbering:** revised Fig. 4E/F panels are included only as an unnumbered candidate. Their final S-number and caption must be reconciled with the existing S1–S22 Figures. The old Fig. 3C historical directional-collapse panel is not included in the main figure set; its hematopoietic part was recomputed, while the neural/lung values were explicitly unverified. Decide whether it should be omitted or retained as a clearly labelled historical-context supplement.
4. **Complete Supporting Information captions:** the manuscript still has a summary paragraph describing S1–S22 Figures and S1–S19 Tables rather than an itemized file-by-file caption list. The complete SI files were not supplied in this integration task, so their exact titles and numbering were not invented.
5. **CRediT roles:** current author-contribution labels include non-standard descriptions such as “Literature collection”, “Mathematical framework”, and “Biological guidance”. These should be mapped to CRediT roles only after the authors confirm each person’s actual contributions.
6. **Corresponding-author details:** the two corresponding-author asterisks remain; their email addresses were not supplied and have not been invented. Confirm who is the submitting corresponding author and provide each corresponding email required by the target journal.
7. **Final SI and submission-system checks:** confirm funding fields, ORCID details, data/code links and licences, the itemized SI inventory, and the journal’s current submission-system requirements.

## Data interpretation safeguards retained

- Fig. 5 rewiring repetitions are averaged within each seed; they are not treated as 90 independent model fits.
- Fig. 6 describes a local response plateau over the tested dose/time ranges, not a universal ceiling.
- Fig. 7 uses `Δκ = κ − 1`; the support-limited global resolution bound is distinguished from target-gene local effects.
- Fig. 8 is explicitly a single-seed descriptive training-budget calibration; the dashed extension is an extrapolation, not an observed condition.
- Fig. 9 distinguishes `κ_gap` from `κ_DK`.
- Fig. 10 reports dataset-specific ratios instead of the former blanket “one twentieth” claim.

## Final render / QA

The final DOCX was re-rendered to PDF and page images after the last text and figure changes. The rendered manuscript contains 71 pages. Page contact sheets for all 71 pages are included under `QA/`; selected pages for the Abstract, Author Summary, Fig. 4, and the local-response-plateau Results section are included at full size.
'''
(ROOT/'README_FINAL_INTEGRATION.md').write_text(readme,encoding='utf-8')

# Recreate ZIP.
zip_path=Path('/mnt/data/V37_PLOS_Final_Integration_Package.zip')
if zip_path.exists(): zip_path.unlink()
with ZipFile(zip_path,'w',ZIP_DEFLATED,compresslevel=6) as z:
    for f in ROOT.rglob('*'):
        if f.is_file(): z.write(f,f.relative_to(ROOT.parent))
print('package',zip_path,'bytes',zip_path.stat().st_size,'files',sum(1 for f in ROOT.rglob('*') if f.is_file()))
