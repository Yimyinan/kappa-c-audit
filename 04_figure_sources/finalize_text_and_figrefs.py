from pathlib import Path
from docx import Document
from docx.oxml.ns import qn
import re, json

SRC=Path('/mnt/data/paper1_V37_PLOS_integrated_FINAL.docx')
OUT=Path('/mnt/data/paper1_V37_PLOS_integrated_FINAL.docx')
doc=Document(SRC)

ABSTRACT = (
"Gene-regulatory network (GRN) priors are increasingly added to single-cell variational autoencoders (VAEs) to make latent representations more biologically structured, yet their presence in a model does not establish that they influence training or improve perturbation prediction. We developed an open dual-spectrum audit that separately measures decoder-level latent-to-expression predictability and encoder latent-covariance structure, each against matched nulls. The diagnostic recovered a synthetic dose response and detected a conditioning effect in the VEGA masked-decoder positive control (aggregate readout ratio 5.04 versus 1.00 for matched prior-free controls). Across 33 prior-injection configurations spanning hematopoietic, neural-cortex, and lung datasets, none crossed the predefined deep decoder-compression threshold. Analytical-Jacobian recomputation was possible for 15 checkpoint-accessible configurations; the remaining 18 Ridge-only estimates also stayed above threshold under an empirical estimator-discrepancy bound. A companion latent analysis showed directional compression in 22 of 29 configurations. Source-level tracing showed that the prior-derived weight vector in the audited PG-UNAGI implementation was not consumed by the training objective, explaining the decoder-readout null. We then explicitly delivered a TRRUST-derived target-set weighting prior into the reconstruction objective. Across 73 distinct model fits on 12,000 hematopoietic cells and three training seeds, mean gene-wise R² increased by 0.0086 across 202 non-overlapping target genes in the primary in-sample analysis. This gain was smaller under five-fold cross-validation of fixed latent codes (+0.0064) and reversed relative to baseline in one seed; no detectable mean change occurred across the other 20,122 genes. Increasing dose or weighted training duration approached a local response plateau, while increasing prior support produced the modest aggregate change predicted by a support-dependent bound. A separate decoder-geometry penalty changed subspace alignment without changing aggregate readout. These results distinguish prior registration, delivery, gradient influence, representational change, and perturbation-prediction utility. The proposed audit tests prior use; it does not certify perturbation readiness."
)
AUTHOR_SUMMARY = (
"Single-cell models are often given gene-regulatory networks as biological guides, with the hope that they will learn representations useful for simulating genetic perturbations. But recording a prior in a model does not show that the model actually uses it. We developed a tool that checks whether a prior reaches the training objective and whether decoder readout and latent covariance change. We also deliberately delivered a regulatory target-gene weighting into training. It modestly improved reconstruction of the genes named by the prior, but did not produce a detectable average change across the other 20,122 genes. Increasing the dose or training duration led to a local response plateau, while a transcriptome-wide summary largely obscured the localized effect. These findings do not show that regulatory knowledge is unimportant. They show that prior registration, use during training, changes in model representation, and prediction of real genetic perturbations are separate claims that need separate tests. Our audit addresses the upstream question of prior use; downstream perturbation benchmarks are still needed to assess predictive utility."
)

def word_count(s):
    return len(re.findall(r"\b[\w]+(?:[’'/-][\w]+)*\b", s))

# Replace Abstract text with one concise paragraph, keeping the heading.
paras=doc.paragraphs
abs_idx=next(i for i,p in enumerate(paras) if p.text.strip()=='Abstract')
auth_idx=next(i for i,p in enumerate(paras) if p.text.strip().lower()=='author summary')
abs_content=[p for p in paras[abs_idx+1:auth_idx] if p.text.strip()]
if not abs_content:
    raise RuntimeError('Abstract content paragraph not found')
first=abs_content[0]
first.clear(); first.add_run(ABSTRACT)
# Remove surplus abstract paragraphs, if present.
for p in abs_content[1:]:
    p._p.getparent().remove(p._p)

# Locate Author Summary heading again after structural edits.
paras=doc.paragraphs
auth_idx=next(i for i,p in enumerate(paras) if p.text.strip().lower()=='author summary')
next_fig=next(i for i in range(auth_idx+1,len(paras)) if paras[i].text.strip().startswith('Fig 1.'))
sum_content=[p for p in paras[auth_idx+1:next_fig] if p.text.strip()]
if not sum_content:
    raise RuntimeError('Author Summary paragraph not found')
sump=sum_content[0]
sump.clear(); sump.add_run(AUTHOR_SUMMARY)
for p in sum_content[1:]:
    p._p.getparent().remove(p._p)

# Replace over-claiming section heading to align with the updated Fig 6 interpretation.
for p in doc.paragraphs:
    if p.text.strip()=='One ceiling':
        p.clear(); p.add_run('A local response plateau')

# Ensure new counts fit PLOS target ranges.
counts={'abstract_words':word_count(ABSTRACT),'author_summary_words':word_count(AUTHOR_SUMMARY)}
if counts['abstract_words']>300 or not 150<=counts['author_summary_words']<=200:
    raise RuntimeError(f'Word limits not met: {counts}')

doc.save(OUT)
Path('/mnt/data/V37_PLOS_text_wordcount_audit.json').write_text(json.dumps(counts,indent=2),encoding='utf-8')
print(counts)
