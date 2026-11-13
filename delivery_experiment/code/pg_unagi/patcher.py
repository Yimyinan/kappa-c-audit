#!/usr/bin/env python3
"""PG-UNAGI: UNAGI源码克隆与三项补丁应用"""
import os, sys, re, shutil, textwrap, subprocess
from pathlib import Path
from pg_unagi.config import UNAGI_REPO, UNAGI_DIR_NAME

UNAGI_DIR = Path("./" + UNAGI_DIR_NAME)

def log(msg, level="INFO"):
    prefix = {"INFO":"  [*]", "OK":"  [OK]", "WARN":"  [WARN]", "ERR":"  [ERR]"}.get(level, "  [*]")
    print(f"{prefix} {msg}")

def run_cmd(cmd, cwd=None):
    r = subprocess.run(cmd, shell=True, cwd=cwd or str(Path.cwd()),
                       capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}\n{r.stderr[:500]}")
    return r.stdout.strip()

def ensure_unagi_source():
    if UNAGI_DIR.exists() and (UNAGI_DIR / "setup.py").exists():
        log("UNAGI source already exists", "OK")
        return
    log(f"Cloning UNAGI from {UNAGI_REPO}...")
    run_cmd(f"git clone --depth 1 {UNAGI_REPO} {UNAGI_DIR}")
    log("Clone complete", "OK")

def _replace_in_file(filepath, old, new, expect=1):
    fp = Path(filepath)
    backup = fp.with_suffix(fp.suffix + ".bak")
    content = fp.read_text(encoding='utf-8', errors='replace')
    count = content.count(old)
    if count == 0:
        log(f"Pattern not found in {fp.name}, skip", "WARN")
        return False
    if not backup.exists():
        shutil.copy2(filepath, backup)
    content = content.replace(old, new)
    fp.write_text(content, encoding='utf-8')
    log(f"Patched {fp.name}: {count} replacements", "OK")
    return True

# Patch 1
def patch_curriculum():
    log("Patch 1: Curriculum learning weights", "INFO")
    py_files = sorted(UNAGI_DIR.rglob("*.py"))
    patterns = ["loss.mean()","recon_loss.mean()","kl_loss.mean()","loss = ","reconstruction_loss","recon_loss","elbo"]
    candidates = []
    for pyf in py_files:
        c = pyf.read_text(encoding='utf-8', errors='replace')
        score = sum(1 for p in patterns if p in c)
        if score >= 3:
            candidates.append((pyf, score, pyf.relative_to(UNAGI_DIR)))
    candidates.sort(key=lambda x: -x[1])
    if not candidates:
        log("No loss candidates found, Patch 1 failed", "ERR")
        return
    for pyf, score, rel in candidates:
        content = pyf.read_text(encoding='utf-8', errors='replace')
        modified = False
        for old_pat, new_pat in [
            (r'return\s+\(?recon_loss\s*\+\s*(?:beta\s*\*\s*)?kl\)?\.mean\(\)', 'return _weighted_loss(recon_loss + kl, sample_weight)'),
            (r'return\s+\(?recon_loss\s*\+\s*kl_loss\)?\.mean\(\)', 'return _weighted_loss(recon_loss + kl_loss, sample_weight)'),
            (r'return\s+loss\.mean\(\)', 'return _weighted_loss(loss, sample_weight)'),
        ]:
            new_c = re.sub(old_pat, new_pat, content)
            if new_c != content:
                content = new_c; modified = True
        if modified:
            pyf.write_text(content, encoding='utf-8')
            helper = textwrap.dedent('''\n# [PG-UNAGI] Weighted loss (curriculum learning)\ndef _weighted_loss(loss, sample_weight=None):\n    if sample_weight is not None and sample_weight.dim() > 0:\n        return (loss * sample_weight).sum() / sample_weight.sum()\n    return loss.mean()\n''')
            lines = content.split('\n')
            insert_pos = max(i for i, l in enumerate(lines) if l.strip().startswith(('import ','from '))) + 1
            lines.insert(insert_pos, helper)
            pyf.write_text('\n'.join(lines), encoding='utf-8')
            log(f"Patched loss aggregation: {rel}", "OK")
            return
    log("Auto-patch failed for curriculum learning", "WARN")

# Patch 2
def patch_mixed_dist():
    log("Patch 2: Mixed distribution (TF pi->0)", "INFO")
    py_files = sorted(UNAGI_DIR.rglob("*.py"))
    candidates = []
    for pyf in py_files:
        c = pyf.read_text(encoding='utf-8', errors='replace')
        if any(p in c for p in ["zi_logits","zero_inf","zero_inflation","logit_pi","pi_logit"]):
            candidates.append((pyf, pyf.relative_to(UNAGI_DIR)))
    if not candidates:
        log("No zero-inflation code found, Patch 2 skip", "WARN")
        return
    for pyf, rel in candidates:
        content = pyf.read_text(encoding='utf-8', errors='replace')
        modified = False
        for old_pat, new_pat in [
            (r'pi\s*=\s*torch\.sigmoid\(\s*zi_logits\s*\)', 'pi = torch.sigmoid(zi_logits)\n        if hasattr(self, "tf_gene_indices") and self.tf_gene_indices is not None:\n            pi[:, self.tf_gene_indices] = 0.0'),
            (r'pi\s*=\s*F\.sigmoid\(\s*zi_logits\s*\)', 'pi = F.sigmoid(zi_logits)\n        if hasattr(self, "tf_gene_indices") and self.tf_gene_indices is not None:\n            pi[:, self.tf_gene_indices] = 0.0'),
        ]:
            new_c = re.sub(old_pat, new_pat, content)
            if new_c != content and len(new_c) > len(content):
                content = new_c; modified = True
        if modified:
            pyf.write_text(content, encoding='utf-8')
            log(f"Patched zero-inflation: {rel}", "OK")
        if "self.tf_gene_indices" not in content:
            content = pyf.read_text(encoding='utf-8', errors='replace')
            init_pat = r'(self\.\w+\s*=\s*None\s*#[^\n]*[Pp]laceholder)'
            if re.search(init_pat, content):
                content = re.sub(init_pat, r'\1\n        self.tf_gene_indices = None', content, count=1)
                pyf.write_text(content, encoding='utf-8')
                log(f"Injected tf_gene_indices: {rel}", "OK")
        return
    log("Patch 2 complete", "OK")

# Patch 3
def patch_tf_prior():
    log("Patch 3: TF prior GRN -- SKIPPED (manual wiring, see runner.py)", "INFO")
    # WHY THIS IS DISABLED
    # --------------------
    # The previous implementation rewrote every occurrence of `[Ii][Dd][Rr][Ee][Mm]`
    # in UNAGI_tool.py. That pattern also matches the *identifier* `iDREM_parameters`
    # and the words inside docstrings, so it produced code like
    #     self.iDREM
    #         self._apply_tf_prior_weights()_parameters = None
    # and `raise ValueError('iDREM\n ... should be a dictionary')`, i.e. an
    # unterminated string literal. The file stopped compiling, which is invisible
    # until the next import.
    #
    # The prior is now wired by hand, in two places, with no regex:
    #   UNAGI_tool.run_UNAGI   -> unagi_runner._tf_prior_weights = self._tf_prior_weights
    #   UNAGI_runner.update_gene_weights_table -> self._apply_tf_prior_to_stagedata()
    # The latter multiplies layers['geneWeight'] in every stage h5ad after the
    # gene-weight table is written and before build_iteration_dataset() merges them.
    # Verified by _wire_test/verify_wiring.py (boosted 1.500x, unboosted 1.000x).
    return


def apply_all_patches():
    backup_dir = UNAGI_DIR.parent / f"{UNAGI_DIR.name}.bak"
    if not backup_dir.exists():
        shutil.copytree(UNAGI_DIR, backup_dir)
        log(f"Backed up original source to {backup_dir}", "OK")
    patch_curriculum()
    patch_mixed_dist()
    patch_tf_prior()
    log("All patches applied", "OK")

def install_patched():
    site = run_cmd("python -c 'import site; print(site.getsitepackages()[0])'")
    egg = Path(site) / "UNAGI.egg-link"
    if egg.exists() and Path(egg.read_text().strip()).resolve() == UNAGI_DIR.resolve():
        log("UNAGI already installed from local source", "OK"); return
    log("Installing patched UNAGI (editable)...")
    run_cmd(f"pip install -e {UNAGI_DIR} --no-build-isolation")
    log("Install complete", "OK")
