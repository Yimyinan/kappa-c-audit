#!/usr/bin/env python3
"""PG-UNAGI: CustomUNAGIWrapper — 三项增强的UNAGI训练封装"""
import os, sys
import numpy as np
from datetime import datetime
from pg_unagi.tf_database import build_prior_matrix

class CustomUNAGIWrapper:
    """UNAGI增强包装器 — 课程学习 + 混合分布 + TF先验GRN"""
    def __init__(self, h5ad_path, curriculum_config=None, tf_list=None,
                 tf_prior_database=None, tf_prior_boost=1.5, device="cuda:0",
                 disable_curriculum=False, disable_mixed_dist=False, disable_tf_prior=False):
        import anndata
        from UNAGI import UNAGI
        self.h5ad_path = h5ad_path; self.device = device
        self.adata = anndata.read_h5ad(h5ad_path).copy(); self.unagi = UNAGI()
        self.tf_prior_boost = tf_prior_boost; self.tf_list = tf_list or []
        self.disable_curriculum = disable_curriculum; self.disable_mixed_dist = disable_mixed_dist
        self.disable_tf_prior = disable_tf_prior
        self.curriculum_config = curriculum_config if not disable_curriculum else None
        self.sample_weights = None
        self.tf_prior_database = tf_prior_database if not disable_tf_prior else {}
        self.tf_gene_indices = None; self.tf_prior_weights = None
        self.total_cells = self.adata.n_obs; self.n_genes = self.adata.n_vars
        if not disable_curriculum: self._prepare_curriculum_weights()
        else: self.sample_weights = np.ones(self.total_cells, dtype=np.float32)
        if not disable_mixed_dist: self._prepare_tf_indices()
        else: self.tf_gene_indices = np.array([], dtype=np.int64)
    # ---- curriculum ----
    def _prepare_curriculum_weights(self):
        cfg = self.curriculum_config; mode = cfg.get("mode", "linear")
        if mode == "custom" and "weights" in cfg:
            sw = cfg["weights"]
        elif mode == "bell":
            stages = sorted(self.adata.obs['stage'].unique()); mid = len(stages)//2
            sw = {int(s):0.5+0.5*np.exp(-0.3*(i-mid)**2) for i,s in enumerate(stages)}
        else:
            stages = sorted(self.adata.obs['stage'].unique()); n=len(stages)
            sw = {int(s):0.6+0.4*(i/n) for i,s in enumerate(stages)}
        self.sample_weights = np.array([sw.get(int(s),1.0) for s in self.adata.obs['stage']], dtype=np.float32)
        self.adata.obs['curriculum_weight'] = self.sample_weights
        print(f"  Curriculum weights: {self.sample_weights.min():.2f}~{self.sample_weights.max():.2f}")
    # ---- mixed dist ----
    def _prepare_tf_indices(self):
        gene_names = self.adata.var_names; gene_upper = [g.upper() for g in gene_names]
        indices = [gene_upper.index(tf.upper()) for tf in self.tf_list if tf.upper() in gene_upper]
        self.tf_gene_indices = np.array(indices, dtype=np.int64) if indices else np.array([], dtype=np.int64)
        print(f"  TF indices: {len(indices)}/{len(self.tf_list)} matched")
    # ---- TF prior ----
    def _prepare_tf_prior(self):
        if not self.tf_prior_database: self.tf_prior_weights = None; return
        self.tf_prior_weights = build_prior_matrix(self.adata, self.tf_prior_database, self.tf_prior_boost)
    # ---- setup + train ----
    def setup_data(self, pre_split_dir=None, **kwargs):
        data_path = pre_split_dir if pre_split_dir else self.h5ad_path
        params = {"data_path":data_path,"stage_key":"stage","total_stage":len(self.adata.obs['stage'].unique()),"neighbors":30,"threads":18}
        self.adata.uns['sample_weights'] = self.sample_weights; params.update(kwargs)
        # 确保 model_save 不存在，避免 UNAGI 报 "already existed"
        import os, shutil
        for d in ['model_save', '0', '1', '2', '3', '4', '5', 'data']:
            p = os.path.join(os.getcwd(), d)
            if os.path.exists(p): shutil.rmtree(p)
        self.unagi.setup_data(**params); self._inject_tf_indices()
        print("  setup_data done")
    def _inject_tf_indices(self):
        import torch
        for attr in ['vae','model','vae_model']:
            vm = getattr(self.unagi, attr, None)
            if vm is not None and isinstance(vm, torch.nn.Module):
                vm.tf_gene_indices = self.tf_gene_indices; return
        for attr in dir(self.unagi):
            if not attr.startswith('_'):
                v = getattr(self.unagi, attr)
                if hasattr(v,'parameters') and hasattr(v,'forward'):
                    try: v.tf_gene_indices=self.tf_gene_indices; return
                    except: pass
    def setup_training(self, task="pg_unagi", dist="ziln", latent_dim=64, hidden_dim=256, graph_dim=1024,
                       epoch_initial=300, epoch_iter=100, max_iter=10, BATCHSIZE=512,
                       lr=1e-3, lr_dis=1e-3, **kwargs):
        self._prepare_tf_prior()
        if self.tf_prior_weights is not None:
            self.unagi._tf_prior_weights = self.tf_prior_weights
        params = {"task":task,"dist":dist,"device":self.device,"GPU":1 if "cuda" in self.device else 0,
                  "hidden_dim":hidden_dim,"latent_dim":latent_dim,"graph_dim":graph_dim,
                  "epoch_initial":epoch_initial,"epoch_iter":epoch_iter,"max_iter":max_iter,
                  "BATCHSIZE":BATCHSIZE,"lr":lr,"lr_dis":lr_dis}
        params.update(kwargs); self.unagi.setup_training(**params)
        print(f"  setup_training done (task={task})")
    def run_unagi(self, idrem_dir=None):
        print(f"  Training started at {datetime.now():%Y-%m-%d %H:%M:%S}")
        print(f"  Cells: {self.total_cells:,}, Genes: {self.n_genes:,}, Device: {self.device}")
        try:
            kwargs = {}
            if idrem_dir: kwargs['idrem_dir'] = idrem_dir
            self.unagi.run_UNAGI(**kwargs)
            print("  Training complete")
        except Exception as e:
            print(f"  Training error: {e}")
            import traceback; traceback.print_exc()
