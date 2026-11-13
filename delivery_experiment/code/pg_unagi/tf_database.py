#!/usr/bin/env python3
"""PG-UNAGI: TF database & prior matrix builder. Uses human TRRUST v2 + curated core priors.
Usage: from pg_unagi.tf_database import build_tf_database
"""
import os, pickle
from pathlib import Path
import pandas as pd
import numpy as np
from urllib.request import urlretrieve

DATA_CACHE = Path("./unagi_custom_data")

def build_tf_database(tf_list, force_rebuild=False):
    """Build TF to target prior knowledge base (human TRRUST v2 + curated core priors).
    TF matching is case-insensitive against TRRUST.
    Args:
        tf_list: TF gene symbols (case-insensitive, human or mouse convention)
        force_rebuild: force re-download and rebuild
    Returns:
        {TF_symbol: [target1, target2, ...]}
    """
    DATA_CACHE.mkdir(parents=True, exist_ok=True)
    db_path = DATA_CACHE / "tf_prior_database_human.pkl"

    if db_path.exists() and not force_rebuild:
        with open(db_path, "rb") as f:
            return pickle.load(f)

    print("  Building TF prior database...")
    tf_database = {tf: [] for tf in tf_list}

    # TRRUST v2 (human) -- case-insensitive TF matching
    trrust_file = DATA_CACHE / "trrust_raw_human.txt"
    trrust_url = "https://www.grnpedia.org/trrust/data/trrust_rawdata.human.txt"
    try:
        if not trrust_file.exists():
            urlretrieve(trrust_url, trrust_file)
        trrust_df = pd.read_csv(trrust_file, sep="\t", header=None, comment="#",
                                names=["TF", "target", "direction", "pmid"])
        # Case-insensitive lookup: build upper to original mapping for TRRUST TFs
        trrust_tf_map = {}
        for tf_orig in trrust_df["TF"].unique():
            trrust_tf_map[tf_orig.upper()] = tf_orig
        for tf in tf_list:
            tf_matched = trrust_tf_map.get(tf.upper())
            if tf_matched:
                targets = trrust_df[trrust_df["TF"] == tf_matched]["target"].tolist()
                if targets:
                    tf_database[tf].extend(targets)
    except Exception as e:
        print(f"  TRRUST download failed: {e}")

    # ---- Curated core priors (species-aware) ----
    # Neurodevelopmental (mouse/human conserved)
    CORE_NEURO = {
        "Pax6": ["Tbr2","Eomes","Neurog1","Neurog2","Ascl1"],
        "Sox2": ["Hes1","Hes5","Pax6","Nestin"],
        "Tbr1": ["Bcl11b","Satb2","Sox5","Fezf2"],
        "Eomes": ["Neurog1","Neurog2","Tbr1","Bcl11b"],
        "Fezf2": ["Bcl11b","Ctip2","Sox5","Tbr1"],
        "Satb2": ["Cux1","Cux2","Brn2","Pou3f2"],
        "Dlx1": ["Dlx2","Dlx5","Dlx6","Gad1","Gad2"],
        "Dlx2": ["Dlx1","Dlx5","Dlx6","Gad1"],
        "Dlx5": ["Dlx6","Gad1","Gad2"],
        "Dlx6": ["Gad1","Gad2"],
        "Neurog2": ["Tbr1","Tbr2","Eomes","Bcl11b"],
        "Ascl1": ["Dlx1","Dlx2","Gad1","Gad2"],
        "Olig2": ["Olig1","Plp1","Mbp","Gfap"],
    }
    # Hematopoietic (human symbols, literature-curated)
    CORE_HEMATO = {
        "GATA2": ["GATA1","RUNX1","SPI1","TAL1","LMO2","GFI1","GFI1B","KLF1","CEBPA","PAX5"],
        "GATA1": ["KLF1","GFI1B","RUNX1","TAL1","LMO2","SPI1"],
        "RUNX1": ["SPI1","GATA1","CEBPA","TAL1","LMO2","PAX5","GFI1"],
        "SPI1": ["CEBPA","RUNX1","GATA1","GFI1","PAX5"],
        "CEBPA": ["SPI1","GATA2","RUNX1","GFI1"],
        "TAL1": ["GATA1","RUNX1","LMO2","GFI1B","KLF1","SPI1"],
        "LMO2": ["TAL1","GATA1","RUNX1","GFI1B"],
        "GFI1": ["SPI1","CEBPA","RUNX1","GATA1"],
        "GFI1B": ["GATA1","KLF1","TAL1","LMO2"],
        "KLF1": ["GATA1","GFI1B","TAL1"],
        "PAX5": ["RUNX1","SPI1"],
    }
    # Pulmonary fibrosis (human symbols, TGF-beta/Hippo/EMT literature-curated)
    CORE_PULMONARY = {
        "SMAD2": ["SMAD3","SMAD4","SNAI1","SNAI2","ZEB1","ZEB2","TWIST1","FOS","JUN"],
        "SMAD3": ["SMAD2","SMAD4","SNAI1","SNAI2","ZEB1","FOS","JUN"],
        "SMAD4": ["SMAD2","SMAD3","SNAI1","ZEB1","JUNB","FOS"],
        "RELA": ["NFKB1","FOS","JUN","SNAI1","TWIST1","HIF1A","MYC"],
        "NFKB1": ["RELA","FOS","JUN","MYC"],
        "YAP1": ["TEAD1","TEAD2","TEAD4","CTGF","SNAI2","ZEB1","FOXM1"],
        "TEAD1": ["YAP1","CTGF","FOXM1"],
        "TEAD4": ["YAP1","CTGF","SNAI2"],
        "STAT3": ["FOS","JUN","MYC","HIF1A","SNAI1","TWIST1"],
        "FOXM1": ["MYC","CCNB1","CCNB2","AURKB","PLK1"],
        "HIF1A": ["SNAI1","SNAI2","TWIST1","ZEB1","VEGFA","TGFB1","LOX"],
        "TP53": ["CDKN1A","BAX","BBC3","MDM2","GADD45A","FAS"],
        "FOS": ["JUN","JUNB","SNAI1","EGR1","MMP9"],
        "JUN": ["FOS","JUNB","SNAI1","EGR1","MMP9"],
        "SNAI1": ["CDH1","SNAI2","TWIST1","ZEB1","ZEB2","MMP2","MMP9"],
        "SNAI2": ["CDH1","TWIST1","ZEB1","MMP2"],
        "TWIST1": ["SNAI1","SNAI2","CDH1","ZEB1"],
        "ZEB1": ["CDH1","SNAI1","SNAI2","TWIST1","ZEB2","MMP2"],
        "MYC": ["CCND1","CDKN1A","HIF1A","FOXM1","SNAI1"],
        "EGR1": ["FOS","JUN","TGFB1","PDGFA"],
        "ETS1": ["MMP1","MMP3","MMP9","FOS","JUN"],
        "KLF4": ["CDH1","SNAI1","SNAI2","ZEB1"],
        "CEBPB": ["FOS","JUN","MYC"],
        "RUNX1": ["SPI1","CEBPA","MYC"],
        "GATA6": ["NKX2-1","FOXA2","FOXF1","SOX9"],
        "NKX2-1": ["SFTPA1","SFTPB","SFTPC","FOXA2","GATA6"],
        "FOXA2": ["NKX2-1","GATA6","SFTPB"],
        "FOXF1": ["SHH","GLI2","TBX4"],
        "SP1": ["FOS","JUN","MYC","TGFB1","CDKN1A"],
    }

    # Merge all core priors (case-insensitive dedup)
    ALL_CORE = {}
    for core_dict in [CORE_NEURO, CORE_HEMATO, CORE_PULMONARY]:
        for tf, targets in core_dict.items():
            key = tf.upper()
            if key not in ALL_CORE:
                ALL_CORE[key] = []
            ALL_CORE[key].extend(targets)

    # Apply core priors (case-insensitive matching to user tf_list)
    for tf in tf_database:
        core_targets = ALL_CORE.get(tf.upper(), [])
        if core_targets:
            tf_database[tf] = list(set(tf_database[tf]) | set(core_targets))

    for tf in tf_database:
        tf_database[tf] = sorted(set(tf_database[tf]))
    tf_database = {k: v for k, v in tf_database.items() if v}

    with open(db_path, "wb") as f:
        pickle.dump(tf_database, f)
    total_edges = sum(len(v) for v in tf_database.values())
    print(f"  TF prior database: {len(tf_database)} TFs, {total_edges} edges (human TRRUST + curated core)")
    return tf_database

def build_prior_matrix(adata, tf_database, boost_factor=1.5):
    """Build gene-weight prior vector from TF to target database."""
    gene_names = adata.var_names
    weights = np.ones(len(gene_names))
    gene_to_idx = {g.upper(): i for i, g in enumerate(gene_names)}
    for tf, targets in tf_database.items():
        for target in targets:
            idx = gene_to_idx.get(target.upper())
            if idx is not None:
                weights[idx] *= boost_factor
    boosted = np.sum(weights > 1.0)
    print(f"  Prior boost: {boosted}/{len(gene_names)} genes ({100*boosted/len(gene_names):.1f}%)")
    return weights