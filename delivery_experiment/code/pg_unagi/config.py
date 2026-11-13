#!/usr/bin/env python3
"""PG-UNAGI 全局配置 — TF基因集、肺纤维化TF先验、颜色方案等"""

# ====== 神经发育 TF 基因集 (小鼠基因符号) ======
NEURO_TFS = [
    # 放射状胶质 / 顶端祖细胞
    "Sox2","Pax6","Hes1","Hes5","Nestin","Vim","Prom1","Sox9",
    # 中间前体细胞
    "Eomes","Tbr2","Neurog1","Neurog2","Ascl1","Mki67","Ccdn1","Ccnd1",
    # 深层神经元
    "Tbr1","Bcl11b","Ctip2","Fezf2","Sox5","Foxy2","Cplx3","Cplx4",
    # 浅层神经元
    "Satb2","Cux1","Cux2","Brn2","Pou3f2","Pou3f3","Svet1","Mef2c",
    # GABA能中间神经元
    "Dlx1","Dlx2","Dlx5","Dlx6","Gad1","Gad2","Lhx6","Nkx2-1",
    "Sst","Pvalb","Vip","Calb2",
    # 胶质细胞发生
    "Gfap","S100b","Aqp4","Olig1","Olig2","Plp1","Mbp","Aldh1l1",
]

# ====== 肺纤维化 TF 先验 (TRRUST v2 肺条目) ======
PULMONARY_TFS = [
    # UNAGI 原论文核心 TF
    "CTCF","RAD21","SMC3","EP300",
    # TGF-β / BMP 通路
    "SMAD2","SMAD3","SMAD4","SMAD1","SMAD5",
    # NF-κB 家族
    "RELA","NFKB1",
    # Hippo 通路
    "YAP1","TEAD1","TEAD2","TEAD4",
    # 肺发育 / 上皮分化
    "GATA6","NKX2-1","FOXA2","FOXF1","SOX9","TBX4",
    # JAK/STAT 通路
    "STAT1","STAT3",
    # 增殖 / 缺氧 / 代谢
    "FOXM1","MYC","HIF1A","TP53",
    # AP-1 家族 (TGF-β 下游纤维化效应器)
    "FOS","JUN","JUNB","EGR1",
    # EMT 因子
    "SNAI1","SNAI2","TWIST1","ZEB1","ZEB2",
    # 纤维化相关
    "ETS1","KLF4","CEBPB","PPARG","RUNX1",
    # SHH 通路
    "GLI2",
    # 通用转录因子
    "SP1",
]

# ====== 造血分化 TF 先验 ======
HEMATOPOIETIC_TFS = [
    "GATA2","GATA1","KLF1",             # 红系分化
    "SPI1","CEBPA",                      # 髓系分化
    "PAX5",                              # 淋系分化
    "RUNX1","TAL1","LMO2",              # HSC 维护
    "GFI1","GFI1B",                      # 造血调控
]

# ====== 课程权重配置 (神经发育) ======
NEURO_CURRICULUM_WEIGHTS = {
    "mode": "custom",
    "weights": {
        0: 0.60, 1: 0.85, 2: 1.00, 3: 1.30, 4: 1.20, 5: 1.05, 6: 0.80,
    }
}

# ====== UNAGI 仓库配置 ======

# ====== ??????/??? TF ?? (Zebrafish Ectoderm/Neural Crest) ======
ZEBRAFISH_TFS = [
    # ??? / ????
    "sox2","sox3","pax3a","pax7a","zic1","zic2a","zic3","gbx2",
    # ??? specifiers
    "sox10","tfap2a","foxd3","snai1a","snai1b","snai2","twist1a","twist1b",
    "ets1","id2a","id3","myb","myca",
    # ?????/EMT
    "crestin","rspo3","wnt5b","fzd7a","cdh6","sox9a","sox9b",
    # ?????
    "neurog1","neurod1","neurod4","elavl3","elavl4","tubb3","huc",
    # ?????
    "tp63","dlx3b","dlx4b","grhl1","grhl3","krt4","krt5","krt18",
    # ?-???
    "otx2","gbx1","hoxb1b","hoxa2b","egr2b","mafba",
]

# ====== ??????? (24hpf ???????) ======
ZEBRAFISH_CURRICULUM_WEIGHTS = {
    "mode": "custom",
    "weights": {
        0: 0.50, 1: 0.70, 2: 0.90, 3: 1.10, 4: 1.30, 5: 1.20, 6: 1.00,
    }
}


# ====== ?????? TF ?? (Zebrafish Mesoderm) ======
ZEBRAFISH_MESODERM_TFS = [
    # ????? / ???
    "ntl","tbx16","tbx6","tbx24","msgn1","tbxta","tbx1",
    # ????? / ??
    "myod1","myf5","myog","myf6","pax3a","pax7a","meox1","mespab","ripply1",
    # ????? / ??
    "hand2","gata4","gata5","gata6","nkx2-5","nkx2.5","mef2ca","mef2cb",
    # ?? / ?? (?????)
    "gata1a","gata2a","tal1","lmo2","spi1b","runx1","lcp1",
    "fli1a","fli1b","etsrp","kdrl","tie1","cdh5",
    # ?? / ?????
    "pax2a","pax8","wt1a","wt1b","lhx1a","osr1","osr2",
    # ???? / ??
    "her1","her7","hes6","deltaC","deltaD",
]

# ====== ?????????? ======
ZEBRAFISH_MESODERM_CURRICULUM = {
    "mode": "custom",
    "weights": {
        0: 0.50, 1: 0.70, 2: 0.90, 3: 1.10, 4: 1.30, 5: 1.20, 6: 1.00,
    }
}

UNAGI_REPO = "https://github.com/mcgilldinglab/UNAGI.git"
UNAGI_DIR_NAME = "UNAGI_source"
