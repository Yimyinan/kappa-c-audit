# Figure 3 V39 Rebuild

This variant keeps the V37 two-panel design but restores all 33
configuration-level observations from `table_s1_audit_body.tex`.

Panel a plots each configuration's corrected global compression, with a
tissue-level minimum-to-maximum range. Panel b plots each corrected kappa value
against both the cross-family G-VAE band (0.937-1.064) and the family-matched
UNAGI band (0.915-1.086).

The script can be rerun from the package root with:

```text
python 04_figure_sources/Fig3_V39/build_fig3_v39.py
```

It writes a new output directory under `04_figure_sources/Fig3_V39/v39_figures/Fig3`.
The packaged 600-dpi TIFF is the source used for the main manuscript and the
separate figure submission.
