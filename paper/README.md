# Technical report source

NeurIPS 2024 *style* (not a conference submission). The compiled PDF is
[`../Technical_report.pdf`](../Technical_report.pdf).

## Compile

Upload this folder as an Overleaf project, or run locally:

```bash
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

If Overleaf already provides `neurips_2024.sty`, keep *their* copy and drop
`main.tex`, `macros.tex`, `refs.bib`, `sections/`, and `figures/` into that
project.
