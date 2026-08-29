# LaTeX manuscript
TEX=paper
LATEX=pdflatex
BIBTEX=bibtex

all: $(TEX).pdf

$(TEX).pdf: $(TEX).tex appendix.tex references.bib figures/*.pdf
	$(LATEX) $(TEX)
	$(BIBTEX) $(TEX)
	$(LATEX) $(TEX)
	$(LATEX) $(TEX)

figures:
	python3 ../../scripts/generate_paper_figures.py

clean:
	rm -f $(TEX).aux $(TEX).bbl $(TEX).blg $(TEX).log $(TEX).out $(TEX).pdf

.PHONY: all figures clean
