all: report.pdf

report.pdf: report.tex
	pdflatex report.tex
	pdflatex report.tex

clean:
	rm -f report.aux report.log report.out report.pdf

.PHONY: all clean
