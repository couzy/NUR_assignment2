#!/bin/bash

echo "Run handin template"

echo "Creating the plotting directory if it does not exist"
if [ ! -d "plots" ]; then
  echo "Directory does not exist create it!"
  mkdir plots
fi

echo "Download data for satellite of halos"
if [ ! -e satgals_m11.txt ]; then
  wget https://home.strw.leidenuniv.nl/~daalen/files/satgals_m11.txt
fi

# Script that returns a plot
echo "Run the first script ..."
python3 exercise1.py > exercise1.txt

# Script that pipes output to a file
echo "Run the second script ..."
python3 exercise2.py > exercise2.txt

# Script that saves data to a file
echo "Run the third script ..."
python3 exercise3.py > exercise3.txt

pdflatex template.tex
bibtex template.aux
pdflatex template.tex
pdflatex template.tex
