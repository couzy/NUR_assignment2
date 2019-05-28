#!/bin/bash

echo "Run handin template"

echo "Creating the plotting directory if it does not exist"
if [ ! -d "plots" ]; then
  echo "Directory does not exist create it!"
  mkdir plots
fi

echo "Download data for random numbers"
if [ ! -e randomnumbers.txt ]; then
  wget https://home.strw.leidenuniv.nl/~nobels/coursedata/randomnumbers.txt
fi

echo "Download data for quad-tree particles"
if [ ! -e randomnumbers.txt ]; then
  wget strw.leidenuniv.nl/~nobels/coursedata/colliding.hdf5
fi

echo "Download data for GRB classification"
if [ ! -e GRBs.txt ]; then
  wget strw.leidenuniv.nl/~nobels/coursedata/GRBs.txt
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

# Script that saves data to a file
echo "Run the fourth script ..."
python3 exercise4.py > exercise4.txt

# Script that saves data to a file
echo "Run the fifth script ..."
python3 exercise5.py > exercise5.txt

# Script that saves data to a file
echo "Run the sixth script ..."
python3 exercise6.py > exercise6.txt

echo "Run the seventh script ..."
python3 exercise7.py > exercise7.txt

echo "Making the movies for the fourth exercise..."

ffmpeg -framerate 30 -pattern_type glob -i "plots/snap*.png" -s:v 640x480 -c:v libx264 -profile:v high -level 4.0 -crf 10 -tune animation -preset slow -pix_fmt yuv420p -r 25 -threads 0 -f mp4 Zeldovich2d.mp4
ffmpeg -framerate 30 -pattern_type glob -i "plots/xsnap*.png" -s:v 640x480 -c:v libx264 -profile:v high -level 4.0 -crf 10 -tune animation -preset slow -pix_fmt yuv420p -r 25 -threads 0 -f mp4 Zeldovichxslice.mp4
ffmpeg -framerate 30 -pattern_type glob -i "plots/ysnap*.png" -s:v 640x480 -c:v libx264 -profile:v high -level 4.0 -crf 10 -tune animation -preset slow -pix_fmt yuv420p -r 25 -threads 0 -f mp4 Zeldovichyslice.mp4
ffmpeg -framerate 30 -pattern_type glob -i "plots/zsnap*.png" -s:v 640x480 -c:v libx264 -profile:v high -level 4.0 -crf 10 -tune animation -preset slow -pix_fmt yuv420p -r 25 -threads 0 -f mp4 Zeldovichzslice.mp4

echo "Create the report"
pdflatex answers.tex
bibtex answers.aux
pdflatex answers.tex
pdflatex answers.tex
