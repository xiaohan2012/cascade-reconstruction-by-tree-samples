#! /bin/zsh

concat() {
    model=$1
    montage lattice-1024-m${model}.png infectious-m${model}.png grqc-m${model}.png ../method_legend.pdf -tile 3x2 -geometry +0+0 together-m${model}.png
}

cd figs/different-cascade-fractions
concat si
concat ic

cd ../different-obs-fractions
concat si
concat ic

cd ../different-obs-fractions-omleaves
concat ic

cd ../different-cascade-fractions-omleaves
concat ic
