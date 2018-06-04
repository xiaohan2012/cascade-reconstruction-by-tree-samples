#! /bin/zsh

concat() {
    model=$1
    legend_suffix=$2
    montage lattice-1024-m${model}.png infectious-m${model}.png grqc-m${model}.png fb-messages-m${model}.png email-univ-m${model}.png ../method_legend${legend_suffix}.pdf -tile 5x2 -geometry +0+0 together-m${model}.png
}

cd figs/different-cascade-fractions-omuniform
concat si
concat ic

cd ../different-obs-fractions-omuniform
concat si
concat ic

cd ../different-obs-fractions-omleaves
concat ic

cd ../different-cascade-fractions-omleaves
concat ic


cd ../small-obs-fraction-ombfs-head
concat si _small

cd ../small-obs-fraction-omleaves
concat si _small
