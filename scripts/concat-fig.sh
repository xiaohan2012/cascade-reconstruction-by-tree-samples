#! /bin/zsh

concat() {
    model=$1
    legend_suffix=$2
    montage lattice-1024-m${model}.png infectious-m${model}.png grqc-m${model}.png fb-messages-m${model}.png email-univ-m${model}.png ../method_legend${legend_suffix}.pdf -tile 5x2 -geometry +0+0 together-m${model}.png
}

cd figs/cmp-baseline-cascade-fractions
concat si
concat ic

cd ../cmp-baseline-obs-fractions
concat si
concat ic

cd ../root_selection-cascade-fractions
concat si _root_selection
concat ic _root_selection

cd ../root_selection-obs-fractions
concat si _root_selection
concat ic _root_selection

cd ../edge-cmp-baseline-obs-fractions
concat si _edge
concat ic _edge


