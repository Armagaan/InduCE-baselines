#!/usr/bin/bash

echo "<----- BA-SHAPES ----->"
python tests/tests.py bashapes
echo

echo "<----- TREE-CYCLES ----->"
python tests/tests.py treecycles
echo

echo "<----- TREE-GRIDS ----->"
python tests/tests.py treegrids
echo

echo "<----- SMALL-AMAZON ----->"
python tests/tests.py small_amazon
echo
