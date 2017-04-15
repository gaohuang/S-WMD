# S-WMD
==================================================

A demo code in Matlab for S-WMD [[Supervised Word Mover's Distance](https://papers.nips.cc/paper/6139-supervised-word-movers-distance.pdf), NIPS 2016] [Oral presentation [video recording](https://channel9.msdn.com/Events/Neural-Information-Processing-Systems-Conference/Neural-Information-Processing-Systems-Conference-NIPS-2016/Supervised-Word-Movers-Distance) by [Matt Kusner](http://mkusner.github.io/)].

Demo code runs on the [bbcsport](http://mlg.ucd.ie/datasets/bbc.html) dataset. Usage: run `swmd.m` in MATLAB. Dataset is preprocessed to contain the following fields:
- `X` is a cell array of all documents, each represented by a dxm matrix where d is the dimensionality of the word embedding and m is the number of unique words in the document
- `Y` is an array of labels
- `BOW_X` is a cell array of word counts for each document
- `indices` is a cell array of global unique IDs for words in a document
- `TR` is a matrix whose ith row is the ith training split of document indices
- `TE` is a matrix whose ith row is the ith testing split of document indices
