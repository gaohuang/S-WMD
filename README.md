# S-WMD


A demo code in Matlab for S-WMD [[Supervised Word Mover's Distance](https://papers.nips.cc/paper/6139-supervised-word-movers-distance.pdf), NIPS 2016] [Oral presentation [video recording](https://channel9.msdn.com/Events/Neural-Information-Processing-Systems-Conference/Neural-Information-Processing-Systems-Conference-NIPS-2016/Supervised-Word-Movers-Distance) by [Matt Kusner](http://mkusner.github.io/)].

Demo code runs on the [bbcsport](http://mlg.ucd.ie/datasets/bbc.html) dataset. Usage: run `swmd.m` in MATLAB. Dataset is preprocessed to contain the following fields:
- `X` is a cell array of all documents, each represented by a dxm matrix where d is the dimensionality of the word embedding and m is the number of unique words in the document
- `Y` is an array of labels
- `BOW_X` is a cell array of word counts for each document
- `indices` is a cell array of global unique IDs for words in a document
- `TR` is a matrix whose ith row is the ith training split of document indices
- `TE` is a matrix whose ith row is the ith testing split of document indices

## Paper Datasets

Here is a Dropbox link to the datasets used in the paper: https://www.dropbox.com/sh/nf532hddgdt68ix/AABGLUiPRyXv6UL2YAcHmAFqa?dl=0

They're all matlab .mat files and have the following variables:

**for bbcsport, twitter, recipe, classic, amazon**
- `X [1,n+ne]`: each cell corresponds to a document and is a `[300,u]` matrix where `u` is the number of unique words in that document, `n` is the number of training points, and `ne` is the number of test points. Each column is the word2vec vector for a particular word.
- `Y [1,n]`: the label of each document
- `BOW_X [1,n+ne]`: each cell in the cell array is a vector corresponding to a document. The size of the vector is the number of unique words in the document, and each entry is how often each unique word occurs.
- `words [1,n+ne]`: each cell corresponds to a document and is itself a `{1,u}` cell where each entry is the actual word corresponding to each unique word
- `TR [5,n]`: each row corresponds to a random split of the training set, each entry is the index with respect to the full dataset. So for example, to get the BOW of the training set for the third split do: `BOW_xtr = BOW_X(TR(3,:))`
- `TE [5,ne]`: same as TR except for the test set


for **ohsumed, reuters (r8), 20news (20ng2_500)**

The only difference with the above datasets is that because there are pre-defined train-test splits, there are already variables `BOW_xtr`, `BOW_xte`, `xtr`, `xte`, `ytr`, `yte`.

