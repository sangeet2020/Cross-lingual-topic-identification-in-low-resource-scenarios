# Cross-lingual topic identification in low resource scenarios

The objective is to develop systems for cross-lingual topic ID without relying on machine translation systems and bilingual word embeddings. In other words, given parallel text between a source (eg. English) and target language, and some set of topic labeled documents in a source language, the goal is to predict topic labels for test documents from the target language.


## Requirements

* Python >= 3.7
* PyTorch >= 1.1
* scipy >= 1.3
* numpy >= 1.16.4
* scikit-learn >= 0.21.2
* h5py >= 2.9.0

## Language Packs used
    Kinyarwanda (IL9)
    Sinhalese (IL10)
    Zulu
    Hindi

## Dataset for training English classifier

    LDC
    Leidos

## Steps

1. Pre-Processing
2. Baseline classification on labeled English data
3. Feature representation
4. Feature transformation from source language space to target language (English) space
5. Training English classifier
6. Predict labels/topics
7. Evaluate avergae precision score

## System block diagram

![alt text](http://url/to/img.png)