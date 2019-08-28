# Cross-lingual topic identification in low resource scenarios

The objective is to develop systems for cross-lingual topic ID without relying on machine translation systems and bilingual word embeddings. In other words, given parallel text between a source (eg. English) and target language, and some set of topic labeled documents in a source language, the goal is to predict topic labels for test documents from the target language.

## Important Note
This project is still in the development stage. NOT all codes can be made public at this stage.


## Requirements

* Python >= 3.7
* scipy >= 1.3
* numpy >= 1.16.4
* scikit-learn >= 0.21.2

## Language packs used
    Kinyarwanda (IL9)
    Sinhalese (IL10)
    Zulu
    Hindi

## Dataset for training English classifier
    LDC
    Leidos

## Steps in brief

1. Pre-Processing
```
python src/XML_parser_Universal.py --help
```
2. Baseline classification on labeled English data
3. Feature representation
```
python src/extract_features_for_test_data.py --help
```
4. Feature transformation from source language space to the target language (English) space
5. Training English classifier
```
python src/train_classifier_on_eng_sf_features.py --help
```
Training English classifier batch-wise
```
python src/train_clf_batchwise.py --help
```
6. Predict labels/topics
7. Evaluate average precision score

## Usage - wrapper files

### 1. Pre-processing

    bash wrapper_preProcess.sh
 Task performed:
*   Generate ground truth annotations
*   Preparing ground truth of the annotations
*   Convert ASR outputs to a text file and document id file
*   Analyze English labeled text data
*   Eliminate out-of-domain docs

### 2. Baseline classification using theme-specific tokens
    wrapper_baseline_best_vacabs.sh

Task performed:
* Generate the best vocabulary from each class of labeled English text data
* Finds the best alpha for each topic/label
* Generates bash file to generate features for parallel text
* Generates bash file to compute final average precision scores for the given test data.
* Display the average precision score for all N (vocab size for each label)

### 3. Feature generation
    wrapper_feats_gen.sh
Performs feature extraction on parallel text data and labeled English text data

### 4. Learning transformation
    wrapper_learn_transformation.sh
Generates bash files to learn transformation and submit jobs as well to GPUs

### 5. Compute average precision score for given text documents
    wrapper_final_scores_generator_new_doc_level.sh
Task performed:
* Training English Classifier
* Combine all sentences of one document into separate sentences.
* Feature extraction of text data
* Apply transformation from IL space to English space
* Predict topics for each document
* Compute average precision scores





## System block diagram

![alt text](https://github.com/sangeet2020/Cross-lingual-topic-identification-in-low-resource-scenarios/blob/development/system_block_diagram.png)
