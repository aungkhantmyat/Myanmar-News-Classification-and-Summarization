# Myanmar-News-Classification-and-Summarization 📰🤖

## Brief
In the era of digital information, efficient processing of textual data has become essential. This project uses ML techniques to classify Myanmar news articles into four categories—Politics, Entertainment, Crime, and Business. The dataset comprises 720 articles, preprocessed using tokenization and stopword removal. Classification experiments employ an RBF Support Vector Machine (SVM) with vectorization techniques, achieving an 83% accuracy using TF-IDF vectorization. For summarization, both extractive and abstractive approaches are explored. Extractive summarization utilizes cosine similarity and the TextRank algorithm, while abstractive summarization leverages the transformer-based mT-5 model. Evaluation metrics such as accuracy for classification and ROUGE scores for summarization validate the system's effectiveness.

## Dataset
- For Myanmar News Classification, **720 news articles** from four categories **(Politics, Entertainment, Crime, and Business)** were collected, segmented into sentences to solve class imbalance, and split into 6,769 training and 1,695 testing sentences using an 80-20 ratio.
- For Summarization, the **XL-Sum dataset**, featuring professionally annotated article-summary pairs in 45 languages, was used, with 5,761 training articles and 719 each for development and testing in the Myanmar language.

## Proposed System Architecture
#### Myanmar News Classification
![Classification](https://github.com/user-attachments/assets/72be9d60-2629-4a3f-9c44-905c60319da9)

#### Extractive Summarization
![Extractive](https://github.com/user-attachments/assets/57a775f1-7823-4749-809d-da9a69796b25)

#### Abstractive Summarization
![Abstractive](https://github.com/user-attachments/assets/9ef4d242-3869-40a2-b89c-f0487374a938)

## Evaluation of Experimental Results
#### Comparison of the Accuracies with Various Vectorization Methods

| Model         | TFID | CountVec | W2Vec |
|---------------|------|----------|-------|
| SVM           | 0.83 | 0.78     | 0.60  |
| Naïve Bayes   | 0.84 | 0.84     | 0.37  |
| Random Forest | 0.78 | 0.78     | 0.66  |

#### Experimental Results for Extractive Summarization

| Metric   | Precision | Recall | F1-Score |
|----------|-----------|--------|----------|
| ROUGE-1  | 39.73     | 51.24  | 43.14    |
| ROUGE-2  | 15.91     | 23.85  | 18.10    |
| ROUGE-L  | 27.77     | 35.72  | 30.04    |

#### Accuracies of Multilingual Model and Monolingual Model with XL-Sum Dataset

| Metric   | Multilingual Model | Monolingual Model |
|----------|---------------------|-------------------|
| ROUGE-1  | 15.96              | 36.67            |
| ROUGE-2  | 5.15               | 16.78            |
| ROUGE-L  | 14.18              | 29.98            |


## Papers
Two research papers on this project have been submitted to **IEEE conferences**:
1. **"Exploring Extractive and Abstractive Approaches for Myanmar Language Text Summarization"**, was published at The _5th  International Conference on Advanced Information Technologies_ (ICAIT 2024). You can find [here](https://ieeexplore.ieee.org/document/10754935).
2. **"Systematic Comparison of Vectorization Methods in Topic Classification for Myanmar Language"**, was published at the _22nd  IEEE International Conference on Computer Applications_ (IEEE-ICCA 2025).

Also, more about the project information can be found [here](https://github.com/aungkhantmyat/Myanmar-News-Classification-and-Summarization/blob/main/Capstone%20Project%20Book.pdf).

