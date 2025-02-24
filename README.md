![Extractive Summarization](https://github.com/user-attachments/assets/02cc1134-486d-41a8-8131-dfab389dac10)# Myanmar-News-Classification-and-Summarization

## Brief
In the era of digital information, efficient processing of textual data has become essential. This project uses ML techniques to classify Myanmar news articles into four categories—Politics, Entertainment, Crime, and Business. The dataset comprises 720 articles, preprocessed using tokenization and stopword removal. Classification experiments employ an RBF Support Vector Machine (SVM) with vectorization techniques, achieving an 83% accuracy using TF-IDF vectorization. For summarization, both extractive and abstractive approaches are explored. Extractive summarization utilizes cosine similarity and the TextRank algorithm, while abstractive summarization leverages the transformer-based mT-5 model. Evaluation metrics such as accuracy for classification and ROUGE scores for summarization validate the system's effectiveness.

## Dataset
- For Myanmar News Classification, **720 news articles** from four categories **(Politics, Entertainment, Crime, and Business)** were collected, segmented into sentences to solve class imbalance, and split into 6,769 training and 1,695 testing sentences using an 80-20 ratio.
- For Summarization, the **XL-Sum dataset**, featuring professionally annotated article-summary pairs in 45 languages, was used, with 5,761 training articles and 719 each for development and testing in the Myanmar language.

## Proposed System Architecture
1. Myanmar News Classification
   ![Classification](https://github.com/user-attachments/assets/614a49ec-7fcd-4876-be52-37838112cdf3)

2. Extractive Summarization
![Extractive Summarization](https://github.com/user-attachments/assets/a1d2f5b5-8789-4e64-9071-7c1a93327bd5)

3. Abstractive Summarization
![Abstractive Summarization](https://github.com/user-attachments/assets/33993832-a661-4525-b75f-c3a2872241ca)

## Evaluation of Experimental Results
1. Comparison of the Accuracies with Various Vectorization Methods

2. Experimental Results for Extractive Summarization

3. Accuracies of multilingual model and monolingual model with XL-Sum Dataset


## Papers
Two research papers on this project have been submitted to IEEE conferences:
1. "Exploring Extractive and Abstractive Approaches for Myanmar Language Text Summarization," was published at The 5th  International Conference on Advanced Information Technologies (ICAIT 2024). You can find [here](https://public.thinkonweb.com/sites/iccr2023/proceeding)
2. "Systematic Comparison of Vectorization Methods in Topic Classification for Myanmar Language," was published at the 22nd  IEEE International Conference on Computer Applications (IEEE-ICCA 2025).

Also, more about the project information can be found [here](https://github.com/Raghu2411/Vaccine-Tweets-Sentiment-Analysis/blob/main/Project%20Report.pdf).

