# COMP61332 Coursework

## DepPath-SVM: Dependency Path-Based SVM for Relation Extraction

### Overview

DepPath-SVM is an SVM-based relation extraction model that utilizes dependency paths and ensemble learning to improve classification performance. It is trained on the WebNLG dataset and extracts relations between entity pairs using linguistic features.

### Dataset Setup

To use the WebNLG dataset, clone the repository:

```bash
!git clone https://gitlab.com/shimorina/webnlg-dataset.git
```

Also, clone and install the WebNLG toolkit:

```bash
!git clone https://github.com/WebNLG/webnlg_toolkit.git
%cd webnlg_toolkit
!pip install -e .
```

### Preprocessing

The dataset is preprocessed by extracting sentences, relations, and entities into structured DataFrames. Dependency paths between entities are extracted using spaCy’s English NER, parser, tagger, and tokenizer.

### Model Pipeline

The model uses a **Voting Classifier** that combines three SVM classifiers with different kernels:

- **Linear Kernel** (spatial relations)
- **RBF Kernel** (temporal relations)
- **Polynomial Kernel** (causal relations)

Feature extraction is performed using:

- **TfidfVectorizer** for sentence text
- **CountVectorizer** for dependency paths

The final classifier is trained using:

```python
svm_pipeline.fit(X_train, y_train)
```

### Model Saving & Loading

After training, the model is saved using:

```python
import pickle
with open('svm_model.pkl', 'wb') as f:
    pickle.dump(svm_pipeline, f)
```

To load the model for inference:

```python
with open('svm_model.pkl', 'rb') as f:
    svm_pipeline = pickle.load(f)
```

### Evaluation

The model is evaluated using accuracy and F1-score:

```python
svm_preds = svm_pipeline.predict(X_test)
svm_accuracy = (svm_preds == y_test).mean()
svm_f1 = f1_score(y_test, svm_preds, average='weighted')
```

Classification performance is analyzed using:

```python
print(classification_report(y_test, svm_preds))
```

### Real-Time Inference

Users can input entity pairs and receive the extracted relation:

```python
def svm_extract_relation(entity1, entity2):
    text = f"{entity1} {entity2}"
    prediction = svm_pipeline.predict([text])
    return prediction
```

Users can select a predefined entity pair from the WebNLG test set or provide their own. Example usage:

```python
entity1, entity2 = "Georgia (US State)", "United States"
relation = svm_extract_relation(entity1, entity2)
print(f"Extracted relation: {relation}")
```

## Requirements

Install required dependencies using the requirements.txt file
