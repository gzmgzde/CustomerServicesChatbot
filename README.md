# Customer Service Intent Classification Chatbot 🤖

This project implements a machine learning–based **Natural Language Processing (NLP)** pipeline to classify user instructions in the customer service domain. Using a dataset of over **26,000 realistic customer interactions** (including typos and natural phrasing), the model predicts user intents (e.g., `cancel_order`, `payment_issue`) to enable automated support responses.

The implementation is designed for clarity, performance, and reproducibility, and reflects the preprocessing logic, libraries, and evaluation results found in the accompanying Jupyter Notebook.

---

## 🚀 Features

- **Text Preprocessing**
  - Automated cleaning using Regular Expressions
  - Lowercasing, punctuation removal, and stripping placeholders (e.g., `{{var}}`)

- **Intent Recognition**
  - Classifies queries into **27 distinct intent categories**

- **High Performance**
  - Achieves **99.37% overall accuracy**
  - Built with **Logistic Regression** and **TF-IDF vectorization**

- **Automated Response Mapping**
  - Lookup system connecting predicted intents to standardized assistant replies

- **Evaluation Suite**
  - Detailed classification reports
  - Visual **Confusion Matrix Heatmap** for model reliability analysis

---

## 🛠️ Tech Stack

**Language**
- Python

**Libraries**
- `scikit-learn` – TF-IDF vectorization, Logistic Regression, evaluation metrics  
- `pandas`, `numpy` – Data manipulation and analysis  
- `matplotlib`, `seaborn` – Data visualization (performance heatmap)  
- `re` – Text cleaning and regex processing  

---

## 📊 Dataset

The model is trained on the **Bitext Customer Support LLM Chatbot Training Dataset**.

- **Total Samples:** 26,872  
- **Structure:**
  - `instruction` – Raw user query (includes realistic typos)
  - `intent` – Target classification label
  - `response` – Expected assistant reply

---

## 📈 Model Performance

The model was evaluated using an **80/20 train-test split**.

### Overall Metrics

| Metric                | Score |
|----------------------|-------|
| Overall Accuracy     | 99.37% |
| Macro Avg F1-Score   | 0.99 |

## 🧠 Methodology

### 1. Data Cleaning
- Converted all text to lowercase for consistency  
- Removed punctuation and placeholder tokens (e.g., `{{variable}}`) to reduce noise and improve model generalization  

### 2. Vectorization
- Transformed text data into numerical feature vectors using **TF-IDF (Term Frequency–Inverse Document Frequency)**
- Limited the feature space to **5,000 features** to balance performance and computational efficiency  

### 3. Model Training
- Trained a **Logistic Regression** classifier for multi-class intent prediction  
- Configured with a maximum of **1,000 iterations** to ensure proper convergence  

### 4. Results Export
- Generated and saved a detailed comparison of **Original Query vs. Predicted Intent**
- Exported predictions to a CSV file for validation, analysis, and external review  
