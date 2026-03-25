# Document Similarity Analysis
[![ML](https://img.shields.io/badge/Machine%20Learning-NLP-blue)](https://en.wikipedia.org/wiki/Natural_language_processing)
[![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)](https://scikit-learn.org/)
[![VectorMath](https://img.shields.io/badge/Math-Cosine%20Similarity-green)](https://en.wikipedia.org/wiki/Cosine_similarity)

## 📄 Project Overview
This repository implements a robust system for calculating the **Semantic Similarity** between multiple documents. Unlike keyword-based matching (Jaccard Similarity), this project leverages **Deep Learning Embeddings** to understand the thematic relationship between texts, even when they use entirely different vocabularies.

---

## 🛠️ Technical Pipeline

### 1. Document Vectorization
* **Embedding Strategy:** Utilizing pre-trained models (OpenAI, Gemini, or Hugging Face) to transform text into fixed-length dense vectors.
* **Global Context:** Capturing the entire semantic meaning of a document rather than just word frequencies.

### 2. Similarity Metrics
* **Cosine Similarity:** Calculating the cosine of the angle between two vectors to determine their orientation and relationship ($S_C = \frac{A \cdot B}{\|A\| \|B\|}$).
* **Euclidean Distance:** Measuring the straight-line distance between document points in high-dimensional space.

### 3. Similarity Matrix Generation
* **All-Pairs Comparison:** Computing a similarity matrix to identify clusters of related documents within a large corpus.
* **Heatmap Visualization:** Using Seaborn to visualize the "closeness" of different document pairs.

### 4. Use Case Applications
* **Duplicate Detection:** Identifying nearly identical documents with high confidence scores.
* **Content Recommendation:** Finding the most similar articles to a given reference document.

---

## 💻 Tech Stack
* **Language:** Python
* **NLP Models:** Sentence-Transformers / OpenAI / Gemini
* **Data Handling:** Pandas, NumPy
* **Visualization:** Seaborn, Matplotlib
* **Mathematics:** Scikit-learn (`cosine_similarity`)

