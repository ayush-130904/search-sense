# 🛍️ SearchSense — Amazon Product Search

A semantic product search engine built with **TF-IDF** and **cosine similarity**, powered by Streamlit. Search through thousands of Amazon India products and get ranked results instantly.

---

## 🚀 Live Demo

> Deployed on Streamlit Community Cloud:(https://search-sense.streamlit.app/)
---

## 📸 Preview

<img width="1920" height="992" alt="Screenshot (92)" src="https://github.com/user-attachments/assets/54ef3e6c-a5ff-4aca-ac26-661c274b5d9f" />


---

## ✨ Features

- 🔍 **Semantic Search** — TF-IDF vectorization with Snowball stemming for smarter matching
- 📊 **Cosine Similarity Ranking** — Results ranked by relevance score
- 🏷️ **Category Tagging** — Each result shows its product category
- ⚡ **Fast & Cached** — Data is indexed once and cached for instant repeat searches
- 🎨 **Clean UI** — Custom-styled Streamlit interface with a responsive card layout

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| NLP / Search | NLTK, Scikit-learn (TF-IDF) |
| Data | Pandas, NumPy |
| Deployment | Streamlit Community Cloud |

---

## 📁 Project Structure

```
shopsense/
├── app.py                      # Main Streamlit application
├── amazon_india_products.csv   # Product dataset
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## ⚙️ How It Works

1. Product titles and descriptions are **tokenized and stemmed** using NLTK's Snowball Stemmer.
2. A **TF-IDF matrix** is built from the stemmed text corpus.
3. When a user enters a query, it is stemmed and **transformed into a TF-IDF vector**.
4. **Cosine similarity** is computed between the query vector and all product vectors.
5. The **top 10 most similar products** are returned and displayed.

---

## 🖥️ Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/your-username/shopsense.git
cd shopsense
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the app**
```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📦 Requirements

```
pandas
numpy
nltk
scikit-learn
streamlit
```

---

## 📄 Dataset

The app uses `amazon_india_products.csv` with the following columns:

| Column | Description |
|---|---|
| `Title` | Product name |
| `Product Description` | Detailed product description |
| `Category` | Product category |

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.

---

## 📝 License

[MIT](https://choosealicense.com/licenses/mit/)
