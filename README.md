# 🛍️ ShopSense — Amazon Product Search

A semantic product search engine built with **TF-IDF** and **cosine similarity**, powered by Streamlit. Search through thousands of Amazon India products and get ranked results instantly.

---

## 🚀 Live Demo

> Deployed on Streamlit Community Cloud: [your-app-link.streamlit.app](https://your-app-link.streamlit.app)

---

## 📸 Preview

![ShopSense Preview](preview.png)

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

## ☁️ Deploy on Streamlit Cloud

1. Push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app** → select your repo → set main file to `app.py`.
4. Click **Deploy**.

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
