import pandas as pd
import numpy as np
import nltk
import warnings
warnings.filterwarnings('ignore')

from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SearchSense · Amazon Product Search",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Sora:wght@600;700;800&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #F7F5F2;
    color: #1A1A2E;
  }

  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding: 0 !important; max-width: 100% !important; }

  .hero-wrapper {
    background: linear-gradient(135deg, #FF6B35 0%, #FF8C42 50%, #FFB347 100%);
    padding: 52px 64px 44px;
    position: relative;
    overflow: hidden;
  }
  .hero-wrapper::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 280px; height: 280px;
    border-radius: 50%;
    background: rgba(255,255,255,0.10);
  }
  .hero-wrapper::after {
    content: '';
    position: absolute;
    bottom: -80px; left: 30%;
    width: 360px; height: 360px;
    border-radius: 50%;
    background: rgba(255,255,255,0.07);
  }
  .hero-eyebrow {
    font-family: 'Inter', sans-serif;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: rgba(255,255,255,0.75);
    margin-bottom: 10px;
  }
  .hero-title {
    font-family: 'Sora', sans-serif;
    font-size: 44px;
    font-weight: 800;
    color: #FFFFFF;
    line-height: 1.1;
    margin: 0 0 10px;
  }
  .hero-subtitle {
    font-size: 16px;
    font-weight: 400;
    color: rgba(255,255,255,0.85);
    margin: 0;
    max-width: 480px;
  }
  .hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.20);
    border: 1px solid rgba(255,255,255,0.35);
    color: #fff;
    font-size: 12px;
    font-weight: 600;
    padding: 4px 12px;
    border-radius: 100px;
    margin-top: 18px;
    letter-spacing: 0.5px;
  }

  .stats-bar {
    display: flex;
    gap: 32px;
    padding: 16px 64px;
    background: #FFF9F5;
    border-top: 1px solid #FFD5BC;
    border-bottom: 1px solid #FFD5BC;
  }
  .stat-item { display: flex; flex-direction: column; }
  .stat-value { font-size: 18px; font-weight: 700; color: #FF6B35; font-family: 'Sora', sans-serif; }
  .stat-label { font-size: 11px; color: #7B7568; text-transform: uppercase; letter-spacing: 0.8px; font-weight: 500; }

  .search-section {
    background: #FFFFFF;
    padding: 36px 64px 28px;
    border-bottom: 1px solid #EDE9E3;
    box-shadow: 0 2px 16px rgba(0,0,0,0.04);
  }
  .search-label {
    font-size: 13px;
    font-weight: 600;
    color: #7B7568;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    margin-bottom: 10px;
  }

  div[data-testid="stTextInput"] input {
    border: 2px solid #E8E3DB !important;
    border-radius: 12px !important;
    padding: 14px 20px !important;
    font-size: 16px !important;
    font-family: 'Inter', sans-serif !important;
    background: #FDFCFB !important;
    color: #1A1A2E !important;
    box-shadow: none !important;
  }
  div[data-testid="stTextInput"] input:focus {
    border-color: #FF6B35 !important;
    box-shadow: 0 0 0 3px rgba(255,107,53,0.10) !important;
  }

  div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #FF6B35, #FF8C42) !important;
    color: #FFFFFF !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    padding: 13px 36px !important;
    border: none !important;
    border-radius: 12px !important;
    letter-spacing: 0.3px !important;
    box-shadow: 0 4px 14px rgba(255,107,53,0.35) !important;
    width: 100% !important;
    margin-top: 4px !important;
  }
  div[data-testid="stButton"] > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(255,107,53,0.45) !important;
  }

  .results-section {
    padding: 36px 64px 64px;
    background: #F7F5F2;
  }
  .results-header {
    margin-bottom: 22px;
    padding-bottom: 14px;
    border-bottom: 2px solid #EDE9E3;
  }
  .results-title {
    font-family: 'Sora', sans-serif;
    font-size: 20px;
    font-weight: 700;
    color: #1A1A2E;
  }
  .results-count {
    font-size: 13px;
    color: #7B7568;
    font-weight: 500;
    margin-top: 2px;
  }

  .product-card {
    background: #FFFFFF;
    border: 1px solid #EDE9E3;
    border-radius: 16px;
    padding: 22px 24px;
    margin-bottom: 14px;
    position: relative;
    overflow: hidden;
    transition: all 0.2s ease;
  }
  .product-card:hover {
    border-color: #FFB347;
    box-shadow: 0 4px 20px rgba(255,107,53,0.10);
  }
  .accent-bar {
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 4px;
    background: linear-gradient(180deg, #FF6B35, #FFB347);
    border-radius: 4px 0 0 4px;
  }
  .card-inner { padding-left: 12px; }
  .card-rank {
    font-size: 11px;
    font-weight: 700;
    color: #FF6B35;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 6px;
  }
  .card-title {
    font-family: 'Sora', sans-serif;
    font-size: 16px;
    font-weight: 700;
    color: #1A1A2E;
    line-height: 1.4;
    margin-bottom: 8px;
  }
  .card-desc {
    font-size: 13.5px;
    color: #5A5650;
    line-height: 1.65;
    margin-bottom: 13px;
  }
  .card-category {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: #FFF4EE;
    border: 1px solid #FFD5BC;
    color: #D4541A;
    font-size: 11.5px;
    font-weight: 600;
    padding: 4px 11px;
    border-radius: 100px;
  }
  .similarity-pill {
    display: inline-block;
    background: #F0FDF4;
    border: 1px solid #BBF7D0;
    color: #15803D;
    font-size: 11px;
    font-weight: 600;
    padding: 3px 9px;
    border-radius: 100px;
    margin-left: 8px;
  }

  .empty-state {
    text-align: center;
    padding: 80px 20px;
  }
  .empty-icon { font-size: 52px; margin-bottom: 14px; }
  .empty-text { font-size: 16px; font-weight: 500; color: #7B7568; }
  .empty-sub { font-size: 14px; color: #B0A99E; margin-top: 6px; }
</style>
""", unsafe_allow_html=True)


# ─── Load & Prepare Data (cached) ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_and_prepare():
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

    df = pd.read_csv('amazon_india_products.csv')
    df = df[['Title', 'Product Description', 'Category']].dropna(
        subset=['Title', 'Product Description']
    )

    stemmer = SnowballStemmer('english')

    def tokenize_stem(text):
        tokens = nltk.word_tokenize(str(text).lower())
        stemmed = [stemmer.stem(token) for token in tokens]
        return " ".join(stemmed)

    df['stemmed_tokens'] = df.apply(
        lambda x: tokenize_stem(str(x['Title']) + " " + str(x['Product Description'])),
        axis=1
    )

    tfidfv = TfidfVectorizer(tokenizer=tokenize_stem)
    tfidf_matrix = tfidfv.fit_transform(df['stemmed_tokens'])

    return df, tfidfv, tfidf_matrix, tokenize_stem


# ─── Search Function ──────────────────────────────────────────────────────────
def search_products(query, df, tfidfv, tfidf_matrix, tokenize_stem):
    stemmed_query = tokenize_stem(query)
    query_vec = tfidfv.transform([stemmed_query])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    df = df.copy()
    df['similarity'] = similarities
    results = df.sort_values(by=['similarity'], ascending=False).head(10)
    return results[['Title', 'Product Description', 'Category', 'similarity']]


# ─── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrapper">
  <div class="hero-eyebrow">Amazon India &middot; Product Intelligence</div>
  <div class="hero-title">SearchSense</div>
  <div class="hero-subtitle">TF-IDF semantic search with cosine similarity ranking &mdash; find exactly what you're looking for.</div>
  <div class="hero-badge">&#128269; Semantic Product Search</div>
</div>
""", unsafe_allow_html=True)

# ─── Index data ───────────────────────────────────────────────────────────────
with st.spinner("Indexing product catalogue…"):
    df, tfidfv, tfidf_matrix, tokenize_stem = load_and_prepare()

total_products = len(df)
categories = df['Category'].nunique()

# ─── Stats Bar ────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="stats-bar">
  <div class="stat-item">
    <span class="stat-value">{total_products:,}</span>
    <span class="stat-label">Products Indexed</span>
  </div>
  <div class="stat-item">
    <span class="stat-value">{categories}</span>
    <span class="stat-label">Categories</span>
  </div>
  <div class="stat-item">
    <span class="stat-value">TF-IDF</span>
    <span class="stat-label">Algorithm</span>
  </div>
  <div class="stat-item">
    <span class="stat-value">Top 10</span>
    <span class="stat-label">Results Shown</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ─── Search Bar ───────────────────────────────────────────────────────────────
st.markdown('<div class="search-section">', unsafe_allow_html=True)
st.markdown('<div class="search-label">Search for a product</div>', unsafe_allow_html=True)

col1, col2 = st.columns([5, 1])
with col1:
    query = st.text_input(
        label="query",
        placeholder="e.g. wireless headphones, running shoes, laptop bag…",
        label_visibility="collapsed",
        key="search_query"
    )
with col2:
    search_btn = st.button("Search →")

st.markdown('</div>', unsafe_allow_html=True)

# ─── Results ─────────────────────────────────────────────────────────────────
st.markdown('<div class="results-section">', unsafe_allow_html=True)

if search_btn and query.strip():
    results = search_products(query.strip(), df, tfidfv, tfidf_matrix, tokenize_stem)

    st.markdown(f"""
    <div class="results-header">
      <div class="results-title">Results for &ldquo;{query}&rdquo;</div>
      <div class="results-count">{len(results)} most relevant products</div>
    </div>
    """, unsafe_allow_html=True)

    for rank, (_, row) in enumerate(results.iterrows(), start=1):
        title = str(row['Title'])
        desc  = str(row['Product Description'])
        cat   = str(row['Category'])
        score_pct = f"{float(row['similarity']) * 100:.1f}%"

        # Truncate description to ~200 chars
        short_desc = desc if len(desc) <= 200 else desc[:200].rstrip() + "…"

        st.markdown(f"""
        <div class="product-card">
          <div class="accent-bar"></div>
          <div class="card-inner">
            <div class="card-rank">#{rank}</div>
            <div class="card-title">{title}</div>
            <div class="card-desc">{short_desc}</div>
            <span class="card-category">&#127991; {cat}</span>
            <span class="similarity-pill">&#10003; {score_pct} match</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

elif search_btn and not query.strip():
    st.markdown("""
    <div class="empty-state">
      <div class="empty-icon">&#128172;</div>
      <div class="empty-text">Type something to search</div>
      <div class="empty-sub">Enter a product name, brand, or description above.</div>
    </div>
    """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="empty-state">
      <div class="empty-icon">&#128722;</div>
      <div class="empty-text">Ready to search</div>
      <div class="empty-sub">Enter a product name or keyword and hit Search &rarr;</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
