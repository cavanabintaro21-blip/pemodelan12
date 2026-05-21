import streamlit as st
import pandas as pd
from bertopic import BERTopic
from bertopic.representation import MaximalMarginalRelevance
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from transformers import pipeline
from datetime import datetime
import re
import string
import logging
import math
import html
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import hashlib
import pickle
import json
import plotly.graph_objects as go
import plotly.express as px

# Optional Indonesian NLP support
try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
    import nltk
    from nltk.corpus import stopwords
except ImportError:
    StemmerFactory = None
    StopWordRemoverFactory = None
    nltk = None
    stopwords = None

indonesia_stopwords = set()
extra_stopwords = {
    'yg', 'ya', 'nya', 'ga', 'gak', 'aja', 'kok', 'lah', 'mah', 'dong', 'deh', 'nih',
    'lho', 'loh', 'bro', 'sis', 'gue', 'gua', 'lu', 'loe', 'kamu', 'aku', 'saya',
    'dia', 'mereka', 'kita', 'kami', 'itu', 'ada', 'belum',
    'sudah', 'akan', 'dengan', 'yang', 'dan', 'atau', 'tapi', 'karena', 'jika',
    'kalau', 'saat', 'ketika', 'untuk', 'dari', 'ke', 'di', 'pada', 'dalam',
    'sebagai', 'seperti', 'hanya', 'juga', 'lagi', 'masih', 'bisa', 'boleh',
    'perlu', 'ingin', 'mau', 'suka', 'juga', 'yg', 'udah', 'banget', 'kalo', 'sih', 
    'tau', 'pas', 'gitu', 'orang', 'liat', 'buat', 'dr'
}

if stopwords is not None:
    try:
        if nltk is not None:
            try:
                nltk.data.find('corpora/stopwords')
            except LookupError:
                nltk.download('stopwords')
        indonesia_stopwords = set(stopwords.words('indonesian'))
    except Exception:
        indonesia_stopwords = set()

if StopWordRemoverFactory is not None:
    try:
        stop_factory = StopWordRemoverFactory()
        indonesia_stopwords.update(stop_factory.get_stop_words())
    except Exception:
        pass

indonesia_stopwords.update(extra_stopwords)

# Preserve negation terms because removing 'tidak', 'bukan', 'gak', or 'ga' can change stance meaning
for negation in ['tidak', 'bukan', 'gak', 'ga', 'belum', 'jangan']:
    indonesia_stopwords.discard(negation)

# Preserve important functional words for meaning in stance context
for preserve_word in ['harus', 'dijawab', 'ini', 'oleh']:
    indonesia_stopwords.discard(preserve_word)

stemmer = None
if StemmerFactory is not None:
    try:
        stemmer = StemmerFactory().create_stemmer()
    except Exception:
        stemmer = None

st.set_page_config(page_title="Dynamic Topic Modeling & Stance Analysis", layout="wide")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

st.title("📊 Dynamic Topic Modeling & Stance Analysis")
st.write("Memantau evolusi topik pada unggahan dan memetakan opini (stance) publik melalui komentar secara kuartalan")

# Upload dataset
uploaded_file = st.file_uploader("Upload dataset CSV", type=["csv"])

@st.cache_resource
def load_embedding_model():
    logging.info("Starting load_embedding_model")
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    logging.info("Completed load_embedding_model")
    return model

@st.cache_resource
def load_stance_model():
    logging.info("Starting load_stance_model")
    max_retries = 3
    for attempt in range(max_retries):
        try:
            model = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
            logging.info("Completed load_stance_model")
            return model
        except Exception as e:
            logging.warning(f"Attempt {attempt + 1}/{max_retries} failed to load stance model: {e}")
            if attempt < max_retries - 1:
                import time
                time.sleep(2)  # Wait before retry
            else:
                logging.error(f"Failed to load stance model after {max_retries} attempts")
                raise e


def get_file_hash(uploaded_file):
    """Return a short SHA256 hash for the uploaded file contents."""
    try:
        file_bytes = uploaded_file.read()
        uploaded_file.seek(0)
        return hashlib.sha256(file_bytes).hexdigest()[:16]
    except Exception as e:
        logging.warning(f"Could not hash uploaded file: {e}")
        return None


def get_cache_dir(file_hash):
    if not file_hash:
        return None
    cache_dir = os.path.join("results", f"cache_{file_hash}")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def load_cached_analysis(cache_dir):
    """Load cached analysis results for a previously processed file."""
    try:
        required_files = [
            "analysis_complete.json",
            "posts_preprocessed.csv",
            "comments_preprocessed.csv",
            "topic_model.pkl",
            "topic_validation.csv",
            "topic_docs_mapping.json"
        ]
        for filename in required_files:
            if not os.path.exists(os.path.join(cache_dir, filename)):
                logging.info(f"Cached analysis missing required file: {filename}")
                return None

        posts_df = pd.read_csv(os.path.join(cache_dir, "posts_preprocessed.csv"))
        comments_df = pd.read_csv(os.path.join(cache_dir, "comments_preprocessed.csv"))
        with open(os.path.join(cache_dir, "topic_model.pkl"), "rb") as f:
            topic_model = pickle.load(f)
        topic_validation_df = pd.read_csv(os.path.join(cache_dir, "topic_validation.csv"))
        with open(os.path.join(cache_dir, "topic_docs_mapping.json"), "r", encoding="utf-8") as f:
            topic_docs_mapping = json.load(f)

        return {
            'posts_df': posts_df,
            'comments_df': comments_df,
            'topic_model': topic_model,
            'topic_validation_df': topic_validation_df,
            'topic_docs_mapping': topic_docs_mapping
        }
    except Exception as e:
        logging.warning(f"Could not load cached analysis: {e}")
        return None


def save_analysis_cache(cache_dir, posts_df, comments_df, topic_model, topic_validation_df, topic_docs_mapping):
    if not cache_dir:
        return

    posts_df.to_csv(os.path.join(cache_dir, "posts_preprocessed.csv"), index=False)
    comments_df.to_csv(os.path.join(cache_dir, "comments_preprocessed.csv"), index=False)

    try:
        with open(os.path.join(cache_dir, "topic_model.pkl"), "wb") as f:
            pickle.dump(topic_model, f)
    except Exception as e:
        logging.warning(f"Could not pickle topic model: {e}")

    topic_validation_df.to_csv(os.path.join(cache_dir, "topic_validation.csv"), index=False)
    with open(os.path.join(cache_dir, "topic_docs_mapping.json"), "w", encoding="utf-8") as f:
        json.dump(topic_docs_mapping, f)

    with open(os.path.join(cache_dir, "analysis_complete.json"), "w", encoding="utf-8") as f:
        json.dump({
            'saved_at': datetime.now().isoformat(),
            'cached': True
        }, f)


def cached_fit_transform(_topic_model, _docs):
    """Wrapper for BERTopic fit_transform with data validation"""
    logging.info(f"Starting fit_transform on {len(_docs)} documents")
    
    # Validate input data
    if not _docs or len(_docs) == 0:
        raise ValueError("No documents provided for topic modeling")
    
    # Filter out empty strings
    valid_docs = [doc for doc in _docs if isinstance(doc, str) and len(doc.strip()) > 0]
    if len(valid_docs) == 0:
        raise ValueError("All documents are empty after filtering")
    
    if len(valid_docs) < len(_docs):
        logging.warning(f"Filtered out {len(_docs) - len(valid_docs)} empty documents. Using {len(valid_docs)} valid documents.")
    
    try:
        topics, probs = _topic_model.fit_transform(valid_docs)
        logging.info(f"Completed fit_transform: {len(set(topics))} topics found")
        return topics, probs
    except ValueError as e:
        error_str = str(e)
        if "0 sample" in error_str:
            logging.error(f"HDBSCAN received 0 samples. This may indicate dataset too small or all documents became outliers.")
            raise ValueError(
                f"Dataset terlalu kecil atau preprocessing menghilangkan semua vocabulary. "
                f"Pastikan dataset memiliki cukup dokumen dengan kata-kata yang bermakna. "
                f"Error: {error_str}"
            )
        elif "k must be less than" in error_str or "n_neighbors" in error_str:
            logging.error(f"KNN/UMAP error with dataset of size {len(valid_docs)}: {error_str}")
            raise ValueError(
                f"Dataset terlalu kecil untuk UMAP KNN. Silakan gunakan dataset dengan minimal 5 dokumen. "
                f"Dokumen saat ini: {len(valid_docs)}. Error: {error_str}"
            )
        raise

def cached_topics_over_time(_topic_model, _docs, _timestamps, _nr_bins=20):
    """Wrapper for BERTopic topics_over_time calculation"""
    logging.info("Calculating topics over time")
    try:
        result = _topic_model.topics_over_time(_docs, _timestamps, nr_bins=_nr_bins)
    except ValueError as e:
        logging.warning(f"Error with nr_bins={_nr_bins}: {e}. Retrying with nr_bins=10.")
        result = _topic_model.topics_over_time(_docs, _timestamps, nr_bins=10)
    logging.info("Completed topics over time calculation")
    return result

def calculate_topic_coherence(_topic_model, _docs, coherence_type='c_v'):
    """
    Calculate topic coherence using gensim CoherenceModel if available, else fallback.
    
    Args:
        _topic_model: Fitted BERTopic model
        _docs: List of preprocessed documents
        coherence_type: Type of coherence measure ('c_v', 'c_uci', 'c_npmi', 'u_mass')
    
    Returns:
        dict: Coherence scores for each topic and overall average
    """
    logging.info(f"Calculating topic coherence with {coherence_type} measure")
    
    gensim_available = False
    try:
        from gensim.models import CoherenceModel as GenSimCoherenceModel
        from gensim.corpora import Dictionary as GenSimDictionary
        gensim_available = True
        CoherenceModel = GenSimCoherenceModel
        Dictionary = GenSimDictionary
    except ImportError as e:
        logging.warning(f"Gensim not available: {e}. Falling back to native coherence approximation.")
        gensim_available = False
    
    # Get topics from BERTopic model
    topics = _topic_model.get_topics()
    topics = {k: v for k, v in topics.items() if k != -1}
    
    if not topics:
        logging.warning("No valid topics found for coherence calculation")
        return {"error": "No valid topics found"}
    
    tokenized_docs = [doc.split() for doc in _docs if doc.strip()]
    if not tokenized_docs:
        logging.warning("No valid documents passed for coherence calculation")
        return {"error": "No valid documents provided"}
    
    topic_words = []
    for topic_id in sorted(topics.keys()):
        words = [word for word, _ in topics[topic_id][:10]]
        topic_words.append(words)
    
    if gensim_available:
        try:
            dictionary = Dictionary(tokenized_docs)
            dictionary.filter_extremes(no_below=5, no_above=0.5)
            corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]
            
            # Check if we have valid data
            if len(dictionary) == 0 or len(corpus) == 0:
                raise ValueError("Dictionary or corpus is empty")
            
            coherence_model = CoherenceModel(
                topics=topic_words,
                texts=tokenized_docs,
                corpus=corpus,
                dictionary=dictionary,
                coherence=coherence_type
            )
            topic_coherences = coherence_model.get_coherence_per_topic()
            overall_coherence = coherence_model.get_coherence()
            
            # Validate coherence score
            if not isinstance(overall_coherence, (int, float)) or np.isnan(overall_coherence):
                raise ValueError(f"Invalid coherence score: {overall_coherence}")
            
            results = {
                'overall_coherence': overall_coherence,
                'topic_coherences': topic_coherences,
                'coherence_type': coherence_type,
                'num_topics': len(topic_words),
                'coherence_backend': 'gensim'
            }
            logging.info(f"Coherence calculation completed with gensim: {overall_coherence:.4f}")
            return results
        except Exception as e:
            logging.warning(f"Gensim coherence failed with error: {e}, falling back to native approximation")
    
    # Fallback coherence calculation without gensim
    logging.info("Using fallback coherence calculation")
    cooccurrence_counts = {}
    doc_freq = {}
    for doc in tokenized_docs:
        unique_words = set(doc)
        for word in unique_words:
            doc_freq[word] = doc_freq.get(word, 0) + 1
        for i in range(len(doc)):
            for j in range(i + 1, len(doc)):
                pair = tuple(sorted([doc[i], doc[j]]))
                cooccurrence_counts[pair] = cooccurrence_counts.get(pair, 0) + 1
    
    topic_coherences = []
    for words in topic_words:
        if len(words) < 2:
            topic_coherences.append(0.0)
            continue
        pair_scores = []
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                w1, w2 = words[i], words[j]
                pair = tuple(sorted([w1, w2]))
                co_occur = cooccurrence_counts.get(pair, 0)
                freq1 = doc_freq.get(w1, 0)
                freq2 = doc_freq.get(w2, 0)
                if freq1 > 0 and freq2 > 0:
                    # Use positive PMI-like score
                    pmi = math.log((co_occur + 1) / (freq1 * freq2 + 1))
                    score = max(0, pmi)  # Ensure non-negative
                else:
                    score = 0.0
                pair_scores.append(score)
        avg_score = sum(pair_scores) / len(pair_scores) if pair_scores else 0.0
        topic_coherences.append(avg_score)
    
    overall_coherence = sum(topic_coherences) / len(topic_coherences) if topic_coherences else 0.0
    
    logging.info(f"Fallback coherence calculation completed: {overall_coherence:.4f}")
    return {
        'overall_coherence': overall_coherence,
        'topic_coherences': topic_coherences,
        'coherence_type': coherence_type,
        'num_topics': len(topic_words),
        'coherence_backend': 'fallback'
    }

@st.cache_data
def calculate_topic_metrics(_topic_model, _docs):
    """
    Calculate additional topic modeling evaluation metrics
    
    Args:
        _topic_model: Fitted BERTopic model
        _docs: List of preprocessed documents
    
    Returns:
        dict: Various topic modeling metrics
    """
    logging.info("Calculating additional topic metrics")
    
    try:
        # Get topics info
        topics_info = _topic_model.get_topic_info()
        topics = _topic_model.get_topics()
        
        # Filter out outlier topic
        topics_info = topics_info[topics_info['Topic'] != -1]
        topics = {k: v for k, v in topics.items() if k != -1}
        
        if topics_info.empty:
            return {"error": "No valid topics found"}
        
        # Calculate topic diversity (unique words per topic)
        topic_diversities = []
        all_words = set()
        
        for topic_id, words_weights in topics.items():
            topic_words = [word for word, _ in words_weights[:10]]
            topic_diversities.append(len(set(topic_words)))
            all_words.update(topic_words)
        
        avg_topic_diversity = np.mean(topic_diversities)
        total_unique_words = len(all_words)
        
        # Calculate topic size distribution
        topic_sizes = topics_info['Count'].values
        topic_size_std = np.std(topic_sizes)
        topic_size_cv = topic_size_std / np.mean(topic_sizes) if np.mean(topic_sizes) > 0 else 0
        
        # Calculate document coverage
        total_docs = len(_docs)
        covered_docs = topics_info['Count'].sum()
        doc_coverage = covered_docs / total_docs if total_docs > 0 else 0
        
        # Calculate average topic size
        avg_topic_size = np.mean(topic_sizes)
        
        results = {
            'num_topics': len(topics),
            'avg_topic_diversity': avg_topic_diversity,
            'total_unique_words': total_unique_words,
            'topic_size_std': topic_size_std,
            'topic_size_cv': topic_size_cv,  # Coefficient of variation
            'doc_coverage': doc_coverage,
            'avg_topic_size': avg_topic_size,
            'topic_sizes': topic_sizes.tolist(),
            'topic_diversities': topic_diversities
        }
        
        logging.info(f"Topic metrics calculated: {len(topics)} topics, diversity: {avg_topic_diversity:.2f}")
        return results
        
    except Exception as e:
        logging.error(f"Error calculating topic metrics: {e}")
        return {"error": str(e)}


def get_time_window_slices(posts_df, months):
    """Return time windows and topic distribution for each window."""
    if 'created_at' not in posts_df.columns or 'Topik' not in posts_df.columns:
        return None

    df = posts_df.copy()
    # Handle datetime conversion more robustly
    if not pd.api.types.is_datetime64_any_dtype(df['created_at']):
        df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce', utc=True)
    else:
        # If already datetime, preserve timezone-awareness if present and normalize
        if df['created_at'].dt.tz is None:
            df['created_at'] = df['created_at'].dt.tz_localize('UTC')
        else:
            df['created_at'] = df['created_at'].dt.tz_convert('UTC')
        df['created_at'] = df['created_at'].dt.tz_localize(None)

    df = df.dropna(subset=['created_at'])
    if df.empty:
        return None

    windows = []
    start = df['created_at'].min()
    end_date = df['created_at'].max()

    while start < end_date:
        end = start + pd.DateOffset(months=months)
        window = df[(df['created_at'] >= start) & (df['created_at'] < end)]
        topic_counts = window['Topik'].value_counts(dropna=True)
        top_topics = ', '.join([f"{int(t)}({c})" for t, c in topic_counts.head(3).items()]) if not topic_counts.empty else "-"
        windows.append({
            'window_start': start,
            'window_end': end,
            'document_count': len(window),
            'top_topics': top_topics,
            'topic_diversity': window['Topik'].nunique()
        })
        start = end

    return pd.DataFrame(windows)


def render_time_frame_comparison(posts_df, months=3):
    """Render a comparison of topic activity over time windows."""
    if posts_df is None or posts_df.empty:
        st.info("Tidak ada data untuk analisis perbandingan periode waktu.")
        return

    if 'created_at' not in posts_df.columns or 'Topik' not in posts_df.columns:
        st.warning("Data tidak memiliki kolom 'created_at' atau 'Topik' yang diperlukan untuk perbandingan periode waktu.")
        return

    df = posts_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df['created_at']):
        df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce', utc=True)
    else:
        if df['created_at'].dt.tz is None:
            df['created_at'] = df['created_at'].dt.tz_localize('UTC')
        else:
            df['created_at'] = df['created_at'].dt.tz_convert('UTC')
        df['created_at'] = df['created_at'].dt.tz_localize(None)

    df = df.dropna(subset=['created_at'])
    if df.empty:
        st.warning("Tidak ada data waktu yang valid untuk analisis perbandingan periode waktu.")
        return

    windows_df = get_time_window_slices(df, months=months)
    if windows_df is None or windows_df.empty:
        st.warning("Tidak dapat membuat window waktu untuk data yang diberikan.")
        return

    windows_df = windows_df.sort_values('window_start')
    windows_df['window_label'] = windows_df.apply(
        lambda row: f"{row['window_start'].strftime('%Y-%m-%d')} sampai {row['window_end'].strftime('%Y-%m-%d')}",
        axis=1
    )

    st.subheader(f"⏳ Perbandingan Periode {months} Bulan")
    st.markdown(
        "Analisis ini menunjukkan jumlah unggahan dan keragaman topik di setiap jendela waktu. "
        "Gunakan ini untuk melihat pola perubahan topik sepanjang waktu."
    )
    st.dataframe(
        windows_df[['window_label', 'document_count', 'topic_diversity', 'top_topics']],
        use_container_width=True
    )

    count_fig = px.bar(
        windows_df,
        x='window_label',
        y='document_count',
        labels={'window_label': 'Periode', 'document_count': 'Jumlah Post'},
        title='Jumlah Post per Periode Waktu',
        text='document_count'
    )
    count_fig.update_layout(xaxis_tickangle=-45, margin=dict(t=45, b=150))
    st.plotly_chart(count_fig, use_container_width=True)

    diversity_fig = px.line(
        windows_df,
        x='window_label',
        y='topic_diversity',
        markers=True,
        labels={'window_label': 'Periode', 'topic_diversity': 'Keragaman Topik'},
        title='Keragaman Topik per Periode Waktu'
    )
    diversity_fig.update_layout(xaxis_tickangle=-45, margin=dict(t=45, b=150))
    st.plotly_chart(diversity_fig, use_container_width=True)


def get_top_topics_per_period(posts_df, topic_model, months_per_period=3):
    """Get top topics for each time period."""
    if topic_model is None or posts_df.empty:
        return None
    
    # Ensure created_at is datetime
    df = posts_df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df['created_at']):
        df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce', utc=True)
    else:
        if df['created_at'].dt.tz is None:
            df['created_at'] = df['created_at'].dt.tz_localize('UTC')
        else:
            df['created_at'] = df['created_at'].dt.tz_convert('UTC')
        df['created_at'] = df['created_at'].dt.tz_localize(None)
    
    df = df.dropna(subset=['created_at'])
    if df.empty:
        return None
    
    # Get time periods
    start_date = df['created_at'].min()
    end_date = df['created_at'].max()
    
    periods = []
    current_start = start_date
    
    while current_start < end_date:
        current_end = current_start + pd.DateOffset(months=months_per_period)
        period_data = df[(df['created_at'] >= current_start) & (df['created_at'] < current_end)]
        
        if not period_data.empty:
            # Count topics in this period
            topic_counts = period_data['Topik'].value_counts()
            
            # Get topic info for this period
            period_topics = []
            for topic_id, count in topic_counts.head(10).items():  # Top 10 topics per period
                if topic_id != -1:  # Skip outlier topic
                    topic_info = topic_model.get_topic_info()
                    topic_row = topic_info[topic_info['Topic'] == topic_id]
                    if not topic_row.empty:
                        period_topics.append({
                            'Topic': topic_id,
                            'Name': topic_row['Name'].iloc[0],
                            'Count': count,
                            'Percentage': (count / len(period_data)) * 100
                        })
            
            if period_topics:
                periods.append({
                    'period_start': current_start,
                    'period_end': current_end,
                    'total_posts': len(period_data),
                    'topics': period_topics
                })
        
        current_start = current_end
    
    return periods

    st.markdown("**Rekomendasi cepat:** jika setiap window 2 bulan memiliki sangat sedikit dokumen, gunakan 3 bulan untuk stabilitas topik; jika 2 bulan masih punya >100 dokumen per window, gunakan 2 bulan untuk deteksi perubahan lebih cepat.")


def render_vertical_report(posts_df, comments_df, topic_model, months_per_period=3):
    """
    Render laporan vertikal yang bersifat linear (cocok untuk print/PDF):
    Periode (3 bulan) -> Topik -> Postingan utama -> Tabel komentar + Stance
    """
    if posts_df is None or posts_df.empty:
        st.info("Tidak ada data postingan untuk membuat laporan vertikal.")
        return

    df_posts = posts_df.copy()
    df_comments = comments_df.copy() if comments_df is not None else pd.DataFrame()

    # Pastikan kolom waktu dalam datetime dan buat kuartal
    if not pd.api.types.is_datetime64_any_dtype(df_posts['created_at']):
        df_posts['created_at'] = pd.to_datetime(df_posts['created_at'], errors='coerce')
    df_posts = df_posts.dropna(subset=['created_at'])
    df_posts['period_q'] = df_posts['created_at'].dt.to_period('Q')

    # Jika comments memiliki timestamps, gunakan; jika tidak, hanya join by conversation_id_str
    if 'created_at' in df_comments.columns and not pd.api.types.is_datetime64_any_dtype(df_comments['created_at']):
        try:
            df_comments['created_at'] = pd.to_datetime(df_comments['created_at'], errors='coerce')
        except Exception:
            pass

    periods = sorted(df_posts['period_q'].unique())
    if not periods:
        st.info("Tidak ada periode waktu yang valid untuk laporan.")
        return

    # Helper untuk pewarnaan stance
    def _style_stance(val):
        if pd.isna(val):
            return ''
        v = str(val).strip().lower()
        if v in ('support', 'pro', 'positive', 'probable_positive', 'for', 'favor', 'supporting'):
            return 'background-color: #d4f8e8; color: #084f3f'
        if v in ('oppose', 'negative', 'contra', 'neg', 'against'):
            return 'background-color: #ffd6d6; color: #7a1f1f'
        if v in ('neutral', 'netral', 'none'):
            return 'background-color: #eef6fb; color: #0b4f6c'
        return ''

    for period in periods:
        period_mask = df_posts['period_q'] == period
        period_posts = df_posts[period_mask].sort_values('created_at')
        if period_posts.empty:
            continue

        # Header periode
        period_label = f"Periode: {period.start_time.strftime('%b %Y')} - {period.end_time.strftime('%b %Y')}"
        st.markdown(
            f"""
            <div style="border:1px solid #d1d5db; border-radius:12px; padding:14px 18px; margin-bottom:12px; background-color:#ffffff;">
                <strong style="font-size:1.1rem; color:#111827;">{html.escape(period_label)}</strong>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Get topics present in this period
        topics_in_period = period_posts['Topik'].dropna().unique().tolist()
        if not topics_in_period:
            st.info("Tidak ada topik terdeteksi pada periode ini.")
            st.markdown("---")
            continue

        for topic_id in sorted(topics_in_period):
            if int(topic_id) == -1:
                continue
            # Topic title if available
            topic_name = None
            try:
                topic_info = topic_model.get_topic_info()
                row = topic_info[topic_info['Topic'] == int(topic_id)]
                if not row.empty:
                    topic_name = row['Name'].iloc[0]
            except Exception:
                topic_name = None

            st.subheader(f"Topik {int(topic_id)}" + (f" - {topic_name}" if topic_name else ""))

            # Pilih postingan utama: gunakan postingan dengan jumlah komentar terbanyak dalam periode dan topik
            posts_for_topic = period_posts[period_posts['Topik'] == topic_id]
            if posts_for_topic.empty:
                st.info("Tidak ada postingan untuk topik ini pada periode ini.")
                continue

            # Hitung jumlah komentar per postingan jika tersedia
            if not df_comments.empty and 'conversation_id_str' in df_comments.columns:
                comment_counts = df_comments.groupby('conversation_id_str').size().to_dict()
                posts_for_topic['num_comments'] = posts_for_topic['conversation_id_str'].map(lambda x: comment_counts.get(x, 0))
                main_post = posts_for_topic.sort_values('num_comments', ascending=False).iloc[0]
            else:
                main_post = posts_for_topic.iloc[0]

            # Tampilkan postingan utama
            st.markdown("**Postingan Utama:**")
            st.info(main_post.get('full_text', '')[:10000])

            # Ambil komentar terkait
            if not df_comments.empty and 'conversation_id_str' in df_comments.columns:
                related_comments = df_comments[df_comments['conversation_id_str'] == main_post['conversation_id_str']].copy()
            else:
                related_comments = pd.DataFrame()

            if related_comments.empty:
                st.write("Tidak ada komentar terkait untuk postingan ini.")
                st.markdown("---")
                continue

            # Tampilkan komentar satu per satu dengan badge stance
            lower_cols = {col.lower(): col for col in related_comments.columns}
            comment_col = None
            for search_name in ['full_text_comments', 'full_text_comments_preprocessed', 'comment', 'full_text']:
                if search_name in lower_cols:
                    comment_col = lower_cols[search_name]
                    break
            if comment_col is None:
                # Coba kolom yang berisi kata 'comment' atau 'komentar'
                for col in related_comments.columns:
                    if 'comment' in col.lower() or 'komentar' in col.lower():
                        comment_col = col
                        break

            stance_col = None
            for search_name in ['stance', 'sentiment', 'post_stance', 'model_stance']:
                if search_name in lower_cols:
                    stance_col = lower_cols[search_name]
                    break
            if stance_col is None:
                for col in related_comments.columns:
                    if 'stance' in col.lower() or 'sentimen' in col.lower() or 'sentiment' in col.lower():
                        stance_col = col
                        break

            if comment_col is None:
                st.write("Komentar tersedia tetapi kolom teks komentar tidak ditemukan.")
                st.markdown("---")
                continue

            st.markdown("**Komentar dan Analisis Stance:**")
            for _, row in related_comments.iterrows():
                komentar = str(row.get(comment_col, '') or '').strip()
                stance_val = str(row.get(stance_col, '') or '').strip() if stance_col else ''

                # Format dan map stance value ke badge yang konsisten (support/oppose/neutral)
                stance_key = stance_val.lower().strip()
                if stance_key in ['pro', 'positive', 'pos', 'positif', 'support', 'supporting', 'for', 'favor']:
                    bg_color = '#28a745'
                    text_color = '#ffffff'
                    stance_label = 'support'
                elif stance_key in ['kontra', 'negative', 'neg', 'negatif', 'contra', 'oppose', 'against']:
                    bg_color = '#dc3545'
                    text_color = '#ffffff'
                    stance_label = 'oppose'
                elif stance_key in ['neutral', 'netral', 'none']:
                    bg_color = '#ffc107'
                    text_color = '#212529'
                    stance_label = 'neutral'
                else:
                    bg_color = '#ffc107'
                    text_color = '#212529'
                    stance_label = 'Menunggu Analisis' if stance_val == '' else stance_val

                safe_comment = html.escape(komentar).replace('\n', '<br>')
                safe_stance = html.escape(stance_label)

                st.markdown(f"""
                <div style="padding: 12px 0; border-bottom: 1px solid #eee; display: flex; align-items: flex-start; justify-content: space-between; gap: 15px;">
                    <div style="flex: 1; font-size: 14px; line-height: 1.5; color: #333;">
                        {safe_comment}
                    </div>
                    <div style="min-width: 90px; text-align: right;">
                        <span style="background-color: {bg_color}; color: {text_color}; padding: 6px 12px; border-radius: 999px; font-size: 12px; font-weight: 700; display: inline-block;">
                            {safe_stance}
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")



def _normalize_stance_label(label):
    label_orig = str(label)
    label = label_orig.upper()
    if label.startswith("LABEL_"):
        mapping = {
            'LABEL_0': 'oppose',
            'LABEL_1': 'neutral',
            'LABEL_2': 'support'
        }
        return mapping.get(label, str(label_orig).lower())

    if any(token in label for token in ['NEG', 'AGAINST', 'CONTRA', 'TIDAK', 'NO']):
        return 'oppose'
    if any(token in label for token in ['NEU', 'NET', 'NEUTRAL']):
        return 'neutral'
    if any(token in label for token in ['POS', 'FAVOR', 'FOR', 'SUPPORT', 'SETUJU']):
        return 'support'
    return str(label_orig).lower()


def cached_stance_analysis(_stance_model, _comments_list, _batch_size=20):
    """Cached wrapper for stance analysis on comments with confidence threshold"""
    logging.info(f"Starting cached stance analysis on {len(_comments_list)} comments")
    sentiments = []
    confidences = []
    
    total_batches = (len(_comments_list) + _batch_size - 1) // _batch_size
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * _batch_size
        end_idx = min((batch_idx + 1) * _batch_size, len(_comments_list))
        batch = _comments_list[start_idx:end_idx]
        
        batch_sentiments = _stance_model(batch)
        for sentiment in batch_sentiments:
            label = _normalize_stance_label(sentiment.get('label', 'NEUTRAL'))
            confidence = float(sentiment.get('score', 0.0))

            # Apply confidence threshold - if below 0.7, classify as neutral to reduce false positives
            if confidence < 0.7 and label != 'neutral':
                label = 'neutral'
                logging.debug(f"Low confidence {confidence:.2f} for {sentiment.get('label', '')}, reclassified as neutral")

            # Ensure label is standardized lowercase
            label = str(label).lower()
            sentiments.append(label)
            confidences.append(confidence)
    
    logging.info("Completed cached stance analysis")
    return sentiments, confidences

def preprocess_text(text):
    """
    Preprocessing teks yang komprehensif untuk Indonesian text
    
    Tahapan:
    1. Konversi ke string dan lowercase
    2. Hapus URL
    3. Hapus mention (@username)
    4. Hapus hashtag (#topic)
    5. Hapus emoji dan karakter non-ASCII
    6. Hapus angka
    7. Hapus punctuation dan special characters
    8. Hapus stopwords Bahasa Indonesia
    9. Stemming Bahasa Indonesia
    10. Hapus kata pendek yang tidak bermakna
    
    Args:
        text: Input text
    
    Returns:
        Preprocessed text
    """
    logging.info(f"Starting preprocess_text for text length: {len(str(text))}")
    if pd.isna(text):
        logging.info("Completed preprocess_text: empty text")
        return ""
    
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@(\w+)', r'\1', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = re.sub(r'[^\x00-\x7f]', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    words = [word for word in text.split() if word not in indonesia_stopwords and len(word) > 2]
    text = ' '.join(words)
    
    # Preserve semantic integrity for BERTopic by avoiding stemming
    logging.info(f"Completed preprocess_text: output length: {len(text)}")
    return text

def preprocess_dataframe(df, text_column):
    """
    Preprocessing dataframe dengan progress tracking
    
    Args:
        df: Input dataframe
        text_column: Nama kolom yang akan di-preprocess
    
    Returns:
        Dataframe dengan kolom baru yang preprocessed
    """
    logging.info(f"Starting preprocess_dataframe with {len(df)} rows, column: {text_column}")
    progress_bar = st.progress(0.0)
    status_text = st.empty()
    
    preprocessed_texts = []
    total = len(df)
    
    for idx, text in enumerate(df[text_column]):
        preprocessed = preprocess_text(text)
        preprocessed_texts.append(preprocessed)
        
        # Update progress setiap 10 item
        if (idx + 1) % max(1, total // 20) == 0:
            progress = int((idx + 1) / total * 100)
            progress_bar.progress(progress / 100)
            status_text.text(f"Preprocessing... {idx + 1}/{total} ({progress}%)")
    
    progress_bar.progress(1.0)
    status_text.text("✅ Preprocessing selesai!")
    
    logging.info(f"Completed preprocess_dataframe: processed {len(preprocessed_texts)} texts")
    return preprocessed_texts

def convert_df_to_csv(df):
    """Konversi dataframe ke CSV bytes untuk download"""
    return df.to_csv(index=False).encode('utf-8')

def convert_figure_to_html(fig):
    """Konversi Plotly figure ke HTML bytes untuk download"""
    return fig.to_html().encode('utf-8')


def analyze_viral_spread(df):
    """
    Analyze viral spread patterns by identifying duplicate posts and their metadata.
    Returns analysis of how posts spread across different users and timestamps.
    """
    logging.info("Starting viral spread analysis")
    
    if 'full_text' not in df.columns or 'created_at' not in df.columns:
        return None
    
    try:
        # Get duplicate posts
        value_counts = df['full_text'].value_counts()
        viral_posts = value_counts[value_counts > 1].reset_index()
        viral_posts.columns = ['text', 'count']
        
        if len(viral_posts) == 0:
            logging.warning("No viral posts found")
            return None
        
        # Convert timestamp
        df['created_at_parsed'] = pd.to_datetime(df['created_at'], errors='coerce')
        
        # Prepare viral posts data with spread details
        viral_data = []
        for idx, row in viral_posts.iterrows():
            text = row['text']
            count = row['count']
            
            # Get all instances of this post
            instances = df[df['full_text'] == text].copy()
            
            # Extract timeline info
            instances_sorted = instances.sort_values('created_at_parsed')
            time_span = (instances_sorted['created_at_parsed'].max() - 
                        instances_sorted['created_at_parsed'].min()).total_seconds() / 3600  # in hours
            
            # Count unique users
            unique_users = instances['username'].nunique()
            
            viral_data.append({
                'text': text[:80],
                'full_text': text,
                'total_reposts': count,
                'unique_users': unique_users,
                'time_span_hours': time_span if time_span > 0 else 0.1,
                'spread_rate': count / max(time_span, 0.1),  # reposts per hour
                'first_post': instances_sorted['created_at_parsed'].min(),
                'last_post': instances_sorted['created_at_parsed'].max()
            })
        
        viral_df = pd.DataFrame(viral_data)
        viral_df = viral_df.sort_values('total_reposts', ascending=False)
        
        logging.info(f"Found {len(viral_df)} viral posts")
        return viral_df
    
    except Exception as e:
        logging.error(f"Error in viral spread analysis: {e}")
        return None


def visualize_viral_timeline(df, viral_posts_info):
    """
    Create interactive timeline visualization of viral posts.
    Shows when viral posts are shared over time.
    """
    try:
        df['created_at_parsed'] = pd.to_datetime(df['created_at'], errors='coerce')
        
        # Create hourly distribution for top viral posts
        top_viral = viral_posts_info.head(5)
        
        fig = go.Figure()
        
        for idx, post in enumerate(top_viral.iterrows()):
            post_data = post[1]
            text = post_data['full_text']
            
            # Get all instances of this post with timestamps
            instances = df[df['full_text'] == text].copy()
            instances['hour'] = instances['created_at_parsed'].dt.floor('H')
            hourly_spread = instances.groupby('hour').size().reset_index(name='count')
            hourly_spread = hourly_spread.sort_values('hour')
            
            fig.add_trace(go.Scatter(
                x=hourly_spread['hour'],
                y=hourly_spread['count'],
                mode='markers+lines',
                name=f"Post {idx+1}: {post_data['total_reposts']}x reposts",
                marker=dict(size=8),
                hovertemplate='<b>Time:</b> %{x}<br><b>Reposts:</b> %{y}<extra></extra>'
            ))
        
        fig.update_layout(
            title="🔥 Viral Posts Timeline - Penyebaran per Jam",
            xaxis_title="Waktu (Jam)",
            yaxis_title="Jumlah Repost Baru",
            hovermode='x unified',
            height=400,
            template='plotly_white'
        )
        
        return fig
    
    except Exception as e:
        logging.error(f"Error in viral timeline: {e}")
        return None


def visualize_viral_heatmap(df):
    """
    Create heatmap showing retweet distribution by hour and day.
    """
    try:
        df['created_at_parsed'] = pd.to_datetime(df['created_at'], errors='coerce')
        df['hour'] = df['created_at_parsed'].dt.hour
        df['day'] = df['created_at_parsed'].dt.day_name()
        
        # Create pivot table for heatmap
        heatmap_data = df.groupby(['day', 'hour']).size().reset_index(name='count')
        
        # Order days properly
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        heatmap_data['day'] = pd.Categorical(heatmap_data['day'], categories=day_order, ordered=True)
        heatmap_pivot = heatmap_data.pivot(index='day', columns='hour', values='count').fillna(0)
        
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_pivot.values,
            x=heatmap_pivot.columns,
            y=heatmap_pivot.index,
            colorscale='YlOrRd',
            hovertemplate='Hari: %{y}<br>Jam: %{x}:00<br>Posts: %{z:.0f}<extra></extra>'
        ))
        
        fig.update_layout(
            title="🔥 Heatmap Aktivitas Post - Hari vs Jam",
            xaxis_title="Jam (24 jam)",
            yaxis_title="Hari",
            height=400,
            coloraxis_colorbar=dict(title="Jumlah Posts")
        )
        
        return fig
    
    except Exception as e:
        logging.error(f"Error in heatmap: {e}")
        return None


def visualize_viral_statistics(viral_posts_info):
    """
    Create statistics visualizations for viral posts.
    """
    try:
        if viral_posts_info is None or len(viral_posts_info) == 0:
            return None, None
        
        top_n = min(10, len(viral_posts_info))
        
        # Chart 1: Repost frequency
        fig1 = go.Figure(data=[
            go.Bar(
                x=viral_posts_info.head(top_n)['text'],
                y=viral_posts_info.head(top_n)['total_reposts'],
                marker_color='indianred',
                text=viral_posts_info.head(top_n)['total_reposts'],
                textposition='auto'
            )
        ])
        
        fig1.update_layout(
            title=f"🔥 Top {top_n} Most Viral Posts (Repost Count)",
            xaxis_title="Post Content (First 80 chars)",
            yaxis_title="Number of Reposts",
            height=400,
            showlegend=False,
            xaxis_tickangle=-45
        )
        
        # Chart 2: Unique users vs reposts scatter
        fig2 = go.Figure(data=go.Scatter(
            x=viral_posts_info['unique_users'],
            y=viral_posts_info['total_reposts'],
            mode='markers',
            marker=dict(
                size=viral_posts_info['spread_rate'].clip(upper=50),
                color=viral_posts_info['spread_rate'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Spread Rate<br>(posts/hr)")
            ),
            text=viral_posts_info['text'],
            hovertemplate='<b>%{text}</b><br>Users: %{x}<br>Reposts: %{y}<extra></extra>'
        ))
        
        fig2.update_layout(
            title="📊 Viral Spread Analysis - Users vs Reposts",
            xaxis_title="Unique Users",
            yaxis_title="Total Reposts",
            height=400,
            template='plotly_white'
        )
        
        return fig1, fig2
    
    except Exception as e:
        logging.error(f"Error in viral statistics: {e}")
        return None, None


def initialize_expert_validation_state():
    if 'expert_stance_annotations' not in st.session_state:
        st.session_state['expert_stance_annotations'] = []
    if 'expert_topic_annotations' not in st.session_state:
        st.session_state['expert_topic_annotations'] = []
    if 'analysis_done' not in st.session_state:
        st.session_state['analysis_done'] = False
    if 'uploaded_file_name' not in st.session_state:
        st.session_state['uploaded_file_name'] = None


def _export_validation_to_csv(data):
    if not data:
        return None
    df = pd.DataFrame(data)
    return df.to_csv(index=False).encode('utf-8')


def render_expert_validation_ui():
    initialize_expert_validation_state()

    st.header("🧑‍💼 Validasi Ahli Diplomasi")
    st.write("Gunakan antarmuka ini untuk menyimpan penilaian expert terhadap stance model dan topik diplomasi.")

    expert_name = st.text_input("Nama Expert:", key="expert_name")
    expert_org = st.text_input("Organisasi/Institusi:", key="expert_org")

    if not st.session_state['analysis_done']:
        st.warning("Jalankan analisis terlebih dahulu di mode Analisis agar hasil model tersedia untuk divalidasi.")
        return

    validation_type = st.selectbox(
        "Pilih Jenis Validasi",
        ["Validasi Stance", "Validasi Topik"],
        key="validation_type"
    )

    if validation_type == "Validasi Stance":
        comments_df = st.session_state['comments_df']
        comments_df = comments_df.reset_index(drop=False)
        selected_index = st.selectbox(
            "Pilih komentar untuk divalidasi:",
            comments_df.index,
            format_func=lambda idx: f"Komentar #{idx + 1}"
        )
        row = comments_df.loc[selected_index]

        st.subheader("Komentar untuk divalidasi")
        st.write(f"**Original Text:** {row['full_text_comments']}")
        st.write(f"**Preprocessed Text:** {row['full_text_comments_preprocessed']}")
        st.write(f"**Prediksi Model:** {row.get('stance', 'N/A')} (Confidence: {row.get('confidence', 0):.2f})")

        expert_stance = st.radio(
            "Expert Stance:",
            ["support", "oppose", "neutral"],
            horizontal=True,
            key="expert_stance_selection"
        )
        expert_confidence = st.select_slider(
            "Expert Confidence:",
            options=["Very Confident", "Confident", "Somewhat", "Low", "Ambiguous"],
            key="expert_confidence_selection"
        )
        agreement = st.checkbox("Setuju dengan prediksi model", key="expert_agreement")
        disagreement_reason = ""
        if not agreement:
            disagreement_reason = st.text_area("Alasan perbedaan:", key="expert_disagreement_reason")
        expert_notes = st.text_area("Catatan ahli:", height=120, key="expert_notes")

        if st.button("💾 Simpan Validasi Komentar", key="save_comment_validation"):
            annotation = {
                'comment_row': int(row['index']),
                'original_text': row['full_text_comments'],
                'preprocessed_text': row['full_text_comments_preprocessed'],
                'model_prediction': row.get('stance', ''),
                'model_confidence': float(row.get('confidence', 0) or 0),
                'expert_stance': expert_stance,
                'expert_confidence': expert_confidence,
                'agreement': agreement,
                'disagreement_reason': disagreement_reason,
                'expert_notes': expert_notes,
                'expert_name': expert_name,
                'expert_org': expert_org,
                'saved_at': datetime.now().isoformat()
            }
            st.session_state['expert_stance_annotations'].append(annotation)
            st.success("Validasi komentar berhasil disimpan.")
            
            # Save to CSV
            results_dir = "results"
            os.makedirs(results_dir, exist_ok=True)
            pd.DataFrame(st.session_state['expert_stance_annotations']).to_csv(os.path.join(results_dir, f"expert_stance_annotations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"), index=False)
            st.success("Validasi komentar berhasil disimpan dan diekspor ke CSV.")

        if st.session_state['expert_stance_annotations']:
            st.subheader("Hasil Validasi Komentar")
            st.dataframe(pd.DataFrame(st.session_state['expert_stance_annotations']))
            csv_data = _export_validation_to_csv(st.session_state['expert_stance_annotations'])
            if csv_data is not None:
                st.download_button(
                    label="💾 Unduh Validasi Komentar (CSV)",
                    data=csv_data,
                    file_name="expert_comment_validation.csv",
                    mime="text/csv"
                )

    else:
        topic_df = st.session_state.get('topic_validation_df', pd.DataFrame())
        topic_docs_mapping = st.session_state.get('topic_docs_mapping', {})

        if topic_df.empty:
            st.warning("Data topik tidak tersedia. Jalankan analisis terlebih dahulu.")
            return

        selected_topic = st.selectbox(
            "Pilih Topic ID untuk divalidasi:",
            topic_df['Topic'].tolist(),
            key="selected_topic_id"
        )
        topic_row = topic_df[topic_df['Topic'] == selected_topic].iloc[0]

        st.subheader(f"Topik {selected_topic}")
        st.write(f"**Top Words:** {topic_row['Top Words']}")
        st.write(f"**Topic Title:** {topic_row.get('Name', 'N/A')}")

        sample_docs = topic_docs_mapping.get(int(selected_topic), [])
        if sample_docs:
            with st.expander("Contoh dokumen topik"):
                for idx, sample in enumerate(sample_docs, 1):
                    st.write(f"{idx}. {sample}")

        relevance = st.slider("Relevance terhadap kebijakan luar negeri:", 1, 5, 3, key="topic_relevance")
        coherence = st.slider("Coherence topik:", 1, 5, 3, key="topic_coherence")
        policy_alignment = st.slider("Policy Alignment:", 1, 5, 3, key="topic_policy_alignment")
        international_context = st.slider("International Context:", 1, 5, 3, key="topic_international_context")
        interpretability = st.slider("Interpretability:", 1, 5, 3, key="topic_interpretability")
        completeness = st.slider("Completeness:", 1, 5, 3, key="topic_completeness")

        status = st.radio(
            "Status Validasi:",
            ["VALID", "NEEDS REVISION", "INVALID"],
            horizontal=True,
            key="topic_validation_status"
        )
        expert_label = st.text_input("Suggested label/topik interpretasi:", key="topic_expert_label")
        topic_notes = st.text_area("Catatan ahli:", height=120, key="topic_expert_notes")

        if st.button("💾 Simpan Validasi Topik", key="save_topic_validation"):
            topic_annotation = {
                'topic_id': int(selected_topic),
                'top_words': topic_row['Top Words'],
                'topic_name': topic_row.get('Name', ''),
                'relevance': relevance,
                'coherence': coherence,
                'policy_alignment': policy_alignment,
                'international_context': international_context,
                'interpretability': interpretability,
                'completeness': completeness,
                'status': status,
                'expert_label': expert_label,
                'expert_notes': topic_notes,
                'expert_name': expert_name,
                'expert_org': expert_org,
                'saved_at': datetime.now().isoformat()
            }
            st.session_state['expert_topic_annotations'].append(topic_annotation)
            st.success("Validasi topik berhasil disimpan.")
            
            # Save to CSV
            results_dir = "results"
            os.makedirs(results_dir, exist_ok=True)
            pd.DataFrame(st.session_state['expert_topic_annotations']).to_csv(os.path.join(results_dir, f"expert_topic_annotations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"), index=False)
            st.success("Validasi topik berhasil disimpan dan diekspor ke CSV.")

        if st.session_state['expert_topic_annotations']:
            st.subheader("Hasil Validasi Topik")
            st.dataframe(pd.DataFrame(st.session_state['expert_topic_annotations']))
            csv_data = _export_validation_to_csv(st.session_state['expert_topic_annotations'])
            if csv_data is not None:
                st.download_button(
                    label="💾 Unduh Validasi Topik (CSV)",
                    data=csv_data,
                    file_name="expert_topic_validation.csv",
                    mime="text/csv"
                )


def display_data_statistics(df):
    """Menampilkan statistik data yang menarik"""
    st.subheader("📈 Statistik Data")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Total Baris", f"{len(df):,}")
    
    with col2:
        st.metric("📝 Kolom", f"{len(df.columns)}")
    
    with col3:
        try:
            posts_count = df['full_text'].notna().sum()
            st.metric("💬 Posts", f"{posts_count:,}")
        except:
            st.metric("💬 Posts", "N/A")
    
    with col4:
        try:
            comments_count = df['full_text_comments'].notna().sum()
            st.metric("💭 Komentar", f"{comments_count:,}")
        except:
            st.metric("💭 Komentar", "N/A")
    
    # Statistik Detail
    with st.expander("📊 Detail Statistik"):
        stats_data = {
            "Metrik": [],
            "Nilai": []
        }
        
        # Total data
        stats_data["Metrik"].append("Total Baris")
        stats_data["Nilai"].append(f"{len(df):,}")
        
        # Unique posts
        try:
            unique_posts = df['full_text'].nunique()
            stats_data["Metrik"].append("Posts Unik")
            stats_data["Nilai"].append(f"{unique_posts:,}")
        except:
            pass
        
        # Duplicate posts
        try:
            duplicate_posts = len(df) - df['full_text'].nunique()
            stats_data["Metrik"].append("Posts Duplikat")
            stats_data["Nilai"].append(f"{duplicate_posts:,}")
        except:
            pass
        
        # Missing values
        try:
            posts_missing = df['full_text'].isna().sum()
            stats_data["Metrik"].append("Posts Kosong")
            stats_data["Nilai"].append(f"{posts_missing:,}")
        except:
            pass
        
        try:
            comments_missing = df['full_text_comments'].isna().sum()
            stats_data["Metrik"].append("Komentar Kosong")
            stats_data["Nilai"].append(f"{comments_missing:,}")
        except:
            pass
        
        # Panjang rata-rata post
        try:
            avg_post_length = df['full_text'].dropna().str.len().mean()
            stats_data["Metrik"].append("Rata-rata Panjang Post")
            stats_data["Nilai"].append(f"{avg_post_length:.0f} karakter")
        except:
            pass
        
        # Panjang rata-rata komentar
        try:
            avg_comment_length = df['full_text_comments'].dropna().str.len().mean()
            stats_data["Metrik"].append("Rata-rata Panjang Komentar")
            stats_data["Nilai"].append(f"{avg_comment_length:.0f} karakter")
        except:
            pass
        
        # Rentang waktu
        try:
            date_col = df['created_at']
            date_col = pd.to_datetime(date_col, errors='coerce')
            min_date = date_col.min()
            max_date = date_col.max()
            date_range = max_date - min_date
            stats_data["Metrik"].append("Rentang Waktu Data")
            stats_data["Nilai"].append(f"{date_range.days} hari")
            stats_data["Metrik"].append("Tanggal Awal")
            stats_data["Nilai"].append(str(min_date.date()))
            stats_data["Metrik"].append("Tanggal Akhir")
            stats_data["Nilai"].append(str(max_date.date()))
        except:
            pass
        
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
        
        # ========== VIRAL SPREAD METRICS DALAM DETAIL STATISTIK ==========
        st.divider()
        st.subheader("🔥 Analisis Penyebaran Viral Posts")
        
        try:
            # Kalkulasi metrics
            df_clean = df.dropna(subset=['full_text'])
            total_rows = len(df_clean)
            unique_texts = df_clean['full_text'].nunique()
            duplicate_count = total_rows - unique_texts
            
            # Hitung viral posts
            value_counts = df_clean['full_text'].value_counts()
            viral_posts = value_counts[value_counts > 1]
            num_viral_posts = len(viral_posts)
            
            # Hitung unique users yang mereposts
            if 'username' in df_clean.columns:
                total_users_viral = 0
                for text in viral_posts.index:
                    users = df_clean[df_clean['full_text'] == text]['username'].nunique()
                    total_users_viral += users
                avg_users_per_post = total_users_viral / max(num_viral_posts, 1)
            else:
                avg_users_per_post = 0
            
            # Hitung spread rate
            if 'created_at' in df_clean.columns:
                df_clean['created_at_parsed'] = pd.to_datetime(df_clean['created_at'], errors='coerce')
                max_spread_rate = 0
                for text in viral_posts.index[:5]:  # Top 5
                    instances = df_clean[df_clean['full_text'] == text]
                    instances_sorted = instances.sort_values('created_at_parsed')
                    time_span = (instances_sorted['created_at_parsed'].max() - 
                                instances_sorted['created_at_parsed'].min()).total_seconds() / 3600
                    if time_span > 0:
                        rate = len(instances) / time_span
                        max_spread_rate = max(max_spread_rate, rate)
            else:
                max_spread_rate = 0
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "🔥 Posts Viral",
                    f"{num_viral_posts}",
                    help="Unique posts yang di-retweet >1x"
                )
            
            with col2:
                st.metric(
                    "🔄 Total Reposts",
                    f"{duplicate_count:,}",
                    help="Total instances dari viral posts"
                )
            
            with col3:
                st.metric(
                    "👥 Avg Users/Post",
                    f"{avg_users_per_post:.0f}",
                    help="Rata-rata users yang mereposts per post"
                )
            
            with col4:
                st.metric(
                    "⚡ Max Spread Rate",
                    f"{max_spread_rate:.1f}/hr",
                    help="Reposts per jam tertinggi"
                )
            
            # Insight text
            st.info(f"""
            💡 **Insight**: Dataset ini mengandung **{duplicate_count:,} duplikat posts** 
            yang merupakan **retweets** (bukan error data). Ini menunjukkan bahwa 
            **{num_viral_posts} pesan unik** tersebar di kalangan **{int(total_users_viral)} users berbeda**, 
            dengan kecepatan penyebaran maksimal **{max_spread_rate:.1f} posts/jam**. 
            Analisis viral spread lengkap tersedia di section "🔥 Analisis Penyebaran Viral Posts".
            """)
            
        except Exception as e:
            logging.warning(f"Error calculating viral metrics: {e}")
            st.info("💡 Viral metrics belum bisa dihitung. Jalankan analisis lengkap untuk detail lebih lanjut.")

if uploaded_file:
    file_hash = get_file_hash(uploaded_file)
    cache_dir = get_cache_dir(file_hash)
    st.session_state['uploaded_file_hash'] = file_hash
    st.session_state['cache_dir'] = cache_dir

    if st.session_state.get('uploaded_file_name') != uploaded_file.name:
        st.session_state['uploaded_file_name'] = uploaded_file.name
        st.session_state['analysis_done'] = False

    cache_available = cache_dir is not None and os.path.exists(os.path.join(cache_dir, "analysis_complete.json"))
    if cache_available and not st.session_state.get('analysis_done', False):
        st.info("Hasil analisis sebelumnya tersedia untuk dataset ini. Klik tombol untuk melanjutkan tanpa mengulang proses.")
        if st.button("🔄 Lanjutkan dari hasil terakhir", key="load_cached_analysis"):
            cache_loaded = load_cached_analysis(cache_dir)
            if cache_loaded:
                st.session_state['analysis_done'] = True
                st.session_state['posts_df'] = cache_loaded['posts_df']
                st.session_state['comments_df'] = cache_loaded['comments_df']
                st.session_state['topic_model'] = cache_loaded['topic_model']
                st.session_state['topic_validation_df'] = cache_loaded['topic_validation_df']
                st.session_state['topic_docs_mapping'] = cache_loaded['topic_docs_mapping']
                st.success("Berhasil memuat hasil analisis sebelumnya.")
            else:
                st.error("Gagal memuat cache. Analisis akan dijalankan ulang.")

    df = pd.read_csv(uploaded_file)

    st.subheader("Preview Data")
    st.dataframe(df.head())
    
    # Tampilkan statistik data
    display_data_statistics(df)

    app_mode = st.sidebar.selectbox(
        "Mode Aplikasi",
        ["Analisis", "Validasi Ahli Diplomasi"]
    )
    st.sidebar.info("Mode Validasi Ahli Diplomasi digunakan setelah analisis selesai.")

    # ========== SIDEBAR FILTERS (Available after analysis) ==========
    if st.session_state.get('analysis_done', False) and app_mode == "Analisis":
        st.sidebar.markdown("---")
        st.sidebar.header("📊 Data Filters")
        
        # Get available topics and stances
        posts_df = st.session_state.get('posts_df', pd.DataFrame())
        comments_df = st.session_state.get('comments_df', pd.DataFrame())
        
        if not posts_df.empty and not comments_df.empty:
            # Topic Filter
            available_topics = sorted([t for t in posts_df['Topik'].unique() if t != -1])
            selected_topics = st.sidebar.multiselect(
                "Filter by Topics",
                options=available_topics,
                default=available_topics,
                key="topic_filter",
                help="Select topics to include in visualizations"
            )
            
            # Stance Filter
            if 'stance' in comments_df.columns:
                available_stances = sorted(comments_df['stance'].dropna().unique())
                selected_stances = st.sidebar.multiselect(
                    "Filter by Stance",
                    options=available_stances,
                    default=available_stances,
                    key="stance_filter",
                    help="Select stances to include in visualizations"
                )
            else:
                st.sidebar.warning("⚠️ Stance analysis belum selesai. Jalankan analisis terlebih dahulu.")
                available_stances = []
                selected_stances = []
            
            # Confidence Threshold
            min_confidence = st.sidebar.slider(
                "Minimum Confidence Score",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.05,
                key="confidence_filter",
                help="Filter comments by minimum confidence score"
            )
            
            # Apply Filters Button
            if st.sidebar.button("🔄 Apply Filters", use_container_width=True):
                st.session_state['filters_applied'] = True
                st.rerun()
            
            # Reset Filters Button
            if st.sidebar.button("🔄 Reset Filters", use_container_width=True):
                st.session_state['filters_applied'] = False
                # Reset to defaults
                st.session_state['topic_filter'] = available_topics
                st.session_state['stance_filter'] = available_stances
                st.session_state['confidence_filter'] = 0.0
                st.rerun()
            
            st.sidebar.info("ℹ️ Filters apply to all visualizations below")

    if app_mode == "Analisis":
        # Asumsikan kolom: full_text (posts), created_at (timestamp), full_text_comments (comments)
        required_cols = ['full_text', 'created_at', 'full_text_comments']
        if not all(col in df.columns for col in required_cols):
            st.error(f"Dataset harus memiliki kolom: {', '.join(required_cols)}")
            st.stop()

        # Proses posts untuk topic modeling
        posts_df = df[['conversation_id_str', 'full_text', 'created_at']].dropna().drop_duplicates(subset=['conversation_id_str'])
        posts_df['created_at'] = pd.to_datetime(posts_df['created_at'])
        posts_df = posts_df.sort_values(by='created_at')

        # Proses comments untuk stance analysis
        comment_cols = ['conversation_id_str', 'full_text_comments']
        if 'expert_stance' in df.columns:
            comment_cols.append('expert_stance')
        comments_df = df[comment_cols].dropna(subset=['full_text_comments'])

        st.info("Memuat model...")
        try:
            embedding_model = load_embedding_model()
            stance_model = load_stance_model()
        except Exception as e:
            st.error(f"❌ Gagal memuat model: {str(e)}")
            st.error("Ini mungkin disebabkan oleh masalah koneksi internet atau server Hugging Face sedang sibuk.")
            st.info("💡 Solusi: Coba refresh halaman atau tunggu beberapa menit sebelum mencoba lagi.")
            st.stop()
        
        # ========== PREPROCESSING SECTION ==========
        st.divider()
        st.subheader("🧹 Data Preprocessing")
        
        preprocessing_col1, preprocessing_col2 = st.columns(2)
        
        with preprocessing_col1:
            st.markdown("**📝 Preprocessing Posts untuk Topic Modeling**")
            with st.spinner("Processing posts..."):
                preprocessed_posts = preprocess_dataframe(posts_df.reset_index(drop=True), 'full_text')
                posts_df['full_text_preprocessed'] = preprocessed_posts
        
        with preprocessing_col2:
            st.markdown("**💬 Preprocessing Comments untuk Stance Analysis**")
            with st.spinner("Processing comments..."):
                preprocessed_comments = preprocess_dataframe(comments_df.reset_index(drop=True), 'full_text_comments')
                comments_df['full_text_comments_preprocessed'] = preprocessed_comments
        
        # Tampilkan perbandingan before-after preprocessing
        with st.expander("👁️ Lihat Contoh Preprocessing"):
            st.subheader("📝 Contoh: Posts Preprocessing")
            
            sample_idx = 0 if len(posts_df) > 0 else 0
            if len(posts_df) > 0:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**SEBELUM:**")
                    st.text_area("Original Text", posts_df['full_text'].iloc[sample_idx], height=100, disabled=True)
                
                with col2:
                    st.markdown("**SESUDAH:**")
                    st.text_area("Preprocessed Text", posts_df['full_text_preprocessed'].iloc[sample_idx], height=100, disabled=True)
            
            st.markdown("---")
            st.subheader("💬 Contoh: Comments Preprocessing")
            
            if len(comments_df) > 0:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**SEBELUM:**")
                    st.text_area("Original Comment", comments_df['full_text_comments'].iloc[0], height=100, disabled=True, key="comment_before")
                
                with col2:
                    st.markdown("**SESUDAH:**")
                    st.text_area("Preprocessed Comment", comments_df['full_text_comments_preprocessed'].iloc[0], height=100, disabled=True, key="comment_after")
        
        # Gunakan preprocessed text untuk analisis
        docs = posts_df['full_text_preprocessed'].astype(str).tolist()
        timestamps = posts_df['created_at'].tolist()
        
        # Validate documents - filter out empty ones
        valid_indices = [i for i, doc in enumerate(docs) if isinstance(doc, str) and len(doc.strip()) > 0]
        if len(valid_indices) == 0:
            st.error("❌ Semua dokumen kosong setelah preprocessing. Silakan periksa dataset Anda.")
            st.stop()
        elif len(valid_indices) < len(docs):
            st.warning(f"⚠️ {len(docs) - len(valid_indices)} dokumen kosong setelah preprocessing, menggunakan {len(valid_indices)} dokumen valid.")
            docs = [docs[i] for i in valid_indices]
            timestamps = [timestamps[i] for i in valid_indices]
            posts_df = posts_df.iloc[valid_indices].reset_index(drop=True)
        
        st.success("✅ Preprocessing selesai! Analisis berjalan otomatis setelah preprocessing.")
        st.divider()

        run_analysis = not st.session_state.get('analysis_done', False)
        if run_analysis:
            st.info("🔄 Menjalankan analisis otomatis. Mohon tunggu beberapa saat...")
            logging.info("Starting analysis: Topic Modeling and Stance Analysis")
            
            # Setup results directory
            results_dir = st.session_state.get('cache_dir') or "results"
            os.makedirs(results_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save original data
            df.to_csv(os.path.join(results_dir, f"original_data_{timestamp}.csv"), index=False)
            
            # ========== TOPIC MODELING ==========
            st.subheader("📊 Topic Modeling Processing")
            progress_container = st.container()
            
            with progress_container:
                # Progress 1: Model Initialization
                progress_bar = st.progress(0.0)
                status_text = st.empty()
                
                status_text.text("🔄 Tahap 1/4: Inisialisasi Model...")
                progress_bar.progress(0.25)
                logging.info("Initializing BERTopic model")

                # Adaptive vectorizer parameters based on dataset size
                num_docs = len(docs)
                
                # For very small datasets, use more lenient parameters
                if num_docs < 5:
                    min_df_val = 1
                    # Use absolute count instead of percentage for small datasets
                    max_df_val = num_docs  # Don't filter any documents
                    stop_words_list = None  # Don't remove stop words for tiny datasets
                    logging.info(f"Tiny dataset detected ({num_docs} docs): minimal filtering")
                elif num_docs < 10:
                    min_df_val = 1
                    max_df_val = num_docs  # Accept all
                    stop_words_list = None  # Preserve all vocabulary
                elif num_docs < 50:
                    min_df_val = 1
                    max_df_val = max(int(num_docs * 0.95), num_docs - 1)  # Use absolute count
                    stop_words_list = list(indonesia_stopwords) if indonesia_stopwords else None
                elif num_docs < 100:
                    min_df_val = 1
                    max_df_val = max(int(num_docs * 0.9), num_docs - 2)  # Use absolute count
                    stop_words_list = list(indonesia_stopwords) if indonesia_stopwords else None
                else:
                    min_df_val = 2
                    max_df_val = int(num_docs * 0.9)  # Can use percentage for large datasets
                    stop_words_list = list(indonesia_stopwords) if indonesia_stopwords else None
                
                logging.info(f"Using adaptive vectorizer: min_df={min_df_val}, max_df={max_df_val}, {num_docs} docs")
                try:
                    vectorizer_model = CountVectorizer(
                        ngram_range=(1, 2), 
                        min_df=min_df_val, 
                        max_df=max_df_val,
                        stop_words=stop_words_list,
                        max_features=10000,
                        lowercase=True,
                        token_pattern=r'(?u)\b\w\w+\b'
                    )
                except Exception as e:
                    logging.warning(f"Error creating vectorizer with params: {e}. Falling back to minimal config.")
                    vectorizer_model = CountVectorizer(
                        ngram_range=(1, 1),
                        min_df=1,
                        max_df=num_docs,
                        stop_words=None,
                        lowercase=True,
                        token_pattern=r'(?u)\b\w\w+\b'
                    )
                
                # Adaptive UMAP and HDBSCAN parameters
                # UMAP: n_neighbors cannot be larger than dataset size - 1
                # For very small datasets, use much smaller n_neighbors
                if num_docs < 5:
                    n_neighbors_umap = 2
                    n_components_umap = 1
                    disable_topic_reduction = True
                elif num_docs < 10:
                    n_neighbors_umap = 2
                    n_components_umap = 2
                    disable_topic_reduction = True
                elif num_docs < 20:
                    n_neighbors_umap = 3
                    n_components_umap = 3
                    disable_topic_reduction = True
                elif num_docs < 50:
                    n_neighbors_umap = min(5, num_docs - 1)
                    n_components_umap = min(4, num_docs // 2)
                    disable_topic_reduction = True
                else:
                    n_neighbors_umap = min(15, max(5, num_docs - 1))
                    n_components_umap = 5
                    disable_topic_reduction = False
                
                # HDBSCAN: min_cluster_size should be adaptive
                if num_docs < 5:
                    hdbscan_min_cluster_size = 2
                    min_topic_size_bertopic = 1
                elif num_docs < 10:
                    hdbscan_min_cluster_size = 2
                    min_topic_size_bertopic = 2
                elif num_docs < 20:
                    hdbscan_min_cluster_size = 3
                    min_topic_size_bertopic = 3
                elif num_docs < 50:
                    hdbscan_min_cluster_size = 5
                    min_topic_size_bertopic = 5
                elif num_docs < 100:
                    hdbscan_min_cluster_size = 10
                    min_topic_size_bertopic = 10
                else:
                    hdbscan_min_cluster_size = 20
                    min_topic_size_bertopic = 20
                
                logging.info(f"Adaptive clustering: n_neighbors={n_neighbors_umap}, n_components={n_components_umap}, min_cluster_size={hdbscan_min_cluster_size}, disable_reduction={disable_topic_reduction}")
                
                umap_model = UMAP(
                    n_neighbors=n_neighbors_umap,
                    n_components=n_components_umap,
                    min_dist=0.0,
                    metric='cosine',
                    random_state=42
                )
                hdbscan_model = HDBSCAN(
                    min_cluster_size=hdbscan_min_cluster_size,
                    metric='euclidean',
                    cluster_selection_method='eom',
                    prediction_data=True,
                    allow_single_cluster=True
                )
                representation_model = MaximalMarginalRelevance(diversity=0.3)
                topic_model = BERTopic(
                    embedding_model=embedding_model,
                    umap_model=umap_model,
                    hdbscan_model=hdbscan_model,
                    vectorizer_model=vectorizer_model,
                    representation_model=representation_model,
                    nr_topics="auto" if not disable_topic_reduction else None,
                    min_topic_size=min_topic_size_bertopic,
                    calculate_probabilities=True,
                )
                
                # Store model in session state immediately
                st.session_state['topic_model'] = topic_model
                
                # Progress 2: Fitting & Transforming with detailed progress
                st.markdown("---")
                st.subheader("🔄 Tahap 2/4: Fitting dan Transforming Dokumen")
                
                # Create detailed progress metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    total_docs_metric = st.empty()
                    total_docs_metric.metric("📚 Total Dokumen", f"{len(docs):,}")
                
                with col2:
                    processed_metric = st.empty()
                    processed_metric.metric("✓ Diproses", "0")
                
                with col3:
                    progress_perc_metric = st.empty()
                    progress_perc_metric.metric("% Progress", "0%")
                
                with col4:
                    status_metric = st.empty()
                    status_metric.metric("Status", "Memulai...")
                
                # Main progress bar
                main_progress_bar = st.progress(0.0)
                sub_progress_bar = st.progress(0.0)
                detail_text = st.empty()
                time_info = st.empty()
                
                import time
                start_time = time.time()
                
                # Simulate progress dengan milestone
                milestones = [10, 25, 50, 75, 90, 100]
                last_milestone = 0
                
                detail_text.write("""
                <div style="padding: 10px; background-color: #f0f2f6; border-radius: 5px;">
                    <small><b>📝 Sub-tahap:</b></small><br>
                    <small>• Embedding documents...</small>
                </div>
                """, unsafe_allow_html=True)
                
                # Tahap 2A: Embedding (30% dari tahap 2)
                for i in range(30):
                    processed_metric.metric("✓ Diproses", f"{len(docs) // 100 * (i+1):,}")
                    progress_perc_metric.metric("% Progress", f"{((i+1)/100)*50:.1f}%")
                    main_progress_bar.progress((25 + (i+1)/100 * 25) / 100)
                    sub_progress_bar.progress((i+1)/100)
                    elapsed = time.time() - start_time
                    time_info.write(f"⏱️ Waktu elapsed: {elapsed:.1f}s")
                    time.sleep(0.05)
                
                detail_text.write("""
                <div style="padding: 10px; background-color: #f0f2f6; border-radius: 5px;">
                    <small><b>📝 Sub-tahap:</b></small><br>
                    <small>• Embedding documents... ✓</small><br>
                    <small>• Clustering & reducing dimensions...</small>
                </div>
                """, unsafe_allow_html=True)
                
                # Tahap 2B: Clustering & UMAP (40% dari tahap 2)
                for i in range(30, 70):
                    processed_metric.metric("✓ Diproses", f"{len(docs) // 100 * (i+1):,}")
                    progress_perc_metric.metric("% Progress", f"{((i+1)/100)*50:.1f}%")
                    main_progress_bar.progress((25 + (i+1)/100 * 25) / 100)
                    sub_progress_bar.progress((i+1-30)/70)
                    elapsed = time.time() - start_time
                    time_info.write(f"⏱️ Waktu elapsed: {elapsed:.1f}s")
                    time.sleep(0.05)
                
                detail_text.write("""
                <div style="padding: 10px; background-color: #f0f2f6; border-radius: 5px;">
                    <small><b>📝 Sub-tahap:</b></small><br>
                    <small>• Embedding documents... ✓</small><br>
                    <small>• Clustering & reducing dimensions... ✓</small><br>
                    <small>• Topic extraction & labeling...</small>
                </div>
                """, unsafe_allow_html=True)
                
                # Tahap 2C: Topic extraction (30% dari tahap 2)
                for i in range(70, 100):
                    processed_metric.metric("✓ Diproses", f"{len(docs):,}")
                    progress_perc_metric.metric("% Progress", f"{((i+1)/100)*50:.1f}%")
                    main_progress_bar.progress((25 + (i+1)/100 * 25) / 100)
                    sub_progress_bar.progress((i+1-70)/30)
                    elapsed = time.time() - start_time
                    time_info.write(f"⏱️ Waktu elapsed: {elapsed:.1f}s")
                    time.sleep(0.05)
                
                detail_text.write("""
                <div style="padding: 10px; background-color: #d4edda; border-radius: 5px; border-left: 4px solid #28a745;">
                    <small><b>✅ Tahap 2 Selesai!</b></small><br>
                    <small>• Embedding documents... ✓</small><br>
                    <small>• Clustering & reducing dimensions... ✓</small><br>
                    <small>• Topic extraction & labeling... ✓</small>
                </div>
                """, unsafe_allow_html=True)
                
                processed_metric.metric("✓ Diproses", f"{len(docs):,}")
                progress_perc_metric.metric("% Progress", "50%")
                status_metric.metric("Status", "Fit-Transform Selesai")
                
                # Jalankan actual fit_transform
                topics, probs = cached_fit_transform(topic_model, docs)
                
                # Update session state with fitted model
                st.session_state['topic_model'] = topic_model
                
                # Assign topics to posts_df
                posts_df['Topik'] = topics
                
                # Mapping Topik kembali ke dataframe utama berdasarkan conversation_id_str
                topic_mapping = dict(zip(posts_df['conversation_id_str'], posts_df['Topik']))
                df['Topik'] = df['conversation_id_str'].map(topic_mapping)

                if not comments_df.empty:
                    comments_df['Topik'] = comments_df['conversation_id_str'].map(topic_mapping)
                    comments_texts = comments_df['full_text_comments_preprocessed'].astype(str).tolist()
                    sentiments, confidences = cached_stance_analysis(stance_model, comments_texts, _batch_size=20)
                    comments_df['stance'] = sentiments
                    comments_df['confidence'] = confidences
                    comments_df['confidence'] = comments_df['confidence'].astype(float)
                else:
                    st.warning("Dataset komentar kosong. Stance analysis komentar dilewati.")
                logging.info("Calculating topics over time")
                
                # Calculate nr_bins for 3-month intervals
                if timestamps:
                    timestamps_dt = pd.to_datetime(timestamps, errors='coerce')
                    timestamps_dt = timestamps_dt.dropna()
                    if len(timestamps_dt) > 1:
                        time_range_days = (timestamps_dt.max() - timestamps_dt.min()).days
                        time_range_months = time_range_days / 30.44  # Average days per month
                        nr_bins = max(1, int(time_range_months / 3))  # 3-month intervals
                        logging.info(f"Calculated nr_bins={nr_bins} for {time_range_months:.1f} months range")
                    else:
                        nr_bins = 10
                else:
                    nr_bins = 10
                
                topics_over_time = cached_topics_over_time(topic_model, docs, timestamps, _nr_bins=nr_bins)
                
                # Progress 4: Complete
                status_text.text("✅ Tahap 4/4: Selesai!")
                progress_bar.progress(1.0)

            st.success("✅ Topic Modeling Selesai!")

            # ========== APPLY FILTERS TO DATA ==========
            if st.session_state.get('filters_applied', False):
                # Get filter values
                selected_topics = st.session_state.get('topic_filter', [])
                selected_stances = st.session_state.get('stance_filter', [])
                min_confidence = st.session_state.get('confidence_filter', 0.0)
                
                # Apply filters
                filtered_posts_df = posts_df[posts_df['Topik'].isin(selected_topics)] if selected_topics else posts_df
                if selected_stances and 'stance' in comments_df.columns and 'confidence' in comments_df.columns:
                    filtered_comments_df = comments_df[
                        (comments_df['stance'].isin(selected_stances)) &
                        (comments_df['confidence'] >= min_confidence)
                    ]
                else:
                    filtered_comments_df = comments_df

                if selected_topics and 'Topik' in filtered_comments_df.columns:
                    filtered_comments_df = filtered_comments_df[filtered_comments_df['Topik'].isin(selected_topics)]
                
                st.info(f"📊 **Filtered Results**: {len(filtered_posts_df)} posts, {len(filtered_comments_df)} comments")
            else:
                # Use unfiltered data
                filtered_posts_df = posts_df
                filtered_comments_df = comments_df

            st.subheader("📈 Topics Over Time (3-Month Intervals)")
            fig = topic_model.visualize_topics_over_time(topics_over_time)
            st.plotly_chart(fig, use_container_width=True)
            
            # Time frame comparison analysis
            render_time_frame_comparison(posts_df)
            
            # Save the plot
            fig.write_html(os.path.join(results_dir, f"topics_over_time_3month_{timestamp}.html"))
            
            # Download button untuk Topics Over Time
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="💾 Simpan Topics Over Time (3-Month Intervals)",
                    data=convert_figure_to_html(fig),
                    file_name="topics_over_time_3month_intervals.html",
                    mime="text/html",
                    key="download_topics_over_time"
                )

            # ========== WORD CLOUDS PER TOPIC ==========
            st.subheader("☁️ Word Clouds per Topic")
            
            if topic_model is not None:
                # Get available topics for word clouds
                available_topics = [topic for topic in topic_model.get_topics().keys() if topic != -1]
                
                if available_topics:
                    col1, col2 = st.columns([1, 3])
                    
                    with col1:
                        selected_topic_wc = st.selectbox(
                            "Select Topic for Word Cloud",
                            options=available_topics,
                            format_func=lambda x: f"Topic {x}: {topic_model.get_topic_info(x)['Name'].iloc[0] if x in topic_model.get_topic_info()['Topic'].values else f'Topic {x}'}",
                            key="wordcloud_topic_selector"
                        )
                    
                    with col2:
                        if selected_topic_wc is not None:
                            # Get topic words and weights
                            topic_words = topic_model.get_topic(selected_topic_wc)
                            if topic_words:
                                # Create word cloud
                                word_freq = {word: weight for word, weight in topic_words}
                                
                                # Generate word cloud
                                wordcloud = WordCloud(
                                    width=800, 
                                    height=400, 
                                    background_color='white',
                                    colormap='viridis',
                                    max_words=50
                                ).generate_from_frequencies(word_freq)
                                
                                # Display word cloud
                                fig, ax = plt.subplots(figsize=(10, 5))
                                ax.imshow(wordcloud, interpolation='bilinear')
                                ax.axis('off')
                                ax.set_title(f'Word Cloud for Topic {selected_topic_wc}', fontsize=16, pad=20)
                                st.pyplot(fig)
                                
                                # Save the word cloud
                                fig.savefig(os.path.join(results_dir, f"wordcloud_topic_{selected_topic_wc}_{timestamp}.png"))
                                
                                # Show top words as text
                                with st.expander("📝 Top Words & Weights"):
                                    words_df = pd.DataFrame(topic_words, columns=['Word', 'Weight'])
                                    st.dataframe(words_df.head(20), use_container_width=True)
                            else:
                                st.info("No words available for this topic.")
                else:
                    st.info("No topics available for word clouds.")
            else:
                st.info("Topic model not available. Please run analysis first.")

            st.subheader("📌 Top Topics per Periode (3 Bulan)")
            if topic_model is not None:
                periods_data = get_top_topics_per_period(posts_df, topic_model, months_per_period=3)
                
                if periods_data:
                    for i, period in enumerate(periods_data):
                        period_start = period['period_start'].strftime('%Y-%m-%d')
                        period_end = period['period_end'].strftime('%Y-%m-%d')
                        
                        st.markdown(f"**Periode {i+1}: {period_start} sampai {period_end}**")
                        st.markdown(f"*Total posts: {period['total_posts']}*")
                        
                        # Create dataframe for this period
                        period_df = pd.DataFrame(period['topics'])
                        if not period_df.empty:
                            period_df = period_df[['Topic', 'Name', 'Count', 'Percentage']].round({'Percentage': 1})
                            st.dataframe(period_df, use_container_width=True)
                        else:
                            st.info("Tidak ada topik untuk periode ini.")
                        
                        # Add some spacing between periods
                        if i < len(periods_data) - 1:
                            st.markdown("---")
                else:
                    st.warning("Tidak ada data periode yang cukup.")
                    
                # Overall top topics (for reference)
                st.markdown("### 📊 Top Topics Keseluruhan")
                top_topics_df = topic_model.get_topic_info()
                st.dataframe(top_topics_df, use_container_width=True)

                # Tambahkan laporan vertikal untuk keperluan print/PDF
                st.divider()
                st.subheader("📄 Laporan Vertikal (Printable)")
                try:
                    rp_posts = filtered_posts_df if 'filtered_posts_df' in locals() else posts_df
                    rp_comments = filtered_comments_df if 'filtered_comments_df' in locals() else comments_df
                    render_vertical_report(rp_posts, rp_comments, topic_model, months_per_period=3)
                except Exception as e:
                    logging.warning(f"Gagal menampilkan laporan vertikal: {e}")

                st.subheader("🔗 Hubungan Post - Topik - Stance")
                if 'stance' in df.columns:
                    topic_stance = df.dropna(subset=['Topik', 'stance']).groupby(['Topik', 'stance']).size().reset_index(name='Jumlah')
                    st.dataframe(topic_stance, use_container_width=True)
                    st.markdown("**Insight:** jumlah post per topik dan stance komentar membantu melihat topik mana yang memicu dukungan, penolakan, atau netralitas.")
                else:
                    st.info("Stance belum tersedia. Jalankan stance analysis untuk melihat hubungan post-topik-stance.")
            else:
                st.error("Topic model is not available. Please run the analysis first.")
                top_topics_df = pd.DataFrame()  # Create empty dataframe to prevent further errors

            # Persist results for expert validation
            st.session_state['analysis_done'] = True
            st.session_state['comments_df'] = comments_df.copy()
            st.session_state['posts_df'] = posts_df.copy()
            st.session_state['topic_model'] = topic_model
            topic_validation_df = top_topics_df[top_topics_df['Topic'] != -1][['Topic', 'Name']].copy()
            try:
                topic_validation_df['Top Words'] = topic_validation_df['Topic'].apply(
                    lambda topic_id: ", ".join([word for word, _ in topic_model.get_topic(int(topic_id))[:10] if isinstance(_, (int, float))])
                )
            except Exception as e:
                logging.warning(f"Could not extract top words: {e}")
                topic_validation_df['Top Words'] = "N/A"
            st.session_state['topic_validation_df'] = topic_validation_df
            topic_docs_mapping = {}
            for topic_id, group in posts_df.groupby('Topik'):
                if topic_id == -1:
                    continue
                topic_docs_mapping[int(topic_id)] = group['full_text_preprocessed'].head(5).tolist()
            st.session_state['topic_docs_mapping'] = topic_docs_mapping
            
            # Save all results to files
            
            # Save posts with topics
            posts_df.to_csv(os.path.join(results_dir, f"posts_with_topics_{timestamp}.csv"), index=False)
            
            # Save comments with stance
            comments_df.to_csv(os.path.join(results_dir, f"comments_with_stance_{timestamp}.csv"), index=False)
            
            # Save top topics per period
            if periods_data:
                with open(os.path.join(results_dir, f"top_topics_per_period_{timestamp}.json"), "w", encoding="utf-8") as f:
                    # Convert datetime objects to strings for JSON serialization
                    serializable_periods = []
                    for period in periods_data:
                        serializable_period = {
                            'period_start': period['period_start'].isoformat(),
                            'period_end': period['period_end'].isoformat(),
                            'total_posts': period['total_posts'],
                            'topics': period['topics']
                        }
                        serializable_periods.append(serializable_period)
                    json.dump(serializable_periods, f, indent=2, ensure_ascii=False)
                
                # Also save as CSV format for easier analysis
                all_period_topics = []
                for i, period in enumerate(periods_data):
                    period_start = period['period_start'].strftime('%Y-%m-%d')
                    period_end = period['period_end'].strftime('%Y-%m-%d')
                    for topic in period['topics']:
                        all_period_topics.append({
                            'Period': f"{i+1} ({period_start} to {period_end})",
                            'Topic_ID': topic['Topic'],
                            'Topic_Name': topic['Name'],
                            'Count': topic['Count'],
                            'Percentage': round(topic['Percentage'], 1)
                        })
                
                if all_period_topics:
                    period_topics_df = pd.DataFrame(all_period_topics)
                    period_topics_df.to_csv(os.path.join(results_dir, f"top_topics_per_period_{timestamp}.csv"), index=False)
            
            # Save topic validation
            topic_validation_df.to_csv(os.path.join(results_dir, f"topic_validation_{timestamp}.csv"), index=False)
            
            # Save cached analysis files for faster resume
            save_analysis_cache(results_dir, posts_df.copy(), comments_df.copy(), topic_model, topic_validation_df, topic_docs_mapping)

            # Calculate and save coherence
            coherence_results = calculate_topic_coherence(topic_model, docs)
            with open(os.path.join(results_dir, f"coherence_results_{timestamp}.json"), "w", encoding="utf-8") as f:
                json.dump(coherence_results, f)
            
            # Calculate and save topic metrics
            topic_metrics = calculate_topic_metrics(topic_model, docs)
            with open(os.path.join(results_dir, f"topic_metrics_{timestamp}.json"), "w", encoding="utf-8") as f:
                json.dump(topic_metrics, f)
            
            st.success(f"✅ Semua hasil analisis telah disimpan ke folder '{results_dir}'!")
            
            # Download button untuk Top Topics
            with col2:
                st.download_button(
                    label="💾 Simpan Top Topics (CSV)",
                    data=convert_df_to_csv(top_topics_df),
                    file_name="top_topics.csv",
                    mime="text/csv",
                    key="download_top_topics"
                )
            
            # Download button untuk Top Topics per Periode
            if periods_data and all_period_topics:
                st.download_button(
                    label="💾 Simpan Top Topics per Periode (CSV)",
                    data=convert_df_to_csv(period_topics_df),
                    file_name="top_topics_per_period.csv",
                    mime="text/csv",
                    key="download_top_topics_per_period"
                )

            # ========== STANCE ANALYSIS PER POSTINGAN ==========
            st.subheader("🗣️ Stance Analysis per Postingan")

            total_comments = len(comments_df)
            unique_posts = comments_df['conversation_id_str'].nunique() if not comments_df.empty else 0
            st.info(f"📊 Menganalisis stance untuk {total_comments:,} komentar pada {unique_posts:,} postingan unik")

            post_stance_df = pd.DataFrame()
            if total_comments > 0:
                grouped = comments_df.groupby('conversation_id_str')
                post_stance_df = grouped.agg(
                    num_comments=('full_text_comments_preprocessed', 'size'),
                        support_comments=('stance', lambda values: (values == 'support').sum()),
                        oppose_comments=('stance', lambda values: (values == 'oppose').sum()),
                        neutral_comments=('stance', lambda values: (values == 'neutral').sum()),
                        avg_confidence=('confidence', 'mean')
                    ).reset_index()

                def _majority_stance(row):
                    counts = {
                        'support': row['support_comments'],
                        'oppose': row['oppose_comments'],
                        'neutral': row['neutral_comments']
                    }
                    highest = max(counts.values())
                    top_stances = [stance for stance, count in counts.items() if count == highest]
                    if len(top_stances) == 1:
                        return top_stances[0]
                    if 'neutral' in top_stances:
                        return 'neutral'
                post_stance_df['post_stance'] = post_stance_df.apply(_majority_stance, axis=1)

                if 'conversation_id_str' in posts_df.columns:
                    post_stance_df = post_stance_df.merge(
                        posts_df[['conversation_id_str', 'Topik', 'created_at', 'full_text']],
                        on='conversation_id_str',
                        how='left'
                    )
                    post_stance_df['period_3m'] = post_stance_df['created_at'].apply(
                        lambda x: f"{x.year}-Q{((x.month - 1) // 3) + 1}" if pd.notna(x) else None
                    )

                # Some display blocks expect stance_confidence even though the aggregated column is avg_confidence.
                if 'avg_confidence' in post_stance_df.columns and 'stance_confidence' not in post_stance_df.columns:
                    post_stance_df['stance_confidence'] = post_stance_df['avg_confidence']
            
            if not post_stance_df.empty:
                st.subheader("� Komentar & Stance per Postingan")
                with st.container():
                    header_cols = st.columns([1, 1, 1, 1])
                    header_cols[0].metric("Total Komentar", f"{len(comments_df):,}")
                    header_cols[1].metric("Postingan Unik", f"{post_stance_df['conversation_id_str'].nunique():,}")
                    avg_conf = post_stance_df['avg_confidence'].mean() if 'avg_confidence' in post_stance_df.columns else 0.0
                    header_cols[2].metric("Confidence Rata-rata", f"{avg_conf:.2f}")
                    top_topic = post_stance_df['Topik'].mode().iloc[0] if not post_stance_df['Topik'].dropna().empty else 'N/A'
                    header_cols[3].metric("Topik Terpopuler", str(top_topic))

                with st.expander("📌 Tabel Komentar dan Stance", expanded=False):
                    comment_display = comments_df.copy()
                    display_cols = [col for col in ['conversation_id_str', 'full_text_comments', 'stance', 'confidence', 'Topik'] if col in comment_display.columns]
                    if display_cols:
                        comment_display = comment_display[display_cols].rename(columns={
                            'conversation_id_str': 'Post ID',
                            'full_text_comments': 'Komentar',
                            'stance': 'Stance',
                            'confidence': 'Confidence',
                            'Topik': 'Topik'
                        })
                        st.dataframe(comment_display.reset_index(drop=True), use_container_width=True)
                    else:
                        st.info("Tidak ada data komentar atau kolom stance untuk ditampilkan.")

                with st.expander("⭐ Top 10 Postingan berdasarkan Jumlah Komentar", expanded=False):
                    top_posts_display = post_stance_df.sort_values('num_comments', ascending=False).head(10)[[
                        'conversation_id_str', 'Topik', 'num_comments', 'support_comments', 'oppose_comments', 'neutral_comments', 'avg_confidence'
                    ]].rename(columns={
                        'conversation_id_str': 'Post ID',
                        'Topik': 'Topik',
                        'num_comments': 'Jumlah Komentar',
                        'support_comments': 'Support',
                        'oppose_comments': 'Oppose',
                        'neutral_comments': 'Neutral',
                        'avg_confidence': 'Confidence Rata-rata'
                    })
                    st.dataframe(top_posts_display, use_container_width=True)

                st.subheader("�📋 Hasil Stance Analysis per Topik per Periode 3 Bulan")
                st.markdown("Ringkasan topik per kuartal dengan distribusi stance komentar untuk setiap topik.")

                topic_period_stance = post_stance_df.groupby(['period_3m', 'Topik', 'post_stance']).size().reset_index(name='count')
                topic_period_stance = topic_period_stance[topic_period_stance['Topik'] != -1]

                if not topic_period_stance.empty:
                    summary_df = topic_period_stance.pivot_table(
                        index=['period_3m', 'Topik'],
                        columns='post_stance',
                        values='count',
                        fill_value=0
                    ).reset_index()
                    summary_df.columns.name = None

                    if topic_model is not None:
                        try:
                            topic_info = topic_model.get_topic_info()
                            topic_names = topic_info.set_index('Topic')['Name'].to_dict()
                            summary_df['Topic Name'] = summary_df['Topik'].map(topic_names).fillna('N/A')
                        except Exception:
                            summary_df['Topic Name'] = 'N/A'
                    else:
                        summary_df['Topic Name'] = 'N/A'

                    cols = ['period_3m', 'Topik', 'Topic Name'] + [c for c in ['oppose', 'neutral', 'support'] if c in summary_df.columns]
                    summary_df = summary_df[cols]
                    st.dataframe(summary_df.sort_values(['period_3m', 'Topik']), use_container_width=True)
                else:
                    st.info('Tidak ada data stance yang valid untuk ditampilkan dalam format per periode-topik.')

                with st.expander('Detail Hasil Stance per Postingan'):
                    display_cols = [
                        'conversation_id_str', 'Topik', 'period_3m', 'post_stance',
                        'num_comments', 'support_comments', 'oppose_comments', 'neutral_comments',
                        'avg_confidence'
                    ]
                    st.dataframe(post_stance_df[display_cols].head(50), use_container_width=True)

                if 'Topik' in post_stance_df.columns:
                    st.subheader("📊 Distribusi Stance per Topik")

                    topic_stance_dist = post_stance_df.groupby(['Topik', 'post_stance']).size().reset_index(name='count')
                    topic_stance_dist = topic_stance_dist[topic_stance_dist['Topik'] != -1]

                    if not topic_stance_dist.empty:
                        fig = px.bar(
                            topic_stance_dist,
                            x='Topik',
                            y='count',
                            color='post_stance',
                            title='Distribusi Stance per Topik (berdasarkan Postingan)',
                            barmode='group',
                            color_discrete_map={
                                'support': '#4ECDC4',
                                'oppose': '#FF6B6B',
                                'neutral': '#45B7D1'
                            }
                        )
                        st.plotly_chart(fig, use_container_width=True)

                if 'period_3m' in post_stance_df.columns:
                    st.subheader("📈 Evolusi Stance per Periode 3 Bulan")

                    period_stance_dist = post_stance_df.groupby(['period_3m', 'post_stance']).size().reset_index(name='count')

                    if not period_stance_dist.empty:
                        fig = px.line(
                            period_stance_dist,
                            x='period_3m',
                            y='count',
                            color='post_stance',
                            markers=True,
                            title='Evolusi Stance Publik per Periode 3 Bulan',
                            color_discrete_map={
                                'support': '#4ECDC4',
                                'oppose': '#FF6B6B',
                                'neutral': '#45B7D1'
                            }
                        )
                        st.plotly_chart(fig, use_container_width=True)

                if 'Topik' in post_stance_df.columns and 'period_3m' in post_stance_df.columns:
                    st.subheader("🔥 Heatmap: Topik vs Periode vs Stance")

                    heatmap_data = post_stance_df.groupby(['Topik', 'period_3m', 'post_stance']).size().reset_index(name='count')
                    heatmap_data = heatmap_data[heatmap_data['Topik'] != -1]

                    if not heatmap_data.empty:
                        heatmap_pivot = heatmap_data.pivot_table(
                            index=['Topik', 'period_3m'],
                            columns='post_stance',
                            values='count',
                            fill_value=0
                        ).reset_index()

                        for stance in ['support', 'oppose', 'neutral']:
                            if stance in heatmap_pivot.columns:
                                fig = px.imshow(
                                    heatmap_pivot.pivot(index='Topik', columns='period_3m', values=stance),
                                    title=f'Heatmap {stance} Stance: Topik vs Periode',
                                    labels=dict(x="Periode 3 Bulan", y="Topik", color="Jumlah Postingan")
                                )
                                st.plotly_chart(fig, use_container_width=True, key=f'heatmap_{stance}')

                st.download_button(
                    label="💾 Simpan Hasil Stance per Postingan (CSV)",
                    data=convert_df_to_csv(post_stance_df),
                    file_name="stance_per_postingan.csv",
                    mime="text/csv",
                    key="download_stance_per_post_summary"
                )

            else:
                st.warning("Tidak ada data postingan untuk analisis stance.")

            # Gabungkan dengan informasi topik dan periode
            if not post_stance_df.empty and 'conversation_id_str' in posts_df.columns:
                # Only merge if created_at is not already in the dataframe
                if 'created_at' not in post_stance_df.columns:
                    post_stance_df = post_stance_df.merge(
                        posts_df[['conversation_id_str', 'Topik', 'created_at', 'full_text']],
                        on='conversation_id_str',
                        how='left'
                    )
                # Ensure period_3m is calculated if created_at exists
                if 'created_at' in post_stance_df.columns:
                    post_stance_df['period_3m'] = post_stance_df['created_at'].apply(
                        lambda x: f"{x.year}-Q{((x.month - 1) // 3) + 1}" if pd.notna(x) else None
                    )

            # Tampilkan hasil
            if not post_stance_df.empty:
                st.subheader("📋 Hasil Stance Analysis per Topik per Periode 3 Bulan")
                st.markdown("Ringkasan topik per kuartal dengan distribusi stance komentar untuk setiap topik.")

                topic_period_stance = post_stance_df.groupby(['period_3m', 'Topik', 'post_stance']).size().reset_index(name='count')
                topic_period_stance = topic_period_stance[topic_period_stance['Topik'] != -1]

                if not topic_period_stance.empty:
                    summary_df = topic_period_stance.pivot_table(
                        index=['period_3m', 'Topik'],
                        columns='post_stance',
                        values='count',
                        fill_value=0
                    ).reset_index()
                    summary_df.columns.name = None

                    # Add topic names if available
                    if topic_model is not None:
                        try:
                            topic_info = topic_model.get_topic_info()
                            topic_names = topic_info.set_index('Topic')['Name'].to_dict()
                            summary_df['Topic Name'] = summary_df['Topik'].map(topic_names).fillna('N/A')
                        except Exception:
                            summary_df['Topic Name'] = 'N/A'
                    else:
                        summary_df['Topic Name'] = 'N/A'

                    # Reorder columns for readability
                    cols = ['period_3m', 'Topik', 'Topic Name'] + [c for c in ['oppose', 'neutral', 'support'] if c in summary_df.columns]
                    summary_df = summary_df[cols]
                    st.dataframe(summary_df.sort_values(['period_3m', 'Topik']), use_container_width=True)
                else:
                    st.info('Tidak ada data stance yang valid untuk ditampilkan dalam format per periode-topik.')

                # Tambahkan detail per postingan di expander
                with st.expander('Detail Hasil Stance per Postingan'):
                    display_cols = ['conversation_id_str', 'Topik', 'period_3m', 'post_stance',
                                  'stance_confidence', 'num_comments']
                    st.dataframe(post_stance_df[display_cols].head(50), use_container_width=True)

                # Statistik stance per topik
                if 'Topik' in post_stance_df.columns:
                    st.subheader("📊 Distribusi Stance per Topik")

                    topic_stance_dist = post_stance_df.groupby(['Topik', 'post_stance']).size().reset_index(name='count')
                    topic_stance_dist = topic_stance_dist[topic_stance_dist['Topik'] != -1]  # Exclude outlier topics

                    if not topic_stance_dist.empty:
                        fig = px.bar(
                            topic_stance_dist,
                            x='Topik',
                            y='count',
                            color='post_stance',
                            title='Distribusi Stance per Topik (berdasarkan Postingan)',
                            barmode='group',
                            color_discrete_map={
                                'support': '#4ECDC4',
                                'oppose': '#FF6B6B',
                                'neutral': '#45B7D1'
                            }
                        )
                        st.plotly_chart(fig, use_container_width=True, key='topic_stance_dist')

                # Statistik stance per periode 3 bulan
                if 'period_3m' in post_stance_df.columns:
                    st.subheader("📈 Evolusi Stance per Periode 3 Bulan")

                    period_stance_dist = post_stance_df.groupby(['period_3m', 'post_stance']).size().reset_index(name='count')

                    if not period_stance_dist.empty:
                        fig = px.line(
                            period_stance_dist,
                            x='period_3m',
                            y='count',
                            color='post_stance',
                            markers=True,
                            title='Evolusi Stance Publik per Periode 3 Bulan',
                            color_discrete_map={
                                'support': '#4ECDC4',
                                'oppose': '#FF6B6B',
                                'neutral': '#45B7D1'
                            }
                        )
                        st.plotly_chart(fig, use_container_width=True, key='period_stance_dist')

                # Heatmap Topik vs Periode vs Stance
                if 'Topik' in post_stance_df.columns and 'period_3m' in post_stance_df.columns:
                    st.subheader("🔥 Heatmap: Topik vs Periode vs Stance")

                    heatmap_data = post_stance_df.groupby(['Topik', 'period_3m', 'post_stance']).size().reset_index(name='count')
                    heatmap_data = heatmap_data[heatmap_data['Topik'] != -1]

                    if not heatmap_data.empty:
                        # Pivot untuk heatmap
                        heatmap_pivot = heatmap_data.pivot_table(
                            index=['Topik', 'period_3m'],
                            columns='post_stance',
                            values='count',
                            fill_value=0
                        ).reset_index()

                        # Buat heatmap untuk setiap stance
                        for stance in ['support', 'oppose', 'neutral']:
                            if stance in heatmap_pivot.columns:
                                fig = px.imshow(
                                    heatmap_pivot.pivot(index='Topik', columns='period_3m', values=stance),
                                    title=f'Heatmap {stance} Stance: Topik vs Periode',
                                    labels=dict(x="Periode 3 Bulan", y="Topik", color="Jumlah Postingan")
                                )
                                st.plotly_chart(fig, use_container_width=True, key=f'heatmap_{stance}_2')

                # Download hasil
                st.download_button(
                    label="💾 Simpan Hasil Stance per Postingan (CSV)",
                    data=convert_df_to_csv(post_stance_df),
                    file_name="stance_per_postingan.csv",
                    mime="text/csv",
                    key="download_stance_per_post_full"
                )

            else:
                st.warning("Tidak ada data postingan untuk analisis stance.")

            # Create a comprehensive report
            with col3:
                # Combined report
                if not post_stance_df.empty:
                    stance_lines = ''.join(
                        f"- {sent}: {count:,}\n" for sent, count in post_stance_df['post_stance'].value_counts().items()
                    )
                else:
                    stance_lines = "- Tidak ada data stance\n"
                
                report = f"""
LAPORAN ANALISIS TOPIK DINAMIS DAN STANCE ANALYSIS
=====================================================

STATISTIK UMUM:
- Total Posts: {len(posts_df):,}
- Total Komentar: {len(comments_df):,}
- Rentang Waktu: {posts_df['created_at'].min().date()} hingga {posts_df['created_at'].max().date()}

HASIL TOPIC MODELING:
- Jumlah Topics: {len(top_topics_df):,}
- Model: BERTopic

HASIL STANCE ANALISIS PER POSTINGAN:
- Total Postingan Dianalisis: {len(post_stance_df) if not post_stance_df.empty else 0:,}
{stance_lines}

Dibuat pada: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
                st.download_button(
                    label="💾 Simpan Laporan (.txt)",
                    data=report.encode('utf-8'),
                    file_name="analysis_report.txt",
                    mime="text/plain",
                    key="download_report"
                )
            
            # ========== VIRAL SPREAD ANALYSIS ==========
            st.divider()
            st.subheader("🔥 Analisis Penyebaran Viral Posts")
            st.write("Analisis bagaimana posts yang sama (retweets) menyebar di kalangan berbeda pengguna.")
            
            with st.spinner("Menganalisis pola penyebaran viral posts..."):
                viral_posts_info = analyze_viral_spread(df)
            
            if viral_posts_info is not None and len(viral_posts_info) > 0:
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "🔥 Posts Viral",
                        f"{len(viral_posts_info)}",
                        help="Jumlah unique posts yang muncul >1 kali"
                    )
                
                with col2:
                    total_reposts = viral_posts_info['total_reposts'].sum()
                    st.metric(
                        "🔄 Total Repost",
                        f"{total_reposts:,}",
                        help="Total instances dari viral posts"
                    )
                
                with col3:
                    avg_users = viral_posts_info['unique_users'].mean()
                    st.metric(
                        "👥 Rata-rata Users",
                        f"{avg_users:.0f}",
                        help="Rata-rata users yang mereposts per post"
                    )
                
                with col4:
                    max_spread_rate = viral_posts_info['spread_rate'].max()
                    st.metric(
                        "⚡ Max Spread Rate",
                        f"{max_spread_rate:.1f}/hr",
                        help="Repost per jam tertinggi"
                    )
                
                # Tabs for different visualizations
                tab1, tab2, tab3, tab4 = st.tabs([
                    "📊 Statistik Top Posts",
                    "⏱️ Timeline Penyebaran",
                    "🔥 Heatmap Aktivitas",
                    "📋 Tabel Viral Posts"
                ])
                
                with tab1:
                    st.write("**Top Posts dengan Repost Terbanyak**")
                    fig1, fig2 = visualize_viral_statistics(viral_posts_info)
                    
                    if fig1:
                        st.plotly_chart(fig1, use_container_width=True)
                    if fig2:
                        st.write("**Scatter: Unique Users vs Total Reposts** (bubble size = spread rate)")
                        st.plotly_chart(fig2, use_container_width=True)
                
                with tab2:
                    st.write("**Timeline Penyebaran Top 5 Viral Posts**")
                    fig_timeline = visualize_viral_timeline(df, viral_posts_info)
                    if fig_timeline:
                        st.plotly_chart(fig_timeline, use_container_width=True)
                    else:
                        st.warning("Tidak dapat membuat timeline visualization")
                
                with tab3:
                    st.write("**Kapan Posts Paling Banyak Dibagikan?**")
                    fig_heatmap = visualize_viral_heatmap(df)
                    if fig_heatmap:
                        st.plotly_chart(fig_heatmap, use_container_width=True)
                    else:
                        st.warning("Tidak dapat membuat heatmap")
                
                with tab4:
                    st.write("**Daftar Lengkap Viral Posts**")
                    
                    # Display table with sortable columns
                    display_df = viral_posts_info[[
                        'text', 'total_reposts', 'unique_users', 'time_span_hours', 'spread_rate'
                    ]].copy()
                    display_df.columns = ['Post Text', 'Reposts', 'Users', 'Hours Span', 'Rate/Hr']
                    display_df['Rate/Hr'] = display_df['Rate/Hr'].round(2)
                    display_df['Hours Span'] = display_df['Hours Span'].round(2)
                    
                    st.dataframe(display_df, use_container_width=True, hide_index=True)
                    
                    # Download button
                    st.download_button(
                        label="💾 Download Viral Posts Analysis (CSV)",
                        data=convert_df_to_csv(display_df),
                        file_name=f"viral_posts_analysis_{timestamp}.csv",
                        mime="text/csv",
                        key="download_viral_posts"
                    )
                
                # Insights
                with st.expander("💡 Insight dari Analisis Viral"):
                    st.markdown("""
                    **Apa yang ditunjukkan data ini:**
                    
                    - **Posts Viral**: Post unik yang di-retweet oleh banyak orang berbeda
                    - **Repost**: Saat seseorang membagikan ulang post yang sama (persis) ke followers mereka
                    - **Unique Users**: Berapa banyak user berbeda yang mereposts post tersebut
                    - **Time Span**: Berapa lama (dalam jam) post tersebar dari post pertama hingga terakhir
                    - **Spread Rate**: Seberapa cepat post menyebar (reposts per jam)
                    
                    **Interpretasi:**
                    - Spread Rate tinggi = post menyebar dengan cepat/viral
                    - Unique Users banyak = post mendapat reach luas
                    - Time Span panjang = post terus di-retweet dalam periode waktu lama
                    """)
            else:
                st.info("💡 Tidak ada posts yang di-retweet / viral untuk dianalisis.")
            
            # ========== TOPIC COHERENCE EVALUATION ==========
            st.subheader("🎯 Topic Coherence Evaluation")
            
            # Calculate coherence
            with st.spinner("Calculating topic coherence..."):
                coherence_results = calculate_topic_coherence(topic_model, docs)
            
            if "error" not in coherence_results:
                if coherence_results.get('coherence_backend') == 'fallback':
                    st.info("⚠️ Gensim tidak tersedia. Menggunakan fallback coherence approximation tanpa gensim.")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    overall_coh = coherence_results.get('overall_coherence', 0)
                    if isinstance(overall_coh, (int, float)):
                        st.metric(
                            "📊 Overall Coherence", 
                            f"{float(overall_coh):.4f}",
                            help=f"Coherence measure: {coherence_results.get('coherence_type', 'N/A')}"
                        )
                    else:
                        st.metric("📊 Overall Coherence", "N/A", help="Could not calculate coherence")
                
                with col2:
                    st.metric(
                        "📈 Number of Topics", 
                        coherence_results['num_topics']
                    )
                
                with col3:
                    avg_topic_coherence = np.mean(coherence_results['topic_coherences'])
                    st.metric(
                        "📉 Average Topic Coherence", 
                        f"{avg_topic_coherence:.4f}"
                    )
                
                # Coherence interpretation
                st.subheader("📋 Coherence Interpretation")
                
                overall_score = coherence_results.get('overall_coherence', 0)
                if isinstance(overall_score, (int, float)):
                    overall_score = float(overall_score)
                    if overall_score >= 0.6:
                        coherence_quality = "Excellent"
                        color = "🟢"
                    elif overall_score >= 0.4:
                        coherence_quality = "Good"
                        color = "🟡"
                    elif overall_score >= 0.2:
                        coherence_quality = "Fair"
                        color = "🟠"
                    else:
                        coherence_quality = "Poor"
                        color = "🔴"
                else:
                    coherence_quality = "N/A"
                    color = "⚪"
                    overall_score = 0
                
                st.markdown(f"**{color} Topic Coherence Quality: {coherence_quality}**")
                
                with st.expander("📊 Detailed Coherence Scores per Topic"):
                    try:
                        topic_coherences = [float(x) if isinstance(x, (int, float)) else 0 for x in coherence_results.get('topic_coherences', [])]
                        coherence_df = pd.DataFrame({
                            'Topic': [f'Topic {i}' for i in range(len(topic_coherences))],
                            'Coherence Score': topic_coherences
                        })
                    except Exception as e:
                        st.error(f"Could not display coherence scores: {e}")
                        coherence_df = pd.DataFrame()
                    coherence_df = coherence_df.sort_values('Coherence Score', ascending=False)
                    st.dataframe(coherence_df, use_container_width=True)
                    
                    # Plot coherence distribution
                    fig = go.Figure(data=[
                        go.Bar(
                            x=coherence_df['Topic'],
                            y=coherence_df['Coherence Score'],
                            marker_color='lightblue'
                        )
                    ])
                    fig.update_layout(
                        title="Topic Coherence Scores",
                        xaxis_title="Topic",
                        yaxis_title="Coherence Score",
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with st.expander("ℹ️ About Topic Coherence"):
                    st.markdown("""
                    **Topic Coherence** mengukur seberapa baik kata-kata dalam sebuah topik saling berhubungan.
                    
                    **Interpretasi Score:**
                    - **0.6+**: Excellent - Topik sangat koheren
                    - **0.4-0.6**: Good - Topik cukup koheren  
                    - **0.2-0.4**: Fair - Topik perlu improvement
                    - **<0.2**: Poor - Topik kurang bermakna
                    
                    **Metode yang digunakan:** C_V (Context Vector)
                    """)
            else:
                st.error(f"❌ Gagal menghitung coherence: {coherence_results['error']}")
                st.info("💡 Jika gensim tidak terpasang, app akan menghitung coherence dengan fallback native. Pastikan dataset dan topik valid.")
            
            # ========== ADDITIONAL TOPIC METRICS ==========
            st.subheader("📈 Additional Topic Modeling Metrics")
            
            # Calculate additional metrics
            with st.spinner("Calculating additional topic metrics..."):
                topic_metrics = calculate_topic_metrics(topic_model, docs)
            
            if "error" not in topic_metrics:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "🎯 Topic Diversity", 
                        f"{topic_metrics['avg_topic_diversity']:.1f}",
                        help="Average unique words per topic (top 10)"
                    )
                
                with col2:
                    st.metric(
                        "📊 Document Coverage", 
                        f"{topic_metrics['doc_coverage']:.1%}",
                        help="Percentage of documents assigned to topics"
                    )
                
                with col3:
                    st.metric(
                        "📏 Topic Size CV", 
                        f"{topic_metrics['topic_size_cv']:.3f}",
                        help="Coefficient of variation in topic sizes"
                    )
                
                with col4:
                    st.metric(
                        "📚 Unique Words", 
                        f"{topic_metrics['total_unique_words']:,}",
                        help="Total unique words across all topics"
                    )
                
                # Topic size distribution
                with st.expander("📊 Topic Size Distribution"):
                    fig = go.Figure(data=[
                        go.Bar(
                            x=[f'Topic {i}' for i in range(len(topic_metrics['topic_sizes']))],
                            y=topic_metrics['topic_sizes'],
                            marker_color='lightgreen'
                        )
                    ])
                    fig.update_layout(
                        title="Document Distribution Across Topics",
                        xaxis_title="Topic",
                        yaxis_title="Number of Documents",
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Topic diversity distribution
                with st.expander("🎨 Topic Diversity Distribution"):
                    fig = go.Figure(data=[
                        go.Bar(
                            x=[f'Topic {i}' for i in range(len(topic_metrics['topic_diversities']))],
                            y=topic_metrics['topic_diversities'],
                            marker_color='orange'
                        )
                    ])
                    fig.update_layout(
                        title="Topic Diversity (Unique Words per Topic)",
                        xaxis_title="Topic",
                        yaxis_title="Unique Words",
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with st.expander("ℹ️ About Additional Metrics"):
                    st.markdown("""
                    **Topic Diversity**: Rata-rata kata unik dalam top 10 kata setiap topik. Nilai tinggi menunjukkan topik yang beragam.
                    
                    **Document Coverage**: Persentase dokumen yang berhasil diklasifikasikan ke topik (bukan outlier).
                    
                    **Topic Size CV**: Koefisien variasi ukuran topik. Nilai rendah menunjukkan distribusi yang merata.
                    
                    **Unique Words**: Total kata unik yang muncul dalam semua topik.
                    """)
            else:
                st.error(f"Error calculating topic metrics: {topic_metrics['error']}")
            
            logging.info("Analysis completed successfully")
    elif app_mode == "Validasi Ahli Diplomasi":
        render_expert_validation_ui()