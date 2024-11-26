import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import TruncatedSVD
from nltk.corpus import stopwords
import gc

# Definiere Stoppwörter am Anfang
CUSTOM_STOP_WORDS = list(stopwords.words('english')) + [
    'show', 'results', 'x', 'et', 'al', 'using', 'one', 'two', 'three',
    'model', 'theory', 'non', 'system', 'systems', 'n', 'k', 'p', 'b',
    'groups', 'production', 'theorem', 'study', 'method', 'based', 'analysis',
    'new', 'approach', 'paper', 'methods', 'data', 'models', 'random', 'algebras',
    'states', 'induced', 'gauge', 'manifolds', 'function', 'functions', 'order',
    'properties', 'type', 'equations', 'problem', 'state'
]

def get_stratified_sample(file_path, sample_size=50000):
    """Erstellt eine stratifizierte Stichprobe basierend auf Jahren"""
    print("Analysiere Jahresverteilung...")

    # Lese zunächst nur das update_date Feld
    years_df = pd.DataFrame()
    chunk_size = 200000

    for chunk in pd.read_json(file_path, lines=True, chunksize=chunk_size):
        years_chunk = pd.DataFrame()
        years_chunk['year'] = pd.to_datetime(chunk['update_date']).dt.year
        years_df = pd.concat([years_df, years_chunk])

    # Berechne Verteilung
    year_dist = years_df['year'].value_counts(normalize=True)

    # Berechne Anzahl der Samples pro Jahr
    samples_per_year = (year_dist * sample_size).round().astype(int)

    print("\nGeplante Stichprobengröße pro Jahr:")
    print(samples_per_year)

    # Ziehe stratifizierte Stichprobe
    sampled_data = []

    # Chunk-weise Verarbeitung
    for chunk in pd.read_json(file_path, lines=True, chunksize=chunk_size):
        chunk_years = pd.to_datetime(chunk['update_date']).dt.year

        # Für jedes Jahr in diesem Chunk
        for year in samples_per_year.index:
            if samples_per_year[year] > 0:
                year_mask = chunk_years == year
                year_data = chunk.loc[year_mask]
                if not year_data.empty:
                    # Ziehe Zufallsstichprobe für dieses Jahr
                    n_samples = min(samples_per_year[year], len(year_data))
                    if n_samples > 0:
                        sampled = year_data.sample(n=n_samples)
                        sampled_data.append(sampled)
                        samples_per_year[year] -= n_samples

    # Kombiniere alle Samples
    final_sample = pd.concat(sampled_data, ignore_index=True)
    print(f"\nEndgültige Stichprobengröße: {len(final_sample)}")

    return final_sample

def clean_data(df):
    """Bereinigt den Datensatz und erstellt grundlegende Features."""
    print("\nBereinige Daten...")

    # Erstelle eine Kopie des DataFrames
    cleaned_df = df.copy()

    # Entferne nicht benötigte Spalten
    columns_to_keep = ['title', 'categories', 'update_date']
    cleaned_df = cleaned_df[columns_to_keep]

    # Erstelle Jahr aus update_date
    cleaned_df.loc[:, 'year'] = pd.to_datetime(cleaned_df['update_date']).dt.year

    # Erstelle Feature für Titellänge
    cleaned_df.loc[:, 'title_length'] = cleaned_df['title'].str.len()

    # Skaliere Titellänge
    scaler = StandardScaler()
    cleaned_df.loc[:, 'title_length_scaled'] = scaler.fit_transform(cleaned_df['title_length'].values.reshape(-1, 1))

    return cleaned_df

def create_features(df):
    """Erstellt Features für die Clustering-Analyse."""
    print("\nErstelle Features...")

    # Kategorie-Features
    mlb = MultiLabelBinarizer()
    categories_encoded = mlb.fit_transform([cats.split() for cats in df['categories']])
    categories_df = pd.DataFrame(categories_encoded, columns=mlb.classes_)

    # TF-IDF Features für Titel
    tfidf = TfidfVectorizer(
        max_features=200,
        stop_words=CUSTOM_STOP_WORDS,
        ngram_range=(1, 2)
    )
    title_tfidf = tfidf.fit_transform(df['title'])

    # Dimensionsreduktion
    print("Führe Dimensionsreduktion durch...")
    svd = TruncatedSVD(n_components=50)
    tfidf_svd = svd.fit_transform(title_tfidf)

    # TSNE
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        max_iter=250,
        random_state=42
    )
    tfidf_tsne = tsne.fit_transform(tfidf_svd)

    # Kombiniere Features
    final_features = pd.concat([
        df[['title_length_scaled']],
        categories_df,
        pd.DataFrame(tfidf_tsne, columns=['tsne_1', 'tsne_2'])
    ], axis=1)

    return final_features, tfidf

def find_optimal_k(features, max_k=8):
    """Optimales k mittels Ellbow-Methode und Silhouette-Score."""
    print("\nFühre Clustering-Analyse durch...")

    inertias = []
    silhouette_scores = []
    K = range(2, max_k + 1)

    for k in K:
        print(f"Teste k={k}")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(features)
        inertias.append(kmeans.inertia_)
        score = silhouette_score(features, kmeans.labels_)
        silhouette_scores.append(score)
        print(f"Silhouette-Score: {score:.4f}")

    # Visualisierung
    plt.figure(figsize=(15, 5))

    # Elbow Plot
    plt.subplot(121)
    plt.plot(K, inertias, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Inertia')
    plt.title('Elbow Plot')
    plt.grid(True, alpha=0.3)

    # Silhouette Score
    plt.subplot(122)
    plt.plot(K, silhouette_scores, 'rx-')
    plt.xlabel('k')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score Analysis')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('Graphen.png')
    plt.close()

    # Bestimme bestes k
    best_k = K[np.argmax(silhouette_scores)]
    print(f"\nBestes k basierend auf Silhouette Score: {best_k}")
    return best_k

def main():
    # Parameter
    SAMPLE_SIZE = 500000
    MAX_K = 7

    # Lade stratifizierte Stichprobe
    sample_df = get_stratified_sample('data.json', SAMPLE_SIZE)

    # Bereinige Daten
    cleaned_data = clean_data(sample_df)

    # Erstelle Features
    features, tfidf = create_features(cleaned_data)

    # Finde optimales k
    best_k = find_optimal_k(features, MAX_K)

    # Speichere Ergebnis
    with open('optimal_k.txt', 'w') as f:
        f.write(str(best_k))
    print(f"\nOptimales k wurde in 'optimal_k.txt' gespeichert")
    print(f"Plots wurden in 'Graphen.png' gespeichert")


if __name__ == "__main__":
    main()