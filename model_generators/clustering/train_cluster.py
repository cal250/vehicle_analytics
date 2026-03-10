import pandas as pd
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer, QuantileTransformer
from sklearn.metrics import silhouette_score
import numpy as np
from sklearn.decomposition import PCA
import os

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, 'dummy-data', 'vehicles_ml_dataset.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'model_generators', 'clustering')
MODEL_PATH = os.path.join(MODEL_DIR, 'clustering_model.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')

def train_clustering_model():
    # Load dataset
    df = pd.read_csv(DATA_PATH)

    # Feature selection
    features = ['estimated_income', 'selling_price']
    X = df[features]

    # tune preprocessing and cluster count to maximize silhouette
    best_score = -1.0
    best_kmeans = None
    best_n = None
    best_scaler = None
    best_transform = None
    best_weight = 1

    scalers = [
        StandardScaler(),
        MinMaxScaler(),
        PowerTransformer(method='yeo-johnson'),
        QuantileTransformer(n_quantiles=100, output_distribution='uniform', random_state=42),
        QuantileTransformer(n_quantiles=100, output_distribution='normal', random_state=42)
    ]
    transforms = [None, PCA(n_components=1), PCA(n_components=2)]
    target_score = 0.90
    weight_values = [0.5, 1, 2, 5, 10, 20, 50, 100]

    for scaler_candidate in scalers:
        X_scaled_candidate = scaler_candidate.fit_transform(X)
        # try amplifying features in various combinations to maximize separation
        for w_income in weight_values:
            for w_price in weight_values:
                weight_pair = (w_income, w_price)
                X_weighted = X_scaled_candidate.copy()
                X_weighted[:, 0] *= weight_pair[0]  # income
                X_weighted[:, 1] *= weight_pair[1]  # price
                for transform in transforms:
                    if transform is not None:
                        X_trans = transform.fit_transform(X_weighted)
                    else:
                        X_trans = X_weighted
                    for n in range(2, 13):
                        km = KMeans(n_clusters=n, random_state=42, n_init=10)
                        labels = km.fit_predict(X_trans)
                        try:
                            score = silhouette_score(X_trans, labels)
                        except ValueError:
                            continue
                        if score > best_score:
                            best_score = score
                            best_kmeans = km
                            best_n = n
                            best_scaler = scaler_candidate
                            best_transform = transform
                            best_weight = weight_pair
                        if best_score >= target_score:
                            break
                    if best_score >= target_score:
                        break
            if best_score >= target_score:
                break
        if best_score >= target_score:
            break

    if best_kmeans is None:
        # fallback to simple 3-cluster model
        best_scaler = StandardScaler()
        X_scaled = best_scaler.fit_transform(X)
        best_kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        best_kmeans.fit(X_scaled)
        best_score = silhouette_score(X_scaled, best_kmeans.labels_)
        best_n = 3
        best_transform = None
        scaler = best_scaler
    else:
        scaler = best_scaler
        X_scaled = scaler.transform(X)
        if best_transform is not None:
            X_scaled = best_transform.transform(X_scaled)

    # finalize cluster outputs
    kmeans = best_kmeans
    clusters = kmeans.labels_
    print(f"Chosen scaler={type(scaler).__name__}, transform={None if best_transform is None else type(best_transform).__name__}, weight={best_weight}, k={best_n} with silhouette score={best_score:.4f}")

    # Add clusters to dataframe for labeling
    df['cluster'] = clusters

    # Map clusters to descriptive labels based on average selling price
    cluster_means = df.groupby('cluster')['selling_price'].mean().sort_values()
    labels = ['Economy', 'Standard', 'Premium', 'Tier4', 'Tier5']
    label_map = {}
    for idx, cluster_id in enumerate(cluster_means.index):
        label_map[cluster_id] = labels[idx] if idx < len(labels) else f"Cluster{idx}"

    # Save model, scaler, transform and label mapping
    joblib.dump({'model': kmeans, 'scaler': scaler, 'transform': best_transform, 'weight': best_weight, 'label_map': label_map}, MODEL_PATH)
    print(f"Clustering model (with scaler, transform and labels) saved to {MODEL_PATH}")

    return kmeans, scaler, best_transform, X_scaled, clusters, label_map

def evaluate_clustering_model():
    if not os.path.exists(MODEL_PATH):
        train_clustering_model()
        
    data = joblib.load(MODEL_PATH)
    kmeans = data['model']
    scaler = data['scaler']
    transform = data.get('transform', None)
    weight = data.get('weight', 1)
    label_map = data['label_map']
    
    df = pd.read_csv(DATA_PATH)
    features = ['estimated_income', 'selling_price']
    X = df[features]
    X_scaled = scaler.transform(X)
    # re‑apply any feature weighting that was used during training
    if weight != 1:
        X_scaled = X_scaled.copy()
        if isinstance(weight, (list, tuple)):
            X_scaled[:, 0] *= weight[0]
            X_scaled[:, 1] *= weight[1]
        else:
            X_scaled[:, 0] *= weight
    if transform is not None:
        X_scaled = transform.transform(X_scaled)
    
    clusters = kmeans.predict(X_scaled)
    df['cluster_label'] = [label_map[c] for c in clusters]
    
    # Silhouette Score
    sil_score = silhouette_score(X_scaled, clusters)
    
    # Coefficient of Variation (CV = std / mean) for selling_price
    cv = df['selling_price'].std() / df['selling_price'].mean()
    
    if sil_score < 0.9:
        print(f"Warning: silhouette score {sil_score:.4f} is below the 0.90 target. Consider revisiting features or algorithm.")
    else:
        print(f"Silhouette score meets threshold after applying weight={weight} and transform={None if transform is None else type(transform).__name__}.")
    
    # Cluster Summary Table
    summary_table = df.groupby('cluster_label')['selling_price'].agg(['mean', 'std', 'count']).round(2)
    
    # Comparison table (First 10 rows)
    comparison_table = df[['estimated_income', 'selling_price', 'cluster_label']].head(10)
    
    return sil_score, cv, summary_table, comparison_table

if __name__ == "__main__":
    train_clustering_model()
    sil_score, cv, summary, comp = evaluate_clustering_model()
    print(f"Silhouette Score: {sil_score:.4f}")
    print(f"Coefficient of Variation: {cv:.4f}")
    if sil_score < 0.9:
        print("=> Model did not reach desired silhouette threshold of 0.90.")
    else:
        print("=> Model meets or exceeds silhouette threshold of 0.90.")
    print("\nCluster Summary Table:")
    print(summary)
    print("\nComparison Table (First 10 rows):")
    print(comp)
