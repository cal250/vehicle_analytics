import pandas as pd
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import numpy as np
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
    
    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # KMeans (3 clusters)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Add clusters to dataframe for labeling
    df['cluster'] = clusters
    
    # Map clusters to Economy, Standard, Premium based on average selling price
    cluster_means = df.groupby('cluster')['selling_price'].mean().sort_values()
    label_map = {
        cluster_means.index[0]: 'Economy',
        cluster_means.index[1]: 'Standard',
        cluster_means.index[2]: 'Premium'
    }
    
    # Save model, scaler and label mapping
    joblib.dump({'model': kmeans, 'scaler': scaler, 'label_map': label_map}, MODEL_PATH)
    print(f"Clustering model (with scaler and labels) saved to {MODEL_PATH}")
    
    return kmeans, scaler, X_scaled, clusters, label_map

def evaluate_clustering_model():
    if not os.path.exists(MODEL_PATH):
        train_clustering_model()
        
    data = joblib.load(MODEL_PATH)
    kmeans = data['model']
    scaler = data['scaler']
    label_map = data['label_map']
    
    df = pd.read_csv(DATA_PATH)
    features = ['estimated_income', 'selling_price']
    X = df[features]
    X_scaled = scaler.transform(X)
    
    clusters = kmeans.predict(X_scaled)
    df['cluster_label'] = [label_map[c] for c in clusters]
    
    # Silhouette Score
    sil_score = silhouette_score(X_scaled, clusters)
    
    # Coefficient of Variation (CV = std / mean) for selling_price
    cv = df['selling_price'].std() / df['selling_price'].mean()
    
    # Cluster Summary Table
    summary_table = df.groupby('cluster_label')['selling_price'].agg(['mean', 'std', 'count']).round(2)
    
    # Comparison table (First 10 rows)
    comparison_table = df[['estimated_income', 'selling_price', 'cluster_label']].head(10)
    
    return sil_score, cv, summary_table, comparison_table

if __name__ == "__main__":
    train_clustering_model()
    sil_score, cv, summary, comp = evaluate_clustering_model()
    print(f"Silhouette Score: {sil_score}")
    print(f"Coefficient of Variation: {cv}")
    print("\nCluster Summary Table:")
    print(summary)
    print("\nComparison Table (First 10 rows):")
    print(comp)
