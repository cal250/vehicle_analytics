from django.shortcuts import render
import pandas as pd
import joblib
import os
from .plotly_dashboard import create_rwanda_map
from model_generators.clustering.train_cluster import evaluate_clustering_model

# Define paths for models and data
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'dummy-data', 'vehicles_ml_dataset.csv')
REGRESSION_MODEL_PATH = os.path.join(BASE_DIR, 'model_generators', 'regression', 'regression_model.pkl')
CLASSIFICATION_MODEL_PATH = os.path.join(BASE_DIR, 'model_generators', 'classification', 'classification_model.pkl')
CLUSTERING_MODEL_PATH = os.path.join(BASE_DIR, 'model_generators', 'clustering', 'clustering_model.pkl')

def data_exploration_view(request):
    df = pd.read_csv(DATA_PATH)
    data_head = df.head(10).to_html(classes='table table-striped table-hover', index=False)
    
    # Plotly Map
    map_html = create_rwanda_map(df)
    
    context = {
        'data_head': data_head,
        'map_html': map_html
    }
    return render(request, 'predictor/index.html', context)

def regression_analysis(request):
    result = None
    if request.method == 'POST':
        year = int(request.POST.get('year'))
        km = float(request.POST.get('km'))
        seats = int(request.POST.get('seats'))
        income = float(request.POST.get('income'))
        
        model = joblib.load(REGRESSION_MODEL_PATH)
        prediction = model.predict([[year, km, seats, income]])[0]
        result = round(prediction, 2)
        
    context = {'result': result}
    return render(request, 'predictor/regression_analysis.html', context)

def classification_analysis(request):
    result = None
    if request.method == 'POST':
        year = int(request.POST.get('year'))
        km = float(request.POST.get('km'))
        seats = int(request.POST.get('seats'))
        income = float(request.POST.get('income'))
        
        model = joblib.load(CLASSIFICATION_MODEL_PATH)
        prediction = model.predict([[year, km, seats, income]])[0]
        result = prediction
        
    context = {'result': result}
    return render(request, 'predictor/classification_analysis.html', context)

def clustering_analysis(request):
    result = None
    sil_score = None
    cv = None
    # evaluate metrics on dataset (useful regardless of POST)
    try:
        sil_score, cv, _, _ = evaluate_clustering_model()
    except Exception:
        # ignore failures, metrics will stay None
        pass

    if request.method == 'POST':
        year = int(request.POST.get('year'))
        km = float(request.POST.get('km'))
        seats = int(request.POST.get('seats'))
        income = float(request.POST.get('income'))
        
        # 1. Predict vehicle price using regression model
        reg_model = joblib.load(REGRESSION_MODEL_PATH)
        predicted_price = reg_model.predict([[year, km, seats, income]])[0]
        
        # 2. Use predicted price + income for clustering
        cluster_data = joblib.load(CLUSTERING_MODEL_PATH)
        kmeans = cluster_data['model']
        scaler = cluster_data['scaler']
        transform = cluster_data.get('transform', None)
        weight = cluster_data.get('weight', 1)
        label_map = cluster_data['label_map']
        
        # scale input features the same way as training
        scaled_input = scaler.transform([[income, predicted_price]])
        # re‑apply any feature weighting
        if weight != 1:
            scaled_input = scaled_input.copy()
            scaled_input[:, 0] *= weight
        # apply transform if used during training
        if transform is not None:
            scaled_input = transform.transform(scaled_input)

        cluster_id = kmeans.predict(scaled_input)[0]
        result = label_map[cluster_id]
        
    context = {
        'result': result,
        'silhouette_score': sil_score,
        'coefficient_variation': cv
    }
    return render(request, 'predictor/clustering_analysis.html', context)
