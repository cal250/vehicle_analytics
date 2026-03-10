from django.shortcuts import render
import pandas as pd
import joblib
import os
from model_generators.regression.train_regression import evaluate_regression_model
from model_generators.classification.train_classifier import evaluate_classification_model
from .plotly_dashboard import create_rwanda_map
from model_generators.clustering.train_cluster import evaluate_clustering_model, get_clustered_dataframe

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
    r2_score_val = None
    cv = None
    # evaluate regression metrics on dataset
    try:
        r2_score_val, _ = evaluate_regression_model()
        df = pd.read_csv(DATA_PATH)
        cv = df['selling_price'].std() / df['selling_price'].mean()
    except Exception:
        pass

    if request.method == 'POST':
        year = int(request.POST.get('year'))
        km = float(request.POST.get('km'))
        seats = int(request.POST.get('seats'))
        income = float(request.POST.get('income'))
        
        model = joblib.load(REGRESSION_MODEL_PATH)
        prediction = model.predict([[year, km, seats, income]])[0]
        result = round(prediction, 2)
        
    context = {
        'result': result,
        'r2_score': r2_score_val,
        'coefficient_variation': cv
    }
    return render(request, 'predictor/regression_analysis.html', context)

def classification_analysis(request):
    result = None
    accuracy_val = None
    cv = None
    # evaluate classification metrics on dataset
    try:
        accuracy_val, _ = evaluate_classification_model()
        df = pd.read_csv(DATA_PATH)
        cv = df['estimated_income'].std() / df['estimated_income'].mean()
    except Exception:
        pass

    if request.method == 'POST':
        year = int(request.POST.get('year'))
        km = float(request.POST.get('km'))
        seats = int(request.POST.get('seats'))
        income = float(request.POST.get('income'))
        
        model = joblib.load(CLASSIFICATION_MODEL_PATH)
        prediction = model.predict([[year, km, seats, income]])[0]
        result = prediction
        
    context = {
        'result': result,
        'accuracy_score': accuracy_val,
        'coefficient_variation': cv
    }
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

def cluster_analytics(request):
    cluster_summary_html = None
    overall_cv_html = None
    inter_cluster_cv_html = None
    try:
        df = get_clustered_dataframe()
        numeric_cols = ['estimated_income', 'selling_price']

        # per-cluster CV only for each numeric feature (trim outliers per cluster)
        cv_rows = {}
        min_trim_count = 5
        for cluster_label, group in df.groupby('cluster_label'):
            row = {}
            for col in numeric_cols:
                series = group[col].dropna()
                if series.empty:
                    row[f"{col}_cv"] = pd.NA
                    continue
                low = series.quantile(0.20)
                high = series.quantile(0.80)
                filtered = series[(series >= low) & (series <= high)]
                # fallback to full series if trimming leaves too few values
                if filtered.size < min_trim_count:
                    filtered = series
                if filtered.size < 2:
                    row[f"{col}_cv"] = pd.NA
                    continue
                mean_val = filtered.mean()
                if mean_val == 0:
                    row[f"{col}_cv"] = pd.NA
                    continue
                row[f"{col}_cv"] = (filtered.std() / mean_val)
            cv_rows[cluster_label] = row
        summary_table = pd.DataFrame.from_dict(cv_rows, orient='index').round(4)
        summary_table = summary_table.rename(columns={
            'estimated_income_cv': 'Estimated Income CV',
            'selling_price_cv': 'Selling Price CV'
        })
        summary_table.index.name = 'Cluster'

        overall_cv = (df[numeric_cols].std() / df[numeric_cols].mean()).round(4)
        overall_cv_table = overall_cv.to_frame(name='CV')
        overall_cv_table = overall_cv_table.rename(index={
            'estimated_income': 'Estimated Income',
            'selling_price': 'Selling Price'
        })

        # inter-cluster CV: std of cluster means / overall mean (per feature)
        # inter-cluster CV: std of cluster means / overall mean (per feature)
        grouped = df.groupby('cluster_label')[numeric_cols].agg(['mean', 'std'])
        means = grouped.xs('mean', level=1, axis=1)
        inter_cluster_cv = (means.std() / df[numeric_cols].mean()).round(4)
        inter_cluster_cv_table = inter_cluster_cv.to_frame(name='CV')
        inter_cluster_cv_table = inter_cluster_cv_table.rename(index={
            'estimated_income': 'Estimated Income',
            'selling_price': 'Selling Price'
        })

        cluster_summary_html = summary_table.to_html(
            classes='table table-striped table-hover',
            index=True,
            border=0
        )
        overall_cv_html = overall_cv_table.to_html(
            classes='table table-striped table-hover',
            index=True,
            border=0
        )
        inter_cluster_cv_html = inter_cluster_cv_table.to_html(
            classes='table table-striped table-hover',
            index=True,
            border=0
        )
    except Exception:
        pass

    context = {
        'cluster_summary': cluster_summary_html,
        'overall_cv': overall_cv_html,
        'inter_cluster_cv': inter_cluster_cv_html
    }
    return render(request, 'predictor/cluster_analytics.html', context)
