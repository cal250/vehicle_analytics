import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import os

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, 'dummy-data', 'vehicles_ml_dataset.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'model_generators', 'regression', 'regression_model.pkl')

def train_regression_model():
    # Load dataset
    df = pd.read_csv(DATA_PATH)
    
    # Feature selection
    features = ['year', 'kilometers_driven', 'seating_capacity', 'estimated_income']
    target = 'selling_price'
    
    X = df[features]
    y = df[target]
    
    # Train/Test split (80/20)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Save model
    joblib.dump(model, MODEL_PATH)
    print(f"Regression model saved to {MODEL_PATH}")
    
    return model, X_test, y_test

def evaluate_regression_model():
    # Load model and test data
    if not os.path.exists(MODEL_PATH):
        train_regression_model()
        
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)
    
    features = ['year', 'kilometers_driven', 'seating_capacity', 'estimated_income']
    target = 'selling_price'
    
    X = df[features]
    y = df[target]
    
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Compute R2 score
    r2 = r2_score(y_test, y_pred)
    
    # Comparison table
    comparison_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred.round(2)})
    comparison_table = comparison_df.head(10)
    
    return r2, comparison_table

if __name__ == "__main__":
    train_regression_model()
    r2, table = evaluate_regression_model()
    print(f"R2 Score: {r2}")
    print("Comparison Table (First 10 rows):")
    print(table)
