from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib
import time

def prepare_data(df):
    """Prepare features and target"""
    feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 
                      'SMA_10', 'SMA_30', 'EMA_10', 'RSI', 'Volatility']
    
    X = df[feature_columns].values
    y = df['Target'].values
    
    # Split data (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, scaler

def train_all_models(X_train, y_train):
    """Train all 5 models"""
    models = {}
    
    print("\n1. Training Linear Regression...")
    start = time.time()
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    models['Linear Regression'] = {
        'model': lr,
        'train_time': time.time() - start
    }
    print(f"   Done in {models['Linear Regression']['train_time']:.2f}s")
    
    print("\n2. Training Decision Tree...")
    start = time.time()
    dt = DecisionTreeRegressor(max_depth=10, random_state=42)
    dt.fit(X_train, y_train)
    models['Decision Tree'] = {
        'model': dt,
        'train_time': time.time() - start
    }
    print(f"   Done in {models['Decision Tree']['train_time']:.2f}s")
    
    print("\n3. Training Random Forest...")
    start = time.time()
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    models['Random Forest'] = {
        'model': rf,
        'train_time': time.time() - start
    }
    print(f"   Done in {models['Random Forest']['train_time']:.2f}s")
    
    print("\n4. Training SVM...")
    start = time.time()
    svm = SVR(kernel='rbf', C=100, gamma=0.1)
    svm.fit(X_train, y_train)
    models['SVM'] = {
        'model': svm,
        'train_time': time.time() - start
    }
    print(f"   Done in {models['SVM']['train_time']:.2f}s")
    
    print("\n5. Training Neural Network...")
    # Using sklearn's MLPRegressor instead of TensorFlow for simplicity
    from sklearn.neural_network import MLPRegressor
    start = time.time()
    nn = MLPRegressor(hidden_layer_sizes=(50, 25), max_iter=100, random_state=42)
    nn.fit(X_train, y_train)
    models['Neural Network'] = {
        'model': nn,
        'train_time': time.time() - start
    }
    print(f"   Done in {models['Neural Network']['train_time']:.2f}s")
    
    return models

def evaluate_models(models, X_test, y_test):
    """Evaluate all models"""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    results = {}
    
    for name, model_data in models.items():
        model = model_data['model']
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Calculate directional accuracy
        actual_direction = np.diff(y_test) > 0
        pred_direction = np.diff(y_pred) > 0
        directional_accuracy = np.mean(actual_direction == pred_direction) * 100
        
        results[name] = {
            'predictions': y_pred,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'directional_accuracy': directional_accuracy,
            'train_time': model_data['train_time']
        }
        
        print(f"\n{name}:")
        print(f"  RMSE: ${rmse:.2f}")
        print(f"  MAE: ${mae:.2f}")
        print(f"  R²: {r2:.4f}")
        print(f"  Directional Accuracy: {directional_accuracy:.1f}%")
    
    return results

def save_models(models, scaler):
    """Save trained models"""
    import os
    os.makedirs('models', exist_ok=True)
    
    for name, model_data in models.items():
        filename = name.lower().replace(' ', '_')
        joblib.dump(model_data['model'], f'models/{filename}.pkl')
    
    joblib.dump(scaler, 'models/scaler.pkl')
    print("\n✅ All models saved to 'models/' folder")