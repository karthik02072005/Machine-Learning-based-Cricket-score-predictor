import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
import joblib
import os

# Set working directory to the script location
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 1. Load data and drop any NaNs (crucial for speed and accuracy)
print("Loading data...")
df = pd.read_csv('ipl_ml_ready.csv').dropna()

# 2. Separate Features and Target
X = df.drop(columns=['final_score'])
y = df['final_score']

# 3. Faster Preprocessing
# OrdinalEncoder is much faster than OneHotEncoder because it doesn't 
# create 50+ extra columns. It keeps the data "slim."
categorical_features = ['batting_team', 'bowling_team', 'venue']
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_features)
    ], remainder='passthrough'
)

# 4. Use the "Speed King" Model
# HistGradientBoosting is designed for speed on modern CPUs.
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', HistGradientBoostingRegressor(
        max_iter=50,       # Fewer iterations for faster output
        learning_rate=0.1, 
        max_leaf_nodes=31, # Standard size for good accuracy/speed balance
        random_state=42
    ))
])

# 5. Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Train (This should take seconds, not minutes)
print(f"Training on {len(df)} rows. This will be fast...")
model_pipeline.fit(X_train, y_train)

# 7. Evaluate and Save
prediction = model_pipeline.predict(X_test)
mae = mean_absolute_error(y_test, prediction)

print("-" * 30)
print(f"DONE! Mean Absolute Error: {round(mae, 2)} runs.")
print("Model saved as 'ipl_score_model.pkl'")
print("-" * 30)

joblib.dump(model_pipeline, 'ipl_score_model.pkl')