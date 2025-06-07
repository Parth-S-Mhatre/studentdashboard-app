import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score,accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler,PolynomialFeatures

df=pd.read_csv('DATA//student_dataset.csv')
print(f"DATASET HEAD:\n{df.head()}")
print(f"DATASET COLUMNS: {df.columns.tolist()}")
print(f"DATASET INFO:")
print(df.info())

# Check what columns actually exist
print(f"All columns: {list(df.columns)}")

le = LabelEncoder()
df['Parental_Education'] = le.fit_transform(df['Parental_Education'])

# Define columns to one-hot encode
one_hot_cols = ['School_Type', 'Location']

ct = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(drop='first'), one_hot_cols)  # drop='first' to avoid dummy variable trap
    ],
    remainder='passthrough'  # keep the other columns as is
)

X = df.drop('Score', axis=1)
print(f"Features after dropping Score: {X.columns.tolist()}")
y = df['Score']

X_transformed = ct.fit_transform(X)

# Convert transformed data back to DataFrame
# Get the new feature names from one-hot encoder
onehot_feature_names = ct.named_transformers_['onehot'].get_feature_names_out(one_hot_cols)

# Combine with the other columns
other_cols = [col for col in X.columns if col not in one_hot_cols]
all_feature_names = list(onehot_feature_names) + other_cols

# Create a DataFrame
X_df = pd.DataFrame(X_transformed.toarray() if hasattr(X_transformed, "toarray") else X_transformed, columns=all_feature_names)
print(f"Final feature names: {all_feature_names}")

# Split the dataset into training and testing sets
#*******************SIMPLE LINEAR REGRESSION MODEL*******************

X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.3, random_state=42)
simple_lr=LinearRegression()
simple_lr.fit(X_train,y_train)
y_pred_simple=simple_lr.predict(X_test)

# Calculate the performance metrics
mse_simple=mean_squared_error(y_test,y_pred_simple)/2
r2_simple=r2_score(y_test,y_pred_simple)
print(f"Simple Linear regression MSE:{mse_simple}")
print(f"Simple Linear regression R2_Score:{r2_simple}")

#*****************CROSS VALIDATION*******************
cv_lr=LinearRegression()
cv_score=cross_val_score(cv_lr,X_transformed,y,cv=5,scoring='neg_mean_squared_error')
mse_cv=-np.mean(cv_score)
print(f"Cross Validation MSE:{mse_cv}")

#***SCALING THE DATASET***
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_transformed)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model_scaled = LinearRegression()
model_scaled.fit(X_train, y_train)
y_pred = model_scaled.predict(X_test)
# Calculate the performance metrics for scaled model
print(f" Scaling MSE: {mean_squared_error(y_test, y_pred):.2f}")
print(f" Scaling R²: {r2_score(y_test, y_pred):.2f}\n")

# Polynomial Regression +Scaling+cross validation
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_scaled)
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Step 4: Scaling + Polynomial Features + Train/Test Split")
print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
print(f"R²: {r2_score(y_test, y_pred):.2f}")

import joblib

# Save model and transformers
joblib.dump(simple_lr, "model.pkl")
joblib.dump(le, "label_encoder.pkl")       # For Parental_Education
joblib.dump(ct, "column_transformer.pkl")  # For OneHotEncoder
joblib.dump(scaler, "scaler.pkl")          # (optional, for scaled model)

# Save the feature columns for reference
joblib.dump(X.columns.tolist(), "feature_columns.pkl")

print("Model and transformers saved successfully!")
print(f"Training completed with {len(X.columns)} features: {X.columns.tolist()}")