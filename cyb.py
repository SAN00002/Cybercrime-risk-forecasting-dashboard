import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ---------------------------
# 1. Load and clean the dataset
# ---------------------------

df = pd.read_csv("enhanced_cybercrime.csv")
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# ✅ Map country codes to names (if needed)
country_map = {
    1: "USA", 2: "India", 3: "Germany", 4: "UK", 5: "Japan",
    6: "China", 7: "France", 8: "Canada", 9: "Brazil", 10: "Australia"
}
if df['country'].dtype in [int, float]:
    df['country'] = df['country'].map(country_map)

# ---------------------------
# 2. Label encode historical data & save encoders
# ---------------------------

le_dict = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    le_dict[col] = le  # Store encoder

# ---------------------------
# 3. Train the regression model
# ---------------------------

X = df.drop(columns=['financial_loss_(in_million_$)', 'severity', 'cluster', 'predicted_loss'], errors='ignore')
y = df['financial_loss_(in_million_$)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("RMSE:", round(rmse, 2))
print("R² Score:", round(r2, 2))

# ---------------------------
# 4. Create future input data (2025–2034) with variation
# ---------------------------

countries = ["USA", "India", "Germany", "UK", "Japan", "France", "China", "Brazil", "Canada", "Australia"]
future_rows = []

for year in range(2025, 2035):
    for i, country in enumerate(countries):
        future_rows.append({
            'year': year,
            'country': country,
            'attack_type': ['Ransomware', 'Phishing', 'Malware', 'DDoS'][i % 4],
            'target_industry': ['IT', 'Finance', 'Healthcare', 'Retail'][i % 4],
            'attack_source': ['Hacker Group', 'Insider', 'Botnet'][i % 3],
            'security_vulnerability_type': ['Unpatched Software', 'Weak Passwords', 'Misconfigurations'][i % 3],
            'defense_mechanism_used': ['Antivirus', 'Firewall', 'Training'][i % 3],
            'number_of_affected_users': 100000 * (i + 1),
            'incident_resolution_time_(in_hours)': 20 + (i * 2)
        })

future_df = pd.DataFrame(future_rows)
future_df['country_name'] = future_df['country']
# ---------------------------
# 5. Encode all columns except 'country'
# ---------------------------

encoded_future = future_df.copy()

for col in encoded_future.select_dtypes(include='object').columns:
    if col in le_dict:
        known_classes = list(le_dict[col].classes_)
        encoded_future[col] = encoded_future[col].apply(
            lambda x: x if x in known_classes else known_classes[0]
        )
        encoded_future[col] = le_dict[col].transform(encoded_future[col].astype(str))

# Ensure column order matches training set
encoded_future = encoded_future[X.columns]  # includes encoded 'country'

# Predict loss
future_df['predicted_loss'] = model.predict(encoded_future)

# Export with readable country names
future_df['country'] = future_df['country_name']
future_df.drop(columns=['country_name'], inplace=True)
future_df.to_csv("predicted_loss_by_country_2025_2034.csv", index=False)
print("Saved to predicted_loss_by_country_2025_2034.csv")


