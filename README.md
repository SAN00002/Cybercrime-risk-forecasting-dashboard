🔐 Cybercrime Financial Loss Prediction Dashboard (2025–2034)

This project is a full-stack data analytics solution to predict and visualize **future financial losses due to cybercrime** across countries and attack types. It uses Python for machine learning and Power BI for interactive data visualization.

📌 Problem Statement

Cyberattacks are becoming more frequent and costlier across the globe. This project forecasts **predicted financial losses by country from 2025 to 2034**, helping policymakers and analysts identify:
- High-risk countries
- Attack types that cause maximum damage
- Severity level trends
- Resolution bottlenecks

🛠 Tools & Technologies

| Tool            | Purpose                                       |
|------------------|-----------------------------------------------|
| Python      | Data cleaning, feature encoding, modeling     |
| Pandas      | Data preprocessing                            |
| Scikit-learn | Machine learning (`RandomForestRegressor`)    |
| Power BI*    | Dashboard creation and insight visualization  |

🧠 Workflow

1️⃣ Data Preparation in Python
- Cleaned the raw dataset and renamed columns
- Applied **Label Encoding** to handle categorical variables
- Log-transformed the target (`financial_loss`) to handle skew

2️⃣ Machine Learning Model
- Used `RandomForestRegressor` to model log-scale financial losses
- Evaluated using **MSE** and **R² score**
- Applied the model to predict losses for future years (2025–2034)
- Inverted log predictions using `np.expm1()`
- Converted predicted loss from **millions to billions** for reporting

3️⃣ Export & Visualization
- Saved predictions to CSV
- Imported the file into Power BI
- Built an interactive dashboard with slicers, maps, line charts, decomposition tree, and influencer analysis

📊 Dashboard Features (Power BI)

- ✅ Total predicted financial loss ($5.41 Billion)
- 🌍 Choropleth map showing country-wise risk exposure
- 🧠 Key Influencers explaining attack-type likelihood
- 🔄 Decomposition Tree to drill from total → year → country → type
- 📈 Year-wise financial loss trend (2015–2034)
- 📊 Bar charts for attack types, incident resolution times
- 🧩 Severity-level donut chart by country

 







