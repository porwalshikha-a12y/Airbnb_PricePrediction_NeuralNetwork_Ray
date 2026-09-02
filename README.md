# Airbnb Pricing Prediction & Clustering Analysis  
**CS5811 – Distributed Data Analysis Project**

---

## Project Overview  
This project analyses Airbnb listings in London to understand pricing patterns using:

- Exploratory Data Analysis (EDA)  
- Machine Learning models  
- Clustering techniques  
- Distributed computing (Apache Spark)  

The objective is to identify key factors affecting Airbnb prices and build scalable predictive models.

---

## Objectives  
- Identify key drivers of Airbnb listing prices  
- Segment listings using clustering techniques  
- Compare performance of multiple ML models  
- Implement scalable pipelines using distributed computing  
- Evaluate accuracy vs scalability trade-offs  

---

## 📂 Project Structure  

airbnb-dda-2026/
│
├── 01_raw_data/          
├── 02_processed_data/    
├── 03_code/              
├── 04_outputs/           
│     ├── plots/          
│     ├── tables/         
│     └── models/         
├── 05_documentation/     
├── 06_reports/           
└── README.md            

---

## 📊 Datasets Used  

The project integrates multiple datasets:

- Airbnb listings dataset (Inside Airbnb – London)  
- London crime dataset  
- Transport for London (TfL) data  
- London income and housing dataset  
- Tourist attractions dataset  

These datasets are combined to capture spatial, socio-economic, and environmental factors affecting pricing.

---

## ⚙️ Technologies Used  

- **Languages:** Python, R  
- **Libraries:** Pandas, NumPy, Scikit-learn  
- **Visualization:** Matplotlib, Seaborn  
- **Big Data Tools:** Apache Spark (PySpark)  
- **Storage:** Azure Data Lake (ADLS Gen2)  
- **Version Control:** GitHub  

---

## 🔍 Methodology  

### 1. Data Preprocessing  
- Handling missing values  
- Feature engineering  
- Encoding categorical variables  
- Scaling numerical features  

### 2. Exploratory Data Analysis  
- Price distribution analysis  
- PCA
- Outlier detection  
- Correlation analysis  

### 3. Clustering  
- K-Means clustering  
- Elbow method  
- Silhouette analysis  

### 4. Machine Learning Models
- Neural Networks  

### 5. Distributed Computing  
- Spark ML pipelines  
- Ray Tune - Parallel model training 
- Performance comparison with single-node models

---

## Key Insights  

- Location and amenities are strong predictors of price  
- Listings can be segmented into distinct pricing groups  
- Tree-based models provide strong performance and interpretability  
- Distributed computing improves scalability significantly  

---

## How to Run  

### 1. Clone Repository  
```bash
git clone https://github.com/porwalshikha-a12y/Airbnb_NeuralNetwork_Ray.git
cd airbnb-neuralnetwork_project