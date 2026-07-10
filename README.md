# XYLOFY AI Internship Projects

Welcome to the XYLOFY AI Internship repository! This workspace contains three comprehensive data analysis and machine learning projects completed during the internship program. Each project demonstrates practical applications of data science, exploratory data analysis, statistical modeling, and predictive analytics.

## 📋 Project Overview

### 1. **Employee Attrition Analysis** (Week 2)
**Location:** `Employee_Attrition/`

A comprehensive HR analytics project focused on analyzing employee attrition patterns and building predictive models to identify factors contributing to employee turnover.

**Key Features:**
- Exploratory Data Analysis (EDA) on HR employee data
- Identification of attrition trends and patterns
- Feature engineering and statistical analysis
- Predictive modeling for employee attrition
- Interactive visualizations and charts

**Files:**
- `analysis.ipynb` - Main Jupyter notebook with complete workflow
- `WA_Fn-UseC_-HR-Employee-Attrition.csv` - HR employee dataset
- `Charts/` - Generated visualizations and charts

**Getting Started:**
```bash
# Open the notebook in VS Code or Jupyter
jupyter notebook Employee_Attrition/analysis.ipynb
```

---

### 2. **House Price Prediction** (Week 1)
**Location:** `House_Price_Prediction/`

An introductory data analysis project focused on exploring housing data and building predictive models for house price estimation.

**Key Features:**
- Exploratory data analysis (EDA) on housing dataset
- Data visualization and statistical insights
- Feature analysis and relationships
- Model development for price prediction
- Visual output generation

**Files:**
- `anaylsis.ipynb` - Jupyter notebook with data exploration and analysis
- `housing.csv` - Housing dataset
- `Charts/` - Generated visualizations and outputs

**Getting Started:**
```bash
# Open the notebook in VS Code or Jupyter
jupyter notebook House_Price_Prediction/anaylsis.ipynb
```

---

### 3. **Sales Forecasting System** (Advanced Project)
**Location:** `Sales_Forecasting_System/`

An end-to-end sales forecasting and demand intelligence system using advanced time-series analysis and machine learning techniques.

**Key Features:**
- Time-series analysis and decomposition
- Multiple forecasting models (SARIMA, Prophet, XGBoost)
- Segment-level forecasting by category and region
- Anomaly detection (Isolation Forest, Rolling Z-Score)
- Customer demand clustering and segmentation
- Interactive Streamlit dashboard

**Models:**
- **SARIMA:** Statistical model for seasonal time-series data
- **Prophet:** Robust business forecasting with trend and seasonality
- **XGBoost:** Machine learning approach with lag-based features

**Files:**
- `analysis.ipynb` - Main notebook with complete workflow
- `app.py` - Streamlit dashboard application
- `train.csv` - Primary sales dataset
- `vgsales.csv` - Supplemental dataset
- `requirements.txt` - Project dependencies
- `Charts/` - Generated forecasts, anomalies, and clustering results

**Getting Started:**

1. Install dependencies:
```bash
pip install -r Sales_Forecasting_System/requirements.txt
```

2. Run the analysis notebook:
```bash
jupyter notebook Sales_Forecasting_System/analysis.ipynb
```

3. Launch the dashboard:
```bash
streamlit run Sales_Forecasting_System/app.py
```

---

## 🛠️ Tech Stack

**Common Technologies Across All Projects:**
- **Python 3.x** - Core programming language
- **Jupyter Notebooks** - Interactive analysis and documentation
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computations
- **Matplotlib & Seaborn** - Data visualization
- **Scikit-learn** - Machine learning models and utilities

**Project-Specific Libraries:**
- **Employee Attrition:** Statistical analysis, classification models
- **House Price Prediction:** Regression models, feature analysis
- **Sales Forecasting:** Statsmodels (SARIMA, ADF testing), Prophet, XGBoost, Streamlit

---

## 📁 Directory Structure

```
XYLOFY_AI_INTERNSHIP/
├── Employee_Attrition/
│   ├── analysis.ipynb
│   ├── WA_Fn-UseC_-HR-Employee-Attrition.csv
│   ├── README.md
│   └── Charts/
├── House_Price_Prediction/
│   ├── anaylsis.ipynb
│   ├── housing.csv
│   ├── README.md
│   └── Charts/
├── Sales_Forecasting_System/
│   ├── analysis.ipynb
│   ├── app.py
│   ├── requirements.txt
│   ├── train.csv
│   ├── vgsales.csv
│   ├── README.md
│   ├── Charts/
│   └── archive/
└── README.md (this file)
```

---

## 🚀 Quick Start Guide

### For First-Time Users:

1. **Clone or navigate to the repository:**
   ```bash
   cd XYLOFY_AI_INTERNSHIP
   ```

2. **Set up a Python environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install common dependencies:**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter
   ```

4. **Start with a project:**
   - **Beginners:** Start with `House_Price_Prediction` for a simple introduction
   - **Intermediate:** Move to `Employee_Attrition` for classification and HR analytics
   - **Advanced:** Explore `Sales_Forecasting_System` for advanced time-series and ML techniques

### Running Specific Projects:

**Employee Attrition:**
```bash
cd Employee_Attrition
jupyter notebook analysis.ipynb
```

**House Price Prediction:**
```bash
cd House_Price_Prediction
jupyter notebook anaylsis.ipynb
```

**Sales Forecasting System:**
```bash
cd Sales_Forecasting_System
pip install -r requirements.txt
jupyter notebook analysis.ipynb
# Optional: Launch dashboard
streamlit run app.py
```

---

## 📊 Key Learning Outcomes

Through these projects, you will learn:

✅ **Data Exploration & Cleaning** - Handle real-world datasets with missing values and outliers  
✅ **Visualization** - Create compelling charts and dashboards for insights  
✅ **Statistical Analysis** - Apply statistical tests and time-series decomposition  
✅ **Predictive Modeling** - Build and evaluate classification and regression models  
✅ **Time-Series Forecasting** - Use SARIMA, Prophet, and XGBoost for forecasting  
✅ **Anomaly Detection** - Identify unusual patterns in business data  
✅ **Clustering & Segmentation** - Group data for business insights  
✅ **Dashboard Development** - Create interactive web apps with Streamlit  

---

## 📝 Notes

- Each project folder contains a detailed `README.md` with project-specific instructions
- All datasets are included in their respective project directories
- Generated charts and outputs are saved in the `Charts/` subfolder of each project
- For reproducibility, use the same Python version and installed package versions
- Refer to individual project READMEs for detailed documentation

---

## 📧 Contact & Support

For questions or issues, refer to the individual project READMEs or the main documentation within each Jupyter notebook.

---

**Happy learning and analyzing! 🎉**
