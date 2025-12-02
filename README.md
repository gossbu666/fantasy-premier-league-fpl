# Fantasy Premier League Points Prediction

> Machine learning system for predicting FPL player points using ensemble methods

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![ML](https://img.shields.io/badge/ML-Ensemble-green.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🎯 Overview

Predicts Fantasy Premier League player points using position-specific ensemble models (XGBoost + LightGBM + Random Forest) with advanced feature engineering.

**Performance**: Models explain 89-97% of variance in player points (2024-25 test set)

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/gossbu666/fantasy-premier-league-fpl.git
cd fantasy-premier-league-fpl

# Install dependencies
pip install -r requirements.txt

# Run web application
python flask_app/app.py

# Or use Streamlit
streamlit run .streamlit/app.py
```

## 📂 Project Structure

```
fantasy-premier-league-fpl/
├── .streamlit/              # Streamlit dashboard
│   └── app.py
│
├── Prepare_Presentation/    # Project documentation & presentation
│   ├── Data Collection & Preparation/
│   ├── Feature Engineering/
│   ├── Model Training & Optimization/
│   ├── Model Evaluation/
│   ├── Prediction & Deployment/
│   ├── Squad Optimization/
│   └── Utilities & Helpers/
│
├── analysis/                # Analysis notebooks & scripts
│   ├── data_exploration.py
│   ├── feature_analysis.py
│   └── model_comparison.py
│
├── flask_app/               # Flask web application
│   ├── app.py
│   ├── templates/
│   ├── static/
│   └── utils/
│
├── notebooks/               # Jupyter notebooks
│   ├── 01_data_collection.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training.ipynb
│
├── src/                     # Source code
│   ├── data/               # Data processing
│   ├── features/           # Feature engineering
│   ├── models/             # Model training
│   ├── evaluation/         # Model evaluation
│   ├── optimization/       # Squad optimization
│   └── utils/              # Utility functions
│
├── requirements.txt
└── README.md
```

## 📊 Model Performance (2024-25 Test Set)

| Position | R² Score | RMSE |
|----------|----------|------|
| GK | 0.898 | 0.831 |
| DEF | 0.891 | 0.856 |
| MID | 0.948 | 0.537 |
| FWD | 0.977 | 0.384 |

**Interpretation**: Models explain 89-97% of variance in player points across all positions.

## 💻 Usage

### 1. Feature Engineering

Build features for each position:

```bash
python src/features/build_gk_features.py
python src/features/build_def_features.py
python src/features/build_mid_features.py
python src/features/build_fwd_features.py
```

### 2. Train Models

```bash
# Train with hyperparameter optimization
python src/models/train_ensemble.py

# Or tune hyperparameters separately
python src/models/tune_optuna.py
```

### 3. Make Predictions

```bash
# Predict next gameweek
python src/evaluation/predict_next_gw.py

# Predict specific player
python src/evaluation/predict_player.py --player_id 123
```

### 4. Optimize Squad

```bash
# Optimize squad under £100M budget
python src/optimization/optimize_squad.py

# Optimize transfers
python src/optimization/optimize_transfers.py
```

### 5. Run Web App

```bash
# Flask app
python flask_app/app.py
# Access at http://localhost:5000

# Streamlit dashboard
streamlit run .streamlit/app.py
# Access at http://localhost:8501
```

## ✨ Features

### Machine Learning
- ✅ Position-specific ensemble models
- ✅ Hyperparameter optimization (Optuna)
- ✅ Temporal cross-validation
- ✅ Feature importance analysis

### Feature Engineering
- 🔄 Rolling statistics (3, 5, 10 games)
- 📊 Form indicators & momentum
- 🏟️ Home/away performance
- 💪 Fixture difficulty rating (FDR)
- ⚽ Expected goals (xG) & assists (xA)

### Optimization
- 🎯 Squad optimization (£100M budget)
- 🔄 Transfer planning
- ⚖️ Formation flexibility
- 🔝 Captain selection

### Deployment
- 🌐 Flask REST API
- 📱 Streamlit dashboard
- 🔌 API endpoints
- ☁️ Cloud-ready

## 🔧 For Developers

### Setup Development Environment

```bash
# Fork repository on GitHub
git clone https://github.com/YOUR_USERNAME/fantasy-premier-league-fpl.git
cd fantasy-premier-league-fpl

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create feature branch
git checkout -b feature/your-feature-name
```

### Make Changes & Contribute

```bash
# Make your changes
# Test thoroughly

# Commit changes
git add .
git commit -m "feat: Your feature description"

# Push to your fork
git push origin feature/your-feature-name

# Create Pull Request on GitHub
```

### Development Workflow

1. **Add Features**: Implement in `src/` with proper tests
2. **Update Notebooks**: Document analysis in `notebooks/`
3. **Update Web App**: Add UI in `flask_app/` or `.streamlit/`
4. **Documentation**: Update relevant docs in `Prepare_Presentation/`

### To-Do List

- [ ] Add LSTM/Transformer models
- [ ] Implement player injury prediction
- [ ] Handle double gameweeks better
- [ ] Add transfer market value prediction
- [ ] Implement unit tests
- [ ] Set up CI/CD pipeline
- [ ] Add Docker support
- [ ] Create mobile app

### Known Issues

- API rate limiting during peak times
- Double gameweek predictions need refinement
- Need better fixture congestion handling

## 💻 Tech Stack

**ML**: XGBoost, LightGBM, Random Forest, Optuna  
**Web**: Flask, Streamlit  
**Data**: Pandas, NumPy, Scikit-learn  
**Optimization**: PuLP, SciPy  
**Visualization**: Matplotlib, Seaborn, Plotly

## 📚 Documentation

Detailed documentation available in `Prepare_Presentation/`:

- **Data Collection & Preparation**: Data pipeline and preprocessing
- **Feature Engineering**: Feature creation methodology
- **Model Training & Optimization**: Training procedures and hyperparameter tuning
- **Model Evaluation**: Performance metrics and validation
- **Prediction & Deployment**: Deployment guide
- **Squad Optimization**: Optimization algorithms
- **Utilities & Helpers**: Helper functions and utilities

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'feat: Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Commit Message Convention

```
feat: Add new feature
fix: Fix bug
docs: Update documentation
style: Format code
refactor: Refactor code
test: Add tests
chore: Update dependencies
```

## 👥 Authors

- **Shah Md Jobayer** - Model Development & Feature Engineering
  - Email: st126404@ait.asia

- **Supanut Kompayak** - Optimization & Web Development
  - Email: st126055@ait.asia

**Institution**: Asian Institute of Technology (AIT)  
**Program**: Data Science and Artificial Intelligence

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [FPL Official API](https://fantasy.premierleague.com/api/) - Data source
- [vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League) - Historical data
- Understat - xG/xA statistics
- FPL community on Reddit

## 📞 Contact

- **Issues**: [GitHub Issues](https://github.com/gossbu666/fantasy-premier-league-fpl/issues)
- **Email**: st126055@ait.asia
- **Repository**: https://github.com/gossbu666/fantasy-premier-league-fpl

---

**⚽ Happy Predicting! May your FPL rank climb high! 🏆**

*Last Updated: December 2, 2025*
