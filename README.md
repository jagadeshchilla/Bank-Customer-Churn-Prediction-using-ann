# Bank Customer Churn Prediction 🏦

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-3.11.3-red.svg)](https://keras.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7.1-green.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Project Overview 🎯

This project implements an Artificial Neural Network (ANN) to predict customer churn in a bank. The model uses various customer features to determine the likelihood of a customer leaving the bank's services.

## Features 📊

- Customer demographic information (Age, Gender, Geography)
- Banking relationship metrics (Credit Score, Balance, Products)
- Advanced preprocessing with feature scaling and encoding
- Hyperparameter tuning for optimal model performance
- TensorBoard integration for model monitoring
- Model persistence for deployment

## Model Architecture 🧠

The ANN model includes:
- Input layer matching feature dimensions
- Configurable hidden layers (1-2 layers)
- Variable neurons per layer (16, 32, 64, or 128)
- Binary classification output layer
- ReLU activation for hidden layers
- Sigmoid activation for output layer

## Project Structure 📁

```
annclassification/
├── app.py                     # Application interface
├── Churn_Modelling.csv       # Dataset
├── experiments.ipynb         # Initial model experiments
├── hyperparametertuningann.ipynb  # Hyperparameter optimization
├── prediction.ipynb          # Model prediction examples
├── salaryregression.ipynb    # Salary prediction model
├── model.h5                  # Saved model
├── scaler.pkl               # Feature scaler
├── label_encoder_gender.pkl  # Gender encoder
├── onehot_encoder_geo.pkl   # Geography encoder
└── logs/                    # TensorBoard logs
    └── fit/
```

## Model Performance 📈

The model achieves high accuracy through:
- Cross-validation during training
- Early stopping to prevent overfitting
- Grid search for optimal hyperparameters
- Performance monitoring via TensorBoard

## Getting Started 🚀

1. Create the virtual environment:
```bash
conda create -p venv python==3.11 -y
```

2. Install required packages:
```python
pip install tensorflow scikit-learn pandas numpy scikeras
```

3. Run the notebooks in the following order:
   - `experiments.ipynb` for initial model development
   - `hyperparametertuningann.ipynb` for model optimization
   - `prediction.ipynb` for making predictions

## Model Usage 💡

```python
# Example prediction
input_data = {
    'CreditScore': 600,
    'Geography': 'France',
    'Gender': 'Male',
    'Age': 40,
    'Tenure': 3,
    'Balance': 60000,
    'NumOfProducts': 2,
    # ... other features
}
```

## Monitoring & Visualization 📊

Use TensorBoard to monitor training:
```bash
tensorboard --logdir logs/fit
```


