# Dual Dynamic Transformer Project

## Overview

The Dual Dynamic Transformer project is an innovative approach to predicting laptop prices using advanced machine learning techniques. This project leverages the power of transformer models, which have revolutionized natural language processing, and applies them to the domain of price prediction. By utilizing a dual transformer architecture, this project aims to achieve higher accuracy and robustness in predictions.

## Unique Innovation

### Dual Transformer Architecture

The core innovation of this project lies in its dual transformer architecture. Unlike traditional models, this approach uses two transformer models in tandem to enhance prediction accuracy. The first transformer model is trained on a comprehensive dataset to learn the general patterns and relationships between features and prices. The second transformer model is then fine-tuned with additional adjustments to further refine the predictions. This allows for the second transformer to create a higher order differential to adjust the first transformer's values when quering unkown data sets. This is a more accurate approach than direct training on the data set as it creates a higher order differential on inferencing.

### Advanced Preprocessing and Encoding

The project employs sophisticated preprocessing and encoding techniques to handle both numerical and categorical data effectively. By using pipelines for standard scaling and one-hot encoding, the data is transformed into a format that is optimal for training transformer models.

### Early Stopping and Exponential Decay

To prevent overfitting and ensure efficient training, the project incorporates early stopping and exponential decay in the learning rate. These techniques help in achieving a balance between training time and model performance.

## Database Explanation

### Data Sources

The project utilizes multiple CSV files containing laptop data from various brands. The primary dataset, `Model1_Full.csv`, includes laptops from Asus, Acer, and Lenovo. Additional datasets like `HP_Dell_laptops.csv` are used for testing and validation purposes.

### Data Preprocessing

1. **Loading and Cleaning**: The datasets are loaded and cleaned by dropping rows with missing values in the target column (`Price`).
2. **Feature Selection**: Relevant features such as `Brand`, `Processor_Speed`, `RAM_Size`, `Storage_Capacity`, `Screen_Size`, and `Weight` are selected for training.
3. **Encoding**: Numerical features are scaled using `StandardScaler`, and categorical features are encoded using `OneHotEncoder`.
4. **Splitting**: The data is split into training and test sets to evaluate model performance.

### Model Training

The transformer models are built and trained using TensorFlow and Keras. The training process includes compiling the model with the Adam optimizer and categorical cross-entropy loss function. Early stopping is used to monitor validation loss and prevent overfitting.

### Predictions and Evaluation

The trained models are used to predict laptop prices on both the test set and unseen data. The predictions are decoded and compared with the true prices to evaluate model accuracy. Visualization of training and validation loss helps in understanding the model's performance over epochs.

## Conclusion

The Dual Dynamic Transformer project represents a significant advancement in the field of price prediction. By combining the strengths of transformer models with advanced preprocessing techniques, this project sets a new benchmark for accuracy and robustness in predictive modeling.
