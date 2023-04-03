# Media Campaign Cost Prediction

This project aims to predict the cost of media campaigns based on various factors, such as store sales, unit sales, total children, and the presence of specific facilities in the store. The dataset for this project (both train and test) was generated from a deep learning model trained on the [Media Campaign Cost Prediction dataset](https://www.kaggle.com/datasets/gauravduttakiit/media-campaign-cost-prediction).

## Data

The dataset contains information about different media campaigns and their associated costs. The features include:

- Store sales (in millions)
- Unit sales (in millions)
- Total children
- Number of children at home
- Average cars at home (approx.)
- Gross weight
- Recyclable package
- Low fat
- Units per case
- Store sqft
- Coffee bar
- Video store
- Salad bar
- Prepared food
- Florist

The target variable is the cost of the media campaign.

## Code

The notebook file is named "work.ipynb"

The project is divided into several steps:

1. Data exploration and preprocessing
2. Model training and evaluation
3. Model prediction and submission
4. Streamlit app for interactive predictions

The code includes data exploration, handling missing values, feature engineering, normalization, and training of various regression models, such as Linear Regression, Decision Trees, Random Forest, Gradient Boosting, XGBoost, and K-Nearest Neighbors. The performance of the models is evaluated using cross-validation and the Root Mean Squared Log Error (RMSLE) metric.

## Results

The best-performing model was XGBoost, with a mean RMSLE of 0.302298 and a standard deviation of 0.000721. The other models had higher RMSLE scores, indicating that the XGBoost model was better at predicting the cost of media campaigns.

## Streamlit App

The Streamlit app provides a simple user interface to make predictions using the trained XGBoost model. Users can adjust various input features using sliders, and the app will display the predicted cost for the media campaign. The app is light, minimalistic, and visually appealing, with proper labeling of input sliders.

## How it works

1. Install Streamlit and other dependencies.
2. Run the Streamlit app using `streamlit run app.py`.
3. Adjust input sliders in the app to set the values for various factors.
4. View the predicted cost for the media campaign.

## Overall Inference

The project demonstrates the effectiveness of using machine learning techniques, such as XGBoost, to predict the cost of media campaigns based on a variety of factors. The app provides an interactive way to explore the impact of different factors on the predicted cost.

## Future Improvements

1. Perform hyperparameter tuning to optimize the performance of the models further.
2. Experiment with other regression models and ensemble techniques to improve prediction accuracy.
3. Use additional feature engineering techniques to create new features or modify existing ones.
4. Explore the use of synthesized data to augment the dataset and potentially improve model performance.
5. Extend the Streamlit app to include visualizations that help users better understand the relationship between input features and predicted cost.
