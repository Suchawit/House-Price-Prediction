# House-Price-Prediction
The purpose of the project is to have a better understanding of machine learning algorithms learnt from the academic lecture by implementing the algorithms to the specific problem and check if the result matches the expectation. The main goal for the project is to predict the price of residential homes located in Ames , Lowa, which requires training the system by letting the system learn from the data provided from the Kaggle about the relationship between each different attribute of the house and the price. Overall, the project uses both XGBoost regression and Deep Learning to achieve the house price prediction, and both machine learning algorithms come up with good prediction rates. This problem provides 80 unique data columns with the final objective to predict the house selling price. With a relatively large quantity of inputs to consider, it is challenging for a human to estimate an accurate sale price. The project plan is illustrated below

<img src="https://github.com/Suchawit/House-Price-Prediction/blob/main/img/Illustrate.PNG" width="800px"/>


## Dataset

Dataset is available to be downloaded from [**here**](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

## Dependency

    pip install -r requirements.txt

## Preprocessing Data

Downloaded data and stored using Pandas library, then checked each column if it contained more than 75% miss values, these columns will not be considered. Since many columns contained object type values including NaN, labeled those object type values to integer type, and set NaN to that column mean ([**Code**](https://github.com/Suchawit/House-Price-Prediction/blob/main/House-Price-Prediction-Project_finished.ipynb))

## Models and Results

### Xgboost with Default Setting

<img src="https://github.com/Suchawit/House-Price-Prediction/blob/main/img/xgb_default.PNG" width="500px"/> 
<img src="https://github.com/Suchawit/House-Price-Prediction/blob/main/img/Result_xgb.PNG" width="300px"/>

### Xgboost with Regularization

<img src="https://github.com/Suchawit/House-Price-Prediction/blob/main/img/xgb_regularization.PNG" width="500px"/> 
<img src="https://github.com/Suchawit/House-Price-Prediction/blob/main/img/Result_xgb_reg.PNG" width="300px"/>

### Deep Learning Keras

<img src="https://github.com/Suchawit/House-Price-Prediction/blob/main/img/big_deep_learning_model_keras.PNG" width="900px"/> 
<img src="https://github.com/Suchawit/House-Price-Prediction/blob/main/img/Result_big_deep_learning_model_keras.PNG" width="300px"/>


### Optimized Deep Learning Keras

<img src="https://github.com/Suchawit/House-Price-Prediction/blob/main/img/optimized_deep_learning_model_keras.PNG" width="900px"/> 
<img src="https://github.com/Suchawit/House-Price-Prediction/blob/main/img/Result_optimized_deep_learning_model_keras.PNG" width="300px"/>

### Optimized Deep Learning Pytorch

<img src="https://github.com/Suchawit/House-Price-Prediction/blob/main/img/deep_learning_pytorch.PNG" width="900px"/> 
<img src="https://github.com/Suchawit/House-Price-Prediction/blob/main/img/Result_deep_learning_pytorch.PNG" width="300px"/>
