The Restaurant Satisfaction Study¶
 
In this project, we use a dataset of a restaurant from Kaggle and use multiple supervised machine learning models to predict customer high satisfaction at the restaurant. The main objective of this analysis is to deliver two goals: a prediction of customer high satisfaction for the restaurant and an interpretation of the existing features so that the restaurant can focus on the features that lead to great customer satisfaction. First we start by conducting an Exploratory Data Analysis and then move on to solving the problem using Machine Learning models.

Upon examining the dataset for EDA and looking for missing and duplicate values, we found that the dataset was very clean already, which is rare in real world problems (for practice, we could generate a random 10% missing/duplicated values in a future exercise). 

The dataset includes many features: CustomerID, Age, Gender, Income, VisitFrequency, AverageSpend, PreferredCuisine, TimeOfVisit, GroupSize, DiningOccasion, MealType, OnlineReservation, DeliveryOrder, LoyaltyProgramMember, WaitTime, ServiceRating, FoodRating, AmbianceRating, HighSatisfaction. Almost all of the features are self-explanatory. We have studied the features and their multicollinearity extensively in the EDA process, and discovered that some of them are more useful than others to be used in the ML models. However, we have used all of the features in the ML models. In a future exercise, we can eliminate some of them and retrain the models to measure their improvements. The EDA analysis has helped us understand the data and its values and limitations, as well as outliers and distributions of the features. 

We identified three types of feature types: binary, numerical, and categorical. During feature engineering, we removed the CustomerID column and created a new column named “income per age” to see if income per age can help in the modeling. Then we split the data and used stratified splitting. We used StratifiedShuffleSplit as well, but for some reason, the simple train_test_split with the stratify option yielded slightly better results (the differences need to be investigated in the future).

After splitting the data into train and test sets, we encoded the numerical and categorical features to make them ready for the models. We used MinMaxScaler and OneHotEncoder. We used OneHotEncoder as opposed to get_dummies so that we can maintain the indices for later use in the notebook if needed. As mentioned, we splitted the data already, so we encoded the train and test sets separately to avoid data leakage. 

We defined multiple functions to get accuracy, evaluate the performance metrics, measure performance errors, and generate the confusion matrix. We constructed multiple tree based classification models: decision trees, random forest, gradient boosting, and XGBoost. For each model, we defined the base model without any constraints, then for some of the models, we defined a second model with some intuitive constraints and then for all models, we used a GridSearchCV to get the best parameters in the model and refit the model based on the “recall” metric. Since our dataset is very unbalanced and we are measuring the positive minority class, we needed to optimize the performance for recall, rather than other metrics. Another metric worth trying is the f1 metric since it balances the precision and recall and it could be useful in optimizing our models. For each model, we printed the evaluation metrics as well as a confusion matrix and feature importance (for XGBoost only).

Dealing with a severely unbalanced dataset, we explored options in the Imbalanced-learn library where we could use the parameter class_weight="balanced" in the classifier object to improve the performance of imbalanced datasets. The Gradient Boosting Classifier did not have the class_weight="balanced" param option to be added to the classifier object, but the rest of the models had the option. There are other ways to improve the performance of the models with imbalanced datasets, such as random under-sampling and using a balanced bagging classifier from imblearn.ensemble that are worth trying.
As an alternative cross validation option other than the GridSearchCV, we examined all the models by using the cross_validate method in sklearn.model_selection and present the findings in the section below.


Key findings:

The results from the cross validation runs can be shown in the table below. We observe that the grid search has improved the “balanced accuracy” (as opposed to accuracy). For recall and precision, they sometimes behave opposite to each other, but looking at the f1 score, we see that the balanced XGboost is the the champion model. In the future, we are planning to explore more in the areas of upsampling, downsampling, or resampling (SMOTE, etc.), putting the models in a pipeline object for automation, using other methods to overcome the severe unbalancing sets issue, as well as classifying using other models such as KNN, SVM, ANN, and logistic regression. 

![Snip20240925_73](https://github.com/user-attachments/assets/3f478504-7abf-41b4-9ecc-2c57e2756eaf)


Published notebook for the Restaurant satisfaction study: 
https://nbviewer.org/github/talinm23/ML/blob/main/Projects/restaurant_customer_satisfaction/Restaurant_satisfaction.ipynb

