# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
The model is a Random Forest classifier developed to predict whether an individual's annual income is greater than $50,000 or less than or equal to $50,000. The model was trained using scikit-learn and categorical features were transformed using one-hot encoding.

## Intended Use
The model is intended as an educational classification model for predicting income categories from Census data. It demonstrates an end-to-end machine learning workflow including preprocessing, model training, evaluation, slice-based performance analysis, and API deployment. It should not be used to make high-stakes employment, lending, eligibility, or other consequential decisions about individuals.

## Training Data
The model was developed using the provided Census dataset. The dataset contains demographic and employment-related features such as age, workclass, education, occupation, relationship, race, sex, hours worked per week, and native country. The target variable is salary, which contains the classes <=50K and >50K.

The data was split into separate training and test datasets. Categorical features were transformed using one-hot encoding, and the salary target was transformed using a label binarizer.

## Evaluation Data
The model performance was evaluated using the test portion of the Census dataset that was held out from model training. Performance was also evaluated across individual values of categorical features, with the slice-level metrics written to slice_output.txt.

## Metrics
The model was evaluated using precision, recall, and F1 score.

The test-set performance was:

Precision: 0.7419
Recall: 0.6384
F1 score: 0.6863
Precision measures how often positive predictions are correct, recall measures the model's ability to identify positive cases, and the F1 score balances precision and recall.

## Ethical Considerations
The dataset contains demographic characteristics, including race and sex, so model predictions may reflect patterns or biases present in the underlying Census data. Performance can also vary across subgroups, which is why the project evaluates model performance across categorical slices.

The model should not be interpreted as determining an individual's worth, ability, or suitability for opportunities and should not be used as the sole basis for decisions that materially affect individuals.

## Caveats and Recommendations
The model is a baseline classifier created for an educational machine learning deployment project. Its performance may not generalize to populations or data that differ from the Census dataset used for development.

Slice-level results should be reviewed to identify categories where performance differs from the overall model. Additional evaluation, bias analysis, model tuning, and validation on current and representative data would be appropriate before considering the model for any real-world application.