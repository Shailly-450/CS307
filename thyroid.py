import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/ann-train.data"
columns = ['T3', 'T4', 'TSH', 'Goiter', 'Tumor', 'Hypothyroid', 'Hyperthyroid', 'Pregnant', 'Thyroiditis', 'Label']

data = pd.read_csv(url, header=None, sep='\s+', names=columns)


data.ffill(inplace=True)  


data['Label'] = data['Label'].map({3: 'Normal', 2: 'Hypothyroid', 1: 'Hyperthyroid'})  

X = data.drop('Label', axis=1)
y = data['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


train_data = pd.concat([X_train, y_train], axis=1)

model = BayesianNetwork([('T3', 'Label'), ('T4', 'Label'), ('TSH', 'Label'), ('Goiter', 'Label'),
                         ('Tumor', 'Label'), ('Hypothyroid', 'Label'), ('Hyperthyroid', 'Label')])

model.fit(train_data, estimator=MaximumLikelihoodEstimator)

inference = VariableElimination(model)
y_pred = []

for _, test_data in X_test.iterrows():
    query_result = inference.map_query(variables=['Label'], evidence=test_data.to_dict())
    y_pred.append(query_result['Label'])


accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

if accuracy >= 0.85:
    print("Model meets the expected accuracy!")
else:
    print("Model accuracy is below 85%. Consider tuning the model or performing further feature engineering.")
