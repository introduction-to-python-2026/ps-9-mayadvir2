import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline

df = pd.read_csv('parkinsons.csv')
df = df.dropna()

print(df.columns.to_list())

selected_columns = ['spread1', 'MDVP:Fo(Hz)', 'status']
sns.pairplot(df[selected_columns], hue='status', diag_kind='kde', corner=True)
plt.show()

features = ['spread1', 'MDVP:Fo(Hz)']
X = df[features]
y = df['status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)

model = make_pipeline(MinMaxScaler(), SVC())
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

joblib.dump(model, 'parkinsons_model.joblib')

config_content = """features: ["spread1", "MDVP:Fo(Hz)"]
path: "parkinsons_model.joblib"
"""
with open("config.yaml", "w") as f:
    f.write(config_content)
