import pandas as pd

df = pd.read_csv('parkinsons.csv')
df = df.dropna()
df.head()
print(df.columns.to_list())
import seaborn as sns
import matplotlib.pyplot as plt

selected_columns = ['spread1', 'MDVP:Fo(Hz)', 'status']
sns.pairplot(df[selected_columns], hue='status', diag_kind='kde', corner=True)
plt.show()

from sklearn.preprocessing import MinMaxScaler

features = ['spread1', 'MDVP:Fo(Hz)']
X = df[features]  

scaler = MinMaxScaler()

X_scaled = scaler.fit_transform(X)

print(X_scaled[:5])

from sklearn.model_selection import train_test_split

y = df['status']

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=0)

print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)

from sklearn.svm import SVC

model = SVC()
model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

import joblib

joblib.dump(model, 'parkinsons_model.joblib')
