
!pip install kagglehub -q
import pandas as pd
import kagglehub
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
path = kagglehub.dataset_download(
    "mlg-ulb/creditcardfraud"
)
print("Dataset Path:", path)
df = pd.read_csv(
    os.path.join(path, "creditcard.csv")
)
print(df.head())
print(df['Class'].value_counts())
X = df.drop('Class', axis=1)
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(
    "Accuracy:",
    accuracy_score(y_test, y_pred)
)
print(
    classification_report(
        y_test,
        y_pred
    )
)
print(
    confusion_matrix(
        y_test,
        y_pred
    )
)