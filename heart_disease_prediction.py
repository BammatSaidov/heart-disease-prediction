# Импорт библиотек
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

# Загрузка датасета
df = pd.read_csv("heart.csv")

# Целевая переменная
target = "HeartDisease"
X = df.drop(columns=[target])
y = df[target]

# Разделение признаков по типу
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

# Трансформеры
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Предобработка
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])

# Разделение выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Модели
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(),
    "Support Vector Machine": SVC(),
    "Decision Tree": DecisionTreeClassifier(),
    "Naive Bayes": GaussianNB()
}

results = {}

# Обучение и оценка моделей
for name, model in models.items():
    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = {
        "model": pipe,
        "accuracy": acc
    }
    print(f"{name}: Accuracy = {acc:.2f}")

# Сравнение точности
plt.figure(figsize=(10, 5))
sns.barplot(x=list(results.keys()), y=[r["accuracy"] for r in results.values()])
plt.ylabel("Accuracy")
plt.title("Сравнение моделей по точности")
plt.xticks(rotation=45)
plt.show()

# Выбор лучшей модели
best_model_name = max(results, key=lambda x: results[x]["accuracy"])
best_model = results[best_model_name]["model"]
print(f"\nЛучшая модель: {best_model_name}")

# Матрица ошибок
y_pred_best = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Нет болезни", "Есть болезнь"])
disp.plot(cmap="Blues")
plt.title("Матрица ошибок лучшей модели")
plt.show()

# Ввод новых пациентов через таблицу
new_patients = pd.DataFrame([
    {
        "Age": 65,
        "Sex": "M",
        "ChestPainType": "ASY",
        "RestingBP": 140,
        "Cholesterol": 289,
        "FastingBS": 1,
        "RestingECG": "LVH",
        "MaxHR": 150,
        "ExerciseAngina": "Y",
        "Oldpeak": 2.3,
        "ST_Slope": "Flat"
    },
    {
        "Age": 45,
        "Sex": "F",
        "ChestPainType": "NAP",
        "RestingBP": 130,
        "Cholesterol": 250,
        "FastingBS": 0,
        "RestingECG": "Normal",
        "MaxHR": 170,
        "ExerciseAngina": "N",
        "Oldpeak": 0.0,
        "ST_Slope": "Up"
    },
    {
        "Age": 58,
        "Sex": "M",
        "ChestPainType": "TA",
        "RestingBP": 138,
        "Cholesterol": 230,
        "FastingBS": 1,
        "RestingECG": "ST",
        "MaxHR": 165,
        "ExerciseAngina": "Y",
        "Oldpeak": 1.2,
        "ST_Slope": "Down"
    }
])

# Предсказания
predictions = best_model.predict(new_patients)

# Вывод
for i, pred in enumerate(predictions):
    status = "Болезнь сердца" if pred == 1 else "Здоров"
    print(f"Пациент {i+1}: {status}")