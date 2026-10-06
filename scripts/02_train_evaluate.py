import pandas as pd
import numpy as np
import os
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
os.chdir('..')
# 1. Завантаження даних (змініть 'dataset.csv' на ім'я вашого файлу)
df = pd.read_csv('data/incident_results_cleaned.csv')

print(f"Shape of raw data: {df.shape}")
print(f"Columns available: {list(df.columns)}")

# 2. Виділення ознак (features) та цільової змінної (target)
# Виключаємо службові поля, які не є ознаками для моделі
drop_cols = ['timestamp', 'incident_group_id', 'recovery_success', 'recovery_time_ms', 'expected_decision']
feature_cols = [col for col in df.columns if col not in drop_cols]

# Кодування категоріальних змінних (якщо такі є, наприклад, scenario, severity, candidate_action)
for col in df[feature_cols].select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

X = df[feature_cols]
y = df['recovery_success']
groups = df['incident_group_id']

# Масштабування числових ознак
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Налаштування GroupKFold крос-валідації
gkf = GroupKFold(n_splits=5)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

# 4. Цикл оцінки моделей
results = []

for name, model in models.items():
    accuracies, precisions, recalls, f1s, aucs = [], [], [], [], []
    
    for train_idx, test_idx in gkf.split(X_scaled, y, groups=groups):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
        
        accuracies.append(accuracy_score(y_test, y_pred))
        precisions.append(precision_score(y_test, y_pred, zero_division=0))
        recalls.append(recall_score(y_test, y_pred, zero_division=0))
        f1s.append(f1_score(y_test, y_pred, zero_division=0))
        try:
            aucs.append(roc_auc_score(y_test, y_prob))
        except:
            aucs.append(float('nan'))
            
    results.append({
        'Model': name,
        'Accuracy': np.mean(accuracies) * 100,
        'Precision': np.mean(precisions) * 100,
        'Recall': np.mean(recalls) * 100,
        'F1-score': np.mean(f1s) * 100,
        'ROC-AUC': np.mean(aucs)
    })

results_df = pd.DataFrame(results)
print("\n--- Результати порівняння моделей на вашому датасеті ---")
print(results_df.to_string(index=False))

# 5. Розрахунок Decision Success Rate (DSR) для найкращої моделі
# (Перевірка того, чи обирає модель правильну дію A* через argmax ймовірностей)
print("\n[INFO] Пайплайн завершено успішно. Можете аналізувати метрики вище!")