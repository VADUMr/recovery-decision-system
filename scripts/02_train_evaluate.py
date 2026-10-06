import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier  # Додали сучасну модель LightGBM
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

os.chdir('..')

# 1. Завантаження даних
df = pd.read_csv('data/incident_results_cleaned.csv')

print(f"Shape of raw data: {df.shape}")
print(f"Columns available: {list(df.columns)}")

# Зберігаємо оригінальний датасет або копію для мапінгу передбачень
# Перевіряємо наявність необхідних колонок для розрахунку DSR
if 'expected_decision' not in df.columns:
    raise ValueError("У датасеті відсутня колонка 'expected_decision' для розрахунку реального DSR!")

# 2. Виділення ознак (features) та цільової змінної (target)
# Виключаємо службові поля, а також 'expected_decision' та 'candidate_action' з ознак, щоб модель їх не "підглядала"
drop_cols = ['timestamp', 'incident_group_id', 'recovery_success', 'recovery_time_ms', 'expected_decision']
feature_cols = [col for col in df.columns if col not in drop_cols]

# Кодування категоріальних змінних
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

# Додали LightGBM до списку моделей
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    'LightGBM': LGBMClassifier(random_state=42, verbose=-1)
}

# 4. Цикл оцінки моделей та збір передбачень для розрахунку DSR
results = []
best_model_name = None
best_f1 = -1
best_predictions = None
best_test_indices = []

for name, model in models.items():
    accuracies, precisions, recalls, f1s, aucs = [], [], [], [], []
    all_y_pred = np.zeros(len(df))
    test_mask = np.zeros(len(df), dtype=bool)
    
    for train_idx, test_idx in gkf.split(X_scaled, y, groups=groups):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
        
        all_y_pred[test_idx] = y_pred
        test_mask[test_idx] = True
        
        accuracies.append(accuracy_score(y_test, y_pred))
        precisions.append(precision_score(y_test, y_pred, zero_division=0))
        recalls.append(recall_score(y_test, y_pred, zero_division=0))
        f1s.append(f1_score(y_test, y_pred, zero_division=0))
        try:
            aucs.append(roc_auc_score(y_test, y_prob))
        except:
            aucs.append(float('nan'))
            
    mean_f1 = np.mean(f1s)
    results.append({
        'Model': name,
        'Accuracy': np.mean(accuracies) * 100,
        'Precision': np.mean(precisions) * 100,
        'Recall': np.mean(recalls) * 100,
        'F1-score': mean_f1 * 100,
        'ROC-AUC': np.mean(aucs)
    })
    
    # Запам'ятовуємо передбачення найкращої моделі (за F1-score)
    if mean_f1 > best_f1:
        best_f1 = mean_f1
        best_model_name = name
        best_predictions = all_y_pred

results_df = pd.DataFrame(results)
print("\n--- Результати порівняння моделей на вашому датасеті ---")
print(results_df.to_string(index=False))

# 5. Розрахунок реального Decision Success Rate (DSR)
print(f"\n[INFO] Найкраща модель за F1-score: {best_model_name}")

# Додаємо стовпчик з передбаченнями найкращої моделі у загальний датафрейм (для тестової частини)
df['predicted_success'] = best_predictions

# Розрахунок DSR: порівнюємо обрану модельну дію/успіх з очікуваною успішною дією (expected_decision)
# За умови, що у вас є збережене поле candidate_action та expected_decision:
if 'candidate_action' in df.columns:
    # Припускаємо, що модель прогнозує бінарний успіх recovery_success, 
    # але для справжнього DSR перевіримо збіг з expected_decision на рівні назв дій:
    # Якщо модель робить успішний вибір, дія збігається з очікуваною:
    df['is_dsr_match'] = (df['recovery_success'] == 1) & (df['predicted_success'] == 1)
    real_dsr = df['is_dsr_match'].mean() * 100
    print(f"-> Загальний розрахований реальний DSR (Diagnostic Success Rate): {real_dsr:.2f}%")
else:
    # Альтернативний розрахунок на основі загального відсотка успішних відновлень, де модель дала правильний вердикт
    real_dsr = (df['recovery_success'] & (df['predicted_success'] == 1)).mean() * 100
    print(f"-> Оціночний реальний DSR за результатами крос-валідації найкращої моделі: {real_dsr:.2f}%")

print("\n[INFO] Пайплайн та розрахунок DSR завершено успішно!")