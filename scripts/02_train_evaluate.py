"""
02_train_evaluate.py — GroupKFold training + decision-level metrics.

Додано:
  - success rate reference-policy дії (Policy SR)
  - Regret = Policy SR − DSR_actual  (від'ємний = модель краща за policy)
  - Brier score (калібрування ймовірностей)
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupKFold, learning_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, balanced_accuracy_score,
    brier_score_loss,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")
plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 160,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.facecolor": "white",
})

os.chdir("..")
os.makedirs("data/plots", exist_ok=True)

# ============================================================
# 1. Завантаження даних
# ============================================================
df = pd.read_csv("data/incident_results_cleaned.csv")
print(f"Shape of raw data: {df.shape}")
print(f"Columns available: {list(df.columns)}")

if "expected_decision" not in df.columns:
    raise ValueError("Відсутня колонка 'expected_decision'!")

# ============================================================
# 2. Фічі (без leakage)
# ============================================================
leakage_cols = [
    "timestamp", "incident_group_id",
    "recovery_success", "recovery_time_ms", "expected_decision",
    "post_cpu", "post_memory", "post_latency_ms",
    "post_error_rate", "post_dependency_available", "post_availability",
]
drop_extra = ["replicate"]

feature_cols = [c for c in df.columns if c not in leakage_cols + drop_extra]
print(f"\n[INFO] Використовуємо фічі: {feature_cols}")

label_encoders = {}
df_enc = df.copy()
for col in df_enc[feature_cols].select_dtypes(include=["object", "string"]).columns:
    le = LabelEncoder()
    df_enc[col] = le.fit_transform(df_enc[col].astype(str))
    label_encoders[col] = le

X = df_enc[feature_cols]
y = df_enc["recovery_success"]
groups = df_enc["incident_group_id"]

# ============================================================
# 3. Моделі
# ============================================================
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, max_depth=10, min_samples_leaf=2, random_state=42, n_jobs=-1
    ),
    "XGBoost": XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        eval_metric="logloss", random_state=42, verbosity=0
    ),
    "LightGBM": LGBMClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        random_state=42, verbose=-1
    ),
}

# ============================================================
# 4. GroupKFold оцінка + OOF predictions
# ============================================================
gkf = GroupKFold(n_splits=5)
results = []
oof_store = {}
best_name, best_f1 = None, -1
best_oof_proba = None

for name, model in models.items():
    accs, precs, recs, f1s, bal_accs, aucs, briers = [], [], [], [], [], [], []
    oof_pred = np.zeros(len(df_enc), dtype=int)
    oof_proba = np.zeros(len(df_enc))

    for train_idx, test_idx in gkf.split(X, y, groups=groups):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        proba = model.predict_proba(X_te)[:, 1]

        oof_pred[test_idx] = pred
        oof_proba[test_idx] = proba

        accs.append(accuracy_score(y_te, pred))
        precs.append(precision_score(y_te, pred, zero_division=0))
        recs.append(recall_score(y_te, pred, zero_division=0))
        f1s.append(f1_score(y_te, pred, zero_division=0))
        bal_accs.append(balanced_accuracy_score(y_te, pred))
        try:
            aucs.append(roc_auc_score(y_te, proba))
        except ValueError:
            aucs.append(np.nan)
        briers.append(brier_score_loss(y_te, proba))

    mean_f1 = float(np.mean(f1s))
    results.append({
        "Model": name,
        "Accuracy, %": round(np.mean(accs) * 100, 2),
        "Precision, %": round(np.mean(precs) * 100, 2),
        "Recall, %": round(np.mean(recs) * 100, 2),
        "F1-score, %": round(mean_f1 * 100, 2),
        "Balanced Accuracy, %": round(np.mean(bal_accs) * 100, 2),
        "ROC-AUC": round(np.nanmean(aucs), 3),
        "Brier": round(np.mean(briers), 4),
    })
    oof_store[name] = (y.values, oof_pred, oof_proba)

    if mean_f1 > best_f1:
        best_f1 = mean_f1
        best_name = name
        best_oof_proba = oof_proba.copy()

results_df = pd.DataFrame(results)
print("\n--- Таблиця 1. Порівняння моделей прогнозування успішності відновлення ---")
print(results_df.to_string(index=False))
print(f"\n[INFO] Найкраща модель за F1-score: {best_name}")

# ============================================================
# 5. Decision-level метрики: DSR + Policy SR + Regret
# ============================================================
df_enc["pred_proba"] = best_oof_proba

# --- 5a. Дія, обрана моделлю (argmax P̂) ---
best_idx = df_enc.groupby("incident_group_id")["pred_proba"].idxmax()
model_choices = df.loc[best_idx].copy()
model_choices = model_choices[
    ["incident_group_id", "candidate_action", "expected_decision",
     "recovery_success", "scenario"]
]
model_choices["pred_proba"] = best_oof_proba[best_idx.values]

dsr_policy_match = (
    model_choices["candidate_action"].values == model_choices["expected_decision"].values
).mean() * 100
dsr_actual = model_choices["recovery_success"].mean() * 100
n_groups = len(model_choices)
mean_p = model_choices["pred_proba"].mean()

# --- 5b. Success rate саме reference-policy дії ---
# Для кожної групи знаходимо рядок, де candidate_action == expected_decision
policy_rows = df[df["candidate_action"] == df["expected_decision"]].copy()
# якщо в групі кілька збігів (не повинно), беремо перший
policy_rows = policy_rows.groupby("incident_group_id", as_index=False).first()
policy_sr = policy_rows["recovery_success"].mean() * 100
n_policy = len(policy_rows)

# --- 5c. Regret (в процентних пунктах) ---
# Regret = Policy_SR − DSR_actual
#   > 0  → policy краща за модель
#   < 0  → модель краща за policy (типовий бажаний випадок)
regret_pp = policy_sr - dsr_actual

print(f"\n--- Decision-level метрики ({best_name}) ---")
print(f"DSR (збіг з expected_decision):          {dsr_policy_match:.2f}%")
print(f"DSR (реальний успіх вибраної дії):       {dsr_actual:.2f}%")
print(f"Policy SR (успіх дії reference policy):  {policy_sr:.2f}%  (n={n_policy})")
print(f"Regret (Policy SR − DSR_actual):         {regret_pp:+.2f} п.п.")
print(f"Кількість інцидентів (груп):             {n_groups}")
print(f"Середня P̂(success) вибраної дії:         {mean_p:.3f}")

brier_best = results_df.loc[results_df["Model"] == best_name, "Brier"].values[0]
print(f"Brier score ({best_name}):               {brier_best:.4f}  (0 = ідеал)")

if mean_p > 0.95 and brier_best > 0.08:
    print("[WARN] Висока середня P̂ при непоганому Brier — можлива надмірна "
          "самовпевненість; розгляньте CalibratedClassifierCV.")
elif brier_best <= 0.08:
    print("[INFO] Brier score низький — ймовірності загалом добре відкалібровані.")

# ============================================================
# 6. Візуалізації
# ============================================================
palette = sns.color_palette("mako", n_colors=len(models))
model_names = list(models.keys())

# ----- 6.1 Фінальна таблиця -----
fig, ax = plt.subplots(figsize=(12, 2.4))
ax.axis("off")
tbl = ax.table(
    cellText=results_df.round(4).values,
    colLabels=results_df.columns,
    loc="center",
    cellLoc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(8.5)
tbl.scale(1.12, 1.55)
for (row, col), cell in tbl.get_celld().items():
    if row == 0:
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", weight="bold")
    elif results_df.iloc[row - 1]["Model"] == best_name:
        cell.set_facecolor("#d5f5e3")
ax.set_title("Таблиця 1 – Порівняння моделей прогнозування успішності відновлення",
             pad=12, fontweight="bold")
plt.tight_layout()
plt.savefig("data/plots/table1_model_comparison.png", bbox_inches="tight")
plt.close()

# ----- 6.2 Barplot метрик -----
metrics_long = results_df.melt(
    id_vars="Model",
    value_vars=["Accuracy, %", "Precision, %", "Recall, %", "F1-score, %", "Balanced Accuracy, %"],
    var_name="Metric", value_name="Value"
)
plt.figure(figsize=(11, 5.5))
sns.barplot(data=metrics_long, x="Metric", y="Value", hue="Model", palette="mako")
plt.ylim(0, 100)
plt.title("Порівняння метрик моделей (GroupKFold)")
plt.ylabel("%")
plt.xlabel("")
plt.legend(title="Модель", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig("data/plots/metrics_comparison.png", bbox_inches="tight")
plt.close()

# ----- 6.3 ROC -----
plt.figure(figsize=(7.5, 6.5))
for i, name in enumerate(model_names):
    y_true, _, y_proba = oof_store[name]
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    plt.plot(fpr, tpr, lw=2.2, color=palette[i],
             label=f"{name} (AUC = {auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6)
plt.xlim([0, 1])
plt.ylim([0, 1.02])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Рисунок 2 – ROC-криві моделей (GroupKFold OOF)")
plt.legend(loc="lower right")
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig("data/plots/roc_curves.png", bbox_inches="tight")
plt.close()

# ----- 6.4 Confusion matrix -----
y_true_best, y_pred_best, _ = oof_store[best_name]
cm = confusion_matrix(y_true_best, y_pred_best)
plt.figure(figsize=(5.8, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["Fail (0)", "Success (1)"],
            yticklabels=["Fail (0)", "Success (1)"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Рисунок 3 – Матриця помилок ({best_name})")
plt.tight_layout()
plt.savefig("data/plots/confusion_matrix_best_model.png", bbox_inches="tight")
plt.close()

# ----- 6.5 Feature importance -----
tree_models = {
    "Random Forest": models["Random Forest"],
    "XGBoost": models["XGBoost"],
    "LightGBM": models["LightGBM"],
}
fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
for ax, (name, model) in zip(axes, tree_models.items()):
    model.fit(X, y)
    if hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_, index=feature_cols)
        imp = imp.sort_values(ascending=True)
        imp.plot(kind="barh", ax=ax, color=sns.color_palette("mako", n_colors=len(imp)))
        ax.set_title(name)
        ax.set_xlabel("Importance")
plt.suptitle("Рисунок 4 – Feature Importance Analysis", fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("data/plots/feature_importance.png", bbox_inches="tight")
plt.close()

# ----- 6.6 Learning curve -----
best_model = models[best_name]
train_sizes, train_scores, val_scores = learning_curve(
    best_model, X, y,
    groups=groups,
    cv=GroupKFold(n_splits=5),
    train_sizes=np.linspace(0.2, 1.0, 6),
    scoring="f1",
    n_jobs=-1,
    random_state=42,
)
train_mean, train_std = train_scores.mean(axis=1), train_scores.std(axis=1)
val_mean, val_std = val_scores.mean(axis=1), val_scores.std(axis=1)

plt.figure(figsize=(8, 5.5))
plt.plot(train_sizes, train_mean, "o-", color="#1abc9c", label="Train F1")
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                 alpha=0.2, color="#1abc9c")
plt.plot(train_sizes, val_mean, "o-", color="#e74c3c", label="Validation F1 (GroupKFold)")
plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                 alpha=0.2, color="#e74c3c")
plt.xlabel("Кількість навчальних прикладів")
plt.ylabel("F1-score")
plt.title(f"Learning Curve – {best_name}")
plt.legend(loc="lower right")
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig("data/plots/learning_curve.png", bbox_inches="tight")
plt.close()

# ----- 6.7 Гістограми pre-метрик -----
pre_cols = [c for c in feature_cols if c.startswith("pre_")]
n = len(pre_cols)
ncols = 3
nrows = int(np.ceil(max(n, 1) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.2 * nrows))
axes = np.array(axes).flatten()
for i, col in enumerate(pre_cols):
    sns.histplot(data=df, x=col, hue="recovery_success", bins=25,
                 palette={0: "#e74c3c", 1: "#27ae60"}, ax=axes[i],
                 alpha=0.65, element="step", common_norm=False)
    axes[i].set_title(col)
    axes[i].set_xlabel("")
for j in range(len(pre_cols), len(axes)):
    axes[j].set_visible(False)
plt.suptitle("Розподіл pre-метрик за результатом відновлення", fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("data/plots/pre_metrics_histograms.png", bbox_inches="tight")
plt.close()

# ----- 6.8 Recovery time by scenario -----
plt.figure(figsize=(9, 5))
sns.boxplot(data=df, x="scenario", y="recovery_time_ms",
            hue="scenario", palette="Set2", legend=False)
plt.title("Час відновлення за типом відмови")
plt.ylabel("recovery_time_ms")
plt.xlabel("scenario")
plt.tight_layout()
plt.savefig("data/plots/recovery_time_by_scenario.png", bbox_inches="tight")
plt.close()

# ----- 6.9 DSR / Policy SR / Regret summary card -----
fig, ax = plt.subplots(figsize=(8, 4.2))
ax.axis("off")
lines = [
    f"Best model: {best_name}",
    f"",
    f"DSR (match expected_decision):  {dsr_policy_match:.2f}%",
    f"DSR (actual success of chosen): {dsr_actual:.2f}%",
    f"Policy SR (success of π_ref):   {policy_sr:.2f}%",
    f"Regret (Policy SR − DSR_act):   {regret_pp:+.2f} pp",
    f"",
    f"Incident groups: {n_groups}",
    f"Mean P̂(chosen action): {mean_p:.3f}",
    f"Brier score: {brier_best:.4f}",
]
ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes,
        fontsize=13, verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="#f8f9fa", edgecolor="#2c3e50", lw=1.5))
ax.set_title("Decision-level summary", fontweight="bold", pad=8)
plt.tight_layout()
plt.savefig("data/plots/dsr_summary.png", bbox_inches="tight")
plt.close()

# ----- 6.10 Reliability diagram (calibration check) -----
from sklearn.calibration import calibration_curve

plt.figure(figsize=(6.5, 6))
for i, name in enumerate(model_names):
    y_true, _, y_proba = oof_store[name]
    frac_pos, mean_pred = calibration_curve(y_true, y_proba, n_bins=8, strategy="quantile")
    plt.plot(mean_pred, frac_pos, "o-", color=palette[i], label=name)
plt.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6, label="Ideal")
plt.xlabel("Mean predicted probability")
plt.ylabel("Fraction of positives")
plt.title("Reliability diagram (calibration)")
plt.legend(loc="lower right")
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig("data/plots/reliability_diagram.png", bbox_inches="tight")
plt.close()

print(f"\n[INFO] Усі графіки збережено у: {os.path.abspath('data/plots')}")
for name in [
    "table1_model_comparison.png", "metrics_comparison.png", "roc_curves.png",
    "confusion_matrix_best_model.png", "feature_importance.png", "learning_curve.png",
    "pre_metrics_histograms.png", "recovery_time_by_scenario.png",
    "dsr_summary.png", "reliability_diagram.png",
]:
    print(f"  • {name}")
print("\n[INFO] Пайплайн завершено успішно!")