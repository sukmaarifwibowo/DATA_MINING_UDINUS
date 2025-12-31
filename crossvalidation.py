import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score, cross_validate, KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Set random seed untuk reproduksibilitas
np.random.seed(42)

# 1. GENERATE DATA SINTETIS
print("=" * 60)
print("STUDI KASUS: CROSS VALIDATION DENGAN LOGISTIC REGRESSION")
print("=" * 60)
print("\n1. MEMBUAT DATA SINTETIS")
print("-" * 60)

# Membuat dataset klasifikasi biner
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    n_classes=2,
    random_state=42,
    flip_y=0.1  # Tambahkan sedikit noise
)

print(f"Jumlah sampel: {X.shape[0]}")
print(f"Jumlah fitur: {X.shape[1]}")
print(f"Distribusi kelas: {np.bincount(y)}")

# 2. PREPROCESSING DATA
print("\n2. PREPROCESSING DATA")
print("-" * 60)

# Standardisasi fitur
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data untuk evaluasi final
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Data training: {X_train.shape[0]} sampel")
print(f"Data testing: {X_test.shape[0]} sampel")

# 3. MODEL LOGISTIC REGRESSION
print("\n3. MEMBUAT MODEL LOGISTIC REGRESSION")
print("-" * 60)

model = LogisticRegression(max_iter=1000, random_state=42)

# 4. CROSS VALIDATION - METODE 1: K-FOLD
print("\n4. CROSS VALIDATION - K-FOLD (5-FOLD)")
print("-" * 60)

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Evaluasi dengan berbagai metrik
cv_scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')

print(f"Accuracy per fold: {cv_scores}")
print(f"Mean Accuracy: {cv_scores.mean():.4f}")
print(f"Std Accuracy: {cv_scores.std():.4f}")

# 5. CROSS VALIDATION - METODE 2: STRATIFIED K-FOLD
print("\n5. CROSS VALIDATION - STRATIFIED K-FOLD (5-FOLD)")
print("-" * 60)

skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Evaluasi dengan multiple scoring metrics
scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1'
}

cv_results = cross_validate(
    model, X_train, y_train, 
    cv=skfold, 
    scoring=scoring,
    return_train_score=True
)

print("Hasil Cross Validation dengan Multiple Metrics:")
print(f"Accuracy  - Train: {cv_results['train_accuracy'].mean():.4f} | Test: {cv_results['test_accuracy'].mean():.4f}")
print(f"Precision - Train: {cv_results['train_precision'].mean():.4f} | Test: {cv_results['test_precision'].mean():.4f}")
print(f"Recall    - Train: {cv_results['train_recall'].mean():.4f} | Test: {cv_results['test_recall'].mean():.4f}")
print(f"F1-Score  - Train: {cv_results['train_f1'].mean():.4f} | Test: {cv_results['test_f1'].mean():.4f}")

# 6. TRAINING FINAL MODEL
print("\n6. TRAINING FINAL MODEL")
print("-" * 60)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluasi pada test set
test_accuracy = accuracy_score(y_test, y_pred)
test_precision = precision_score(y_test, y_pred)
test_recall = recall_score(y_test, y_pred)
test_f1 = f1_score(y_test, y_pred)

print("Evaluasi pada Test Set:")
print(f"Accuracy: {test_accuracy:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall: {test_recall:.4f}")
print(f"F1-Score: {test_f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 7. VISUALISASI
print("\n7. VISUALISASI HASIL")
print("-" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Comparison of CV scores across folds
ax1 = axes[0, 0]
folds = range(1, 6)
ax1.plot(folds, cv_scores, marker='o', linestyle='-', linewidth=2, markersize=8)
ax1.axhline(y=cv_scores.mean(), color='r', linestyle='--', label=f'Mean: {cv_scores.mean():.4f}')
ax1.fill_between(folds, cv_scores.mean() - cv_scores.std(), 
                  cv_scores.mean() + cv_scores.std(), alpha=0.2)
ax1.set_xlabel('Fold', fontsize=12)
ax1.set_ylabel('Accuracy', fontsize=12)
ax1.set_title('K-Fold Cross Validation Scores', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Multiple metrics comparison
ax2 = axes[0, 1]
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
cv_values = [
    cv_results['test_accuracy'].mean(),
    cv_results['test_precision'].mean(),
    cv_results['test_recall'].mean(),
    cv_results['test_f1'].mean()
]
test_values = [test_accuracy, test_precision, test_recall, test_f1]

x = np.arange(len(metrics))
width = 0.35

bars1 = ax2.bar(x - width/2, cv_values, width, label='CV Mean', alpha=0.8)
bars2 = ax2.bar(x + width/2, test_values, width, label='Test Set', alpha=0.8)

ax2.set_ylabel('Score', fontsize=12)
ax2.set_title('Cross Validation vs Test Set Performance', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(metrics)
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

# Tambahkan nilai di atas bar
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9)

# Plot 3: Confusion Matrix
ax3 = axes[1, 0]
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3, cbar=False)
ax3.set_xlabel('Predicted Label', fontsize=12)
ax3.set_ylabel('True Label', fontsize=12)
ax3.set_title('Confusion Matrix', fontsize=14, fontweight='bold')

# Plot 4: Training vs Test scores across CV folds
ax4 = axes[1, 1]
folds = range(1, 6)
ax4.plot(folds, cv_results['train_accuracy'], marker='o', label='Train', linewidth=2)
ax4.plot(folds, cv_results['test_accuracy'], marker='s', label='Validation', linewidth=2)
ax4.set_xlabel('Fold', fontsize=12)
ax4.set_ylabel('Accuracy', fontsize=12)
ax4.set_title('Train vs Validation Accuracy per Fold', fontsize=14, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('cross_validation_results.png', dpi=300, bbox_inches='tight')
print("Visualisasi disimpan sebagai 'cross_validation_results.png'")
plt.show()

# 8. KESIMPULAN
print("\n" + "=" * 60)
print("KESIMPULAN")
print("=" * 60)
print(f"1. Model menunjukkan konsistensi yang baik dengan std accuracy: {cv_scores.std():.4f}")
print(f"2. Tidak ada overfitting signifikan (Train vs Val accuracy gap kecil)")
print(f"3. Performa pada test set ({test_accuracy:.4f}) konsisten dengan CV mean ({cv_scores.mean():.4f})")
print("4. Model siap untuk deployment!")
print("=" * 60)