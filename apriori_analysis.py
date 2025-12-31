import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

print("=" * 70)
print("ANALISIS ASOSIASI MENGGUNAKAN ALGORITMA APRIORI")
print("=" * 70)

# 1. DATA TRANSAKSI
print("\n1. DATA TRANSAKSI")
print("-" * 70)

# Dataset dari soal
transactions = [
    ['Roti', 'Selai', 'Mentega'],           # T1
    ['Roti', 'Mentega'],                     # T2
    ['Roti', 'Susu', 'Mentega'],            # T3
    ['Cokelat', 'Roti', 'Susu', 'Mentega'], # T4
    ['Cokelat', 'Susu']                      # T5
]

# Tampilkan transaksi
for i, trans in enumerate(transactions, 1):
    print(f"T{i}: {', '.join(trans)}")

print(f"\nTotal transaksi: {len(transactions)}")

# 2. KONVERSI DATA KE FORMAT ONE-HOT ENCODING
print("\n2. KONVERSI DATA (ONE-HOT ENCODING)")
print("-" * 70)

te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)
df.index = [f'T{i}' for i in range(1, len(transactions)+1)]

print("\nDataFrame One-Hot Encoded:")
print(df)

# 3. MENJALANKAN ALGORITMA APRIORI
print("\n3. FREQUENT ITEMSETS (APRIORI)")
print("-" * 70)

# Parameter: min_support = 30% = 0.3
min_support = 0.3
print(f"Minimum Support: {min_support*100}%")

# Generate frequent itemsets
frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
frequent_itemsets = frequent_itemsets.sort_values('support', ascending=False)

print(f"\nJumlah Frequent Itemsets: {len(frequent_itemsets)}")
print("\nFrequent Itemsets:")
print(frequent_itemsets.to_string(index=False))

# 4. GENERATE ASSOCIATION RULES
print("\n4. ASSOCIATION RULES")
print("-" * 70)

# Parameter: min_confidence = 60% = 0.6
min_confidence = 0.6
print(f"Minimum Confidence: {min_confidence*100}%")

# Generate rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

if len(rules) > 0:
    # Format untuk display yang lebih baik
    rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
    rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
    
    # Pilih kolom yang relevan
    display_rules = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
    display_rules = display_rules.sort_values('confidence', ascending=False)
    
    print(f"\nJumlah Association Rules: {len(rules)}")
    print("\nAssociation Rules yang ditemukan:")
    print(display_rules.to_string(index=False))
    
    # 5. INTERPRETASI RULES
    print("\n5. INTERPRETASI TOP RULES")
    print("-" * 70)
    
    for idx, row in display_rules.head(5).iterrows():
        print(f"\nRule: {row['antecedents']} → {row['consequents']}")
        print(f"  • Support: {row['support']:.2%} (muncul pada {row['support']*len(transactions):.0f} dari {len(transactions)} transaksi)")
        print(f"  • Confidence: {row['confidence']:.2%} (jika beli {row['antecedents']}, {row['confidence']:.2%} kemungkinan beli {row['consequents']})")
        print(f"  • Lift: {row['lift']:.2f} ({'positif' if row['lift'] > 1 else 'negatif'} correlation)")
else:
    print("\nTidak ada rules yang memenuhi threshold confidence!")

# 6. VISUALISASI
print("\n6. VISUALISASI")
print("-" * 70)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Support Distribution of Frequent Itemsets
ax1 = axes[0, 0]
itemset_by_length = frequent_itemsets.groupby('length')['support'].mean()
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(itemset_by_length)))
bars = ax1.bar(itemset_by_length.index, itemset_by_length.values, color=colors, alpha=0.8, edgecolor='black')
ax1.set_xlabel('Jumlah Item dalam Itemset', fontsize=12, fontweight='bold')
ax1.set_ylabel('Average Support', fontsize=12, fontweight='bold')
ax1.set_title('Support Distribution by Itemset Length', fontsize=14, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

# Tambahkan nilai di atas bar
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2%}', ha='center', va='bottom', fontsize=10)

# Plot 2: Top Frequent Itemsets
ax2 = axes[0, 1]
top_itemsets = frequent_itemsets.nlargest(10, 'support')
itemset_names = [', '.join(list(x)) for x in top_itemsets['itemsets']]
y_pos = np.arange(len(itemset_names))

bars = ax2.barh(y_pos, top_itemsets['support'].values, color=plt.cm.Greens(np.linspace(0.4, 0.9, len(top_itemsets))))
ax2.set_yticks(y_pos)
ax2.set_yticklabels(itemset_names)
ax2.invert_yaxis()
ax2.set_xlabel('Support', fontsize=12, fontweight='bold')
ax2.set_title('Top 10 Frequent Itemsets', fontsize=14, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)

# Tambahkan nilai
for i, bar in enumerate(bars):
    width = bar.get_width()
    ax2.text(width, bar.get_y() + bar.get_height()/2.,
            f'{width:.2%}', ha='left', va='center', fontsize=9, 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

# Plot 3: Support vs Confidence (jika ada rules)
ax3 = axes[1, 0]
if len(rules) > 0:
    scatter = ax3.scatter(rules['support'], rules['confidence'], 
                         c=rules['lift'], s=rules['lift']*100, 
                         alpha=0.6, cmap='coolwarm', edgecolors='black', linewidth=1)
    ax3.set_xlabel('Support', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Confidence', fontsize=12, fontweight='bold')
    ax3.set_title('Support vs Confidence (size & color = Lift)', fontsize=14, fontweight='bold')
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label('Lift', fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Tambahkan garis threshold
    ax3.axhline(y=min_confidence, color='r', linestyle='--', linewidth=2, alpha=0.5, label=f'Min Confidence ({min_confidence:.0%})')
    ax3.axvline(x=min_support, color='b', linestyle='--', linewidth=2, alpha=0.5, label=f'Min Support ({min_support:.0%})')
    ax3.legend()
else:
    ax3.text(0.5, 0.5, 'Tidak ada rules yang memenuhi threshold', 
            ha='center', va='center', fontsize=12, transform=ax3.transAxes)
    ax3.set_title('Support vs Confidence', fontsize=14, fontweight='bold')

# Plot 4: Heatmap Item Co-occurrence
ax4 = axes[1, 1]
# Buat matriks co-occurrence
items = list(df.columns)
cooccurrence = np.zeros((len(items), len(items)))

for i, item1 in enumerate(items):
    for j, item2 in enumerate(items):
        cooccurrence[i][j] = ((df[item1] == True) & (df[item2] == True)).sum()

sns.heatmap(cooccurrence, annot=True, fmt='.0f', cmap='YlOrRd', 
            xticklabels=items, yticklabels=items, ax=ax4, cbar_kws={'label': 'Co-occurrence Count'})
ax4.set_title('Item Co-occurrence Matrix', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('apriori_analysis.png', dpi=300, bbox_inches='tight')
print("\nVisualisasi disimpan sebagai 'apriori_analysis.png'")
plt.show()

# 7. KESIMPULAN
print("\n" + "=" * 70)
print("KESIMPULAN")
print("=" * 70)
print(f"1. Total Frequent Itemsets: {len(frequent_itemsets)}")
print(f"2. Item paling sering muncul: {frequent_itemsets.iloc[0]['itemsets']}")
print(f"3. Support tertinggi: {frequent_itemsets.iloc[0]['support']:.2%}")

if len(rules) > 0:
    best_rule = rules.nlargest(1, 'confidence').iloc[0]
    print(f"4. Best Association Rule:")
    print(f"   {best_rule['antecedents']} → {best_rule['consequents']}")
    print(f"   Confidence: {best_rule['confidence']:.2%}, Lift: {best_rule['lift']:.2f}")
    print(f"5. Total Rules yang valid: {len(rules)}")
else:
    print("4. Tidak ada association rules yang memenuhi threshold")

print("\n💡 Rekomendasi: Letakkan item yang sering dibeli bersama berdekatan!")
print("=" * 70)