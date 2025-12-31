import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (18, 12)

print("=" * 90)
print("ANALISIS CLUSTERING DATA COVID-19 PER PROVINSI DI INDONESIA")
print("=" * 90)

# 1. LOAD DATA
print("\n1. LOADING DATA")
print("-" * 90)

# Baca file CSV
df = pd.read_csv('Statistik_Harian_per_Provinsi_COVID19_Indonesia_Rev.csv')

print("Data berhasil dimuat!")
print(f"\nShape: {df.shape}")
print(f"\nKolom yang tersedia:")
print(df.columns.tolist())
print(f"\nPreview Data:")
print(df.head())

# 2. EKSPLORASI DATA
print("\n2. EKSPLORASI DATA")
print("-" * 90)

print("\nInfo Dataset:")
print(df.info())

print("\nStatistik Deskriptif:")
print(df.describe())

print(f"\nJumlah Provinsi Unik: {df['Provinsi'].nunique()}")
print(f"Provinsi: {df['Provinsi'].unique()}")

print(f"\nRentang Tanggal: {df['Tanggal'].min()} s/d {df['Tanggal'].max()}")

# 3. DATA PREPROCESSING DAN AGREGASI
print("\n3. DATA PREPROCESSING DAN AGREGASI")
print("-" * 90)

# Agregasi data per provinsi (total akumulatif dan rata-rata harian)
agg_functions = {
    'Kasus_Terkonfirmasi_Akumulatif': 'max',  # Total kasus
    'Penambahan_Harian_Kasus_Terkonf': 'mean',  # Rata-rata penambahan harian
    'Kasus_Sembuh_Akumulatif': 'max',  # Total sembuh
    'Penambahan_Harian_Kasus_Sembuh': 'mean',  # Rata-rata sembuh harian
    'Kasus_Meninggal_Akumulatif': 'max',  # Total meninggal
    'Penambahan_Harian_Kasus_Meningg': 'mean',  # Rata-rata meninggal harian
    'Kasus_Aktif_Akumulatif': 'mean'  # Rata-rata kasus aktif
}

df_provinsi = df.groupby('Provinsi').agg(agg_functions).reset_index()

# Tambahkan fitur tambahan
df_provinsi['Tingkat_Kesembuhan'] = (df_provinsi['Kasus_Sembuh_Akumulatif'] / 
                                      df_provinsi['Kasus_Terkonfirmasi_Akumulatif'] * 100)
df_provinsi['Tingkat_Kematian'] = (df_provinsi['Kasus_Meninggal_Akumulatif'] / 
                                    df_provinsi['Kasus_Terkonfirmasi_Akumulatif'] * 100)

print(f"\nData agregat per provinsi: {df_provinsi.shape}")
print("\nSample data agregat:")
print(df_provinsi.head())

# Handle missing values dan inf
df_provinsi = df_provinsi.replace([np.inf, -np.inf], np.nan)
df_provinsi = df_provinsi.fillna(0)

# Pilih fitur untuk clustering
feature_cols = [
    'Kasus_Terkonfirmasi_Akumulatif',
    'Penambahan_Harian_Kasus_Terkonf',
    'Kasus_Sembuh_Akumulatif',
    'Kasus_Meninggal_Akumulatif',
    'Tingkat_Kesembuhan',
    'Tingkat_Kematian'
]

X = df_provinsi[feature_cols].values
provinsi = df_provinsi['Provinsi'].values

print(f"\nFitur yang digunakan untuk clustering:")
for col in feature_cols:
    print(f"  • {col}")

# Standardisasi data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"\nData telah distandardisasi: shape = {X_scaled.shape}")

# 4. MENENTUKAN JUMLAH CLUSTER OPTIMAL
print("\n4. MENENTUKAN JUMLAH CLUSTER OPTIMAL")
print("-" * 90)

# Elbow Method dan Silhouette Score
inertias = []
silhouette_scores = []
K_range = range(2, min(11, len(provinsi)))

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

print("\nElbow Method & Silhouette Score Results:")
print(f"{'K':<5} {'Inertia':<15} {'Silhouette':<15}")
print("-" * 35)
for k, inertia, silhouette in zip(K_range, inertias, silhouette_scores):
    print(f"{k:<5} {inertia:<15.2f} {silhouette:<15.4f}")

# Pilih K dengan silhouette score tertinggi
optimal_k = K_range[np.argmax(silhouette_scores)]
print(f"\n✓ Jumlah cluster optimal (berdasarkan Silhouette): {optimal_k}")

# 5. K-MEANS CLUSTERING
print("\n5. K-MEANS CLUSTERING")
print("-" * 90)

kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)

# Evaluasi
silhouette_kmeans = silhouette_score(X_scaled, kmeans_labels)
davies_bouldin_kmeans = davies_bouldin_score(X_scaled, kmeans_labels)
calinski_kmeans = calinski_harabasz_score(X_scaled, kmeans_labels)

print(f"\nMetrik Evaluasi K-Means:")
print(f"  • Silhouette Score: {silhouette_kmeans:.4f} (range: -1 to 1, lebih tinggi = lebih baik)")
print(f"  • Davies-Bouldin Index: {davies_bouldin_kmeans:.4f} (lebih rendah = lebih baik)")
print(f"  • Calinski-Harabasz Score: {calinski_kmeans:.2f} (lebih tinggi = lebih baik)")

# Tambahkan hasil clustering ke dataframe
df_provinsi['KMeans_Cluster'] = kmeans_labels

print(f"\nDistribusi Cluster K-Means:")
for cluster in range(optimal_k):
    count = (kmeans_labels == cluster).sum()
    print(f"  Cluster {cluster}: {count} provinsi")

# Karakteristik setiap cluster
print(f"\n{'='*90}")
print("KARAKTERISTIK SETIAP CLUSTER (K-MEANS)")
print(f"{'='*90}")

for cluster in range(optimal_k):
    print(f"\n--- CLUSTER {cluster} ---")
    cluster_data = df_provinsi[df_provinsi['KMeans_Cluster'] == cluster]
    print(f"Jumlah provinsi: {len(cluster_data)}")
    print(f"Provinsi: {', '.join(cluster_data['Provinsi'].values)}")
    print("\nRata-rata karakteristik:")
    print(cluster_data[feature_cols].mean().to_string())
    print("-" * 90)

# 6. HIERARCHICAL CLUSTERING
print("\n6. HIERARCHICAL CLUSTERING")
print("-" * 90)

# Coba berbagai metode linkage
linkage_methods = ['ward', 'complete', 'average', 'single']
best_silhouette = -1
best_method = None
best_hier_labels = None

print("Evaluasi berbagai metode linkage:")
print(f"{'Method':<15} {'Silhouette':<15} {'Davies-Bouldin':<20} {'Calinski-Harabasz':<20}")
print("-" * 70)

for method in linkage_methods:
    try:
        hier = AgglomerativeClustering(n_clusters=optimal_k, linkage=method)
        hier_labels = hier.fit_predict(X_scaled)
        silhouette = silhouette_score(X_scaled, hier_labels)
        davies_bouldin = davies_bouldin_score(X_scaled, hier_labels)
        calinski = calinski_harabasz_score(X_scaled, hier_labels)
        
        print(f"{method:<15} {silhouette:<15.4f} {davies_bouldin:<20.4f} {calinski:<20.2f}")
        
        if silhouette > best_silhouette:
            best_silhouette = silhouette
            best_method = method
            best_hier_labels = hier_labels
    except:
        print(f"{method:<15} Error - tidak kompatibel")

print(f"\n✓ Metode linkage terbaik: {best_method.upper()}")

# Evaluasi Hierarchical Clustering terbaik
davies_bouldin_hier = davies_bouldin_score(X_scaled, best_hier_labels)
calinski_hier = calinski_harabasz_score(X_scaled, best_hier_labels)

print(f"\nMetrik Evaluasi Hierarchical Clustering ({best_method}):")
print(f"  • Silhouette Score: {best_silhouette:.4f}")
print(f"  • Davies-Bouldin Index: {davies_bouldin_hier:.4f}")
print(f"  • Calinski-Harabasz Score: {calinski_hier:.2f}")

# Tambahkan hasil clustering ke dataframe
df_provinsi['Hierarchical_Cluster'] = best_hier_labels

print(f"\nDistribusi Cluster Hierarchical:")
for cluster in range(optimal_k):
    count = (best_hier_labels == cluster).sum()
    print(f"  Cluster {cluster}: {count} provinsi")

# 7. PCA UNTUK VISUALISASI
print("\n7. REDUKSI DIMENSI DENGAN PCA")
print("-" * 90)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print(f"Variance explained by PC1: {pca.explained_variance_ratio_[0]:.2%}")
print(f"Variance explained by PC2: {pca.explained_variance_ratio_[1]:.2%}")
print(f"Total variance explained: {pca.explained_variance_ratio_.sum():.2%}")

# 8. VISUALISASI
print("\n8. VISUALISASI")
print("-" * 90)

fig = plt.figure(figsize=(20, 14))

# Plot 1: Elbow Method
ax1 = plt.subplot(3, 4, 1)
ax1.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
ax1.set_xlabel('Jumlah Cluster (K)', fontweight='bold', fontsize=10)
ax1.set_ylabel('Inertia', fontweight='bold', fontsize=10)
ax1.set_title('Elbow Method', fontweight='bold', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal K={optimal_k}')
ax1.legend()

# Plot 2: Silhouette Score
ax2 = plt.subplot(3, 4, 2)
ax2.plot(K_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
ax2.set_xlabel('Jumlah Cluster (K)', fontweight='bold', fontsize=10)
ax2.set_ylabel('Silhouette Score', fontweight='bold', fontsize=10)
ax2.set_title('Silhouette Score vs K', fontweight='bold', fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal K={optimal_k}')
ax2.legend()

# Plot 3: K-Means Clusters (PCA)
ax3 = plt.subplot(3, 4, 3)
scatter = ax3.scatter(X_pca[:, 0], X_pca[:, 1], 
                     c=kmeans_labels, cmap='viridis', 
                     s=200, alpha=0.7, edgecolors='black', linewidth=1.5)
centroids_pca = pca.transform(kmeans.cluster_centers_)
ax3.scatter(centroids_pca[:, 0], centroids_pca[:, 1],
           c='red', marker='X', s=500, edgecolors='black', linewidths=3, label='Centroids')
ax3.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontweight='bold', fontsize=10)
ax3.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontweight='bold', fontsize=10)
ax3.set_title('K-Means Clustering (PCA)', fontweight='bold', fontsize=12)
ax3.legend()
plt.colorbar(scatter, ax=ax3, label='Cluster')

# Plot 4: Hierarchical Clusters (PCA)
ax4 = plt.subplot(3, 4, 4)
scatter = ax4.scatter(X_pca[:, 0], X_pca[:, 1], 
                     c=best_hier_labels, cmap='plasma', 
                     s=200, alpha=0.7, edgecolors='black', linewidth=1.5)
ax4.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontweight='bold', fontsize=10)
ax4.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontweight='bold', fontsize=10)
ax4.set_title(f'Hierarchical Clustering (PCA) - {best_method}', fontweight='bold', fontsize=12)
plt.colorbar(scatter, ax=ax4, label='Cluster')

# Plot 5: Dendrogram
ax5 = plt.subplot(3, 4, 5)
linkage_matrix = linkage(X_scaled, method=best_method)
dendrogram(linkage_matrix, ax=ax5, truncate_mode='lastp', p=15, 
           leaf_font_size=8, leaf_rotation=90)
ax5.set_title(f'Dendrogram ({best_method.capitalize()})', fontweight='bold', fontsize=12)
ax5.set_xlabel('Sample Index', fontweight='bold', fontsize=10)
ax5.set_ylabel('Distance', fontweight='bold', fontsize=10)
ax5.axhline(y=linkage_matrix[-optimal_k, 2], color='r', linestyle='--', label=f'Cut for {optimal_k} clusters')
ax5.legend()

# Plot 6: Perbandingan Metrik
ax6 = plt.subplot(3, 4, 6)
metrics = ['Silhouette\n(↑ better)', 'Davies-Bouldin\n(↓ better)', 'Calinski-H.\n(↑ better)']
kmeans_metrics_norm = [
    silhouette_kmeans, 
    1 - davies_bouldin_kmeans/10,  # Inverse & scale untuk visualisasi
    calinski_kmeans/1000  # Scale
]
hier_metrics_norm = [
    best_silhouette, 
    1 - davies_bouldin_hier/10, 
    calinski_hier/1000
]

x = np.arange(len(metrics))
width = 0.35

bars1 = ax6.bar(x - width/2, kmeans_metrics_norm, width, label='K-Means', alpha=0.8, color='steelblue')
bars2 = ax6.bar(x + width/2, hier_metrics_norm, width, label='Hierarchical', alpha=0.8, color='coral')

ax6.set_ylabel('Normalized Score', fontweight='bold', fontsize=10)
ax6.set_title('Perbandingan Metrik Evaluasi', fontweight='bold', fontsize=12)
ax6.set_xticks(x)
ax6.set_xticklabels(metrics, fontsize=9)
ax6.legend()
ax6.grid(axis='y', alpha=0.3)

# Plot 7: Distribusi Cluster K-Means
ax7 = plt.subplot(3, 4, 7)
cluster_counts_km = pd.Series(kmeans_labels).value_counts().sort_index()
bars = ax7.bar(cluster_counts_km.index, cluster_counts_km.values, 
               color=plt.cm.viridis(np.linspace(0.2, 0.8, optimal_k)), 
               alpha=0.8, edgecolor='black', linewidth=1.5)
ax7.set_xlabel('Cluster', fontweight='bold', fontsize=10)
ax7.set_ylabel('Jumlah Provinsi', fontweight='bold', fontsize=10)
ax7.set_title('Distribusi K-Means', fontweight='bold', fontsize=12)
ax7.grid(axis='y', alpha=0.3)

for bar in bars:
    height = bar.get_height()
    ax7.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}', ha='center', va='bottom', fontweight='bold', fontsize=10)

# Plot 8: Distribusi Cluster Hierarchical
ax8 = plt.subplot(3, 4, 8)
cluster_counts_hc = pd.Series(best_hier_labels).value_counts().sort_index()
bars = ax8.bar(cluster_counts_hc.index, cluster_counts_hc.values,
               color=plt.cm.plasma(np.linspace(0.2, 0.8, optimal_k)),
               alpha=0.8, edgecolor='black', linewidth=1.5)
ax8.set_xlabel('Cluster', fontweight='bold', fontsize=10)
ax8.set_ylabel('Jumlah Provinsi', fontweight='bold', fontsize=10)
ax8.set_title('Distribusi Hierarchical', fontweight='bold', fontsize=12)
ax8.grid(axis='y', alpha=0.3)

for bar in bars:
    height = bar.get_height()
    ax8.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}', ha='center', va='bottom', fontweight='bold', fontsize=10)

# Plot 9-12: Heatmap karakteristik per cluster
cluster_means_km = df_provinsi.groupby('KMeans_Cluster')[feature_cols].mean()

ax9 = plt.subplot(3, 4, 9)
sns.heatmap(cluster_means_km.T, annot=True, fmt='.0f', cmap='YlOrRd', 
            ax=ax9, cbar_kws={'label': 'Nilai'}, linewidths=0.5)
ax9.set_title('Karakteristik Cluster K-Means', fontweight='bold', fontsize=11)
ax9.set_ylabel('Fitur', fontweight='bold', fontsize=9)
ax9.set_xlabel('Cluster', fontweight='bold', fontsize=9)
ax9.tick_params(axis='y', labelsize=8)

# Plot 10: Top 10 Provinsi - Kasus Terkonfirmasi
ax10 = plt.subplot(3, 4, 10)
top_provinces = df_provinsi.nlargest(10, 'Kasus_Terkonfirmasi_Akumulatif')
bars = ax10.barh(range(len(top_provinces)), top_provinces['Kasus_Terkonfirmasi_Akumulatif'],
                 color=plt.cm.Reds(np.linspace(0.4, 0.9, len(top_provinces))))
ax10.set_yticks(range(len(top_provinces)))
ax10.set_yticklabels(top_provinces['Provinsi'], fontsize=9)
ax10.invert_yaxis()
ax10.set_xlabel('Total Kasus', fontweight='bold', fontsize=10)
ax10.set_title('Top 10 Provinsi - Kasus Terkonfirmasi', fontweight='bold', fontsize=11)
ax10.grid(axis='x', alpha=0.3)

# Plot 11: Tingkat Kesembuhan vs Kematian
ax11 = plt.subplot(3, 4, 11)
scatter = ax11.scatter(df_provinsi['Tingkat_Kesembuhan'], 
                      df_provinsi['Tingkat_Kematian'],
                      c=kmeans_labels, s=150, cmap='viridis', 
                      alpha=0.7, edgecolors='black', linewidth=1)
ax11.set_xlabel('Tingkat Kesembuhan (%)', fontweight='bold', fontsize=10)
ax11.set_ylabel('Tingkat Kematian (%)', fontweight='bold', fontsize=10)
ax11.set_title('Kesembuhan vs Kematian (K-Means)', fontweight='bold', fontsize=11)
ax11.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax11, label='Cluster')

# Plot 12: Correlation Matrix
ax12 = plt.subplot(3, 4, 12)
corr_matrix = df_provinsi[feature_cols].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, ax=ax12, cbar_kws={'label': 'Correlation'},
            linewidths=0.5, square=True)
ax12.set_title('Correlation Matrix', fontweight='bold', fontsize=11)
ax12.tick_params(axis='both', labelsize=8)

plt.tight_layout()
plt.savefig('clustering_covid19_provinsi.png', dpi=300, bbox_inches='tight')
print("Visualisasi disimpan sebagai 'clustering_covid19_provinsi.png'")
plt.show()

# 9. SAVE HASIL
print("\n9. MENYIMPAN HASIL")
print("-" * 90)

# Simpan hasil clustering
df_provinsi.to_excel('hasil_clustering_covid19.xlsx', index=False)
print("Hasil clustering disimpan ke 'hasil_clustering_covid19.xlsx'")

# 10. KESIMPULAN
print("\n" + "=" * 90)
print("KESIMPULAN ANALISIS CLUSTERING COVID-19")
print("=" * 90)
print(f"✓ Jumlah cluster optimal: {optimal_k}")
print(f"✓ K-Means Silhouette Score: {silhouette_kmeans:.4f}")
print(f"✓ Hierarchical Silhouette Score: {best_silhouette:.4f}")
print(f"✓ Metode terbaik: {'K-Means' if silhouette_kmeans > best_silhouette else 'Hierarchical (' + best_method + ')'}")
print(f"✓ Linkage terbaik untuk Hierarchical: {best_method}")

print("\n💡 INTERPRETASI CLUSTER:")
for cluster in range(optimal_k):
    cluster_data = df_provinsi[df_provinsi['KMeans_Cluster'] == cluster]
    avg_cases = cluster_data['Kasus_Terkonfirmasi_Akumulatif'].mean()
    avg_recovery = cluster_data['Tingkat_Kesembuhan'].mean()
    avg_mortality = cluster_data['Tingkat_Kematian'].mean()
    
    print(f"\n   Cluster {cluster} ({len(cluster_data)} provinsi):")
    print(f"   • Rata-rata kasus: {avg_cases:,.0f}")
    print(f"   • Tingkat kesembuhan: {avg_recovery:.1f}%")
    print(f"   • Tingkat kematian: {avg_mortality:.1f}%")
    
    if avg_cases > df_provinsi['Kasus_Terkonfirmasi_Akumulatif'].median():
        severity = "TINGGI"
    elif avg_cases > df_provinsi['Kasus_Terkonfirmasi_Akumulatif'].quantile(0.25):
        severity = "SEDANG"
    else:
        severity = "RENDAH"
    print(f"   • Kategori dampak: {severity}")

print("\n" + "=" * 90)