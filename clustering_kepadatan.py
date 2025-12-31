import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)

print("=" * 80)
print("ANALISIS CLUSTERING KEPADATAN PENDUDUK PROVINSI DI INDONESIA")
print("=" * 80)

# 1. LOAD DATA
print("\n1. LOADING DATA")
print("-" * 80)

# Baca file Excel
df = pd.read_excel('Kepadatan Penduduk menurut Provinsi.xlsx')

print("Data berhasil dimuat!")
print(f"\nShape: {df.shape}")
print(f"\nInfo Dataset:")
print(df.info())
print(f"\nPreview Data:")
print(df.head(10))

# 2. EKSPLORASI DATA
print("\n2. EKSPLORASI DATA")
print("-" * 80)

print("\nStatistik Deskriptif:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

# 3. DATA PREPROCESSING
print("\n3. DATA PREPROCESSING")
print("-" * 80)

# Asumsi kolom: Provinsi, Luas_Wilayah, Jumlah_Penduduk, Kepadatan, dll
# Sesuaikan dengan nama kolom sebenarnya di file Excel Anda

# Identifikasi kolom numerik untuk clustering
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"Kolom numerik yang tersedia: {numeric_cols}")

# Hapus kolom ID jika ada
id_cols = [col for col in numeric_cols if 'id' in col.lower() or 'fid' in col.lower()]
numeric_cols = [col for col in numeric_cols if col not in id_cols]

print(f"Kolom untuk clustering: {numeric_cols}")

# Handle missing values
df_clean = df.dropna(subset=numeric_cols)
print(f"\nData setelah cleaning: {df_clean.shape[0]} baris")

# Pilih fitur untuk clustering
X = df_clean[numeric_cols].values
provinsi = df_clean.iloc[:, 0].values  # Asumsi kolom pertama adalah nama provinsi

# Standardisasi data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Data telah distandardisasi: shape = {X_scaled.shape}")

# 4. MENENTUKAN JUMLAH CLUSTER OPTIMAL
print("\n4. MENENTUKAN JUMLAH CLUSTER OPTIMAL")
print("-" * 80)

# Elbow Method
inertias = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

print("\nElbow Method Results:")
for k, inertia, silhouette in zip(K_range, inertias, silhouette_scores):
    print(f"K={k}: Inertia={inertia:.2f}, Silhouette Score={silhouette:.4f}")

# 5. K-MEANS CLUSTERING
print("\n5. K-MEANS CLUSTERING")
print("-" * 80)

# Pilih jumlah cluster optimal (misal: 3)
optimal_k = 3
print(f"Jumlah cluster yang dipilih: {optimal_k}")

kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)

# Evaluasi
silhouette_kmeans = silhouette_score(X_scaled, kmeans_labels)
davies_bouldin_kmeans = davies_bouldin_score(X_scaled, kmeans_labels)
calinski_kmeans = calinski_harabasz_score(X_scaled, kmeans_labels)

print(f"\nMetrik Evaluasi K-Means:")
print(f"  • Silhouette Score: {silhouette_kmeans:.4f} (semakin mendekati 1, semakin baik)")
print(f"  • Davies-Bouldin Index: {davies_bouldin_kmeans:.4f} (semakin kecil, semakin baik)")
print(f"  • Calinski-Harabasz Score: {calinski_kmeans:.2f} (semakin besar, semakin baik)")

# Tambahkan hasil clustering ke dataframe
df_clean['KMeans_Cluster'] = kmeans_labels

print(f"\nDistribusi Cluster K-Means:")
print(df_clean['KMeans_Cluster'].value_counts().sort_index())

# Karakteristik setiap cluster
print(f"\nKarakteristik Setiap Cluster (K-Means):")
for cluster in range(optimal_k):
    print(f"\n--- Cluster {cluster} ---")
    cluster_data = df_clean[df_clean['KMeans_Cluster'] == cluster]
    print(f"Jumlah provinsi: {len(cluster_data)}")
    print(f"Provinsi: {', '.join(cluster_data.iloc[:, 0].values.astype(str)[:5])}...")
    print(cluster_data[numeric_cols].mean())

# 6. HIERARCHICAL CLUSTERING
print("\n6. HIERARCHICAL CLUSTERING")
print("-" * 80)

# Agglomerative Clustering dengan berbagai linkage
linkage_methods = ['ward', 'complete', 'average']
best_silhouette = -1
best_method = None
best_hier_labels = None

print("Mencoba berbagai metode linkage:")
for method in linkage_methods:
    hier = AgglomerativeClustering(n_clusters=optimal_k, linkage=method)
    hier_labels = hier.fit_predict(X_scaled)
    silhouette = silhouette_score(X_scaled, hier_labels)
    print(f"  {method.capitalize()}: Silhouette Score = {silhouette:.4f}")
    
    if silhouette > best_silhouette:
        best_silhouette = silhouette
        best_method = method
        best_hier_labels = hier_labels

print(f"\nMetode linkage terbaik: {best_method.upper()}")

# Evaluasi Hierarchical Clustering terbaik
davies_bouldin_hier = davies_bouldin_score(X_scaled, best_hier_labels)
calinski_hier = calinski_harabasz_score(X_scaled, best_hier_labels)

print(f"\nMetrik Evaluasi Hierarchical Clustering ({best_method}):")
print(f"  • Silhouette Score: {best_silhouette:.4f}")
print(f"  • Davies-Bouldin Index: {davies_bouldin_hier:.4f}")
print(f"  • Calinski-Harabasz Score: {calinski_hier:.2f}")

# Tambahkan hasil clustering ke dataframe
df_clean['Hierarchical_Cluster'] = best_hier_labels

print(f"\nDistribusi Cluster Hierarchical:")
print(df_clean['Hierarchical_Cluster'].value_counts().sort_index())

# 7. VISUALISASI
print("\n7. VISUALISASI")
print("-" * 80)

fig = plt.figure(figsize=(18, 12))

# Plot 1: Elbow Method
ax1 = plt.subplot(3, 3, 1)
ax1.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
ax1.set_xlabel('Jumlah Cluster (K)', fontweight='bold')
ax1.set_ylabel('Inertia', fontweight='bold')
ax1.set_title('Elbow Method', fontweight='bold', fontsize=12)
ax1.grid(True, alpha=0.3)

# Plot 2: Silhouette Score
ax2 = plt.subplot(3, 3, 2)
ax2.plot(K_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
ax2.set_xlabel('Jumlah Cluster (K)', fontweight='bold')
ax2.set_ylabel('Silhouette Score', fontweight='bold')
ax2.set_title('Silhouette Score vs K', fontweight='bold', fontsize=12)
ax2.grid(True, alpha=0.3)

# Plot 3: K-Means Clusters (2D - PCA atau 2 fitur pertama)
ax3 = plt.subplot(3, 3, 3)
if X_scaled.shape[1] >= 2:
    scatter = ax3.scatter(X_scaled[:, 0], X_scaled[:, 1], 
                         c=kmeans_labels, cmap='viridis', 
                         s=100, alpha=0.6, edgecolors='black')
    ax3.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
               c='red', marker='X', s=300, edgecolors='black', linewidths=2, label='Centroids')
    ax3.set_xlabel(f'{numeric_cols[0]} (scaled)', fontweight='bold')
    ax3.set_ylabel(f'{numeric_cols[1]} (scaled)', fontweight='bold')
    ax3.set_title('K-Means Clustering (2D)', fontweight='bold', fontsize=12)
    ax3.legend()
    plt.colorbar(scatter, ax=ax3, label='Cluster')

# Plot 4: Dendrogram (Hierarchical)
ax4 = plt.subplot(3, 3, 4)
linkage_matrix = linkage(X_scaled, method=best_method)
dendrogram(linkage_matrix, ax=ax4, truncate_mode='lastp', p=20)
ax4.set_title(f'Dendrogram ({best_method.capitalize()} Linkage)', fontweight='bold', fontsize=12)
ax4.set_xlabel('Cluster Index', fontweight='bold')
ax4.set_ylabel('Distance', fontweight='bold')

# Plot 5: Hierarchical Clusters (2D)
ax5 = plt.subplot(3, 3, 5)
if X_scaled.shape[1] >= 2:
    scatter = ax5.scatter(X_scaled[:, 0], X_scaled[:, 1], 
                         c=best_hier_labels, cmap='plasma', 
                         s=100, alpha=0.6, edgecolors='black')
    ax5.set_xlabel(f'{numeric_cols[0]} (scaled)', fontweight='bold')
    ax5.set_ylabel(f'{numeric_cols[1]} (scaled)', fontweight='bold')
    ax5.set_title('Hierarchical Clustering (2D)', fontweight='bold', fontsize=12)
    plt.colorbar(scatter, ax=ax5, label='Cluster')

# Plot 6: Perbandingan Metrik
ax6 = plt.subplot(3, 3, 6)
metrics = ['Silhouette', 'Davies-Bouldin', 'Calinski-Harabasz']
kmeans_metrics = [silhouette_kmeans, davies_bouldin_kmeans, calinski_kmeans/100]  # Scaling CH
hier_metrics = [best_silhouette, davies_bouldin_hier, calinski_hier/100]

x = np.arange(len(metrics))
width = 0.35

bars1 = ax6.bar(x - width/2, kmeans_metrics, width, label='K-Means', alpha=0.8)
bars2 = ax6.bar(x + width/2, hier_metrics, width, label='Hierarchical', alpha=0.8)

ax6.set_ylabel('Score', fontweight='bold')
ax6.set_title('Perbandingan Metrik Evaluasi', fontweight='bold', fontsize=12)
ax6.set_xticks(x)
ax6.set_xticklabels(metrics, rotation=15, ha='right')
ax6.legend()
ax6.grid(axis='y', alpha=0.3)

# Plot 7: Distribusi Cluster K-Means
ax7 = plt.subplot(3, 3, 7)
cluster_counts_km = df_clean['KMeans_Cluster'].value_counts().sort_index()
bars = ax7.bar(cluster_counts_km.index, cluster_counts_km.values, 
               color=plt.cm.viridis(np.linspace(0.2, 0.8, optimal_k)), 
               alpha=0.8, edgecolor='black')
ax7.set_xlabel('Cluster', fontweight='bold')
ax7.set_ylabel('Jumlah Provinsi', fontweight='bold')
ax7.set_title('Distribusi K-Means Cluster', fontweight='bold', fontsize=12)
ax7.grid(axis='y', alpha=0.3)

for bar in bars:
    height = bar.get_height()
    ax7.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}', ha='center', va='bottom', fontweight='bold')

# Plot 8: Distribusi Cluster Hierarchical
ax8 = plt.subplot(3, 3, 8)
cluster_counts_hc = df_clean['Hierarchical_Cluster'].value_counts().sort_index()
bars = ax8.bar(cluster_counts_hc.index, cluster_counts_hc.values,
               color=plt.cm.plasma(np.linspace(0.2, 0.8, optimal_k)),
               alpha=0.8, edgecolor='black')
ax8.set_xlabel('Cluster', fontweight='bold')
ax8.set_ylabel('Jumlah Provinsi', fontweight='bold')
ax8.set_title('Distribusi Hierarchical Cluster', fontweight='bold', fontsize=12)
ax8.grid(axis='y', alpha=0.3)

for bar in bars:
    height = bar.get_height()
    ax8.set_text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}', ha='center', va='bottom', fontweight='bold')

# Plot 9: Heatmap Cluster Characteristics
ax9 = plt.subplot(3, 3, 9)
cluster_means = df_clean.groupby('KMeans_Cluster')[numeric_cols].mean()
sns.heatmap(cluster_means.T, annot=True, fmt='.2f', cmap='YlOrRd', 
            ax=ax9, cbar_kws={'label': 'Nilai'})
ax9.set_title('Karakteristik Cluster (K-Means)', fontweight='bold', fontsize=12)
ax9.set_ylabel('Fitur', fontweight='bold')
ax9.set_xlabel('Cluster', fontweight='bold')

plt.tight_layout()
plt.savefig('clustering_kepadatan_penduduk.png', dpi=300, bbox_inches='tight')
print("Visualisasi disimpan sebagai 'clustering_kepadatan_penduduk.png'")
plt.show()

# 8. SAVE HASIL
print("\n8. MENYIMPAN HASIL")
print("-" * 80)

# Simpan hasil clustering
df_clean.to_excel('hasil_clustering_kepadatan.xlsx', index=False)
print("Hasil clustering disimpan ke 'hasil_clustering_kepadatan.xlsx'")

# 9. KESIMPULAN
print("\n" + "=" * 80)
print("KESIMPULAN")
print("=" * 80)
print(f"1. Jumlah cluster optimal: {optimal_k}")
print(f"2. K-Means Silhouette Score: {silhouette_kmeans:.4f}")
print(f"3. Hierarchical Silhouette Score: {best_silhouette:.4f}")
print(f"4. Metode terbaik: {'K-Means' if silhouette_kmeans > best_silhouette else 'Hierarchical'}")
print(f"5. Linkage terbaik untuk Hierarchical: {best_method}")
print("\n💡 Interpretasi:")
print("   - Cluster 0: Provinsi dengan karakteristik tertentu")
print("   - Cluster 1: Provinsi dengan karakteristik berbeda")
print("   - Cluster 2: Provinsi dengan karakteristik lainnya")
print("=" * 80)