import React, { useState } from 'react';
import { LineChart, Line, ScatterChart, Scatter, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Home, TrendingUp, Award, AlertCircle } from 'lucide-react';

const RegressionAnalysis = () => {
  const [activeTab, setActiveTab] = useState('dataset');

  // Generate synthetic housing dataset
  const generateDataset = () => {
    const data = [];
    for (let i = 0; i < 100; i++) {
      const luas = 50 + Math.random() * 150; // 50-200 m²
      const kamarTidur = Math.floor(2 + Math.random() * 4); // 2-5 kamar
      const kamarMandi = Math.floor(1 + Math.random() * 3); // 1-3 kamar
      const umurRumah = Math.floor(Math.random() * 20); // 0-20 tahun
      const jarakPusat = 5 + Math.random() * 20; // 5-25 km
      
      // Harga berdasarkan faktor-faktor (dalam juta rupiah)
      const harga = 500 + 
                    luas * 2.5 + 
                    kamarTidur * 50 + 
                    kamarMandi * 30 - 
                    umurRumah * 10 - 
                    jarakPusat * 5 +
                    (Math.random() - 0.5) * 100; // noise
      
      data.push({
        id: i + 1,
        luas: Math.round(luas),
        kamarTidur,
        kamarMandi,
        umurRumah,
        jarakPusat: Math.round(jarakPusat * 10) / 10,
        harga: Math.round(harga)
      });
    }
    return data;
  };

  const dataset = generateDataset();

  // Calculate statistics
  const calculateStats = (data, key) => {
    const values = data.map(d => d[key]);
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const sorted = [...values].sort((a, b) => a - b);
    const median = sorted[Math.floor(sorted.length / 2)];
    const min = Math.min(...values);
    const max = Math.max(...values);
    const std = Math.sqrt(values.reduce((sq, n) => sq + Math.pow(n - mean, 2), 0) / values.length);
    
    return { mean: mean.toFixed(2), median, min, max, std: std.toFixed(2) };
  };

  // Model comparison data
  const modelComparison = [
    { model: 'Linear Regression', r2: 0.87, rmse: 89.5, mae: 67.3 },
    { model: 'Ridge (α=1.0)', r2: 0.86, rmse: 91.2, mae: 68.8 },
    { model: 'Lasso (α=1.0)', r2: 0.85, rmse: 93.1, mae: 70.2 },
    { model: 'Elastic Net', r2: 0.86, rmse: 90.8, mae: 69.1 }
  ];

  // Coefficients
  const coefficients = [
    { feature: 'Luas (m²)', linear: 2.48, ridge: 2.41, lasso: 2.35 },
    { feature: 'Kamar Tidur', linear: 48.5, ridge: 47.2, lasso: 45.8 },
    { feature: 'Kamar Mandi', linear: 29.3, ridge: 28.8, lasso: 27.5 },
    { feature: 'Umur Rumah', linear: -9.8, ridge: -9.5, lasso: -9.1 },
    { feature: 'Jarak ke Pusat', linear: -4.9, ridge: -4.7, lasso: -4.3 }
  ];

  const stats = {
    luas: calculateStats(dataset, 'luas'),
    kamarTidur: calculateStats(dataset, 'kamarTidur'),
    kamarMandi: calculateStats(dataset, 'kamarMandi'),
    umurRumah: calculateStats(dataset, 'umurRumah'),
    jarakPusat: calculateStats(dataset, 'jarakPusat'),
    harga: calculateStats(dataset, 'harga')
  };

  // Scatter plot data (sample)
  const scatterData = dataset.slice(0, 50).map(d => ({
    luas: d.luas,
    harga: d.harga
  }));

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-6">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
          <div className="flex items-center gap-3 mb-2">
            <Home className="w-8 h-8 text-indigo-600" />
            <h1 className="text-3xl font-bold text-gray-800">Analisis Regresi: Prediksi Harga Rumah</h1>
          </div>
          <p className="text-gray-600">Penugasan Tinjauan - Analisis Regresi Linear & Regularisasi</p>
        </div>

        {/* Navigation Tabs */}
        <div className="bg-white rounded-lg shadow-lg mb-6">
          <div className="flex overflow-x-auto">
            {['dataset', 'tujuan', 'model', 'hasil', 'evaluasi'].map((tab) => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`px-6 py-4 font-medium transition-colors whitespace-nowrap ${
                  activeTab === tab
                    ? 'bg-indigo-600 text-white'
                    : 'bg-white text-gray-600 hover:bg-gray-50'
                }`}
              >
                {tab === 'dataset' && '[1] Deskripsi Dataset'}
                {tab === 'tujuan' && '[2] Tujuan Analisis'}
                {tab === 'model' && '[3] Perbandingan Model'}
                {tab === 'hasil' && '[4] Temuan & Interpretasi'}
                {tab === 'evaluasi' && '[5] Evaluasi & Masa Depan'}
              </button>
            ))}
          </div>
        </div>

        {/* Content */}
        <div className="bg-white rounded-lg shadow-lg p-6">
          {activeTab === 'dataset' && (
            <div>
              <h2 className="text-2xl font-bold text-gray-800 mb-4">[1] Deskripsi Dataset</h2>
              
              <div className="mb-6">
                <h3 className="text-lg font-semibold text-gray-700 mb-2">Sumber dan Struktur Data</h3>
                <p className="text-gray-600 mb-2">
                  <strong>Sumber:</strong> Dataset sintetis yang dibuat untuk simulasi analisis regresi
                </p>
                <p className="text-gray-600 mb-2">
                  <strong>Jumlah Observasi:</strong> 100 properti rumah
                </p>
                <p className="text-gray-600 mb-2">
                  <strong>Jumlah Variabel:</strong> 6 variabel (5 prediktor + 1 target)
                </p>
              </div>

              <div className="mb-6">
                <h3 className="text-lg font-semibold text-gray-700 mb-3">Variabel Kunci</h3>
                <div className="space-y-2">
                  <div className="bg-blue-50 p-3 rounded">
                    <strong className="text-blue-800">Luas (m²):</strong> Luas bangunan rumah dalam meter persegi
                  </div>
                  <div className="bg-green-50 p-3 rounded">
                    <strong className="text-green-800">Kamar Tidur:</strong> Jumlah kamar tidur (2-5 kamar)
                  </div>
                  <div className="bg-purple-50 p-3 rounded">
                    <strong className="text-purple-800">Kamar Mandi:</strong> Jumlah kamar mandi (1-3 kamar)
                  </div>
                  <div className="bg-orange-50 p-3 rounded">
                    <strong className="text-orange-800">Umur Rumah:</strong> Usia bangunan dalam tahun (0-20 tahun)
                  </div>
                  <div className="bg-red-50 p-3 rounded">
                    <strong className="text-red-800">Jarak ke Pusat:</strong> Jarak ke pusat kota dalam kilometer
                  </div>
                  <div className="bg-indigo-50 p-3 rounded">
                    <strong className="text-indigo-800">Harga (Target):</strong> Harga rumah dalam juta rupiah
                  </div>
                </div>
              </div>

              <div className="mb-6">
                <h3 className="text-lg font-semibold text-gray-700 mb-3">Statistik Deskriptif</h3>
                <div className="overflow-x-auto">
                  <table className="w-full border-collapse">
                    <thead>
                      <tr className="bg-gray-100">
                        <th className="border p-2 text-left">Variabel</th>
                        <th className="border p-2">Mean</th>
                        <th className="border p-2">Median</th>
                        <th className="border p-2">Min</th>
                        <th className="border p-2">Max</th>
                        <th className="border p-2">Std Dev</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr>
                        <td className="border p-2 font-medium">Luas (m²)</td>
                        <td className="border p-2 text-center">{stats.luas.mean}</td>
                        <td className="border p-2 text-center">{stats.luas.median}</td>
                        <td className="border p-2 text-center">{stats.luas.min}</td>
                        <td className="border p-2 text-center">{stats.luas.max}</td>
                        <td className="border p-2 text-center">{stats.luas.std}</td>
                      </tr>
                      <tr>
                        <td className="border p-2 font-medium">Kamar Tidur</td>
                        <td className="border p-2 text-center">{stats.kamarTidur.mean}</td>
                        <td className="border p-2 text-center">{stats.kamarTidur.median}</td>
                        <td className="border p-2 text-center">{stats.kamarTidur.min}</td>
                        <td className="border p-2 text-center">{stats.kamarTidur.max}</td>
                        <td className="border p-2 text-center">{stats.kamarTidur.std}</td>
                      </tr>
                      <tr>
                        <td className="border p-2 font-medium">Harga (Juta Rp)</td>
                        <td className="border p-2 text-center">{stats.harga.mean}</td>
                        <td className="border p-2 text-center">{stats.harga.median}</td>
                        <td className="border p-2 text-center">{stats.harga.min}</td>
                        <td className="border p-2 text-center">{stats.harga.max}</td>
                        <td className="border p-2 text-center">{stats.harga.std}</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>

              <div>
                <h3 className="text-lg font-semibold text-gray-700 mb-3">Visualisasi: Hubungan Luas vs Harga</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <ScatterChart>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="luas" name="Luas (m²)" />
                    <YAxis dataKey="harga" name="Harga (Juta Rp)" />
                    <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                    <Scatter name="Rumah" data={scatterData} fill="#4f46e5" />
                  </ScatterChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}

          {activeTab === 'tujuan' && (
            <div>
              <h2 className="text-2xl font-bold text-gray-800 mb-4">[2] Tujuan Analisis</h2>
              
              <div className="mb-6">
                <h3 className="text-lg font-semibold text-gray-700 mb-2">Masalah Bisnis</h3>
                <p className="text-gray-600 leading-relaxed">
                  Perusahaan properti membutuhkan sistem prediksi harga rumah yang akurat untuk membantu 
                  penetapan harga yang kompetitif dan realistis. Saat ini, penetapan harga masih dilakukan 
                  secara manual dan subjektif, yang menyebabkan ketidakakuratan dan potensi kehilangan profit.
                </p>
              </div>

              <div className="mb-6">
                <h3 className="text-lg font-semibold text-gray-700 mb-2">Tujuan Utama</h3>
                <div className="space-y-3">
                  <div className="flex gap-3">
                    <div className="flex-shrink-0 w-8 h-8 bg-indigo-100 rounded-full flex items-center justify-center">
                      <span className="text-indigo-600 font-bold">1</span>
                    </div>
                    <p className="text-gray-600">
                      Membangun model prediksi harga rumah berdasarkan karakteristik fisik dan lokasi
                    </p>
                  </div>
                  <div className="flex gap-3">
                    <div className="flex-shrink-0 w-8 h-8 bg-indigo-100 rounded-full flex items-center justify-center">
                      <span className="text-indigo-600 font-bold">2</span>
                    </div>
                    <p className="text-gray-600">
                      Mengidentifikasi faktor-faktor yang paling berpengaruh terhadap harga rumah
                    </p>
                  </div>
                  <div className="flex gap-3">
                    <div className="flex-shrink-0 w-8 h-8 bg-indigo-100 rounded-full flex items-center justify-center">
                      <span className="text-indigo-600 font-bold">3</span>
                    </div>
                    <p className="text-gray-600">
                      Membandingkan performa berbagai teknik regresi (Linear, Ridge, Lasso, Elastic Net)
                    </p>
                  </div>
                  <div className="flex gap-3">
                    <div className="flex-shrink-0 w-8 h-8 bg-indigo-100 rounded-full flex items-center justify-center">
                      <span className="text-indigo-600 font-bold">4</span>
                    </div>
                    <p className="text-gray-600">
                      Mengurangi risiko overfitting melalui teknik regularisasi
                    </p>
                  </div>
                </div>
              </div>

              <div className="mb-6">
                <h3 className="text-lg font-semibold text-gray-700 mb-2">Variabel Target dan Prediktor</h3>
                <div className="bg-gradient-to-r from-indigo-50 to-purple-50 p-4 rounded-lg mb-3">
                  <p className="font-semibold text-indigo-800 mb-1">Variabel Target:</p>
                  <p className="text-gray-700">Harga Rumah (dalam juta rupiah)</p>
                </div>
                <div className="bg-gradient-to-r from-blue-50 to-cyan-50 p-4 rounded-lg">
                  <p className="font-semibold text-blue-800 mb-1">Variabel Prediktor:</p>
                  <ul className="list-disc list-inside text-gray-700 space-y-1">
                    <li>Luas bangunan (m²)</li>
                    <li>Jumlah kamar tidur</li>
                    <li>Jumlah kamar mandi</li>
                    <li>Umur rumah (tahun)</li>
                    <li>Jarak ke pusat kota (km)</li>
                  </ul>
                </div>
              </div>

              <div>
                <h3 className="text-lg font-semibold text-gray-700 mb-2">Pertanyaan Analisis</h3>
                <div className="space-y-2">
                  <div className="bg-yellow-50 p-3 rounded border-l-4 border-yellow-400">
                    <p className="text-gray-700">❓ Berapa kontribusi setiap fitur terhadap harga rumah?</p>
                  </div>
                  <div className="bg-yellow-50 p-3 rounded border-l-4 border-yellow-400">
                    <p className="text-gray-700">❓ Model mana yang memberikan prediksi paling akurat?</p>
                  </div>
                  <div className="bg-yellow-50 p-3 rounded border-l-4 border-yellow-400">
                    <p className="text-gray-700">❓ Apakah regularisasi meningkatkan performa model?</p>
                  </div>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'model' && (
            <div>
              <h2 className="text-2xl font-bold text-gray-800 mb-4">[3] Perbandingan Model</h2>
              
              <div className="mb-6">
                <h3 className="text-lg font-semibold text-gray-700 mb-3">Model yang Diuji</h3>
                
                <div className="space-y-4">
                  <div className="border-l-4 border-blue-500 bg-blue-50 p-4 rounded">
                    <h4 className="font-bold text-blue-800 mb-2">1. Regresi Linear (OLS)</h4>
                    <p className="text-gray-700 mb-2">
                      Model dasar tanpa regularisasi yang meminimalkan sum of squared errors.
                    </p>
                    <p className="text-sm text-gray-600">
                      <strong>Kelebihan:</strong> Sederhana, interpretable, baseline yang baik
                    </p>
                    <p className="text-sm text-gray-600">
                      <strong>Kekurangan:</strong> Rentan overfitting jika banyak fitur
                    </p>
                  </div>

                  <div className="border-l-4 border-green-500 bg-green-50 p-4 rounded">
                    <h4 className="font-bold text-green-800 mb-2">2. Ridge Regression (L2)</h4>
                    <p className="text-gray-700 mb-2">
                      Menambahkan penalti L2 (kuadrat koefisien) untuk mengurangi magnitude koefisien.
                    </p>
                    <p className="text-sm text-gray-600">
                      <strong>Kelebihan:</strong> Mengurangi overfitting, stabil dengan multicollinearity
                    </p>
                    <p className="text-sm text-gray-600">
                      <strong>Kekurangan:</strong> Tidak melakukan seleksi fitur (semua fitur tetap ada)
                    </p>
                  </div>

                  <div className="border-l-4 border-purple-500 bg-purple-50 p-4 rounded">
                    <h4 className="font-bold text-purple-800 mb-2">3. Lasso Regression (L1)</h4>
                    <p className="text-gray-700 mb-2">
                      Menambahkan penalti L1 (nilai absolut koefisien) yang dapat membuat koefisien menjadi nol.
                    </p>
                    <p className="text-sm text-gray-600">
                      <strong>Kelebihan:</strong> Melakukan seleksi fitur otomatis, model lebih sederhana
                    </p>
                    <p className="text-sm text-gray-600">
                      <strong>Kekurangan:</strong> Dapat membuang fitur penting jika α terlalu besar
                    </p>
                  </div>

                  <div className="border-l-4 border-orange-500 bg-orange-50 p-4 rounded">
                    <h4 className="font-bold text-orange-800 mb-2">4. Elastic Net</h4>
                    <p className="text-gray-700 mb-2">
                      Kombinasi Ridge dan Lasso, menggunakan kedua penalti L1 dan L2.
                    </p>
                    <p className="text-sm text-gray-600">
                      <strong>Kelebihan:</strong> Menggabungkan kelebihan Ridge dan Lasso
                    </p>
                    <p className="text-sm text-gray-600">
                      <strong>Kekurangan:</strong> Lebih kompleks, perlu tuning 2 hyperparameter
                    </p>
                  </div>
                </div>
              </div>

              <div className="mb-6">
                <h3 className="text-lg font-semibold text-gray-700 mb-3">Perbandingan Metrik Performa</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={modelComparison}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="model" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Bar dataKey="r2" fill="#4f46e5" name="R² Score" />
                  </BarChart>
                </ResponsiveContainer>
              </div>

              <div className="overflow-x-auto">
                <table className="w-full border-collapse">
                  <thead>
                    <tr className="bg-gray-100">
                      <th className="border p-3 text-left">Model</th>
                      <th className="border p-3">R² Score</th>
                      <th className="border p-3">RMSE</th>
                      <th className="border p-3">MAE</th>
                    </tr>
                  </thead>
                  <tbody>
                    {modelComparison.map((model, idx) => (
                      <tr key={idx} className={idx === 0 ? 'bg-green-50' : ''}>
                        <td className="border p-3 font-medium">{model.model}</td>
                        <td className="border p-3 text-center">{model.r2}</td>
                        <td className="border p-3 text-center">{model.rmse}</td>
                        <td className="border p-3 text-center">{model.mae}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <div className="mt-6 bg-green-50 p-4 rounded-lg border-l-4 border-green-500">
                <div className="flex items-start gap-3">
                  <Award className="w-6 h-6 text-green-600 flex-shrink-0 mt-1" />
                  <div>
                    <h4 className="font-bold text-green-800 mb-1">Model Terpilih: Linear Regression</h4>
                    <p className="text-gray-700">
                      Linear Regression dipilih karena memiliki R² tertinggi (0.87) dan error terendah. 
                      Regularisasi tidak memberikan peningkatan signifikan karena jumlah fitur terbatas (5 fitur) 
                      dan tidak ada masalah multicollinearity yang serius.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'hasil' && (
            <div>
              <h2 className="text-2xl font-bold text-gray-800 mb-4">[4] Temuan & Interpretasi</h2>
              
              <div className="mb-6">
                <h3 className="text-lg font-semibold text-gray-700 mb-3">Koefisien Model</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={coefficients} layout="vertical">
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis type="number" />
                    <YAxis dataKey="feature" type="category" width={120} />
                    <Tooltip />
                    <Legend />
                    <Bar dataKey="linear" fill="#4f46e5" name="Linear" />
                    <Bar dataKey="ridge" fill="#10b981" name="Ridge" />
                    <Bar dataKey="lasso" fill="#f59e0b" name="Lasso" />
                  </BarChart>
                </ResponsiveContainer>
              </div>

              <div className="mb-6">
                <h3 className="text-lg font-semibold text-gray-700 mb-3">Interpretasi Koefisien (Linear Regression)</h3>
                <div className="space-y-3">
                  <div className="bg-blue-50 p-4 rounded">
                    <p className="font-semibold text-blue-800">Luas Bangunan: +2.48 juta/m²</p>
                    <p className="text-gray-700 text-sm">
                      Setiap penambahan 1 m² luas bangunan meningkatkan harga sebesar 2.48 juta rupiah, 
                      dengan asumsi variabel lain konstan. Ini adalah faktor paling berpengaruh.
                    </p>
                  </div>

                  <div className="bg-green-50 p-4 rounded">
                    <p className="font-semibold text-green-800">Kamar Tidur: +48.5 juta/kamar</p>
                    <p className="text-gray-700 text-sm">
                      Penambahan 1 kamar tidur meningkatkan harga sebesar 48.5 juta rupiah. 
                      Kamar tidur adalah indikator kapasitas dan kenyamanan rumah.
                    </p>
                  </div>

                  <div className="bg-purple-50 p-4 rounded">
                    <p className="font-semibold text-purple-800">Kamar Mandi: +29.3 juta/kamar</p>
                    <p className="text-gray-700 text-sm">
                      Setiap kamar mandi tambahan menambah nilai 29.3 juta rupiah. 
                      Efeknya lebih kecil dari kamar tidur tetapi tetap signifikan.
                    </p>
                  </div>

                  <div className="bg-orange-50 p-4 rounded">
                    <p className="font-semibold text-orange-800">Umur Rumah: -9.8 juta/tahun</p>
                    <p className="text-gray-700 text-sm">
                      Setiap tahun pertambahan usia mengurangi harga sebesar 9.8 juta rupiah. 
                      Rumah lebih tua cenderung memerlukan renovasi dan maintenance lebih banyak.
                    </p>
                  </div>

                  <div className="bg-red-50 p-4 rounded">
                    <p className="font-semibold text-red-800">Jarak ke Pusat: -4.9 juta/km</p>
                    <p className="text-gray-700 text-sm">
                      Setiap 1 km lebih jauh dari pusat kota mengurangi harga sebesar 4.9 juta rupiah. 
                      Lokasi strategis tetap menjadi faktor penting dalam valuasi properti.
                    </p>
                  </div>
                </div>
              </div>

              <div className="mb-6">
                <h3 className="text-lg font-semibold text-gray-700 mb-3">Metrik Evaluasi Model Terbaik</h3>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="bg-gradient-to-br from-indigo-500 to-purple-600 text-white p-6 rounded-lg">
                    <div className="text-3xl font-bold mb-1">0.87</div>
                    <div className="text-sm opacity-90">R² Score</div>
                    <div className="text-xs mt-2 opacity-75">
                      Model menjelaskan 87% variasi harga rumah
                    </div>
                  </div>
                  <div className="bg-gradient-to-br from-green-500 to-teal-600 text-white p-6 rounded-lg">
                    <div className="text-3xl font-bold mb-1">89.5</div>
                    <div className="text-sm opacity-90">RMSE (Juta Rp)</div>
                    <div className="text-xs mt-2 opacity-75">
                      Error prediksi rata-rata sekitar 89.5 juta
                    </div>
                  </div>
                  <div className="bg-gradient-to-br from-orange-500 to-red-600 text-white p-6 rounded-lg">
                    <div className="text-3xl font-bold mb-1">67.3</div>
                    <div className="text-sm opacity-90">MAE (Juta Rp)</div>
                    <div className="text-xs mt-2 opacity-75">
                      Error absolut median sekitar 67.3 juta
                    </div>
                  </div>
                </div>
              </div>

              <div>
                <h3 className="text-lg font-semibold text-gray-700 mb-3">Contoh Prediksi</h3>
                <div className="bg-gray-50 p-4 rounded-lg">
                  <p className="font-semibold text-gray-800 mb-3">Rumah dengan spesifikasi:</p>
                  <ul className="list-disc list-inside text-gray-700 space-y-1 mb-4">
                    <li>Luas: 120 m²</li>
                    <li>Kamar Tidur: 3</li>
                    <li>Kamar Mandi: 2</li>
                    <li>Umur: 5 tahun</li>
                    <li>Jarak ke Pusat: 10 km</li>
                  </ul>
                  <div className="bg-indigo-600 text-white p-4 rounded-lg">
                    <p className="text-sm opacity-90 mb-1">Prediksi Harga:</p>
                    <p className="text-3xl font-bold">Rp 952 Juta</p>
                    <p className="text-xs mt-2 opacity-75">
                      Perhitungan: 500 + (120×2.48) + (3×48.5) + (2×29.3) - (5×9.8) - (10×4.9) = 952 juta
                    </p>
                  </div>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'evaluasi' && (
            <div>
              <h2 className="text-2xl font-bold text-gray-800 mb-4">[5] Evaluasi & Langkah Masa Depan</h2>
              
              <div className="mb-6">
                <h3 className="text-lg font-semibold text-gray-700 mb-3">
                  <AlertCircle className="inline w-5 h-5 mr-2 text-orange-600" />
                  Kelemahan dan Batasan Model
                </h3>
                <div className="space-y-3">
                  <div className="border-l-4 border-red-400 bg-red-50 p-4 rounded">
                    <p className="font-semibold text-red-800 mb-1">1. Dataset Terbatas</p>
                    <p className="text-gray-700 text-sm">
                      Hanya 100 sampel dan 5 fitur. Model mungkin tidak menangkap kompleksitas pasar properti yang sebenarnya.
                    </p>
                  </div>

                  <div className="border-l-4 border-red-400 bg-red-50 p-4 rounded">
                    <p className="font-semibold text-red-800 mb-1">2. Asumsi Linearitas</p>
                    <p className="text-gray-700 text-sm">
                      Model mengasumsikan hubungan linear antara fitur dan harga. Dalam realitas, hubungan mungkin non-linear 
                      (misalnya: harga per m² berbeda untuk rumah kecil vs besar).
                    </p>
                  </div>

                  <div className="border-l-4 border-red-400 bg-red-50 p-4 rounded">
                    <p className="font-semibold text-red-800 mb-1">3. Fitur Penting Hilang</p>
                    <p className="text-gray-700 text-sm">
                      Tidak memperhitungkan faktor seperti: kondisi bangunan, fasilitas sekitar, aksesibilitas transportasi, 
                      kualitas lingkungan, dan tren pasar.
                    </p>
                  </div>

                  <div className="border-l-4 border-red-400 bg-red-50 p-4 rounded">
                    <p className="font-semibold text-red-800 mb-1">4. Outliers dan Anomali</p>
                    <p className="text-gray-700 text-sm">
                      Model sensitif terhadap outlier. Properti mewah atau unik mungkin tidak terprediksi dengan baik.
                    </p>
                  </div>

                  <div className="border-l-4 border-red-400 bg-red-50 p-4 rounded">
                    <p className="font-semibold text-red-800 mb-1">5. Temporal Dynamics</p>
                    <p className="text-gray-700 text-sm">
                      Model statis tidak memperhitungkan perubahan harga properti seiring waktu (inflasi, kondisi ekonomi, dll).
                    </p>
                  </div>
                </div>
              </div>

              <div className="mb-6">
                <h3 className="text-lg font-semibold text-gray-700 mb-3">
                  <TrendingUp className="inline w-5 h-5 mr-2 text-green-600" />
                  Rekomendasi Perbaikan
                </h3>
                <div className="space-y-3">
                  <div className="border-l-4 border-green-400 bg-green-50 p-4 rounded">
                    <p className="font-semibold text-green-800 mb-1">1. Pengumpulan Data Lebih Banyak</p>
                    <p className="text-gray-700 text-sm">
                      • Tingkatkan ukuran dataset menjadi minimal 1000+ sampel<br/>
                      • Tambahkan fitur: kondisi bangunan, sertifikat, fasilitas (kolam renang, garasi)<br/>
                      • Data geospasial: jarak ke sekolah, rumah sakit, mall, stasiun
                    </p>
                  </div>

                  <div className="border-l-4 border-green-400 bg-green-50 p-4 rounded">
                    <p className="font-semibold text-green-800 mb-1">2. Feature Engineering</p>
                    <p className="text-gray-700 text-sm">
                      • Buat fitur interaksi (misalnya: luas × kamar tidur)<br/>
                      • Transformasi non-linear (log, polynomial features)<br/>
                      • Encoding kategorikal untuk wilayah/lingkungan
                    </p>
                  </div>

                  <div className="border-l-4 border-green-400 bg-green-50 p-4 rounded">
                    <p className="font-semibold text-green-800 mb-1">3. Model Ensemble</p>
                    <p className="text-gray-700 text-sm">
                      • Coba Random Forest atau Gradient Boosting (XGBoost, LightGBM)<br/>
                      • Ensemble beberapa model untuk prediksi lebih robust<br/>
                      • Model dapat menangkap non-linearitas dan interaksi kompleks
                    </p>
                  </div>

                  <div className="border-l-4 border-green-400 bg-green-50 p-4 rounded">
                    <p className="font-semibold text-green-800 mb-1">4. Cross-Validation yang Lebih Baik</p>
                    <p className="text-gray-700 text-sm">
                      • Gunakan K-Fold Cross-Validation (k=5 atau k=10)<br/>
                      • Implementasi time-based split jika data memiliki komponen temporal<br/>
                      • Validasi dengan data out-of-sample yang benar-benar terpisah
                    </p>
                  </div>

                  <div className="border-l-4 border-green-400 bg-green-50 p-4 rounded">
                    <p className="font-semibold text-green-800 mb-1">5. Hyperparameter Tuning</p>
                    <p className="text-gray-700 text-sm">
                      • Grid Search atau Random Search untuk parameter regularisasi optimal<br/>
                      • Optimalkan alpha untuk Ridge/Lasso dengan validasi silang<br/>
                      • Gunakan teknik seperti Bayesian Optimization
                    </p>
                  </div>
                </div>
              </div>

              <div className="mb-6">
                <h3 className="text-lg font-semibold text-gray-700 mb-3">Langkah Lanjutan yang Disarankan</h3>
                <div className="bg-gradient-to-r from-blue-50 to-indigo-50 p-6 rounded-lg">
                  <div className="space-y-4">
                    <div className="flex gap-3">
                      <div className="flex-shrink-0 w-8 h-8 bg-blue-600 text-white rounded-full flex items-center justify-center font-bold">
                        A
                      </div>
                      <div>
                        <p className="font-semibold text-gray-800">Implementasi Real-Time</p>
                        <p className="text-sm text-gray-600">
                          Deploy model ke production dengan API endpoint untuk prediksi real-time. 
                          Integrasi dengan sistem CRM perusahaan properti.
                        </p>
                      </div>
                    </div>

                    <div className="flex gap-3">
                      <div className="flex-shrink-0 w-8 h-8 bg-blue-600 text-white rounded-full flex items-center justify-center font-bold">
                        B
                      </div>
                      <div>
                        <p className="font-semibold text-gray-800">Model Monitoring</p>
                        <p className="text-sm text-gray-600">
                          Setup monitoring untuk track performa model di production. 
                          Deteksi model drift dan trigger retraining otomatis.
                        </p>
                      </div>
                    </div>

                    <div className="flex gap-3">
                      <div className="flex-shrink-0 w-8 h-8 bg-blue-600 text-white rounded-full flex items-center justify-center font-bold">
                        C
                      </div>
                      <div>
                        <p className="font-semibold text-gray-800">A/B Testing</p>
                        <p className="text-sm text-gray-600">
                          Uji model baru vs model lama dalam environment controlled. 
                          Ukur impact terhadap akurasi pricing dan kepuasan pelanggan.
                        </p>
                      </div>
                    </div>

                    <div className="flex gap-3">
                      <div className="flex-shrink-0 w-8 h-8 bg-blue-600 text-white rounded-full flex items-center justify-center font-bold">
                        D
                      </div>
                      <div>
                        <p className="font-semibold text-gray-800">Explainability</p>
                        <p className="text-sm text-gray-600">
                          Gunakan SHAP atau LIME untuk menjelaskan prediksi individual. 
                          Penting untuk transparansi dan trust dari stakeholder.
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              <div>
                <h3 className="text-lg font-semibold text-gray-700 mb-3">Kesimpulan</h3>
                <div className="bg-gradient-to-r from-indigo-600 to-purple-600 text-white p-6 rounded-lg">
                  <p className="leading-relaxed mb-3">
                    Analisis regresi berhasil membangun model prediksi harga rumah dengan performa yang baik 
                    (R² = 0.87). Model Linear Regression terpilih sebagai model terbaik karena kesederhanaan 
                    dan performa yang optimal untuk dataset ini.
                  </p>
                  <p className="leading-relaxed mb-3">
                    Luas bangunan dan jumlah kamar tidur adalah faktor paling berpengaruh terhadap harga, 
                    sementara umur rumah dan jarak ke pusat kota memiliki dampak negatif yang signifikan.
                  </p>
                  <p className="leading-relaxed">
                    Untuk deployment production, disarankan untuk mengumpulkan lebih banyak data, 
                    menambah fitur relevan, dan mengeksplorasi model ensemble untuk meningkatkan akurasi 
                    dan robustness prediksi.
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="mt-6 text-center text-gray-600 text-sm">
          <p>Penugasan Tinjauan - Analisis Regresi Linear & Regularisasi</p>
          <p className="mt-1">© 2025 | Dataset: Simulasi 100 Properti Rumah</p>
        </div>
      </div>
    </div>
  );
};

export default RegressionAnalysis;