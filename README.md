# Recommendation-Systems
# Collaborative Filtering: User-Based vs Item-Based

**BSM585 Öneri Sistemleri - Ödev 1**  
**User-Based (PCC) ve Item-Based (ACS) Yöntemlerin Karşılaştırmalı Analizi**

![Python](https://img.shields.io/badge/Python-3.x-blue)
![NumPy](https://img.shields.io/badge/NumPy-1.x-orange)
![Pandas](https://img.shields.io/badge/Pandas-2.x-green)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.x-yellow)

## 📝 Proje Açıklaması
Bu proje, **MovieLens 100K** veri seti üzerinde **User-Based Collaborative Filtering (Pearson Korelasyon Katsayısı - PCC)** ve **Item-Based Collaborative Filtering (Adjusted Cosine Similarity - ACS)** yöntemlerini **sıfırdan** (Surprise kütüphanesi kullanılmadan) implemente ederek karşılaştırmalı analiz etmektedir.

Leave-One-Out Cross-Validation (seed=42) ile k = {10, 25, 50, 100} değerleri için MAE metriği ve hesaplama süreleri ölçülmüş, ayrıca %20 veri silme senaryosu ile sparsity etkisi incelenmiştir.

**Sonuç:** Item-Based CF (k=10), User-Based CF’ye göre hem daha düşük MAE (**0.5853**) hem de **yaklaşık 90 kat daha hızlı** performans göstermiştir.

## ✨ Özellikler
- PCC ve ACS benzerlik metrikleri NumPy/Pandas ile sıfırdan yazıldı
- Leave-One-Out Cross-Validation (tam ödev protokolü)
- k = {10, 25, 50, 100} için kapsamlı deneyler
- MAE + hesaplama süresi karşılaştırması
- %20 sparsity artışı analizi
- Tüm grafikler ve tablolar otomatik üretilir

## 🛠️ Kullanılan Teknolojiler
- **Python 3.x**
- NumPy, Pandas, Matplotlib
- Google Colab (önerilen) / Yerel ortam

## 📊 Veri Seti
- **MovieLens 100K** (Grouplens)
- 943 kullanıcı, 1682 film, ~100.000 oy
- Sparsity: %93.70
