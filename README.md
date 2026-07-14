# BSM585 – Öneri Sistemleri (Recommendation Systems)

Bilgisayar Mühendisliği Yüksek Lisans, Öneri Sistemleri kapsamında hazırlanan projeler ve sunumların tutulduğu depo. Süreç boyunca **2 uygulama projesi** ve **2 sunum** hazırlanmıştır.

---

## İçindekiler

| # | Tür | Başlık |  
|---|---|---|---|---|
| 1 | Work-1 | Collaborative Filtering: User-Based vs Item-Based | Klasik CF yöntemlerinin sıfırdan implementasyonu ve karşılaştırılması | 
| 2 | Presentation-1 | Graph Recommendation Evolution | Öneri sistemlerinin matris çarpanlarına ayırmadan graf/GNN tabanlı yöntemlere evrimi | 
| 3 | Work-2 | IGMC ile Graph-Based Matrix Completion | Graf sinir ağı tabanlı inductive öneri sistemi ve MF baseline karşılaştırması | 
| 4 | Pre-2/P2 | Pinterest Graph Recommendation Architecture | Pinterest'in büyük ölçekli graf tabanlı öneri mimarisinin incelenmesi |

---

## 1. Proje: Collaborative Filtering — User-Based vs Item-Based (Ödev 1)

MovieLens-100K veri seti üzerinde iki klasik Collaborative Filtering (CF) yöntemi sıfırdan (NumPy/Pandas ile) geliştirilmiş ve karşılaştırılmıştır:

- **User-Based CF** — Pearson Korelasyon Katsayısı (PCC) ile kullanıcı benzerliği
- **Item-Based CF** — Adjusted Cosine Similarity (ACS) ile öğe benzerliği

### Kapsam

- Veri ön işleme: mükerrer (user_id, item_id) çiftlerinin tespiti/temizlenmesi, eksik değer (NaN) raporlaması, sparsity hesaplama (`sparsity = 1 − |R| / (|U|×|I|)`)
- `compute_pcc()` ve `compute_acs()` fonksiyonlarının sıfırdan implementasyonu
- Leave-One-Out Cross-Validation (seed=42) protokolü ile MAE hesaplama
- k = {10, 25, 50, 100} komşuluk boyutları için karşılaştırmalı deneyler ve çalışma süresi ölçümü
- %20 veri silme senaryosu ile sparsity artışının model performansına etkisinin analizi
- En kötü performans gösteren kullanıcıların incelenmesi, zaman karmaşıklığı analizi ve production ortamı için yöntem önerisi

### Veri Seti

| Özellik | Değer |
|---|---|
| Kullanıcı sayısı | 943 |
| Film sayısı | 1.682 |
| Rating sayısı | ~100.000 |
| Sparsity | %93.70 |

### Sonuçlar

**k Değeri Karşılaştırması**

| Metrik | k=10 | k=25 | k=50 | k=100 |
|---|---|---|---|---|
| User-Based MAE | 0.7811 | 0.7529 | 0.7457 | 0.7456 |
| Item-Based MAE | 0.6198 | 0.6325 | 0.6397 | 0.6465 |
| User-Based Süre (sn) | 172.60 | 174.49 | 180.43 | 187.27 |
| Item-Based Süre (sn) | 2.55 | 4.10 | 3.34 | 3.52 |

- User-Based CF'de k arttıkça MAE azalır (en iyi sonuç k=100: 0.7456); Item-Based CF'de ise k arttıkça MAE artar (en iyi sonuç k=10: 0.6198).
- Item-Based CF, hem doğruluk hem de hız açısından belirgin şekilde üstündür (943 kullanıcıya karşı 1.682 film sayısı ve önceden hesaplanıp önbelleklenen item-item benzerlik matrisi sayesinde).

**Sparsity (%20 Veri Silme) Senaryosu**

| Metrik | Orijinal | Seyrek (%20 silinmiş) | Değişim |
|---|---|---|---|
| Sparsity | %93.70 | %95.97 | — |
| User-Based MAE (k=100) | 0.7456 | 0.7567 | %+1.49 |
| Item-Based MAE (k=10) | 0.6198 | 0.5791 | %−6.56 |

- Veri eksikliği User-Based CF'yi bilgi kaybı nedeniyle zayıflatırken, Item-Based CF üzerinde beklenmedik biçimde iyileştirici etki göstermiştir; bu durum Item-Based yöntemin yüksek sparsity karşısındaki göreli sağlamlığına (robustness) işaret etmektedir.

**Genel Sonuç:** Doğruluk, maliyet ve ölçeklenebilirlik açısından gerçek bir üretim sisteminde **Item-Based CF** tercih edilmelidir.
---

## 2. Proje: IGMC ile Graph-Based Matrix Completion (Final Ödev)

**Referans Makale:** Zhang, M. & Chen, Y. (2020). *Inductive Matrix Completion Based on Graph Neural Networks.* ICLR 2020.

Bu projede, kullanıcı-öğe etkileşimlerini iki parçalı graf olarak modelleyen ve hedef çift etrafındaki **1-hop enclosing subgraph**'ı R-GCN ile işleyerek rating tahmini yapan IGMC yöntemi MovieLens-100K üzerinde sıfırdan (PyTorch) uygulanmış, NumPy tabanlı Matrix Factorization (Funk-SVD) baseline ile karşılaştırılmıştır.

### Kapsam

- Enclosing subgraph çıkarımı (BFS), 4 sınıflı node labeling, rating-tipi bazlı R-GCN mesaj iletimi (basis-decomposition ile)
- `MatrixFactorizationSGD`, `SparseRGCNLayer`, `extract_and_build_sparse_subgraph()`, `IGMC_Model` sınıflarının sıfırdan implementasyonu
- 5-fold cross-validation, 3 farklı öğrenme oranı denemesi (0.001 / 0.005 / 0.01), 2 ablation varyasyonu (w/o ARR, w/o Node Labeling)
- Alt graf önbellekleme (full cache) ve erken durdurma (patience=2) ile hız optimizasyonu

### Sonuçlar (Özet)

| Yöntem | RMSE | MAE |
|---|---|---|
| MF Baseline (Funk-SVD) | 1.2489 | 0.9796 |
| **IGMC (5-fold ort.)** | **1.0099 ± 0.0096** | **0.8126** |
| IGMC — Makale (Zhang & Chen, 2020) | 0.905 | — |

- IGMC, MF baseline'a göre RMSE'de ~%19.1 iyileşme sağlamıştır.
- Ablation sonucunda **node labeling** bileşeninin kritik önemde olduğu; kaldırıldığında RMSE'nin 0.9946'dan 1.0160'a yükseldiği görülmüştür.
- Uygulama sonucu, orijinal makaledeki RMSE'den (0.905) yüksektir; bunun temel nedenleri %30 stratified örneklem kullanımı, maksimum 5 epoch sınırı ve sadeleştirilmiş node labeling/ARR uygulamasıdır.

---

## 3. Sunum: Graph Recommendation Evolution

**Dosya:** `Graph_Recommendation_Evolution.pptx`

Öneri sistemlerinin zaman içindeki gelişimini ele alan sunum; klasik matris çarpanlarına ayırma (Matrix Factorization) ve komşuluk tabanlı Collaborative Filtering yöntemlerinden başlayarak, derin öğrenme tabanlı yaklaşımlara ve ardından **graf sinir ağları (GNN)** temelli öneri mimarilerine (GraphSAGE, PinSage, IGMC gibi) uzanan evrimi; son olarak da büyük ölçekli, uzun kullanıcı davranış dizilerini işleyen (özel C++ çıkarım motoru ve CUDA çekirdekleri ile optimize edilmiş) güncel nesil öneri modellerine geçişi tarihsel bir perspektifle sunar.

Bu sunum, IGMC projesiyle doğrudan bağlantılıdır: IGMC, sunumda ele alınan graf tabanlı öneri sistemleri evriminin inductive/lokal-graf temsilli bir örneğini oluşturur.

---

## 4. Sunum: Pinterest Graph Recommendation Architecture

**Dosya:** `Pinterest_Graph_Recommendation_Architecture.pptx`

Pinterest'in üretim ortamında kullandığı büyük ölçekli, graf tabanlı öneri sistemi mimarisini (PinSage / GraphSAGE tabanlı temsil öğrenme, random-walk örnekleme, üretim düzeyinde ölçeklenebilirlik ve çıkarım optimizasyonları) inceleyen sunum. Akademik IGMC/GNN yaklaşımlarının endüstride nasıl büyük ölçekte üretime alındığına dair pratik bir bakış açısı sağlar.

---

## Genel Değerlendirme

Bu dört çalışma birlikte, öneri sistemleri alanının üç temel katmanını kapsayacak şekilde tasarlanmıştır:

1. **Klasik yöntemler** — komşuluk tabanlı Collaborative Filtering (Ödev 1)
2. **Graf sinir ağı tabanlı akademik yöntemler** — IGMC (Final Ödev)
3. **Kavramsal/endüstriyel bağlam** — graf tabanlı öneri sistemlerinin evrimi ve Pinterest'in üretim mimarisi (Sunumlar)

Ortak veri seti olarak her iki projede de **MovieLens-100K** kullanılmıştır; bu sayede klasik CF ve graf tabanlı yöntemlerin doğrudan karşılaştırılması mümkün olmuştur.

---

## Repo Yapısı

```
.
├── Work-1                                             # Ödev 1 raporu (CF: User-Based vs Item-Based)
│   ├── AssignmentGuidlines.pdf
│   ├── Final Reports.pdf                                     
│   ├── item_sim_matrix.pkl                            
│   ├── ÖneriSis.ipynb                                 # Ödev 1 kodu  
├── Woek-2                                             # Ödev 1 kodu
│   ├── 1904.12058v3.pdf                               # Referans Makale
│   ├── FinalWork_Guidlines2026.pdf                    # Ödev 1 kodu
│   ├── IGMC_Final_Doc.pdf                             # Final Ödev raporu (IGMC)
|   ├── IGMC-Rec-Sys_2.ipynb                           # Final Ödev kodu
├── Pre-1
│   ├── P-1GraphRecommenddation_Evolution.pptx
│   ├── Pre-Guidline.docx
│   ├── Rec-Sys-Evolution.docx
├── P2_Pinterest_Oneri_Sistemi_Analizi.docx
├── P2-Pinterest_Graph_Recommendation_Architecture.pptx   # Sunum 2
├── Presentation-2Guide.pdf
└── README.md                                          # Bu dosya (ana / özet README)
```

## Gereksinimler

```bash
pip install numpy pandas scikit-learn torch tqdm matplotlib
```

Not defterleri Google Colab için hazırlanmıştır; GPU (T4) yalnızca IGMC projesinde önerilir, CF projesi CPU üzerinde de hızlıca çalışır.
