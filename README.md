# 🚴‍♂️ Yemek Teslimat Sürelerinin Makine Öğrenmesi ile Sınıflandırılması

Bu proje, yemek teslimat hizmetlerinde teslimat sürelerini doğru şekilde sınıflandırmak amacıyla çeşitli makine öğrenimi algoritmalarının uygulanmasını kapsamaktadır. Teslimat süreleri **Hızlı**, **Orta** ve **Yavaş** olarak üç gruba ayrılmış; bu sınıflandırmayı etkileyen çevresel ve operasyonel faktörler analiz edilmiştir.

## 📌 Proje Amacı

Müşteri memnuniyetini artırmak ve operasyonel verimliliği iyileştirmek amacıyla, teslimat sürelerinin doğru tahmin edilmesi hedeflenmiştir. Bu kapsamda sınıflandırma modelleri ile:

- Teslimatın süresi tahmin edilmiş,
- En uygun algoritma seçilmiş,
- İşletmeler için iyileştirici öneriler sunulmuştur.

## 📊 Kullanılan Veri Seti

Veri seti, yemek teslimat hizmetlerinden toplanan gerçek verilere dayanmaktadır ve şu sütunları içermektedir:

- `Order_ID` – Sipariş kimliği
- `Distance_km` – Teslimat mesafesi
- `Weather` – Hava durumu
- `Traffic_Level` – Trafik yoğunluğu
- `Time_of_Day` – Günün saati
- `Vehicle_Type` – Kurye aracı
- `Preparation_Time_min` – Hazırlık süresi
- `Courier_Experience_yrs` – Kurye deneyimi
- `Delivery_Time_min` – Teslimat süresi (hedef değişken)

## ⚙️ Yöntem

### Veri Ön İşleme
- Eksik veriler dolduruldu (mod/medyan ile)
- Aykırı değerler belirlendi ve sınırlandırıldı
- Kategorik veriler `LabelEncoder` ile sayısallaştırıldı
- Sayısal veriler `StandardScaler` ile normalize edildi
- `Delivery_Time_min`, üç kategoriye bölündü:
  - `Hızlı`: 0–40 dk
  - `Orta`: 40–60 dk
  - `Yavaş`: 60+ dk

### Kullanılan Algoritmalar
- Decision Tree
- Random Forest
- K-Nearest Neighbors (KNN)
- Naive Bayes
- Gradient Boosting

### Değerlendirme Metrikleri
- Accuracy
- Precision
- Recall
- F1-Score

## 🏆 Sonuçlar

| Model             | Doğruluk (Accuracy) |
|------------------|---------------------|
| ✅ Random Forest | %74.00              |
| Gradient Boosting| %73.00              |
| Naive Bayes      | %72.50              |
| Decision Tree    | %69.50              |
| KNN              | %65.50              |

**Random Forest**, özellikle “Hızlı” ve “Yavaş” kategorilerinde yüksek kesinlik ve duyarlılıkla en iyi performansı göstermiştir.

## 💡 İşletmeye Yönelik İçgörüler

- **Hava durumu** ve **trafik seviyesi** teslimat süresini önemli ölçüde etkiler.
- **Kurye deneyimi** ve **araç tipi** daha hızlı teslimat ile ilişkilidir.
- **Hazırlık süresi**, toplam teslimat süresini uzatabilir.

## 🔧 Kullanılan Kütüphaneler

- [Scikit-learn](https://scikit-learn.org/)
- [Pandas](https://pandas.pydata.org/)
- [Matplotlib](https://matplotlib.org/)
- [Seaborn](https://seaborn.pydata.org/)

## 📁 Proje Yapısı

