# Veri Madenciliği Projesi: Yemek Teslimat Sürelerinin Sınıflandırılması

## 1. Kapak Sayfası
- **Proje Adı:** Yemek Teslimat Sürelerinin Makine Öğrenmesi ile Sınıflandırılması
- **Öğrenci:** [Öğrenci Adı ve Numarası]
- **Ders:** Veri Madenciliği
- **Teslim Tarihi:** [Tarih]

## 2. Giriş
### Proje Konusu ve Amacı
Bu proje, yemek teslimat hizmetlerinde teslimat sürelerini tahmin etmek ve sınıflandırmak amacıyla veri madenciliği tekniklerini kullanmaktadır. Proje, teslimat sürelerini 'Hızlı', 'Orta' ve 'Yavaş' kategorilerine ayırarak, hangi faktörlerin teslimat performansını etkilediğini belirlemeyi hedeflemektedir.

### Seçilen Veri Madenciliği Tekniği
Bu projede **Sınıflandırma (Classification)** tekniği kullanılmıştır. Teslimat süreleri kategorilere ayrılarak, çeşitli faktörlerin (hava durumu, trafik, mesafe vb.) teslimat süresini nasıl etkilediği tahmin edilmektedir.

### Kısa Yöntem Özeti
1. Veri seti yükleme ve keşifsel veri analizi
2. Eksik veri temizleme ve ön işleme
3. Kategorik değişkenlerin sayısallaştırılması
4. Teslimat sürelerinin kategorilere ayrılması
5. Farklı sınıflandırma algoritmalarının uygulanması
6. Model performanslarının karşılaştırılması
7. İşletme için içgörüler ve öneriler

## 3. Veri Seti Tanıtımı
### Veri Seti Kaynağı
Veri seti, yemek teslimat hizmetlerinden toplanan gerçek verilerden oluşmaktadır.

### Veri Seti Açıklaması
Veri seti, yemek teslimat süreçlerini etkileyen çeşitli faktörleri içermektedir. Toplam 1001 sipariş kaydı bulunmaktadır.

### Veri Seti Alanları
- **Order_ID:** Sipariş numarası
- **Distance_km:** Teslimat mesafesi (kilometre)
- **Weather:** Hava durumu (Clear, Rainy, Snowy, Foggy, Windy)
- **Traffic_Level:** Trafik seviyesi (Low, Medium, High)
- **Time_of_Day:** Günün saati (Morning, Afternoon, Evening, Night)
- **Vehicle_Type:** Araç tipi (Bike, Scooter, Car)
- **Preparation_Time_min:** Hazırlık süresi (dakika)
- **Courier_Experience_yrs:** Kurye deneyimi (yıl)
- **Delivery_Time_min:** Teslimat süresi (dakika) - Hedef değişken

### Veri Seti Boyutu
- Satır sayısı: 1000
- Sütun sayısı: 15

## 4. Veri Ön İşleme
### Eksik Veri İşleme
- Weather: En sık görülen değer ile dolduruldu
- Traffic_Level: En sık görülen değer ile dolduruldu
- Time_of_Day: En sık görülen değer ile dolduruldu
- Courier_Experience_yrs: Medyan değer ile dolduruldu

### Kategorik Veri Dönüştürme
- Tüm kategorik değişkenler LabelEncoder ile sayısallaştırıldı
- Teslimat süreleri kategorilere ayrıldı (Hızlı: 0-40 dk, Orta: 40-60 dk, Yavaş: 60+ dk)

### Normalizasyon
- Sayısal değişkenler StandardScaler ile normalize edildi
- Eğitim ve test setleri %80-%20 oranında ayrıldı

## 5. Yöntem ve Uygulama
### Kullanılan Algoritmalar
1. **Decision Tree (Karar Ağacı)**
2. **Random Forest (Rastgele Orman)**
3. **K-Nearest Neighbors (K-En Yakın Komşu)**
4. **Naive Bayes**
5. **Gradient Boosting**

### Eğitim/Test Ayrımı
- Eğitim seti: %80 (800 örnek)
- Test seti: %20 (201 örnek)
- Stratified sampling kullanıldı

### Başarı Ölçütleri
- **Accuracy (Doğruluk):** Doğru tahmin edilen örneklerin oranı
- **Precision (Kesinlik):** Pozitif tahminlerin doğruluk oranı
- **Recall (Duyarlılık):** Gerçek pozitiflerin yakalanma oranı
- **F1-Score:** Precision ve Recall'un harmonik ortalaması

### Model Performans Sonuçları
                  accuracy precision recall  f1_score
Decision Tree        0.695  0.705836  0.695  0.698201
Random Forest         0.74   0.74126   0.74  0.740243
K-NN                 0.655  0.670077  0.655  0.659802
Naive Bayes          0.725  0.726309  0.725  0.725457
Gradient Boosting     0.73  0.731305   0.73  0.730572

## 6. Sonuç ve Yorumlar
### Elde Edilen Bulgular
1. **Hava durumu** teslimat sürelerini önemli ölçüde etkilemektedir
2. **Trafik seviyesi** yüksek olduğunda teslimat süreleri artmaktadır
3. **Araç tipi** mesafeye göre optimize edilmelidir
4. **Kurye deneyimi** teslimat performansını etkilemektedir

### Uygulamanın Güçlü Yönleri
- Çoklu algoritma karşılaştırması
- Kapsamlı veri ön işleme
- İşletme odaklı içgörüler
- Görsel analiz ve raporlama

### Geliştirilebilecek Yönler
- Daha fazla veri toplanması
- Hiperparametre optimizasyonu
- Derin öğrenme modellerinin denenmesi
- Gerçek zamanlı tahmin sistemi

### İşletme Yorumları
Bu analiz sonuçları, yemek teslimat hizmetlerinde:
- Operasyonel verimliliği artırmak
- Müşteri memnuniyetini yükseltmek
- Maliyetleri optimize etmek
- Kaynak planlamasını iyileştirmek
için kullanılabilir.

## 7. Kaynakça
1. Scikit-learn Documentation: https://scikit-learn.org/
2. Pandas Documentation: https://pandas.pydata.org/
3. Matplotlib Documentation: https://matplotlib.org/
4. Seaborn Documentation: https://seaborn.pydata.org/
5. Veri Madenciliği: Kavramlar ve Teknikler - Jiawei Han

## 8. Ekler
- Python kodu: veri_madenciligi_projesi.py
- Veri seti: Food_Delivery_Times.csv
- Grafik dosyaları: veri_analizi.png, model_karsilastirma.png, ozellik_onem.png
- Rapor: veri_madenciligi_raporu.md
