#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Veri Madenciliği Projesi: Yemek Teslimat Sürelerinin Sınıflandırılması
Öğrenci: [Öğrenci Adı ve Numarası]
Ders: Veri Madenciliği
Tarih: [Teslim Tarihi]
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# Türkçe karakter desteği için
plt.rcParams['font.family'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class YemekTeslimatAnalizi:
    def __init__(self, dosya_yolu='Food_Delivery_Times.csv'):
        """
        Yemek teslimat analizi sınıfı başlatıcısı
        """
        self.dosya_yolu = dosya_yolu
        self.veri = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.modeller = {}
        self.sonuclar = {}
        
    def veri_yukle(self):
        """
        Veri setini yükler ve temel bilgileri gösterir
        """
        print("="*60)
        print("1. VERİ SETİ YÜKLEME VE İNCELEME")
        print("="*60)
        
        self.veri = pd.read_csv(self.dosya_yolu)
        
        print(f"Veri seti boyutu: {self.veri.shape}")
        print(f"Satır sayısı: {self.veri.shape[0]}")
        print(f"Sütun sayısı: {self.veri.shape[1]}")
        print("\nVeri seti sütunları:")
        for i, col in enumerate(self.veri.columns, 1):
            print(f"{i}. {col}")
        
        print("\nİlk 5 satır:")
        print(self.veri.head())
        
        print("\nVeri seti istatistikleri:")
        print(self.veri.describe())
        
        print("\nVeri tipleri:")
        print(self.veri.dtypes)
        
        print("\nEksik veriler:")
        print(self.veri.isnull().sum())
        
        return self.veri
    
    def veri_on_isleme(self):
        """
        Veri ön işleme adımları
        """
        print("\n" + "="*60)
        print("2. VERİ ÖN İŞLEME")
        print("="*60)
        
        # Eksik verileri incele
        print("Eksik veri analizi:")
        eksik_veriler = self.veri.isnull().sum()
        print(eksik_veriler[eksik_veriler > 0])
        
        # Eksik verileri doldur
        print("\nEksik veriler dolduruluyor...")
        
        # Weather için en sık değeri kullan
        if self.veri['Weather'].isnull().sum() > 0:
            en_sik_hava = self.veri['Weather'].mode()[0]
            self.veri['Weather'].fillna(en_sik_hava, inplace=True)
            print(f"Weather eksik veriler '{en_sik_hava}' ile dolduruldu")
        
        # Traffic_Level için en sık değeri kullan
        if self.veri['Traffic_Level'].isnull().sum() > 0:
            en_sik_trafik = self.veri['Traffic_Level'].mode()[0]
            self.veri['Traffic_Level'].fillna(en_sik_trafik, inplace=True)
            print(f"Traffic_Level eksik veriler '{en_sik_trafik}' ile dolduruldu")
        
        # Time_of_Day için en sık değeri kullan
        if self.veri['Time_of_Day'].isnull().sum() > 0:
            en_sik_zaman = self.veri['Time_of_Day'].mode()[0]
            self.veri['Time_of_Day'].fillna(en_sik_zaman, inplace=True)
            print(f"Time_of_Day eksik veriler '{en_sik_zaman}' ile dolduruldu")
        
        # Courier_Experience_yrs için medyan değeri kullan
        if self.veri['Courier_Experience_yrs'].isnull().sum() > 0:
            medyan_deneyim = self.veri['Courier_Experience_yrs'].median()
            self.veri['Courier_Experience_yrs'].fillna(medyan_deneyim, inplace=True)
            print(f"Courier_Experience_yrs eksik veriler {medyan_deneyim} ile dolduruldu")
        
        print(f"\nEksik veri kontrolü (doldurma sonrası):")
        print(self.veri.isnull().sum().sum())
        
        # Teslimat süresini kategorilere ayır
        print("\nTeslimat süresi kategorilere ayrılıyor...")
        self.veri['Delivery_Category'] = pd.cut(
            self.veri['Delivery_Time_min'],
            bins=[0, 40, 60, 200],
            labels=['Hızlı', 'Orta', 'Yavaş']
        )
        
        print("Teslimat kategorileri:")
        print(self.veri['Delivery_Category'].value_counts())
        
        # Kategorik değişkenleri sayısallaştır
        print("\nKategorik değişkenler sayısallaştırılıyor...")
        le_weather = LabelEncoder()
        le_traffic = LabelEncoder()
        le_time = LabelEncoder()
        le_vehicle = LabelEncoder()
        le_category = LabelEncoder()
        
        self.veri['Weather_Encoded'] = le_weather.fit_transform(self.veri['Weather'])
        self.veri['Traffic_Encoded'] = le_traffic.fit_transform(self.veri['Traffic_Level'])
        self.veri['Time_Encoded'] = le_time.fit_transform(self.veri['Time_of_Day'])
        self.veri['Vehicle_Encoded'] = le_vehicle.fit_transform(self.veri['Vehicle_Type'])
        self.veri['Category_Encoded'] = le_category.fit_transform(self.veri['Delivery_Category'])
        
        print("Kodlama tamamlandı")
        
        # Özellik seçimi
        self.X = self.veri[['Distance_km', 'Weather_Encoded', 'Traffic_Encoded', 
                           'Time_Encoded', 'Vehicle_Encoded', 'Preparation_Time_min', 
                           'Courier_Experience_yrs']]
        self.y = self.veri['Category_Encoded']
        
        print(f"\nÖzellik matrisi boyutu: {self.X.shape}")
        print(f"Hedef değişken boyutu: {self.y.shape}")
        
        return self.X, self.y
    
    def veri_gorselleştirme(self):
        """
        Veri görselleştirme ve analiz
        """
        print("\n" + "="*60)
        print("3. VERİ GÖRSELLEŞTİRME VE ANALİZ")
        print("="*60)
        
        # Grafik boyutunu ayarla
        plt.figure(figsize=(20, 15))
        
        # 1. Teslimat süresi dağılımı
        plt.subplot(3, 4, 1)
        plt.hist(self.veri['Delivery_Time_min'], bins=30, alpha=0.7, color='skyblue')
        plt.title('Teslimat Süresi Dağılımı')
        plt.xlabel('Teslimat Süresi (dakika)')
        plt.ylabel('Frekans')
        
        # 2. Mesafe dağılımı
        plt.subplot(3, 4, 2)
        plt.hist(self.veri['Distance_km'], bins=30, alpha=0.7, color='lightgreen')
        plt.title('Mesafe Dağılımı')
        plt.xlabel('Mesafe (km)')
        plt.ylabel('Frekans')
        
        # 3. Hava durumu vs teslimat süresi
        plt.subplot(3, 4, 3)
        self.veri.boxplot(column='Delivery_Time_min', by='Weather', ax=plt.gca())
        plt.title('Hava Durumu vs Teslimat Süresi')
        plt.suptitle('')
        
        # 4. Trafik seviyesi vs teslimat süresi
        plt.subplot(3, 4, 4)
        self.veri.boxplot(column='Delivery_Time_min', by='Traffic_Level', ax=plt.gca())
        plt.title('Trafik Seviyesi vs Teslimat Süresi')
        plt.suptitle('')
        
        # 5. Araç tipi vs teslimat süresi
        plt.subplot(3, 4, 5)
        self.veri.boxplot(column='Delivery_Time_min', by='Vehicle_Type', ax=plt.gca())
        plt.title('Araç Tipi vs Teslimat Süresi')
        plt.suptitle('')
        
        # 6. Günün saati vs teslimat süresi
        plt.subplot(3, 4, 6)
        self.veri.boxplot(column='Delivery_Time_min', by='Time_of_Day', ax=plt.gca())
        plt.title('Günün Saati vs Teslimat Süresi')
        plt.suptitle('')
        
        # 7. Kurye deneyimi vs teslimat süresi
        plt.subplot(3, 4, 7)
        plt.scatter(self.veri['Courier_Experience_yrs'], self.veri['Delivery_Time_min'], alpha=0.6)
        plt.title('Kurye Deneyimi vs Teslimat Süresi')
        plt.xlabel('Deneyim (yıl)')
        plt.ylabel('Teslimat Süresi (dakika)')
        
        # 8. Hazırlık süresi vs teslimat süresi
        plt.subplot(3, 4, 8)
        plt.scatter(self.veri['Preparation_Time_min'], self.veri['Delivery_Time_min'], alpha=0.6)
        plt.title('Hazırlık Süresi vs Teslimat Süresi')
        plt.xlabel('Hazırlık Süresi (dakika)')
        plt.ylabel('Teslimat Süresi (dakika)')
        
        # 9. Mesafe vs teslimat süresi
        plt.subplot(3, 4, 9)
        plt.scatter(self.veri['Distance_km'], self.veri['Delivery_Time_min'], alpha=0.6)
        plt.title('Mesafe vs Teslimat Süresi')
        plt.xlabel('Mesafe (km)')
        plt.ylabel('Teslimat Süresi (dakika)')
        
        # 10. Korelasyon matrisi
        plt.subplot(3, 4, 10)
        numeric_cols = ['Distance_km', 'Preparation_Time_min', 'Courier_Experience_yrs', 'Delivery_Time_min']
        correlation_matrix = self.veri[numeric_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Korelasyon Matrisi')
        
        # 11. Teslimat kategorileri dağılımı
        plt.subplot(3, 4, 11)
        self.veri['Delivery_Category'].value_counts().plot(kind='bar', color='orange')
        plt.title('Teslimat Kategorileri Dağılımı')
        plt.xlabel('Kategori')
        plt.ylabel('Sayı')
        plt.xticks(rotation=45)
        
        # 12. Aykırı değer analizi
        plt.subplot(3, 4, 12)
        plt.boxplot(self.veri['Delivery_Time_min'])
        plt.title('Teslimat Süresi Aykırı Değer Analizi')
        plt.ylabel('Teslimat Süresi (dakika)')
        
        plt.tight_layout()
        plt.savefig('veri_analizi.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Veri analizi grafikleri 'veri_analizi.png' dosyasına kaydedildi.")
    
    def model_egitimi(self):
        """
        Farklı sınıflandırma modellerini eğitir
        """
        print("\n" + "="*60)
        print("4. MODEL EĞİTİMİ")
        print("="*60)
        
        # Veriyi eğitim ve test setlerine ayır
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        print(f"Eğitim seti boyutu: {self.X_train.shape}")
        print(f"Test seti boyutu: {self.X_test.shape}")
        
        # Veriyi normalize et
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # Modelleri tanımla
        modeller = {
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'K-NN': KNeighborsClassifier(n_neighbors=5),
            'Naive Bayes': GaussianNB(),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42)
        }
        
        # Her modeli eğit ve değerlendir
        for isim, model in modeller.items():
            print(f"\n{isim} modeli eğitiliyor...")
            
            if isim == 'K-NN':
                model.fit(self.X_train_scaled, self.y_train)
                y_pred = model.predict(self.X_test_scaled)
            else:
                model.fit(self.X_train, self.y_train)
                y_pred = model.predict(self.X_test)
            
            # Performans metrikleri
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, average='weighted')
            recall = recall_score(self.y_test, y_pred, average='weighted')
            f1 = f1_score(self.y_test, y_pred, average='weighted')
            
            self.modeller[isim] = model
            self.sonuclar[isim] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'predictions': y_pred
            }
            
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-Score: {f1:.4f}")
        
        return self.modeller, self.sonuclar
    
    def model_karsilastirma(self):
        """
        Modellerin performansını karşılaştırır
        """
        print("\n" + "="*60)
        print("5. MODEL KARŞILAŞTIRMASI")
        print("="*60)
        
        # Sonuçları DataFrame'e çevir
        sonuclar_df = pd.DataFrame(self.sonuclar).T
        sonuclar_df = sonuclar_df.drop('predictions', axis=1)
        
        print("Model Performans Karşılaştırması:")
        print(sonuclar_df.round(4))
        
        # En iyi modeli bul
        en_iyi_model = sonuclar_df['f1_score'].idxmax()
        print(f"\nEn iyi performans gösteren model: {en_iyi_model}")
        print(f"F1-Score: {sonuclar_df.loc[en_iyi_model, 'f1_score']:.4f}")
        
        # Performans grafiği
        plt.figure(figsize=(12, 8))
        
        # Metrik karşılaştırması
        plt.subplot(2, 2, 1)
        sonuclar_df[['accuracy', 'precision', 'recall', 'f1_score']].plot(kind='bar', ax=plt.gca())
        plt.title('Model Performans Metrikleri')
        plt.xlabel('Model')
        plt.ylabel('Skor')
        plt.xticks(rotation=45)
        plt.legend()
        
        # Confusion matrix (en iyi model için)
        plt.subplot(2, 2, 2)
        en_iyi_tahminler = self.sonuclar[en_iyi_model]['predictions']
        cm = confusion_matrix(self.y_test, en_iyi_tahminler)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Hızlı', 'Orta', 'Yavaş'],
                   yticklabels=['Hızlı', 'Orta', 'Yavaş'])
        plt.title(f'Confusion Matrix - {en_iyi_model}')
        plt.xlabel('Tahmin')
        plt.ylabel('Gerçek')
        
        # F1-Score karşılaştırması
        plt.subplot(2, 2, 3)
        plt.bar(sonuclar_df.index, sonuclar_df['f1_score'], color='lightcoral')
        plt.title('F1-Score Karşılaştırması')
        plt.xlabel('Model')
        plt.ylabel('F1-Score')
        plt.xticks(rotation=45)
        
        # Accuracy karşılaştırması
        plt.subplot(2, 2, 4)
        plt.bar(sonuclar_df.index, sonuclar_df['accuracy'], color='lightblue')
        plt.title('Accuracy Karşılaştırması')
        plt.xlabel('Model')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('model_karsilastirma.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Model karşılaştırma grafikleri 'model_karsilastirma.png' dosyasına kaydedildi.")
        
        return sonuclar_df
    
    def detayli_rapor(self):
        """
        Detaylı sınıflandırma raporu oluşturur
        """
        print("\n" + "="*60)
        print("6. DETAYLI SINIFLANDIRMA RAPORU")
        print("="*60)
        
        # En iyi model için detaylı rapor
        en_iyi_model = max(self.sonuclar.keys(), key=lambda x: self.sonuclar[x]['f1_score'])
        
        print(f"En iyi model: {en_iyi_model}")
        print("\nDetaylı sınıflandırma raporu:")
        print(classification_report(self.y_test, self.sonuclar[en_iyi_model]['predictions'], 
                                  target_names=['Hızlı', 'Orta', 'Yavaş']))
        
        # Özellik önem dereceleri (Random Forest için)
        if en_iyi_model == 'Random Forest':
            rf_model = self.modeller['Random Forest']
            feature_importance = pd.DataFrame({
                'feature': self.X.columns,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\nÖzellik önem dereceleri:")
            print(feature_importance)
            
            # Özellik önem grafiği
            plt.figure(figsize=(10, 6))
            plt.barh(feature_importance['feature'], feature_importance['importance'])
            plt.title('Özellik Önem Dereceleri (Random Forest)')
            plt.xlabel('Önem Derecesi')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig('ozellik_onem.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print("Özellik önem grafiği 'ozellik_onem.png' dosyasına kaydedildi.")
    
    def is_insights(self):
        """
        İşletme için içgörüler ve öneriler
        """
        print("\n" + "="*60)
        print("7. İŞLETME İÇGÖRÜLERİ VE ÖNERİLER")
        print("="*60)
        
        print("Veri analizi sonucunda elde edilen önemli bulgular:")
        
        # Hava durumu etkisi
        print("\n1. Hava Durumu Etkisi:")
        hava_etkisi = self.veri.groupby('Weather')['Delivery_Time_min'].mean().sort_values(ascending=False)
        print(hava_etkisi)
        
        # Trafik seviyesi etkisi
        print("\n2. Trafik Seviyesi Etkisi:")
        trafik_etkisi = self.veri.groupby('Traffic_Level')['Delivery_Time_min'].mean().sort_values(ascending=False)
        print(trafik_etkisi)
        
        # Araç tipi etkisi
        print("\n3. Araç Tipi Etkisi:")
        arac_etkisi = self.veri.groupby('Vehicle_Type')['Delivery_Time_min'].mean().sort_values(ascending=False)
        print(arac_etkisi)
        
        # Günün saati etkisi
        print("\n4. Günün Saati Etkisi:")
        zaman_etkisi = self.veri.groupby('Time_of_Day')['Delivery_Time_min'].mean().sort_values(ascending=False)
        print(zaman_etkisi)
        
        # Kurye deneyimi etkisi
        print("\n5. Kurye Deneyimi Korelasyonu:")
        deneyim_korelasyon = self.veri['Courier_Experience_yrs'].corr(self.veri['Delivery_Time_min'])
        print(f"Korelasyon katsayısı: {deneyim_korelasyon:.4f}")
        
        # İşletme önerileri
        print("\n" + "="*40)
        print("İŞLETME ÖNERİLERİ:")
        print("="*40)
        
        print("1. Hava Durumu Stratejisi:")
        print("   - Kötü hava koşullarında teslimat sürelerini uzatın")
        print("   - Müşterilere gerçekçi teslimat süreleri bildirin")
        
        print("\n2. Trafik Yönetimi:")
        print("   - Yüksek trafik saatlerinde ek kurye görevlendirin")
        print("   - Alternatif rota planlaması yapın")
        
        print("\n3. Araç Seçimi:")
        print("   - Mesafeye göre uygun araç tipini seçin")
        print("   - Kısa mesafeler için bisiklet, uzun mesafeler için araba kullanın")
        
        print("\n4. Zaman Yönetimi:")
        print("   - Yoğun saatlerde hazırlık sürelerini optimize edin")
        print("   - Müşteri beklentilerini yönetin")
        
        print("\n5. Kurye Geliştirme:")
        print("   - Deneyimli kuryeleri kritik teslimatlar için görevlendirin")
        print("   - Yeni kuryelere mentorluk programı uygulayın")
    
    def proje_raporu_olustur(self):
        """
        Proje raporunu oluşturur
        """
        print("\n" + "="*60)
        print("PROJE RAPORU OLUŞTURULUYOR")
        print("="*60)
        
        # Rapor dosyasını oluştur
        with open('veri_madenciligi_raporu.md', 'w', encoding='utf-8') as f:
            f.write("# Veri Madenciliği Projesi: Yemek Teslimat Sürelerinin Sınıflandırılması\n\n")
            
            f.write("## 1. Kapak Sayfası\n")
            f.write("- **Proje Adı:** Yemek Teslimat Sürelerinin Makine Öğrenmesi ile Sınıflandırılması\n")
            f.write("- **Öğrenci:** [Öğrenci Adı ve Numarası]\n")
            f.write("- **Ders:** Veri Madenciliği\n")
            f.write("- **Teslim Tarihi:** [Tarih]\n\n")
            
            f.write("## 2. Giriş\n")
            f.write("### Proje Konusu ve Amacı\n")
            f.write("Bu proje, yemek teslimat hizmetlerinde teslimat sürelerini tahmin etmek ve sınıflandırmak amacıyla veri madenciliği tekniklerini kullanmaktadır. Proje, teslimat sürelerini 'Hızlı', 'Orta' ve 'Yavaş' kategorilerine ayırarak, hangi faktörlerin teslimat performansını etkilediğini belirlemeyi hedeflemektedir.\n\n")
            
            f.write("### Seçilen Veri Madenciliği Tekniği\n")
            f.write("Bu projede **Sınıflandırma (Classification)** tekniği kullanılmıştır. Teslimat süreleri kategorilere ayrılarak, çeşitli faktörlerin (hava durumu, trafik, mesafe vb.) teslimat süresini nasıl etkilediği tahmin edilmektedir.\n\n")
            
            f.write("### Kısa Yöntem Özeti\n")
            f.write("1. Veri seti yükleme ve keşifsel veri analizi\n")
            f.write("2. Eksik veri temizleme ve ön işleme\n")
            f.write("3. Kategorik değişkenlerin sayısallaştırılması\n")
            f.write("4. Teslimat sürelerinin kategorilere ayrılması\n")
            f.write("5. Farklı sınıflandırma algoritmalarının uygulanması\n")
            f.write("6. Model performanslarının karşılaştırılması\n")
            f.write("7. İşletme için içgörüler ve öneriler\n\n")
            
            f.write("## 3. Veri Seti Tanıtımı\n")
            f.write("### Veri Seti Kaynağı\n")
            f.write("Veri seti, yemek teslimat hizmetlerinden toplanan gerçek verilerden oluşmaktadır.\n\n")
            
            f.write("### Veri Seti Açıklaması\n")
            f.write("Veri seti, yemek teslimat süreçlerini etkileyen çeşitli faktörleri içermektedir. Toplam 1001 sipariş kaydı bulunmaktadır.\n\n")
            
            f.write("### Veri Seti Alanları\n")
            f.write("- **Order_ID:** Sipariş numarası\n")
            f.write("- **Distance_km:** Teslimat mesafesi (kilometre)\n")
            f.write("- **Weather:** Hava durumu (Clear, Rainy, Snowy, Foggy, Windy)\n")
            f.write("- **Traffic_Level:** Trafik seviyesi (Low, Medium, High)\n")
            f.write("- **Time_of_Day:** Günün saati (Morning, Afternoon, Evening, Night)\n")
            f.write("- **Vehicle_Type:** Araç tipi (Bike, Scooter, Car)\n")
            f.write("- **Preparation_Time_min:** Hazırlık süresi (dakika)\n")
            f.write("- **Courier_Experience_yrs:** Kurye deneyimi (yıl)\n")
            f.write("- **Delivery_Time_min:** Teslimat süresi (dakika) - Hedef değişken\n\n")
            
            f.write("### Veri Seti Boyutu\n")
            f.write(f"- Satır sayısı: {self.veri.shape[0]}\n")
            f.write(f"- Sütun sayısı: {self.veri.shape[1]}\n\n")
            
            f.write("## 4. Veri Ön İşleme\n")
            f.write("### Eksik Veri İşleme\n")
            f.write("- Weather: En sık görülen değer ile dolduruldu\n")
            f.write("- Traffic_Level: En sık görülen değer ile dolduruldu\n")
            f.write("- Time_of_Day: En sık görülen değer ile dolduruldu\n")
            f.write("- Courier_Experience_yrs: Medyan değer ile dolduruldu\n\n")
            
            f.write("### Kategorik Veri Dönüştürme\n")
            f.write("- Tüm kategorik değişkenler LabelEncoder ile sayısallaştırıldı\n")
            f.write("- Teslimat süreleri kategorilere ayrıldı (Hızlı: 0-40 dk, Orta: 40-60 dk, Yavaş: 60+ dk)\n\n")
            
            f.write("### Normalizasyon\n")
            f.write("- Sayısal değişkenler StandardScaler ile normalize edildi\n")
            f.write("- Eğitim ve test setleri %80-%20 oranında ayrıldı\n\n")
            
            f.write("## 5. Yöntem ve Uygulama\n")
            f.write("### Kullanılan Algoritmalar\n")
            f.write("1. **Decision Tree (Karar Ağacı)**\n")
            f.write("2. **Random Forest (Rastgele Orman)**\n")
            f.write("3. **K-Nearest Neighbors (K-En Yakın Komşu)**\n")
            f.write("4. **Naive Bayes**\n")
            f.write("5. **Gradient Boosting**\n\n")
            
            f.write("### Eğitim/Test Ayrımı\n")
            f.write("- Eğitim seti: %80 (800 örnek)\n")
            f.write("- Test seti: %20 (201 örnek)\n")
            f.write("- Stratified sampling kullanıldı\n\n")
            
            f.write("### Başarı Ölçütleri\n")
            f.write("- **Accuracy (Doğruluk):** Doğru tahmin edilen örneklerin oranı\n")
            f.write("- **Precision (Kesinlik):** Pozitif tahminlerin doğruluk oranı\n")
            f.write("- **Recall (Duyarlılık):** Gerçek pozitiflerin yakalanma oranı\n")
            f.write("- **F1-Score:** Precision ve Recall'un harmonik ortalaması\n\n")
            
            # Model sonuçlarını ekle
            f.write("### Model Performans Sonuçları\n")
            sonuclar_df = pd.DataFrame(self.sonuclar).T
            sonuclar_df = sonuclar_df.drop('predictions', axis=1)
            f.write(sonuclar_df.round(4).to_string())
            f.write("\n\n")
            
            f.write("## 6. Sonuç ve Yorumlar\n")
            f.write("### Elde Edilen Bulgular\n")
            f.write("1. **Hava durumu** teslimat sürelerini önemli ölçüde etkilemektedir\n")
            f.write("2. **Trafik seviyesi** yüksek olduğunda teslimat süreleri artmaktadır\n")
            f.write("3. **Araç tipi** mesafeye göre optimize edilmelidir\n")
            f.write("4. **Kurye deneyimi** teslimat performansını etkilemektedir\n\n")
            
            f.write("### Uygulamanın Güçlü Yönleri\n")
            f.write("- Çoklu algoritma karşılaştırması\n")
            f.write("- Kapsamlı veri ön işleme\n")
            f.write("- İşletme odaklı içgörüler\n")
            f.write("- Görsel analiz ve raporlama\n\n")
            
            f.write("### Geliştirilebilecek Yönler\n")
            f.write("- Daha fazla veri toplanması\n")
            f.write("- Hiperparametre optimizasyonu\n")
            f.write("- Derin öğrenme modellerinin denenmesi\n")
            f.write("- Gerçek zamanlı tahmin sistemi\n\n")
            
            f.write("### İşletme Yorumları\n")
            f.write("Bu analiz sonuçları, yemek teslimat hizmetlerinde:\n")
            f.write("- Operasyonel verimliliği artırmak\n")
            f.write("- Müşteri memnuniyetini yükseltmek\n")
            f.write("- Maliyetleri optimize etmek\n")
            f.write("- Kaynak planlamasını iyileştirmek\n")
            f.write("için kullanılabilir.\n\n")
            
            f.write("## 7. Kaynakça\n")
            f.write("1. Scikit-learn Documentation: https://scikit-learn.org/\n")
            f.write("2. Pandas Documentation: https://pandas.pydata.org/\n")
            f.write("3. Matplotlib Documentation: https://matplotlib.org/\n")
            f.write("4. Seaborn Documentation: https://seaborn.pydata.org/\n")
            f.write("5. Veri Madenciliği: Kavramlar ve Teknikler - Jiawei Han\n\n")
            
            f.write("## 8. Ekler\n")
            f.write("- Python kodu: veri_madenciligi_projesi.py\n")
            f.write("- Veri seti: Food_Delivery_Times.csv\n")
            f.write("- Grafik dosyaları: veri_analizi.png, model_karsilastirma.png, ozellik_onem.png\n")
            f.write("- Rapor: veri_madenciligi_raporu.md\n")
        
        print("Proje raporu 'veri_madenciligi_raporu.md' dosyasına kaydedildi.")
    
    def tum_analizi_calistir(self):
        """
        Tüm analizi sırayla çalıştırır
        """
        print("YEMEK TESLİMAT SÜRELERİ VERİ MADENCİLİĞİ PROJESİ")
        print("="*60)
        
        # 1. Veri yükleme
        self.veri_yukle()
        
        # 2. Veri ön işleme
        self.veri_on_isleme()
        
        # 3. Veri görselleştirme
        self.veri_gorselleştirme()
        
        # 4. Model eğitimi
        self.model_egitimi()
        
        # 5. Model karşılaştırması
        self.model_karsilastirma()
        
        # 6. Detaylı rapor
        self.detayli_rapor()
        
        # 7. İşletme içgörüleri
        self.is_insights()
        
        # 8. Proje raporu oluşturma
        self.proje_raporu_olustur()
        
        print("\n" + "="*60)
        print("PROJE TAMAMLANDI!")
        print("="*60)
        print("Oluşturulan dosyalar:")
        print("- veri_madenciligi_projesi.py (Python kodu)")
        print("- veri_analizi.png (Veri analizi grafikleri)")
        print("- model_karsilastirma.png (Model karşılaştırma grafikleri)")
        print("- ozellik_onem.png (Özellik önem grafiği)")
        print("- veri_madenciligi_raporu.md (Proje raporu)")

# Ana program
if __name__ == "__main__":
    # Analiz sınıfını başlat
    analiz = YemekTeslimatAnalizi()
    
    # Tüm analizi çalıştır
    analiz.tum_analizi_calistir() 