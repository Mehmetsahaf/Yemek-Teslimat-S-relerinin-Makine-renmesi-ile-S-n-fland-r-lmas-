#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
İstatistiksel Analiz ve Görselleştirme
Yemek Teslimat Süreleri Veri Seti
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Türkçe karakter desteği
plt.rcParams['font.family'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class IstatistikselAnaliz:
    def __init__(self, dosya_yolu='Food_Delivery_Times.csv'):
        self.dosya_yolu = dosya_yolu
        self.veri = None
        
    def veri_yukle(self):
        """Veri setini yükler ve temel istatistikleri hesaplar"""
        self.veri = pd.read_csv(self.dosya_yolu)
        
        # Eksik verileri doldur
        self.veri['Weather'].fillna(self.veri['Weather'].mode()[0], inplace=True)
        self.veri['Traffic_Level'].fillna(self.veri['Traffic_Level'].mode()[0], inplace=True)
        self.veri['Time_of_Day'].fillna(self.veri['Time_of_Day'].mode()[0], inplace=True)
        self.veri['Courier_Experience_yrs'].fillna(self.veri['Courier_Experience_yrs'].median(), inplace=True)
        
        return self.veri
    
    def temel_istatistikler(self):
        """Temel istatistikleri hesaplar ve görselleştirir"""
        print("="*60)
        print("TEMEL İSTATİSTİKLER")
        print("="*60)
        
        # Sayısal değişkenler için istatistikler
        numeric_cols = ['Distance_km', 'Preparation_Time_min', 'Courier_Experience_yrs', 'Delivery_Time_min']
        stats_df = self.veri[numeric_cols].describe()
        
        print("Temel İstatistikler:")
        print(stats_df.round(2))
        
        # Görselleştirme
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Temel İstatistikler - Dağılım Grafikleri', fontsize=16, fontweight='bold')
        
        # 1. Teslimat Süresi Dağılımı
        axes[0, 0].hist(self.veri['Delivery_Time_min'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].axvline(self.veri['Delivery_Time_min'].mean(), color='red', linestyle='--', 
                          label=f'Ortalama: {self.veri["Delivery_Time_min"].mean():.1f}')
        axes[0, 0].axvline(self.veri['Delivery_Time_min'].median(), color='green', linestyle='--', 
                          label=f'Medyan: {self.veri["Delivery_Time_min"].median():.1f}')
        axes[0, 0].set_title('Teslimat Süresi Dağılımı')
        axes[0, 0].set_xlabel('Teslimat Süresi (dakika)')
        axes[0, 0].set_ylabel('Frekans')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Mesafe Dağılımı
        axes[0, 1].hist(self.veri['Distance_km'], bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0, 1].axvline(self.veri['Distance_km'].mean(), color='red', linestyle='--', 
                          label=f'Ortalama: {self.veri["Distance_km"].mean():.1f}')
        axes[0, 1].axvline(self.veri['Distance_km'].median(), color='green', linestyle='--', 
                          label=f'Medyan: {self.veri["Distance_km"].median():.1f}')
        axes[0, 1].set_title('Mesafe Dağılımı')
        axes[0, 1].set_xlabel('Mesafe (km)')
        axes[0, 1].set_ylabel('Frekans')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Hazırlık Süresi Dağılımı
        axes[1, 0].hist(self.veri['Preparation_Time_min'], bins=25, alpha=0.7, color='orange', edgecolor='black')
        axes[1, 0].axvline(self.veri['Preparation_Time_min'].mean(), color='red', linestyle='--', 
                          label=f'Ortalama: {self.veri["Preparation_Time_min"].mean():.1f}')
        axes[1, 0].axvline(self.veri['Preparation_Time_min'].median(), color='green', linestyle='--', 
                          label=f'Medyan: {self.veri["Preparation_Time_min"].median():.1f}')
        axes[1, 0].set_title('Hazırlık Süresi Dağılımı')
        axes[1, 0].set_xlabel('Hazırlık Süresi (dakika)')
        axes[1, 0].set_ylabel('Frekans')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Kurye Deneyimi Dağılımı
        axes[1, 1].hist(self.veri['Courier_Experience_yrs'], bins=20, alpha=0.7, color='purple', edgecolor='black')
        axes[1, 1].axvline(self.veri['Courier_Experience_yrs'].mean(), color='red', linestyle='--', 
                          label=f'Ortalama: {self.veri["Courier_Experience_yrs"].mean():.1f}')
        axes[1, 1].axvline(self.veri['Courier_Experience_yrs'].median(), color='green', linestyle='--', 
                          label=f'Medyan: {self.veri["Courier_Experience_yrs"].median():.1f}')
        axes[1, 1].set_title('Kurye Deneyimi Dağılımı')
        axes[1, 1].set_xlabel('Deneyim (yıl)')
        axes[1, 1].set_ylabel('Frekans')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('temel_istatistikler.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Temel istatistik grafikleri 'temel_istatistikler.png' dosyasına kaydedildi.")
        
        return stats_df
    
    def korelasyon_analizi(self):
        """Korelasyon analizi yapar ve görselleştirir"""
        print("\n" + "="*60)
        print("KORELASYON ANALİZİ")
        print("="*60)
        
        # Sayısal değişkenler arası korelasyon
        numeric_cols = ['Distance_km', 'Preparation_Time_min', 'Courier_Experience_yrs', 'Delivery_Time_min']
        correlation_matrix = self.veri[numeric_cols].corr()
        
        print("Korelasyon Matrisi:")
        print(correlation_matrix.round(3))
        
        # Korelasyon testleri
        print("\nKorelasyon Testleri:")
        
        # Mesafe vs Teslimat Süresi
        corr_distance, p_value_distance = stats.pearsonr(self.veri['Distance_km'], self.veri['Delivery_Time_min'])
        print(f"Mesafe vs Teslimat Süresi: r = {corr_distance:.3f}, p = {p_value_distance:.6f}")
        
        # Hazırlık Süresi vs Teslimat Süresi
        corr_prep, p_value_prep = stats.pearsonr(self.veri['Preparation_Time_min'], self.veri['Delivery_Time_min'])
        print(f"Hazırlık Süresi vs Teslimat Süresi: r = {corr_prep:.3f}, p = {p_value_prep:.6f}")
        
        # Kurye Deneyimi vs Teslimat Süresi
        corr_exp, p_value_exp = stats.pearsonr(self.veri['Courier_Experience_yrs'], self.veri['Delivery_Time_min'])
        print(f"Kurye Deneyimi vs Teslimat Süresi: r = {corr_exp:.3f}, p = {p_value_exp:.6f}")
        
        # Görselleştirme
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Korelasyon Analizi', fontsize=16, fontweight='bold')
        
        # 1. Korelasyon ısı haritası
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, ax=axes[0, 0], cbar_kws={'shrink': 0.8})
        axes[0, 0].set_title('Korelasyon Isı Haritası')
        
        # 2. Mesafe vs Teslimat Süresi
        axes[0, 1].scatter(self.veri['Distance_km'], self.veri['Delivery_Time_min'], alpha=0.6, color='blue')
        z = np.polyfit(self.veri['Distance_km'], self.veri['Delivery_Time_min'], 1)
        p = np.poly1d(z)
        axes[0, 1].plot(self.veri['Distance_km'], p(self.veri['Distance_km']), "r--", alpha=0.8)
        axes[0, 1].set_xlabel('Mesafe (km)')
        axes[0, 1].set_ylabel('Teslimat Süresi (dakika)')
        axes[0, 1].set_title(f'Mesafe vs Teslimat Süresi\nr = {corr_distance:.3f}')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Hazırlık Süresi vs Teslimat Süresi
        axes[1, 0].scatter(self.veri['Preparation_Time_min'], self.veri['Delivery_Time_min'], alpha=0.6, color='green')
        z = np.polyfit(self.veri['Preparation_Time_min'], self.veri['Delivery_Time_min'], 1)
        p = np.poly1d(z)
        axes[1, 0].plot(self.veri['Preparation_Time_min'], p(self.veri['Preparation_Time_min']), "r--", alpha=0.8)
        axes[1, 0].set_xlabel('Hazırlık Süresi (dakika)')
        axes[1, 0].set_ylabel('Teslimat Süresi (dakika)')
        axes[1, 0].set_title(f'Hazırlık Süresi vs Teslimat Süresi\nr = {corr_prep:.3f}')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Kurye Deneyimi vs Teslimat Süresi
        axes[1, 1].scatter(self.veri['Courier_Experience_yrs'], self.veri['Delivery_Time_min'], alpha=0.6, color='purple')
        z = np.polyfit(self.veri['Courier_Experience_yrs'], self.veri['Delivery_Time_min'], 1)
        p = np.poly1d(z)
        axes[1, 1].plot(self.veri['Courier_Experience_yrs'], p(self.veri['Courier_Experience_yrs']), "r--", alpha=0.8)
        axes[1, 1].set_xlabel('Kurye Deneyimi (yıl)')
        axes[1, 1].set_ylabel('Teslimat Süresi (dakika)')
        axes[1, 1].set_title(f'Kurye Deneyimi vs Teslimat Süresi\nr = {corr_exp:.3f}')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('korelasyon_analizi.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Korelasyon analizi grafikleri 'korelasyon_analizi.png' dosyasına kaydedildi.")
        
        return correlation_matrix
    
    def kategorik_analiz(self):
        """Kategorik değişkenlerin teslimat süresine etkisini analiz eder"""
        print("\n" + "="*60)
        print("KATEGORİK DEĞİŞKEN ANALİZİ")
        print("="*60)
        
        # Kategorik değişkenler için analiz
        categorical_vars = ['Weather', 'Traffic_Level', 'Vehicle_Type', 'Time_of_Day']
        
        # İstatistiksel testler ve görselleştirme
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Kategorik Değişkenlerin Teslimat Süresine Etkisi', fontsize=16, fontweight='bold')
        
        for i, var in enumerate(categorical_vars):
            row = i // 2
            col = i % 2
            
            # Grup istatistikleri
            group_stats = self.veri.groupby(var)['Delivery_Time_min'].agg(['mean', 'std', 'count']).round(2)
            print(f"\n{var} - Grup İstatistikleri:")
            print(group_stats)
            
            # ANOVA testi
            groups = [group['Delivery_Time_min'].values for name, group in self.veri.groupby(var)]
            f_stat, p_value = stats.f_oneway(*groups)
            print(f"ANOVA Testi - F: {f_stat:.3f}, p: {p_value:.6f}")
            
            # Boxplot
            self.veri.boxplot(column='Delivery_Time_min', by=var, ax=axes[row, col])
            axes[row, col].set_title(f'{var}\nANOVA p = {p_value:.4f}')
            axes[row, col].set_xlabel(var)
            axes[row, col].set_ylabel('Teslimat Süresi (dakika)')
            axes[row, col].tick_params(axis='x', rotation=45)
        
        plt.suptitle('')  # Ana başlığı temizle
        plt.tight_layout()
        plt.savefig('kategorik_analiz.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Kategorik analiz grafikleri 'kategorik_analiz.png' dosyasına kaydedildi.")
        
        # Detaylı istatistikler
        print("\n" + "="*40)
        print("DETAYLI KATEGORİK İSTATİSTİKLER")
        print("="*40)
        
        for var in categorical_vars:
            print(f"\n{var} - Ortalama Teslimat Süreleri:")
            means = self.veri.groupby(var)['Delivery_Time_min'].mean().sort_values(ascending=False)
            for category, mean_time in means.items():
                print(f"  {category}: {mean_time:.2f} dakika")
    
    def aykirı_deger_analizi(self):
        """Aykırı değer analizi yapar"""
        print("\n" + "="*60)
        print("AYKIRI DEĞER ANALİZİ")
        print("="*60)
        
        # Sayısal değişkenler için aykırı değer analizi
        numeric_cols = ['Distance_km', 'Preparation_Time_min', 'Courier_Experience_yrs', 'Delivery_Time_min']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Aykırı Değer Analizi', fontsize=16, fontweight='bold')
        
        for i, col in enumerate(numeric_cols):
            row = i // 2
            col_idx = i % 2
            
            # Boxplot
            axes[row, col_idx].boxplot(self.veri[col])
            axes[row, col_idx].set_title(f'{col} - Aykırı Değer Analizi')
            axes[row, col_idx].set_ylabel(col)
            axes[row, col_idx].grid(True, alpha=0.3)
            
            # İstatistikler
            Q1 = self.veri[col].quantile(0.25)
            Q3 = self.veri[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = self.veri[(self.veri[col] < lower_bound) | (self.veri[col] > upper_bound)]
            
            print(f"\n{col}:")
            print(f"  Q1: {Q1:.2f}, Q3: {Q3:.2f}, IQR: {IQR:.2f}")
            print(f"  Alt sınır: {lower_bound:.2f}, Üst sınır: {upper_bound:.2f}")
            print(f"  Aykırı değer sayısı: {len(outliers)} ({len(outliers)/len(self.veri)*100:.1f}%)")
        
        plt.tight_layout()
        plt.savefig('aykirı_deger_analizi.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Aykırı değer analizi grafikleri 'aykirı_deger_analizi.png' dosyasına kaydedildi.")
    
    def ozet_istatistikler(self):
        """Özet istatistikleri tablo halinde oluşturur"""
        print("\n" + "="*60)
        print("ÖZET İSTATİSTİKLER TABLOSU")
        print("="*60)
        
        # Özet tablo oluştur
        summary_data = {
            'Değişken': ['Teslimat Süresi (dk)', 'Mesafe (km)', 'Hazırlık Süresi (dk)', 'Kurye Deneyimi (yıl)'],
            'Ortalama': [
                f"{self.veri['Delivery_Time_min'].mean():.2f}",
                f"{self.veri['Distance_km'].mean():.2f}",
                f"{self.veri['Preparation_Time_min'].mean():.2f}",
                f"{self.veri['Courier_Experience_yrs'].mean():.2f}"
            ],
            'Medyan': [
                f"{self.veri['Delivery_Time_min'].median():.2f}",
                f"{self.veri['Distance_km'].median():.2f}",
                f"{self.veri['Preparation_Time_min'].median():.2f}",
                f"{self.veri['Courier_Experience_yrs'].median():.2f}"
            ],
            'Standart Sapma': [
                f"{self.veri['Delivery_Time_min'].std():.2f}",
                f"{self.veri['Distance_km'].std():.2f}",
                f"{self.veri['Preparation_Time_min'].std():.2f}",
                f"{self.veri['Courier_Experience_yrs'].std():.2f}"
            ],
            'Minimum': [
                f"{self.veri['Delivery_Time_min'].min():.2f}",
                f"{self.veri['Distance_km'].min():.2f}",
                f"{self.veri['Preparation_Time_min'].min():.2f}",
                f"{self.veri['Courier_Experience_yrs'].min():.2f}"
            ],
            'Maksimum': [
                f"{self.veri['Delivery_Time_min'].max():.2f}",
                f"{self.veri['Distance_km'].max():.2f}",
                f"{self.veri['Preparation_Time_min'].max():.2f}",
                f"{self.veri['Courier_Experience_yrs'].max():.2f}"
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
        
        # Tabloyu kaydet
        summary_df.to_csv('ozet_istatistikler.csv', index=False, encoding='utf-8-sig')
        print("\nÖzet istatistikler 'ozet_istatistikler.csv' dosyasına kaydedildi.")
        
        return summary_df
    
    def tum_analizi_calistir(self):
        """Tüm istatistiksel analizi çalıştırır"""
        print("İSTATİSTİKSEL ANALİZ VE GÖRSELLEŞTİRME")
        print("="*60)
        
        # Veri yükle
        self.veri_yukle()
        
        # Tüm analizleri çalıştır
        self.temel_istatistikler()
        self.korelasyon_analizi()
        self.kategorik_analiz()
        self.aykirı_deger_analizi()
        self.ozet_istatistikler()
        
        print("\n" + "="*60)
        print("İSTATİSTİKSEL ANALİZ TAMAMLANDI!")
        print("="*60)
        print("Oluşturulan dosyalar:")
        print("- temel_istatistikler.png")
        print("- korelasyon_analizi.png")
        print("- kategorik_analiz.png")
        print("- aykirı_deger_analizi.png")
        print("- ozet_istatistikler.csv")

# Ana program
if __name__ == "__main__":
    analiz = IstatistikselAnaliz()
    analiz.tum_analizi_calistir() 