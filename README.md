# Customer Behavior Analysis with PySpark & Hadoop 🐘🐍

Bu proje, **Sivas Cumhuriyet Üniversitesi** Büyük Veri Analistliği dersi kapsamında, Hadoop altyapısı üzerinde PySpark kütüphanesi kullanılarak gerçekleştirilmiştir. Yaklaşık 100.000 satırlık perakende verisi üzerinde makine öğrenmesi modelleri eğitilmiştir.

### 🛠️ Kullanılan Teknolojiler

* **Infrastructure:** Hadoop (HDFS)
* **Processing:** Apache Spark (PySpark)
* **Machine Learning:** Spark MLlib (Pipeline, Logistic Regression, Decision Tree)
* **Data Tools:** Pandas, HDFS CLI

### 🚀 Proje Öne Çıkanlar

* **Pipeline Mimarisi:** Veri ön işleme (StringIndexer, VectorAssembler, StandardScaler) süreçleri Spark Pipeline ile otomatikleştirildi.
* **Çoklu Sınıflandırma:** Müşteri özelliklerine göre ürün kategorisi tahmini yapıldı.
* **Stratejik Analizler:** Cinsiyet yönelimi, AVM bazlı yoğunluk ve yaş gruplarına göre kategori tercihleri analiz edildi.
* **Risk Modelleme:** Yüksek fiyatlı ve yüksek adetli alışverişler için "Satış Kaybı ve İade Riski" tahmin modeli geliştirildi.

### 📈 Model Performansları

* **Lojistik Regresyon Doğruluğu:** % [Koddaki sonuç buraya]
* **Karar Ağacı Doğruluğu:** % [Koddaki sonuç buraya]
