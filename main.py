import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

import os
import joblib
from scipy.stats import shapiro




data_rw = pd.read_csv('./data/winequality-red.csv', sep=';')
data_ww = pd.read_csv('./data/winequality-white.csv', sep=';')




# Veri çerçevesinin boyutunu kontrol etme
print(f'Red Wine Dataset: {data_rw.shape}')
print(f'White Wine Dataset: {data_ww.shape}')

# İlk 5 satırı yazdırma
print(data_rw.head())
print(data_ww.head())


# Kırmızı şarap veri setinde eksik verileri kontrol etme
print(data_rw.isnull().any())
# Beyaz şarap veri setinde eksik verileri kontrol etme
print(data_ww.isnull().any())


# Kırmızı şarap veri seti hakkında bilgi alma
print(data_rw.info())
# Beyaz şarap veri seti hakkında bilgi alma
print(data_ww.info())


# 'type' adında yeni bir sütun ekleyip kırmızı şarap veri setine "red" değeri atıyoruz
data_rw.insert(0, 'type', 'red')
# 'type' adında yeni bir sütun ekleyip beyaz şarap veri setine "white" değeri atıyoruz
data_ww.insert(0, 'type', 'white')


# Kırmızı şarap veri setinin ilk 5 satırını yazdırma
print(data_rw.head())
# Beyaz şarap veri setinin ilk 5 satırını yazdırma
print(data_ww.head())


# Veri setlerini birleştirme
wines = pd.concat([data_rw, data_ww], ignore_index=True)

# Yeni veri çerçevesinin boyutunu kontrol etme
print(wines.shape)
# Veri çerçevesinin temel bilgilerini kontrol etme
wines.info()


# 'quality class' sütununu ekliyoruz
wines['quality class'] = wines.quality.apply(lambda q: 'low' if q <= 5 \
                                             else 'high' if q > 7 else 'medium')
# İlk 5 satırı yazdırma
print(wines.head())
# Veri çerçevesinin yapısı hakkında bilgi almak
wines.info()



# Benzersiz değerleri görüntüleme
print(wines.apply(lambda c: c.unique()))
# Benzersiz değerlerin sayısını kontrol etme
print(wines.apply(lambda c: c.unique().shape[0]))


# Veri türlerinin ve her birinin sayısının kontrolü
print(wines.dtypes.value_counts())
# Veri çerçevesinin istatistiksel özet bilgilerini kontrol etme
print(wines.describe())


# Veri çerçevesindeki sütun isimlerini yazdırma
print(wines.columns)


rws = round(wines.loc[wines.type == 'red', wines.columns].describe(), 2).T
wws = round(wines.loc[wines.type == 'white', wines.columns].describe(), 2).T
# Kırmızı ve beyaz şaraplar için tanımlayıcı istatistikleri birleştirme
result = pd.concat([rws, wws], axis=1, keys=['Red Wine', 'White Wine'])
print(result)



# Düşük kalite şaraplar için tanımlayıcı istatistikler
lqs = round(wines.loc[wines['quality class'] == 'low', wines.columns].describe(), 2).T
# Orta kalite şaraplar için tanımlayıcı istatistikler
mqs = round(wines.loc[wines['quality class'] == 'medium', wines.columns].describe(), 2).T
# Yüksek kalite şaraplar için tanımlayıcı istatistikler
hqs = round(wines.loc[wines['quality class'] == 'high', wines.columns].describe(), 2).T
# Veri çerçevelerini birleştirme
quality_stats = pd.concat([lqs, mqs, hqs], axis=1, keys=['Low Quality Wine', 'Medium Quality Wine', 'High Quality Wine']).T
print(quality_stats)







numeric_columns = wines.select_dtypes(include=['float64', 'int64'])
# Normallik testi (Shapiro-Wilk)

# Sadece sayısal sütunlar üzerinde test yapıyoruz
print("\nShapiro-Wilk Normality Test Sonuçları:\n")
for column in numeric_columns.columns:
    stat, p_value = shapiro(wines[column])
    sonuc = "Uygun" if p_value >= 0.05 else "Uygun Değil"
    print(f"{column}: Test İstatistiği = {stat:.6f}, p-değeri = {p_value:.6e}, Sonuç = {sonuc}")


# Grafik düzenini oluşturma
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
f.suptitle('Wine Type vs Quality Classes', fontsize=14)
f.subplots_adjust(top=0.85, wspace=0.3)

# Kırmızı şarap için kalite dağılımı
sns.countplot(x='quality class', data=wines[wines.type == 'red'],
              color='red', order=['low', 'medium', 'high'], edgecolor='black', ax=ax1)
ax1.set_title('Red Wine')
ax1.set_xlabel('Quality Class')
ax1.set_ylabel('Frequency', size=12)
ax1.set_ylim([0, 3200])

# Beyaz şarap için kalite dağılımı
sns.countplot(x='quality class', data=wines[wines.type == 'white'],
              color='gray', order=['low', 'medium', 'high'], edgecolor='black', ax=ax2)
ax2.set_title('White Wine')
ax2.set_xlabel('Quality Class')
ax2.set_ylabel('Frequency', size=12)
ax2.set_ylim([0, 3200])

# Grafiği gösterme
plt.show()



# Veriyi karıştırma
wines = wines.sample(frac=1, random_state=77).reset_index(drop=True)
from sklearn.preprocessing import LabelEncoder

# LabelEncoder nesnesi oluşturma
le = LabelEncoder()

# "type" sütunundaki kategorik verileri sayısal verilere dönüştürme
y_type = le.fit_transform(wines.type.values)  # 0 = Kırmızı, 1 = Beyaz

# Yeni "color" sütunu ekleme
wines['color'] = y_type

# "y_type" veri tipini kontrol etme
print(type(y_type))

# Veri çerçevesinin ilk 5 satırını yazdırma
print(wines.head())




# Kategorik kalite sınıflarını sayısal verilere dönüştürme
qcl = {'low': 0, 'medium': 1, 'high': 2}
y_qclass = wines['quality class'].map(qcl)
print(y_qclass.head())
# Veri tipini kontrol etme
print(type(y_qclass))





# Yalnızca sayısal sütunları seçmek
numeric_columns = wines.select_dtypes(include=['float64', 'int64'])

# Korelasyon matrisini hesaplama
wcorr = numeric_columns.corr()

# 'color' sütununa göre sıralama (önce numeric_columns'dan 'color' sütununu seçtiğinizden emin olun)
if 'color' in numeric_columns.columns:
    sort_corr_cols = wcorr['color'].sort_values(ascending=False).keys()
    sort_corr_t = wcorr.loc[sort_corr_cols, sort_corr_cols]

    # Sonuçları yazdırma
    print(sort_corr_t)
else:
    print("'color' sütunu korelasyon için uygun değil.")






# Isı haritasını çizme
plt.figure(figsize=(13.5, 11.5))
sns.heatmap(sort_corr_t, 
            annot=True, 
            annot_kws={'fontsize': 14}, 
            fmt='.2f', 
            cmap='coolwarm', 
            square=True)

# Başlık ve eksen ayarları
plt.title('Wine Attributes Correlations by Wine Type', 
          fontsize=14, 
          fontweight='bold', 
          pad=10)
plt.xticks(rotation=50, fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')

# Grafiği göster
plt.show()







# Önemli sütunları seçme
important_columns = ['type', 'alcohol', 'volatile acidity', 'citric acid', 'quality']
selected_wines = wines[important_columns]

# Pairplot oluşturma (sadece seçilen sütunlar)
g = sns.pairplot(selected_wines,
                 hue='type',  # Türlere göre renklendirme
                 palette={'red': 'red', 'white': 'palegreen'},  # Renk paleti
                 plot_kws=dict(edgecolor='b', linewidth=0.5))  # Nokta kenar ayarları

# Başlık ekleme
fig = g.fig
fig.subplots_adjust(top=0.95, wspace=0.2)  # Grafik düzenleme
fig.suptitle('Selected Wine Attributes by Wine Types', 
             fontsize=26, 
             fontweight='bold')

# Grafiği kaydetme
g.savefig('./Figures/selected_pairplot.png')  # Klasör ve dosya ismini uygun şekilde ayarlayın

# Grafiği gösterme
plt.show()





# 'quality' sütununa göre sıralama
if 'quality' in wcorr.columns:
    sort_corr_cols = wcorr['quality'].sort_values(ascending=False).keys()
    sort_corr_q = wcorr.loc[sort_corr_cols, sort_corr_cols]
else:
    print("'quality' sütunu bulunamadı.")
    sort_corr_q = None

# Isı haritasını çizme (sort_corr_q tanımlandıysa)
if sort_corr_q is not None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(13.5, 11.5))
    sns.heatmap(sort_corr_q, 
                annot=True, 
                annot_kws={'fontsize': 14}, 
                square=True, 
                fmt='.2f', 
                cmap='coolwarm')

    plt.title('Wine Attributes Correlations by Wine Quality Classes', 
              fontsize=14, 
              fontweight='bold', 
              pad=10)
    plt.xticks(rotation=50, fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    plt.show()
else:
    print("Isı haritası oluşturulamadı çünkü sort_corr_q tanımlanmadı.")


    # Önemli sütunları seçme
important_columns = ['quality class', 'alcohol', 'volatile acidity', 'citric acid', 'density', 'quality']  # Önemli gördüğünüz sütunları seçin
wines_pq = wines[important_columns]  # Veri çerçevesinden yalnızca bu sütunları seçiyoruz

# Pairplot oluşturma
g = sns.pairplot(wines_pq,
                 hue='quality class',  # Kalite sınıfına göre renklendirme
                 palette={'high': 'coral', 'medium': 'palegreen', 'low': 'dodgerblue'},  # Renk paleti
                 plot_kws=dict(edgecolor='b', linewidth=0.5))  # Nokta kenar ayarları

# Başlık ekleme
fig = g.fig
fig.subplots_adjust(top=0.95, wspace=0.2)  # Grafik düzenleme
fig.suptitle('Wine Attributes by Selected Important Features',
             fontsize=26,
             fontweight='bold')

# Grafiği kaydetme
import os
os.makedirs('./Figures', exist_ok=True)  # Figures klasörünü oluştur
g.savefig('./Figures/selected_pairplot.png')  # Grafiği kaydet

# Grafiği gösterme
plt.show()




# Geriye kalan sütun isimlerini yazdırma
for f in wines.drop(['type', 'quality', 'quality class', 'color'], axis=1).columns:
    print(f)


# Önemli sütunları seçme
important_columns = ['alcohol', 'volatile acidity', 'citric acid', 'density', 'residual sugar']

# Sadece önemli sütunlar için döngü oluştur
for attr in important_columns:
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    f.suptitle('Wine Type - Quality - ' + attr, fontsize=14)
    f.subplots_adjust(top=0.80, wspace=0.3)

    # Birinci boxplot: Quality (kalite skoru)
    sns.boxplot(x='quality', y=attr, hue='type', data=wines,
                palette={'red': 'coral', 'white': 'palegreen'}, ax=ax1)
    ax1.set_xlabel('Quality')
    ax1.set_ylabel(attr, size=12)
    ax1.legend(title='Wine Type', bbox_to_anchor=(1.1, 1.15))

    # İkinci boxplot: Quality Class (kalite sınıfı)
    sns.boxplot(x='quality class', y=attr, hue='type', data=wines,
                order=['low', 'medium', 'high'],
                palette={'red': 'coral', 'white': 'palegreen'}, ax=ax2)
    ax2.set_xlabel('Quality Class')
    ax2.set_ylabel(attr)
    ax2.legend(loc=1, title='Wine Type', bbox_to_anchor=(1.1, 1.15))

    # Grafiği göster
    plt.show()




# Önemli sütunları seçme
important_columns = ['alcohol', 'volatile acidity', 'citric acid', 'density', 'residual sugar']

# Önemli sütunlarla violin plot oluşturma
for attr in important_columns:
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))  # Her sütun için iki alt grafik
    f.suptitle(f'Wine Type - Quality - {attr}', fontsize=16, fontweight='bold')  # Grafik başlığı
    f.subplots_adjust(top=0.85, wspace=0.4)  # Üst boşluk ve yatay aralık

    # İlk violin plot: Quality ile ilişki
    sns.violinplot(x='quality', y=attr, hue='type', data=wines, split=True, inner='quart',
                   palette={'red': 'coral', 'white': 'palegreen'}, ax=ax1)
    ax1.set_xlabel('Quality', fontsize=12)
    ax1.set_ylabel(attr, fontsize=12)
    ax1.legend(title='Wine Type', bbox_to_anchor=(1.05, 1), loc='upper left')

    # İkinci violin plot: Quality Class ile ilişki
    sns.violinplot(x='quality class', y=attr, hue='type', data=wines, split=True, inner='quart',
                   order=['low', 'medium', 'high'],
                   palette={'red': 'coral', 'white': 'palegreen'}, ax=ax2)
    ax2.set_xlabel('Quality Class', fontsize=12)
    ax2.set_ylabel(attr, fontsize=12)
    ax2.legend(title='Wine Type', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Grafiği gösterme
    plt.show()


    # lmplot ile görselleştirme
g = sns.lmplot(
    x='alcohol', 
    y='density', 
    col='type', 
    col_order=['red', 'white'], 
    hue='quality class', 
    hue_order=['low', 'medium', 'high'], 
    data=wines, 
    palette=sns.light_palette("navy", 4), 
    scatter_kws=dict(alpha=0.95, edgecolor="k", linewidths=0.5),  # 'linewidth' yerine 'linewidths'
    fit_reg=True, 
    legend=False  # Legend'i manuel ekleyeceğiz
)

# Grafik düzenlemeleri
fig = g.fig
fig.subplots_adjust(top=0.85, wspace=0.3)  # Üst boşluk ve yatay aralık
fig.suptitle('Wine Type - Density - Alcohol - Quality', fontsize=14)  # Genel başlık

# Legend ekleme
g.add_legend(title='Wine Quality Class')

# Grafiği gösterme
plt.show()


print(wines.head())

# Özellikleri (features) ayıklama
features = wines.drop(['type', 'quality', 'quality class', 'color'], axis=1).columns
X = wines[features].copy()

# İlk 5 satırı yazdırma
print("Extracted Features (X):")
print(X.head())

# Hedef değişkeni (target) ayrıştırma
y = wines.color.copy()

# Hedef değişkenin ilk 5 satırını görüntüleme
print("First 5 values of the target variable (y):")
print(y.head())

# Hedef değişkenin dağılımını görüntüleme
color_distribution = wines.groupby('color').color.count()
print("\nDistribution of the target variable (color):")
print(color_distribution) 




# 'color' sütununu kopyalayarak y hedef değişkenini oluşturma
y = wines.color.copy()

# İlk 5 satırı görüntüleme
print("First 5 rows of 'y':")
print(y.head())

# 'color' sütununun dağılımını kontrol etme
color_distribution = wines.groupby('color')['color'].count()
print("\nDistribution of 'color':")
print(color_distribution)





# Özellikleri (X) ve hedef değişkeni (y) ayırma
X = numeric_columns.drop(['color'], axis=1)  # 'color' sütununu çıkartıyoruz çünkü bu hedef değişken
y = wines['color']  # Hedef değişken: 0 = kırmızı, 1 = beyaz

# Eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=77, stratify=y)

# Model pipeline oluşturma (standartlaştırma + lojistik regresyon)
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Veriyi standartlaştırma
    ('model', LogisticRegression(random_state=77))  # Lojistik Regresyon
])

# Modeli eğitme
pipeline.fit(X_train, y_train)

# Test setinde tahmin yapma
y_pred = pipeline.predict(X_test)

# Model performansını değerlendirme
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))





# Hiperparametre ızgarasını tanımlama
param_grid = {
    'model__C': [0.1, 1, 10, 100],
    'model__tol': [0.001, 0.0001]
}

# GridSearchCV ile hiperparametre optimizasyonu
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# En iyi parametreleri ve en iyi skoru yazdırma
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Score:", grid_search.best_score_)

# Test seti üzerinde en iyi modeli değerlendirme
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

from sklearn.metrics import classification_report
print("Classification Report for Best Model:")
print(classification_report(y_test, y_pred))






# Pipeline oluşturma
pipeline = Pipeline([
    ('scl', StandardScaler()),  # Veriyi standartlaştırma
    ('lr', LogisticRegression(random_state=77))  # Lojistik Regresyon
])

# Hiperparametre ızgarası
param_grid = {
    'lr__C': [0.1, 1, 10, 100],
    'lr__tol': [0.001, 0.0001]
}

# GridSearchCV tanımlama
clf = GridSearchCV(pipeline, param_grid, cv=10)

# Modeli eğitme ve hiperparametreleri optimize etme
clf.fit(X_train, y_train)

# En iyi parametreler ve model
print("Best Parameters:", clf.best_params_)
print("Best Estimator:", clf.best_estimator_)

# Test setinde en iyi modelle tahmin
y_pred = clf.predict(X_test)

# Model performansını değerlendirme
print("Classification Report:")
print(classification_report(y_test, y_pred))









# Test seti üzerinde tahmin yapma
y_pred = clf.predict(X_test)

# Model performansını değerlendirme
target_names = ['red', 'white']
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=target_names), '\n')

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred)) 








# Veri setinin ilk 5 satırını kontrol edin
print("Veri setinin ilk 5 satırı:")
print(wines.head())

# Özellikleri seçme (gereksiz sütunları çıkarma)
features = wines.drop(['type', 'quality', 'quality class', 'color'], axis=1).columns

# Özelliklerin bir kopyasını oluşturma
X = wines[features].copy()

# İlk 5 satırı kontrol etme
print("\nSeçilen özelliklerin ilk 5 satırı:")
print(X.head())









y_qclass = wines['quality class'].map({'low': 0, 'medium': 1, 'high': 2})

# Hedef değişkeni atanması
y = y_qclass

# İlk 5 satırı görüntüleme
print("Hedef değişkenin ilk 5 değeri:")
print(y.head())

# Sınıf dağılımını kontrol etme
print("\nHedef değişkenin sınıf dağılımı:")
print(y.value_counts())








# Çubuk grafiği oluşturma
sns.countplot(
    x='quality class', 
    data=wines, 
    edgecolor='black', 
    order=['low', 'medium', 'high']
)

# Başlık ekleme
plt.title(
    'Wine Sample Distribution by Quality Class', 
    fontsize=14, 
    pad=10
)

# Grafiği gösterme
plt.show()






# Eğitim ve test setlerine bölme
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=77, stratify=y)

# Pipeline tanımlama
pipeline = Pipeline([
    ('scl', StandardScaler()),  # Ölçeklendirme
    ('rfc', RandomForestClassifier(random_state=77))  # Model
])

print("Eğitim ve test setleri oluşturuldu, pipeline tanımlandı.")

# Model dosyası adı
model_file = "best_model.pkl"

# Model dosyası var mı kontrol et
if os.path.exists(model_file):
    print(f"{model_file} bulundu. Model yükleniyor...")
    best_model = joblib.load(model_file)  # Kaydedilmiş modeli yükle
    print("Model başarıyla yüklendi.")
else:
    print("Model bulunamadı, model eğitiliyor...")

    # Pipeline parametrelerini listeleme
    print(pipeline.get_params())

    # Hiperparametre ızgarasını tanımlama
    param_grid = {
        'rfc__min_samples_leaf': [1, 2, 3],           # Yaprak düğümdeki minimum örnek sayısı
        'rfc__min_samples_split': [2, 3, 4],         # Dallanma için minimum örnek sayısı
        'rfc__n_estimators': [150, 175, 200],        # Ağaç sayısı
        'rfc__max_depth': [20, 40, None],            # Maksimum derinlik
        'rfc__criterion': ['gini', 'entropy'],       # Bölünme ölçütü
        'rfc__class_weight': ['balanced', None]      # Sınıf ağırlıkları
    }

    print("Hiperparametre ızgarası:")
    print(param_grid)

    # GridSearchCV işlemini başlatma
    clf = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=10, n_jobs=-1)

    # Modeli eğitme
    clf.fit(X_train, y_train)

    # En iyi hiperparametreleri yazdırma
    print("Best Parameters:", clf.best_params_)

    # GridSearchCV'den en iyi modeli alın
    best_model = clf.best_estimator_

    # En iyi modelin detaylarını yazdırın
    print("Best Model:")
    print(best_model)

    # Modeli kaydet
    joblib.dump(best_model, model_file)
    print(f"Model başarıyla {model_file} dosyasına kaydedildi.")

# Eğitilmiş modeli kullanabilirsiniz
# Örneğin: y_pred = best_model.predict(X_test)



# Test seti üzerinde tahminler yap
y_pred = best_model.predict(X_test)

# Performans değerlendirme raporu
target_names = ['low', 'medium', 'high']
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))

# Karışıklık matrisi
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Model doğruluğunu hesapla
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy Score:", accuracy)





# Karışıklık matrisi hesapla
mat = confusion_matrix(y_test, y_pred)

# Subplot oluştur
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
f.suptitle('Random Forest Classifier', fontsize=14)
f.subplots_adjust(top=0.85, wspace=0.3)

# Normalizasyon olmadan karışıklık matrisi
sns.heatmap(mat, annot=True, fmt='d', cbar=True, square=True, cmap='Oranges', ax=ax1)
ax1.set_xticklabels(labels=['low', 'medium', 'high'])
ax1.set_yticklabels(labels=['low', 'medium', 'high'])
ax1.set_title('Confusion Matrix w/o Normalization')
ax1.set_xlabel('Predicted Label')
ax1.set_ylabel('True Label')

# Normalizasyon ile karışıklık matrisi
matn = mat.astype('float') / mat.sum(axis=1)[:, np.newaxis]
sns.heatmap(matn, annot=True, fmt='.2f', cbar=True, square=True, cmap='Oranges', ax=ax2)
ax2.set_xticklabels(labels=['low', 'medium', 'high'])
ax2.set_yticklabels(labels=['low', 'medium', 'high'])
ax2.set_title('Normalized Confusion Matrix')
ax2.set_xlabel('Predicted Label')
ax2.set_ylabel('True Label')

# Grafikleri göster
plt.show()









from scipy.stats import skew, kurtosis

# Sadece sayısal sütunları seçme
numeric_columns = wines.select_dtypes(include=['float64', 'int64'])

# Medyan Hesaplama
median_values = numeric_columns.median()
print("\nSayısal Sütunların Medyan Değerleri:")
print(median_values)

# Mod Hesaplama
mode_values = numeric_columns.mode().iloc[0]
print("\nSayısal Sütunların Mod Değerleri:")
print(mode_values)

# Varyans Hesaplama
variance_values = numeric_columns.var()
print("\nSayısal Sütunların Varyans Değerleri:")
print(variance_values)

# Standart Sapma Hesaplama
std_dev_values = numeric_columns.std()
print("\nSayısal Sütunların Standart Sapma Değerleri:")
print(std_dev_values)

# Çarpıklık Hesaplama
skewness_values = numeric_columns.apply(skew)
print("\nSayısal Sütunların Çarpıklık (Skewness) Değerleri:")
print(skewness_values)

# Basıklık Hesaplama
kurtosis_values = numeric_columns.apply(kurtosis)
print("\nSayısal Sütunların Basıklık (Kurtosis) Değerleri:")
print(kurtosis_values)
