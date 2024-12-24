import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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
