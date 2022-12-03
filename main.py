##################################################
# Rating Product & Sorting Reviews in Amazon
##################################################

#### İş Problemi ####
# E-ticaretteki en önemli problemlerden bir tanesi ürünlere satış
# sonrası verilen puanların doğru şekilde hesaplanmasıdır. Bu
# problemin çözümü e-ticaret sitesi için daha fazla müşteri
# memnuniyeti sağlamak, satıcılar için ürünün öne çıkması ve satın
# alanlar için sorunsuz bir alışveriş deneyimi demektir. Bir diğer
# problem ise ürünlere verilen yorumların doğru bir şekilde
# sıralanması olarak karşımıza çıkmaktadır. Yanıltıcı yorumların öne
# çıkması ürünün satışını doğrudan etkileyeceğinden dolayı hem
# maddi kayıp hem de müşteri kaybına neden olacaktır. Bu 2 temel
# problemin çözümünde e-ticaret sitesi ve satıcılar satışlarını
# arttırırken müşteriler ise satın alma yolculuğunu sorunsuz olarak
# tamamlayacaktır.

##### Veri Seti Hikayesi #####
# Amazon ürün verilerini içeren bu veri seti ürün kategorileri ile çeşitli metadataları içermektedir.
# Elektronik kategorisindeki en fazla yorum alan ürünün kullanıcı puanları ve yorumları vardır.

##### Değişkenler #####
# reviewerID    : Kullanıcı ID’si
# asin          : Ürün ID’si
# reviewerName  : Kullanıcı Adı
# helpful       : Faydalı değerlendirme derecesi
# reviewText    : Değerlendirme
# overall       : Ürün rating’i
# summary       : Değerlendirme özeti
# unixReviewTime: Değerlendirme zamanı
# reviewTime    : Değerlendirme zamanı Raw
# day_diff      : Değerlendirmeden itibaren geçen gün sayısı
# helpful_yes   : Değerlendirmenin faydalı bulunma sayısı
# total_vote    : Değerlendirmeye verilen oy sayısı
# 12 Değişken, 4915 Gözlem, 71.9 MB


##### Gerekli Kütüphaneler #####
import pandas as pd
import datetime as dt
import math
import scipy.stats as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df_ = pd.read_csv("amazon_review.csv")

df = df_.copy()
df.head()
df.shape
df.info()
df.describe().T
df.isnull().sum()
df.dropna(inplace=True)

# aynı ürüne 4915 yorum yapılmış
df["asin"].nunique()
df["asin"].value_counts()

# her yorumu farklı biri yapmış
df["reviewerID"].nunique()

##############################################################
# Görev 1: Average Rating’i güncel yorumlara göre hesaplayınız
# ve var olan average rating ile kıyaslayınız
##############################################################

# Adım 1: Ürünün ortalama puanını hesaplayınız. (var olan average rating)
df["overall"].mean()

# Adım 2: Tarihe göre ağırlıklı puan ortalamasını hesaplayınız.
df.info()
df["reviewTime"] = df["reviewTime"].apply(pd.to_datetime)
current_date = df["reviewTime"].max()  #2014-12-07
df["day_diff_recency"] = (current_date - df["reviewTime"]).dt.days

df[["day_diff", "day_diff_recency"]].head(10)

quantile1 = df["day_diff_recency"].quantile(0.25)  #280.0
quantile2 = df["day_diff_recency"].quantile(0.50)  #430.0
quantile3 = df["day_diff_recency"].quantile(0.75)  #580.0

# güncel yorumlara göre average rating
print(df.loc[df["day_diff_recency"] <= quantile1, "overall"].mean())
print(df.loc[(df["day_diff_recency"] > quantile1) & (df["day_diff_recency"] <= quantile2), "overall"].mean())
print(df.loc[(df["day_diff_recency"] > quantile2) & (df["day_diff_recency"] <= quantile3), "overall"].mean())
print(df.loc[df["day_diff_recency"] > quantile3, "overall"].mean())


# Adım 3: Ağırlıklandırılmış puanlamada her bir zaman diliminin ortalamasını karşılaştırıp yorumlayınız.
# 1. çeyrek: 4.6957928802588995
# 2. çeyrek: 4.636140637775961
# 3. çeyrek: 4.571661237785016
# 4. çeyrek: 4.4462540716612375

##############################################################
# Görev 2: Ürün için ürün detay sayfasında görüntülenecek 20 review'ı seçiniz.
##############################################################

# Adım 1: helpful_no değişkenini üretiniz.
df["helpful_no"] = df["total_vote"] - df["helpful_yes"]

# Adım 2: score_pos_neg_diff, score_average_rating ve wilson_lower_bound skorlarını hesaplayıp veriye ekleyiniz.
def score_pos_neg_diff(helpful_yes, helpful_no):
    return helpful_yes - helpful_no

def score_average_rating(helpful_yes, helpful_no):
    if helpful_yes + helpful_no == 0:
        return 0
    return helpful_yes / (helpful_yes + helpful_no)

def wilson_lower_bound(helpful_yes, helpful_no, confidence=0.95):
    """
    Wilson Lower Bound Score hesapla

    - Bernoulli parametresi p için hesaplanacak güven aralığının(confidence) alt sınırı WLB skoru olarak kabul edilir.
    - Hesaplanacak skor ürün sıralaması için kullanılır.
    - Not:
    Eğer skorlar 1-5 arasıdaysa 1-3 negatif, 4-5 pozitif olarak işaretlenir ve bernoulli'ye uygun hale getirilebilir.
    Bu beraberinde bazı problemleri de getirir. Bu sebeple bayesian average rating yapmak gerekir.

    Parameters
    ----------
    up: int
        up count
    down: int
        down count
    confidence: float
        confidence

    Returns
    -------
    wilson score: float

    """

    n = helpful_yes + helpful_no
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = helpful_yes / n
    return (phat + z ** 2 / (2 * n) - z * math.sqrt((phat * (1 - phat) + z ** 2 / (4 * n)) / n)) / (1 + z ** 2 / n)

df["score_pos_neg_diff"] = df.apply(lambda x: score_pos_neg_diff(x["helpful_yes"], x["helpful_no"]), axis=1)
df["score_average_rating"] = df.apply(lambda x: score_average_rating(x["helpful_yes"], x["helpful_no"]), axis=1)
df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"], x["helpful_no"]), axis=1)
df.head()


# Adım 3: 20 Yorumu belirleyiniz ve sonuçları Yorumlayınız
df.sort_values("wilson_lower_bound", ascending=False).head(20)

# ürün rating
df["overall"].value_counts()
sns.countplot(x=df["overall"], data=df)
plt.show()

# wlb score
df["wilson_lower_bound"].value_counts()
sns.histplot(x=df["wilson_lower_bound"], data=df)
plt.show()

# yıllara göre yorum sayısı
year = df["reviewTime"].dt.year
year.value_counts()
sns.countplot(x=year, data=df)
plt.show()








