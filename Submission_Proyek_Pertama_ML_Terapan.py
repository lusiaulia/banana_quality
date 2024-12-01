#!/usr/bin/env python
# coding: utf-8

# # Klasifikasi Kualitas Buah Pisang
# - **Nama:** Lusi Aulia Jati
# - **Email:** lusiauliajati@gmail.com
# - **Sumber Data:** https://www.kaggle.com/datasets/l3llff/banana

# ##  Load Data

# In[1]:


#mengimpor library yang diperlukan
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


# In[2]:


# Clone dataset dari github
get_ipython().system('git clone https://github.com/lusiaulia/banana_quality.git')
# mendefinisikan dataframe
bnn = pd.read_csv("./banana_quality.csv")
bnn.head()


# Terlihat tampilan dari data, terdapat 7 data berbentuk kuantitatif dan 1 data kualitatif. Akan dilihat berapa banyak variasi data kualitatif pada data Quality.

# In[3]:


jumlah_jenis = bnn["Quality"].nunique()
print(f"Jumlah jenis data kualitatif: {jumlah_jenis}")
jenis_data = bnn["Quality"].value_counts()
print(jenis_data)


# Terdapat 2 jenis data kualitatif (Bad dan Good). Sehingga akan didefinisikan Good = 1 dan Bad = 0.

# In[4]:


bnn["Quality"] = bnn["Quality"].replace({"Bad": 0, "Good": 1})
bnn.head()


# In[5]:


#melihat informasi jumlah dan jenis data 
bnn.info()


# Total terdapat 8000 data, dengan 7 data bertipe float dan 1 jenis data integer (biner 1 dan 0).

# In[6]:


#melihat statistik deskriptif dari data
bnn.describe()


# Ketujuh data memiliki rentang berkisar -8 hingga 8, sebelum digunakan untuk proses prediksi akan dilebih sempitkan lagi interval range datanya menjadi -1 hingga 1 agar lebih algoritma bisa bekerja lebih maksimal

# ## Exploratory Data Analysis
# ### Data Cleaning

# In[7]:


#mengecek apakah ada data kosong
bnn.isnull().sum()


# In[8]:


#outlier dari data
sns.boxplot(bnn)


# Masih terdapat cukup banyak data outlier sehingga perlu dilakukan penghapusan outlier supaya tidak mempengaruhi hasil prediksi

# In[9]:


Q1 = bnn.quantile(0.25)
Q3 = bnn.quantile(0.75)
IQR=Q3-Q1
bnn=bnn[~((bnn<(Q1-1.5*IQR))|(bnn>(Q3+1.5*IQR))).any(axis=1)]
 
# Cek ukuran dataset setelah drop outliers
bnn.shape


# Setelah dihapus outlier diperoleh sebanyak 7645 data yang tersisa, coba dilihat secara visualisasi boxplotnya kembali

# In[10]:


sns.boxplot(bnn)


# terlihat sudah cukup bersih dari outlier

# In[11]:


bnn["Quality"].value_counts()


# Jumlah perbandingan sampel data kategori 'Bad' dan 'Good' memiliki jumlah yang hampir seimbang 3919 dan 3726 sehingga dipenelitian ini tidak dilakukan perubahan lagi untuk jumlah sampel.

# ### Feature Selection

# In[12]:


plt.figure()
correlation_matrix = bnn.corr().round(2)
 
# Untuk menge-print nilai di dalam kotak, gunakan parameter anot=True
sns.heatmap(data=correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Matrix untuk Fitur Numerik ", size=10)


# Secara korelasi data karakteristik buah memiliki korelasi yang kecil sekali terhadap kualitas buah pisang, seperti tingkat keasaman dan tekstur buah yang memiliki korelasi sebesar -0,02 dan -0,01. Namun untuk karakteristik lainnya menunjukkan korelasi positif yang cukup baik hampir diangka 0,4. Sehingga untuk data input akan digunakan karakteristik yang menunjukkan korelasi positif (Ukuran, Berat, Kemanisan, Waktu Panen, dan Kematangan Buah).  

# Menggunakan model K-Nearest Neighbor (KNN), Logistic Regression, dan XGBoost. Akan digunakan data train 80% dan test 20%.

# ### Data Transforms

# In[13]:


X = bnn.drop(['Quality','Acidity','Softness'], 1)
y = bnn['Quality']

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.2, random_state = 123)


# In[14]:


print(f'Total # of sample in whole dataset: {len(X_scaled)}')
print(f'Total # of sample in train dataset: {len(X_train)}')
print(f'Total # of sample in test dataset: {len(X_test)}')


# ## Modeling & Evaluate Model
# ### KNN Model

# In[15]:


knn = KNeighborsClassifier()
knn.fit(X_train, y_train)


# In[16]:


# membuat fungsi evaluasi model
def evaluasi_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)


# ![image.png](attachment:image.png)

# Evaluasi yang digunakan pada penelitian ini yaitu menggunakan skor akurasi, yang mengukur nilai akurasi yang didapatkan dari jumlah data bernilai positif yang diprediksi positif dan data bernilai negatif yang diprediksi negatif dibagi dengan jumlah seluruh data di dalam dataset.

# In[17]:


evaluasi_model(knn, X_test, y_test)


# Dihasilkan akurasi sebesar 91,8% pada model KNN pada data test.

# ### Logistic Regression

# In[18]:


LR = LogisticRegression()
LR.fit(X_train, y_train)


# In[19]:


evaluasi_model(LR, X_test, y_test)


# Dihasilkan akurasi sebesar 87,2% pada model regresi logistik pada data test.

# ### XG Boost

# In[20]:


# !pip install xgboost

XGB = XGBClassifier()
XGB.fit(X_train, y_train)


# In[21]:


evaluasi_model(XGB, X_test, y_test)


# Dihasilkan akurasi sebesar 92,1% pada model XGBoost pada data test.

# ## Conclusion

# Dari ketiga model terlihat model dengan akurasi yang paling besar yaitu model XGBoost dan KNN dengan akurasi berkisar 92%. Mengindikasikan bahwa model sudah bisa mengklasifikasikan data karakteristik buah pisang apakah berkualitas baik atau tidak dengan baik. Namun untuk penelitian selanjutnya akan lebih baik jika digunakan evaluasi lainnya seperti precision, recall, F1-score, atau AUC-ROC dan pengecekan apakah model overfit atau tidak supaya jika digunakan data baru model dapat memprediksi dengan baik.
