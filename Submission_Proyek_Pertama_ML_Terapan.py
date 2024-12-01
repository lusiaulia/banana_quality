#!/usr/bin/env python
# coding: utf-8

# # Proyek Pertama ML Terapan
# - **Nama:** Lusi Aulia Jati
# - **Email:** lusiauliajati@gmail.com
# - **Sumber Data:** https://www.kaggle.com/datasets/l3llff/banana

# ###  Data Loading

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

# load the dataset
bnn = pd.read_csv("./banana_quality.csv")
bnn.head()


# In[2]:


# # Clone dataset dari github
# !git clone https://github.com/lusiaulia/banana_quality.git
# # mendefinisikan dataframe
# bnn = pd.read_csv("./banana_quality.csv")
# bnn.head()


# In[3]:


jumlah_jenis = bnn["Quality"].nunique()
print(f"Jumlah jenis data kualitatif: {jumlah_jenis}")
jenis_data = bnn["Quality"].value_counts()
print(jenis_data)


# Karena data Quality merupakan data kualitatif (Bad dan Good). Sehingga akan didefinisikan Good = 1 dan Bad = 0.

# In[4]:


bnn["Quality"] = bnn["Quality"].replace({"Bad": 0, "Good": 1})
bnn.head()


# In[5]:


#melihat informasi jumlah dan jenis data 
bnn.info()


# In[6]:


#melihat statistik deskriptif dari data
bnn.describe()


# ### Exploratory Data Analysis

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

# In[12]:


plt.figure()
correlation_matrix = bnn.corr().round(2)
 
# Untuk menge-print nilai di dalam kotak, gunakan parameter anot=True
sns.heatmap(data=correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Matrix untuk Fitur Numerik ", size=10)


# Secara korelasi data karakteristik buah memiliki korelasi yang kecil sekali terhadap kualitas buah pisang, seperti tingkat keasaman dan tekstur buah yang memiliki korelasi sebesar -0,02 dan -0,01. Namun untuk karakteristik lainnya menunjukkan korelasi positif yang cukup baik hampir diangka 0,4. 

# Menggunakan model K-Nearest Neighbor (KNN), Logistic Regression, dan XGBoost. Akan digunakan data train 80% dan test 20%.

# In[13]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

X = bnn.drop(['Quality'], 1)
y = bnn['Quality']

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.2, random_state = 123)


# In[14]:


print(f'Total # of sample in whole dataset: {len(X_scaled)}')
print(f'Total # of sample in train dataset: {len(X_train)}')
print(f'Total # of sample in test dataset: {len(X_test)}')


# ### KNN Model

# In[15]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

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


# ### Logistic Regression

# In[18]:


from sklearn.linear_model import LogisticRegression

LR = LogisticRegression()
LR.fit(X_train, y_train)


# In[19]:


evaluasi_model(LR, X_test, y_test)


# ### XG Boost

# In[20]:


# !pip install xgboost
from xgboost import XGBClassifier

XGB = XGBClassifier()
XGB.fit(X_train, y_train)


# In[21]:


evaluasi_model(XGB, X_test, y_test)


# ### Hasil

# Dari ketiga model terlihat model dengan akurasi yang paling besar yaitu model XGBoost dan KNN dengan akurasi 97%. Mengindikasikan bahwa model sudah bisa mengklasifikasikan data karakteristik buah pisang apakah berkualitas baik atau tidak dengan baik. Namun untuk penelitian selanjutnya akan lebih baik jika digunakan evaluasi lainnya seperti precision, recall, F1-score, atau AUC-ROC dan pengecekan apakah model overfit atau tidak supaya jika digunakan data baru model dapat memprediksi dengan baik.
