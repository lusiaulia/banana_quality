# Laporan Proyek Machine Learning - Lusi Aulia Jati

![Gambar Banana](https://raw.githubusercontent.com/lusiaulia/banana_quality/a781ff7e77c14a3588bac17b933cd8956b945b12/banana_quality.jpeg)

## Domain Proyek Pertanian
Pada sektor agribisnis banyak sekali yang perlu diperhatikan dari proses awal hingga akhir seperti pengelolaan lahan, pemilihan bibit, perawatan tanaman dan pendistribusian baik ekspor maupun domestik. Mengambil contoh
pada tanaman buah pisang, ekspor buah pisang Indonesia sendiri memiliki potensi yang besar dibuktikan pada 2021, volume ekspor pisang menduduki posisi kedua tertinggi setelah manggis dengan angka 5.500 ton per Mei 2021 (mediaindonesia.com,2021).
Namun meski memiliki potensi yang besar, masih terdapat hambatan yang membuat sulitnya memaksimalkan potensi yang ada seperti persyaratan yang perlu dipenuhi dalam mengekspor buah salah satunya yaitu kualitas buah
yang harus memenuhi standar ekspor dan penyesuaian sesuai preferensi konsumen pasar dunia [Rara,2021](https://journal.ipb.ac.id/index.php/jagbi/article/view/36753/24583). Sehingga, dengan hal ini diperlukannya peningkatan kualitas buah yang dihasilkan. 
Pada penelitian ini, akan dilakukan percobaan klasifikasi baik tidaknya kualitas buah pisang berdasarkan data karakteristik buah dengan harapan dapat diketahui apakah ada kaitan antara karakteristik buah dengan kualitas. 
Jika karakteristik buah dapat menggambarkan kualitas buah, maka dapat dilakukan penelitian lebih lanjut karakteristik yang paling mewakili dan yang tidak, serta bagaimana mengoptimalkan tiap rakteristik agar buah memiliki kualitas optimal.  

## Business Understanding
### Problem Statements
- Apakah data-data karakteristik buah pisang yang dipanen bisa menggambarkan kualitas buah pisang?
- Dari data-data karakteristik buah pisang yang dipanen seperti Ukuran, Berat, Kemanisan, Keasaman, Waktu Panen, Tekstur dan Kematangan Buah, data mana sajakah yang berpengaruh terhadap kualitas buah?

### Goals
- Mengetahui apakah data-data karakteristik buah pisang yang dipanen bisa menggambarkan kualitas buah pisang.
- Mengetahui data karakteristik buah mana saja yang berpengaruh terhadap kualitas buah pisang.

### Solution Statements


## Data Understanding
Data yang digunakan yaitu data karakteristik buah pisang yang dipanen dengan total berisi 8000 data. Karakterisitik buah yang dimaksud adalah ukuran, berat, tingkat kemanisan, tekstur softness, waktu panen, kematangan, keasaman dan kualitas buah. Data bersumber dari [Kaggle Banana Quality Dataset](https://www.kaggle.com/datasets/l3llff/banana). Berikut nama variabel karakteristik buah pisang pada dataset yang akan digunakan dalam proses pemodelan : 

- Size : data ukuran buah
- Weight : data berat buah pisang
- Sweetness : data tingkat kemanisan buah
- Softness : data tekstur buah (lembut tidaknya)
- HarvestTime : waktu panen, bisa berupa usia buah ketika panen
- Ripeness : tingkat kematangan buah, asumsi sudah melalui proses pematangan
- Acidity : tingkat keasaman buah
- Quality : kualitas buah (good atau bad)
  
Ketujuh data kecuali kualitas adalah data numerik, data tersebut sudah dilakukan proses standarisasi sehingga bukan merupakan data mentah. Namun dalam proses kali ini akan dilakukan penskalaan ulang dengan rentang yang lebih sempit untuk memudahkan algoritma memahami data untuk memprediksi kualitas buah. 

## Data Preparation
Tahapan pengolahan data yang dilakukan meliputi proses Exploratory Data Analysis (EDA) yang mencakup : 
### Label Encoding
Data kualitas buah merupakan data kualitatif, sehingga diperlukan proses encoding menjadi data numerik agar dapat dilakukan pemrosesan data. Pada penelitian ini, proses ini dilakukan lebih dahulu sebelum proses data preparation lainnya.
### Data Cleaning
Pada proses ini akan dilihat apakah data utuh atau terdapat data kosong, jika terdapat data kosong (null) maka bisa dilakukan pengisian data atau dihapus.Kemudian apakah data berisi outlier atau tidak, jika iya untuk menghindari kesalahan model dalam membaca pola data dilakukan penghapusan data outlier. 
### Feature Selection
Pada tahap ini dilihat korelasi antar data satu sama lain dan terutama terhadap data kualitas buah. Karakteristik buah yang memiliki korelasi baik akan dipilih menjadi variabel penentu kualitas pada model yang dibuat nantinya. 
### Data Transform & Split Data
Seperti yang sudah dijelaskan pada bagian 'Data Understanding', akan dilakukan proses transformasi data melalui penskalaan yang lebih sempit dalam penelitian ini menggunakan MinMax Scaler. Selain itu, data juga dibagi menjadi data training dan test, dengan proporsi 80% data training dan 20% data test.

## Modeling
Pada penelitian ini dibuat 3 jenis model yang biasa digunakan dalam proses klasifikasi, yaitu : 
### K-Nearest Neighbor (KNN)
Metode yang digunakan untuk melakukan klasifikasi dari suatu data berdasarkan atribut-atributnya dengan mengambil sejumlah K data terdekat (tetangganya), dalam proses mengukur data terdekat menggunakan rumus jarak euclidian.
### Logistic Regression
Metode yang melakukan prediksi dari probabilitas suatu kejadian, dalam prosesnya logistic regression mencari fungsi logistik (sigmoid) terbaik untuk mengklasifikasikan data. 
### XGBoost  
Metode yang bekerja dengan cara menggabungkan hasil prediksi dari berbagai model Decision Tree sehingga menjadi model dengan akurasi dan kinerja yang cukup baik.

Ketiga model sudah terdapat library masing-masing sehingga bisa diimplementasikan dengan mudah tanpa perlu membuat model dari awal (scratch). 

## Evaluation & Conclusion
- Proses evaluasi menggunakan accuracy score, yang mengukur nilai akurasi yang didapatkan dari jumlah data bernilai positif yang diprediksi positif dan data bernilai negatif yang diprediksi negatif dibagi dengan jumlah seluruh data di dalam dataset. Dari ketiga model terlihat model dengan akurasi yang paling besar yaitu model XGBoost dan KNN dengan akurasi berkisar 92%, sedangkan Logistic Regression menghasilkan akurasi 87%. 
- Hasil akurasi mengindikasikan bahwa model sudah bisa mengklasifikasikan data karakteristik buah pisang apakah berkualitas baik atau tidak dengan baik. Sehingga bisa dikatakan bahwa dalam penelitian ini data karakteristik buah bisa menggambarkan kualitas buah pisang. 
- Dilihat dari skor korelasi dan hasil prediksi maka data karakteristik Ukuran, Berat, Kemanisan, Waktu Panen, dan Kematangan Buah memiliki pengaruh terhadap kualitas buah. Sedangkan keasaman dan tekstur memiliki korelasi yang sangat kecil bahkan hanya berada diangka 0,x.
- Terkadang dalam data nilai korelasi belum tentu menggambarkan sebab-akibat, sehingga untuk penelitian selanjutnya akan lebih baik jika dicoba juga mengikutsertakan kedua karakteristik buah tersebut dalam pemodelan atau mencoba kombinasi karakteristik lainnya. 
- Kemudian dapat juga digunakan evaluasi lain seperti precision, recall, F1-score, atau AUC-ROC serta pengecekan apakah model overfit atau tidak supaya jika digunakan data baru model yang sudah dibuat dapat memprediksi dengan kualitas buah dengan baik.
