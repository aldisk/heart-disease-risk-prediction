# Laporan Proyek Machine Learning - Aldiansyah Satrio Kabisat

## Domain Proyek

Penyakit jantung merupakan penyebab utama kematian di banyak negara, dan seringkali gejalanya tidak terdeteksi hingga mencapai tahap yang parah. Dengan mengembangkan teknologi deteksi dini yang akurat dan efisien, seperti melalui analisis data medis yang mudah didapat seperti BMI, umur, waktu tidur, dll. serta penggunaan teknologi seperti machine learning, proyek ini bertujuan untuk memberikan akses lebih awal terhadap diagnosis penyakit jantung, memungkinkan intervensi yang tepat waktu dan penanganan yang lebih efektif, serta mengurangi angka kematian dan beban kesehatan global yang ditimbulkan oleh penyakit ini. Namun, dengan kemajuan metode machine learning yang sudah mampu menyelesaikan banyak permasalahan sehari-hari, muncul kembali permasalahan AI yang cenderung bersifat black-box sehingga banyak orang enggan percaya terhadap AI dan ML secara umum.

**Mengapa Permasalahan Harus Diselesaikan**:
- Tren penyakit jantung terus naik selama 28 tahun terakhir terutama pada lansia dan keluarga dengan ekonomi menengah kebawah, hal ini dapat mengakibatkan menurunnya kualitas hidup, kerugian ekonomi, dan kematian. (Lippi dan Sanchis-Gomar, 2020)
- Dari data di tahun 2018, Gagal jantung menyebabkan 13.4% kematian, salah satu penyebab kematian tertinggi (CDC, 2023)
- Penggunaan AI dan ML dalam bidang kesehatan masih rendah dikarenakan tingkat kepercayaan yang rendah terhadap sistem yang mana proses perhitungan sulit dijelaskan (Jermutus dkk., 2022)

## Business Understanding

### Problem Statements

- Algoritma Machine Learning Mana yang Memiliki Tingkat Explainability Tinggi dan Sesuai Untuk Data Penyakit Jantung?
- Apakah Algoritma Machine Learning Tersebut dapat Dikembangkan Agar Memiliki Akurasi yang Lebih Baik?

### Goals

- Mendapatkan Algoritma Machine Learning yang Dapat Dijelaskan Dengan Baik Sehingga Dapat Meningkatkan Kepercayaan Pengguna.
- Mengeksplorasi Kemungkinan Pengembangan Algoritma Agar Lebih Baik.

Semua poin di atas harus diuraikan dengan jelas. Anda bebas menuliskan berapa pernyataan masalah dan juga goals yang diinginkan.

### Solution statements
- Menggunakan Metode yang Sederhana Seperti Decision Tree
- Menggunakan Metode Ensemble untuk Mengembangkan Algoritma

## Data Understanding
Dataset dapat diakses pada laman berikut https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease. Data didasarkan pada BRFSS dari CDC pada tahun 2020. Data yang digunakan telah diproses untuk memproses data yang tidak konsisten dan seleksi fitur terhadap fitur yang paling berpengaruh terhadap kemungkinan seseorang memiliki penyakit jantung 

### Fitur pada dataset meliputi:
- HeartDisease : Variabel Target Biner, Apakah seseorang memiliki penyakit jantung
- BMI : Variabel Kontinu, Body Mass Index dari seseorang
- Smoking : Variabel Biner, Apakah seseorang perokok aktif
- Alcohol Drinking : Variabel Biner, Apakah seseorang sering meminum alkohol
- Stroke : Variabel Biner, Apakah seseorang memiliki riwayat stroke
- PhysicalHealth : Variabel Numerik (1-30), Dalam 30 terakhir berapa hari seseorang merasa memiliki sakit fisik
- MentalHealth : Variabel Numerik (1-30), Dalam 30 terakhir berapa hari seseorang merasa memiliki sakit mental
- DiffWalking : Variabel Biner, Apakah seseorang memiliki kesulitan berjalan
- Sex : Variabel Biner, Jenis kelamin seseorang
- AgeCategory : Variabel Ordinal, Golongan umur seseorang
- Race : Variabel Kategorikal, Suku dan ras dari seseorang
- Diabetic : Variabel Biner, Apakah seseorang memiliki riwayat diabetes
- PhysicalActivity : Variabel Biner, Apakah seseorang aktif beraktivitas fisik
- GenHealth : Variabel Ordinal (1-5), Kesehatan seseorang secara umum
- SleepTime : Variabel Numerik, Jumlah jam tidur rata-rata
- Asthma : Variabel Biner, Apakah seseorang memiliki riwayat asma
- KidneyDisease : Variabel Biner, Apakah seseorang memiliki riwayat penyakit ginjal
- SkinCancer : Variabel Biner, Apakah seseorang memiliki riwayat kanker kulit

**Data Analysis**:
Dari pengecekan singkat, data tidak memiliki nilai null dan memiliki 319795 data. Untuk distribusi data dapat dilihat pada kolase grafik berikut

![image](https://github.com/aldisk/heart-disease-risk-prediction/assets/95540779/4d5ff593-c57f-4c86-8824-2db609d6fc89)

![image](https://github.com/aldisk/heart-disease-risk-prediction/assets/95540779/6045f3e0-2a7a-4d27-8335-0982c50ae62c)

Dapat dilihat bahwa data pada data kategorikan memiliki distribusi yang tidak rata baik pada variabel target maupun fitur, Hal ini dapat terjadi mengingat data diambil dari Amerika Serikat sehingga data cenderung menggambarkan populasi penduduk Amerika Serikat. Perhatian lebih perlu diarahkan terhadap variabel target yang tidak seimbang sehingga perlu adanya pemrosesan di tahap selanjutnya. Pada data kontinu data berkecenderungan memiliki distribusi normal dengan beberapa noise. Hal tersebut dapat berupa outlier, namun ketika dijelajahi lebih lanjut data tersebut merupakan data minoritas yang masih dalam range yang wajar. Seperti pada MentalHealth dimana data memiliki satu kelompok dengan nilai di sekitar 30 yang dapat berarti orang tersebut sedang mengalami depresi.


## Data Preparation
Persiapan data berupa Imbalance Handling dan Data Encoding. Mengingat algoritma yang digunakan berupa Decision Tree, encoding yang dilakukan berupa label encoding dengan memperhatikan urutan ordinal apabila ada.

**Proses dan Alasan**: 
- Proses dimulai dengan Imbalance Handling menggunakan SMOTE-NC untuk memastikan data sintetik yang dihasilkan semirip mungkin dengan populasi data kelas minoritas. Kemudian dilakukan encoding menggunakan label encoding pada setiap fitur yang ada dengan memperhatikan urutan ordinal apabila ada. Normalisasi tidak dikarenakan Decision Tree tidak sensitif terhadap skala data karena mengambil keputusan berdasarkan persebaran data dan variabel target.
- Imbalance Handling dilakukan karena algoritma ML umumnya sensitif terhadap bias kelas yang ada. Meskipun Decision Tree umumnya cukup tahan terhadap bias tersebut, terdapat kemungkinan data yang tidak seimbang dapat berakibat pada akurasi dan proses splitting yang tidak optimal. Penggunaan SMOTE-NC diharapkan dapat menambahkan data namun tidak menambah repetisi data secara masif.
- Encoding menggunakan Label dikarenakan Decision Tree bekerja dengan memperhitungkan setiap splitting point yang mungkin sehingga Label Encoding sudah dapat dibedakan oleh Decision Tree. Selain itu One-hot Encoding dapat mendatangkan masalah dikarenakan Decision Tree akan memperhitungkan setiap fitur yang dihasilkan untuk melakukan splitting yang dapat meningkatkan kompleksitas tree namun menurunkan akurasi algoritma.

## Modeling
Digunakan 2 Algoritma dalam pemodelan, yaitu Decision Tree dan pengembangannya yaitu Random Forest

**Decision Tree
Decision Tree merupakan salah satu metode Machine Learning tradisional yang masih digunakan hingga sekarang. Cara kerja Decision Tree didasari dengan melakukan splitting pada target variabel berdasarkan nilai statistik fitur-fitur yang ada. Hal ini dilakukan dari fitur yang paling membedakan (diskriminatif) hingga didapatkan data yang pure (kondisi statistik yang menghasilkan data dengan satu kelas). Decision Tree memiliki kelebihan seperti efisiensi, seleksi fitur intrinsik, dan explainability yang sangat baik (Xu, 2019). Kelemahan metode ini adalah mungkin perlu dilakukannya feature engineering apabila data tidak dapat dipisahkan menggunakan fitur secara satu per satu.

**Random Forest
Random Forest adalah pengembangan dari Decision Tree dengan memanfaatkan beberapa Decision Tree disaat bersamaan. Random Forest menggunakan teknik bagging dimana data training diambil secara acak untuk setiap Decision Tree yang ada. Subset data acak tersebut kemudian akan dilatih pada setiap Decision Tree yang dibuat. Ketika melakukan inferensi, data akan dimasukkan ke seluruh Decision Tree yang ada dan akan diambil hasil mayoritas dari seluruh Decision Tree. Pendekatan ini meningkatkan performa Decision Tree dengan mencegah overfitting dan meningkatkan generalisasi algortima. Pengembangan ini namun memiliki kelemahan dimana terdapat sebuah keacakan yang dapat sangat berpengaruh terhadap performa algoritma dan peningkatan waktu pelatihan dan inferensi yang cukup signifikan.

## Evaluation
Untuk melakukan evaluasi terhadap algoritma yang diujikan, digunakan metrik performa akurasi, precision, recall, dan F1 score. Penggunaan akurasi sebagai metrik performa dikarenakan akurasi umumnya merupakan metrik performa yang cukup menggambarkan performa klasifikasi sebenarnya dari algoritma. Kemudian penggunaan precision, recall, dan F1 score untuk mengantisipasi apabila terdapat bias dalam evaluasi yang menyebabkan akurasi tinggi namun masih terdapat kekurangan dalam proses klasifikasi.

Perhitungan akurasi dapat dituliskan sebagai berikut

$$
\text{Akurasi} = \frac{\text{Prediksi Benar}}{\text{Total Prediksi}}
$$

Akurasi menggambarkan persentase jumlah prediksi yang benar. Namun dalam kasus tertentu akurasi kurang dapat menggambarkan performa algoritma, sehingga diperlukan metrik lainnya seperti precision, recall, dan F1 score

Perhitungan ketiga metrik tersebut dituliskan sebagai berikut

$$
\text{Recall} = \frac{\text{TruePositives}}{\text{TruePositives} + \text{FalseNegatives}}
$$

$$
\text{Precision} = \frac{\text{TruePositives}}{\text{TruePositives} + \text{FalsePositives}}
$$

$$
\text{F1 Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
$$

untuk menghitung nilai dari precision dan recall diperlukan metrik perhitungan confusion matrix yang dapat dilihat pada gambar berikut

![image](https://github.com/aldisk/heart-disease-risk-prediction/assets/95540779/0713367a-9082-4ebc-b16e-5b1f4fb10042)

Recall menggambarkan "senstivitas" algoritma terhadap data pada suatu kelas, sedangkan precision menggambarkan seberapa akurat "sensitivitas" tersebut. metrik tersebut dapat mengukur performa algoritma dalam kasus klasifikasi yang bias dengan baik dikarenakan juga memperhitungkan ketepatan klasifikasi setiap kelas secara individu. Untuk memudahkan mengukur performa algoritma dengan satu nilai, umumnya digunakan F1-Score yang menggabungkan kedua Recall dan Precision 

Dari hasil ujicoba, didapatkan akurasi pada data test sebagai berikut :

| Algoritma    | Akurasi     | F1-Score    |
|--------------|-------------|-------------|
|Decision Tree | 88%         | 88%         |
|Random Forest | 92%         | 92%         |

Dari hasil ujicoba diatas didapatkan bahwa algoritma Random Forest lebih baik dalam memperkirakan resiko seseorang terkena penyakit jantung. Akurasi dari kedua algoritma belum cukup baik untuk dijadikan sebagai sistem deteksi utama penyakit, meskipun demikian, kedua algoritma dapat digunakan sebagai sistem pembantu untuk menseleksi orang dengan resiko penyakit jantung secara cepat. Selain itu dikarenakan tingkat explainability dari Decision Tree dan Random Forest tinggi dibandingkan metode ML lainnya, knowledge yang didapatkan dari kedua metode diatas dapat diaplikasikan secara langsung untuk pengembangan metode deteksi penyakit jantung lainnya dengan basis statistika ataupun untuk memahami alur kerja algoritma untuk meningkatkan kepercayaan terhadap model yang telah dibuat

## Reference

- Lippi, G., & Sanchis-Gomar, F. (2020). Global epidemiology and future trends of heart failure. AME medical journal, 5.
- CDC. (2023, January 5th). Heart Failure. Centers for Disease Control and Prevention. https://www.cdc.gov/heartdisease/heart_failure.htm
- Jermutus, E., Kneale, D., Thomas, J., & Michie, S. (2022). Influences on user trust in healthcare artificial intelligence: A systematic review. Wellcome Open Research, 7(65), 65.
- Xu, F., Uszkoreit, H., Du, Y., Fan, W., Zhao, D., & Zhu, J. (2019). Explainable AI: A brief survey on history, research areas, approaches and challenges. In Natural Language Processing and Chinese Computing: 8th CCF International Conference, NLPCC 2019, Dunhuang, China, October 9–14, 2019, Proceedings, Part II 8 (pp. 563-574). Springer International Publishing.
