# Laporan Proyek Machine Learning - Asgarindo Dwiki Ibrahim Adji

## Project Overview

Di era digital saat ini, akses terhadap informasi dan bahan bacaan semakin luas. Berbagai platform digital dan toko buku daring seperti Gramedia, Togamas, dan lainnya menyediakan ribuan koleksi buku yang dapat diakses kapan saja. Namun, semakin banyak pilihan yang tersedia justru membuat pengguna mengalami kesulitan dalam memilih buku yang sesuai dengan minat atau kebutuhannya. Masalah ini dikenal sebagai information overload atau kelebihan informasi (Eppler & Mengis, 2004).

Untuk mengatasi permasalahan tersebut, banyak platform mulai mengadopsi sistem rekomendasi (recommender system), yaitu sistem yang dirancang untuk menyarankan item yang relevan berdasarkan preferensi atau riwayat pengguna. Sistem ini telah terbukti efektif di berbagai industri seperti Netflix (film), Spotify (musik), dan Amazon (produk). Dalam konteks literasi, sistem rekomendasi dapat membantu pengguna menemukan buku yang relevan dengan lebih cepat dan efisien.

Data dari UNESCO menyebut bahwa indeks minat baca masyarakat Indonesia hanya sebesar 0,001%, yang berarti dari 1.000 orang Indonesia, hanya 1 orang yang memiliki minat baca yang tinggi. Hal ini menunjukkan bahwa tingkat literasi di Indonesia masih tergolong sangat rendah dan memerlukan perhatian khusus dari berbagai pihak (Radio Republik Indonesia, 2019).

Selain itu, Kementerian Komunikasi dan Informatika Republik Indonesia (Kemenkominfo) juga pernah merilis hasil riset bertajuk World’s Most Literate Nations Ranked yang dilakukan oleh Central Connecticut State University pada Maret 2016. Dalam riset tersebut, Indonesia menempati peringkat ke-60 dari 61 negara dalam hal minat membaca, hanya berada satu tingkat di atas Botswana (61), dan tepat di bawah Thailand (59) (Radio Republik Indonesia, 2019).

oleh karena itu, membangun minat baca bukan hanya soal menyediakan bahan bacaan, tetapi juga tentang bagaimana membuat proses menemukan bahan bacaan menjadi lebih mudah, personal, dan relevan. Di sinilah sistem rekomendasi berperan penting, terutama dalam menyajikan pilihan buku berdasarkan preferensi pengguna

Referensi:
- Eppler, M. J., & Mengis, J. (2004). The Concept of Information Overload: A Review of Literature from Organization Science, Accounting, Marketing, MIS, and Related Disciplines. The Information Society, 20(5), 325–344. https://doi.org/10.1080/01972240490507974
- Radio Republik Indonesia (RRI). (2019). UNESCO Sebut Minat Baca Orang Indonesia Masih Rendah. Diakses dari: https://www.rri.co.id/daerah/649261/unesco-sebut-minat-baca-orang-indonesia-masih-rendah

## Business Understanding

### Problem Statements

- Bagaimana membangun sistem rekomendasi buku yang mampu memberikan saran buku berdasarkan judul buku?
- Bagaimana membangun sistem rekomendasi buku yang mampu memberikan saran buku berdasarkan interaksi pengguna sebelumnya?

### Goals

- Mengembangkan sistem rekomendasi yang dapat memberikan daftar buku berdasarkan content preferensi pengguna.

### Solution Statements

- Mengimplementasikan pendekatan *Content based filtering* berbasis matrix factorization.
- Menggunakan pendekatan deep learning *Collaborative Filtering* untuk meningkatkan performa model.

## Data Understanding

Dataset yang digunakan diperoleh dari [Kaggle - Book Recommendation Dataset](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset). Dataset ini berisi data pengguna, buku, dan interaksi berupa rating yang digunakan sebagai dasar dalam membangun sistem rekomendasi buku.

### 1. Deskripsi Dataset
Dataset ini terdiri dari tiga file utama:
- Users berisi data pengguna yang sudah disamarkan ID-nya (User-ID). Beberapa pengguna juga punya data tambahan seperti Location (lokasi) dan Age (usia).
- Books menyimpan informasi buku seperti ISBN, Book-Title, Book-Author, Year-Of-Publication, dan Publisher. Selain itu, ada juga link gambar sampul buku dalam tiga ukuran: kecil (Image-URL-S), sedang (Image-URL-M), dan besar (Image-URL-L).
- Ratings mencatat penilaian yang diberikan pengguna ke buku melalui Book-Rating. Penilaian ini bisa berupa nilai eksplisit (1 sampai 10), atau nilai implisit yang ditandai dengan angka 0.

| Dataset   | Jumlah Baris | Jumlah Kolom | Deskripsi Singkat                                    |
|-----------|---------------|--------------|------------------------------------------------------|
| Users     | 278.858       | 3            | Data pengguna seperti User-ID, Location, dan Age     |
| Books     | 271.379       | 8            | Informasi buku seperti ISBN, Judul, Penulis, dll     |
| Ratings   | 1.149.780     | 3            | Data interaksi berupa User-ID, ISBN, dan Rating      |

### 2. Penjelasan Fitur dan Kondisinya
#### a. Dataset `Users`
| Fitur       | Tipe Data | Deskripsi                          | Kondisi                                                                 |
|-------------|-----------|------------------------------------|--------------------------------------------------------------------------|
| `User-ID`   | Integer   | ID unik untuk setiap pengguna      | Tidak ada nilai duplikat dan tidak ada missing values.                   |
| `Location`  | String    | Lokasi pengguna (Kota, Negara)     | Tidak ada missing values, tetapi format tidak konsisten.                 |
| `Age`       | Float     | Usia pengguna                      | Terdapat nilai kosong.                        |

#### b. Dataset `Books`

| Fitur                 | Tipe Data | Deskripsi                            | Kondisi                                                                 |
|-----------------------|-----------|--------------------------------------|--------------------------------------------------------------------------|
| `ISBN`                | String    | ID unik buku                         | Tidak ada missing values dan tidak ada data duplikat.                      |
| `Book-Title`          | String    | Judul buku                           | Tidak ada missing values, ada duplikasi dengan ISBN berbeda.             |
| `Book-Author`         | String    | Nama penulis                         | Terdapat missing value.                          |
| `Year-Of-Publication` | Integer   | Tahun terbit                         | Tidak ada missing value.                                   |
| `Publisher`           | String    | Nama penerbit                        | Terdapat missing value.                                  |
| `Image-URL-S`         | String    | URL gambar kecil                     | Tidak digunakan, akan dihapus.                                           |
| `Image-URL-M`         | String    | URL gambar sedang                    | Tidak digunakan, akan dihapus.                                           |
| `Image-URL-L`         | String    | URL gambar besar                     | Terdapat mising value, Tidak digunakan, akan dihapus.                                           |

#### c. Dataset `Ratings`

| Fitur         | Tipe Data | Deskripsi                                             | Kondisi                                                                 |
|---------------|-----------|-------------------------------------------------------|--------------------------------------------------------------------------|
| `User-ID`     | Integer   | ID pengguna                                           | Tidak ada missing values, dan tidak ada data duplikat.
| `ISBN`        | String    | ID buku yang diberi rating                           | Tidak ada missing values, dan tidak ada data duplikat.                          |
| `Book-Rating` | Integer   | Nilai rating eksplisit (1–10) atau implisit (0)       | Tidak ada missing value tetapi banyak nilai 0 sebagai rating implisit (sekitar 716.000+ baris).         |

---

### 3. Analisis Kualitas Data

#### a. Missing Values

- **Age**: terdapat nilai kosong sekitar 110762
- **Book-Author**: terdapat 2 missing value
- **Publisher**: terdapat 2 missing value
- **Image URLs**: terdapat 3 missing value

#### b. Duplikasi

- Tidak ada duplikasi pada `User-ID` dan `ISBN`.
- Terdapat duplikasi judul buku sebanyak 29224 (dengan ISBN yang berbeda)

#### c. Outlier

- **Age**: tidak ada data yang <5 atau >100 tetapi tetap dibersihkan.
- **Year-Of-Publication**: Tidak ada outlier (tidak ada data yang kurang dari Tahun <1900 atau >2025) teteapi tetap di bersihkan.
- **Book-Rating**: Distribusi sangat tidak seimbang, dominan di nilai 0 (nilai implisit).

## Data Prepocessing dan Preparation

### Data Preprocessing

Data preprocessing dilakukan untuk membersihkan dan menyatukan data dari beberapa sumber agar siap digunakan untuk pemodelan sistem rekomendasi. Tahapan preprocessing dilakukan pada tiga variabel utama: books, users, dan ratings

#### 1. Preprocessing Variabel `books`

##### Cek Missing Value

- Diperiksa jumlah nilai kosong (missing value) pada setiap kolom:

    ```python
    books.isnull().sum()
    ```

##### Menghapus Data Kosong

- Menghapus baris yang memiliki nilai kosong pada kolom penting:

    ```python
    books = books.dropna(subset=['Book-Author', 'Year-Of-Publication', 'Publisher', 'Image-URL-L'])
    ```

##### Menghapus Kolom Gambar

- Karena gambar tidak digunakan dalam modeling (Drop kolom gambar):

    ```python
    books = books.drop(columns=['Image-URL-S', 'Image-URL-M', 'Image-URL-L'])
    ```

##### Cek dan Hapus Judul Buku Duplikat

- Duplikat dicek berdasarkan `Book-Title`, kemudian hanya satu data per judul yang dipertahankan:

    ```python
    books = books.drop_duplicates(subset='Book-Title', keep='first')
    ```

##### Konversi Tipe Data

- Mengubah tipe data `Year-Of-Publication` ke integer dan menyaring nilai tahun yang valid (1900–2025):

    ```python
    books['Year-Of-Publication'] = books['Year-Of-Publication'].astype(int)
    books = books[(books['Year-Of-Publication'] > 1900) & (books['Year-Of-Publication'] <= 2025)]
    ```

#### 2. Preprocessing Variabel `users`

##### Cek dan Bersihkan Umur Kosong dan Outlier

- Nilai kosong pada `Age` diisi dengan 0 lalu disaring hanya untuk rentang usia 5–80:

    ```python
    users['Age'] = users['Age'].fillna(0).astype(int)
    users = users[(users['Age'] >= 5) & (users['Age'] <= 80)]
    ```

##### Pemisahan Lokasi

- Kolom `Location` dipecah menjadi `City`, `State`, dan `Country`, lalu kolom `Location` dihapus:

    ```python
    users[['City', 'State', 'Country']] = users['Location'].str.split(',', expand=True, n=2)
    users['Country'] = users['Country'].str.strip()
    users = users.drop(columns=['Location'])
    users.dropna(subset=['City', 'State', 'Country'], inplace=True)
    ```
    
#### 3. Preprocessing Variabel `ratings`

##### Menghapus Baris dengan Rating 0

- Rating dengan nilai 0 dianggap tidak memberikan penilaian yang valid:

    ```python
    ratings = ratings[ratings['Book-Rating'] > 0]
    ```
    
#### 4. Menggabungkan Dataset
- Dataset ratings, books, dan users digabungkan menjadi satu
    ```python
    ratings_books = ratings.merge(books, on='ISBN', how='inner')
    full_data = ratings_books.merge(users, on='User-ID', how='inner')
    ```
#### 5. Menghapus Nilai 'n/a
- Membersihkan string 'n/a' dari semua kolom
    ```python
    mask = (full_data.astype(str)
             .apply(lambda col: col.str.strip().str.lower()) == 'n/a').any(axis=1)
    full_data = full_data[~mask]
    ```

### Data Preparation
Tahapan yang dilakukan dalam proses persiapan data untuk sistem rekomendasi adalah sebagai berikut:
#### 1. Sampling Data
- Mengambil sampel sebanyak 10.000 baris dari dataset awal (`full_data`) secara acak menggunakan `random_state=42`.
- Hal ini dilakukan untuk mengurangi beban komputasi saat pelatihan awal, namun tetap mempertahankan keberagaman data.

### 2. Content-Based Filtering Preparation
- Membuat kolom `all_features` dengan menggabungkan tiga fitur penting: `Book-Title`, `Book-Author`, dan `Publisher`.
- Format gabungan ini akan digunakan untuk menghitung kemiripan antar buku berdasarkan kontennya.

    ```python
    full_data["all_features"] = full_data[["Book-Title", "Book-Author", "Publisher"]].astype(str).agg(" ".join, axis=1)
    ```

### 3. Collaborative Filtering Preparation
- Mengambil kolom User-ID, ISBN, dan Book-Rating dari full_data.

    ```python
    df_collab = full_data[['User-ID', 'ISBN', 'Book-Rating']].copy()
    df_collab.rename(columns={
        'User-ID': 'userID',
        'ISBN': 'bookID',
        'Book-Rating': 'rating'
    }, inplace=True)
    ```
- Melakukan encoding manual terhadap kolom userID dan bookID ke dalam bentuk integer agar dapat digunakan oleh model

    ```python
    user_ids = df_collab['userID'].unique().tolist()
    user_to_encoded = {x: i for i, x in enumerate(user_ids)}
    df_collab['user'] = df_collab['userID'].map(user_to_encoded)
    
    book_ids = df_collab['bookID'].unique().tolist()
    book_to_encoded = {x: i for i, x in enumerate(book_ids)}
    df_collab['book'] = df_collab['bookID'].map(book_to_encoded)
    ```

### 4. Normalisasi Rating
- Mengubah nilai rating ke skala 0–1 agar sesuai dengan kebutuhan model pembelajaran mesin.
Alasan:

    ```python
    df_collab['rating'] = df_collab['rating'].astype(np.float32)
    min_rating = df_collab['rating'].min()
    max_rating = df_collab['rating'].max()
    y = df_collab['rating'].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values
    ```

### 5. Train-Test Split
- Data diacak agar distribusi data tidak bias.
- Dibagi menjadi 80% data latih dan 20% data validasi.
     ```python
    df_collab = df_collab.sample(frac=1, random_state=42)
    x = df_collab[['user', 'book']].values
    train_size = int(0.8 * len(df_collab))
    x_train, x_val = x[:train_size], x[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    ```

##### Alasan dan Tujuan
- Sampling: Mempercepat eksperimen awal dengan mengurangi jumlah data.
- Encoding ID: Agar ID pengguna dan buku bisa diproses oleh model sebagai input numerik.
- Normalisasi: Membantu model belajar lebih cepat dan stabil.
- Train-Test Split: Penting untuk mengukur performa model dalam memprediksi data baru (unseen data).

## Modeling

Pada tahap ini, dilakukan implementasi dan evaluasi dua pendekatan utama dalam sistem rekomendasi buku:
- Content-Based Filtering
- Collaborative Filtering

Setiap pendekatan disesuaikan dengan karakteristik data dan tujuan sistem, serta diuji menggunakan data nyata untuk menghasilkan rekomendasi berbasis input pengguna

### Content-Based Filtering

Pendekatan ini merekomendasikan buku berdasarkan kemiripan kontennya, dalam hal ini fitur judul, penulis, dan penerbit, yang telah digabung ke dalam kolom all_features. Dua metode digunakan untuk mengekstraksi fitur teks: CountVectorizer dan TfidfVectorizer. Kemiripan antar buku dihitung menggunakan Cosine Similarity.

Cara kerja Cosine Similarity:
Cosine similarity menghitung sudut kosinus antara dua vektor dalam ruang multidimensi. Nilai hasil berkisar antara 0 (tidak mirip) hingga 1 (identik).

rumus => (A, B) = (A . B) / (||A|| * ||B||)
 
Metode ini cocok untuk teks karena mengukur kemiripan struktur kata, bukan nilai absolut.

### 1. CountVectorizer
#### Cara Kerja
- CountVectorizer mengubah teks menjadi representasi numerik berbasis frekuensi kata.
- Setiap dokumen (dalam hal ini data fitur buku) dikonversi menjadi vektor berdasarkan jumlah kemunculan kata (bag of words)
- Kata-kata umum akan memiliki bobot tinggi, sehingga buku dengan banyak kata sama akan memiliki vektor yang mirip


#### Kemiripan
Digunakan Cosine Similarity untuk mengukur kedekatan antar-vektor, Cosine similarity menghitung sudut antara dua vektor; nilai 1 artinya identik, dan 0 artinya tidak mirip.

#### Implementasi

    ```python
    vectorizer_cv = CountVectorizer()
    vectors_cv = vectorizer_cv.fit_transform(full_data["all_features"])
    similarity_cv = cosine_similarity(vectors_cv)
    ```
#### Hasil (Result)

Dalam kode, pengguna mencari rekomendasi berdasarkan buku "Stanislaski Sisters".

| Book Title                                | Author        | Publisher   | Similarity Score |
|-------------------------------------------|---------------|-------------|------------------|
| Stanislaski Sisters                       | Nora Roberts  | Silhouette  | 1.000            |
| Stanislaski Brothers (Silhouette Promo)   | Nora Roberts  | Silhouette  | 0.745            |
| Mysterious                                | Nora Roberts  | Silhouette  | 0.671            |
| Summer Pleasures                          | Nora Roberts  | Silhouette  | 0.600            |
| Cordina's Crown Jewel                     | Nora Roberts  | Silhouette  | 0.548            |

Semua hasil yang direkomendasikan memiliki kesamaan tinggi dalam hal penulis dan penerbit, menunjukkan efektivitas pendekatan berbasis konten (content-based filtering) dalam menyarankan buku-buku yang serupa

### 2. TfidfVectorizer
#### Cara Kerja
- TfidfVectorizer (Term Frequency-Inverse Document Frequency) memberikan bobot penting berdasarkan seberapa unik sebuah kata dalam keseluruhan dokumen.
- Kata yang sering muncul di satu dokumen tapi jarang di dokumen lain akan diberi bobot lebih tinggi.
- Ini membantu menurunkan pengaruh kata-kata umum dan meningkatkan akurasi kemiripan semantik.

#### Kemiripan
Masih menggunakan Cosine Similarity.

#### Implementasi
    ```python
    vectorizer_tv = TfidfVectorizer()
    vectors_tv = vectorizer_tv.fit_transform(full_data["all_features"])
    similarity_tv = cosine_similarity(vectors_tv)
    ```
##### Hasil (Result)
Dengan input buku "Stanislaski Sisters", sistem juga memberikan hasil yang serupa, namun skor kemiripan bisa sedikit berbeda tergantung bobot kata

| Book Title                                | Author        | Publisher   | Similarity Score |
|-------------------------------------------|---------------|-------------|------------------|
| Stanislaski Sisters                       | Nora Roberts  | Silhouette  | 1.000            |
| Stanislaski Brothers (Silhouette Promo)   | Nora Roberts  | Silhouette  | 0.683            |
| Mysterious                                | Nora Roberts  | Silhouette  | 0.516            |
| Time And Again                            | Nora Roberts  | Silhouette  | 0.469            |
| Summer Pleasures                          | Nora Roberts  | Silhouette  | 0.451            |

Rekomendasi yang dihasilkan menunjukkan bahwa pendekatan TF-IDF berhasil mengidentifikasi buku-buku dengan kemiripan konten teks yang tinggi.

### Collaborative Filtering

Pendekatan ini merekomendasikan buku berdasarkan pola interaksi pengguna, yaitu rating. Model ini mempelajari representasi (embedding) pengguna dan buku dalam bentuk vektor laten untuk memprediksi kecocokan

### Arsitektur Model

RecommenderNet adalah sebuah custom model Keras (tf.keras.Model) yang dibangun untuk mempelajari representasi pengguna dan item (buku) dalam bentuk embedding vector, sehingga dapat menghitung kecocokan (similarity) antara user dan item. Semakin cocok, semakin besar kemungkinan user menyukai item tersebut.

```python
class RecommenderNet(tf.keras.Model):
    def __init__(self, num_users, num_books, embedding_size=50, **kwargs):
        super(RecommenderNet, self).__init__(**kwargs)
        ...
        
    def call(self, inputs):
        ...
        return tf.nn.sigmoid(x)
```

### Spesifikasi Model:
- Loss Function: Binary Crossentropy
- Optimizer: Adam
- Validation Split: 20%
- Epochs: max 100, menggunakan EarlyStopping dengan patience 5
- Output: Probabilitas rating tinggi

### Cara Kerja:
- Model mempelajari preferensi pengguna dan karakteristik buku melalui embedding.
- Hasil dot product antara dua embedding menunjukkan kecocokan.
- Semakin tinggi nilai sigmoid(dot_product), semakin besar kemungkinan user akan menyukai buku tersebut.

#### Hasil (Result)
Hasil Rekomendasi Buku untuk User ID: 211426
Buku-Buku dengan Rating Tinggi dari User
| No. | ISBN        | Book Title                                                       |
|-----|-------------|-------------------------------------------------------------------|
| 1   | 0345337662  | *Interview with the Vampire*                                     |
| 2   | 0060973897  | *Lakota Woman*                                                   |
| 3   | 0140236864  | *The Penguin Gandhi Reader*                                      |
| 4   | 0671617028  | *The Color Purple*                                               |
| 5   | 080213095X  | *World of the Buddha: An Introduction to Buddhist Literature*    |

Top 10 Book Recommendations

| No. | ISBN        | Book Title                                                                 |
|-----|-------------|-----------------------------------------------------------------------------|
| 1   | 0385316895  | *Legacy of Silence*                                                         |
| 2   | 0671003364  | *Ransom*                                                                    |
| 3   | 0671705091  | *A Knight in Shining Armor*                                                 |
| 4   | 0671741039  | *Swan Song*                                                                 |
| 5   | 0842329218  | *Tribulation Force: The Continuing Drama of Those Left Behind*             |
| 6   | 0684835983  | *Before I Say Good-Bye: A Novel*                                            |
| 7   | 0786868015  | *The Diary of Ellen Rimbauer: My Life at Rose Red*                         |
| 8   | 1558748865  | *Chicken Soup for the Gardener's Soul*                                     |
| 9   | 0064407683  | *The Wide Window (A Series of Unfortunate Events, Book 3)*                 |
| 10  | 0941524841  | *Empowerment Through Reiki*                                                |

Model berhasil merekomendasikan buku yang sejalan dengan minat user berdasarkan histori rating yang diberikan. Buku-buku yang direkomendasikan mencakup genre fiksi, misteri, fantasi, hingga thriller, yang cocok dengan preferensi pembaca berdasarkan buku yang telah dinilai tinggi sebelumnya.

# Evaluation

### Evalusasi Model
##### Content-Based Filtering – Top-K Ranking Metrics
Content-based tidak menggunakan rating eksplisit, sehingga evaluasi lebih fokus pada relevansi hasil rekomendasi. Berikut adalah metrik evaluasi yang sesuai (belum seluruhnya diimplementasikan dalam kode, tapi menjadi rujukan penting ke depan):
- Precision@K: Seberapa banyak dari top-K rekomendasi yang relevan.
- Recall@K: Seberapa banyak item relevan berhasil ditemukan dari seluruh item relevan.
- F1-Score@K: Harmonik dari precision dan recall

Rumus Evaluasi:
Jika:
- TP = jumlah buku yang direkomendasikan dan relevan (benar),
- FP = jumlah buku yang direkomendasikan tapi tidak relevan (salah),
- FN = jumlah buku yang relevan tapi tidak direkomendasikan,

Maka:
```
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1-Score = 2 * (Precision * Recall) / (Precision + Recall)
```
Rumus-rumus ini diterapkan dalam fungsi `evaluasi_rekomendasi(...)` untuk mengevaluasi hasil rekomendasi buku berdasarkan judul referensi dan ground truth yang diberikan.

Buku yang digunakan sebagai acuan rekomendasi:  
**"Stanislaski Sisters"**

Ground truth (judul yang relevan secara tematik atau penulis):
- Stanislaski Brothers (Silhouette Promo)
- Mysterious
- Time And Again
- Summer Pleasures

##### Hasil Evaluasi
Hasil evaluasi metrik content based filtering
| Metode             | Precision | Recall | F1-Score |
|--------------------|-----------|--------|----------|
| CountVectorizer    | 0.6	    | 0.75  | 0.667    |
| TfidfVectorizer    | 0.8	    | 1.00 | 0.889    |

Hasil evaluasi menunjukkan bahwa TF-IDF Vectorizer memberikan performa yang lebih unggul dibanding CountVectorizer pada ketiga metrik utama: Precision, Recall, dan F1-Score. Nilai Recall sebesar 1.00 menandakan bahwa semua item relevan berhasil direkomendasikan, dan F1-Score yang tinggi mencerminkan keseimbangan antara ketepatan dan kelengkapan sistem rekomendasi ini.

##### Collaborative Filtering - RMSE (Root Mean Squared Error)
Rumus Evaluasi:

```
RMSE = √(1/n ∑(yᵢ - ŷᵢ)²)
```

- RMSE rendah menandakan prediksi model mendekati rating aktual pengguna.
- Digunakan untuk mengevaluasi performa model collaborative filtering selama proses training dan validasi.

Buku yang digunakan sebagai acuan rekomendasi:
**"Stanislaski Sisters"**

##### Hasil Evaluasi:
Hasil evaluasi training model collaborative filtering:
| Epoch | Train RMSE | Validation RMSE |
|-------|------------|-----------------|
| 1     | ≈ 0.31     | ≈ 0.30          |
| 100   | ≈ 0.04     | ≈ 0.27          |

Train RMSE menunjukkan penurunan yang signifikan dan konsisten, dari sekitar 0.31 menjadi sekitar 0.04, yang mengindikasikan bahwa model mampu mempelajari pola dari data pelatihan dengan sangat baik. Validation RMSE juga menurun, namun dengan laju yang lebih lambat. Pada akhir epoch, nilai RMSE validasi berada di sekitar 0.27.

### Hubungan dengan Business Understanding
Model yang dievaluasi dalam proyek ini menunjukkan dampak yang relevan dan signifikan terhadap elemen-elemen utama dalam Business Understanding, yaitu problem statement, goals, dan solution statement.

##### Apakah Sudah Menjawab Setiap Problem Statement?
- **Bagaimana membangun sistem rekomendasi buku yang mampu memberikan saran buku berdasarkan judul buku?**
Model Content-Based Filtering berhasil menjawab tantangan ini dengan memanfaatkan representasi fitur teks (CountVectorizer dan TfidfVectorizer). Evaluasi menunjukkan bahwa model dapat merekomendasikan buku yang mirip secara konten terhadap buku yang disukai pengguna.
- **Bagaimana membangun sistem rekomendasi buku yang mampu memberikan saran buku berdasarkan interaksi pengguna sebelumnya?**
Model Collaborative Filtering berbasis neural network mampu mempelajari preferensi pengguna melalui histori rating dan memberikan rekomendasi yang relevan. Hal ini terbukti dari nilai RMSE yang rendah pada data validasi.

##### Apakah Goals Tercapai?
- Kedua pendekatan berhasil diimplementasikan. Content-Based Filtering menangani cold start dan memberi rekomendasi berbasis konten, sementara Collaborative Filtering memberikan rekomendasi berdasarkan pola interaksi pengguna.
- Model Collaborative Filtering menunjukkan performa yang baik dengan RMSE validasi sekitar 0.27, sementara Content-Based Filtering mencapai metrik evaluasi Precision, Recall, dan F1-Score yang tinggi, khususnya dengan TfidfVectorizer.
- Rekomendasi yang dihasilkan mencerminkan preferensi pengguna berdasarkan data historis dan konten, sesuai dengan tujuan sistem rekomendasi personalisasi.

##### Apakah Setiap Solusi Statement Berdampak?
- Sistem ini bisa memberikan rekomendasi buku berdasarkan kemiripan isi buku, dan tetap bisa bekerja meskipun pengguna belum pernah memberikan rating sebelumnya (masalah ini disebut cold start).
- Model juga menunjukkan hasil yang akurat saat dilatih, dan cukup stabil saat diuji. Ini menunjukkan bahwa sistem mampu menyesuaikan rekomendasi sesuai kebiasaan membaca pengguna dari waktu ke waktu.

### Kesimpulan

Proyek ini berhasil membuktikan bahwa sistem rekomendasi hybrid dengan content-based dan collaborative filtering dapat dibangun untuk meningkatkan pengalaman pengguna dalam memilih buku. Evaluasi menunjukkan bahwa:
- Collaborative filtering memberikan prediksi akurat terhadap rating (RMSE rendah).
- Content-based filtering memberikan alternatif relevan, terutama saat data pengguna terbatas.
