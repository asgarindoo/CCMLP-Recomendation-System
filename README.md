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

Dataset yang digunakan diperoleh dari [kagle](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset), yang mencakup informasi mengenai:
- Users berisi data pengguna yang sudah disamarkan ID-nya (User-ID). Beberapa pengguna juga punya data tambahan seperti Location (lokasi) dan Age (usia).
- Books menyimpan informasi buku seperti ISBN, Book-Title, Book-Author, Year-Of-Publication, dan Publisher. Selain itu, ada juga link gambar sampul buku dalam tiga ukuran: kecil (Image-URL-S), sedang (Image-URL-M), dan besar (Image-URL-L).
- Ratings mencatat penilaian yang diberikan pengguna ke buku melalui Book-Rating. Penilaian ini bisa berupa nilai eksplisit (1 sampai 10), atau nilai implisit yang ditandai dengan angka 0.

Jumlah data:
- Jumlah pengguna: 278,858
- Jumlah buku: 271,379
- Jumlah interaksi (rating): 1,149,780

Fitur penting:
- `user_id` : ID unik pengguna
- `ISBN` : ID unik buku
- `book_title` : Judul buku
- `book_author` : Penulis buku
- `rating` : Nilai rating dari pengguna (1–10)

## Data Preparation

Tahapan yang dilakukan dalam proses persiapan data untuk sistem rekomendasi adalah sebagai berikut:
### 1. Sampling Data
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

### Content-Based Filtering

Pendekatan ini merekomendasikan buku berdasarkan kemiripan konten (judul).

### 1. CountVectorizer
- Mengubah data teks menjadi representasi frekuensi kata.
- Kemiripan dihitung menggunakan **cosine similarity**.
- Fungsi `content_based_CountVectorizer(book_title, n_recommendations)` digunakan untuk mencari buku yang mirip berdasarkan judul yang dimasukkan.
    ```python
    vectorizer_cv = CountVectorizer()
    vectors_cv = vectorizer_cv.fit_transform(full_data["all_features"])
    similarity_cv = cosine_similarity(vectors_cv)
    ```

### 2. TfidfVectorizer
- Menggunakan bobot TF-IDF untuk memberi nilai penting pada kata-kata unik.
- Kemiripan dihitung dengan **cosine similarity**.
- Fungsi `content_based_TfidfVectorizer(book_title, n_recommendations)` mencari rekomendasi berdasarkan judul buku.
    ```python
    vectorizer_tv = TfidfVectorizer()
    vectors_tv = vectorizer_tv.fit_transform(full_data["all_features"])
    similarity_tv = cosine_similarity(vectors_tv)
    ```

### Collaborative Filtering

Pendekatan ini menggunakan **interaksi pengguna dan buku (rating)**.

- Model dibangun menggunakan Neural Network (TensorFlow Keras) dengan teknik embedding.
- Prediksi rating dilakukan dengan fungsi aktivasi sigmoid.
- Dataset dibagi 80:20 untuk training dan validasi.
- Menggunakan binary crossentropy sebagai loss function.
- EarlyStopping digunakan untuk mencegah overfitting.

### Arsitektur Model

```python
class RecommenderNet(tf.keras.Model):
    def __init__(self, num_users, num_books, embedding_size=50, **kwargs):
        super(RecommenderNet, self).__init__(**kwargs)
        ...
        
    def call(self, inputs):
        ...
        return tf.nn.sigmoid(x)
```

# Evaluation

### Metrik yang Digunakan - RMSE (Root Mean Squared Error)

Rumus:

```
RMSE = √(1/n ∑(yᵢ - ŷᵢ)²)
```

- RMSE rendah menandakan prediksi model mendekati rating aktual pengguna.
- Digunakan untuk mengevaluasi performa model collaborative filtering selama proses training dan validasi.

### Hasil Evaluasi:

- RMSE Awal (Epoch 1):
  - Train: ≈ 0.31
  - Validation: ≈ 0.30
- RMSE Akhir (Epoch 100):
  - Train: ≈ 0.04
  - Validation: ≈ 0.27

**Train RMSE** menunjukkan penurunan yang signifikan dan konsisten, dari sekitar 0.31 menjadi sekitar 0.04, yang mengindikasikan bahwa model mampu mempelajari pola dari data pelatihan dengan sangat baik. **Validation RMSE** juga menurun, namun dengan laju yang lebih lambat. Pada akhir epoch, nilai RMSE validasi berada di sekitar 0.27.

## Kesimpulan

Proyek ini membuktikan bahwa sistem rekomendasi dapat dibangun dengan pendekatan **content-based** dan **collaborative filtering** untuk memberikan rekomendasi buku yang lebih personal. Berikut beberapa temuan penting:

### Keberhasilan

- **Content-based filtering** memberikan rekomendasi yang cukup relevan berdasarkan judul dan metadata buku.
- **Collaborative filtering berbasis neural network** mampu belajar dari pola interaksi pengguna dengan performa RMSE yang cukup baik.
- Kombinasi kedua pendekatan ini berpotensi meningkatkan pengalaman pengguna dalam memilih buku sesuai preferensi mereka.

### Tantangan

- **Overfitting** menjadi kendala utama pada model collaborative filtering, ditunjukkan oleh gap besar antara RMSE training dan validation.
- **Keterbatasan data eksplisit** (rating) menyulitkan model untuk benar-benar memahami preferensi pengguna secara mendalam.
- **Cold start problem** masih menjadi tantangan, terutama untuk pengguna atau buku baru yang belum memiliki cukup interaksi.