# SMS Spam Classification - Eksperimen ML dengan DagsHub

**Author:** Mochammad Ariel Sulton  
**Username Dicoding:** arielsulton  
**Dataset:** SMS Spam Collection dari UCI Machine Learning Repository  
**DagsHub Repository:** https://dagshub.com/arielsulton/sms-spam-mlops

---

## 📋 Deskripsi Proyek

Proyek ini merupakan submission untuk kelas **"Membangun Sistem Machine Learning"** di Dicoding. Proyek ini mengimplementasikan sistem klasifikasi SMS spam menggunakan Machine Learning dengan pendekatan MLOps yang lengkap, terintegrasi dengan **DagsHub** untuk experiment tracking.

### Fitur Utama
- ✅ Preprocessing otomatis dengan GitHub Actions
- ✅ Eksplorasi data lengkap dengan Jupyter Notebook
- ✅ Training 3 model dengan hyperparameter tuning
- ✅ Integrasi DagsHub untuk MLflow tracking
- ✅ Visualisasi lengkap (confusion matrix, ROC curve, model comparison)
- ✅ Artifact versioning dengan DagsHub

---

## 📁 Struktur Folder

```
Eksperimen_SML_MochammadArielSulton/
├── .github/
│   └── workflows/
│       └── preprocessing.yml                # ✅ GitHub Actions untuk preprocessing otomatis
│
├── .workflow/
│   └── preprocessing.yml
│
├── sms_spam_raw/
│   ├── SMSSpamCollection                    # Dataset asli (5,574 SMS)
│   └── readme                               # Dokumentasi dataset UCI
│
├── preprocessing/
│   ├── automate_MochammadArielSulton.py     # ✅ Script preprocessing otomatis
│   ├── Eksperimen_MochammadArielSulton.ipynb
│   ├── sms_spam_preprocessing.csv           # Dataset hasil preprocessing (5,169 SMS)
│   └── requirements.txt                     # Dependencies untuk preprocessing
│
├── Membangun_model/
│   ├── modelling.py                         # Training model basic dengan MLflow autolog
│   ├── modelling_tuning.py                  # ✅ Training dengan hyperparameter tuning + DagsHub
│   ├── sms_spam_preprocessing.csv           # Dataset untuk training
│   ├── requirements.txt                     # Dependencies untuk model training
│   ├── DagsHub.txt                          # Informasi DagsHub repository
│   ├── DOCKER_RUN_ID.txt                    # Run ID untuk Docker build
│   │
│   ├── artifacts/                           # ✅ Model artifacts
│   │   ├── model_comparison_tuned.csv       # Hasil perbandingan model
│   │   ├── vectorizer_logistic_regression_(tuned).pkl
│   │   ├── vectorizer_naive_bayes_(tuned).pkl
│   │   └── vectorizer_random_forest_(tuned).pkl
│   │
│   ├── screenshots/                         # ✅ Visualisasi hasil training
│   │   ├── confusion_matrix_logistic_regression_(tuned).png
│   │   ├── confusion_matrix_naive_bayes_(tuned).png
│   │   ├── confusion_matrix_random_forest_(tuned).png
│   │   ├── roc_curve_logistic_regression_(tuned).png
│   │   ├── roc_curve_naive_bayes_(tuned).png
│   │   ├── roc_curve_random_forest_(tuned).png
│   │   └── model_comparison_tuned.png
│   │
│   ├── screenshoot_dashboard.png            # ✅ Screenshot DagsHub dashboard
│   └── screenshoot_artifak.png              # ✅ Screenshot DagsHub artifacts
│
├── .workflow/                               # (Legacy - bisa dihapus)
│   └── preprocessing.yml                    # Versi lama workflow
│
├── README.md                                # Dokumentasi proyek (file ini)
└── Eksperimen_SML_MochammadArielSulton.txt  # File informasi tambahan
```

---

## 🚀 Cara Menggunakan

### 1. Setup Environment

```bash
# Clone repository
git clone https://github.com/arielsulton/Eksperimen_SML_MochammadArielSulton.git
cd Eksperimen_SML_MochammadArielSulton

# Install dependencies untuk preprocessing
cd preprocessing
pip install -r requirements.txt
```

### 2. Dataset

Dataset **SMS Spam Collection** sudah tersedia di folder `sms_spam_raw/`.

**Statistik Dataset:**
- Total SMS: 5,574 pesan
- Setelah preprocessing: 5,169 pesan
- Kelas: 2 (ham/legitimate dan spam)
- Format: Tab-separated values
- Sumber: [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)

Jika perlu download ulang:
```bash
cd sms_spam_raw
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip
unzip smsspamcollection.zip
```

### 3. Jalankan Preprocessing Otomatis

```bash
cd preprocessing
python automate_MochammadArielSulton.py
```

**Output:**
- `sms_spam_preprocessing.csv` - Dataset siap untuk training

### 4. Eksplorasi Data dengan Jupyter Notebook

```bash
jupyter notebook Eksperimen_MochammadArielSulton.ipynb
```

**Isi Notebook:**
- Data loading dan inspeksi
- Exploratory Data Analysis (EDA)
- Visualisasi distribusi spam vs ham
- Text preprocessing
- Feature engineering dengan TF-IDF

### 5. Training Model

#### Setup DagsHub (Opsional - untuk Advanced Level)

```bash
# Install dagshub
pip install dagshub

# Set environment variables (atau gunakan .env file)
export DAGSHUB_USER_TOKEN="your_token_here"
```

**DagsHub Repository:** https://dagshub.com/arielsulton/sms-spam-mlops

#### Opsi A: Training Dasar dengan MLflow Autolog

```bash
cd Membangun_model
pip install -r requirements.txt
python modelling.py
```

**Fitur:**
- MLflow autolog untuk tracking otomatis
- Model: Logistic Regression, Naive Bayes, Random Forest
- Metrics: Accuracy, Precision, Recall, F1-Score

#### Opsi B: Training dengan Hyperparameter Tuning (Recommended - Advanced Level)

```bash
cd Membangun_model
python modelling_tuning.py
```

**Fitur:**
- ✅ Hyperparameter tuning dengan GridSearchCV
- ✅ Manual logging ke DagsHub
- ✅ Artifact logging (vectorizer, model comparison, plots)
- ✅ 3 Model comparison:
  - Logistic Regression (Best: **0.9894 accuracy**)
  - Naive Bayes
  - Random Forest

**Best Model:**
- Model: Logistic Regression (Tuned)
- Accuracy: **0.9894** (98.94%)
- Run ID: `0e742c818f084587a063836c0060db99`
- DagsHub URL: https://dagshub.com/arielsulton/sms-spam-mlops/experiments

### 6. Melihat Hasil Eksperimen

#### DagsHub (Recommended)
Buka: https://dagshub.com/arielsulton/sms-spam-mlops

**Fitur:**
- Experiment comparison
- Artifact versioning
- Model registry
- Collaboration tools

#### MLflow Local (Opsional)
```bash
mlflow ui --port 5000
```
Buka browser: `http://localhost:5000`

---

## 📊 Hasil Eksperimen

### Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **Logistic Regression (Tuned)** | **0.9894** | 0.9856 | 0.9726 | 0.9791 |
| Naive Bayes (Tuned) | 0.9806 | 0.9565 | 0.9589 | 0.9577 |
| Random Forest (Tuned) | 0.9787 | 0.9711 | 0.9452 | 0.9580 |

### Visualisasi

Lihat folder `Membangun_model/screenshots/` untuk:
- ✅ Confusion matrix untuk setiap model
- ✅ ROC curves dengan AUC scores
- ✅ Model comparison chart
- ✅ Feature importance (Random Forest)

### DagsHub Screenshots

- ✅ `screenshoot_dashboard.png` - DagsHub experiments dashboard
- ✅ `screenshoot_artifak.png` - DagsHub artifacts page

---

## 🎯 Kriteria Submission

### ✅ Kriteria 1: Melakukan Eksperimen Dataset (4/4 Points)

**Basic (2 pts):**
- ✅ Template eksperimen digunakan sebagai struktur notebook
- ✅ Preprocessing otomatis dengan `automate_MochammadArielSulton.py`

**Skilled (3 pts):**
- ✅ Data loading, EDA, dan preprocessing lengkap di notebook
- ✅ Visualisasi dan insights dari data

**Advanced (4 pts):**
- ✅ GitHub workflow untuk preprocessing automation
- ✅ Automated data validation
- ✅ File: `.github/workflows/preprocessing.yml`

---

### ✅ Kriteria 2: Membangun Model Machine Learning (4/4 Points)

**Basic (2 pts):**
- ✅ `modelling.py` dengan MLflow autolog
- ✅ Minimal 1 model trained

**Skilled (3 pts):**
- ✅ Manual logging (parameters, metrics)
- ✅ Minimal 2 artifact tambahan selain autolog

**Advanced (4 pts):**
- ✅ `modelling_tuning.py` dengan hyperparameter tuning
- ✅ 3 models trained dan dibandingkan
- ✅ **DagsHub integration** untuk remote tracking
- ✅ Screenshot dashboard MLflow/DagsHub
- ✅ 6+ artifacts logged:
  1. TF-IDF Vectorizers (3 files)
  2. Model comparison CSV
  3. Confusion matrices (3 files)
  4. ROC curves (3 files)

---

## 🔧 Teknologi yang Digunakan

### Core Technologies
- Python 3.10+
- Scikit-learn (ML algorithms)
- Pandas & NumPy (Data processing)
- Matplotlib & Seaborn (Visualization)

### MLOps Stack
- **MLflow 2.19.0** - Experiment tracking
- **DagsHub** - Remote MLflow tracking & collaboration
- **GitHub Actions** - CI/CD automation

### ML Models
- Logistic Regression (with TF-IDF)
- Multinomial Naive Bayes
- Random Forest Classifier

---

## 📝 Catatan Penting

### Untuk Submission Dicoding:

1. **DagsHub Screenshots:** Sudah tersedia di `Membangun_model/`
   - `screenshoot_dashboard.png`
   - `screenshoot_artifak.png`

2. **GitHub Workflow:** Tersedia di `.github/workflows/preprocessing.yml`
   - Jangan gunakan `.workflow/` (folder legacy)

3. **Best Model Run ID:** `0e742c818f084587a063836c0060db99`
   - Simpan untuk Docker build di Kriteria 3

4. **Artifacts:** Semua artifacts tersimpan di:
   - DagsHub: https://dagshub.com/arielsulton/sms-spam-mlops
   - Local: `Membangun_model/artifacts/`

### Tips:

- ✅ Pastikan DagsHub token tersedia untuk training
- ✅ Gunakan `modelling_tuning.py` untuk hasil terbaik
- ✅ Screenshot harus menunjukkan username `arielsulton`
- ✅ Push ke GitHub sebagai repository PUBLIC

---

## 🔗 Links

- **DagsHub Repository:** https://dagshub.com/arielsulton/sms-spam-mlops
- **GitHub Repository:** https://github.com/arielsulton/Eksperimen_SML_MochammadArielSulton
- **Best Run:** https://dagshub.com/arielsulton/sms-spam-mlops/experiments/#/0e742c818f084587a063836c0060db99

---

## 👨‍💻 Author

**Mochammad Ariel Sulton**  
Dicoding Username: `arielsulton`  
DagsHub: https://dagshub.com/arielsulton

---

## 📜 License

Dataset: [UCI Machine Learning Repository License](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)

---