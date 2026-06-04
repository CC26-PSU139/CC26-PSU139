# CareerPath AI 🚀

**CareerPath AI** adalah platform cerdas berbasis web yang membantu pencari kerja menemukan lowongan yang paling sesuai dengan kualifikasi mereka. Cukup upload CV dalam format PDF, dan sistem akan menganalisis isinya secara otomatis menggunakan teknologi Deep Learning untuk mencocokkan dengan ribuan lowongan kerja yang tersedia.

---

## 🌐 Live Demo

| Service | URL |
|---|---|
| Frontend | https://careerpathai-front-end.vercel.app |
| Backend API | https://careerpathai-backend-chi.vercel.app |
| AI Service | https://terijuky-careerpathai.hf.space |
| API Docs (Swagger) | https://terijuky-careerpathai.hf.space/docs |

---

## 🧠 Tentang Proyek

CareerPath AI dibangun sebagai solusi atas permasalahan mismatch antara kualifikasi pencari kerja dengan kriteria lowongan yang tersedia. Sistem ini menggunakan pendekatan **Semantic Matching** berbasis Deep Learning — bukan sekadar keyword matching — sehingga mampu memahami makna dari isi CV meskipun menggunakan terminologi yang berbeda dari deskripsi lowongan.

### Alur Kerja

```
User upload CV (PDF)
    ↓
Frontend ekstrak dan kirim ke Backend
    ↓
Backend teruskan ke AI Service
    ↓
Model Deep Learning analisis dan cocokkan dengan 4000+ lowongan
    ↓
Kembalikan 10 rekomendasi terbaik
    ↓
Ditampilkan ke user beserta link lowongan
```

---

## 🏗️ Arsitektur Sistem

```
careerpath-ai/
├── client/          ← Frontend (React + Vite + Tailwind CSS)
├── server/          ← Backend (Express.js + Supabase)
├── ai-service/      ← AI Service referensi lokal
└── careerpathai/    ← AI Service deployment (Hugging Face Spaces)
```

---

## 🤖 AI Model

Model yang digunakan adalah **Dual Encoder Transformer** yang dibangun dari nol menggunakan TensorFlow (Model Subclassing), tanpa pre-trained weights.

**Arsitektur:**
- Two Tower / Dual Encoder — CV dan Job Description masing-masing punya encoder terpisah
- Custom Transformer Block (Multi-Head Attention + Feed Forward)
- Custom Layers: MyCustomDense, MyCustomLeakyReLU, MyCustomDropout, MyCustomAttention
- Custom Loss Function: Contrastive Loss
- Custom Training Loop: tf.GradientTape
- Pre-computed job vectors untuk inference yang cepat (~2-3 detik)

**Performa:**
- Accuracy: 94.37%
- Precision: 95.11%
- Recall: 93.51%
- F1-Score: 94.30%
- NDCG@10: 0.967

---

## 🛠️ Tech Stack

| Layer | Teknologi |
|---|---|
| Frontend | React, Vite, Tailwind CSS, Axios |
| Backend | Express.js, Node.js, Multer, pdf-parse |
| Database | Supabase |
| AI Model | TensorFlow, Keras (Model Subclassing) |
| AI Service | FastAPI, Uvicorn |
| Deployment | Vercel (Frontend + Backend), Hugging Face Spaces (AI Service) |

---

## 📡 API Endpoints

### Backend (Express)

| Method | Endpoint | Deskripsi |
|---|---|---|
| POST | `/api/analyze` | Upload CV (PDF atau teks) dan dapatkan rekomendasi |
| GET | `/` | Health check backend |

### AI Service (FastAPI)

| Method | Endpoint | Deskripsi |
|---|---|---|
| POST | `/predict` | Prediksi rekomendasi dari teks CV |
| GET | `/health` | Health check AI service |

Dokumentasi API lengkap tersedia di [Swagger UI](https://terijuky-careerpathai.hf.space/docs).

---

## 📊 Dataset

Data lowongan kerja diperoleh melalui scraping dari platform LinkedIn dan telah melalui proses Data Wrangling secara menyeluruh:
- Gathering & Assessing
- Cleaning (deduplication, missing values, format standardization)
- Feature Engineering (experience level extraction)
- EDA dan visualisasi tren pasar kerja

Total: **4.014 lowongan** setelah proses cleaning.

---

## 📈 Dashboard Data Science

Analisis dan visualisasi tren pasar kerja tersedia di Streamlit Cloud — menampilkan statistik skill yang paling dibutuhkan, distribusi level pengalaman, dan tren kategori pekerjaan.
Link Dashboard : https://careerpathai-capstone.streamlit.app/
---

## ⚠️ Catatan

- AI Service di Hugging Face Spaces akan sleep otomatis jika tidak ada traffic. Request pertama setelah sleep membutuhkan waktu 30-60 detik.
- Untuk hasil terbaik, gunakan CV dalam bahasa Indonesia atau Inggris dengan minimal 50 karakter teks yang dapat dibaca.
- Similarity score di bawah 50% bukan berarti tidak cocok — ini adalah jarak cosine dari model, bukan persentase kecocokan absolut. Fokus pada urutan rekomendasi, bukan angkanya.
