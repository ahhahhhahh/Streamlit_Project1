# ============================================================
# APLIKASI MACHINE LEARNING DENGAN STREAMLIT
# Regresi & Klasifikasi - Toy Datasets dari scikit-learn
# ============================================================

# --- Import library ---
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# Scikit-learn: dataset & utilitas
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

# Scikit-learn: model regresi
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# Scikit-learn: model klasifikasi
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

# --- Konfigurasi halaman (layout wide) ---
st.set_page_config(
    page_title="ML Regresi & Klasifikasi",
    page_icon="🤖",
    layout="wide",
)

# ============================================================
# DAFTAR PILIHAN
# ============================================================
DATASET_REGRESI = ["Diabetes", "Linnerud (Target: Weight)"]
DATASET_KLASIFIKASI = ["Iris", "Wine", "Breast Cancer", "Digits"]

MODEL_REGRESI = [
    "Linear Regression", "Ridge Regression", "Lasso Regression",
    "Decision Tree", "Random Forest",
]
MODEL_KLASIFIKASI = [
    "Logistic Regression", "KNN", "Decision Tree",
    "Random Forest", "SVM",
]


# ============================================================
# FUNGSI BANTU
# ============================================================
@st.cache_data
def muat_dataset(jenis: str, nama: str):
    """Memuat toy dataset sklearn dan di-cache supaya tidak dimuat ulang."""
    kelas = None
    if jenis == "Regresi":
        if nama == "Diabetes":
            d = datasets.load_diabetes(as_frame=True)
            X, y = d.data, d.target
        else:  # Linnerud: ambil kolom Weight sebagai target
            d = datasets.load_linnerud(as_frame=True)
            X, y = d.data, d.target["Weight"]
    else:
        pemuat = {
            "Iris": datasets.load_iris,
            "Wine": datasets.load_wine,
            "Breast Cancer": datasets.load_breast_cancer,
            "Digits": datasets.load_digits,
        }[nama]
        d = pemuat(as_frame=True)
        X, y = d.data, d.target
        kelas = [str(k) for k in d.target_names]
    return X, y, kelas


def buat_model(jenis: str, nama: str, seed: int):
    """Membuat model sesuai pilihan. Model sensitif skala dibungkus StandardScaler."""
    if jenis == "Regresi":
        model = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=1.0))]),
            "Lasso Regression": Pipeline([("sc", StandardScaler()), ("m", Lasso(alpha=0.1))]),
            "Decision Tree": DecisionTreeRegressor(max_depth=5, random_state=seed),
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=seed),
        }
    else:
        model = {
            "Logistic Regression": Pipeline([("sc", StandardScaler()), ("m", LogisticRegression(max_iter=1000))]),
            "KNN": Pipeline([("sc", StandardScaler()), ("m", KNeighborsClassifier(n_neighbors=5))]),
            "Decision Tree": DecisionTreeClassifier(random_state=seed),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=seed),
            "SVM": Pipeline([("sc", StandardScaler()), ("m", SVC(kernel="rbf"))]),
        }
    return model[nama]


def latih_dan_evaluasi(jenis, nama_dataset, nama_model, test_size, seed):
    """Melatih model, menghitung metrics, lalu mengembalikan semuanya dalam dict."""
    X, y, kelas = muat_dataset(jenis, nama_dataset)

    # Pembagian data (stratify untuk klasifikasi agar proporsi kelas terjaga)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed,
        stratify=y if jenis == "Klasifikasi" else None,
    )

    model = buat_model(jenis, nama_model, seed)
    mulai = time.time()
    model.fit(X_train, y_train)
    waktu = time.time() - mulai
    y_pred = model.predict(X_test)

    hasil = {
        "jenis": jenis, "dataset": nama_dataset, "model": nama_model,
        "waktu": waktu, "n_train": len(X_train), "n_test": len(X_test),
        "y_test": np.asarray(y_test), "y_pred": np.asarray(y_pred), "kelas": kelas,
    }

    if jenis == "Regresi":
        mse = mean_squared_error(y_test, y_pred)
        hasil["metrics"] = {
            "MAE": mean_absolute_error(y_test, y_pred),
            "MSE": mse,
            "RMSE": float(np.sqrt(mse)),
            "R²": r2_score(y_test, y_pred),
        }
    else:
        hasil["metrics"] = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "Recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "F1-Score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        }
        hasil["report_dict"] = classification_report(
            y_test, y_pred, target_names=kelas, output_dict=True, zero_division=0)
        hasil["report_text"] = classification_report(
            y_test, y_pred, target_names=kelas, zero_division=0)
    return hasil


# ============================================================
# SIDEBAR: PENGATURAN MODEL
# ============================================================
with st.sidebar:
    st.header("⚙️ Pengaturan Model")
    st.caption("Atur parameter di bawah, lalu klik **Train Model**.")

    jenis = st.radio("🎯 Pilih Jenis Model", ["Regresi", "Klasifikasi"], horizontal=True)

    daftar_dataset = DATASET_REGRESI if jenis == "Regresi" else DATASET_KLASIFIKASI
    daftar_model = MODEL_REGRESI if jenis == "Regresi" else MODEL_KLASIFIKASI

    nama_dataset = st.selectbox("📊 Pilih Dataset", daftar_dataset)
    nama_model = st.selectbox("🧠 Pilih Model", daftar_model)
    test_size = st.slider("📐 Test Size", 0.10, 0.50, 0.20, 0.05,
                          help="Proporsi data yang dipakai untuk pengujian.")
    seed = st.number_input("🎲 Random State", min_value=0, max_value=9999, value=42, step=1)

    st.divider()
    tombol_train = st.button("🚀 Train Model", type="primary", use_container_width=True)
    st.divider()
    st.caption("💡 Tips: coba berbagai kombinasi dataset & model!")

# Jika tombol ditekan, latih model dan simpan hasilnya ke session_state
if tombol_train:
    with st.spinner("⏳ Sedang melatih model..."):
        st.session_state["hasil"] = latih_dan_evaluasi(
            jenis, nama_dataset, nama_model, test_size, int(seed))

# ============================================================
# HALAMAN UTAMA
# ============================================================
st.title("🤖 Regresi / Klasifikasi Machine Learning")
st.caption("Aplikasi demo Machine Learning menggunakan toy datasets dari scikit-learn")
st.divider()

hasil = st.session_state.get("hasil")

if hasil is None:
    st.info("👈 Silakan atur parameter di sidebar, lalu klik tombol 🚀 `Train Model` untuk memulai training.")
    c1, c2, c3 = st.columns(3)
    c1.subheader("📊 Dataset")
    c1.write("Pilih dari toy datasets scikit-learn")
    c2.subheader("🧠 Model")
    c2.write("5 model regresi & 5 model klasifikasi")
    c3.subheader("📈 Evaluasi")
    c3.write("Metrics lengkap + visualisasi")
    st.stop()

# --- Ringkasan hasil training ---
st.success(f"✅ Model berhasil dilatih dalam {hasil['waktu']:.3f} detik!")
k1, k2, k3, k4 = st.columns(4)
k1.metric("🎯 Task", hasil["jenis"])
k2.metric("📊 Dataset", hasil["dataset"].split(" (")[0])
k3.metric("🧠 Model", hasil["model"])
k4.metric("⏱️ Training Time", f"{hasil['waktu']:.3f}s")

# Pratinjau dataset
with st.expander("🔍 Lihat pratinjau dataset"):
    X_prev, y_prev, _ = muat_dataset(hasil["jenis"], hasil["dataset"])
    st.write(f"Jumlah sampel: **{X_prev.shape[0]}** | Jumlah fitur: **{X_prev.shape[1]}**")
    st.dataframe(X_prev.head(10))

st.divider()
tab1, tab2 = st.tabs(["📊 Training & Visualisasi", "📈 Evaluation Metrics"])

# ------------------------------------------------------------
# TAB 1: VISUALISASI
# ------------------------------------------------------------
with tab1:
    st.subheader("📊 Hasil Visualisasi")
    a, b = st.columns(2)
    a.info(f"**Data Training:** {hasil['n_train']} sampel")
    b.info(f"**Data Testing:** {hasil['n_test']} sampel")

    y_test, y_pred = hasil["y_test"], hasil["y_pred"]

    if hasil["jenis"] == "Regresi":
        kiri, kanan = st.columns(2)

        # Scatter: Actual vs Predicted
        with kiri:
            st.markdown("#### 📈 Actual vs Predicted (Scatter)")
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.scatterplot(x=y_test, y=y_pred, ax=ax, alpha=0.7, edgecolor="k")
            batas = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
            ax.plot(batas, batas, "r--", lw=2, label="Perfect Prediction")
            ax.set_xlabel("Nilai Aktual")
            ax.set_ylabel("Nilai Prediksi")
            ax.set_title("Actual vs Predicted")
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)

        # Garis: perbandingan per index (50 sampel pertama)
        with kanan:
            st.markdown("#### 📉 Perbandingan per Index (50 pertama)")
            n = min(50, len(y_test))
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot(range(n), y_test[:n], "o-", label="Actual")
            ax.plot(range(n), y_pred[:n], "x-", label="Predicted")
            ax.set_xlabel("Index Sampel")
            ax.set_ylabel("Nilai")
            ax.set_title("Actual vs Predicted per Index")
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)

    else:
        st.markdown("#### 🧩 Confusion Matrix")
        normal = st.checkbox("Tampilkan dalam persentase (normalisasi per baris)")
        cm = confusion_matrix(y_test, y_pred)
        if normal:
            cm = cm / cm.sum(axis=1, keepdims=True)
        kelas = hasil["kelas"]
        ukuran = 5 if len(kelas) <= 4 else 8
        fig, ax = plt.subplots(figsize=(ukuran, ukuran * 0.85))
        sns.heatmap(cm, annot=True, fmt=".2f" if normal else "d", cmap="Blues",
                    xticklabels=kelas, yticklabels=kelas, ax=ax)
        ax.set_xlabel("Prediksi")
        ax.set_ylabel("Aktual")
        ax.set_title("Confusion Matrix")
        _, tengah, _ = st.columns([1, 2, 1])
        with tengah:
            st.pyplot(fig)
        plt.close(fig)

# ------------------------------------------------------------
# TAB 2: EVALUATION METRICS
# ------------------------------------------------------------
with tab2:
    st.subheader("📈 Evaluation Metrics")
    m = hasil["metrics"]
    kolom = st.columns(len(m))
    for kol, (nama, nilai) in zip(kolom, m.items()):
        kol.metric(nama, f"{nilai:.4f}")

    if hasil["jenis"] == "Regresi":
        st.caption("💡 R² mendekati 1 berarti model menjelaskan variasi data dengan baik. "
                   "MAE/RMSE lebih kecil berarti kesalahan prediksi lebih kecil.")
    else:
        st.caption("💡 Precision, Recall, dan F1 dihitung dengan rata-rata *weighted* "
                   "agar sesuai dengan jumlah sampel tiap kelas.")
        st.markdown("#### 📋 Classification Report")
        df_report = pd.DataFrame(hasil["report_dict"]).T.round(3)
        st.dataframe(df_report)
        with st.expander("📝 Lihat dalam format teks"):
            st.code(hasil["report_text"])