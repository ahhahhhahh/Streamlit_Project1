# ============================================================
# APLIKASI MACHINE LEARNING DENGAN STREAMLIT  (Versi 2)
# Regresi, Klasifikasi & Clustering | Sklearn Toy + UCI Repository
# ============================================================

# --- Import library ---
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    silhouette_score, davies_bouldin_score, adjusted_rand_score,
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

st.set_page_config(page_title="ML Regresi, Klasifikasi & Clustering",
                   page_icon="🤖", layout="wide")

# ============================================================
# DAFTAR PILIHAN
# ============================================================
SUMBER = ["Sklearn (Toy)", "UCI Repository"]

# Dataset sklearn. Clustering memakai daftar "Klasifikasi" (label hanya untuk pembanding)
DATASET_SKLEARN = {
    "Regresi": ["Diabetes", "Linnerud (Target: Weight)"],
    "Klasifikasi": ["Iris", "Wine", "Breast Cancer", "Digits"],
}

# Dataset UCI: nama tampilan -> ID di archive.ics.uci.edu
DATASET_UCI = {
    "Regresi": {
        "Auto MPG (id 9)": 9,
        "Abalone (id 1)": 1,
        "Concrete Strength (id 165)": 165,
        "Real Estate Valuation (id 477)": 477,
    },
    "Klasifikasi": {
        "Iris (id 53)": 53,
        "Wine (id 109)": 109,
        "Breast Cancer Wisconsin (id 17)": 17,
        "Wine Quality (id 186)": 186,
        "Seeds (id 236)": 236,
    },
}

MODEL_REGRESI = ["Linear Regression", "Ridge Regression", "Lasso Regression",
                 "Decision Tree", "Random Forest"]
MODEL_KLASIFIKASI = ["Logistic Regression", "KNN", "Decision Tree",
                     "Random Forest", "SVM"]


# ============================================================
# FUNGSI DATA
# ============================================================
def bersihkan(X, y, jenis):
    """Merapikan dataset UCI: one-hot kolom teks, isi nilai kosong, encode label."""
    X = pd.get_dummies(X, dtype=float)                 # kolom kategori -> angka
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median())                           # nilai kosong -> median
    ada = y.notna()                                    # buang baris tanpa target
    X, y = X[ada].reset_index(drop=True), y[ada].reset_index(drop=True)
    kelas = None
    if jenis == "Klasifikasi":
        unik = sorted(y.unique(), key=str)
        y = y.map({k: i for i, k in enumerate(unik)})
        kelas = [str(k) for k in unik]
    else:
        y = pd.to_numeric(y, errors="coerce")
    return X, y, kelas


@st.cache_data(show_spinner="📥 Memuat dataset...")
def muat_data(sumber, jenis, nama, id_uci=None):
    """Memuat dataset (sklearn / UCI) dan di-cache. Mengembalikan X, y, nama kelas."""
    if sumber == SUMBER[1]:
        from ucimlrepo import fetch_ucirepo
        d = fetch_ucirepo(id=id_uci)
        return bersihkan(d.data.features.copy(), d.data.targets.iloc[:, 0].copy(), jenis)

    if jenis == "Regresi":
        if nama == "Diabetes":
            d = datasets.load_diabetes(as_frame=True)
            return d.data, d.target, None
        d = datasets.load_linnerud(as_frame=True)       # Linnerud: target kolom Weight
        return d.data, d.target["Weight"], None

    pemuat = {"Iris": datasets.load_iris, "Wine": datasets.load_wine,
              "Breast Cancer": datasets.load_breast_cancer,
              "Digits": datasets.load_digits}[nama]
    d = pemuat(as_frame=True)
    return d.data, d.target, [str(k) for k in d.target_names]


# ============================================================
# FUNGSI MODEL
# ============================================================
def buat_model(jenis, nama, seed):
    """Membuat model sesuai pilihan. Model sensitif skala dibungkus StandardScaler."""
    sc = lambda m: Pipeline([("sc", StandardScaler()), ("m", m)])
    if jenis == "Regresi":
        return {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": sc(Ridge(alpha=1.0)),
            "Lasso Regression": sc(Lasso(alpha=0.1)),
            "Decision Tree": DecisionTreeRegressor(max_depth=5, random_state=seed),
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=seed),
        }[nama]
    return {
        "Logistic Regression": sc(LogisticRegression(max_iter=1000)),
        "KNN": sc(KNeighborsClassifier(n_neighbors=5)),
        "Decision Tree": DecisionTreeClassifier(random_state=seed),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=seed),
        "SVM": sc(SVC(kernel="rbf")),
    }[nama]


def latih_dan_evaluasi(X, y, kelas, jenis, nama_model, test_size, seed):
    """Melatih model supervised (regresi/klasifikasi) lalu menghitung metrics."""
    # stratify hanya jika setiap kelas punya minimal 2 sampel
    strat = y if (jenis == "Klasifikasi" and y.value_counts().min() >= 2) else None
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=strat)

    model = buat_model(jenis, nama_model, seed)
    mulai = time.time()
    model.fit(X_tr, y_tr)
    waktu = time.time() - mulai
    y_pred = model.predict(X_te)

    hasil = {"jenis": jenis, "model": nama_model, "waktu": waktu, "X": X,
             "n_train": len(X_tr), "n_test": len(X_te),
             "y_test": np.asarray(y_te), "y_pred": np.asarray(y_pred), "kelas": kelas}

    if jenis == "Regresi":
        mse = mean_squared_error(y_te, y_pred)
        hasil["metrics"] = {"MAE": mean_absolute_error(y_te, y_pred), "MSE": mse,
                            "RMSE": float(np.sqrt(mse)), "R²": r2_score(y_te, y_pred)}
    else:
        hasil["metrics"] = {
            "Accuracy": accuracy_score(y_te, y_pred),
            "Precision": precision_score(y_te, y_pred, average="weighted", zero_division=0),
            "Recall": recall_score(y_te, y_pred, average="weighted", zero_division=0),
            "F1-Score": f1_score(y_te, y_pred, average="weighted", zero_division=0)}
        semua = list(range(len(kelas)))
        kw = dict(labels=semua, target_names=kelas, zero_division=0)
        hasil["report_dict"] = classification_report(y_te, y_pred, output_dict=True, **kw)
        hasil["report_text"] = classification_report(y_te, y_pred, **kw)
    return hasil


def jalankan_clustering(X, y, k, standar, seed):
    """K-Means + PCA 2D + kurva elbow/silhouette untuk k = 2..10."""
    Xs = StandardScaler().fit_transform(X.values) if standar else X.values
    n_sil = min(2000, len(Xs))                          # sampel silhouette agar cepat

    km = KMeans(n_clusters=k, n_init=10, random_state=seed)
    mulai = time.time()
    label = km.fit_predict(Xs)
    waktu = time.time() - mulai

    pca = PCA(n_components=2, random_state=seed)
    Z = pca.fit_transform(Xs)
    C = pca.transform(km.cluster_centers_)

    ks = list(range(2, min(10, len(Xs) - 1) + 1))
    inertia, sil = [], []
    for kk in ks:
        m = KMeans(n_clusters=kk, n_init=10, random_state=seed).fit(Xs)
        inertia.append(m.inertia_)
        sil.append(silhouette_score(Xs, m.labels_, sample_size=n_sil, random_state=seed))

    metrics = {"Inertia": km.inertia_,
               "Silhouette": silhouette_score(Xs, label, sample_size=n_sil, random_state=seed),
               "Davies-Bouldin": davies_bouldin_score(Xs, label)}
    if y is not None:
        metrics["ARI (vs label asli)"] = adjusted_rand_score(y, label)

    return {"jenis": "Clustering", "model": f"K-Means (k={k})", "waktu": waktu, "X": X,
            "k": k, "label": label, "Z": Z, "C": C, "ks": ks, "inertia": inertia,
            "sil": sil, "var_pca": float(pca.explained_variance_ratio_.sum()),
            "metrics": metrics}


def tampil(fig):
    """Menampilkan figure matplotlib lalu menutupnya agar memori tidak bocor."""
    st.pyplot(fig)
    plt.close(fig)


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.header("⚙️ Pengaturan Model")
    st.caption("Atur parameter di bawah, lalu klik tombol Train.")

    jenis = st.radio("🎯 Pilih Jenis Tugas", ["Regresi", "Klasifikasi", "Clustering"])
    kunci = "Klasifikasi" if jenis == "Clustering" else jenis

    sumber = st.radio("🗂️ Sumber Dataset", SUMBER, horizontal=True)
    if sumber == SUMBER[0]:
        nama_dataset = st.selectbox("📊 Pilih Dataset", DATASET_SKLEARN[kunci])
        id_uci = None
    else:
        nama_dataset = st.selectbox("📊 Pilih Dataset UCI", list(DATASET_UCI[kunci]))
        id_uci = DATASET_UCI[kunci][nama_dataset]

    if jenis == "Clustering":
        k = st.slider("🔢 Jumlah Cluster (k)", 2, 10, 3)
        standar = st.checkbox("📏 Standarisasi fitur", value=True,
                              help="Disarankan agar fitur berskala besar tidak mendominasi.")
        label_tombol = "🚀 Jalankan Clustering"
    else:
        nama_model = st.selectbox("🧠 Pilih Model",
                                  MODEL_REGRESI if jenis == "Regresi" else MODEL_KLASIFIKASI)
        test_size = st.slider("📐 Test Size", 0.10, 0.50, 0.20, 0.05,
                              help="Proporsi data yang dipakai untuk pengujian.")
        label_tombol = "🚀 Train Model"

    seed = int(st.number_input("🎲 Random State", 0, 9999, 42, 1))

    st.divider()
    tombol = st.button(label_tombol, type="primary", use_container_width=True)
    st.divider()
    st.caption("💡 Tips: coba berbagai kombinasi dataset & model!")

# ============================================================
# HALAMAN UTAMA
# ============================================================
st.title("🤖 Regresi, Klasifikasi & Clustering")
st.caption("Aplikasi demo Machine Learning dengan dataset scikit-learn dan UCI Repository")
st.divider()

# Jika tombol ditekan: muat data, latih, simpan hasil ke session_state
if tombol:
    try:
        X, y, kelas = muat_data(sumber, kunci, nama_dataset, id_uci)
    except Exception as e:
        st.error(f"❌ Gagal memuat dataset: {e}")
        st.info("Jika memakai dataset UCI, pastikan ada koneksi internet. "
                "Coba dataset sklearn sebagai pembanding.")
        st.stop()

    with st.spinner("⏳ Sedang memproses..."):
        if jenis == "Clustering":
            hasil = jalankan_clustering(X, y, k, standar, seed)
        else:
            hasil = latih_dan_evaluasi(X, y, kelas, jenis, nama_model, test_size, seed)
    hasil["dataset"] = nama_dataset
    hasil["sumber"] = sumber
    st.session_state["hasil"] = hasil

hasil = st.session_state.get("hasil")

if hasil is None:
    st.info("👈 Silakan atur parameter di sidebar, lalu klik tombol 🚀 untuk memulai.")
    c1, c2, c3 = st.columns(3)
    c1.subheader("📊 Dataset")
    c1.write("Toy datasets sklearn atau UCI Repository")
    c2.subheader("🧠 Model")
    c2.write("Regresi, klasifikasi, dan clustering K-Means")
    c3.subheader("📈 Evaluasi")
    c3.write("Metrics lengkap + visualisasi")
    st.stop()

# --- Ringkasan hasil ---
st.success(f"✅ Selesai diproses dalam {hasil['waktu']:.3f} detik!")
k1, k2, k3, k4 = st.columns(4)
k1.metric("🎯 Task", hasil["jenis"])
k2.metric("📊 Dataset", hasil["dataset"].split(" (")[0])
k3.metric("🧠 Model", hasil["model"])
k4.metric("🗂️ Sumber", hasil["sumber"].split(" (")[0])

with st.expander("🔍 Lihat pratinjau dataset"):
    st.write(f"Jumlah sampel: **{hasil['X'].shape[0]}** | "
             f"Jumlah fitur: **{hasil['X'].shape[1]}**")
    st.dataframe(hasil["X"].head(10))

st.divider()
tab1, tab2 = st.tabs(["📊 Training & Visualisasi", "📈 Evaluation Metrics"])

# ------------------------------------------------------------
# TAB 1: VISUALISASI
# ------------------------------------------------------------
with tab1:
    st.subheader("📊 Hasil Visualisasi")

    if hasil["jenis"] == "Clustering":
        a, b, c = st.columns(3)
        a.info(f"**Jumlah Sampel:** {hasil['X'].shape[0]}")
        b.info(f"**Jumlah Cluster:** {hasil['k']}")
        c.info(f"**Variansi PCA (2 komponen):** {hasil['var_pca']:.1%}")

        st.markdown("#### 🎨 Sebaran Cluster (proyeksi PCA 2D)")
        nama_c = [f"Cluster {i}" for i in range(hasil["k"])]
        hue = pd.Series(hasil["label"]).map(lambda i: f"Cluster {i}")
        fig, ax = plt.subplots(figsize=(8, 5.5))
        sns.scatterplot(x=hasil["Z"][:, 0], y=hasil["Z"][:, 1], hue=hue, hue_order=nama_c,
                        palette="tab10", s=45, edgecolor="k", alpha=0.8, ax=ax)
        ax.scatter(hasil["C"][:, 0], hasil["C"][:, 1], c="red", marker="X", s=260,
                   edgecolor="white", linewidth=1.5, label="Centroid", zorder=5)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title("Hasil K-Means pada ruang PCA")
        ax.legend(title="Keterangan", fontsize=8)
        _, tengah, _ = st.columns([1, 3, 1])
        with tengah:
            tampil(fig)

    else:
        a, b = st.columns(2)
        a.info(f"**Data Training:** {hasil['n_train']} sampel")
        b.info(f"**Data Testing:** {hasil['n_test']} sampel")
        y_test, y_pred = hasil["y_test"], hasil["y_pred"]

        if hasil["jenis"] == "Regresi":
            kiri, kanan = st.columns(2)
            with kiri:
                st.markdown("#### 📈 Actual vs Predicted (Scatter)")
                fig, ax = plt.subplots(figsize=(6, 5))
                sns.scatterplot(x=y_test, y=y_pred, ax=ax, alpha=0.7, edgecolor="k")
                batas = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
                ax.plot(batas, batas, "r--", lw=2, label="Perfect Prediction")
                ax.set_xlabel("Nilai Aktual")
                ax.set_ylabel("Nilai Prediksi")
                ax.legend()
                tampil(fig)
            with kanan:
                st.markdown("#### 📉 Perbandingan per Index (50 pertama)")
                n = min(50, len(y_test))
                fig, ax = plt.subplots(figsize=(6, 5))
                ax.plot(range(n), y_test[:n], "o-", label="Actual")
                ax.plot(range(n), y_pred[:n], "x-", label="Predicted")
                ax.set_xlabel("Index Sampel")
                ax.set_ylabel("Nilai")
                ax.legend()
                tampil(fig)
        else:
            st.markdown("#### 🧩 Confusion Matrix")
            normal = st.checkbox("Tampilkan dalam persentase (normalisasi per baris)")
            kelas = hasil["kelas"]
            cm = confusion_matrix(y_test, y_pred, labels=list(range(len(kelas))))
            if normal:
                cm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
            ukuran = 5 if len(kelas) <= 4 else 8
            fig, ax = plt.subplots(figsize=(ukuran, ukuran * 0.85))
            sns.heatmap(cm, annot=True, fmt=".2f" if normal else "d", cmap="Blues",
                        xticklabels=kelas, yticklabels=kelas, ax=ax)
            ax.set_xlabel("Prediksi")
            ax.set_ylabel("Aktual")
            _, tengah, _ = st.columns([1, 2, 1])
            with tengah:
                tampil(fig)

# ------------------------------------------------------------
# TAB 2: EVALUATION METRICS
# ------------------------------------------------------------
with tab2:
    st.subheader("📈 Evaluation Metrics")
    m = hasil["metrics"]
    for kol, (nama, nilai) in zip(st.columns(len(m)), m.items()):
        kol.metric(nama, f"{nilai:.4f}")

    if hasil["jenis"] == "Regresi":
        st.caption("💡 R² mendekati 1 berarti model menjelaskan variasi data dengan baik. "
                   "MAE/RMSE lebih kecil berarti kesalahan prediksi lebih kecil.")

    elif hasil["jenis"] == "Klasifikasi":
        st.caption("💡 Precision, Recall, dan F1 memakai rata-rata *weighted* "
                   "agar sesuai jumlah sampel tiap kelas.")
        st.markdown("#### 📋 Classification Report")
        st.dataframe(pd.DataFrame(hasil["report_dict"]).T.round(3))
        with st.expander("📝 Lihat dalam format teks"):
            st.code(hasil["report_text"])

    else:
        st.caption("💡 Silhouette (−1 sampai 1): makin tinggi makin baik. "
                   "Davies-Bouldin: makin rendah makin baik. "
                   "ARI membandingkan cluster dengan label asli (1 = identik).")

        st.markdown("#### 🔎 Memilih Jumlah Cluster (Elbow & Silhouette)")
        kiri, kanan = st.columns(2)
        with kiri:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(hasil["ks"], hasil["inertia"], "o-")
            ax.axvline(hasil["k"], color="red", ls="--", label=f"k dipilih = {hasil['k']}")
            ax.set_xlabel("Jumlah Cluster (k)")
            ax.set_ylabel("Inertia")
            ax.set_title("Metode Elbow")
            ax.legend()
            tampil(fig)
        with kanan:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(hasil["ks"], hasil["sil"], "s-", color="green")
            ax.axvline(hasil["k"], color="red", ls="--", label=f"k dipilih = {hasil['k']}")
            ax.set_xlabel("Jumlah Cluster (k)")
            ax.set_ylabel("Silhouette Score")
            ax.set_title("Silhouette per k")
            ax.legend()
            tampil(fig)

        st.markdown("#### 📋 Ringkasan Tiap Cluster (rata-rata fitur)")
        df = hasil["X"].copy()
        df["Cluster"] = hasil["label"]
        ringkas = df.groupby("Cluster").mean().round(3)
        ringkas.insert(0, "Jumlah", df["Cluster"].value_counts().sort_index())
        st.dataframe(ringkas)