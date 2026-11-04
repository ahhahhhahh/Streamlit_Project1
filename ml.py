"""Logika Machine Learning: dataset, model, evaluasi, clustering, dan retrain."""
import json
import time
import urllib.request

import numpy as np
import pandas as pd
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
    classification_report,
    silhouette_score, davies_bouldin_score, adjusted_rand_score,
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.compose import TransformedTargetRegressor

# ============================================================
# DAFTAR PILIHAN & SKEMA DATA
# ============================================================
SUMBER = ["Sklearn (Toy)", "UCI Repository", "Real-time (USGS Gempa)"]

# Feed gempa USGS (USGS memperbarui tiap beberapa menit; di aplikasi di-cache 5 menit)
URL_USGS = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/{}.geojson"
FEED_GEMPA = {"M2.5+ 7 hari": "2.5_week", "M4.5+ 30 hari": "4.5_month", "Semua 24 jam": "all_day"}
DATASET_SKLEARN = {
    "Regresi": ["Diabetes", "Linnerud (Target: Weight)"],
    "Klasifikasi": ["Iris", "Wine", "Breast Cancer", "Digits"],
}
# Hanya dataset yang BISA diimpor lewat ucimlrepo (tidak semua dataset UCI bisa, mis. Seeds id 236).
# Daftar resminya: archive.ics.uci.edu/datasets (centang filter "Python").
DATASET_UCI = {
    "Regresi": {
        "Auto MPG (id 9)": 9,
        "Abalone (id 1)": 1,
        "Real Estate Valuation (id 477)": 477,
        "Wine Quality (id 186)": 186,
    },
    "Klasifikasi": {
        "Iris (id 53)": 53,
        "Wine (id 109)": 109,
        "Breast Cancer Wisconsin (id 17)": 17,
        "Wine Quality (id 186)": 186,
        "Heart Disease (id 45)": 45,
    },
}
MODEL_REGRESI = ["Linear Regression", "Ridge Regression", "Lasso Regression",
                 "Decision Tree", "Random Forest", "Neural Network (MLP)"]
MODEL_KLASIFIKASI = ["Logistic Regression", "KNN", "Decision Tree",
                     "Random Forest", "SVM", "Neural Network (MLP)"]
FITUR_IRIS = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
SPESIES = ["setosa", "versicolor", "virginica"]
PETA_SPESIES = {s: i for i, s in enumerate(SPESIES)}   # setosa=0, versicolor=1, virginica=2


def bersihkan(X, y, jenis):
    """Merapikan dataset UCI: one-hot kolom teks, isi nilai kosong, encode label."""
    X = pd.get_dummies(X, dtype=float)                 # kolom kategori -> angka
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.dropna(axis=1, how="all")                    # buang kolom yang seluruhnya kosong
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
def muat_statis(sumber, jenis, nama, id_uci=None):
    """Memuat dataset statis (sklearn / UCI), di-cache. Mengembalikan X, y, nama kelas."""
    if sumber == SUMBER[1]:
        from ucimlrepo import fetch_ucirepo
        try:
            d = fetch_ucirepo(id=id_uci)
        except Exception as e:
            if "not available for import" in str(e):
                raise RuntimeError(f"Dataset UCI id={id_uci} ada di repositori, tetapi tidak bisa diimpor "
                                   "lewat ucimlrepo. Silakan pilih dataset lain.") from e
            raise RuntimeError(f"Gagal mengambil dataset UCI id={id_uci} "
                               f"({type(e).__name__}). Periksa koneksi internet atau pilih dataset lain.") from e
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


@st.cache_data(ttl=300, show_spinner="🌋 Mengambil data gempa terbaru dari USGS...")
def muat_realtime(jenis, feed):
    """Mengambil data gempa real-time dari USGS (di-cache 5 menit).

    Fitur   : longitude, latitude, depth, nst, gap, dmin, rms
    Regresi : target = magnitudo
    Klasifikasi: target = kategori magnitudo (Kecil / Sedang / Besar)
    """
    req = urllib.request.Request(URL_USGS.format(feed), headers={"User-Agent": "streamlit-ml-app"})
    with urllib.request.urlopen(req, timeout=20) as r:
        data = json.load(r)

    baris = []
    for f in data.get("features", []):
        p = f.get("properties") or {}
        g = (f.get("geometry") or {}).get("coordinates") or [None, None, None]
        baris.append({"longitude": g[0], "latitude": g[1], "depth": g[2],
                      "nst": p.get("nst"), "gap": p.get("gap"),
                      "dmin": p.get("dmin"), "rms": p.get("rms"), "mag": p.get("mag")})
    df = pd.DataFrame(baris).dropna(subset=["mag", "longitude", "latitude", "depth"])
    if len(df) < 30:
        raise ValueError("Data gempa dari USGS terlalu sedikit. Coba feed dengan rentang lebih luas.")

    X = df.drop(columns="mag")
    if jenis == "Regresi":
        y = df["mag"]
    else:   # awalan angka agar urutan kelas rapi saat di-encode
        y = pd.cut(df["mag"], [-10, 3.5, 4.5, 20],
                   labels=["1-Kecil (M<3.5)", "2-Sedang (M3.5-4.5)", "3-Besar (M>=4.5)"]).astype(str)
    return bersihkan(X, y, jenis)


def muat_data(sumber, jenis, nama, id_uci=None):
    """Titik masuk pemuatan data. Real-time (id_uci = kode feed) di-cache 5 menit,
    dataset statis di-cache selamanya."""
    if sumber == SUMBER[2]:
        return muat_realtime(jenis, id_uci)
    return muat_statis(sumber, jenis, nama, id_uci)


def kurva_loss(model):
    """Mengambil kurva loss training jika modelnya MLP (selain itu mengembalikan None)."""
    m = getattr(model, "regressor_", model)            # TransformedTargetRegressor -> regresor dalam
    if hasattr(m, "named_steps"):
        m = m.named_steps["m"]
    kurva = getattr(m, "loss_curve_", None)
    return list(kurva) if kurva is not None else None


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
            # Deep learning ringan: jaringan saraf 2 lapis tersembunyi (64 dan 32 neuron).
            # Target juga distandarisasi agar training stabil.
            "Neural Network (MLP)": TransformedTargetRegressor(
                regressor=sc(MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=2000,
                                          random_state=seed)),
                transformer=StandardScaler()),
        }[nama]
    return {
        "Logistic Regression": sc(LogisticRegression(max_iter=1000)),
        "KNN": sc(KNeighborsClassifier(n_neighbors=5)),
        "Decision Tree": DecisionTreeClassifier(random_state=seed),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=seed),
        "SVM": sc(SVC(kernel="rbf")),
        "Neural Network (MLP)": sc(MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000,
                                                 random_state=seed)),
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
    hasil["loss_curve"] = kurva_loss(model)            # hanya terisi untuk model MLP

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


def retrain_iris(dok, seed=42):
    """Menggabungkan Iris sklearn (150 sampel) + dokumen MongoDB (`dok`), lalu melatih Random Forest."""
    iris = datasets.load_iris(as_frame=True)
    X0 = iris.data.copy()
    X0.columns = FITUR_IRIS
    y0 = iris.target.copy()
    X, y, n_mongo = X0, y0, 0

    if dok:
        dm = pd.DataFrame(dok).reindex(columns=FITUR_IRIS + ["species"])
        dm["label"] = dm["species"].map(PETA_SPESIES)  # string -> integer
        dm = dm.dropna()
        n_mongo = len(dm)
        X = pd.concat([X0, dm[FITUR_IRIS].astype(float)], ignore_index=True)
        y = pd.concat([y0, dm["label"].astype(int)], ignore_index=True)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    model = RandomForestClassifier(n_estimators=100, random_state=seed)
    mulai = time.time()
    model.fit(X_tr, y_tr)
    waktu = time.time() - mulai
    y_pred = model.predict(X_te)

    laporan = classification_report(y_te, y_pred, labels=[0, 1, 2], target_names=SPESIES,
                                    output_dict=True, zero_division=0)
    return {"akurasi": accuracy_score(y_te, y_pred), "waktu": waktu,
            "n_sklearn": len(X0), "n_mongo": n_mongo, "n_total": len(X),
            "laporan": pd.DataFrame(laporan).T.round(3),
            "pentingnya": pd.Series(model.feature_importances_, index=FITUR_IRIS).sort_values()}