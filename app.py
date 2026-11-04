"""Aplikasi utama: konfigurasi halaman, sidebar, alur training, dan susunan tab.

Struktur proyek:
    app.py        -> file ini (titik masuk, `streamlit run app.py`)
    ml.py         -> dataset, model, evaluasi, clustering, retrain
    db/database.py -> koneksi & operasi MongoDB
    interface.py  -> isi tiap tab (grafik, tabel, form)
    llm.py        -> integrasi Groq (ringkasan & sentimen komentar)
"""
import streamlit as st

# set_page_config harus menjadi perintah Streamlit pertama
st.set_page_config(page_title="ML Regresi, Klasifikasi & Clustering",
                   page_icon="🤖", layout="wide")

from ml import (  # noqa: E402
    SUMBER, FEED_GEMPA, DATASET_SKLEARN, DATASET_UCI, MODEL_REGRESI, MODEL_KLASIFIKASI,
    muat_data, latih_dan_evaluasi, jalankan_clustering,
)
from interface import isi_tab1, isi_tab2, isi_tab3, isi_tab4  # noqa: E402


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.header("⚙️ Pengaturan Model")
    st.caption("Atur parameter di bawah, lalu klik tombol Train.")

    jenis = st.radio("🎯 Pilih Jenis Tugas", ["Regresi", "Klasifikasi", "Clustering"])
    kunci = "Klasifikasi" if jenis == "Clustering" else jenis

    sumber = st.radio("🗂️ Sumber Dataset", SUMBER)
    if sumber == SUMBER[0]:
        nama_dataset = st.selectbox("📊 Pilih Dataset", DATASET_SKLEARN[kunci])
        id_uci = None
    elif sumber == SUMBER[1]:
        nama_dataset = st.selectbox("📊 Pilih Dataset UCI", list(DATASET_UCI[kunci]))
        id_uci = DATASET_UCI[kunci][nama_dataset]
    else:   # real-time: id_uci dipakai untuk membawa kode feed USGS
        nama_dataset = st.selectbox("🌋 Pilih Feed Gempa USGS", list(FEED_GEMPA))
        id_uci = FEED_GEMPA[nama_dataset]
        st.caption("ℹ️ Data real-time di-cache 5 menit. "
                   "Regresi: prediksi magnitudo. Klasifikasi: kategori magnitudo. "
                   "Clustering: zona seismik dari lokasi dan kedalaman.")

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

st.caption("Aplikasi demo Machine Learning dengan dataset scikit-learn, UCI Repository, dan data gempa real-time")

st.divider()

# Jika tombol ditekan: muat data, latih, simpan hasil ke session_state
if tombol:
    try:
        X, y, kelas = muat_data(sumber, kunci, nama_dataset, id_uci)
    except Exception as e:
        st.error(f"❌ Gagal memuat dataset: {e}")
        st.info("Jika memakai dataset UCI atau real-time, pastikan ada koneksi internet. "
                "Coba dataset sklearn sebagai pembanding.")
    else:
        if jenis == "Clustering" and sumber == SUMBER[2]:
            X, y = X[["longitude", "latitude", "depth"]], None   # klaster zona seismik
        with st.spinner("⏳ Sedang memproses..."):
            if jenis == "Clustering":
                hasil_baru = jalankan_clustering(X, y, k, standar, seed)
            else:
                hasil_baru = latih_dan_evaluasi(X, y, kelas, jenis, nama_model, test_size, seed)
        hasil_baru["dataset"] = nama_dataset
        hasil_baru["sumber"] = sumber
        st.session_state["hasil"] = hasil_baru

hasil = st.session_state.get("hasil")

if hasil is None:
    st.info("👈 Atur parameter di sidebar, lalu klik tombol 🚀 untuk memulai. "
            "Tab 🗄️ Database & Retrain dan 💬 Komentar bisa dipakai kapan saja.")
else:
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

# ============================================================
# SUSUNAN TAB
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs(["📊 Training & Visualisasi", "📈 Evaluation Metrics",
                                  "🗄️ Database & Retrain", "💬 Komentar"])

with tab1:
    if hasil is None:
        st.info("Klik tombol 🚀 di sidebar terlebih dahulu.")
    else:
        isi_tab1(hasil)

with tab2:
    if hasil is None:
        st.info("Klik tombol 🚀 di sidebar terlebih dahulu.")
    else:
        isi_tab2(hasil)

with tab3:
    isi_tab3()

with tab4:
    isi_tab4()