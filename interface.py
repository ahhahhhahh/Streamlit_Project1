"""Fungsi tampilan: isi tiap tab Streamlit."""
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import confusion_matrix

from db.database import (
    NAMA_DB, KOLEKSI_IRIS, KOLEKSI_KOMENTAR,
    ambil_db, ambil_data, ambil_semua_iris, tambah_dokumen, hapus_dokumen,
    panduan_koneksi, waktu_wib,
)
from llm import (BATAS_ANALISIS, LABEL_SENTIMEN, analisis_komentar, kunci_groq, model_groq)
from ml import FITUR_IRIS, SPESIES, retrain_iris

sns.set_theme(style="whitegrid")


def tampil(fig):
    """Menampilkan figure matplotlib lalu menutupnya agar memori tidak bocor."""
    st.pyplot(fig)
    plt.close(fig)


def isi_tab1(hasil):
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
    # --- Kurva loss (khusus model Neural Network / MLP) ---
    if hasil.get("loss_curve"):
        st.divider()
        st.markdown("#### 🧠 Kurva Loss Training (Neural Network)")
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.plot(hasil["loss_curve"])
        ax.set_xlabel("Iterasi (epoch)")
        ax.set_ylabel("Loss")
        ax.set_title("Loss yang turun berarti jaringan sedang belajar")
        tampil(fig)

    # --- Peta sebaran cluster (khusus data gempa real-time) ---
    if hasil["jenis"] == "Clustering" and {"longitude", "latitude"} <= set(hasil["X"].columns):
        st.divider()
        st.markdown("#### 🌍 Peta Sebaran Gempa per Cluster")
        hue = pd.Series(hasil["label"]).map(lambda i: f"Cluster {i}")
        fig, ax = plt.subplots(figsize=(9, 4.5))
        sns.scatterplot(x=hasil["X"]["longitude"].values, y=hasil["X"]["latitude"].values,
                        hue=hue, hue_order=[f"Cluster {i}" for i in range(hasil["k"])],
                        palette="tab10", s=25, alpha=0.8, ax=ax)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title("Zona seismik hasil K-Means (longitude, latitude, kedalaman)")
        ax.legend(fontsize=8)
        tampil(fig)


def isi_tab2(hasil):
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


def isi_tab3():
    st.subheader("🗄️ MongoDB CRUD & Retrain Model")
    db, err = ambil_db()
    if db is None:
        st.error(f"❌ {err}")
        panduan_koneksi()
        return

    try:
        total, dok = ambil_data(KOLEKSI_IRIS)
    except Exception as e:
        st.error(f"❌ Gagal membaca data: {e}")
        return

    # --- Statistik collection ---
    c1, c2, c3 = st.columns(3)
    c1.metric("📦 Total Dokumen", total)
    c2.metric("🗃️ Database", NAMA_DB)
    c3.metric("📁 Collection", KOLEKSI_IRIS)
    st.divider()

    # --- Form tambah data ---
    st.markdown("### ➕ Tambah Data Iris")
    with st.form("form_iris"):
        kiri, kanan = st.columns(2)
        sl = kiri.number_input("Sepal Length (cm)", 0.0, 15.0, 5.1, 0.1)
        sw = kiri.number_input("Sepal Width (cm)", 0.0, 15.0, 3.5, 0.1)
        pl = kanan.number_input("Petal Length (cm)", 0.0, 15.0, 1.4, 0.1)
        pw = kanan.number_input("Petal Width (cm)", 0.0, 15.0, 0.2, 0.1)
        sp = st.selectbox("Species", SPESIES)
        kirim = st.form_submit_button("💾 Simpan ke MongoDB")
    if kirim:
        try:
            tambah_dokumen(db, KOLEKSI_IRIS, {
                "sepal_length": float(sl), "sepal_width": float(sw),
                "petal_length": float(pl), "petal_width": float(pw),
                "species": sp})
            st.toast("✅ Data berhasil disimpan!")
            st.rerun()
        except Exception as e:
            st.error(f"❌ Gagal menyimpan: {e}")
    st.divider()

    # --- Tabel data + hapus ---
    st.markdown("### 📋 Data di MongoDB")
    if not dok:
        st.info("Belum ada data. Tambahkan lewat form di atas.")
    else:
        df = pd.DataFrame(dok).reindex(columns=["_id"] + FITUR_IRIS + ["species", "timestamp"])
        tabel = df.copy()
        tabel["timestamp"] = tabel["timestamp"].map(waktu_wib)
        st.dataframe(tabel)
        if total > len(dok):
            st.caption(f"Menampilkan {len(dok)} data terbaru dari {total}.")

        label = {r["_id"]: (f"{r['species']} | {r['sepal_length']}, {r['sepal_width']}, "
                            f"{r['petal_length']}, {r['petal_width']} | {waktu_wib(r['timestamp'])}")
                 for _, r in df.iterrows()}
        id_hapus = st.selectbox("🗑️ Pilih data yang akan dihapus", list(label), format_func=label.get)
        if st.button("Hapus data terpilih"):
            try:
                hapus_dokumen(db, KOLEKSI_IRIS, id_hapus)
                st.toast("🗑️ Data dihapus.")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Gagal menghapus: {e}")
    st.divider()

    # --- Retrain ---
    st.markdown("### 🚀 Retrain Model (Random Forest, Iris)")
    st.caption("Menggabungkan data Iris sklearn (150 sampel) dengan data dari MongoDB.")
    if st.button("🚀 Retrain Sekarang", type="primary"):
        with st.spinner("⏳ Melatih ulang model..."):
            try:
                st.session_state["retrain"] = retrain_iris(ambil_semua_iris(db))
            except Exception as e:
                st.error(f"❌ Retrain gagal: {e}")

    rt = st.session_state.get("retrain")
    if rt:
        a, b, c, d = st.columns(4)
        a.metric("🎯 Akurasi", f"{rt['akurasi']:.2%}")
        b.metric("📚 Data sklearn", rt["n_sklearn"])
        c.metric("🗄️ Data MongoDB", rt["n_mongo"])
        d.metric("➕ Total data latih+uji", rt["n_total"])
        kiri, kanan = st.columns(2)
        with kiri:
            st.markdown("#### 📋 Classification Report")
            st.dataframe(rt["laporan"])
        with kanan:
            st.markdown("#### 🌟 Feature Importance")
            fig, ax = plt.subplots(figsize=(6, 3.5))
            sns.barplot(x=rt["pentingnya"].values, y=rt["pentingnya"].index, ax=ax)
            ax.set_xlabel("Tingkat Kepentingan")
            tampil(fig)


IKON_SENTIMEN = {"positif": "😊", "netral": "😐", "negatif": "😠"}
WARNA_SENTIMEN = {"positif": "#2e9e5b", "netral": "#8a8f98", "negatif": "#d64545"}


def bagian_ai(dok):
    """Analisis komentar dengan LLM Groq: ringkasan, sentimen, dan topik utama."""
    st.divider()
    st.markdown("### 🤖 Analisis AI (Groq)")
    if not kunci_groq():
        st.info("Fitur ini butuh API key Groq. Buat di console.groq.com, lalu isi "
                "`GROQ_API_KEY = \"gsk_...\"` di `.streamlit/secrets.toml` (lokal) dan "
                "di Secrets Streamlit Cloud.")
        return
    if not dok:
        st.caption("Belum ada komentar untuk dianalisis.")
        return

    st.caption(f"Menganalisis hingga {BATAS_ANALISIS} komentar terbaru. "
               "Hanya isi komentar yang dikirim ke Groq, tanpa nama penulis.")
    if st.button("✨ Analisis Komentar dengan AI"):
        sekarang = time.time()
        if sekarang - st.session_state.get("ai_terakhir", 0) < 10:
            st.warning("⏳ Tunggu beberapa detik sebelum menganalisis lagi.")
        else:
            st.session_state["ai_terakhir"] = sekarang
            terpilih = dok[:BATAS_ANALISIS]
            try:
                with st.spinner("🤖 AI sedang membaca komentar..."):
                    ai = analisis_komentar(tuple(d.get("komentar", "") for d in terpilih),
                                           model_groq())
                ai = {**ai, "peta": {d["_id"]: lb for d, lb in zip(terpilih, ai["label"])}}
                st.session_state["hasil_ai"] = ai
                st.rerun()                             # muat ulang agar ikon sentimen tampil di daftar
            except Exception as e:
                st.error(f"❌ Analisis gagal: {e}")

    ai = st.session_state.get("hasil_ai")
    if ai:
        st.markdown("#### 📝 Ringkasan")
        with st.container(border=True):
            st.text(ai["ringkasan"])                   # st.text: keluaran LLM tidak dirender sebagai markdown
        hitung = pd.Series(ai["label"]).value_counts().reindex(LABEL_SENTIMEN, fill_value=0)
        kiri, kanan = st.columns([3, 2])
        with kiri:
            st.markdown("#### 📊 Sentimen")
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.bar(hitung.index, hitung.values, color=[WARNA_SENTIMEN[x] for x in hitung.index])
            ax.set_ylabel("Jumlah komentar")
            tampil(fig)
        with kanan:
            st.markdown("#### 🏷️ Topik utama")
            st.text("\n".join(f"• {t}" for t in ai["topik"]) or "-")
        st.caption(f"Model: {ai['model']} • hasil di-cache 10 menit • Analisis AI bisa keliru, "
                   "anggap sebagai gambaran umum.")


def isi_tab4():
    st.subheader("💬 Komentar Publik")
    db, err = ambil_db()
    if db is None:
        st.error(f"❌ {err}")
        panduan_koneksi()
        return

    with st.form("form_komentar", clear_on_submit=True):
        nama = st.text_input("Nama (opsional)", max_chars=50)
        isi = st.text_area("Komentar", max_chars=500, placeholder="Tulis komentar kamu di sini...")
        kirim = st.form_submit_button("📨 Kirim Komentar")

    if kirim:
        isi, nama = isi.strip(), nama.strip()
        sekarang = time.time()
        if not isi:
            st.warning("⚠️ Komentar tidak boleh kosong.")
        elif sekarang - st.session_state.get("terakhir_kirim", 0) < 10:
            st.warning("⏳ Tunggu beberapa detik sebelum mengirim komentar lagi.")
        else:
            try:
                tambah_dokumen(db, KOLEKSI_KOMENTAR, {"nama": nama or "Anonim", "komentar": isi})
                st.session_state["terakhir_kirim"] = sekarang
                st.toast("✅ Komentar terkirim, terima kasih!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Gagal menyimpan komentar: {e}")

    st.divider()
    try:
        total, dok = ambil_data(KOLEKSI_KOMENTAR)
    except Exception as e:
        st.error(f"❌ Gagal membaca komentar: {e}")
        return

    st.markdown(f"### 📜 Daftar Komentar ({total})")
    if not dok:
        st.info("Belum ada komentar. Jadilah yang pertama!")
    else:
        peta_ai = (st.session_state.get("hasil_ai") or {}).get("peta", {})
        with st.container(height=450):
            for d in dok:
                with st.container(border=True):
                    # st.text dipakai agar isi komentar tidak dirender sebagai markdown/HTML
                    ikon = IKON_SENTIMEN.get(peta_ai.get(d["_id"]), "")
                    st.text(f"{ikon} 👤 {d.get('nama', 'Anonim')}  •  🕒 {waktu_wib(d.get('timestamp'))}")
                    st.text(d.get("komentar", ""))

    bagian_ai(dok)