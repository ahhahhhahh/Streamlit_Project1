from datetime import datetime, timedelta, timezone

import streamlit as st

try:
    from pymongo import MongoClient
    from bson import ObjectId
    ADA_PYMONGO = True
except ImportError:
    ADA_PYMONGO = False

from ml import FITUR_IRIS

# ============================================================
# KONFIGURASI
# ============================================================
NAMA_DB = "db_streamlit"  # nama database di MongoDB Atlas
KOLEKSI_IRIS = "iriss"  # nama koleksi untuk dataset Iris
KOLEKSI_KOMENTAR = "komentar"
ZONA_WIB = timedelta(hours=7)


@st.cache_resource
def koneksi_mongo():
    """Satu client MongoDB dipakai bersama semua sesi (di-cache)."""
    client = MongoClient(st.secrets["MONGODB_URI"], serverSelectionTimeoutMS=5000)
    client.admin.command("ping")                       # uji koneksi
    return client


def ambil_db():
    """Mengembalikan (database, None) jika sukses, atau (None, pesan_error)."""
    if not ADA_PYMONGO:
        return None, "Library pymongo belum terpasang. Jalankan: pip install pymongo dnspython"
    try:
        return koneksi_mongo()[NAMA_DB], None
    except Exception as e:
        return None, ("Gagal terhubung ke MongoDB. Periksa MONGODB_URI di secrets, "
                      f"password database user, dan Network Access. (Detail: {type(e).__name__})")


@st.cache_data(ttl=5, show_spinner=False)
def ambil_data(koleksi, batas=100):
    """Mengambil total dokumen dan dokumen terbaru. Di-cache 5 detik."""
    db, err = ambil_db()
    if db is None:
        raise RuntimeError(err)
    total = db[koleksi].count_documents({})
    dok = list(db[koleksi].find().sort("timestamp", -1).limit(batas))
    for d in dok:
        d["_id"] = str(d["_id"])                       # ObjectId -> string agar bisa di-cache
    return total, dok


def waktu_wib(ts):
    """Mengubah timestamp UTC dari MongoDB menjadi teks WIB."""
    if not isinstance(ts, datetime):
        return "-"
    return (ts + ZONA_WIB).strftime("%d %b %Y, %H:%M") + " WIB"


def panduan_koneksi():
    with st.expander("🛠️ Cara mengatur koneksi MongoDB"):
        st.markdown(
            "1. Di MongoDB Atlas, buka **Network Access** lalu tambahkan IP `0.0.0.0/0`.\n"
            "2. Salin connection string (Drivers → Python) dan ganti `<db_password>`.\n"
            "3. **Lokal:** buat file `.streamlit/secrets.toml` berisi:\n"
            "   `MONGODB_URI = \"mongodb+srv://...\"`\n"
            "4. **Streamlit Cloud:** Manage app → Settings → Secrets, isi baris yang sama.\n"
            "5. Pastikan `pymongo` dan `dnspython` ada di `requirements.txt`.")


def ambil_semua_iris(db):
    """Mengambil semua dokumen Iris (hanya kolom yang dibutuhkan untuk retrain)."""
    return list(db[KOLEKSI_IRIS].find({}, {f: 1 for f in FITUR_IRIS + ["species"]}))


def tambah_dokumen(db, koleksi, dokumen):
    """Menyimpan satu dokumen + timestamp UTC, lalu membuang cache agar langsung tampil."""
    dokumen["timestamp"] = datetime.now(timezone.utc)
    db[koleksi].insert_one(dokumen)
    ambil_data.clear()


def hapus_dokumen(db, koleksi, id_str):
    """Menghapus satu dokumen berdasarkan _id (string -> ObjectId)."""
    db[koleksi].delete_one({"_id": ObjectId(id_str)})
    ambil_data.clear()