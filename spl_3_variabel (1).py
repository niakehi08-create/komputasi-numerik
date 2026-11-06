# CELL 1: Memanggil pustaka NumPy
# NumPy digunakan untuk operasi matriks.
import numpy as np


# CELL 2: Memasukkan data sistem persamaan
# Setiap baris berisi koefisien x, y, z dan konstanta.
A = np.array([
    [2, 1, -1, 8],
    [-3, -1, 2, -11],
    [-2, 1, 2, -3]
], dtype=float)


# CELL 3: Fungsi penyelesaian Eliminasi Gauss
def eliminasi_gauss_3_variabel(matriks):

    # Menyalin matriks agar data asli tidak berubah.
    M = np.array(matriks, dtype=float, copy=True)

    # Memeriksa ukuran matriks.
    if M.shape != (3, 4):
        raise ValueError("Matriks harus berukuran 3 x 4")

    # Melakukan proses eliminasi.
    for kolom in range(2):

        # Mencari posisi pivot terbesar.
        baris_pivot = kolom + np.argmax(
            np.abs(M[kolom:, kolom])
        )

        # Menukar posisi baris jika diperlukan.
        if baris_pivot != kolom:
            M[[kolom, baris_pivot]] = M[[baris_pivot, kolom]]

        # Memeriksa apakah pivot bernilai nol.
        if np.isclose(M[kolom, kolom], 0):
            raise ValueError("Tidak ada solusi tunggal")

        # Menghilangkan elemen di bawah pivot.
        for baris in range(kolom + 1, 3):

            # Menghitung faktor eliminasi.
            faktor = M[baris, kolom] / M[kolom, kolom]

            # Melakukan operasi pengurangan baris.
            M[baris, kolom:] = (
                M[baris, kolom:]
                - faktor * M[kolom, kolom:]
            )

    # Memeriksa pivot terakhir.
    if np.isclose(M[2, 2], 0):
        raise ValueError("Tidak ada solusi tunggal")

    # Menghitung nilai z.
    z = M[2, 3] / M[2, 2]

    # Menghitung nilai y.
    y = (M[1, 3] - M[1, 2] * z) / M[1, 1]

    # Menghitung nilai x.
    x = (M[0, 3] - M[0, 1] * y - M[0, 2] * z) / M[0, 0]

    # Mengembalikan nilai x, y, dan z.
    return np.array([x, y, z])


# CELL 4: Menjalankan perhitungan
solusi = eliminasi_gauss_3_variabel(A)

# Menampilkan hasil perhitungan.
print("Hasil Penyelesaian SPL:")
print("Nilai x =", solusi[0])
print("Nilai y =", solusi[1])
print("Nilai z =", solusi[2])


# CELL 5: Memeriksa hasil perhitungan
# Mengambil koefisien dan konstanta.
koefisien = A[:, :3]
konstanta = A[:, 3]

# Menghitung ruas kiri persamaan.
hasil = koefisien @ solusi

# Menampilkan pemeriksaan setiap persamaan.
print("Pemeriksaan Hasil:")

for i in range(3):
    print(
        "Persamaan", i + 1,
        "=", hasil[i],
        "Konstanta =", konstanta[i]
    )

# Memeriksa kesesuaian hasil.
if np.allclose(hasil, konstanta):
    print("Perhitungan berhasil!")
else:
    print("Perhitungan belum sesuai!")
