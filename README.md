# Sistem Monitoring dan Deteksi DDoS BPS

Proyek ini berisi persiapan data dan eksperimen awal deteksi trafik. Belum ada
model yang layak disebut atau digunakan sebagai detektor DDoS produksi, dan
belum ada integrasi ke jaringan BPS.

## Status berdasarkan pemeriksaan 3 November 2026

| Area | Kondisi aktual |
|---|---|
| Dataset flow berlabel publik | `training_features_public_attack_labeled.csv`: 20.984 window, 2.576 berlabel serangan (`-1`), 18.408 belum berlabel, dan tidak ada label normal (`+1`). Belum dapat dipakai untuk training biner. |
| Ground truth BPS | Belum tersedia. `data/labels/` belum memiliki manifest terverifikasi; template hanya contoh dan seluruh entri masih `pending`. Data BPS tidak boleh dipakai untuk klaim evaluasi. |
| Data train/test model lama | Ada file harmonized dengan label biner, tetapi sumber train/test mencakup file CIC-IDS2018. Pipeline lama menganggap setiap label selain `BENIGN` sebagai serangan, termasuk kelas non-DDoS. Hasilnya bukan dataset DDoS yang valid. |
| Model tersimpan | `models/traffic_sgd.joblib` adalah baseline `SGDClassifier`, bukan Decision Tree. Artefak dipertahankan untuk jejak eksperimen, tetapi jangan digunakan untuk deteksi DDoS. |
| Evaluasi model lama | Recall attack 1,77%, precision attack 10,77%, F1 attack 3,04%, ROC-AUC 0,634. Kinerja serangan tidak memadai, selain masalah validitas label/sumber data. |
| Aplikasi dashboard | Belum ada entry point aplikasi; folder `dashboard/` masih berupa kerangka. |

### Kesimpulan kualitas data

- Label numerik pada dataset turunan memang ada, tetapi keberadaan label tidak
  menjamin label itu benar untuk tugas DDoS. Model lama memasukkan kelas seperti
  brute force, bot, dan infiltration sebagai `attack`.
- `Random_combine_final.csv` memiliki label DDoS, tetapi provenance dan proses
  penggabungannya belum terverifikasi.
- Dataset fitur dari capture publik berisi sebagian label attack yang dipetakan
  dari rentang waktu terdokumentasi. Window di luar rentang itu tetap kosong,
  bukan `BENIGN`.
- Persiapan CSV kini hanya menerima label yang tercantum dalam
  `config/label_mapping.json`. Label lain akan ditolak, bukan otomatis dianggap
  serangan. Data CIC-IDS2018 yang memuat label non-DDoS harus dikeluarkan atau
  diproses dengan pemetaan tugas yang sesuai sebelum training.
- Jangan retrain dari `data/processed/csv_train_test/` atau
  `data/processed/harmonized/` yang sudah ada: hasil tersebut dibuat sebelum
  validasi label diperketat.

## Struktur saat ini

```text
config/                 Konfigurasi dan pemetaan label yang diizinkan
data/
  labels/               Template/manifes ground truth
  raw/
    normal/             Capture yang label normalnya sudah diverifikasi
    attack/             Capture DDoS yang label serangannya sudah diverifikasi
  processed/            Hasil pipeline dan dataset turunan
  reports/              Laporan profiling dan evaluasi
models/                 Artefak eksperimen (bukan model produksi)
src/
  data/                 Profiling data
  ml/                   Area modul ML
  monitoring/           Area modul monitoring (belum diimplementasikan)
  database/             Area modul database (belum diimplementasikan)
  *.py                  Skrip pipeline/eksperimen yang masih digunakan
tests/                  Tes unit dan integrasi
docs/                   Dokumen kebutuhan dan rencana
dashboard/              Kerangka dashboard; aplikasi belum tersedia
notebooks/              Notebook eksperimen
```

Dataset mentah, hasil olahan besar, laporan lokal, dan model tidak dimasukkan ke
Git. File `.gitkeep` dipakai untuk mempertahankan direktori kosong.

### Pengelompokan data raw

Simpan sumber yang labelnya telah diverifikasi di `data/raw/normal/` atau
`data/raw/attack/`. Folder `attack/` khusus untuk DDoS yang terkonfirmasi,
bukan semua jenis serangan. Data yang belum jelas label atau provenancenya
jangan dimasukkan ke salah satu kelas; biarkan di luar folder kelas sampai
bukti dan manifest ground truth diverifikasi. Nama folder tidak menggantikan
bukti label, dan file sumber harus dipertahankan tanpa perubahan.

### Ekstraksi PCAP/PCAPNG ke fitur CSV

Fitur trafik dapat diekstrak per detik tanpa mengubah file capture:

```powershell
python -m src.prepare_training_data "data/raw/normal/nama-capture.pcapng" `
  --output "data/processed/normal_features.csv" --window-seconds 1
```

Jika manifest ground truth terverifikasi tidak diberikan, kolom `label` sengaja
dibiarkan kosong. CSV hasil ekstraksi memiliki 16 fitur window proyek; skemanya
berbeda dari fitur CICFlowMeter yang dipakai artefak kandidat lama, sehingga
tidak boleh langsung diberikan ke model tersebut. Model harus dilatih dan
dievaluasi ulang dengan fitur yang sama dan label yang terverifikasi.

### Harmonisasi fitur per-flow

Untuk membandingkan capture normal dengan CSV attack CIC-DDoS2019, ekstrak
capture normal memakai CICFlowMeter, lalu selaraskan nama fitur tanpa
mengubah nilai:

```powershell
python -m src.harmonize_cicflowmeter_features `
  --normal-csv "C:\path\ke\hasil-cicflowmeter.csv" `
  --attack-csv "data/raw/attack/cicddos2019_attack.csv" `
  --output-dir "data/processed/cicflowmeter_harmonized"
```

Skrip membuang identifier/alamat/port dan indeks baris, menyelaraskan alias
kolom, serta menghasilkan dua CSV dengan urutan fitur yang sama. `Class` pada
normal tetap kosong dan `Label` CICFlowMeter tetap `NeedManualLabel` sampai
provenance serta ground truth dicatat di manifest. Kolom `CWR Flag Count` dan
`CWE Flag Count` dikecualikan karena kesetaraan semantiknya belum dibuktikan.
Nilai negatif, kosong, dan non-finite dipertahankan serta dilaporkan, bukan
dihapus atau dikoreksi otomatis.

Kesamaan header belum membuktikan kesamaan algoritma, satuan, atau proses
ekstraksi di kedua sumber. File hasil belum siap dipakai melatih model sampai
provenance/label ditinjau, nilai anomali diperiksa, dan evaluasi memakai capture
independen yang dibekukan.

### Status provenance dan kualitas hasil flow

`data/labels/dataset_provenance_inventory.csv` mencatat hash, lineage, jumlah
record, klaim label, dan kekurangan bukti untuk sumber serta turunannya.
Inventaris ini **bukan** ground-truth manifest dan tidak boleh dipakai untuk
memberi label training. Semua review masih `pending_verification`; nama file,
folder, atau klaim pengguna bukan bukti independen bahwa sumber maupun label
benar.

Pemeriksaan lokal memastikan CSV attack-only berisi tepat 333.540 baris
`Class=Attack` yang cocok dengan sumber; 97.831 baris `Benign` dikeluarkan.
Semua 17 nilai `Label` ditemukan dalam `config/label_mapping.json`. Ini
membuktikan konsistensi/filtering internal, bukan provenance atau akurasi
ground truth.

Pengguna mengidentifikasi capture Monday sebagai CIC-IDS2017 dari
<https://www.unb.ca/cic/datasets/ids-2017.html>. Dokumentasi resmi menyatakan
Monday, 3 Juli 2017, hanya berisi trafik benign. Ini mendukung label yang
diharapkan, tetapi file lokal belum dicocokkan dengan checksum resmi; label
capture tetap pending. CSV attack diidentifikasi pengguna sebagai
`cicddos2019_dataset.csv` dari CIC-DDoS2019; dokumentasi umum dataset tersedia di
<https://www.unb.ca/cic/datasets/ddos-2019.html>, tetapi URL/identitas rilis
unduhan yang persis dan review label per baris belum dicatat.

Ekstraksi normal menggunakan CICFlowMeter 4.0 (commit
`98a5ebad0df579cc8b43eedd3421b3ae87699901`), memproses 11.662.315 dan
membuang 47.656 dari 11.709.971 paket. Nilai non-finite dan negatif ditemukan
pada kedua sumber; nilainya dipertahankan dan belum ditinjau sebagai sentinel,
overflow, atau nilai valid. Jangan melatih model dari CSV tersebut.

### Rencana pengumpulan trafik normal tambahan (P0)

Capture Monday adalah satu hari benchmark publik, bukan sampel yang mewakili
semua layanan atau kondisi operasional. Untuk menambah variasi tanpa menyalin
baris, kumpulkan capture asli yang terpisah pada jaringan lab yang dimiliki
atau secara eksplisit diizinkan:

| Dimensi | Rencana pencatatan |
|---|---|
| Titik capture | Interface mirror/SPAN yang disetujui atau gateway lab; catat interface, topologi, filter, snap length, versi alat, dan packet-drop counter. |
| Perangkat | Catat peran dan OS perangkat klien, server, gateway, serta perangkat layanan; hindari menyimpan identitas sensitif yang tidak diperlukan. |
| Layanan | Pilih layanan yang benar-benar digunakan pada lab, misalnya DNS, DHCP, web HTTPS, email, dan transfer berkas; catat aplikasi/protokol dan aktivitasnya. |
| Waktu/kondisi | Ambil beberapa hari kerja dan akhir pekan serta jam sibuk/sepi; dokumentasikan aktivitas, perubahan konfigurasi, maintenance, insiden, dan timezone. Sinkronkan jam ke UTC. |
| Label | Mulai sebagai `unlabeled`. Beri label normal hanya setelah pemeriksaan log/insiden untuk rentang capture tersebut dan review oleh penanggung jawab; interval yang tidak pasti tetap unlabeled. Simpan bukti dan reviewer di manifest, jangan menyimpulkan normal dari tidak adanya alert. |
| Keamanan | Jangan capture jaringan BPS atau jaringan pihak lain tanpa izin tertulis. Mulai dari mode offline/alert-only; batasi akses dan retensi PCAP karena payload dapat memuat data sensitif. Jangan membuat serangan untuk menambah dataset normal. |

Tentukan kecukupan jumlah capture berdasarkan target false-positive rate dan
interval ketidakpastian yang disepakati, bukan dengan menggandakan capture atau
menetapkan jumlah baris secara arbitrer. Bekukan hari/capture uji terpisah
sebelum pemilihan model maupun threshold.

### Dataset seimbang untuk eksperimen pengembangan

CSV sumber CIC-DDoS2019 campuran yang tersedia di Temp memiliki 97.831 baris
`Benign` dan 333.540 baris `Attack`. Untuk eksperimen awal, skrip mengambil
seluruh baris benign dan mengambil 97.831 attack tanpa penggantian, dengan
sampling terstratifikasi pada subtype. Tidak ada baris yang disalin.

```powershell
python -m src.prepare_balanced_cicddos2019 `
  --input (Join-Path $env:TEMP "cicddos2019_dataset.csv") `
  --output "data/processed/cicflowmeter_harmonized/cicddos2019_balanced_training.csv" `
  --report "data/reports/cicddos2019_balanced_training.json"

python -m src.train_ddos_candidate `
  --input "data/processed/cicflowmeter_harmonized/cicddos2019_balanced_training.csv" `
  --model "models/cicddos2019_development_candidate.joblib" `
  --report "data/reports/cicddos2019_development_candidate.json"
```

Dataset kandidat saat ini berisi 97.831 flow per kelas dan 71 fitur numerik.
Enam kolom dikecualikan: `CWE Flag Count` (semantik belum diselaraskan),
`Fwd/Bwd Header Length` dan `Fwd Seg Size Min` (nilai overflow negatif), serta
`Init Fwd/Bwd Win Bytes` (sentinel `-1` yang sangat umum). Fitur terpakai
divalidasi numerik, finite, dan non-negatif; data mentah tidak ditimpa.
Sampling attack menjaga setiap subtype, tetapi subtype langka tetap memiliki
sampel sedikit.

Pelatihan membandingkan Decision Tree dan ExtraTrees pada split internal yang
menahan subtype attack; baris benign dibagi acak karena sumber tidak menyertakan
capture/day ID. Laporan menampilkan confusion matrix, precision/recall/F1,
PR-AUC, false-positive rate normal beserta interval Wilson, dan recall per
subtype. Semua skor tersebut hanya diagnostik dalam satu dataset, **bukan**
test capture/hari independen. Model tersimpan berstatus provisional untuk
eksperimen pipeline dan tidak boleh dipakai untuk mitigasi atau klaim akurasi
operasional.

Pemeriksaan diagnostik lintas-sumber pada 4 November 2026 menjalankan kandidat
tersebut pada 453.650 flow dari capture Monday yang masih berstatus
`NeedManualLabel`. Kandidat memprediksi 52.571 flow sebagai attack pada threshold
0,5 dan 71.561 pada threshold validation 0,3849. Angka ini bukan false-positive
rate terverifikasi karena label dan identitas capture belum direview; selain
itu, fitur normal diekstrak dengan CICFlowMeter 4.0, sementara fitur training
berasal dari CICFlowMeter-V3. Namun sinyalnya cukup kuat untuk menghentikan
deployment: jangan memakai kandidat atau menyesuaikan threshold dari data ini.
Lihat `data/reports/cicddos2019_cross_source_normal_diagnostic.json`.

### Rancangan test independen (P0)

- Bagi berdasarkan **capture/hari asal**, bukan baris atau flow acak; satu
  capture beserta semua window/flow-nya hanya boleh berada di satu split.
- Sediakan development untuk training, validation terpisah untuk memilih model
  dan threshold, serta final test dari hari/capture berbeda yang dibekukan.
- Catat hash PCAP/CSV, manifest label dan bukti review, versi ekstraktor, skema,
  serta aturan preprocessing sebelum membuka hasil final test.
- Jangan gunakan final test untuk pemilihan fitur/model, threshold, atau
  perbaikan data. Jika aturan berubah setelah melihat test, siapkan test baru.
- CSV attack saat ini tidak memuat identitas capture/hari yang dapat dipakai
  untuk split independen, sedangkan normal yang tersedia baru satu hari Monday.
  Karena itu belum ada pembagian final test yang sah; diperlukan sumber attack
  berlabel dengan capture/day ID dan capture normal tambahan yang terverifikasi.

### Dataset uji diagnostik sementara

Untuk menguji alur prediksi offline sambil menunggu capture independen, skrip
berikut membuat sampel seimbang dari sumber lokal. Skrip mengambil 20.000 flow
normal dan 20.000 flow attack secara deterministik, lalu mengeluarkan baris
attack yang memiliki fingerprint fitur dan label sama dengan data training.
Label normal `BENIGN` mengikuti konfirmasi pengguna dan dokumentasi Monday
CIC-IDS2017; identitas/checksum capture lokal masih perlu diverifikasi.

```powershell
python -m src.prepare_cicddos2019_diagnostic_test
```

CSV dibuat di
`data/processed/cicflowmeter_harmonized/cicddos2019_cross_source_diagnostic_test.csv`
dan audit sumber, hash, sampling, serta keterbatasannya di
`data/reports/cicddos2019_cross_source_diagnostic_test.json`. Untuk menjalankan
kandidat dan menghitung metrik dari label tersebut:

```powershell
python -m src.predict_ddos_candidate `
  --input "data/processed/cicflowmeter_harmonized/cicddos2019_cross_source_diagnostic_test.csv" `
  --output "data/reports/cicddos2019_cross_source_diagnostic_predictions.csv" `
  --label-column Label
```

Ini hanya **uji diagnostik lintas-sumber**, bukan pengganti test final:
fingerprint identik yang dikeluarkan tidak membuktikan pemisahan capture,
capture/day ID attack tidak tersedia, extractor normal CICFlowMeter 4.0 berbeda
dari sumber attack CICFlowMeter-V3, dan distribusi kelas sengaja dibuat 50:50.
Jangan gunakan metriknya untuk tuning threshold atau menyatakan kesiapan
operasional.

Pada eksekusi 5 November 2026, kandidat mendapat akurasi 91,98% dan recall
attack 99,83%, tetapi salah menandai 3.174 dari 20.000 flow normal sebagai
attack (false-positive rate 15,87%). Angka ini hanya menggambarkan sampel
diagnostik dengan label/provenance dan extractor yang memiliki batasan di atas;
ini bukan estimasi performa pada trafik nyata. Rincian tersimpan di
`data/reports/cicddos2019_cross_source_diagnostic_predictions.report.json`.

## Menjalankan pemeriksaan

Jalankan dari root proyek di Windows:

```powershell
python -m pytest -q
python -m src.data.profiling --file data/raw/Random_combine_final.csv
```

Perintah training belum direkomendasikan sampai sumber data, label benign, dan
label DDoS diverifikasi. Setelah dataset terverifikasi dipilih, siapkan subset
input khusus DDoS dan jalankan pipeline pada subset tersebut; jangan arahkan
pipeline ke seluruh `data/raw/`.

### Eksperimen kandidat pada data gabungan (provisional)

Jika ingin mereproduksi eksperimen pada `Random_combine_final.csv`, jalankan:

```powershell
python -m src.train_ddos_candidate
```

Eksperimen ini menyimpan kandidat terpisah di
`models/traffic_ddos_candidate.joblib` beserta laporan
`data/reports/traffic_ddos_candidate_evaluation.json`; model lama tidak ditimpa.
Dataset hanya memiliki sedikit sampel `BENIGN`, provenance belum diverifikasi,
dan evaluasi menahan keluarga label serangan, tetapi tetap berasal dari satu
file. Karena itu hasilnya provisional dan tidak boleh dianggap validasi
independen atau siap produksi.

### Gate kesiapan kandidat CIC-DDoS2019 (4 November 2026)

Pengguna menyatakan seluruh data telah diverifikasi. Pernyataan tersebut dicatat,
namun tidak mengubah perbedaan teknis pada sumber: CSV attack CIC-DDoS2019 dibuat
dengan CICFlowMeter-V3, sedangkan capture normal Monday diekstrak dengan
CICFlowMeter 4.0. Pada 453.650 flow dari capture normal tersebut, kandidat
menandai 52.571 flow (11,59%) sebagai attack pada threshold 0,5. Pemisahan
berdasarkan jam juga menunjukkan 5,1%-17,8% prediksi attack per blok waktu.
Karena source CSV masih mencatat `NeedManualLabel`, satu hari/capture saja
tersedia, dan extractor berbeda, angka tersebut adalah diagnostik bersyarat,
bukan FPR tersertifikasi.

**Keputusan: no-go untuk deployment atau mitigasi otomatis.** Threshold internal
0,3849 menghasilkan lebih banyak prediksi attack pada capture normal; jangan
gunakan atau retune threshold itu pada capture ini. Sebelum kandidat dapat
dipertimbangkan untuk pemakaian, samakan versi/konfigurasi ekstraktor,
pertahankan capture/day ID, dan validasi pada beberapa capture normal serta
attack yang independen dan dibekukan sebelum tuning. Hasil lengkap tercatat di
`data/reports/cicddos2019_cross_source_normal_diagnostic.json` dan
`data/reports/cicddos2019_normal_time_block_diagnostic.json`.

### Pemeriksaan grup PCAP baru (5 November 2026)

Folder `E:\PCAP-01-12_0250-0499` berisi 250 chunk PCAP berurutan bernomor
0250-0499 (total 50.000.154.070 byte), bukan 250 capture independen. Header PCAP
dan waktu paket sampel konsisten dengan tanggal 1 Desember 2018; nama saja belum
membuktikan sumber CIC-DDoS2019. Folder tidak memiliki manifest/label per file.
Jika grup ini memang CIC-DDoS2019 hari pertama, jadwal resmi menunjukkan periode
yang berpotensi mencakup interval attack dan benign; label final tetap unknown
sampai sumber dan timezone terverifikasi.

CSV training CIC-DDoS2019 yang tersedia tidak menyimpan capture/day ID atau
timestamp, sehingga overlap persis dengan 250 chunk tidak dapat diperiksa.
Apabila PCAP ini bagian dari rilis yang sama dengan CSV tersebut, jangan
perlakukan sebagai test independen. Rincian fingerprint inventaris dan bukti
terbatas dicatat di
`data/reports/pcap_group_identity_label_overlap_20261105.json`; status provenance
tetap `pending_verification`.

Pilot ekstraksi belum dijalankan: lingkungan yang tersedia hanya memiliki
CICFlowMeter 4.0, sedangkan fitur kandidat berasal dari CSV CICFlowMeter-V3.
Menjalankan V4 sebagai pengganti lalu menganggapnya validasi model tidak dapat
membedakan kegagalan model dari pergeseran ekstraktor. Folder juga belum
memiliki manifest label. Sebelum pilot, sediakan CICFlowMeter-V3 dan sumber
tepercaya/label ber-timestamp; ekstrak hanya satu chunk/slice ke direktori
sementara, bukan seluruh 50 GB. Gate dan prasyarat tercatat di
`data/reports/pcap_pilot_gate_20261105.json`.

### Uji offline flow CSV dengan kandidat

Kandidat dapat dijalankan untuk **uji offline** pada CSV flow CICFlowMeter. PCAP
harus diekstrak terlebih dahulu; perintah ini tidak menangkap jaringan, tidak
mengubah PCAP, dan tidak melakukan mitigasi. Kolom fitur wajib harus cocok
dengan 71 fitur model. Alias header CICFlowMeter yang sudah dikenal dipetakan
otomatis; fitur yang hilang ditolak. Baris tanpa satu pun fitur numerik valid
ditandai `NOT_SCORED`, bukan dipaksa menjadi normal/attack.

```powershell
python -m src.predict_ddos_candidate `
  --input "data/processed/cicflowmeter_harmonized/normal_features_unlabeled.csv" `
  --output "data/reports/normal_candidate_predictions.csv" `
  --capture-id "monday-normal-diagnostic"
```

Jika CSV uji memiliki label yang sudah diverifikasi, tambahkan
`--label-column Class` (atau nama kolom label sebenarnya) untuk menghitung
metrik uji. Label harus sesuai pemetaan `config/label_mapping.json` atau bernilai
`NORMAL`/`ATTACK`. Laporan JSON mencatat hash input/model, threshold, ringkasan
prediksi, dan metrik jika label tersedia. Default threshold berasal dari
validation internal model; mengubah threshold berdasarkan hasil test tidak
dibenarkan. Output yang sudah ada tidak ditimpa kecuali dengan `--overwrite`.

Runner ini membuat kandidat dapat dieksekusi dan diaudit pada CSV, tetapi tidak
memperbaiki generalisasi model: hasil tetap provisional, extractor CICFlowMeter
harus konsisten, dan tidak ada persetujuan untuk penggunaan operasional atau
mitigasi otomatis.

### Mencoba prediksi melalui aplikasi lokal

PCAP klasik yang diunggah dikonversi otomatis oleh distribusi CICFlowMeter
sebelum prediksi. Pasang distribusi CICFlowMeter (yang menyertakan folder
`bin`, `lib`, dan `lib/native`), lalu tunjuk folder distribusinya lewat
`CICFLOWMETER_HOME`. Aplikasi menjalankan kelas CLI `cic.cs.unb.ca.ifm.Cmd`
secara langsung melalui Java; pengguna dashboard tidak perlu membuat CSV atau
menulis perintah konversi setiap kali mengunggah file.

```powershell
$env:CICFLOWMETER_HOME = "C:\tools\CICFlowMeter-4.0"
$env:CICFLOWMETER_VERSION = "4.0"
python app.py
```

Folder distribusi dapat disimpan di `tools\CICFlowMeter` agar tidak perlu
mengatur `CICFLOWMETER_HOME`. Java dan library native jNetPcap yang sesuai
dengan OS harus tersedia. Opsi `CICFLOWMETER_COMMAND` tetap tersedia untuk
instalasi khusus, sebagai JSON array argumen dengan placeholder `{input}` dan
`{output_dir}`. Ekstraktor saat ini mendukung PCAP klasik; PCAPNG perlu
dikonversi ke PCAP klasik terlebih dahulu.

Buka `http://127.0.0.1:5000`. Pilih CSV contoh yang tersedia, atau unggah CSV
flow maupun PCAP klasik (maksimal 512 MB). Untuk PCAP, aplikasi menjalankan
ekstraksi lokal terlebih dahulu, memeriksa bahwa hasil ekstraksi tepat satu CSV,
lalu memeriksa skema fitur model dan memprediksi tiap flow. File unggahan dan CSV
sementara dihapus setelah proses; CSV prediksi dan laporan audit disimpan di
`data/reports/` dan dapat diunduh. Untuk CSV normal Monday, biarkan kolom label
kosong: labelnya masih `NeedManualLabel`, jadi metrik akurasi tidak dapat
dihitung.

Model memakai fitur turunan CICFlowMeter-V3. Versi/konfigurasi lain dapat
mengubah fitur atau nilai; kesamaan nama kolom saja tidak menjamin hasil setara.
Aplikasi hanya memproses file lokal, bukan capture jaringan langsung atau
mitigasi.

Hasil eksperimen 3 November 2026: kandidat `ExtraTreesClassifier` mendapat
akurasi 99,99% dan recall serangan 99,995% pada test (threshold 0,5). Test
memuat 59.745 sampel serangan tetapi hanya 98 sampel normal; 2 sampel normal
salah terdeteksi sebagai serangan. Metrik tinggi ini belum membuktikan performa
operasional karena provenance file tidak diketahui dan test bukan capture
independen. Artefak lama `models/traffic_sgd.joblib` tidak ditimpa.

## Batasan operasional

1. Metrik dari dataset publik tidak membuktikan performa pada trafik BPS.
2. Tidak ada mitigasi otomatis; sistem hanya boleh menghasilkan peringatan.
3. Integrasi jaringan BPS memerlukan izin dan pengujian terpisah.
