# Formal Counterexample Report

This report summarizes all **SAT / FAIL** solver cases found in the formal fairness checks.

## Overview

| Dataset | Model | Sensitive Attribute | Counterexample Pair | Prediction x | Prediction y | Score x | Score y | Changed Features |
|---|---|---|---|---:|---:|---|---|---:|
| scholarship | Logistic Regression | Asal_Daerah | 3T -> Pedesaan | 0 | 1 | 0 | 19715089434651473/100000000000000000 | 1 |
| adult | Logistic Regression | sex | Female -> Male | 0 | 1 | 0 | 10368279548611263/12500000000000000 | 1 |
| german_credit | Logistic Regression | sex | female -> male | 0 | 1 | 0 | 3727060284807609/40000000000000000 | 1 |

## Detailed Counterexamples

### scholarship / Logistic Regression

- **Sensitive attribute**: Asal_Daerah
- **Counterexample pair**: 3T -> Pedesaan
- **Prediction before**: 0
- **Prediction after**: 1
- **Score before**: 0
- **Score after**: 19715089434651473/100000000000000000
- **Encoding scope**: exact linear threshold over transformed feature space (one-hot categoricals + standardized numerics)
- **Fairness property**: existence of two inputs equal on all non-sensitive transformed features, different on Asal_Daerah, with different model predictions

#### Changed Feature(s)

- **Asal_Daerah**: `3T` -> `Pedesaan`

#### Example x

```json
{
  "IPK": 2.819370651831912,
  "Semester": 4.4685,
  "Pendapatan_Orang_Tua": 5328574.04,
  "Jumlah_Tanggungan": 3.503,
  "Jumlah_Kegiatan_Relawan": 2.4925,
  "Skor_Motivasi_Tertulis": 99.99999999999386,
  "Skor_Keaktifan_Kampus": 50.00000000001464,
  "Rekam_Ketidakhadiran": 19.999999999999538,
  "Skor_Literasi_Finansial": 99.99999999998096,
  "Jurusan": "Manajemen",
  "Pernah_Mengulang": "Ya",
  "Pekerjaan_Orang_Tua": "Buruh",
  "Status_Tempat_Tinggal": "Kontrak",
  "Asal_Daerah": "3T",
  "Aktif_Organisasi": "Tidak",
  "Pengalaman_Magang": "Tidak",
  "Surat_Rekomendasi": "Ada",
  "Pernah_Mendapat_Beasiswa": "Tidak",
  "Pernah_Melawan_Hukum": "Tidak",
  "Jenis_Kelamin": "Laki-laki"
}
```

#### Example y

```json
{
  "IPK": 2.819370651831912,
  "Semester": 4.4685,
  "Pendapatan_Orang_Tua": 5328574.04,
  "Jumlah_Tanggungan": 3.503,
  "Jumlah_Kegiatan_Relawan": 2.4925,
  "Skor_Motivasi_Tertulis": 99.99999999999386,
  "Skor_Keaktifan_Kampus": 50.00000000001464,
  "Rekam_Ketidakhadiran": 19.999999999999538,
  "Skor_Literasi_Finansial": 99.99999999998096,
  "Jurusan": "Manajemen",
  "Pernah_Mengulang": "Ya",
  "Pekerjaan_Orang_Tua": "Buruh",
  "Status_Tempat_Tinggal": "Kontrak",
  "Asal_Daerah": "Pedesaan",
  "Aktif_Organisasi": "Tidak",
  "Pengalaman_Magang": "Tidak",
  "Surat_Rekomendasi": "Ada",
  "Pernah_Mendapat_Beasiswa": "Tidak",
  "Pernah_Melawan_Hukum": "Tidak",
  "Jenis_Kelamin": "Laki-laki"
}
```

### adult / Logistic Regression

- **Sensitive attribute**: sex
- **Counterexample pair**: Female -> Male
- **Prediction before**: 0
- **Prediction after**: 1
- **Score before**: 0
- **Score after**: 10368279548611263/12500000000000000
- **Encoding scope**: exact linear threshold over transformed feature space (one-hot categoricals + standardized numerics)
- **Fairness property**: existence of two inputs equal on all non-sensitive transformed features, different on sex, with different model predictions

#### Changed Feature(s)

- **sex**: `Female` -> `Male`

#### Example x

```json
{
  "age": 38.58164675532078,
  "fnlwgt": 1484704.999999985,
  "education_num": 15.999999999999371,
  "capital_gain": 1077.6488437087312,
  "capital_loss": 87.303829734959,
  "hours_per_week": 37.66761380793423,
  "workclass": "State-gov",
  "education": "10th",
  "marital_status": "Divorced",
  "occupation": "Adm-clerical",
  "relationship": "Husband",
  "race": "Asian-Pac-Islander",
  "sex": "Female",
  "native_country": "Cambodia"
}
```

#### Example y

```json
{
  "age": 38.58164675532078,
  "fnlwgt": 1484704.999999985,
  "education_num": 15.999999999999371,
  "capital_gain": 1077.6488437087312,
  "capital_loss": 87.303829734959,
  "hours_per_week": 37.66761380793423,
  "workclass": "State-gov",
  "education": "10th",
  "marital_status": "Divorced",
  "occupation": "Adm-clerical",
  "relationship": "Husband",
  "race": "Asian-Pac-Islander",
  "sex": "Male",
  "native_country": "Cambodia"
}
```

### german_credit / Logistic Regression

- **Sensitive attribute**: sex
- **Counterexample pair**: female -> male
- **Prediction before**: 0
- **Prediction after**: 1
- **Score before**: 0
- **Score after**: 3727060284807609/40000000000000000
- **Encoding scope**: exact linear threshold over transformed feature space (one-hot categoricals + standardized numerics)
- **Fairness property**: existence of two inputs equal on all non-sensitive transformed features, different on sex, with different model predictions

#### Changed Feature(s)

- **sex**: `female` -> `male`

#### Example x

```json
{
  "duration_months": 21.048571428571428,
  "credit_amount": 250.00000000128057,
  "installment_rate": 1.0000000000007323,
  "present_residence_since": 1.0000000000008276,
  "age_years": 40.00368622547428,
  "existing_credits": 1.4142857142857144,
  "people_liable": 1.1628571428571428,
  "status_checking": "A13",
  "credit_history": "A32",
  "purpose": "A40",
  "savings": "A61",
  "employment_since": "A71",
  "personal_status_sex": "A91",
  "other_debtors": "A101",
  "property": "A121",
  "other_installment_plans": "A141",
  "housing": "A151",
  "job": "A172",
  "telephone": "A191",
  "foreign_worker": "A201",
  "sex": "female"
}
```

#### Example y

```json
{
  "duration_months": 21.048571428571428,
  "credit_amount": 250.00000000128057,
  "installment_rate": 1.0000000000007323,
  "present_residence_since": 1.0000000000008276,
  "age_years": 40.00368622547428,
  "existing_credits": 1.4142857142857144,
  "people_liable": 1.1628571428571428,
  "status_checking": "A13",
  "credit_history": "A32",
  "purpose": "A40",
  "savings": "A61",
  "employment_since": "A71",
  "personal_status_sex": "A91",
  "other_debtors": "A101",
  "property": "A121",
  "other_installment_plans": "A141",
  "housing": "A151",
  "job": "A172",
  "telephone": "A191",
  "foreign_worker": "A201",
  "sex": "male"
}
```
