import streamlit as st
import numpy as np
import pandas as pd

# Fungsi untuk SAW Method
def saw_method(alternatives, weights, criteria):
    normalized = []
    for i in range(len(weights)):
        if criteria[i] == 'keuntungan':
            max_val = max([alt[i] for alt in alternatives])
            normalized.append([alt[i] / max_val for alt in alternatives])
        elif criteria[i] == 'biaya':
            min_val = min([alt[i] for alt in alternatives])
            normalized.append([min_val / alt[i] for alt in alternatives])
    normalized = np.array(normalized).T
    scores = [sum([n * w for n, w in zip(alt, weights)]) for alt in normalized]
    return normalized, scores

# Fungsi untuk WP Method
def wp_method(alternatives, weights, criteria):
    normalized = []
    for i in range(len(weights)):
        if criteria[i] == 'keuntungan':
            max_val = max([alt[i] for alt in alternatives])
            normalized.append([alt[i] / max_val for alt in alternatives])
        elif criteria[i] == 'biaya':
            min_val = min([alt[i] for alt in alternatives])
            normalized.append([min_val / alt[i] for alt in alternatives])
    normalized = np.array(normalized).T
    scores = []
    for alt in normalized:
        product = 1
        for n, w in zip(alt, weights):
            product *= n ** w
        scores.append(product)
    return normalized, scores

# Fungsi untuk TOPSIS Method
def topsis_method(alternatives, weights, criteria):
    norm_matrix = np.array(alternatives) / np.sqrt(np.sum(np.square(alternatives), axis=0))
    weighted_matrix = norm_matrix * weights
    ideal_best = []
    ideal_worst = []
    for i in range(len(criteria)):
        if criteria[i] == 'keuntungan':
            ideal_best.append(np.max(weighted_matrix[:, i]))
            ideal_worst.append(np.min(weighted_matrix[:, i]))
        elif criteria[i] == 'biaya':
            ideal_best.append(np.min(weighted_matrix[:, i]))
            ideal_worst.append(np.max(weighted_matrix[:, i]))
    distance_best = np.sqrt(np.sum((weighted_matrix - ideal_best) ** 2, axis=1))
    distance_worst = np.sqrt(np.sum((weighted_matrix - ideal_worst) ** 2, axis=1))
    scores = distance_worst / (distance_best + distance_worst)
    return weighted_matrix, ideal_best, ideal_worst, scores

# Fungsi untuk AHP Method: Menghitung bobot kriteria
def ahp_method(criteria_matrix):
    column_sums = np.sum(criteria_matrix, axis=0)
    normalized_matrix = criteria_matrix / column_sums
    weights = np.mean(normalized_matrix, axis=1)
    return weights

# Fungsi untuk AHP Method: Menghitung skor alternatif berdasarkan perbandingan berpasangan untuk setiap kriteria
def ahp_alternatives(alternative_matrices, weights):
    final_scores = []
    for matrix in alternative_matrices:
        column_sums = np.sum(matrix, axis=0)
        normalized_matrix = matrix / column_sums
        scores = np.mean(normalized_matrix, axis=1)
        final_scores.append(scores)
    
    final_scores = np.array(final_scores).T
    weighted_scores = np.dot(final_scores, weights)
    return final_scores, weighted_scores

# Fungsi untuk menampilkan ranking alternatif
def rank_alternatives(scores):
    ranking = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)
    return ranking

# Fungsi untuk memberi label alternatif
def label_alternatives(indexes):
    return {i + 1: f"A{index + 1}" for i, index in enumerate(indexes)}

# Fungsi utama untuk Streamlit
def main():
    st.title("Aplikasi Pengambilan Keputusan")

    # Pilih metode
    method = st.selectbox("Pilih metode pengambilan keputusan", ["SAW", "WP", "TOPSIS", "AHP"])

    # Input jumlah kriteria dan alternatif jika metode selain AHP
    if method != "AHP":
        num_criteria = st.number_input("Masukkan jumlah kriteria", min_value=1, value=3)
        num_alternatives = st.number_input("Masukkan jumlah alternatif", min_value=1, value=3)

        # Input nama kriteria
        criteria = []
        criteria_types = []
        for i in range(num_criteria):
            crit_name = st.text_input(f"Masukkan nama kriteria {i+1}")
            crit_type = st.selectbox(f"Apakah kriteria {i+1} adalah biaya atau keuntungan?", ["biaya", "keuntungan"], key=i)
            criteria.append(crit_name)
            criteria_types.append(crit_type)

        # Input bobot kriteria
        weights = []
        for i in range(num_criteria):
            weight = st.number_input(f"Masukkan bobot untuk kriteria {i+1} (misal 0.2)", min_value=0.0, max_value=1.0, value=0.1)
            weights.append(weight)

        # Input nilai untuk setiap alternatif
        alternatives = []
        for i in range(num_alternatives):
            st.subheader(f"Alternatif {i+1}")
            alternative = []
            for j in range(num_criteria):
                value = st.number_input(f"Masukkan nilai untuk kriteria {criteria[j]} alternatif {i+1}", key=f"{i}-{j}")
                alternative.append(value)
            alternatives.append(alternative)

    # Input matriks AHP (pairwise comparison) jika metode AHP
    if method == "AHP":
        num_criteria = st.number_input("Masukkan jumlah kriteria", min_value=2, value=3)
        num_alternatives = st.number_input("Masukkan jumlah alternatif", min_value=2, value=3)
        
        st.subheader("Matriks Perbandingan Kriteria untuk AHP")
        ahp_matrix = []
        for i in range(num_criteria):
            row = []
            for j in range(num_criteria):
                value = st.number_input(f"Nilai perbandingan kriteria {i+1} dengan kriteria {j+1}", key=f"ahp-{i}-{j}", value=1.0)
                row.append(value)
            ahp_matrix.append(row)

        # Input perbandingan berpasangan untuk alternatif di setiap kriteria
        alternative_matrices = []
        for i in range(num_criteria):
            st.subheader(f"Perbandingan Alternatif untuk Kriteria {i+1}")
            alt_matrix = []
            for j in range(num_alternatives):
                row = []
                for k in range(num_alternatives):
                    value = st.number_input(f"Alternatif {j+1} vs Alternatif {k+1} di Kriteria {i+1}", key=f"alt-{i}-{j}-{k}", value=1.0)
                    row.append(value)
                alt_matrix.append(row)
            alternative_matrices.append(alt_matrix)

    # Proses perhitungan berdasarkan metode yang dipilih
    if st.button("Hitung"):
        st.subheader("Hasil Perhitungan")
        
        if method == "SAW":
            normalized, saw_scores = saw_method(alternatives, weights, criteria_types)
            saw_ranking = rank_alternatives(saw_scores)
            
            # Menampilkan tabel normalisasi
            normalized_df = pd.DataFrame(normalized, columns=criteria, index=[f"A{i+1}" for i in range(len(alternatives))])
            st.write("Tabel Normalisasi (SAW):")
            st.dataframe(normalized_df)

            st.write("SAW Scores:", dict(zip(label_alternatives(range(len(saw_scores))), saw_scores)))
            st.write("Ranking Alternatif (SAW):", label_alternatives(saw_ranking))
        
        elif method == "WP":
            normalized, wp_scores = wp_method(alternatives, weights, criteria_types)
            wp_ranking = rank_alternatives(wp_scores)

            # Menampilkan tabel normalisasi
            normalized_df_wp = pd.DataFrame(normalized, columns=criteria, index=[f"A{i+1}" for i in range(len(alternatives))])
            st.write("Tabel Normalisasi (WP):")
            st.dataframe(normalized_df_wp)

            st.write("WP Scores:", dict(zip(label_alternatives(range(len(wp_scores))), wp_scores)))
            st.write("Ranking Alternatif (WP):", label_alternatives(wp_ranking))
        
        elif method == "TOPSIS":
            weighted_matrix, ideal_best, ideal_worst, topsis_scores = topsis_method(alternatives, weights, criteria_types)
            
            # Menampilkan tabel hasil TOPSIS
            weighted_df = pd.DataFrame(weighted_matrix, columns=criteria, index=[f"A{i+1}" for i in range(len(alternatives))])
            st.write("Tabel Matriks Tertimbang (TOPSIS):")
            st.dataframe(weighted_df)

            st.write("Ideal Best:", ideal_best)
            st.write("Ideal Worst:", ideal_worst)
            st.write("TOPSIS Scores:", topsis_scores)
            topsis_ranking = rank_alternatives(topsis_scores)
            st.write("Ranking Alternatif (TOPSIS):", label_alternatives(topsis_ranking))

        elif method == "AHP":
            ahp_weights = ahp_method(np.array(ahp_matrix))
            final_scores, weighted_scores = ahp_alternatives(alternative_matrices, ahp_weights)
            
            st.write("Bobot Kriteria (AHP):", ahp_weights)
            for i, score in enumerate(weighted_scores):
                st.write(f"Bobot Alternatif {i + 1}: {score}")
                
            ahp_ranking = rank_alternatives(weighted_scores)
            st.write("Ranking Alternatif (AHP):", label_alternatives(ahp_ranking))

if __name__ == "__main__":
    main()
