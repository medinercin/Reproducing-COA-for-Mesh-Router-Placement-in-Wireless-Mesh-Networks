# Table 11 olusturmak icin script
# COA referans degerleri ile karsilastirma yapiyor

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# sutun ismini bulmak icin fonksiyon
def find_column(df, possible_names, case_insensitive=True):
    df_columns = df.columns.tolist()
    
    if case_insensitive:
        df_columns_lower = [c.lower() for c in df_columns]
        for name in possible_names:
            name_lower = name.lower()
            if name_lower in df_columns_lower:
                idx = df_columns_lower.index(name_lower)
                return df_columns[idx]
    else:
        for name in possible_names:
            if name in df_columns:
                return name
    
    return None


def process_convergence_file(csv_path):
    # csv dosyasini oku
    df = pd.read_csv(csv_path)
    
    # fitness veya cost sutununu bul
    fitness_col = find_column(df, ["fitness", "best_fitness", "objective", "best_objective", "obj", "best", "cost"])
    
    if fitness_col is None:
        raise ValueError(f"Sutun bulunamadi: {csv_path}. Mevcut sutunlar: {df.columns.tolist()}")
    
    # iteration sutununu bul
    iter_col = find_column(df, ["iteration", "iter", "t"])
    
    # degerleri al
    values = df[fitness_col].values
    
    # cost ise fitness'e cevir
    if fitness_col.lower() == "cost":
        fitness_values = 1.0 - values
        best_idx = np.argmin(values)
    else:
        if "fitness" in fitness_col.lower():
            fitness_values = values
            best_idx = np.argmax(values)
        else:
            fitness_values = 1.0 - values
            best_idx = np.argmin(values)
    
    best_fitness = fitness_values[best_idx]
    
    # iteration degerini al (1'den baslamali)
    if iter_col is not None:
        iterations = df[iter_col].values
        best_iteration = iterations[best_idx]
        # 0'dan basliyorsa 1'e cevir
        if best_iteration == 0 and best_idx == 0:
            if iterations[0] == 0:
                best_iteration = best_idx + 1
            else:
                best_iteration = int(best_iteration)
        else:
            best_iteration = int(best_iteration)
    else:
        best_iteration = best_idx + 1
    
    return best_fitness, best_iteration


def create_table11(input_dir="results/tables", output_dir="results/tables"):
    # paper'daki referans degerler
    paper_values = {
        1: {"fitness": 0.85, "iteration": 308.67},
        2: {"fitness": 0.77, "iteration": 721.60},
        3: {"fitness": 0.68, "iteration": 817.26},
        4: {"fitness": 0.61, "iteration": 911.46},
    }
    
    rows = []
    # her instance icin isle
    for instance_num in range(1, 5):
        csv_path = os.path.join(input_dir, f"convergence_instance{instance_num}.csv")
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Dosya bulunamadi: {csv_path}")
        
        # kendi degerlerimizi hesapla
        reproduced_fitness, reproduced_iteration = process_convergence_file(csv_path)
        
        # paper degerlerini al
        paper_fitness = paper_values[instance_num]["fitness"]
        paper_iteration = paper_values[instance_num]["iteration"]
        
        rows.append({
            "Instance": f"Instance{instance_num}",
            "COA (Paper) Fitness": paper_fitness,
            "COA (Paper) Iteration": paper_iteration,
            "COA (Reproduced) Fitness": reproduced_fitness,
            "COA (Reproduced) Iteration": reproduced_iteration,
        })
    
    # dataframe olustur
    df = pd.DataFrame(rows)
    
    # formatla
    df_display = df.copy()
    df_display["COA (Paper) Fitness"] = df_display["COA (Paper) Fitness"].apply(lambda x: f"{x:.2f}")
    df_display["COA (Paper) Iteration"] = df_display["COA (Paper) Iteration"].apply(lambda x: f"{x:.2f}")
    df_display["COA (Reproduced) Fitness"] = df_display["COA (Reproduced) Fitness"].apply(lambda x: f"{x:.2f}")
    # iteration formatla
    df_display["COA (Reproduced) Iteration"] = df_display["COA (Reproduced) Iteration"].apply(
        lambda x: f"{int(x)}" if x == int(x) else f"{x:.2f}"
    )
    
    return df, df_display


def save_csv(df, output_path):
    # klasor yoksa olustur
    dir_path = os.path.dirname(output_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"CSV kaydedildi: {output_path}")


def save_latex(df, output_path):
    dir_path = os.path.dirname(output_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    
    # latex tablosu olustur
    latex_lines = []
    latex_lines.append("\\begin{tabular}{l|cc|cc}")
    latex_lines.append("\\hline")
    
    # baslik satiri
    headers = [
        "\\textbf{Instance}",
        "\\textbf{COA (Paper) Fitness}",
        "\\textbf{COA (Paper) Iteration}",
        "\\textbf{COA (Reproduced) Fitness}",
        "\\textbf{COA (Reproduced) Iteration}",
    ]
    latex_lines.append(" & ".join(headers) + " \\\\")
    latex_lines.append("\\hline")
    
    # veri satirlari
    for _, row in df.iterrows():
        # degerleri formatla
        iter_val = row['COA (Reproduced) Iteration']
        if iter_val == int(iter_val):
            iter_str = f"{int(iter_val)}"
        else:
            iter_str = f"{iter_val:.2f}"
        
        values = [
            str(row["Instance"]),
            f"{row['COA (Paper) Fitness']:.2f}",
            f"{row['COA (Paper) Iteration']:.2f}",
            f"{row['COA (Reproduced) Fitness']:.2f}",
            iter_str,
        ]
        latex_lines.append(" & ".join(values) + " \\\\")
    
    latex_lines.append("\\hline")
    latex_lines.append("\\end{tabular}")
    
    latex_str = "\n".join(latex_lines)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(latex_str)
    
    print(f"LaTeX kaydedildi: {output_path}")


def save_png(df, output_path, title="Table 11: Convergence Analysis (COA)"):
    # figure olustur
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.axis("tight")
    ax.axis("off")
    
    # tablo verilerini hazirla
    table_data = []
    for _, row in df.iterrows():
        repro_iter = row['COA (Reproduced) Iteration']
        if repro_iter == int(repro_iter):
            repro_iter_str = f"{int(repro_iter)}"
        else:
            repro_iter_str = f"{repro_iter:.2f}"
        
        table_data.append([
            str(row["Instance"]),
            f"{row['COA (Paper) Fitness']:.2f}",
            f"{row['COA (Paper) Iteration']:.2f}",
            f"{row['COA (Reproduced) Fitness']:.2f}",
            repro_iter_str,
        ])
    
    # sutun basliklari
    col_labels = [
        "Instance",
        "COA (Paper)\nFitness",
        "COA (Paper)\nIteration",
        "COA (Reproduced)\nFitness",
        "COA (Reproduced)\nIteration",
    ]
    
    # tablo olustur
    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    
    # tablo stilleri
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.2)
    
    # baslik stilini ayarla
    for i in range(len(col_labels)):
        cell = table[(0, i)]
        cell.set_facecolor("#D0D0D0")
        cell.set_text_props(weight="bold", fontsize=11)
        cell.set_edgecolor("black")
        cell.set_linewidth(2.0)
    
    # veri hucrelerini stillendir
    for i in range(1, len(table_data) + 1):
        for j in range(len(col_labels)):
            cell = table[(i, j)]
            cell.set_edgecolor("black")
            cell.set_linewidth(1.5)
            cell.set_text_props(fontsize=11)
            if j == 0:
                cell.set_facecolor("#F8F8F8")
            else:
                cell.set_facecolor("white")
    
    # dikey ayirici cizgiler
    for i in range(len(table_data) + 1):
        cell = table[(i, 1)]
        cell.set_linewidth(2.0)
    for i in range(len(table_data) + 1):
        cell = table[(i, 3)]
        cell.set_linewidth(2.0)
    
    # baslik ekle
    plt.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
    
    # kaydet
    dir_path = os.path.dirname(output_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close()
    
    print(f"PNG kaydedildi: {output_path}")


def main():
    # dosya yollari
    INPUT_DIR = "results/tables"
    OUTPUT_DIR = "results/tables"
    
    print("=" * 70)
    print("Table 11 olusturuluyor...")
    print("=" * 70)
    
    # tabloyu olustur
    print("\nCSV dosyalari isleniyor...")
    df, df_display = create_table11(input_dir=INPUT_DIR, output_dir=OUTPUT_DIR)
    
    # tabloyu goster
    print("\n" + "=" * 70)
    print("Table 11: Convergence Analysis")
    print("=" * 70)
    print(df_display.to_string(index=False))
    print("=" * 70)
    
    # dosyalari kaydet
    print("\nDosyalar kaydediliyor...")
    
    # CSV
    csv_path = os.path.join(OUTPUT_DIR, "table11_coa_reference_vs_reproduced.csv")
    save_csv(df_display, csv_path)
    
    # ham veri CSV
    csv_raw_path = os.path.join(OUTPUT_DIR, "table11_coa_reference_vs_reproduced_raw.csv")
    save_csv(df, csv_raw_path)
    
    # LaTeX
    tex_path = os.path.join(OUTPUT_DIR, "table11_coa_reference_vs_reproduced.tex")
    save_latex(df, tex_path)
    
    # PNG
    png_path = os.path.join(OUTPUT_DIR, "table11_coa_reference_vs_reproduced.png")
    save_png(df, png_path)
    
    print("\n" + "=" * 70)
    print("Table 11 hazir!")
    print("=" * 70)
    print(f"\nDosyalar kaydedildi: {OUTPUT_DIR}")
    print(f"  - {os.path.basename(csv_path)}")
    print(f"  - {os.path.basename(tex_path)}")
    print(f"  - {os.path.basename(png_path)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
