# COA ile WMN Router Placement

Bu proje ders kapsamında yapılmıştır. Coyote Optimization Algorithm (COA) kullanarak Wireless Mesh Network'lerde router yerleşimi optimizasyonu gerçekleştirilmiştir.

## Proje Yapısı

```
BBL512E_COA_WMN/
├── scripts/              # Çalıştırma scriptleri
├── src/wmn/              # Ana kod modülleri
├── results/              # Sonuçlar (tablolar, grafikler)
│   ├── figures/         # Grafikler
│   ├── tables/          # Tablolar (CSV, PNG, LaTeX)
│   └── logs/            # Log dosyaları
└── requirements.txt      # Gereksinimler
```

## Kurulum

```bash
pip install -r requirements.txt
```

## Kullanım

### Temel Placement
```bash
python scripts/run_placement.py
```

### Convergence Analizi
```bash
python scripts/run_convergence.py
python generate_convergence_figures.py
python make_table11.py
```

### Parametre Sweep Deneyleri
```bash
python scripts/run_clients_sweep_parallel.py    # Client sayısı sweep
python scripts/run_routers_sweep_parallel.py    # Router sayısı sweep
python scripts/run_radius_sweep_parallel.py     # Coverage radius sweep
```

## Sonuçlar

Sonuçlar `results/` klasöründe:
- **figures/**: Convergence grafikleri ve placement görselleri
- **tables/**: Deney sonuçları (CSV formatında)

### Örnek Sonuçlar

![Placement](results/tables/fig_placement_coa.png)

![Convergence](results/figures/Fig17_instance1_COA_reproduced.png)

## Notlar

- Tüm deneyler parallel processing ile hızlandırılmıştır
- Sonuçlar paper'daki referans değerlerle karşılaştırılmıştır
- Detaylı loglar `results/logs/` klasöründe bulunmaktadır
