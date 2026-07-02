# geodesic-anomaly-test

Reference code for the article **"Information-Geometric Hypothesis Testing on the
Categorical Statistical Manifold: A Geodesic Test Statistic for Anomaly Detection
in Decision Support Systems"** (Emre Öztürk).

It reproduces every figure, table, and reported statistic: the geodesic
(Fisher–Rao) test statistic on the categorical manifold, its asymptotic and
finite-sample theory (curvature/Bartlett correction and the non-correctability
result), the composite/submanifold tests, the Gaussian (hyperbolic) extension,
and the NSL-KDD real-data application.

## Requirements
- Python 3.10+
- `numpy`, `scipy`, `matplotlib`

```bash
pip install numpy scipy matplotlib
```

## Reproducing the results

| Script | Produces |
|--------|----------|
| `simulation.py` | point-null calibration, ROC, power, and the manifold geometry figure (`fig_null_calibration`, `fig_roc`, `fig_power`, `fig_geometry`) |
| `curvature_sim.py` | second-order bias law + Bartlett calibration (`fig_curvature`) |
| `composite_sim.py` | independence-submanifold test (`fig_composite`) |
| `compbart_sim.py` | composite Bartlett correction (`fig_composite_bartlett`) |
| `gauss_sim.py` | Gaussian/hyperbolic χ²₂ validation (`fig_gaussian`) |
| `noncorr.py` | certifies v ≠ 4b (non-correctability numbers) |
| `noncorr_fig.py` | non-correctability figure (`fig_noncorr`) |
| `nslkdd_experiment.py` | NSL-KDD real-data application (numbers; saves `data/nsl_results.npz`) |
| `fig_nslkdd.py` | NSL-KDD figure (`fig_nslkdd`) |

```bash
python simulation.py
python nslkdd_experiment.py
python fig_nslkdd.py
# ...etc
```

Figures are written to `figs/` (created automatically) as both PDF and PNG.
All scripts use a fixed random seed; results are deterministic.

## Data
The NSL-KDD 20% subset is the public benchmark of Tavallaee et al. (2009),
DOI `10.1109/CISDA.2009.5356528`. Save the `KDDTrain+_20Percent.txt` benchmark
as `data/nslkdd20.txt` (the filename `nslkdd_experiment.py` and `fig_nslkdd.py`
read).

## Citation
If you use this code, please cite the article and this archive (see
`CITATION.cff`). A DOI is minted automatically by Zenodo on each GitHub release.

## License
MIT — see `LICENSE`.
