# Results

Numbers behind the figures. Logs are in `logs/` and are not tracked.

## Final detector results: `step18_same_as_original/`

Produced by `src/ap_step18_same_as_original.py` on the setup of the paper: ten pooled datasets, random 75/25 read level split, SGD, early stopping, at most 1,000 epochs, seed 0.

```
size{fragment}_ret{retention}_line_pool10_es100m500[_wide]_ep1000_seed0.txt   confusion matrix rates and test accuracy
size{fragment}_ret{retention}_line_pool10_es100m500[_wide]_ep1000_seed0.npz   test labels and scores, for ROC and PR curves
```

- No `_wide` in the name: Reimplemented CNN (the two layer model of the paper).
- `_wide`: our Wide CNN.
- `es100m500`: early stopping with patience 100, not before epoch 500.

The `.txt` file has two lines: `TN, FN, FP, TP, accuracy`, then the values. The accuracy is in percent. The `TN`, `FN`, `FP` and `TP` columns are the normalized confusion matrix cells in the order the authors' script writes them.

Earlier, exploratory runs on one dataset (`*_ds0_ep3000_seed0.txt`, split `line` or `host`) are also here. They are not used in the main results.

## Runs of the authors' own Keras script

- `step9_authors_cnn/`: the authors' CNN as released (two conv layers of 32 filters, kernel 5), one dataset, read level split.
- `step9_authors_cnn_fig7b/`: the same script with the architecture changed to the paper's two layer model (16 filters, kernel 8, max pool 4).

## Other files

| File | Content |
|---|---|
| `fig5b.json`, `fig6.json` | Alignment score reductions (MaxNW, Random, MinNW) |
| `step5_kmer.json`, `step5_retention.json` | k-mer baseline |
| `step8_report_numbers.json` | Earlier host level detector numbers (superseded) |
| `step10_consensus_tradeoff.json` | Consensus decoding trade off simulation |
| `step14_full_dataset_boxplot.json` | Amplification simulation: seven platforms, 1 to 9 copies |
| `generation_log.txt` | Log of the dataset generation |
