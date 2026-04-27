Put the jury CSV file in this folder during the presentation.

In `ads.ipynb`, open the "Jury CSV Batch Inference" cell and change only:

```python
JURY_CSV_FILENAME = "jury_test.csv"
```

The notebook writes predictions to `artifacts/jury_predictions_<file_name>.csv`.

If the jury gives an answer key for local evaluation, put it here as:

```csv
test_id,text,label
```

Use the file name `JURI_CEVAP_ANAHTARI.csv` so the final notebook evaluator cell can find it automatically.
