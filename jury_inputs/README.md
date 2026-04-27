Put the jury CSV file in this folder during the presentation.

In `ads.ipynb`, open the "Jury CSV Batch Inference" cell and change only:

```python
JURY_CSV_FILENAME = "jury_test.csv"
```

The notebook writes predictions to `artifacts/jury_predictions_<file_name>.csv`.
