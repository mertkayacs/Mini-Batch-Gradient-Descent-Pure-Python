# Mini-Batch-Gradient-Descent-Pure-Python

Mini-batch gradient descent for polynomial regression, written in Python. It reads a dataset and a model definition from text files, fits the model weights, and prints the final test MSE.

## How to run

Requires Python 3 and pandas (`pip install pandas`). There is no requirements file.

Generate a sample dataset and model file:

```
python dataset_generator.py 1000
```

This writes `data.csv` and `model.txt` into the current directory. Then train:

```
python minibatchgradientdescent.py data.csv model.txt
```

The model file is a polynomial in the CSV feature columns, one line, terms joined with `+`:

```
a_0*F1^2 + a_1*F2 + a_2
```

Terms without a feature are constants. Not every feature has to appear in the model.

The dataset is a CSV whose last column is the target. Data is shuffled and split 80/20 into train and test. Default settings in the code: learning rate 0.1, batch size 32, 10000 iterations, mean squared error as the loss. Final test MSE is printed at the end.

## Tech used

- Python 3
- pandas

## Status

Coursework, 2020.

## License

MIT License. See [LICENSE](LICENSE).
