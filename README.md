### Regression Metrics (Diabetes dataset)

| Model          | RMSE (Train) | RMSE (Test) | R² (Train) | R² (Test) |
|----------------|--------------|-------------|------------|-----------|
| Linear         | 53.559       | 53.853      | 0.528      | 0.453     |
| Ridge          | 53.928       | 53.465      | 0.521      | 0.460     |
| Lasso          | 54.119       | 52.904      | 0.518      | 0.472     |

# Висновки:
- Різниця між RMSE_train і RMSE_test невелика, що свідчить про добру узагальнюваність.
- Lasso має найнижчий RMSE_test і найвищий R²_test, тобто краще справляється з тестовими даними.
- Linear має найвищий R²_train, але гірший R²_test — можливо, трохи перенавчився на тренувальних даних.
- Загалом Lasso найкраще узагальнює — можливо, завдяки відбору ознак.



### Classification Metrics (Breast Cancer dataset)

| Model          | Accuracy (Train) | Accuracy (Test)  | ROC AUC (Train)  | ROC AUC (Test)  |
|----------------|------------------|------------------|------------------|-----------------|
| Logistic       | 1.000            | 0.965            | 1.000            | 0.976           |
| Logistic L2    | 0.985            | 1.000            | 0.996            | 1.000           |
| Logistic L1    | 0.987            | 0.991            | 0.998            | 1.000           |

# Висновки:
- Є ознаки перенавчання, краще було б провести стандартизацію даних після розподілу на тренувальні і тестові дані.
- Logistic (без регуляризації) має ідеальні метрики на тренуванні (1.000), але помітно гірші на тесті — це класичний симптом перенавчання.
- L2 і L1 регуляризовані моделі мають трохи нижчі метрики на тренуванні, але вищі на тесті — це ознака кращої узагальнюваності.
- L2 виглядає найстабільнішою: висока точність і AUC на обох наборах.


# Linear algebra introduction practice

The repository contains practical homework from the course "Incorrect Data Processing Problems".

Please, use NumPy, SciPy, scikit-learn and similar libs to implement the tasks.

## Practices

1. Vectors;
2. Matrices;
3. Linear and affine mappings;
4. Matrix decompositions;
5. Regularization.


## Useful links

* [Introduction to Linear Algebra for Applied Machine Learning with Python](https://pabloinsente.github.io/intro-linear-algebra)
* [Regularization 1](https://github.com/ethen8181/machine-learning/blob/master/regularization/regularization.ipynb)
* [Regularization 2](https://nbviewer.org/github/justmarkham/DAT8/blob/master/notebooks/20_regularization.ipynb)
