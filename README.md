### Regression Metrics (Diabetes dataset)

| Model          | RMSE (Train) | RMSE (Test) | R² (Train) | R² (Test) |
|----------------|--------------|-------------|------------|-----------|
| Linear         | 52.603       | 57.419      | 0.503      | 0.520     |
| Ridge          | 52.960       | 57.793      | 0.496      | 0.514     |
| Lasso          | 52.603       | 57.419      | 0.503      | 0.520     |




### Classification Metrics (Breast Cancer dataset)

| Model          | Accuracy (Train) | Accuracy (Test)  | ROC AUC (Train)  | ROC AUC (Test)  |
|----------------|------------------|------------------|------------------|-----------------|
| Logistic       | 1.000            | 0.965            | 1.000            | 0.976           |
| Logistic L2    | 0.985            | 1.000            | 0.996            | 1.000           |
| Logistic L1    | 0.987            | 0.991            | 0.998            | 1.000           |



💡 Поради для README
- Додай коротке пояснення перед таблицями:
- "Нижче наведено порівняння моделей регресії та класифікації за основними метриками. Тестові значення показують узагальнюваність моделей на нових даних."
- Якщо хочеш, можеш додати висновки після таблиць у вигляді списку:
**Висновки:**
- Logistic L2 показує найкращу узагальнюваність.
- Ridge Regression трохи краще узагальнює, ніж Linear.
- Lasso і Linear мають однакові результати — можливо, Lasso не занулює ознаки.
Хочеш, я допоможу сформувати повний шаблон README з заголовками, описом задачі, кодом запуску та результатами?
| Model               | Accuracy (Train) | Accuracy (Test) | ROC AUC (Train) | ROC AUC (Test) |
|---------------------|------------------|------------------|------------------|-----------------|
| Logistic (no penalty) | 1.000            | 0.965            | 1.000            | 0.976           |
| Logistic L2          | 0.985            | 1.000            | 0.996            | 1.000           |
| Logistic L1          | 0.987            | 0.991            | 0.998            | 1.000           |







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
