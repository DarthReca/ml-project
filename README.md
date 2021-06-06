# Machine Learning Project

## Modules

**data_loading**: Load test data and training data

**data_plotting**: It contains function to plot data in various way

**models**: It contains all classes of models

## Notes

### Correlation

- Using Pearson Correlation Coefficient we can see that before preprocessing there is correlation between data
- Gaussianization reduces a bit correlation

### Gaussian Model

| Method \ prior | 0.1  | 0.5   | 0.9   |
| -------------- | ---- | ----- | ----- |
| No prep        | 1.0  | 1.0   | 1.055 |
| Prep           | 1.0  | 0.613 | 0.771 |
| Gaussianized   | 1.0  | 1.0   | 1.055 |

*Naive*

| Method \ prior | 0.1  | 0.5  | 0.9   |
| -------------- | ---- | ---- | ----- |
| No prep        | 1.0  | 1.0  | 1.055 |
| Prep           | 1.0  | 1.0  | 1.055 |
| Gaussianized   | 1.0  | 1.0  | 1.055 |

*Tied covariance*

| Method \ prior | 0.1  | 0.5   | 0.9   |
| -------------- | ---- | ----- | ----- |
| No prep        | 1.0  | 1.0   | 1.055 |
| Prep           | 1.0  | 0.997 | 1.008 |
| Gaussianized   | 1.0  | 0.999 | 1.055 |

- Data of pulsar and non-pulsar are not well separated
- Linear models will not achieve good result because they are not linearly separable

### Linear Regression

- Analizing the risk the natural choice seems $\lambda \to 0$

*$\lambda = 1e-5$*

| Method \ prior | 0.1   | 0.5   | 0.9   |
| -------------- | ----- | ----- | ----- |
| No prep        | 0.579 | 0.191 | 0.700 |
| Prep           | 0.274 | 0.124 | 0.554 |
| Gaussianized   | 0.440 | 0.173 | 0.464 |

- Gaussianization is bad, Preprocessing is good
- Linear Regression is better than MVG

### Support Vector Machine
