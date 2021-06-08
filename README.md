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

#### Linear Case

- Analizing the risk the natural choice seems $C \to 0$ for preprocessed and gaussianized data

$C = 1e-3$, rebalanced

| Method \ prior | 0.1   | 0.5   | 0.9   |
| -------------- | ----- | ----- | ----- |
| No prep        | 0.607 | 0.221 | 0.774 |
| Prep           | 0.226 | 0.110 | 0.517 |
| Gaussianized   | 0.202 | 0.111 | 0.531 |

$C = 1e-3$, no rebalanced

| Method \ prior | 0.1   | 0.5   | 0.9   |
| -------------- | ----- | ----- | ----- |
| No prep        |       |       |       |
| Prep           | 0.226 | 0.167 | 0.702 |
| Gaussianized   | 0.227 | 0.150 | 0.686 |

#### Quadratic Case

- Analizing the risk the natural choice seems $C \to 0$ for preprocessed and gaussianized data

$C = 1e-3$, rebalanced

| Method \ prior | 0.1   | 0.5   | 0.9   |
| -------------- | ----- | ----- | ----- |
| No prep        |       |       |       |
| Prep           | 0.203 | 0.110 | 0.489 |
| Gaussianized   | 0.221 | 0.110 | 0.499 |

#### RBF

- Analizing the risk the natural choice seems $C \to 0$ and $\gamma = 0.1$ for Gaussianized and $\gamma = 10$ for preprocessed

$C=1e-3, \gamma = 10$

| Method \ prior | 0.1   | 0.5   | 0.9   |
| -------------- | ----- | ----- | ----- |
| No prep        |       |       |       |
| Prep           | 0.200 | 0.107 | 0.581 |
| Gaussianized   | 0.217 | 0.115 | 0.644 |

$C = 1e-3, \gamma = 0.1$

| Method \ prior | 0.1   | 0.5   | 0.9   |
| -------------- | ----- | ----- | ----- |
| No prep        |       |       |       |
| Prep           | 0.290 | 0.150 | 0.537 |
| Gaussianized   | 0.214 | 0.113 | 0.647 |

