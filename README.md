# Machine Learning Project

## Modules

**data_loading**: Load test data and training data

**data_plotting**: It contains function to plot data in various way

**models**: It contains all classes of models

## Notes

### Target

It is better to target the 0.1 prior application, because we know there are few pulsar

### Correlation

- Using Pearson Correlation Coefficient we can see that before preprocessing there is correlation between data
- Gaussianization reduces a bit correlation

### Gaussian Model

| Method \ prior | 0.1   | 0.5   | 0.9   |
| -------------- | ----- | ----- | ----- |
| No prep        | 0.266 | 0.139 | 0.629 |
| Prep           | 0.330 | 0.139 | 0.589 |
| Gaussianized   | 0.234 | 0.130 | 0.552 |

*Naive* 

| Method \ prior | 0.1   | 0.5   | 0.9   |
| -------------- | ----- | ----- | ----- |
| No prep        | 0.304 | 0.182 | 0.667 |
| Prep           | 0.243 | 0.133 | 0.589 |
| Gaussianized   | 0.213 | 0.113 | 0.541 |

*Tied covariance*

| Method \ prior | 0.1   | 0.5   | 0.9   |
| -------------- | ----- | ----- | ----- |
| No prep        | 0.213 | 0.106 | 0.501 |
| Prep           | 0.213 | 0.122 | 0.547 |
| Gaussianized   | 0.213 | 0.106 | 0.500 |

### Logistic Regression

- Analizing the risk the natural choice seems $\lambda \to 0$

$\lambda = 1e-5$, rebalanced 0.5

| Method \ prior | 0.1   | 0.5   | 0.9   |
| -------------- | ----- | ----- | ----- |
| No prep        | 0.579 | 0.191 | 0.700 |
| Prep           | 0.207 | 0.124 | 0.554 |
| Gaussianized   | 0.440 | 0.173 | 0.464 |

$\lambda = 1e-5$, rebalanced 0.1

| Method \ prior | 0.1   | 0.5   | 0.9   |
| -------------- | ----- | ----- | ----- |
| No prep        |       |       |       |
| Prep           | 0.198 | 0.107 | 0.542 |
| Gaussianized   | 0.194 | 0.104 | 0.503 |

#### Quadratic Case

- Natural choice seems $\lambda \to 0$

$\lambda = 1e-5$, rebalanced 0.1

| Method \ prior | 0.1   | 0.5   | 0.9   |
| -------------- | ----- | ----- | ----- |
| No prep        |       |       |       |
| Prep           | 0.195 | 0.108 | 0.481 |
| Gaussianized   | 0.222 | 0.119 | 0.545 |

### Support Vector Machine

#### Linear Case

- Analizing the risk the natural choice seems $C \to 0$ for preprocessed and gaussianized data

$C = 1e-3$, rebalanced 0.5

| Method \ prior | 0.1   | 0.5   | 0.9   |
| -------------- | ----- | ----- | ----- |
| No prep        | 0.607 | 0.221 | 0.774 |
| Prep           | 0.226 | 0.110 | 0.517 |
| Gaussianized   | 0.202 | 0.111 | 0.531 |

$C = 1e-3$, rebalanced 0.1

| Method \ prior | 0.1   | 0.5   | 0.9   |
| -------------- | ----- | ----- | ----- |
| No prep        | 0.288 | 0.160 | 0.741 |
| Prep           | 0.244 | 0.149 | 0.658 |
| Gaussianized   | 0.223 | 0.148 | 0.706 |

$C = 1e-3$, no rebalanced

| Method \ prior | 0.1   | 0.5   | 0.9   |
| -------------- | ----- | ----- | ----- |
| No prep        |       |       |       |
| Prep           | 0.226 | 0.167 | 0.702 |
| Gaussianized   | 0.227 | 0.150 | 0.686 |

#### Quadratic Case

- Analizing the risk the natural choice seems $C \to 0$ for preprocessed and gaussianized data

$C = 1e-3$, rebalanced 0.5

| Method \ prior | 0.1   | 0.5   | 0.9   |
| -------------- | ----- | ----- | ----- |
| No prep        |       |       |       |
| Prep           | 0.203 | 0.110 | 0.489 |
| Gaussianized   | 0.221 | 0.110 | 0.499 |

$C = 1e-3$, rebalanced 0.1

| Method \ prior | 0.1   | 0.5   | 0.9   |
| -------------- | ----- | ----- | ----- |
| No prep        | 0.297 | 0.160 | 0.781 |
| Prep           | 0.194 | 0.107 | 0.518 |
| Gaussianized   | 0.206 | 0.118 | 0.535 |

#### Cubic Case

- Analizing the risk the natural choice seems $C \to 0$ 
- From the table we don’t see improvements from quadratic model

$C = 1e-3$, rebalanced 0.1

| Method \ prior | 0.1   | 0.5   | 0.9   |
| -------------- | ----- | ----- | ----- |
| No prep        |       |       |       |
| Prep           | 0.198 | 0.109 | 0.622 |
| Gaussianized   |       |       |       |

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

- With prior for rebalancing 0.1

$C=1e-3, \gamma = 10$

| Method \ prior | 0.1   | 0.5   | 0.9   |
| -------------- | ----- | ----- | ----- |
| No prep        |       |       |       |
| Prep           | 0.199 | 0.119 | 0.609 |
| Gaussianized   | 0.912 | 0.391 | 0.898 |

$C = 1e-3, \gamma = 0.1$

| Method \ prior | 0.1   | 0.5   | 0.9   |
| -------------- | ----- | ----- | ----- |
| No prep        |       |       |       |
| Prep           | 0.208 | 0.122 | 0.580 |
| Gaussianized   | 0.204 | 0.123 | 0.577 |

$C= 1e-3, \gamma = 100$

| Method \ prior | 0.1   | 0.5   | 0.9   |
| -------------- | ----- | ----- | ----- |
| No prep        |       |       |       |
| Prep           | 0.308 | 0.151 | 0.817 |
| Gaussianized   |       |       |       |

### Gaussian Mixture Model

- Gaussianization doesn’t achieve good results in general
- Preprocessing is useful only for prior 0.5 e 0.9
- Gaussian Mixture Model is not benefiting from increasing the number of gaussians per class

number of gaussian = 2

| Method \ prior | 0.1   | 0.5   | 0.9   |
| -------------- | ----- | ----- | ----- |
| No prep        | 0.248 | 0.135 | 0.647 |
| Prep           | 0.315 | 0.125 | 0.568 |
| Gaussianized   | 0.840 | 0.277 | 0.680 |

number of gaussian = 4

| Method \ prior | 0.1   | 0.5   | 0.9   |
| -------------- | ----- | ----- | ----- |
| No prep        | 0.248 | 0.135 | 0.647 |
| Prep           | 0.315 | 0.125 | 0.568 |
| Gaussianized   | 0.840 | 0.277 | 0.680 |

number of gaussian = 8

| Method \ prior | 0.1   | 0.5   | 0.9   |
| -------------- | ----- | ----- | ----- |
| No prep        | 0.248 | 0.135 | 0.647 |
| Prep           | 0.315 | 0.125 | 0.568 |
| Gaussianized   | 0.840 | 0.277 | 0.680 |

number of gaussian = 32

| Method \ prior | 0.1   | 0.5   | 0.9   |
| -------------- | ----- | ----- | ----- |
| No prep        | 0.249 | 0.130 | 0.635 |
| Prep           | 0.325 | 0.127 | 0.552 |
| Gaussianized   |       |       |       |

number of gaussian = 64

| Method \ prior | 0.1   | 0.5   | 0.9   |
| -------------- | ----- | ----- | ----- |
| No prep        | 0.253 | 0.133 | 0.618 |
| Prep           | 0.318 | 0.125 | 0.555 |
| Gaussianized   | 0.741 | 0.315 | 0.741 |

## Roc Analysis

Plotting the different ROCS of Logistic Regression, Linear SVM, Quadratic SVM -> Linear SVM is the most stable, Quadratic can sometimes achieve better results, also Logistic Regression can work well a little less than linear svm

RBF SVM works better with gaussianized and can be good but seems there is some miscalibration

## Miscalibration Analysis

From DCF analysis:

- Log Regr is better calibrated for different type of prior and has a wider range of values that can achieve good results
- Linear SVM (k=1.0, C=1e-3, prior=0.5, grade=1.0, c=1) has less miscalibration than Log Regr with prior 0.5
- RBF SVM is really miscalibrated

From score analysis:

- Log Regr: we get a threshold of -0.4 / -0.25 (low acc, medium acc)
- RBF SVM: we get -1.08

Prior 0.1

| Model               | min DCF | DCF with theoretical threshold | DCF with estimated threshold | threshold           |
| ------------------- | ------- | ------------------------------ | ---------------------------- | ------------------- |
| Quadratic SVM Prep  | 0.205   | 1.0                            | 0.237                        | -0.8493689904960244 |
| Linear Log Reg Prep | 0.203   | 0.295                          | 0.211                        | -0.2023909147748068 |

Logistic regression is better calibrated so it is better to choose this model.

## Extra Step

This data are collected improving precision of the model to 1e7 instead of 1e12

$\lambda = 1e-5$, rebalanced 0.1

| Method \ prior | 0.1  | 0.5  | 0.9  |
| -------------- | ---- | ---- | ---- |
| No prep        |      |      |      |
| Prep           |      |      |      |
| Gaussianized   |      |      |      |

 

