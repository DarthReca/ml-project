# Machine Learning Project

## Modules

**data_loading**: Load test data and training data

**data_plotting**: It contains function to plot data in various way

**models**: It contains all classes of models

## Notes

### Correlation

- Using Pearson Correlation Coefficient we can see that before preprocessing there is correlation between data
- Gaussianization reduces a bit correlation

### Cross Validation

- Overfitting or Underfitting can be good ([Shiring](https://shiring.github.io/machine_learning/2017/04/02/unbalanced))

### Gaussian Model

- Min DCF show that there is miscalibration
    - Without preprocessing the miscalibration is higher
    - miscalibration is higher with naive and tied cov assumptions 
    - Gaussianization is not improving min-dcf

- After preprocessing can be obtained better results
- Data of pulsar and non-pulsar are not well separated
- Linear models will not achieve good result because they are not linearly separable

### Linear Regression

### Support Vector Machine
