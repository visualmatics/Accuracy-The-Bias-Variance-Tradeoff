# Accuracy-The-Bias-Variance-Tradeoff
In the article “Which Machine Learning (ML) to choose? [1]”, which helps you to choose the right ML for your data, we indicated that “From a business perspective, two of the most significant measurements are accuracy and interpretability.”

We also claimed that “Evaluating the accuracy of a machine learning model is critical in selecting and deploying a machine learning model.”

-       But, what factors affect model accuracy?

Accuracy is affected by your model fitting. And, model fitting depends on Bias-Variance Tradeoff in machine learning. Balancing bias and variance can solve overfitting and underfitting.



Bullseye Diagram: The distribution of model predictions. Image adapted: Domingo 2012[2]
Bullseye Diagram: The distribution of model predictions
Image adapted: Domingo 2012 [2]

-       Definitions:

"Model fitting is a measure of how well [optimize] a machine learning model generalizes to similar [evaluation] data to that on which it was trained. A model that is well-fitted [optimal-fitted] produces more accurate outcomes. A model that is overfitted matches the data too closely. A model that is under-fitted does not match closely enough [3]."

"In machine learning, overfitting occurs when a learning model customizes itself too much to describe the relationship between training data and the labels. Overfitting tends to make the model very complex by having too many parameters. By doing this, it loses its generalization power, which leads to poor performance on new [evaluation] data [4]."

"Your model is underfitting the training data when the model performs poorly on the training data. This is because the model is unable to capture the relationship between the input examples (often called X) and the target values (often called Y) [5].”

-       Root causes: 

Model fit depends on solving the issue and balancing the tradeoff between bias and variance.

"Understanding model fit is important for understanding the root cause for poor model accuracy. This understanding will guide you to take corrective steps. We can determine whether a predictive model is underfitting or overfitting the training data by looking at the prediction error on the training data and the evaluation data [6]."

Bias is the difference between the estimated value and the true value of the parameter being evaluated. High bias results in underfitting leading to an inaccurate [not valid] model. It can be caused by training on a small data set, building a simple model to capture complex patterns, or not taking into account all the features given for training which causes learning incorrect relations. Generally, high-bias models learn faster and are easy to understand, but they are less flexible [7].

Variance is the degree of spread in a data set which indicates how far a set of data points are spread out from their mean [average] value. The variance of an estimated function indicates how much the function is capable of adjusting to the change in a data set. High variance results in overfitting leading to inconsistent [not reliable] model. It can be caused by having too many features, building a more complex model than necessary, or capturing a high noise level. Generally, high variance models tune themselves and are more robust to a changing data set, but they are more complex and overly flexible.

“A major difference between machine learning and statistics is their purpose. Machine learning models are designed to make the most accurate predictions possible. Statistical models are designed for inference about the relationships between variables.”

Statistical bias is a systematic tendency that causes differences between results and facts. Statistical bias may be introduced at all stages of data analysis: data selection, hypothesis testing, estimator selection, analysis methods, and interpretation.

tatistical bias sources from stages of data analysis. Image: Visual Science Informatics, LLC
Statistical bias sources from stages of data analysis. Image: Visual Science Informatics, LLC

Systematic error (bias) introduces noisy data with high bias but low variance. Although measurements are inaccurate (not valid), they are consistent (reliable). Repeatable systematic error is associated with faulty equipment or a flawed experimental design and influences a measurement's accuracy.

Reproducibility error (variance) introduces noisy data with low bias but high variance. Although measurements are accurate (valid), they are inconsistence (not reliable). The repeatable error is due to a measurement process and primarily influences a measurement's accuracy. Reproducibility refers to the variation in measurements made on a subject under changing conditions.



Bias-Variance Tradeoff. Images adapted from Scott Fortmann-Roe[8], Abhishek Shrivastava[9], and Andrew Ng[10]
Bias-Variance Tradeoff


Underfitting, Optimal-fitting, and Overfitting in Machine Learning  Images adapted from Scott Fortmann-Roe[8], Abhishek Shrivastava[9], and Andrew Ng[10]
Underfitting, Optimal-fitting, and Overfitting in Machine Learning
Images adapted from Scott Fortmann-Roe[8], Abhishek Shrivastava[9], and Andrew Ng[10]

-       Trade-off:

“The expected test error of an ML model can be decomposed into its bias and variance through the following formula:

𝙩𝙚𝙨𝙩 𝙚𝙧𝙧𝙤𝙧 = 𝙗𝙞𝙖𝙨² + 𝙫𝙖𝙧𝙞𝙖𝙣𝙘𝙚 + 𝙞𝙧𝙧𝙚𝙙𝙪𝙘𝙞𝙗𝙡𝙚 𝙚𝙧𝙧𝙤𝙧

So, to decrease the estimation error [to improve accuracy], you need to decrease both the bias and variance, which in general are inversely proportional and hence the trade-off [11].”

Bias and variance trade-off needs to be balanced in order to address the differences in health care in this country and around the world. Increasing bias (not always) reduces variance and vice-versa.

-       Remedies:

 Early stopping:
One additional effective technique in solving the issue of overfitting and underfitting and building an optimal-fitting ML model is early stopping.

"Early stopping is one of the most commonly used strategies because it is straightforward and effective. It refers to the process of stopping the training when the training error is no longer decreasing but the validation error is starting to rise [12]."

Ensemble methods:
Combine models via the Boosting ensemble method to decrease the bias. Combine models via the Bagging ensemble method to reduce the variance.

Visualization
Data visualization is a graphical representation of information and data. Using visual elements, such as charts, graphs, and maps, data visualization techniques provide a visual way to see and understand trends, outliers, and patterns in data. Visualization tools provide capabilities that help discover new insights by demonstrating relationships between data.

Anscombe's Quartet [Image: Schutz ]
Anscombe's Quartet
Image: Schutz [13]

An additional benefit for visualizing data is that data sets that have similar descriptive statistics, such as mean, variance, correlation, linear regression, and coefficient of determination of the linear regression, yet have very different distributions and appear very different when graphed.

Anscombe's quartet [14], in the above image, comprises four data sets that demonstrate both the importance of graphing data when analyzing it and the effect of outliers and other influential observations on statistical properties.

In ML, the three major reasons, for data visualization, are for understanding, diagnosis, and refinement of your model.

One important purpose, you need to visualize your model, is for providing an interpretable (reasoning) predictive model and explainability of your model. Other significant purposes are visualizing your model architecture, parameters, and metrics.  

Also, you might need to visualize your model during debugging and improvements, comparison and selection, and teaching concepts.

Visualization is most relevant during training for monitoring and observing a number of metrics and tracking of model training progression. After training, visualizing model inference is the process of drawing conclusions out of a trained model. Visualizing the results helps in interpreting and retracing how the model generates its estimates (Visualizing Machine Learning Models: Guide and Tools [15]).

Confusion Matrix and Classification Evaluation Metrics
Once you fit your ML model, you must evaluate its performance on a test dataset.

Evaluating your model performance is critical, as your model performance allows you to choose between candidate models and to communicate how reasonable the model is at solving the problem.

Measuring, for instance, a binary output prediction (Classification) is captured in a specific table layout - a Confusion Matrix, which visualizes whether a model is confusing two classes. Each row of the matrix represents the instances in an actual class, while each column represents the instances in a predicted class. Four measures are captured: True Positive, False Negative, False Positive, and True Negative.

Calculating accuracy is derived from the four values in a confusion matrix. Additional metrics with formulas on the right and below are Classification Evaluation Metrics. These metrics include but are not limited to the following: Sensitivity, Specificity, Accuracy, Negative Predictive Value, and Precision.

Confusion Matrix and Classification Evaluation Metrics. Image: Maninder Virk
Confusion Matrix and Classification Evaluation Metrics. Image: Maninder Virk

In addition to accuracy, there are numerous model evaluation metrics. Three metrics that are commonly reported for a model on a binary classification problem are:

Precision
Recall
F1 score
Precision quantifies the number of positive class predictions that actually belong to the positive class. Recall quantifies the number of positive class predictions made out of all positive examples in the dataset. The F1 score combines the precision and recall of a classifier into a single metric by taking their harmonic mean. It is primarily used to compare the performance of two finer-grained classifiers.

Hierarchy of Metrics from raw measurements / labeled data to F1-Score. Adapted Image: Teemu Kanstrén
Hierarchy of Metrics from labeled training data and classifier predictions to F1 score. Adapted Image: Teemu Kanstrén

"The metrics form a hierarchy that starts by counting the true/false negatives/positives, at the bottom, continues by calculating the Precision and Recall/Sensitivity metrics, and builds up by combining them to calculate the F1 score [16]."

The importance and interpretation of evaluation metrics depend on the domain and context of your ML model. For instance, medical tests are evaluated by specificity and sensitivity, while information retrieval systems are evaluated by precision and recall. Understanding the differences between precision and recall vs. specificity and sensitivity is significant in your model evaluation within a specific domain [17].  

Bias vs. Variance of ML Algorithms. Image: Ega Skura
Bias vs. Variance of ML Algorithms. Image: Ega Skura
Bias vs. Variance of ML Algorithms. Image: Ega Skura

For ML model builders understanding how accuracy is affected by their model fitting is essential. Building an accurate classification model can correctly classify positives from negatives.

-       In essence:

“Balancing bias and variance ... is the best way to ensure that model is sufficiently [optimally] fit on the data and performs well on new [evaluation] data.” Solving the issue of bias and variance is about dealing with overfitting and underfitting and building an optimal model.

Next, read my "Complexity - Time, Space, & Sample" article at https://www.linkedin.com/pulse/complexity-time-space-sample-yair-rajwan-ms-dsc

---------------------------------------------------------

[1] https://www.linkedin.com/pulse/machine-learning-101-which-ml-choose-yair-rajwan-ms-dsc

[2] https://homes.cs.washington.edu/~pedrod/papers/cacm12.pdf

[3] https://www.datarobot.com/wiki/fitting

[4] https://prateekvjoshi.com/2013/06/09/overfitting-in-machine-learning

[5] https://docs.aws.amazon.com/machine-learning/latest/dg/model-fit-underfitting-vs-overfitting.html

[6] https://medium.com/ml-research-lab/under-fitting-over-fitting-and-its-solution-dc6191e34250

[7] https://medium.datadriveninvestor.com/determining-perfect-fit-for-your-ml-model-339459eef670

[8] http://scott.fortmann-roe.com/docs/BiasVariance.html

[9] https://www.kaggle.com/getting-started/166897

[10] https://www.coursera.org/lecture/deep-neural-network/bias-variance-ZhclI

[11] https://towardsdatascience.com/understanding-the-bias-variance-tradeoff-and-visualizing-it-with-example-and-python-code-7af2681a10a7

[12] https://theaisummer.com/regularization

[13] https://commons.wikimedia.org/wiki/User:Schutz

[14] https://www.tandfonline.com/doi/abs/10.1080/00031305.1973.10478966

[15] https://neptune.ai/blog/visualizing-machine-learning-models

[16] https://towardsdatascience.com/a-look-at-precision-recall-and-f1-score-36b5fd0dd3ec

[17] https://towardsdatascience.com/should-i-look-at-precision-recall-or-specificity-sensitivity-3946158aace1
