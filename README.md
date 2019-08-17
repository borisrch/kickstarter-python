# Machine Learning + Data Mining techniques on datasets using Python

## Getting Started
Dataset Download URL: [https://www.kaggle.com/kemical/kickstarter-projects](https://www.kaggle.com/kemical/kickstarter-projects)

Make sure that both 'ks-projects-201612.csv' and 'ks-projects-201801.csv' are located in the same working directory as the python file.

## Kickstarter Dataset

Kickstarter is an online crowdfunding platform where creators can share and gather interest on a
creative project they would like to launch. Crowdfunding is a method of raising capital through the general public to send these projects into production. A crowdfunding campaign can make or break a business idea, so it is advantageous to investigate attributes of successful Kickstarter projects. 

The datasets that we worked with was sourced from the Kickstarter platform. The data reflects the results of funded Kickstarter projects. This 2018 dataset includes 375764 records and 15 feature variables.

All models were performed with k-fold cross validation, as a result each models' accuracies represents mean accuracies (to k = 10 degrees of freedom) at 95% CI.

- Random Forests Classifier (0.794)
- Logistic Regression (0.718)
- K-Nearest Neighbours Classifier (0.743)
- Ada Boost Classifier (0.806)

Lines 327 to 363 display results from holdout (i.e. 1 partition of training and 1 partition of testing dataset) along with accuracy, precision, recall and AUC score. For more detailed analysis refer to PDF.

### Conclusion (Section 8 of Report)

In summary, our findings showed that the best model depends on the measure that is valued by the individual when looking for Kickstarter projects to invest in. Precision and Recall are popular measures that can be used depending on the situation, however we argued that for Kickstarter projects, the cost of false-positive is high, therefore the selected model should have a high precision score. The results in Section 7.6 show that the Logistic Regression model had the highest precision. 

If both precision and recall are important, then the model with the highest f-score should be considered, which for this paper was the AdaBoost model. Furthermore, in hypothesis testing, the AdaBoost model proved to be a statistically significant result in having the lowest error rate, compared to the Random forests model. The analysis of ROC curves indicates that the AdaBoost model also had the highest AUC value. Overall, the AdaBoost model performs notably well in comparison to the other classifiers, and should be highly considered by stakeholders to model the success of Kickstarter campaigns.

 