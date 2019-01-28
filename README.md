# Titanic Dataset Exploration

A small project exploring the [Titanic Dataset](https://www.kaggle.com/c/titanic) on [Kaggle](https://www.kaggle.com/). In this project, we'll explore survivorship for passengers on board the [RMS Titanic](https://en.wikipedia.org/wiki/RMS_Titanic).

This project is written in Python and uses Jupyter Notebooks. Machine learning using the SkLearn package for Python. Dataframes from the Pandas library. Plots from the MatPlotLib library.

The following is an overview of what is covered in the iPython Notebook:

## Setup

Read in the training dataset, which has labels indicating whether or not each passenger survived.

```python
df = pd.read_csv('data/train.csv')
```

In the training set, there are 891 rows, with the following columns:

| **Column Name** | **Type** | Description                                                  |
| --------------- | -------- | ------------------------------------------------------------ |
| `PassengerId`   | int      | Id of the passenger in this dataset                          |
| `Survived`      | int      | Boolean indicating whether or not the passenger survived the voyage (0 = No, 1 = Yes) |
| `Pclass`        | int      | Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)                     |
| `Name`          | string   | Name of the passenger                                        |
| `Sex`           | string   | Sex of the passenger                                         |
| `Age`           | float    | Age of the passenger in years                                |
| `SibSp`         | int      | Number of siblings or spouses also aboard the Titanic        |
| `Parch`         | int      | Number of parents or children also aboard the Titanic        |
| `Ticket`        | string   | Ticket Number                                                |
| `Fare`          | float    | Passenger fare (in USD)                                      |
| `Cabin`         | string   | Cabin Number                                                 |
| `Embarked`      | string   | Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton) |

## Data Exploration

**Note**: these figures are for the training set only.

### Overview

Start off by getting a sense of the data and look for patterns to measure statistically.

![Head of DataFrame](https://raw.githubusercontent.com/henrywoody/titanic/master/img/df-head.png)

### General Trends

Explore some general trends regarding survivorship of passengers on the Titanic. Overall, 61.6% of passengers died, while 38.4% survived.

Here is a breakdown of passenger statistics for the average surviving passenger and the average passenger who did not make it:

![Survivorship pivot table](https://raw.githubusercontent.com/henrywoody/titanic/master/img/survivorship-pivot.png)

A few things to notice here:

- Survivors tended to be younger than non-survivors on average
- Survivors tended to have paid a higher fare and stayed in a higher class cabin (as indicated by a lower value for `Pclass`) than non-survivors
- In general, having a greater number of parents or children on board is correlated with a greater survival rate, but having *fewer* siblings/spouses was related to a higher rate of survival

### Sex

Explore differences in survival rates between the sexes. Of the 891 passengers in our dataset, 577 were male and 314 were female. For male passengers, the survival rate was 18.9%, while that of female passengers was 74.2%.

Here is a breakdown of some passenger statistics split by sex and whether or not the passenger survived:

![Sex-Survivorship pivot table](https://raw.githubusercontent.com/henrywoody/titanic/master/img/survivorship-sex-pivot.png)

So far we can see that female passengers were much more likely to survive the ship's sinking than men. We can also see that female passengers who survived tended to be older than female passengers who did not, but that the opposite relationship holds for men.

### Age

Explore the impact of age on survivorship for passengers. The minimum age of a passenger in our dataset was 0.42 years and the maximum was 80 years.

Here is a plot of the density of passengers by age divided by survival status:

![Age density of survivors split by age](https://raw.githubusercontent.com/henrywoody/titanic/master/img/age-density-survivorship.png)

Looking at the plot above, we can see that the age trends for people in both survival groups are quite similar. We can see that survivors had a somewhat wider range of ages, while those who died were more concentrated at around 20-30 years old. Also there is a large concentration of very young children (less than 5-7 years old) that survived.



### Socio-Economics

To explore socio-economic status of passengers, we'll look at the `Fare` and `Pclass` for each passenger. The minimum fare for any passenger in the set is $0 and the maximum fare paid was $512.

First looking at fare price:

![Fare price density of passengers split by survivorship status](https://raw.githubusercontent.com/henrywoody/titanic/master/img/fare-density-survivorship.png)

From the above plot we can see that most of the passengers who died paid quite a low fair price. While those who lived tended to have paid higher prices to board.

Now turning to the relationship between cabin class and survival. Of the passengers in our set, 24.2% were in 1st class, 20.7% were in 2nd class, and 55.1% were in 3rd class.

![Exploration of class and survivorship](https://raw.githubusercontent.com/henrywoody/titanic/master/img/class-survivorship.png)

So we can see that of those who did not survive, a large majority (67.8%) were in 3rd class. Of those who did survive, the most common class is 1st (39.8%). Breaking passengers into classes, we find that the probability of surviving is highest among those in 1st class, and steadily decreases as the cabin becomes less exclusive.

## Constructing a Model

We'll use the SkLearn library to construct a model to predict whether a passenger will survive the voyage of the Titanic given the features outlined above (less `Survived`). For these, we'll first split our dataset into a training set and test set. (Note: this is not to be confused with the Kaggle train/test split, since we are only using the training set from Kaggle for our models).

I chose to use a Decision Tree Classifier and Gradient Boosting Classifier in this section because they are often quite accurate and are able to handle mixed data-types easily.

### Baseline

First we'll create a baseline model, nothing fancy, just to see how much better a more advanced model is than a simple approach and see if we're heading in the right direction.

We'll use a Decision Tree Classifier, which works by successively splitting the dataset into two groups. At each step the model finds a way to split the current pool of observations such that the two groups are as homogeneous as possible with respect to the target variable (here survivorship). Then to make a prediction for a new observation, the model just follows the decision tree until reaching a terminal node and taking the predicted label (survive or not) of that node. For example, the model might start by splitting on Sex, then, for male, whether age is less than 30 or not, and, for female, whether `SibSp` is less than 2 or not, and so on.

Using a Decision Tree Classifier and the columns `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, `Embarked` as features we get the following results:

- Accuracy on training set: 0.9925093632958801
- Accuracy on test set: 0.7528089887640449
- Area under the ROC curve score on test set: 0.7631237006237006

For reference, the receiver operating characteristic (ROC) curve is a graph that depicts the performance of a classification model (such as ours) at all classification thresholds. The curve compares the True Positive Rate to the False Positive Rate. The area under the ROC curve gives a more accurate picture of the performance of our model than just accuracy alone mainly because it accounts for skew in the dataset. For example, if 99% of passengers survived the voyage, then simply always predicting that a passenger would survive results in a 99% accuracy, but a much lower area under the ROC curve score. You can read more about this metric [here](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc).

### A More Advanced Model

Now we'll create a model using a Gradient Boosting Classifier. A Gradient Boosting Classifier is a more complex model that is composed of many individual, and simple, Decision Tree Classifiers, where each successive decision tree attempts to correct the mistakes/bias of the previous.

Using a Gradient Boosting Classifier with the following parameters:

- `learning_rate`: `0.1` — corresponds to the emphasis on correcting the mistakes of previous trees in the chain, a higher learning rate generally corresponds to more complex trees
- `max_depth`: `4` — corresponds to the maximum allowed depth of any tree in the classifier

We find the following results:

- Accuracy on training set: 0.9588014981273408
- Accuracy on test set: 0.8202247191011236
- Area under the ROC curve score on test set: 0.8696075883575884

So we're doing quite a bit better with the Gradient Boosting Classifier than the simple Decision Tree Classifier—compare area under the ROC curve of 0.87 for the Gradient Boosting Classifier to 0.76 for the Decision Tree Classifier.

Here are the feature importances found by our model:

| **Feature** | Importance |
| ----------- | ---------- |
| `Sex`       | 36.0%      |
| `Age`       | 21.2%      |
| `Fare`      | 19.3%      |
| `Pclass`    | 15.0%      |
| `SibSp`     | 5.7%       |
| `Parch`     | 1.4%       |
| `Embarked`  | 1.1%       |

![Feature importances](https://raw.githubusercontent.com/henrywoody/titanic/master/img/feature-importances.png)

By a wide margin, Sex is the most important factor in predicting the probability that an individual would survive the voyage of the Titanic.  The next most important feature was Age. Then followed by Fare and Pclass (corresponding to wealth).

### Splitting by Sex

We'll now use the same model as above, but create separate models for men and women since the factors that impact the probability of survival for each seem to differ. With this set up, we find the following results:

- Male
  - Accuracy on training set: 0.976401179941003
  - Accuracy on test set: 0.8157894736842105
  - Area under the Curve score on test set: 0.6536824180502341
- Female
  - Accuracy on training set: 0.9948453608247423
  - Accuracy on test set: 0.8153846153846154
  - Area under the Curve score on test set: 0.876

Initially we can see that the model performs much worse for male passengers than for female passengers. This might be because the survival rate of males overall was much lower than for female passengers, so luck might have played a greater role in survival for men than for women, making survival harder to predict among men.

Now looking at the feature importances of the two models:

![Feature importances split by sex](https://raw.githubusercontent.com/henrywoody/titanic/master/img/sex-split-feature-importances.png)



Now we can see that Age is by far the most important feature in determining the probability of survival of a male passenger, but that it is not as important for female passengers. It seems that socio-economics were relatively more important for female passengers than for males. These differences should, however, be taken with a grain of salt due to the inaccuracy of the model for males.