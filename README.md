
# Predicting Housing Prices in King's County
![House Size vs. Sale Price](/imgs/sqft_above_vs_target.png)

## README Outline
Within this README file you will find:
1. Introduction
2. Overview of Repository Contents
3. Project Objectives
4. Overview of the Process
5. Findings & Recommendations
6. Conclusion / Summary

## Introduction
I iteratively build and evaluate a multivariate regression to predict house sale prices in King's County as accurately as possible.  This exercise was completed in the context of a private equity / real estate investment firm looking for diligence and valuation support.  

## Repository Contents
Within this github repository you will find the following:
1. [`README.md`](https://github.com/akaigraham/KC-House-Price-Regression/blob/main/README.md) - this file
2. [`House-Price_Regression.ipynb`](https://github.com/akaigraham/KC-House-Price-Regression/blob/main/House-Price_Regression.ipynb) - jupyter notebook containing all code and analyses / models
3. [`House-Price-Analysis_Presentation.pdf`](https://github.com/akaigraham/KC-House-Price-Regression/blob/main/House-Price-Analysis_Presentation.pdf) - non-technical presentation presenting methodology, findings, and recommendations of analytical work
4. [`imgs`](https://github.com/akaigraham/KC-House-Price-Regression/tree/main/imgs) - directory containing images found within this file

## Project Objectives
Accurately predict King's County house sale price using multivariate regression.  Regression to be used in context of real estate investor looking to make asset purchases / sales or determine what potential investments to make in an existing asset to drive up potential sale value.

In addition to the primary objective, I am hoping to answer the following questions:
1. What does the current KC housing market look like? What do the majority of houses have in common?
2. Are there specific times (months) that might be related with a higher sale price?
3. Are provided rankings (grade, condition, etc.) trustworthy / relevant to predicting sale price or will additional diligence be required to build our own rankings?

## Overview of the Process
I followed the Cross-Industry Standard Process for Data Mining (CRISP-DM), comprised of the following steps:
1. Business Understanding
2. Data Understanding
3. Data Preparation
4. Modeling
5. Evaluation

### 1. Business Understanding
This process aims to build a regression to accurately predict house sale prices in King's County, with the goal of providing investors in the area a robust tool for making smarter investments.  This regression can be leveraged by investment professionals to validate other valuation methodologies and ultimately determine purchase price of housing assets, ensuring purchasers are not overpaying.  

In addition to valuation predictions, unpacking which features and characteristics are most significant can help guide purchasing decisions and provide a checklist of features to look for when buying new assets.  

In addition to aiding buyers in making smarter purchases, this tool can be leveraged by existing property owners to guide investments into existing properties.  Property owners can target projects, renovations, and expansions that are most likely to improve sale price.  


### 2. Data Understanding
This project uses the King's County House Sales dataset from [Kaggle](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction).  Sale price data comes from May 2014 through may 2015.  

The raw dataset takes up just over 3.5MB of storage, and contains 21 columns:
* `id` - unique identifier for a house
* `date` - date house was sold
* `price` - sale price and the prediction target
* `bedrooms` - number of bedrooms/house
* `bathrooms` - number of bathrooms/bedrooms
* `sqft_living` - square footage of the home
* `sqft_lot` - square footage of the lot
* `floors` - total floors (levels) in the house
* `waterfront` - house which has a view to a waterfront
* `view` - has been viewed
* `condition` - how good the condition is (Overall)
* `grade` - overall grade given to the housing unit, based on King County grading system
* `sqft_above` - square footage of house apart from basement
* `sqft_basement` - square footage of the basement
* `yr_built` - year built
* `yr_renovated` - year when house was renovated
* `zipcode` - zipcode
* `lat` - latitude coordinate
* `long` - longitude coordinate
* `sqft_living15` - the square footage of interior housing living space for the nearest 15 neighbors
* `sqft_lot15` - the square footage of the land lots of the nearest 15 neighbors

After loading the dataset, plotting a histogram of price helps show the distribution of our target variable:
![Target Variable Distribution](/imgs/target_hist.png)

In addition to the target, understanding distribution of predictors provides a good starting point of understanding which values are available to us in which columns, and start to identify any potential issues:
![Predictors Histograms](/imgs/predictors_hist.png)

Some key observations on the different features are listed here:
* `id`: Again, not very helpful given this is just a unique identifier
* `bedrooms`: We see the majority of KC houses have 1 to 5 bedrooms, with some outliers having more than 5.  The median number of bedrooms in KC is 3.   
* `bathrooms`: The majority of KC houses have less than 3 bathrooms, with houses having 1 or 2.5 bathrooms being most common. The median is 2.25, with 75% of all houses in the dataset having 2.5 bathrooms or fewer.
* `sqft_living`: Median living size is just under 2000 sq. ft at 1,910 sq.ft.  75% of KC houses have houses with 2,550 sq. ft. and less.
* `sqft_lot`: The presence of outliers is clear, given the small unimodal bump close to 0, with a significant right skew.  Median lot size is 7,620 sqft.
* `floors`: 75% of houses in KC have 2 floors or fewer.  The highest house has 3.5 floors.
* `waterfront`: Very few houses in KC have waterfront views
* `view`: Most houses in KC have not been viewed
* `condition`: Median condition is 3 - seems we have a good representation of mid-condition homes.
* `grade`: Median house grade is 7, with a max grade of 13 in the dataset.  Minimum grade is 3 - - seems the scale may have been shifted to remove 0 grades by 3 to the right.
* `sqft_above`: Median sqft above is 1,560 sq. ft, with 75% of all KC houses having sqft above of 2,210 or less
* `sqft_basement`: A large number of houses do not have basements
* `yr_built`: The earliest KC house in the dataset was built in 1900, the earlies built in 2015.  Median yr_built is 1975.  75% of all houses were built in 1997 or prior.
* `yr_renovated`: Majority have not been renovated.
* `zipcode`: Good range of values
* `lat`: see notes above
* `long`: see notes above
* `sqft_living15`: median of 1,840, max of 6,210 sqft.
* `sqft_lot15`: median of 7,626 sqft., max of 871,200 sqft. (this may be a mistake, definitely an outlier at the minimum).
* `datetime`: We see date sold ranges from May 2014 to May 2015

We now have a solid understanding of the dataset and what features are available to us. Move forward with data preparation.

### 3. Data Preparation
To prepare the raw dataset for modeling, this section handles a number of preprocessing and cleaning steps including: checking and handling of missing values and duplicates.  Missing values found in the following three columns: `waterfront`, `view`, and `yr_renovated`.

Comparing price of houses with missing data to those houses with data provides insight into how to best handle the missing values.

![Price of Houses Split by Waterfront](/imgs/waterfront_comparison.png)
![Price of Houses Split by Reno](/imgs/reno_comp.png)

With a good understanding of how price breaks down by waterfront and renovations, plotting the missing data points reveals distributions more inline with houses that do not have waterfront and houses that have not had renovations.
![Missing Waterfront Prices](/imgs/missing_water.png)
![Missing Renovation Prices](/imgs/missing_reno.png)

Next step is to handle typing of various columns to ensure issues will not pop up later when modeling.  One example of this, was converting the date column and extracting the month information, as shown below:
```
# convert date to datetime column
predictors['date_sold'] = pd.to_datetime(predictors['date'], infer_datetime_format=True)

# drop initial date column
predictors = predictors.drop('date', axis=1

# create month sold column
clean_df['month_sold'] = clean_df['date_sold'].map(lambda x: x.month)
```

Reviewing the distribution of month sold, it is easy to see that the majority of houses are sold during spring months.
![Month Sold Distribution](/imgs/month.png)

#### Evaluate relationship with target
Prior to moving forward with modeling, understanding how each of our predictor's is related with the target variable is important to making sure the right features are included in the model.  

To do this, I plotted scatterplots against price for each continuous column, and created boxplots against price for each categorical column, as shown below:
![Cont Feats](/imgs/cont_feats.png)
![Cat Feats](/imgs/cat_feats.png)


## Findings & Recommendations

#### Question 1 - Current KC Housing Market & Discussion of Distributions
 - bathrooms: appears to be categorical with the majority of entries falling between 0.5 and 3 bathrooms. there are a number of outliers to the right, with one entry having 8 bedrooms.  Room to clean these outliers up after the baseline model is generated.  Summary stats presented above, shows a median of 2.25 bathrooms, and a standard deviation of just over 0.75 bathrooms
 - bedrooms: appears to be categorical with the majority of entries having 1 to 4 bedrooms.  Significant outlier with one entry having 33 bedrooms.  Median number of bedrooms is 3, with a standard deviation of just under 1 bedroom
 - condition: appears categorical, max of 5, median of 3 - - looks like the majority are condition ratings of 3 and 4.  standard deviation of 0.65 (just over half one condition rating)
 - day_sold: appears to be categorical.  Looks like there is no deviation other than near the beginning of the month (day 1) and the middle of the month (day ~20) showing more houses sold near these days than on others
 - floors: categorical with a significant falloff after 2 floors
 - grade: appears somewhat normally distributed although it is categorical.  Some grades above 10, with the most between 6 and 9.  Median grade is 7 with a std of 1.17
 - month_sold: categorical, but interesting to see that there are more houses sold during the summer months than winter months, which makes sense
 - price, sqft_above, sqft_living, sqft_living15 appear log normally distributed, with right skew and likely some positive outliers
 - sqft_lot and sqft_lot15 have have significant right skew and outliers
 - view: very few houses have been viewed (> 0 value)
 - waterfront: very few houses have waterfront
 - year_sold: only two years of sale data is included here (2014 and 2015), with more entries coming from 2014 than 2015
 - yr_built: the majority of houses represented were built after 1950, with the earliest house included being built in 1900
 - yr_renovated: only a handful have received renovations

### Question 2 - Is it beneficial to sell a house during specific times of the year?
 - When looking at month_sold vs. price, there is a slight bump in sale price during the spring months or month 3, month 4, and month5.  Additionally, our final model includes `month_3`, `month_5`, and `month_6` as coefficiencts that are statistically signficant from zero. This all supports the fact that a seller should target Spring to maximize the month in which sales price tends to be highest, albeit fairly small impacts

### Question 3 - Are the KC provided measures of quality (`condition` and `grade`) accurate for predicting price / can they be trusted?
 - Looiking at boxplots of grade and condition vs. price, we can see that houses with both high grades and high condition values are correlated with high sale prices.  `grade` shows a clear ordinal relationship with `price_boxcox`, and when evaluating `condition`, it's apparent that condition values of 3-5 are correlated with higher sales price.  Additionally, `condition_5` and `condition_4` are coefficients in our final model and statistically signifcant from zero at the 0.05 level.

### Final Model:
the final model is comprised of 13 predictor variables, of a constant
 - Final target variable: `price_boxcox`
 - Final predictor variables:
     - `t_sqft_liv_15`
     - `grade`
     - `basement_1`
     - `floors_15`
     - `condition_5`
     - `reno_1`
     - `condition_4`
     - `month_4`
     - `floors_20`
     - `month_3`
     - `month_5`
     - `day_26`

Because price was transformed to `price_boxcox` using a boxcox transformation with a fitted lambda value of -.24, meaning our coefficients will be representing changes in `price_boxcox` (price raised to the -.24 exponent).  Our model has an adjusted r-squared value of 0.548, representing the amount of variance in `price_boxcox` that can be explained by our predictors.

Coefficients
 - `grade`: overall grade given to the house based on the King County grading system.
     - coefficient: 0.0105 -- a one unit change in grade, holding all other variables constant, relates to a 0.0105 change in box cox transformed price
 - `basement_1`: house has a basement
     - coefficient: 0.007 -- having a basement is associated with an increase to `price_boxcox` by 0.007, holding others constant
 - `reno_1`: house has been renovated
     - coefficient: 0.011 -- holding others constant, a renovated house is asssociated with an increase to `price_boxcox` by 0.011
 - `condition_5`: overall condition rating=5
     - while coefficients are relatively small, a condition rating of 5 is associated with an increase to `price_boxcox` by 0.009

Effect sizes are really small, even when translating back from boxcox to regular price.  We have an intercept of 3.58 `price_boxcox`, which when translated back to price, is $3,053.  While coefficients are statistically signficant, with most confidence intervals containing non-zeros, a such a small effect size makes it hard to pull real insight from this data.

### Recommendations:
1. When buying houses, look for houses that have no history of renovations, a low KC grade, and a house with 2 floors.  Houses with history of renovation are related to higher priced houses, similar with high-graded homes.  Targeting purchases with no history of renovation and a low grade should be related with a lower house purchase price based on our model.
2. When selling assets, target selling during the spring months, as `month_3`, `month_4`, and `month_5` are associated with higher sale prices.
3. Change management - driving asset improvement.  As we can see from the `reno_1` coefficient, a renovation is related to higher sale prices, along with houses with basements, and high grades / conditions.  As a result, it is recommended to implement these changes if possible to purchased assets before selling to try and maximize sale price. If possible, add basement, invest in renovations and improvements, with a goal of driving up condition ranking as close to 5 as possible, and driving grade as high as possible.
