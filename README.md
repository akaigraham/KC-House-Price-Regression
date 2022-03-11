
# Module 2 Final Project


## README Outline

Within this README file you will find:
1. Introduction
2. Overview of Repository Contents
3. Project Objectives
4. Overview of the Process
5. Findings & Recommendations
6. Conclusion / Summary

## Introduction
Build multivariate regression on house sale price using King's County housing dataset. The ultimate goal is to produce a model that will be used by real estate investors in conjunction with traditional due diligence processes to better compute valuation and factors around buying / selling houses.

## Repository Contents

Within this github repository you will find the following:
1. `README.md`
2. `student.ipynb` - jupyter notebook containing all code and analyses / models
3. `presentation.pdf` - non-technical presentation presenting methodology, findings, and recommendations
4. `column_names.md` - file containing descriptors of column names 
5. `kc_house_data.csv` - dataset containing all target and predictor variables
6. `backup_files` - backup jupyter files and in process files used to arrive at final `student.ipynb` file

## Project Objectives

Accurately predict King's County house sale price using multivariate regression.  Regression to be used in context of real estate investor looking to make asset purchases / sales or determine what potential investments to make in an existing asset to drive up potential sale value.

In addition to the primary objectie, I am also hoping to answer the following questions:
1. What does the current KC housing market look like? What do the majority of houses have in common? What does the typical KC house look like?
2. Are there specific times (years, months, days) that might be related with a higher sale price?
3. Are provided rankings (grade, condition, etc.) trustworthy / relevant to predicting sale price or will additional diligence be required to build our own rankings?

## Overview of the Process

1. Load dataset and handle data issues (missing values, weird values, column data types, etc.)
2. Identify continuous, ordinal categorical, and non-ordinal categorical variables
3. Drop continous variables that do not meet linearity requirments
4. Handle multi-collinearity
5. Handle categorical variables - treat ordinal categorical variables as single columns and one-hot encode non-ordinal categorical columns
6. Use stepwise selection to select features that meet p-val thresholds (0.05)
6. Run baseline model with these features to get preliminary results to compare subsequent models against
7. Evaluate regression diagnostics, check where assumptions on linearity, normality of residuals and heteroscedasticity of residuals
8. Remove clear outliers and ontinue preprocessing / transforming to improve results

## Findings & Recommendations

#### Question 1 - Current KC Housing Market & Discussion of Distributions
 - bathrooms: appears to be categorical with the majority of entries falling between 0.5 and 3 bathrooms. there are a number of outliers to the right, with one entry having 8 bedrooms.  Room to clean these outliers up after the baseline model is generated.  Summary stats presented above, shows a median of 2.25 bathrooms, and a standard deviation of just over 0.75 bathrooms
 - bedrooms: appears to be categorical with the majority of entries having 1 to 4 bedrooms.  Significant oulier with one entry having 33 bedrooms.  Median number of bedrooms is 3, with a standard deviation of just under 1 bedroom
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