
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

#### Remove multicollinearity
To check for highly correlated predictors, I wrote the following function to return highly correlated pairs.  This information was used to remove one member of the highly correlated pair to ensure this correlation was not included in final models.

```
def multi_collinearity(predictors, threshold=0.75):
    """
    Function to identify multi-collinearity amongst predictors based on a threshold value.
    Parameters include:
    - predictors: dataframe of features to be tested for multi-collinearity
    - threshold: correlation threshold used to determine how many pairs are returned
    """

    df = predictors.corr().abs().stack().reset_index().sort_values(0, ascending=False)

    # zip the variable name columns (named level_0 and level_1 by default)
    df['pairs'] = list(zip(df['level_0'], df['level_1']))

    # set index to pairs
    df.set_index(['pairs'], inplace=True)

    # drop level columns
    df.drop(columns=['level_1', 'level_0'], inplace=True)

    # rename correlation column as cc rather than 0
    df.columns = ['cc']

    # drop duplicates and return
    df.drop_duplicates(inplace=True)
    return df[(df['cc'] > threshold) & (df['cc'] < 1)]
```

#### Encode categorical columns
As a final data preparation step, I one hot encoded categorical columns, an example of which is included below:
```
# bathrooms
ohe = OneHotEncoder()
bathrooms_ohe = ohe.fit_transform(cat_feats[['scaled_bathrooms']])
bathrooms_ohe = pd.DataFrame(bathrooms_ohe,
                             columns=ohe.categories_[0][1:], # dropped first
                             index=cat_feats.index)

# add suffix to column headers
bathrooms_ohe.columns = [f'{col}-baths' for col in bathrooms_ohe.columns]
bathrooms_ohe.head()
```

### 4. Modeling
After preparing the dataset, I moved on to iteratively modeling a multivariate regression to predict house sale price in King's County.  

To ensure only statistically significant features were included in the model, forward-backward feature selection based on p-value was used.  

In total, three model iterations were completed, including a baseline and two additional iterations.  Assumptions required for linear regression were checked during each iteration, to ensure assumptions were not being violated.  

These checks included producing Q-Q plots to evaluate if residuals were distributed normally, in addition to regression plots to show that assumptions of homoscedasticity hold. Examples of these two visualizations can be found below:

![Q-Q Plot](/imgs/qqplot.png)
![Residual Plot](/imgs/residplot.png)

### 5. Evaluation
During the evaluation stage, results of all different model iterations were compared, looking first to see if any assumptions required for linear regression were being violated, then moving to look at performance of the regression.  Analysis of final coefficients and adjusted r-squared metrics was completed to identify the best performing model.  

The final model benefitted from a log transformation on the target variable (price) and the removal of some outliers from independent variables.  

The positie effects resulting from log-transforming price can bee seen in the two distributions below:

![Price vs. Log-Price](/imgs/pricelog.png)

As can be seen, the log-transformed variable is much more normally distributed, helping improve performance of the regression.

The best identified model ended up being comprised of 7 predictor variables, and produced an adjusted r-squared value of 0.716, meaning 71.6% of variation in the target variable was explained by our model.  

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
 - When looking at month_sold vs. price, there is a slight bump in sale price during the spring months or month 4, month 5, and month 6.  This all supports the fact that a seller should target Spring to maximize the month in which sales price tends to be highest, albeit fairly small impacts

### Question 3 - Are the KC provided measures of quality (`condition` and `grade`) accurate for predicting price / can they be trusted?
 - Looking at box plots of grade and condition vs. price, we can see that houses with both high grades and high condition values are correlated with high sale prices.  `grade` shows a clear ordinal relationship with `price`, and when evaluating `condition`, it's apparent that condition values of 3-5 are correlated with higher sales price.  

### Final Model:
the final model is comprised of 7 predictor variables, inclusive of a constant
 - Final target variable: `log_price`
 - Final predictor variables and coefficients:
     - `sqft_basement`: 0.000267
     - `sqft_above`: 0.000214
     - `lat`: 1.455790
     - `grade`: 0.162307
     - `1-waterfront`: 0.639687
     - `1-reno`: 0.18009

Coefficients can be interpreted in the following manner: a 1-unit increase in the independent variable, results in a coefficient sized change in the target / dependent variable.  Because price was transformed to `log_price`, our coefficients will be representing changes in `log_price`.  The following code can be applied to reverse the effects of the log transformation and make the coefficients easier to interpret:

```
scaled_coef = 10 ** coef # reverses effects of log transformation
```

After reversing the effects of the log transformation, we see the following coefficients:
- `sqft_basement`: 1.00
- `sqft_above`: 1.00
- `lat`: 28.56
- `grade`: 1.45
- `1-waterfront`: 4.36
- `1-reno`: 1.51

Effect sizes are really small, even when translating back from log transformed to regular price.  A 1-unit increase in house square footage, according to our model, is associated with a 1 dollar increase in sale price.  While coefficients are statistically significant, the small effect sizes make it somewhat challenging to pull a lot of insight from our model.  Latitude has the biggest effect size on sale price, indicating location of houses is important.  

### Recommendations:
1. When buying houses, target the purchase of larger homes, within the 47.6 to 47.7 degree latitude band.  Additionally, look for houses that have watefront view or have received renovations, to ensure you are buying an asset the market values.  If looking to buy asset where immediate investment can improve and drive up sale price, looking for a house that has not yet seen renovations will be key, as the investor should be able to realize these gains going forward.
2. When selling assets, target selling during the spring months, as `month_4`, `month_5`, and `month_6` as these months are associated with slightly higher selling prices.
3. Property management - given renovations and square footage being positively correlated with sale price, when looking to increase value of an existing property, adding square footage above or below ground, in addition to other types of renovations, should help drive gains.  
4. When performing due diligence processes, if time is short or decisions need to be made quickly, `grade` has been shown to be a strong predictor of overall sale price.  Use this information strategically, or when in a pinch, to make quick decisions on limited information. 
