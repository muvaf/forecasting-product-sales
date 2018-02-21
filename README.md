Forecasting sales play important role in the operations management of most of the industries. After all, if you know how many products you'll sell, you'd do the 
best stock optimization even if you are a small store. In this project, I'll be using historical data of Rossmann Stores that is <a 
href="https://www.kaggle.com/c/rossmann-store-sales/data">publicly available</a>.

Technically, the most differentiating factor of forecasting sales is that you got a time series data and are trying to predict what will be the next value. That 
aspect of prediction alone makes many machine learning models unusable. As an instance, take the decision trees. How would you inject your data to the tree? Does 
it really make sense to have a leaf for every date?

In the sense of businesses, while there is so much data around, analyzing most of them doesn't give results that can be incorporated in the decision making 
processes hence the analysis doesn't turn into profits. However, sales data prediction is the common problem of most of the businesses. Stores of all kinds, 
software companies, consultancies etc. However, it's not always possible to make good predictions because of the nature of the company. Some firms sell only a few 
units of products but make big profit out of them, but technically it's really hard to come up with a good enough prediction with so small data.

Rossmann has published historical data of sales that include many products and the dates covered are wide enough to have meaningful training data. So, I'll use 
both statistical methods and machine learning models to predict sales of the products day by day.
