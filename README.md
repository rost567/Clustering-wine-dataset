# Clustering wine dataset
Learning about methods that we can understand how to treat the data as a set of general patterns rather just individual points. All of these points based in the book  "Python: Advanced Predictive Analytics" (Ashish Kumar, Joseph Babcock)
When examining data from the financial market or an e-commerce business, clustering algorithms can identify similar patterns in the data without having a specific response variable. This can provide useful information for predictive models or general summaries of the data.
This project explains about grouping or clustering algorithms, those are know unsupervised learning, it means, we have no response value, such as sale price or clicking-trought rate, which are used to determine the optimal parameters of the algorithm.

**Step 1**: identify similar datapoints using only the features.  
**Step 2**: Whether the clusters we identify share a common pattern in their responses.  
**Step 3**: So, suggets the cluster is useful in finding groups associated with the outcome we are interested in.  

Finding these clusters has steps which vary between algorithms.
1) It is a notion of distance or similarity between items, which alows us to quantitatively compare thhem.
2) Number of groups we wish to identify, this can be using domain knowledge, or by running an algorithm with different numbers of clusters. This number is important because help us to describe a dataset through statistics, numerical variance within the groups determined by the algorithm, or visual inspection.  
<br>

**GOALS**  
> - Normalise data for use in a clustering algorithm, calculate similarity measures for categorical and numerical data. <br>
> - Use k-means clustering to identify the optimal number of clusters, using the quadratic error function.  
> - Use agglomerative clustering to identify clusters at different levels.
> - Use affinity propagation to automatically identify the number of clusters is a dataset.   
> - Use spectral methods to cluster data with non-linear relationships between points.

## Similarity and distance metrics
First step is to decide how to compare the similarity or dissimilarity between items. Can be according the properties of the data or according in our interests.  
There are methods of distance for:  
> - numerical, 
> - categorical, 
> - time series, 
> - and set-based data. 
> - Also, normalizations for different data types prior to running clustering algorithms.

### Numerical distance metric
For this we are using a wine.csv data.
How is possible to calculate a similarity measurement between wines based on the information in each row? It means with data.describe()  
1. Would be to consider each of the wines as a point in a thirteen-dimensional space specified by its dimensions, each column.  
2. We cannot directly visualize a thirteen dimension space in a scatterplot to see if they are nearby, so, we can calculate distances for a more familiar 2 or 3 dimensional space using the Euclidean distance formula (length of the straight line between two points). This formula can be used whether the points are in a 2 or 13 -dimensional plot.  
<br>

> $ D(x,y) = \sqrt{\sum_{i=1}^{n}(x_i - y_i)^2} $<br>
> - x and y are rows 
> - n is number of columns
> - An important aspect of Euclidean distance formula is that columns whose scale is much different from others can dominate the overall result of the calculation, i.e the mean of magnesium content is 100x aprox, so it can dominate. 

**data.describe() RESULTS**  
The distance of these data-points would be clearly determined by the megnesium concentration.  
This might sometime be desirable, in the case of the column with the largest numerical values is the one we most care about for judging similarity.  
In most cases, this situation does not have favor, one feature over other, is better to give equal weight to all columns.  
<br>

**APPLYING EUCLIDEAN**<br>
So for a fair distance comparison we need **normalize the columns**, because in this way all the columns fall into the same numerical range **(means have similar max and min values)**.  
<br>
How to do this? 
> - scale() function in scikit-learn. 
> - This function will subtract the mean value of a column from each element and then divide each point by the standard deviation of the column. The output give us a numpy array.   
> - This normalization centers the data in each column at mean 0 with variance.  
> - In this case od normally distributed data this results in a standard normal distribution.  
> - Now that we have scaled the data, we can calculate Euclidean distances between the rows using the following commands.

The output is a square matrix of dimension: (178, 178)
We have converted the dataset into a square matrix, given the distances between each of these rows. $Row_i$, $column_j$ represent the Euclidean distance between rows i and j in the dataset. Now, this **distance matrix**  is the input we will use for clustering inputs.  

**VISUALIZATION**<br>
To create a visualization of how the points are distributed relative to one another using a given distance metric, is possible to use **multidimensional scaling(MDS)**.  
This method attempts to find the set of lower dimensional coordinates (here using 2 dimensions) that best represents the distances between points in the original, higher dimensions of a dataset (here using the pairwise Euclidean distances we calculated from the 13 dismension)<br>  
MDS finds the optimal 2-d coordinates according to the function.<br> For that we will use the distances calculated between points, the 2-d coordinates that minimize this function are found using **Singular Value Descomposition (SVD)**. After obtaining the coordinates from MDS, we can plot the results.<br>
<br>
The coordinates themselves have no interpretation, they change due to numerical randomness.  
Rather, it is the relative position of points that we are interested in.  

**EUCLIDIAN RESULTS**
Is the Euclidean distance a good choice here? Visually:<br>
> - There is separation between the classes based on the features we have used to calculate distance. Conceptually it appears that this is a reasonable choice in this case.
> - The decision also depends on what we are trying to compare. 
> - If we are interested in detecting wines with similar attributes in absolute values (absolute composition of the wine), then it is a good metric.
> - However, if that is not the interest, but what if we want to know whether its variables follow similar trends among wines with different alcohol contents? It mean not the absolute difference in values, but rather the **correlation between columns**. This is common for time series, so we consider that next.

### Correlation similarity metrics and time series
We are concerned with whether the patterns between series exhibit the same **variation over time, rather than their absolute differences in value.**
<br>
Comparing stocks, we might want to identify groups of stocks whose prices move up and down in similar patterns over time.
<br>
The absolute price is of less interest than this pattern of increase and decrease.
<br>
Example: Using the variation in prices of stocks in the Dow Jones Industrial Average (DJIA) over time.
**RESULTS**
All the numerical values (prices) are on the same scale, we won't normalize.<br>
Although we notice two issues.<br>
**First:** the closing price per week, this variabble is useful for the calculation of the correlation, is presented as string.
<br>
**Second:** The date is not present in the correct format for plotting.
<br>
**Solution:** Converting these columns to a float and datetime object, respectively. 


We have only two columns to calculate correlations between rows, as the first two columns are the index and stock ticker symbol.<br>
Let's calculate the correlation between these time series of stock prices by selecting the second column to end columns of the dataframe for each row.<br>
Calculating the pairwise correlations distance metric, and visualizing it using MDS, as before:

The Pearson coefficient, which we have calculated here, is a measure of linear correlation between these time series. In other words, it captures the linear increase (or decrease) of the trend in one price relative to another, but won't necessarily capture nonlinear trends (such as a parabola or sigmoidal pattern). We can see this by looking at the formula for the Pearson correlation.  

**SPEARMAN RESULTS**  
The **Spearman** correlation distances, based on the x and y axes, **appear closer to each other than the Pearson** distances, suggesting from the perspective of rank correlation that the time series are more similar.

**Which method to use?**
Euclidean distance between articles could be computed, but because each coordinate is either 0 or 1, it does not provide the continuous distribution of distances we would like (we will get many ties, since there are only so many ways to add and subtract 1 and 0).  
Measurements of correlation between these binary vectors are less than ideal because the values can only be identical or non-identical, again leading to many duplicate correlation values.  
So, we can use Jaccard coefficient, Cosine or Hamming distance, Manhattan distance:
> 1. Jaccard coefficient: This is the number of intersecting items (positions where both a and b are set to 1 in our example) divided by the union (the total number of positions where either a or b are set to 1).This measure could be biased, however, if the articles have very different numbers of keywords, as a larger set of words will have a greater probability of being similar to another article. 
> 2. We could use the cosine similarity, which measure the angle between vectors and is sensitive to the number of elements in each cosine.
> 3. The Hamming distance which simply sums whether the elements of two sets are identical or not:  
Clearly, this measure will be best if we are primarily looking for matches and mismatches. It is also, like the Jaccard coefficient, sensitive to the number of items in each set, as simply increasing the number of elements increases the possible upper bound of the distance.
> 4. Similar to Hamming is the Manhattan distance, which does not require the data to be binary. If we use the Manhattan distance as an example, we can use MDS again to plot the arrangement of the documents in keyword space.


