
# coding: utf-8

# ## Recommendations with Spark ALS

# In[1]:


import pyspark


# In[2]:


sc = pyspark.SparkContext("local[*]")


# In[3]:


sc.version


# In[4]:


from pyspark.mllib.recommendation import ALS
from pyspark.mllib.recommendation import Rating


# In[5]:


def expand_user(a, user):
    return [Rating(user, item, ranking) for item, ranking in enumerate(a) if ranking != 0]


# In[6]:


def expand_all(a):
    return [expand_user(items, user) for user, items in enumerate(a)]


# ### Here we have ratings from eight users for six different movies: Titanic, Dirty Dancing, Die Hard, Terminator 2, Wayne's World, and Zoolander. Or in other words, two romantic films, two action films, and two comedies. Each row is a user, each column is a movie.
# 
# ### The ratings are constructed so that if a user has seen both movies in one of these pairs, their ratings for the two movies are similar.
# 
# ### There is no evidence in this data that anyone likes all three film genres.

# In[7]:


rawdata = [
    [5,5,0,0,0,0],
    [0,0,5,5,0,0],
    [0,0,0,0,5,5],
    [0,1,5,5,5,0],
    [1,1,5,0,5,5],
    [5,5,0,5,1,1],
    [5,0,0,5,0,1],
    [5,5,5,0,1,0]
    ]
list_of_ratings = expand_all(rawdata)


# In[8]:


# construct an RDD of Ratings for every non-zero rating
ratings = [val for sublist in list_of_ratings for val in sublist]
ratingsRDD = sc.parallelize(ratings)
ratingsRDD.take(5)


# In[9]:


rank = 3
numIterations = 5
als_lambda = 0.1
model = ALS.train(ratingsRDD, rank, numIterations, als_lambda, seed=4242, nonnegative=True)
# there is also a trainImplicit method that one uses when
# working with implicit ratings (it uses a different cost function)


# In[10]:


# here we see the model's vector of features for each user
users = model.userFeatures().collect()
sorted(users, key=lambda x: x[0])


# In[11]:


# and the features for the "products"
products = model.productFeatures().collect()
sorted(products, key=lambda x: x[0])


# In[12]:


# recommend 3 items for user 2
model.recommendProducts(2, 3)


# ### Display the original matrix side-by-side with the reconstructed matrix. The values that were originally non-zero should be closely approximated, and the values that were zero (empty) now have predictions.

# In[13]:


import sys
print(" original      reconstructed")
for user in range(0, len(rawdata)):
    for product in range (0, len(rawdata[0])):
        sys.stdout.write("%d " % rawdata[user][product])
    sys.stdout.write("    ")
    for product in range (0, len(rawdata[0])):
        sys.stdout.write("%0.0f " % model.predict(user, product))
    print(" ")


# In[14]:


print(" original         errors        predictions")
for user in range(0, len(rawdata)):
    for product in range (0, len(rawdata[0])):
        sys.stdout.write("%d " % rawdata[user][product])
    sys.stdout.write("    ")
    for product in range (0, len(rawdata[0])):
        if rawdata[user][product] != 0:
            prediction = model.predict(user, product)
            if rawdata[user][product] != round(prediction, 0):
                sys.stdout.write("%0.0f " % prediction)
            else:
                sys.stdout.write("- ")
        else:
            sys.stdout.write("- ")
    sys.stdout.write("    ")
    for product in range (0, len(rawdata[0])):
        if rawdata[user][product] == 0:
            prediction = model.predict(user, product)
            sys.stdout.write("%0.0f " % prediction)
        else:
            sys.stdout.write("- ")
    print(" ")


# ### Compute the mean squared error of the reconstructed matrix. This can be used to decide if the rank is sufficiently large.

# In[15]:


evalRDD = ratingsRDD.map(lambda p: (p[0], p[1]))
evalRDD.take(5)


# In[16]:


predictions = model.predictAll(evalRDD).map(lambda r: ((r[0], r[1]), r[2]))
predictions.take(5)


# In[17]:


ratingsAndPreds = ratingsRDD.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
ratingsAndPreds.take(5)


# In[ ]:


ratingsAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()


# With a larger dataset we would separate the rating data into training and test sets, and see how well our predicted ratings match the actual data.

