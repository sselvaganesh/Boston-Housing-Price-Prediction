
# coding: utf-8

# In[2]:



# coding: utf-8

# In[53]:


from sklearn.cross_validation import KFold
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, SGDRegressor
import numpy as np
import pylab as pl


# In[54]:


from sklearn.datasets import load_boston
boston = load_boston()


# In[55]:


print(boston.feature_names)


# In[56]:


print(boston.data.shape)
print(boston.target.shape)


# In[57]:


np.set_printoptions(precision=2, linewidth=120, suppress=True, edgeitems=4)


# In[58]:


print(boston.data)


# In[59]:


# In order to do multiple regression we need to add a column of 1s for x0
x = np.array([np.concatenate((v,[1])) for v in boston.data])
y = boston.target


# In[60]:


# First 10 elements of the data
print(x[:10])


# In[61]:


# First 10 elements of the response variable
print(y[:10])


# In[62]:


linreg=LinearRegression()
linreg.fit(x,y)


# In[63]:


# Let's see predictions for the first 10 instances
print(linreg.predict(x[:10]))


# In[64]:


# Compute RMSE on training data
# p = np.array([linreg.predict(xi) for xi in x])
p = linreg.predict(x)
# Now we can constuct a vector of errors
err = abs(p-y)

# Let's see the error on the first 10 predictions
print(err[:10])


# In[65]:


# Dot product of error vector with itself gives us the sum of squared errors
total_error = np.dot(err,err)
# Compute RMSE
rmse_train = np.sqrt(total_error/len(p))
print(rmse_train)


# In[66]:


# We can view the regression coefficients
print ('Regression Coefficients: \n', linreg.coef_)


# In[67]:


# Plot outputs
get_ipython().run_line_magic('matplotlib', 'inline')
pl.plot(p, y,'ro')
pl.plot([0,50],[0,50], 'g-')
pl.xlabel('predicted')
pl.ylabel('real')
pl.show()


# In[68]:


# Now let's compute RMSE using 10-fold x-validation
kf = KFold(len(x), n_folds=10)
xval_err = 0
for train,test in kf:
    linreg.fit(x[train],y[train])
    # p = np.array([linreg.predict(xi) for xi in x[test]])
    p = linreg.predict(x[test])
    e = p-y[test]
    xval_err += np.dot(e,e)
    
rmse_10cv = np.sqrt(xval_err/len(x))


# In[69]:


method_name = 'Simple Linear Regression'
print('Method: %s' %method_name)
print('RMSE on training: %.4f' %rmse_train)
print('RMSE on 10-fold CV: %.4f' %rmse_10cv)


# In[70]:


# Create linear regression object with a ridge coefficient 0.5
ridge = Ridge(fit_intercept=True, alpha=0.5)

# Train the model using the training set
ridge.fit(x,y)


# In[71]:


# Compute RMSE on training data
# p = np.array([ridge.predict(xi) for xi in x])
p = ridge.predict(x)
err = p-y
total_error = np.dot(err,err)
rmse_train = np.sqrt(total_error/len(p))

# Compute RMSE using 10-fold x-validation
kf = KFold(len(x), n_folds=10)
xval_err = 0
for train,test in kf:
    ridge.fit(x[train],y[train])
    p = ridge.predict(x[test])
    e = p-y[test]
    xval_err += np.dot(e,e)
rmse_10cv = np.sqrt(xval_err/len(x))

method_name = 'Ridge Regression'
print('Method: %s' %method_name)
print('RMSE on training: %.4f' %rmse_train)
print('RMSE on 10-fold CV: %.4f' %rmse_10cv)


# In[72]:


print('Ridge Regression')
print('alpha\t RMSE_train\t RMSE_10cv\n')
alpha = np.linspace(.01,20,50)
t_rmse=np.array([])
cv_rmse=np.array([])

for a in alpha:
    ridge=Ridge(fit_intercept=True, alpha=a)
    # computing the RMSE on training data
    ridge.fit(x,y)
    p=ridge.predict(x)
    error=p-y
    total_error=np.dot(err,err)
    rmse_train=np.sqrt(total_error/len(p))
    # Computing rmse using 10 fold cross validation
    
    kf = KFold(len(x), n_folds=10)
    xval_err=0
    for train,test in kf:
        ridge.fit(x[train], y[train])
        p=ridge.predict(x[test])
        err=p-y[test]
        xval_err+=np.dot(err,err)
    rmse_10cv=np.sqrt(xval_err/len(x))
    
    t_rmse=np.append(t_rmse,[rmse_train])
    cv_rmse=np.append(cv_rmse,[rmse_10cv])
    print('{:.3f}\t {:.4f}\t\t {:.4f}'.format(a,rmse_train,rmse_10cv))


# In[73]:


pl.plot(alpha, t_rmse, label='RMSE_Train')
pl.plot(alpha, cv_rmse, label='RMSE_XVal')
pl.legend(('RMSE-TRAIN', 'RMSE_XVal'))
pl.ylabel('RMSE')
pl.xlabel('Alpha')
pl.show()


# In[84]:


a=0.3
for name,met in [
    ('linear regression', LinearRegression()),
    ('lasso', Lasso(fit_intercept=True, alpha=a)),
    ('ridge', Ridge(fit_intercept=True, alpha=a)),
    ('elastic-net', ElasticNet(fit_intercept=True, alpha=a))
    ]:
    met.fit(x,y)
    p=met.predict(x)
    e=p-y
    total_error=np.dot(e,e)
    rmse_train=np.sqrt(total_error/len(p))
    #print(p)
    #print(y)
    print(y.shape)
    print(p.shape)
    pl.scatter(p,y)
    #pl.plot(p, y,'ro')
    pl.plot([0,50],[0,50], 'g-')
    pl.xlabel('predicted')
    pl.ylabel('real')
    pl.show()
    kf=KFold(len(x), n_folds=10)
    err=0
    for train,test in kf:
        met.fit(x[train],y[train])
        p=met.predict(x[test])
        e=p-y[test]
        err+=np.dot(e,e)
        #print(err)
        
        
    rmse_10cv=np.sqrt(err/len(x))
    print('Method: %s' %name)
    print('RMSE on training: %.4f' %rmse_train)
    print('RMSE on 10-fold CV: %.4f' %rmse_10cv)
    print('\n')
    
a=0.3
for name,met in [
    ('linear regression', LinearRegression()),
    ('lasso', Lasso(fit_intercept=True, alpha=a)),
    ('ridge', Ridge(fit_intercept=True, alpha=a)),
    ('elastic-net', ElasticNet(fit_intercept=True, alpha=a))
    ]:
    met.fit(x,y)
    p=met.predict(x)
    e=p-y
    total_error=np.dot(e,e)
    rmse_train=np.sqrt(total_error/len(p))
    #print(p)
    #print(y)
    print(y.shape)
    print(p.shape)
    pl.scatter(p,y)
    #pl.plot(p, y,'ro')
    pl.plot([0,50],[0,50], 'g-')
    pl.xlabel('predicted')
    pl.ylabel('real')
    #pl.show()
    kf=KFold(len(x), n_folds=10)
    err=0
    for train,test in kf:
        met.fit(x[train],y[train])
        p=met.predict(x[test])
        e=p-y[test]
        err+=np.dot(e,e)
        #print(err)
        
        
    rmse_10cv=np.sqrt(err/len(x))
    print('Method: %s' %name)
    print('RMSE on training: %.4f' %rmse_train)
    print('RMSE on 10-fold CV: %.4f' %rmse_10cv)
    print('\n')
  
  

