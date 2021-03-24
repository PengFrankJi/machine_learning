#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import statsmodels.api as sm
import pylab as pl
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc  ###计算roc和auc


# In[2]:


# read the excel file
file_path = "/Users/jipeng/Documents/Study/Study_myself/Logistic_Regression/data.xlsx"
file = pd.ExcelFile(file_path)
data = file.parse("data")


# Variables unrelated to the model：id, member_id,  
# Variables having only one value: term  
# Variables have repeated meaning: label: loan_status, \[home_mort, home_own\]: home_ownership, 

# In[3]:


columns_to_keep = ["label", "loan_amnt", "issue_d", "int_rate", "installment", "grade", "emp_length", "home_mort",                    "home_own", "FICO", "annual_inc", "loantoincome", "dti", "delinq_2yrs", "inq_last_6mths",                    "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc", "tot_cur_bal", "total_rev_hi_lim",                    "acc_open_past_24mths", "avg_cur_bal", "bc_open_to_buy", "bc_util", "mo_sin_old_il_acct",                    "mo_sin_old_rev_tl_op", "mo_sin_rcnt_rev_tl_op", "mo_sin_rcnt_tl", "mort_acc",                    "mths_since_recent_bc", "num_accts_ever_120_pd", "num_actv_bc_tl", "num_actv_rev_tl", "num_bc_sats",                    "num_bc_tl", "num_il_tl", "num_op_rev_tl", "num_rev_accts", "num_rev_tl_bal_gt_0", "num_sats",                    "num_tl_op_past_12m", "pct_tl_nvr_dlq", "percent_bc_gt_75", "tot_hi_cred_lim", "total_bal_ex_mort",                    "total_bc_limit", "total_il_high_credit_limit"]
data = data[columns_to_keep]


# In[4]:


def convert_date_to_month(x):
    if x == np.datetime64('2020-01-14'):
        return 1
    elif x == np.datetime64('2020-02-14'):
        return 2
    elif x == np.datetime64('2020-03-14'):
        return 3
    elif x == np.datetime64('2020-04-14'):
        return 4
    elif x == np.datetime64('2020-05-14'):
        return 5
    elif x == np.datetime64('2020-06-14'):
        return 6
    elif x == np.datetime64('2020-07-14'):
        return 7
    elif x == np.datetime64('2020-08-14'):
        return 8

data["issue_d"] = [convert_date_to_month(i) for i in data['issue_d']]


# In[5]:


data = pd.concat([data, pd.get_dummies(data['grade'], prefix = 'grade').iloc[:, 1:]], axis = 1)
data = pd.concat([data, pd.get_dummies(data['emp_length'], prefix = 'emp_length').iloc[:, 1:]], axis = 1)
data = data.drop(['grade', 'emp_length'], axis = 1)


# In[6]:


data.head(30)


# In[7]:


# check how many NA's
print(np.where(np.isnan(data)))
print(data.columns[24])
print(np.where(np.isnan(data))[0].shape)

# there are 3075 NA's. We can delete observations containing NA
data = data.dropna()


# In[8]:


f = lambda x: (x - x.min()) / (x.max()-x.min())
data = data.apply(f)


# In[9]:


y = data['label']
x = data.iloc[:, 1:]
x['intercept'] = 1.0

np.random.seed(seed = 9)
row = x.shape[0]
shuffle_indexes = np.random.permutation(row)
train_ratio = 0.7
train_size = int(row * train_ratio)
train_x = x.iloc[shuffle_indexes[0: train_size], :]
test_x = x.iloc[shuffle_indexes[train_size: ], :]
train_y = y.iloc[shuffle_indexes[0: train_size]]
test_y = y.iloc[shuffle_indexes[train_size: ]]


# In[10]:


model1 = sm.Logit(train_y, train_x).fit()
model1.summary()


# According to the P-values, we pick the variables that are significant.

# In[11]:


columns_to_keep = ["issue_d", "int_rate", "home_mort", "home_own", "FICO", "loantoincome", "dti", "delinq_2yrs",                    "inq_last_6mths", "pub_rec", "acc_open_past_24mths", "mo_sin_old_il_acct", "mort_acc",                    "mths_since_recent_bc", "num_rev_accts", "num_rev_tl_bal_gt_0", "pct_tl_nvr_dlq",                    "percent_bc_gt_75", "total_bal_ex_mort", "total_il_high_credit_limit", "grade_B", "grade_C",                    "grade_D", "grade_E", "grade_F", "grade_G", "emp_length_10+ years", "emp_length_2 years",                    "emp_length_3 years", "emp_length_4 years", "emp_length_5 years", "emp_length_6 years",                   "emp_length_7 years", "emp_length_8 years", "emp_length_9 years", "emp_length_< 1 year", "intercept"]
train_x = train_x[columns_to_keep]
test_x = test_x[columns_to_keep]
model2 = sm.Logit(train_y, train_x).fit()
model2.summary()


# In[12]:


predict_prob = model2.predict(test_x)
threshold_position = sum(test_y == 1) 
threshold = sorted(predict_prob, reverse=True)[threshold_position]
predict_y = predict_prob.apply(lambda x: 1 if x > threshold else 0)

test_y_list = test_y.values.tolist()
predict_y_list = predict_y.values.tolist()

tp = sum((predict_y == 1).values.tolist() and (test_y == 1).values.tolist()) # True Positive
fn = sum((predict_y == 0).values.tolist() and (test_y == 1).values.tolist()) # False Negative
fp = sum((predict_y == 1).values.tolist() and (test_y == 0).values.tolist()) # False Positive
tn = sum((predict_y == 0).values.tolist() and (test_y == 0).values.tolist()) # True Negative

matrix_of_confusion = pd.DataFrame([[tp, fp, tp + fp],
                                    [fn, tn, fn + tn],
                                    [tp + fn, fp + tn, tp + fp + fn + tn]],
                                  columns = ["actual good", "actual bad", "total"], 
                                  index = ["predicted good", "predicted bad", "total"])

matrix_of_confusion


# In[13]:


f,ax=plt.subplots()

conf_matrix = confusion_matrix(test_y, predict_y)
print(conf_matrix) #打印出来看看

sns.heatmap(conf_matrix, annot=True, ax=ax) #画热力图
ax.set_title('confusion matrix') #标题
ax.set_xlabel('predict') #x轴
ax.set_ylabel('actual') #y轴


# In[14]:


fpr, tpr, thresholds  =  roc_curve(test_y, predict_prob)
roc_auc =auc(fpr, tpr) 

plt.figure()
lw = 2
plt.figure(figsize=(10,10))
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

print(roc_auc)


# In[ ]:




