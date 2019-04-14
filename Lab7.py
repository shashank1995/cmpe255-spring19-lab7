
# coding: utf-8

# In[21]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[22]:


def linear_svm():
    # download data set: https://drive.google.com/file/d/13nw-uRXPY8XIZQxKRNZ3yYlho-CYm_Qt/view
    # info: https://archive.ics.uci.edu/ml/datasets/banknote+authentication

    # load data
    bankdata = pd.read_csv("./bill_authentication.csv")  

    # see the data
    bankdata.shape  

    # see head
    bankdata.head()  

    # data processing
    X = bankdata.drop('Class', axis=1)  
    y = bankdata['Class']  

    from sklearn.model_selection import train_test_split  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)  

    # train the SVM
    from sklearn.svm import SVC  
    svclassifier = SVC(kernel='linear')  
    svclassifier.fit(X_train, y_train)  

    # predictions
    y_pred = svclassifier.predict(X_test)  

    # Evaluate model
    from sklearn.metrics import classification_report, confusion_matrix
    print("--------------------Linear kernel--------------------------")
    print(confusion_matrix(y_test,y_pred))  
    print(classification_report(y_test,y_pred))
    
linear_svm()


# In[23]:


# Iris dataset  https://archive.ics.uci.edu/ml/datasets/iris4
def import_iris():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

    # Assign colum names to the dataset
    colnames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

    # Read dataset to pandas dataframe
    irisdata = pd.read_csv(url, names=colnames) 
    
    # process
    X = irisdata.drop('Class', axis=1)  
    y = irisdata['Class']  

    # train
    from sklearn.model_selection import train_test_split  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
    return  X_train, X_test, y_train, y_test


# In[24]:


def polynomial_kernel(X_train, X_test, y_train, y_test):
    # TODO
    # NOTE: use 8-degree in the degree hyperparameter. 
    # Trains, predicts and evaluates the model
    from sklearn.svm import SVC  
    svclassifier = SVC(kernel='poly', degree=8)  
    svclassifier.fit(X_train, y_train)
    y_pred = svclassifier.predict(X_test)
    from sklearn.metrics import classification_report, confusion_matrix
    print("------------------------Polynomial Kernel data----------------------")
    print(confusion_matrix(y_test, y_pred))  
    print(classification_report(y_test, y_pred)) 

def gaussian_kernel(X_train, X_test, y_train, y_test):
    # TODO
    # Trains, predicts and evaluates the model
    from sklearn.svm import SVC  
    svclassifier = SVC(kernel='rbf')  
    svclassifier.fit(X_train, y_train)
    y_pred = svclassifier.predict(X_test)
    from sklearn.metrics import classification_report, confusion_matrix  
    print("-----------------------Gaussian Kernel data------------------------")
    print(confusion_matrix(y_test, y_pred))  
    print(classification_report(y_test, y_pred))

def sigmoid_kernel(X_train, X_test, y_train, y_test):
    # TODO
    # Trains, predicts and evaluates the model
    from sklearn.svm import SVC  
    svclassifier = SVC(kernel='sigmoid')  
    svclassifier.fit(X_train, y_train)
    y_pred = svclassifier.predict(X_test)
    from sklearn.metrics import classification_report, confusion_matrix  
    print("-----------------------Sigmoid Kernel data------------------------")
    print(confusion_matrix(y_test, y_pred))  
    print(classification_report(y_test, y_pred))

def test():
    X_train, X_test, y_train, y_test = import_iris()
    polynomial_kernel(X_train, X_test, y_train, y_test)
    gaussian_kernel(X_train, X_test, y_train, y_test)
    sigmoid_kernel(X_train, X_test, y_train, y_test)

test()


# In[25]:


from sklearn import svm, datasets
def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target


C = 1.0  
models = (svm.SVC(kernel='linear', C=C),
          svm.LinearSVC(C=C),
          svm.SVC(kernel='rbf', gamma=0.7, C=C),
          svm.SVC(kernel='poly', degree=3, C=C))
models = (clf.fit(X, y) for clf in models)

titles = ('Linear SVC kernel',
          'LinearSVC (linear kernel)',
          'Gaussian SVC kernel',
          'Polynomial SVC (degree 3) kernel')

fig, sub = plt.subplots(2, 2)
plt.subplots_adjust(wspace=0.8, hspace=0.8)

X0, X1 = X[:, 0], X[:, 1]
xx, yy = make_meshgrid(X0, X1)

for clf, title, ax in zip(models, titles, sub.flatten()):
    plot_contours(ax, clf, xx, yy,
                  cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('Length')
    ax.set_ylabel('Width')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)

plt.show()

