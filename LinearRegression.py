import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#visulation with graphs using seaborn
class Visulation:
    def HeatMap(self,data):
        sns.heatmap(data.isnull(),yticklabels=False,cbar=False,cmap='viridis')
        plt.show()
    
    def LRScatterGraph(self,y_test,prediction):
        sns.scatterplot(x=y_test,y=prediction)
        plt.show()
   
#Dataset Anylasis
def Anylasis(data):
    print('\n',data.shape)
    print('\n',data.columns)
    print('\n',data.info())
    print('\n',data.corr())
    print('\n',data.isnull().sum(axis=0))

#Removing Null Values
def NullFilter(data):
    data_before_drop = data.shape[0]
    data = data.dropna()
    data_after_drop = data.shape[0]
    print('\nTotal Rows before drop: ',data_before_drop)
    print('Total Rows after drop: ', data_after_drop)
    return data

#Encoding Categorical Data
def DataEncoding(data):
    from sklearn.preprocessing import LabelEncoder
    lb = LabelEncoder()
    data['Status'] = lb.fit_transform(data['Status'])
    return data

def LinearRegression(X,y):
    #Splitting Dataset into train and test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.6,random_state=77)
    
    #Multiple Linear Regression using scikit-learn library
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(X_train,y_train)

    #Predection
    y_prediction = lr.predict(X_test)

    # accuracy and error
    from sklearn.metrics import mean_squared_error, accuracy_score, mean_absolute_error
    print('\naccuracy(R^2): {}%'.format(lr.score(X_test, y_test)*100))
    print('\nmae: {}'.format(mean_absolute_error(y_test, y_prediction)))
    print('\nrmse: {}'.format(np.sqrt(mean_squared_error(y_test, y_prediction))))

    #Scatter graph to analyse your predicted values
    v1.LRScatterGraph(y_test,y_prediction)

if __name__ == "__main__":
    #Reading Dataset
    dataset = pd.read_csv('LED.csv')
    
    v1 = Visulation()
    
    #Use Heatmap graph to analyle null values
    v1.HeatMap(dataset)

    Anylasis(dataset)
    
    dataset = NullFilter(dataset)

    dataset = DataEncoding(dataset)

    #delecting non numering columns
    dataset = dataset.drop(['Country'], axis=1)
    X = dataset.drop(['Life expectancy '], axis=1).values
    y = dataset['Life expectancy '].values

    LinearRegression(X,y)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#visulation with graphs using seaborn
class Visulation:
    def HeatMap(self,data):
        sns.heatmap(data.isnull(),yticklabels=False,cbar=False,cmap='viridis')
        plt.show()
    
    def LRScatterGraph(self,y_test,prediction):
        sns.scatterplot(x=y_test,y=prediction)
        plt.show()
   
#Dataset Anylasis
def Anylasis(data):
    print('\n',data.shape)
    print('\n',data.columns)
    print('\n',data.info())
    print('\n',data.corr())
    print('\n',data.isnull().sum(axis=0))

#Removing Null Values
def NullFilter(data):
    data_before_drop = data.shape[0]
    data = data.dropna()
    data_after_drop = data.shape[0]
    print('\nTotal Rows before drop: ',data_before_drop)
    print('Total Rows after drop: ', data_after_drop)
    return data

#Encoding Categorical Data
def DataEncoding(data):
    from sklearn.preprocessing import LabelEncoder
    lb = LabelEncoder()
    data['Status'] = lb.fit_transform(data['Status'])
    return data

def LinearRegression(X,y):
    #Splitting Dataset into train and test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.6,random_state=77)
    
    #Multiple Linear Regression using scikit-learn library
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(X_train,y_train)

    #Predection
    y_prediction = lr.predict(X_test)

    # accuracy and error
    from sklearn.metrics import mean_squared_error, accuracy_score, mean_absolute_error
    print('\naccuracy(R^2): {}%'.format(lr.score(X_test, y_test)*100))
    print('\nmae: {}'.format(mean_absolute_error(y_test, y_prediction)))
    print('\nrmse: {}'.format(np.sqrt(mean_squared_error(y_test, y_prediction))))

    #Scatter graph to analyse your predicted values
    v1.LRScatterGraph(y_test,y_prediction)

if __name__ == "__main__":
    #Reading Dataset
    dataset = pd.read_csv('LED.csv')
    
    v1 = Visulation()
    
    #Use Heatmap graph to analyle null values
    v1.HeatMap(dataset)

    Anylasis(dataset)
    
    dataset = NullFilter(dataset)

    dataset = DataEncoding(dataset)

    #delecting non numering columns
    dataset = dataset.drop(['Country'], axis=1)
    X = dataset.drop(['Life expectancy '], axis=1).values
    y = dataset['Life expectancy '].values

    LinearRegression(X,y)
