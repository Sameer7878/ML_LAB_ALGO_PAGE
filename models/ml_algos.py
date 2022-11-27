
import random

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn import metrics
from sklearn.cluster import KMeans
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination


class ML_ALGOS:

    def __init__(self,data):#initializing the data
        self.data=data

    def FIND_S(self): #FIND_S ALGO
        res={}
        data=self.data.values
        msh=[0]*(len(data[0])-1)
        for row in data:
            if row[-1]=='n' or row[-1]=='no':
                continue
            for i in range(len(row)-1):
                if msh[i]==0:
                    msh[i]=row[i]
                elif msh[i]==row[i]:
                    continue
                else:
                    msh[i]='?'
        res['Most Specific Hypothesis of the given data: ']=msh
        return res

    def CANDIDATE_ALGO(self): #candidate elimination algo
        res={}
        concepts=np.array(self.data.iloc [:, 0:-1])
        res['concepts']=concepts
        target=np.array(self.data.iloc [:, -1])
        res['target']=target

        def learn(concepts, target):
            specific_h=concepts [0].copy()
            general_h=[["?" for i in range(len(specific_h))] for i in range(len(specific_h))]
            res ["initialization of specific_h and general_h"]=[specific_h,general_h]

            for i, h in enumerate(concepts):
                #print("For Loop Starts")
                if target[i] == "yes":
                    #print("If instance is Positive ")
                    for x in range(len(specific_h)):
                        if h [x] != specific_h [x]:
                            specific_h [x]='?'
                            general_h [x] [x]='?'

                if target [i] == "no":
                    #print("If instance is Negative ")
                    for x in range(len(specific_h)):
                        if h [x] != specific_h [x]:
                            general_h [x] [x]=specific_h [x]
                        else:
                            general_h [x] [x]='?'

                res["steps of Candidate Elimination Algorithm"]= [i+1,specific_h,general_h]

            indices=[i for i, val in enumerate(general_h) if val == ['?', '?', '?', '?', '?', '?']]
            for i in indices:
                general_h.remove(['?', '?', '?', '?', '?', '?'])
            return specific_h, general_h

        s_final, g_final=learn(concepts, target)
        res["Final Specific_h:"]=s_final
        res["Final General_h:"]=g_final
        return res
    def BackPropagation(self):
        self.PreProcessing()
        res={}
        X=np.array(self.data [self.data.columns.values [:-1]].values, dtype=float)  # two inputs [sleep,study]
        y=np.array(self.data [self.data.columns.values [-1]].values, dtype=float)
        # one output [Expected % in Exams]
        X=X / np.amax(X, axis=0)  # maximum of X array longitudinally
        y=y / 100
        y=[[x] for x in y]

        # Sigmoid Function
        def sigmoid(x):
            return 1 / (1+np.exp(-x))

        # Derivative of Sigmoid Function
        def derivatives_sigmoid(x):
            return x * (1-x)

        # Variable initialization
        epoch=5000  # Setting training iterations
        lr=0.2  # Setting learning rate
        inputlayer_neurons= len(self.data.columns.values [:-1]) # number of features in data set
        hiddenlayer_neurons=3  # number of hidden layers neurons
        output_neurons=1  # number of neurons at output layer

        # weight and bias initialization
        wh=np.random.uniform(
            size=(inputlayer_neurons, hiddenlayer_neurons))  # weight of the link from input node to hidden node
        bh=np.random.uniform(size=(1, hiddenlayer_neurons))  # bias of the link from input node to hidden node
        wout=np.random.uniform(
            size=(hiddenlayer_neurons, output_neurons))  # weight of the link from hidden node to output node
        bout=np.random.uniform(size=(1, output_neurons))  # bias of the link from hidden node to output node

        # draws a random range of numbers uniformly of dim x*y
        for i in range(epoch):
            # Forward Propogation
            hinp1=np.dot(X, wh)
            hinp=hinp1+bh
            hlayer_act=sigmoid(hinp)
            outinp1=np.dot(hlayer_act, wout)
            outinp=outinp1+bout
            output=sigmoid(outinp)
            # Backpropagation
            EO=y-output
            outgrad=derivatives_sigmoid(output)
            d_output=EO * outgrad

            EH=d_output.dot(wout.T)

            # how much hidden layer weights contributed to error
            hiddengrad=derivatives_sigmoid(hlayer_act)
            d_hiddenlayer=EH * hiddengrad

            # dotproduct of nextlayererror and currentlayerop
            wout+=hlayer_act.T.dot(d_output) * lr
            wh+=X.T.dot(d_hiddenlayer) * lr

        res["Input: "]=X
        res["Actual Output: "]=y
        res["Predicted Output: "]=output
        return res

    def PreProcessing(self):
        le=preprocessing.LabelEncoder()
        for i in list(self.data):
            self.data [i]=le.fit_transform(self.data[i].values)
    def ID3(self):
        self.PreProcessing()
        res={}
        clf=DecisionTreeClassifier(random_state=0)
        x=self.data [self.data.columns.values [:-1]].values
        y=self.data [self.data.columns.values [-1]].values
        x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.3)
        clf.fit(x_train, y_train)
        fig, axes=plt.subplots(nrows=1, ncols=1, figsize=(4, 4), dpi=300)
        y_pred=clf.predict(x_test)
        tree.plot_tree(clf,
                       feature_names=self.data.columns.values [:-1],
                       class_names=self.data.columns.values [-1],
                       filled=True)
        res['Confusion matrix']=metrics.confusion_matrix(y_test,y_pred)
        res['Accuracy of the classifier is']=metrics.accuracy_score(y_test,y_pred)
        fig.savefig('static/img/DTP.png')
        res['path']='static/img/DTP.png'
        return res
    def naive_bayes_GNB(self):
        self.PreProcessing()
        res={}
        # splitting the dataset into train and test data
        X=self.data [self.data.columns.values [:-1]].values
        y=self.data [self.data.columns.values [-1]].values
        xtrain, xtest, ytrain, ytest=train_test_split(X, y, test_size=0.33)

        res['the total number of Training Data :']=ytrain.shape
        res['the total number of Test Data :']=ytest.shape

        # Training Naive Bayes (NB) classifier on training data.

        clf=GaussianNB().fit(xtrain, ytrain.ravel())
        predicted=clf.predict(xtest)
        # printing Confusion matrix, accuracy, Precision and Recall
        res['Confusion matrix']=metrics.confusion_matrix(ytest, predicted)
        res['Accuracy of the classifier is']= metrics.accuracy_score(ytest, predicted)

        res['The value of Precision']=metrics.precision_score(ytest, predicted)

        res['The value of Recall']=metrics.recall_score(ytest, predicted)
        return res
    def naive_bayes_MNB(self):
        self.PreProcessing()
        res={}
        X=self.data [self.data.columns.values [:-1]].values
        y=self.data [self.data.columns.values [-1]].values
        xtrain, xtest, ytrain, ytest=train_test_split(X, y, test_size=0.33)

        res ['the total number of Training Data :']=ytrain.shape
        res ['the total number of Test Data :']=ytest.shape

        # Training Naive Bayes (NB) classifier on training data.

        clf=MultinomialNB().fit(xtrain, ytrain.ravel())
        predicted=clf.predict(xtest)
        # printing Confusion matrix, accuracy, Precision and Recall
        res ['Confusion matrix']=metrics.confusion_matrix(ytest, predicted)
        res ['Accuracy of the classifier is']=metrics.accuracy_score(ytest, predicted)

        res ['The value of Precision']=metrics.precision_score(ytest, predicted)

        res ['The value of Recall']=metrics.recall_score(ytest, predicted)
        return res
    def naive_bayes_BNB(self):
        self.PreProcessing()
        res={}
        X=self.data [self.data.columns.values [:-1]].values
        y=self.data [self.data.columns.values [-1]].values
        xtrain, xtest, ytrain, ytest=train_test_split(X, y, test_size=0.33)

        res ['the total number of Training Data :']=ytrain.shape
        res ['the total number of Test Data :']=ytest.shape

        # Training Naive Bayes (NB) classifier on training data.

        clf=BernoulliNB().fit(xtrain, ytrain.ravel())
        predicted=clf.predict(xtest)
        # printing Confusion matrix, accuracy, Precision and Recall
        res ['Confusion matrix']=metrics.confusion_matrix(ytest, predicted)
        res ['Accuracy of the classifier is']=metrics.accuracy_score(ytest, predicted)

        res ['The value of Precision']=metrics.precision_score(ytest, predicted)

        res ['The value of Recall']=metrics.recall_score(ytest, predicted)
        return res
    def bayesian_network(self):
        self.PreProcessing()
        res={}
        dataset=self.data
        dataset=dataset.replace('?', np.nan)
        res['Sample instances from the dataset are given below']=dataset.head()

        res['Attributes and datatypes']=dataset.dtypes
        train_li=[(x,self.data.columns.values [-1]) for x in self.data.columns.values [:-1] ]
        model=BayesianNetwork(train_li)
        res['Learning CPD using Maximum likelihood estimators']=''
        model.fit(dataset, estimator=MaximumLikelihoodEstimator)

        res['Inferencing with Bayesian Network:']=''
        HeartDiseasetest_infer=VariableElimination(model)
        li=list(self.data.columns.values [:-1])
        a=random.choice(li)
        li.remove(a)
        b=random.choice(li)
        q1=HeartDiseasetest_infer.query(variables=[self.data.columns.values [-1]], evidence={a: 1})
        res [f'1. Probability of {self.data.columns.values [-1]} given evidence= {a}']=q1

        q2=HeartDiseasetest_infer.query(variables=[self.data.columns.values [-1]], evidence={b: 2})
        res [f'2. Probability of {self.data.columns.values [-1]} given evidence= {b}']=q2
        return res
    def EM_GMM(self):
        self.PreProcessing()
        res={}
        X=self.data [self.data.columns.values [:-1]]
        y=self.data [self.data.columns.values [-1]]
        X.columns=self.data.columns.values [:-1]
        y.columns=self.data.columns.values [-1]
        scaler=preprocessing.StandardScaler()
        scaler.fit(X)
        xsa=scaler.transform(X)
        xs=pd.DataFrame(xsa, columns=X.columns)
        # xs.sample(5)

        from sklearn.mixture import GaussianMixture
        gmm=GaussianMixture(n_components=3)
        gmm.fit(xs)

        y_gmm=gmm.predict(xs)
        # y_cluster_gmm

        res['The accuracy score of EM: ']=metrics.accuracy_score(y, y_gmm)
        res['The Confusion matrix of EM: ']=metrics.confusion_matrix(y, y_gmm)
        return res
    def K_Means(self):
        self.PreProcessing()
        res={}
        X=self.data [self.data.columns.values [:-1]]
        y=self.data [self.data.columns.values [-1]]
        X.columns=self.data.columns.values [:-1]
        y.columns=self.data.columns.values [-1]
        model=KMeans(n_clusters=3)
        model.fit(X)
        """
        plt.figure(figsize=(14, 7))

        colormap=np.array(['red', 'lime', 'black'])

        # Plot the Original Classifications
        li=list(X.columns)
        a=random.choice(li)
        li.remove(a)
        b=random.choice(li)
        print(a,b)
        plt.subplot(1, 2, 1)
        plt.scatter(X[a], X[b], c=colormap [y[0]], s=40)
        plt.title('Real Classification')
        plt.xlabel('Petal Length')
        plt.ylabel('Petal Width')
        plt.show()
        # Plot the Models Classifications
        plt.subplot(1, 2, 2)
        plt.scatter(X[a], X[b], c=colormap [model.labels_], s=40)
        plt.title('K Mean Classification')
        plt.xlabel('Petal Length')
        plt.ylabel('Petal Width')
        plt.show()"""
        res['The accuracy score of K-Mean: ']= metrics.accuracy_score(y, model.labels_)
        res['The Confusion matrixof K-Mean: ']= metrics.confusion_matrix(y, model.labels_)
        return res

