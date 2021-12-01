#tryout if these make better vectors
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
#tryout ensembling
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
#tryout clustering
from sklearn.cluster import KMeans
#tryout machine learning algorithm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
#display result
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

import time
import pandas as pd
import numpy as np
from MyParser import MyParser #my python file
import ml_algorithm #my python file

#test module SVC
#returns prediction


#train module SVC
#returns fitted model

#combined module (class)
#that as soon as it's given path
#it runs MyParser
#trains model
#and has testing function 
bs=100 #fixed batch size for training
tp=0.2 #fixed test set size for notifying trained model result
class SVC_result:
    def __init__(self,trainpath,testpath):
        self.trainpath=trainpath
        self.testpath=testpath
        self.parser=None
    def train(self):
        if not self.trainpath:
            print("\nPath not appropriately set")
            return -1
        self.parser=MyParser(self.trainpath)
        self.parser.parse()
        self.parser.rebatch(bs)
        
        testx,testy,trainx,trainy=self.parser.set_splitter(tp)
        #give train result with little test
        
        #model
        tsneSVC,pcaSVC=try_SVC(trainx,trainy)
        #test
        predtsne=tsneSVC.predict(testx)
        predpca=pcaSVC.predict(testx)
        #accuracy
        acctsne=accuracy_score(testy,predtsne)
        accpca=accuracy_score(testy,predpca)
        
        #notification
        print("\n|||Training SVC models completed")
        print("\n|||accuracy of trained SVC models")
        print("|||\ttsne->SVC:",acctsne)
        print("|||\tpca->SVC:",accpca)
        
        return tsneSVC,pcaSVC
    
    def test(self,model):
        files=[]
        for filename in listdir(self.testpath):
            files.append(self.parser.parse_single(filename)) #MyDocument object
        
        #expected only one file
        pred=[]
        for file in files:
            pred.append(model.predict(file.X))
        
        return pred
    
    def run(self):
        tsneSVC,pcaSVC=self.train()
        predtsne=self.test(tsneSVC)
        predpca=self.test(pcaSVC)
        
        return predtsne,predpca
        