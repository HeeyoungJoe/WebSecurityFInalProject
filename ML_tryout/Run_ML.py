from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from MyParser import * #my python file
from ml_algorithm import try_SVC,try_simple_SVC
from os import listdir
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
        

    def run(self):
        if not self.trainpath:
            print("\nPath not appropriately set")
            return -1
        
        #train parser
        self.parser=MyParser(self.trainpath)
        self.parser.parse()
        #parser now has multiple different data
        #they are not combine-able because they have
        #different number of features
        
        #test parser
        test_parser=MyParser(self.testpath)
        test_parser.parse()
        
        ttr,ttc=self.parser.batch_data.shape
        tr,tc=test_parser.batch_data.shape
        
        #set target's sizes
        self.parser.batch_target=np.reshape(self.parser.batch_target,(self.parser.batch_target.size,1))
        test_parser.batch_target=np.reshape(test_parser.batch_target,(test_parser.batch_target.size,1))
        
        #attach 
        x=np.concatenate((self.parser.batch_data,test_parser.batch_data),axis=0)
        y=np.concatenate((self.parser.batch_target,test_parser.batch_target),axis=0)
        
        #야발 여기서 하면 안되는데...
        plane=ttr
        pred_tsneSVC,pred_pcaSVC=try_simple_SVC(x,y,plane)
        
        """ '''
        #문제 발생! tsne가 train data와 test data 둘 다 함게
        #붙인 후에 실행시켜야 함. 
        datasets=self.parser.prepare_data(tp)
        
            testx,testy,trainx,trainy=dataset
            tsneSVC,pcaSVC=try_SVC(trainx,trainy)
            #test
            predtsne=tsneSVC.predict(testx)
            predpca=pcaSVC.predict(testx)
            #accuracy
            train_accuracy['tsne'].append(accuracy_score(testy,predtsne))
            train_accuracy['pca'].append(accuracy_score(testy,predpca))
        ''' """
        #don't take out accuracy or display of accuracy as separate module
        #I don't have any spice to give
        
        print("\n|||Training complete!")
        print("\n|||Trained model is validated to have accuracy of...")
        print("\n|||File[%d]:accuracy for tsne\t%f"%(i,accuracy_score(pred_tsneSVC,testy)))
        print("\n|||File[%d]:accuracy for pca\t%f"%(i,accuracy_score(pred_pcaSVC,testy)))
        
        '''
        for i in range(len(train_accuracy)):#need work, check if len is valid for dict
            print("\n|||File[%d]:accuracy for tsne\t%f"%(i,train_accuracy['tsne'][i]))
            print("\n|||File[%d]:accuracy for pca\t%f"%(i,train_accuracy['pca'][i]))
        '''
        #return tsneSVC,pcaSVC
    '''
    def test(self,model):
        #need work
        files=[] #2D representation of csv data on input
        for filename in listdir(self.testpath):
            files.append(self.parser.parse_single(filename)) #MyDocument object
        
        #expected only one file
        pred=[]
        for file in files:
            pred.append(model.predict(file))
            #need work
            #collect accuracy
            #check if model has name
            #check if file has index
            #connect file and model in order for easy search
        
        return pred
    
    def run(self):
        tsneSVC,pcaSVC=self.train()
        predtsne=self.test(tsneSVC)
        predpca=self.test(pcaSVC)
        
        return predtsne,predpca 
    #list of prediction numpy array
    #number of rows-->number of documents saved in test path
      '''  
if __name__=='__main__':
    trainpath='./pdf2csv/testcsv' #fill in
    testpath='./pdf2csv/testcsv' #fill in
    get_result=SVC_result(trainpath,testpath)
    get_result.run() #desired result
    
    