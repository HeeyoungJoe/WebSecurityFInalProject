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
from MyParser import MyParser
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
def make_2D(x,y):
    #t-sne
    #t-sne simply returns
    print("\n\n=============T-SNE=============\n")
    tsne=TSNE(n_components=2,learning_rate='auto',init='random').fit_transform(x)
    return tsne
    '''
    #pca
    print("\n\n=============PCA=============\n")
    pca=PCA(n_compoenets=2).fit_transform(x)
    '''

def print_2D(x,y):
    '''
    code from:
    https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60
    '''
    fg=plt.figure(figsize=(10,10))
    ax=fg.add_subplot(1,1,1)
    ax.set_xlabel('Principal Component 1',fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('2 component PCA', fontsize = 20)

    targets=['malicious','benign']
    colors=['r','b']

    data=pd.DataFrame(np.concatenate((x,y.resize(y.size,1)),axis=1),columns=['pc1','pc2','target'])

    for target, color in zip(targets,colors):
        indices=data['target']==target
        ax.scatter(data.loc[indices,'pc1'],data.loc[indices,'pc2'],c=color,s=50)
    ax.legend(targets)
    ax.grid()

def unsupervised_algorithm(version,dataX):
    
    clusters={}

    #k means clustering -->this isn't supervised. Does this have meaning?
    print("\n\n=============K Means=============\n")
    km=KMeans(n_clusters=2,random_state=0)
    km.fit(dataX)
    clusters['km']=km.labels

    
    return clusters

if __name__=="__main__":
    a=MyParser('./pdf2csv/testcsv')
    a.parse()
    x=a.batch_data[:5]
    y=a.batch_target[:5]
    x.resize((5,:))

    print("\n\n\\\\yoyo x size %d and y size %d\n\n"%(x.size,y.size))

    x_new=make_2D(x,y)
    print_2D(x_new,y)
    unsupervised_algorithm(x)