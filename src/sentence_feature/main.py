import importlib
import src.sentence_feature.load_model as load_model
from sentence_transformers import SentenceTransformer, util

import glob
import os
import io
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import KMeans

from sklearn.datasets import make_blobs
from sklearn.preprocessing import normalize

from sklearn.cluster import AffinityPropagation


def run():
    # 比較する文章
    sentence1 = "自然言語処理は非常に興味深い分野です。"
    sentence2 = "自然言語処理には多くの挑戦がありますが、面白いです。"

    model = load_model.getModel()

    
    # 文章をベクトルに変換
    embeddings1 = model.encode(sentence1, convert_to_tensor=True)
    embeddings2 = model.encode(sentence2, convert_to_tensor=True)

    print( embeddings1.shape, embeddings2.shape )
    
    # コサイン類似度の計算
    cosine_score = util.pytorch_cos_sim(embeddings1, embeddings2)[0][0]
    
    print(f"文章1と文章2の類似度: {cosine_score}")

def loadContent():
    model = load_model.getModel()

    path2vec = {}
    for dir in glob.glob( "/home/hoge/work/ifritJP.github.io/blog2/content/posts/*" ):
        for file in glob.glob( os.path.join( dir, "*.org" ) ):
            fileObj = io.open( file )
            content = fileObj.read()
            path2vec[ file ] = model.encode(content )
    return path2vec


def dispCluster( labels, path_list ):
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    print( labels )

    label_path_list = sorted( list( zip( labels, path_list ) ), key=lambda X:X[0] )
    
    for val in label_path_list:
        print( *val )
    
    print("number of estimated clusters : %d" % n_clusters_)
    

def execMeanShift( path2vec ):
    path_list = list( path2vec.keys() )
    
    x = np.array( [ path2vec[ path ] for path in path_list ] )
    print( x.shape )


    # 距離が小さい点を近傍とする
    #ms = MeanShift( bandwidth=bandwidth )
    #ms = MeanShift( bandwidth=2 )
    #ms = MeanShift( bandwidth=0.5 )

    ms = AffinityPropagation(random_state=5, damping=0.6 )
    
    ms.fit( x )
    labels = ms.labels_
    #cluster_centers = ms.cluster_centers_

    dispCluster( labels, path_list )

def execkMeans( path2vec, num ):
    path_list = list( path2vec.keys() )
    
    x = np.array( [ path2vec[ path ] for path in path_list ] )
    x = normalize(x, norm='l2')
    print( x.shape )

    km = KMeans(n_clusters=num, random_state=0, n_init="auto")
    kmeans = km.fit( x )
    
    labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_

    dispCluster( labels, path_list )
    

def test():
    path2vec = loadContent()
#    execMeanShift( path2vec )
    execkMeans( path2vec, 20 )
