from sentence_transformers import SentenceTransformer, util

# モデルのロード
#model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
#model = SentenceTransformer('studio-ousia/luke-japanese-large-lite')
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2' )



def getModel():
    return model
