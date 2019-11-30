import sys
sys.path.append("..")
from spacyClean.clean import Clean
import gensim 
from gensim.models import Word2Vec
from bert_embedding import BertEmbedding
class vectorization(Clean):
    
    def __init__(self):
        '''get all the methods from Clean module'''
        super().__init__()

    def word2Vec(self,text,**kwargs):
        tokens = self.lemma(self.standardCleaning(text))
        data = [i.lower() for i in tokens]
        model = gensim.models.Word2Vec(data, **kwargs)
        return model

    def bertEmbedding(self,text,**kwargs):
        tokens = self.lemma(self.standardCleaning(text))
        data = [i.lower() for i in tokens]
        bert_embedding = BertEmbedding()
        model = bert_embedding(data,**kwargs)
        return model
    
    def Glove(self,text):
        #to do
    
    def tfidf(self,text):
        #to do
    
    