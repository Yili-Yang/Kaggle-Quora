import spacy

class Clean():
    def __init__():
        self.nlp = spacy.load('en_core_web_sm')

    def lemma(self,text):
        return ' '.join([i.lemma_ for i in self.nlp(text)])
    
    def word_tokenize(self,text):
        return [i.text for i in self.nlp(text)]
    
    def posTagger(self,text):
        return [i.pos_ for i in self.nlp(text)]
    
    def removeStopWords(self,text):
        return [i.text for i in self.nlp(text) if not i.stop_]
    
    def 
    
