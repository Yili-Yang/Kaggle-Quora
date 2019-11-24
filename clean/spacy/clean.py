import spacy
from spacymoji import Emoji

class Clean():

    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        emoji = Emoji(self.nlp)
        self.nlp.add_pipe(emoji, first=True)

    def lemma(self,text):
        return ' '.join([i.lemma_ for i in self.nlp(text)])
    
    def word_tokenize(self,text):
        return [i.text for i in self.nlp(text)]
    
    def posTagger(self,text):
        return [i.pos_ for i in self.nlp(text)]
    
    def removeStopWord(self,text):
        return [i.text for i in self.nlp(text) if not i.stop_]
    
    def removePunct(self,text):
        return [i.text for i in self.nlp(text) if not i.is_punct]
    
    def removeEmoji(self,text):
        return [i.text for i in self.nlp(text) if not i.is_emoji]