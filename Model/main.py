import errno

import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('wordnet')
nltk.download('omw-1.4')
import spacy
from spacy.language import Language
from spacy import displacy
from spacy.tokens import Span
from spacy.matcher import PhraseMatcher
from spacy.matcher import Matcher
from nltk.tokenize import word_tokenize
import re

#%%

@Language.factory('make_street_entity_matcher')
def make_street_entity_matcher(nlp, name, entity_name:str, file_name:str, pattern=None):
    return EntityMatcher(nlp, file_name, entity_name, pattern)

#%%

@Language.factory('make_directions_entity_matcher')
def make_directions_entity_matcher(nlp, name,entity_name:str, file_name=None, pattern=None):
    return EntityMatcher(nlp, file_name, entity_name, pattern)

#%%

class EntityMatcher:
    def __init__(self, nlp, file_name, entity_name, pattern):
        if file_name:
            try:
                line_list = open(file_name,'r').readlines()

            except IOError as ex:
                if ex.errno != errno.ENOENT:
                    raise
            else:
                phrase_list = []
                for l in line_list:
                    phrase_list.append(l.strip())
                phrase_patterns = [nlp(text) for text in phrase_list]
                self.phrase_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
                self.phrase_matcher.add(entity_name, None, *phrase_patterns )
        if pattern:
            self.matcher = Matcher(nlp.vocab)
            self.matcher.add(entity_name, pattern )

    def __call__(self, doc):
        matches = set()
        if hasattr(self, "matcher"):
            matches.update(self.matcher(doc))
        if hasattr(self, "phrase_matcher"):
            matches.update(self.phrase_matcher(doc))
        seen_tokens = set()
        new_entities = []
        entities = doc.ents
        for match_id, start, end in matches:
            if start not in seen_tokens and end - 1 not in seen_tokens:
                new_entities.append(Span(doc, start, end, label=match_id))
                entities = [
                    e for e in entities if not (e.start < end and e.end > start)
                ]
                seen_tokens.update(range(start, end))

        doc.ents = tuple(entities) + tuple(new_entities)
        return doc

#%%



#%%

#nlp = spacy.load("nl_core_news_sm")
nlp = spacy.load("en_core_web_sm")
change = "wjdA"

regex =[[{"TEXT": {"REGEX": "^[A-Z][0-9]{1,5}.*"}}]]
nlp.add_pipe("make_street_entity_matcher", config={"file_name":'../Data/street_names_north_holland.txt',"entity_name": "STREET", "pattern": regex}, after="ner")

#%%

directions =[[{"LOWER": 'left'}], [{"LOWER": 'right'}]]
nlp.add_pipe("make_directions_entity_matcher", config={"entity_name": "DIRECTION", "pattern": directions}, after="ner")


#%%

file = '../Data/AH/Haarlemmerdijk 1.txt'
tokenisedList = []
with open(file,'r') as f:
    line = f.readline()
    while line:
        line = f.readline()
        nlpLine = nlp(line)
        tokenisedList.append(nlpLine)

#%%

# if you want to keep specific stop words
# nlp.Defaults.stop_words.remove()
# nlp.vocab['beyond'].is_stop = False

#%%

colors = {"STREET": '#981231', "DIRECTION": '#33DCFF'}
options={'distance':50,'colors': colors}
displacy.render(tokenisedList, style='ent',options=options, jupyter=True)
#displacy.serve(tokenisedList, style='ent', options=options )
