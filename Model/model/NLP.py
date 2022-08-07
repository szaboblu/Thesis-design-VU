#!/usr/bin/env python
# coding: utf-8

# In[1]:


import errno

import spacy
from spacy.language import Language
from spacy import displacy
from spacy.tokens import Span
from spacy.matcher import PhraseMatcher
from spacy.matcher import Matcher
from spacy_wordnet.wordnet_annotator import WordnetAnnotator
import re
import sys
import pickle


# In[2]:


@Language.factory('make_street_entity_matcher')
def make_street_entity_matcher(nlp, name, entity_name:str, file_name:str, pattern=None):
    return EntityMatcher(nlp, file_name, entity_name, pattern)


# In[3]:


@Language.factory('make_city_entity_matcher')
def make_city_entity_matcher(nlp, name,entity_name:str, file_name=None, pattern=None):
    return EntityMatcher(nlp, file_name, entity_name, pattern)


# In[4]:


@Language.factory('make_landmark_entity_matcher')
def make_directions_entity_matcher(nlp, name,entity_name:str, file_name=None, pattern=None):
    return EntityMatcher(nlp, file_name, entity_name, pattern)


# In[5]:


@Language.factory('make_directions_entity_matcher')
def make_directions_entity_matcher(nlp, name,entity_name:str, file_name=None, pattern=None):
    return EntityMatcher(nlp, file_name, entity_name, pattern)


# In[6]:


@Language.factory('make_neighbourhood_entity_matcher')
def make_neighbourhood_entity_matcher(nlp, name,entity_name:str, file_name=None, pattern=None):
    return EntityMatcher(nlp, file_name, entity_name, pattern)


# In[7]:


@Language.factory('make_ordinals_entity_matcher')
def make_ordinals_entity_matcher(nlp, name,entity_name:str, file_name=None, pattern=None):
    return EntityMatcher(nlp, file_name, entity_name, pattern)


# In[8]:


@Language.factory('make_organisation_entity_matcher')
def make_organisation_entity_matcher(nlp, name, entity_name:str, file_name=None, pattern=None):
    return EntityMatcher(nlp, file_name, entity_name, pattern)


# In[9]:


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


# In[10]:


def regex_part_of_string(filename):
    try:
        line_list = open(filename,'r').readlines()
    except IOError as ex:
        if ex.errno != errno.ENOENT:
            raise
    else:
        phrase_list = '('
        for l in line_list:
            if phrase_list == '(':
                phrase_list += f'{l.strip()})'
            else:
                phrase_list += f'|({l.strip()})'
        return f"(\w*({phrase_list})\w*)"


# In[11]:


labels={'street':"STREET",
        'city':"CITY",
        'landmark':"LANDMARK",
        'neighbourhood':"NEIGHBOURHOOD",
        'ordinals':"ORDINALS",
        'organisation':"ORGANISATION",
        'direction':"DIRECTION"}


# In[12]:


nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)


# In[13]:


filename_dutch_streets_types = '../../Data/street_types_dutch.txt'
street_regex = regex_part_of_string(filename_dutch_streets_types)
regex =[[{"TEXT": {"REGEX": "^[A-Z][0-9]{1,5}.*"}}],[{"TEXT": {"REGEX": street_regex}}]]
filename_street = '../../Data/street_names_north_holland.txt'
nlp.add_pipe("make_street_entity_matcher", config={"file_name":filename_street,"entity_name": labels['street'], "pattern": regex}, after="ner")


# In[14]:


filename_city = '../../Data/cities.txt'
nlp.add_pipe("make_city_entity_matcher", config={ "file_name": filename_city, "entity_name": labels['city']}, after="ner")


# In[15]:


filename_dutch_landmark_types = '../../Data/landmark_types_dutch.txt'
landmark_regex = regex_part_of_string(filename_dutch_landmark_types)
regex =[[{"TEXT": {"REGEX": landmark_regex}}]]
filename_landmarks = '../../Data/landmarks.txt'
nlp.add_pipe("make_landmark_entity_matcher", config={ "file_name": filename_landmarks, "entity_name": labels['landmark'], "pattern": regex }, after="ner")


# In[16]:


filename_landmarks = '../../Data/neighbourhoods_amsterdam.txt'
nlp.add_pipe("make_neighbourhood_entity_matcher", config={ "file_name": filename_landmarks, "entity_name": labels['neighbourhood'],}, after="ner")


# In[17]:


filename_landmarks = '../../Data/ordinals.txt'
nlp.add_pipe("make_ordinals_entity_matcher", config={ "file_name": filename_landmarks, "entity_name": labels['ordinals'],}, after="ner")


# In[18]:


filename_landmarks = '../../Data/organisations.txt'
nlp.add_pipe("make_organisation_entity_matcher", config={ "file_name": filename_landmarks, "entity_name": labels['organisation'],}, after="ner")


# In[19]:


directions =[[{"LOWER": 'left'}], [{"LOWER": 'right'}], [{"LOWER": 'straight'}]]
nlp.add_pipe("make_directions_entity_matcher", config={"entity_name": labels['direction'], "pattern": directions}, after="ner")


# In[20]:


def getAddressFromFileName(filename):
    return filename.split("/")[-1].split(".",1)[0]


# In[21]:


class DirectionInstruction:
    def __init__(self, text):
        self.text=text
        self.data={}
    def extend(self,label='null', value='null'):
        self.data[label] = value


# In[22]:


class RouteDescription:
    def __init__(self, destination):
        #self.origin = origin
        self.destination = destination
        self.steps = []

    def addStep(self, directionInstruction):
        self.steps.append(directionInstruction)


# In[23]:


def display_ent(tokenisedList):
    colors = {labels['street']: '#28821e',
              labels['direction']: '#33DCFF',
              labels['city']:'#742318',
              labels['landmark']:'#ba03fc',
              labels['neighbourhood']:'#1212cc',
              labels['ordinals']:'#cc1212'}
    options={'distance':50,'colors': colors}
    displacy.render(tokenisedList, style='ent',options=options, jupyter=True)
    #displacy.render(tokenisedList, style="dep")


# In[24]:


def read_route(filename):
    tokenisedList = []
    address = getAddressFromFileName(filename)
    route= RouteDescription(destination=address)

    with open(file,'r') as f:
        line = f.readline()
        is_travel_direction = False
        while line:
            line = f.readline()
            nlpLine = nlp(line)
            pattern = [{"LOWER": "directions"}]
            matcher.add("BEGIN", [pattern])
            matches = matcher(nlpLine)

            if line == '\n':
                is_travel_direction = False
            if is_travel_direction:
                tokenisedList.append(nlpLine)
                instruction = DirectionInstruction(line)
                if(nlpLine.ents):
                    for entity in nlpLine.ents:
                        for label in labels:
                            if(labels[label] == entity.label_):
                                instruction.extend(label,entity.text)
                    route.addStep(instruction)
            for match_id, _, _ in matches:
                string_id = nlp.vocab.strings[match_id]
                if string_id == 'BEGIN':
                    is_travel_direction = True
    display_ent(tokenisedList)
    with open(f'../../Data/Routes/{address}.pickle', 'wb') as pickleFile:
        pickle.dump(route, pickleFile)
    return route


# In[25]:


file = '../../Data/AH/Haarlemmerdijk 1.txt'
Route = read_route(file)
print(getAddressFromFileName(file))


# In[26]:


if __name__ == '__main__':
    print('__name__',sys.argv)
    read_route(sys.argv[1])


# In[27]:


get_ipython().system('jupyter nbconvert --to script *.ipynb')

