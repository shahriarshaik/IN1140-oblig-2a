from itertools import count
import random
import nltk
from nltk import probability
from nltk.inference.tableau import Categories
from nltk.tokenize import regexp, word_tokenize
import numpy as np
nltk.download('gutenberg')
nltk.download('punkt')
from nltk.corpus import gutenberg
from nltk.util import bigrams, trigrams

gutenberg.fileids()

gutenberg.raw("bible-kjv.txt")

#Hvor mange tokens er det i teksten, a)
gutenberg_ord = gutenberg.words("bible-kjv.txt")
antall_token = len(gutenberg_ord)
print("Antall tokens er", antall_token)

#Hva er totalt antall ord-typer i teksten, b)
ordtyper = []
for token in gutenberg_ord:
    typer = token.lower()
    ordtyper.append(typer)

antall_typer = len(set(ordtyper))
print("Antall ordtyper er:", antall_typer)

#Hva er de 20 mest frekvente ordtypene i teksten, c)
from collections import Counter, defaultdict

frekvens = Counter(gutenberg_ord)

for ord in frekvens.most_common(20):
    print(ord)

#Hva er frekvensen til ordene heaven, death og life, d)
death_forekomst = frekvens["death"]
print("Death forekommer:", death_forekomst, "ganger")
life_forekomst = frekvens["heaven"]
print("Life forekommer:", life_forekomst, "ganger")
heaven_forekomst = frekvens["life"]
print("Heaven forekommer:", heaven_forekomst, "ganger")

#Hvilke bigrammer forekommer i setning 4, e)
gutenberg_setninger = gutenberg.sents("bible-kjv.txt")
#print(gutenberg_setninger[3])
bigrammer = bigrams(gutenberg_setninger[3])
print("Bigram som forekommer i den fjerde setningen er:\n", list(bigrammer))

#Hvilke trigrammer forekommer i setning 5, f)
trigrammer = trigrams(gutenberg_setninger[4])
print("Trigram som forekommer i den femte setningen er:\n", list(trigrammer))

from collections import defaultdict
from nltk import bigrams, trigrams

bigram_counts = defaultdict(lambda: defaultdict(lambda: 0))
bigram_model = defaultdict(lambda: defaultdict(lambda: 0.0))

for sentence in gutenberg_setninger:
    for w1, w2 in bigrams(sentence, pad_right= True, pad_left = True):
        bigram_counts[w1][w2] += 1
    
for w1 in bigram_counts:
    total_bigramcount = sum(bigram_counts[w1].values())
    for w2 in bigram_counts[w1]:
        if total_bigramcount:
            bigram_model[w1][w2] = bigram_counts[w1][w2]/total_bigramcount
    

text = [None]
sentence_is_finished = False

while not sentence_is_finished:
    key = text[-1]
    
    ord = list(bigram_model[key].keys())

    probs = list(bigram_model[key].values())
  
    text.append(np.random.choice(ord, p=probs)) 
    
    if text[-1] == None:
        sentence_is_finished = True

generert_tekst = " ".join([t for t in text if t])
print(generert_tekst)

#Finn sannsynligheten til den genererte teksten
generert_tekst1 = generert_tekst.split()
bigram_generert_tekst = list(bigrams(generert_tekst1))
print(bigram_generert_tekst)

fd_generert = Counter(bigram_generert_tekst)
print(fd_generert)
probabilities = {}

for word, count in fd_generert.items():
    probabilities[word] = count/len(generert_tekst)
print("sans:", np.prod(sum(probabilities.values())))