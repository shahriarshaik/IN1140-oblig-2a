
"""
1. Hvilke bigrammer forekommer i korpuset?

det forkommer n + 1 bigrammer per linje,
L_1: n = 3, 3 + 1 = 4
L_2: n = 2, 2 + 1 = 3
L_3: n = 3, 3 + 1 = 4

samlet er det 11 bigramer  

<<s>, Per>, <Per, synger>, <synger, ikke>, <ikke,<\s>>
<<s>, Kari>, <Kari, synger, <synger,<\s>>
<<s>, Ola>, <Ola, synger>, <synger, ikke>, <ikke,<\s>>

"""


"""
2. Hvordan beregner vi sannsynligheten for et ord gitt det foregående ordet P (wi|wi-1) fra et
korpus?

Når vi skal beregne basert foregående ordet beregner man ved å brukedenne formelen: P(wi|wi-1)/P(wi-1)

"""

"""
3. Du skal nå bruke bigrammodellen og tekstkorpuset til å beregne sannsynligheten for setningen
"<s> Kari synger ikke <\s>". Vis hvilke sannsynligheter du trenger, og hvordan disse
beregnes fra korpuset. Du trenger ikke å regne ut den totale sannsynligheten for setningen.

P(Kari|<s>) * P(synger|Kari) * P(ikke|synger) * P(<\s>|ikke) = P(<s>,Kari)/P(<s>) * P(Kari,synger)/P(Kari) * P(synger,ikke)/P(ikke) * P(ikke, <\s>)/P(ikke) =
1/3 * 1/1 * 1/1 * 3/3    =   0,33 * 1 * 1 * 1    =   0,33. 33,3% sjanse
"""

"""
Gitt ordklassene i tabellen under, tildel ordklasser til alle ordene i setningene. Du må velge ett
alternativ for hvert ord.

a)

"Jesper drikker saft uten sugerør"

sugerør: NN
saft: NN
drikker: VB
Jesper: PO
uten: RB

b)

"Hun løper raskt som en atlet"

hun: PO
løper: VB
raskt: JJ
som: SB
en: DET
atlet: NN

"""


import nltk
from nltk.corpus import gutenberg
from itertools import count
import random
import nltk
#from nltk import probability
#from nltk.inference.tableau import Categories
#from nltk.tokenize import regexp, word_tokenize
import numpy as np
nltk.download('gutenberg')
nltk.download('punkt')
from nltk.corpus import gutenberg
from nltk.util import bigrams, trigrams
from collections import Counter, defaultdict



gutenberg.fileids()
gutenberg.raw("bible-kjv.txt")
gutenberg_ord = gutenberg.words("bible-kjv.txt")
antall_token = len(gutenberg_ord)
print("\n\noppgave 3.1")
print("Antall tokens er", antall_token)


ordtyper = []
for token in gutenberg_ord:
    typer = token.lower()
    ordtyper.append(typer)
antall_typer = len(set(ordtyper))
print("\n\noppgave 3.2")
print("Antall ordtyper er:", antall_typer)


frekvens = Counter(gutenberg_ord)
print("\n\noppgave 3.3 \n20 mest frekvente ordtypene:")
for ord in frekvens.most_common(20):
    print(ord)

print("\n\noppgave 3.4")
death_forekomst = frekvens["death"]
print("Death forekommer:", death_forekomst, "ganger")
life_forekomst = frekvens["heaven"]
print("Life forekommer:", life_forekomst, "ganger")
heaven_forekomst = frekvens["life"]
print("Heaven forekommer:", heaven_forekomst, "ganger")


print("\n\n\noppgave 3.5")
gutenberg_setninger = gutenberg.sents("bible-kjv.txt")
bigrammer = bigrams(gutenberg_setninger[6])
print("Bigram som forekommer i den syvente setningen er:\n", list(bigrammer))


print("\n\n\noppgave 3.6")
gutenberg_setninger = gutenberg.sents("bible-kjv.txt")
trigram = trigrams(gutenberg_setninger[7])
print("Bigram som forekommer i den aattende setningen er:\n", list(trigram))


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
antall = 0
while not sentence_is_finished:
    key = text[-1]
    ord = list(bigram_model[key].keys())
    probs = list(bigram_model[key].values())
    text.append(np.random.choice(ord, p=probs)) 
    antall = antall + 1
    if text[-1] == None and antall > 51:
        sentence_is_finished = True
generert_tekst = " ".join([t for t in text if t])
print("\n\noppgave 3.7")
print(generert_tekst)


generert_tekst1 = generert_tekst.split()
bigram_generert_tekst = list(bigrams(generert_tekst1))
print("\n\noppgave 3.8")
fd_generert = Counter(bigram_generert_tekst)
print(fd_generert)
probabilities = {}
for word, count in fd_generert.items():
    probabilities[word] = count/len(generert_tekst)
print("sans:", np.prod(sum(probabilities.values())))
