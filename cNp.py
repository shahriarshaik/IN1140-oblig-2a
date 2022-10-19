#oppgave 1.1:
"""
bigrammer forekommer er (2=n)
⟨<s>, Per⟩, ⟨per, synger⟩, ⟨synger, ikke⟩,⟨ikke <\s>⟩
⟨<s>, Kari ⟩, ⟨Kari, synger⟩, ⟨synger, <\s>⟩
⟨<s>, Ola⟩, ⟨Ola, synger⟩, ⟨synger, ikke ⟩, ⟨ikke, <\s>⟩  
"""
#oppgave 1.2:
"""
for å regne sannsynnlighet for et ord ved  foregående ordet 𝑃 (𝑤𝑖|𝑤𝑖−1) som for eksempel: ordet ⟨Kari, synger⟩ 𝑃 (Kari| synger) = Count (synger, Kari)/Count (synger)
"""
#oppgave1.3 :
""" 
vi må finne sannsynlighet for alle par ordene og gange svarene sammen. 
som her: "<s> Kari synger ikke <\s>"  finner sannsynligheten P(Kari|<s>) * P(synger|Kari) * P(ikke|synger) * P(<\s>|ikke), metoden for det ble besvaret på oppgaven 1.2
"""
#oppagve 2a:
"""
Jesper:substantiv "NN"
drikker:verb "VB"
saft:substantiv "NN"
uten:preposisjon "PR"
sugerør:substantiv  "NN"
"""
#oppgave 2b:
"""
Hun:pronomen "PO"
løper:verb "VB"
raskt:adjektiv "JJ"
som:subjunksjon "SB"
en:determinativ "DET"
atlet:substantiv "NN"
"""
#oppgave 3:
from tokenize import tokenize
import nltk
from collections import Counter
from nltk import bigrams, trigrams
from collections import defaultdict
from nltk.corpus import brown
from nltk.corpus import gutenberg
import numpy as np
#3.1
bible_words = gutenberg.words("bible-kjv.txt")
total_words = len(bible_words)
print("--------------3.1--------------")
print("Antall ord: " , total_words)
#3.2
liste = []
for x in bible_words: # dette går gjennom ordene 
    liste.append(x.lower())#dette gjør alt til små bokstaver
total_type = len(set(liste)) # her finner hvor mange ord som er forskjellige, jeg bruker set for å fjerne de som ligner
print("--------------3.2--------------")
print("antall type: ", total_type) 
#3.3
fleste = Counter(bible_words)
print("--------------3.3--------------")
print("meste 20" , fleste.most_common(20))
#3.4
earth = fleste["earth"]
death = fleste["death"]
life = fleste["life"]
print("--------------3.4--------------")
print("frekvensen av 'earth': ",earth)
print("frekvensen av 'death': ",death)
print("frekvensen av 'life': ",life)
#3.5
setninger = gutenberg.sents('bible-kjv.txt')
syvende = setninger[6]
print("--------------3.5--------------")
print(list(bigrams(syvende, pad_left=True, pad_right=True)))
#3.6
åttende = setninger[7]
print("--------------3.6--------------")
print(list(trigrams(åttende, pad_left=True, pad_right=True)))
#3.7
BigramTelling = defaultdict(lambda:defaultdict(lambda: 0))
BigramModel = defaultdict(lambda:defaultdict(lambda:0.0))
for x in setninger:
    for forsteOrd,andreOrd in bigrams(x, pad_right = True, pad_left = True):
        BigramTelling[forsteOrd][andreOrd] += 1 
for x in BigramTelling:
    tellingTotalt = sum(BigramTelling[x].values())
    for y in BigramTelling[x]:
        BigramModel[x][y]=BigramTelling[x][y] / tellingTotalt
texte = [None]
finish = False
while not finish: 
    nokkel = texte[-1]
    ordene = list(BigramModel[nokkel].keys())
    sann = list(BigramModel[nokkel].values())
    texte.append(np.random.choice(ordene, p=sann))
    if texte [-1:] == [None]:
        if len(texte) > 50:
            finish = True
tt = " ".join([t for t in texte if t])
print("--------------3.7--------------")
print(tt)
#3.8
ttSplit = tt.split()
bigramtekst = list(bigrams(ttSplit))
ff = Counter(bigramtekst)
#print(ff)
ordbok = {}
for x, y in ff.items():
    ordbok[x] = y/len(tt)
print("--------------3.8--------------")
print(np.prod(sum(ordbok.values())))
#oppgave 4
#4.1
patterns = [
    (r'.*ing$', 'VBG'),                # gerunds: "developing"
    (r'.*ed$', 'VBD'),                 # simple past: "called"
    (r'.*es$', 'VBZ'),                 # 3rd singular present: "bases"
    (r'.*ould$', 'MD'),                # modals: "could"
    (r'.*\'s$', 'NN$'),                # possessive nouns: "Samsung's"
    (r'.*s$', 'NNS'),                  # plural nouns: "Windows"
    (r'^-?[0-9]+(\.[0-9]+)?$', 'CD'),  # cardinal numbers: "£425"
    (r'.er*', 'NN'),                   # nouns (default): "sea"
    (r'\b[Tt]he|[Aa]n?\b', 'DT'),      # determiner (default): "the"
    (r'.*s', 'PRP$'),                  # possessive pronoun: "hem"
    (r'.*', 'NNP'),                    # proper noun: "Abdul"
]
#4.2
browne = brown.tagged_sents(categories='fiction')
sents = brown.sents(categories='fiction')
reg = nltk.RegexpTagger(patterns)
Noya = reg.evaluate(browne)
print("--------------4.2--------------")
print( Noya)

#4.3
file = open("setninger.txt", "r", encoding="utf8").read()
hent = nltk.tokenize.sent_tokenize(file)
liste = []
for x in hent:
    liste.append(x.split())
setninger = nltk.RegexpTagger(patterns)
tag = setninger.tag(liste[0])
new_fil = open("taggede_setninger.txt", "a", encoding="utf-8")
for x in str(tag):
    new_fil.write(x)
new_fil.close() 
new_fil = open("taggede_setninger.txt", "r", encoding="utf-8")
print("--------------4.3--------------")
print(new_fil.read())
print(tag)
"""
første feil er at "." står for NNP. i steden å sette bare . i DT må vare mer prisis.
andre feil er "with" står for NNP. with skal være preposisjon
tredje er "use" står for PRP$. dette er verb. må sette type VB opp på patterns. 
"""