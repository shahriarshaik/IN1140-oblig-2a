
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
2. Hvordan beregner vi sannsynligheten for et ord gitt det foregående ordet 𝑃 (𝑤𝑖
|𝑤𝑖−1) fra et
korpus?

Når vi skal beregne basert foregående ordet beregner man ved å brukedenne formelen: P(wi|wi-1)/P(wi-1)

"""

"""
3. Du skal nå bruke bigrammodellen og tekstkorpuset til å beregne sannsynligheten for setningen
"<s> Kari synger ikke <\s>". Vis hvilke sannsynligheter du trenger, og hvordan disse
beregnes fra korpuset. Du trenger ikke å regne ut den totale sannsynligheten for setningen.
"""
import nltk

print("helloooo")



