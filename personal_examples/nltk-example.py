# from nltk.corpus import brown
#
# sw = brown.words()
# print(len(sw))

from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer("english")
stemmer.stem("responsiveness")
