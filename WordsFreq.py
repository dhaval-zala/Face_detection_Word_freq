import nltk

class WordFreq:

    def __init__(self):

        self.file = open('firefox.txt', mode='r')

    def freq(self):

        wt_words = self.file.read()

        wt_words = wt_words.split()

        data_analysis = nltk.FreqDist(wt_words)

        for key in sorted(data_analysis):
            print("%s: %s" % (key, data_analysis[key]))

        data_analysis = nltk.FreqDist(data_analysis)

        return data_analysis.plot(100, cumulative=False)


wfreq = WordFreq()
wfreq.freq()
