# SentimentalAnalysis
Project done in Summer '19

Input : URLs which have the link to the annual financial reports of the SEC.

Output : Variables like Sentence Score, Positive Score, Negative Score, Polarity Score, in the form of a CSV file after analysing the financial reports.

Methodology Used :
 -> Preprocessing : Tokenizing the sentences into words, removal of HTML tags, removal of extra space, removal of stop words, removal of punctuation.
 -> Sentiment Analysis : Comparing the words obtained after preprocessing to a master dictionary and then performing sentiment analysis, alloting a positivity/negativity score to each word, and then calculating the net sentiment score.
 -> Output : Converting the final list consiting of variables into a CSV file.

Tools Used : 
 -> Python 3.x
 -> Modules used :
    -nltk
    -string
    -csv
    -pandas
    -contractions



