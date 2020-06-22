# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 11:35:48 2019

@author: User
"""

import nltk.data
import pandas as pd
from nltk.corpus import stopwords
import re
import string
from string import punctuation
from contractions import contractions_dict
import csv


###Cleaning the Text###

#Step 1: removal of numbers
def removalOfNumbers(lcase_text):
    lcase_text = re.sub(r'\d+', '', lcase_text)
    return lcase_text

#Step 2: Removing contractions
def expand_contractions(lcase_text, contractions_dict):
    contractions_pattern = re.compile('({})'.format('|'.join(contractions_dict.keys())),
                                      flags=re.IGNORECASE | re.DOTALL)

    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contractions_dict.get(match) \
            if contractions_dict.get(match) \
            else contractions_dict.get(match.lower())
        expanded_contraction = expanded_contraction
        return expanded_contraction

    expanded_text = contractions_pattern.sub(expand_match, lcase_text)
    expanded_text = re.sub("'", "", expanded_text)
    expanded_text = re.sub("-", " ", expanded_text)
    return expanded_text

#Step 3: Removal of Punctuations
def removalOfPunctuations(lcase_text):
    lcase_text = lcase_text.translate(str.maketrans("","", string.punctuation))
    return lcase_text

#Step 4: Removal of White Spaces
def removalOfWhiteSpaces(lcase_text):
    lcase_text = lcase_text.strip()
    return lcase_text

#Step 5 : Removing tags
def removalOfTags(lcase_text):
    lcase_text = re.sub('<[^<]+?>','', lcase_text)
    return lcase_text

#Step 6 : Converting the cleaned text into a list
def convertToList(lcase_text):
    lcase_text = lcase_text.split()
    return lcase_text

#step 7: Removal of Stop Words
def removalOfStopWords(lcase_text):
    filenames = ["StopWords_GenericLong.txt","StopWords_DatesAndNumbers.txt",
             "StopWords_Geographic.txt","StopWords_Currencies.txt","StopWords_Generic.txt",
             "StopWords_Auditor.txt","StopWords_Names.txt"]
    with open('output_file', 'w') as outfile:
        for fname in filenames:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)
    stopWords = []
    with open('output_file','r') as outfile:
        for word in outfile:
            stopWords.append(word.lower())
    eng_stopwords = set(stopwords.words('english'))
    for word in eng_stopwords:
        if word not in stopWords:
            stopWords.append(word.lower())
        
    new_text = []
    for word in lcase_text:
        if word not in stopWords:
            new_text.append(word)
    new_text = ' '.join([w for w in new_text])
    return new_text

#Usage of Master Dictionary to make positive and negative dictionary and therby calculating the scores
def reducingNoOfWords(new_text, my_dict):
    #Adding the words that make sense into a list, provided they are a 
    #part of the master dictionary created above"""
    new_text = nltk.word_tokenize(new_text)
    sense_words = []
    for word in my_dict.keys():
        if word in new_text:
            sense_words.append(word)
    sense_words = set(sense_words)
    sense_words = list(sense_words)
    return sense_words

#Calculating the Negaative Score
def negativeScore(sense_words, my_dict):
    neg_dict = {'Negative': []}
    #print(my_dict.keys())
    for word in sense_words:
        if word in my_dict:
            if (my_dict[word][0] > 0) and (my_dict[word][1] <= 0):
                neg_dict['Negative'].append(word)
    neg_dict['Negative'] = set(neg_dict['Negative'])           
    neg_dict['Negative'] = list(neg_dict['Negative'])
    neg_score = []
    nscore1 = 0
    for word in neg_dict['Negative']:
        neg_score.append(-1)
        nscore1 = (nscore1-1)
    nscore = nscore1*(-1)
    return nscore

#Calculating the Positive Score
def positiveScore(sense_words, my_dict):
    pos_dict = {'Positive' : []}
    for word in sense_words:
        if word in my_dict:
            if (my_dict[word][0] <=0) and (my_dict[word][1] > 0):
                pos_dict['Positive'].append(word)
    pos_dict['Positive'] = set(pos_dict['Positive'])           
    pos_dict['Positive'] = list(pos_dict['Positive'])
    pos_score = []
    pscore = 0
    for word in pos_dict['Positive']:
        pos_score.append(1)
        pscore = pscore+1
    return pscore

#Calculating Word Count
def totalNumberOfWords(sense_words):
    wordCount = 0
    for word in sense_words:
        wordCount = wordCount+1
    return wordCount

#Calculating the Polarity Score
def polarityScore(pscore, nscore):
    polarityScore = (pscore-nscore)/((pscore+nscore)+0.000001)
    return polarityScore

#Calculating Positive Word Proportion
def positiveWordProportion(pscore, wordCount):
    positiveWordProp = pscore/wordCount
    return positiveWordProp

#Calculating the average sentence lenth
def sentence(text_data, sense_words):
    lcaseSent = ''.join([w.lower() for w in text_data])
    sentences = nltk.sent_tokenize(lcaseSent)
    #removing punctuation
    sentences = ''.join(c for c in sentences if c not in punctuation)
    sentences = nltk.sent_tokenize(sentences)
    avg_sent = len(sense_words)/len(sentences)
    return avg_sent

#Calculating the percentage of complex words
def complexWordCount(sense_words):
    
    def syllable_count(word):
        word = word.lower()
        count = 0
        vowels = "aeiouy"
        if word[0] in vowels:
            count += 1
        for index in range(1, len(word)):
            if word[index] in vowels and word[index - 1] not in vowels:
                count += 1
            if word.endswith("e"):
                count -= 1
            if count == 0:
                count += 1
        return count

    syllables_dict = {}
    complexCount = 0
#For every word where the number of syllables is greater than 2, it is a 
#complex word, hence count gets incremented.
    for word in sense_words:
        no_of_syllables = syllable_count(word)
        syllables_dict[word] = no_of_syllables
        if(no_of_syllables > 2):
            complexCount = complexCount+1
       
    return complexCount

def complexPercentage(sense_words, complexCount):
    perc_complex_words = (complexCount/len(sense_words))*100
    return perc_complex_words

#Calculating the FOG Index
def calculatingFOGIndex(avgSentLength, perc_complex_words):
    fog_index = (avgSentLength+perc_complex_words)*(0.4)
    return fog_index

#Uncertainty
def uncertainty(sense_words):
    uncertainty = pd.read_csv('uncertainty_dictionary.csv')
    listUnc = uncertainty.iloc[:, 0].values
    listUnc = list(listUnc)
    listUnc = [word.lower() for word in listUnc]
    uncertainty_count = 0
    for word in sense_words:
        if word in listUnc:
            uncertainty_count = uncertainty_count+1
    uncertainty_prop = uncertainty_count/len(sense_words)
    return uncertainty_prop


#Constraining 
def constraining(sense_words):
    constraining = pd.read_csv('constraining_dictionary.csv')
    listCon = constraining.iloc[:, 0].values
    listCon = list(listCon)
    listCon = [word.lower() for word in listCon]
    constraining_count = 0
    for word in sense_words:
         if word in listCon:
             constraining_count = constraining_count+1
    constraining_prop = constraining_count/len(sense_words)
    return constraining_prop


def main():
    #Reading the Dataset
    dataset = pd.read_csv('codeWithFunctions.csv')
    
    ##Extracting the column of URL's from the dataset
    url = dataset.iloc[:, -1].values
    url = list(url)
    print(url[2])
    site = 0
    
    download_dir = "output.csv" #where you want the file to be downloaded to 
       
    columnTitleRow = ['positive_score','negative_score','polarity_score','average_sentence_length','percentage_of_complex_words','fog_index','complex_word_count','word_count','uncertainty_score','constraining_score','positive_word_proportion']
    
    with open(download_dir, 'w', newline = '') as outfile:
        csvwriter = csv.writer(outfile)
        csvwriter.writerow(columnTitleRow)
    
    while (site<len(url)):
        #the text stored in the first url
        text_data = nltk.data.load('https://www.sec.gov/Archives/'+url[site])
        
        #Converting the text to lower case
        lcase_text = ''.join([w.lower() for w in text_data])
        
        lcase_text = removalOfNumbers(lcase_text)
        lcase_text = expand_contractions(lcase_text, contractions_dict)
        lcase_text = removalOfPunctuations(lcase_text)
        lcase_text = removalOfWhiteSpaces(lcase_text)
        lcase_text = removalOfTags(lcase_text)
        lcase_text = convertToList(lcase_text)
        new_text = removalOfStopWords(lcase_text)
        
        
        #extracting the words from the csv file
        master_dict = pd.read_csv("masterDict.csv")
        df = master_dict[['Word','Negative','Positive']]
        #Words that appear in the master dictionary and the cleaned data,
        #are added to the string sense_Words
        #Converting the the three columns of the dataframe to a dictionary of the 
        #form {'Word' : [Negative, Positive]}
        my_dict = df.set_index('Word').T.to_dict('list')
        #Converting the keys of the above formed dictionary into lowercase
        my_dict =  {str(k).lower(): v for k, v in my_dict.items()}
        
        sense_words = reducingNoOfWords(new_text, my_dict)
        
        outputList = []
        
        pscore = positiveScore(sense_words, my_dict)
        outputList.append(str(pscore))
        
        nscore = negativeScore(sense_words, my_dict)
        outputList.append(str(nscore))
        
        polScore = polarityScore(pscore, nscore)
        outputList.append(str(polScore))
        
        avgSentLength = sentence(text_data, sense_words)
        outputList.append(str(avgSentLength))
        
        complexCount = complexWordCount(sense_words)    
        perc_complex_words = complexPercentage(sense_words,complexCount)
        outputList.append(str(perc_complex_words))
        
        fog_index = calculatingFOGIndex(avgSentLength, perc_complex_words)
        outputList.append(str(fog_index))
        
        outputList.append(str(complexCount))
        
        wordCount = totalNumberOfWords(sense_words)
        outputList.append(str(wordCount))
        
        uncertainty_prop = uncertainty(sense_words)
        outputList.append(str(uncertainty_prop))
        
        constraining_prop = constraining(sense_words)
        outputList.append(str(constraining_prop))
        
        posWordProp = positiveWordProportion(pscore, wordCount)
        outputList.append(str(posWordProp))
        
           
        
        
        
        with open(download_dir, 'a', newline = '') as outfile:
            csvwriter = csv.writer(outfile)
            for row in download_dir:
                csvwriter.writerow(outputList)
                break
            
        site = site+1
        

    
if __name__ == "__main__":
    main()
    
    
