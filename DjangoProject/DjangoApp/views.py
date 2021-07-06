from bson import ObjectId
from django.http import HttpResponse
from django.shortcuts import render
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.pipeline import make_pipeline
# Create your views here.
from django.views.decorators.csrf import csrf_exempt

from . models import Comments
import joblib
from apiclient.discovery import build
import nltk
import pandas as pd
import numpy as np
from tqdm import tqdm
import copy
import re
import string
import math
import regex
import csv
from csv import QUOTE_NONE
import unidecode

from sklearn import model_selection, naive_bayes, svm
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report as creport
import functools
import operator
import emoji
from multiprocessing import Pool
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class pretraitement:
    def __init__(self):
        self.arabic_punctuations = '''`÷« »×؛<>٩٨'٧٦٥٤٣٢١٠_()↗*•&^%][ـ،/:".,'{}⋮≈~¦+|٪”…“–ـ/[]%=#*+\\•~@£·_{}©^®`→°€™›♥←×§″′Â█à…“★”–●â►−¢¬░¶↑±▾	═¦║―¥▓—‹─▒：⊕▼▪†■’▀¨▄♫☆é¯♦¤▲è¸Ã⋅‘∞∙）↓、│（»，♪╩╚³・╦╣╔╗▬❤ïØ¹≤‡₹´'''
        self.english_punctuations = string.punctuation
        self.english_punctuations = self.english_punctuations.replace("!", "")
        self.english_punctuations = self.english_punctuations.replace("?", "")
        self.punctuations_list = self.arabic_punctuations + self.english_punctuations
        with open('media/emoji_signification.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()
            self.emojis_fr = {}
            for line in lines:
                line = line.strip('\n').split('\t')
                self.emojis_fr.update({line[0].strip(): line[1].strip()})
        # emojis_fr

        with open('media/emot2.txt', 'r', encoding='utf-8-sig') as f:
            lines = f.readlines()
            self.emot_fr = {}
            # emojis_ar2 = []
            for line in lines:
                line = line.strip('\n').split('\t')
                # line2 = line.strip('\n')
                self.emot_fr.update({line[0].strip(): line[1].strip()})
                # emojis_ar2.append(line[0])
        # emot_fr
        with open('media/stopwords-arabizi.txt', 'r', encoding='utf-8-sig') as f:
            lines = f.readlines()
            self.stop_word_comp = []
            # emojis_ar2 = []
            for line in lines:
                line = line.strip('\n').split('\t')
                # line2 = line.strip('\n')
                self.stop_word_comp.append(line[0].strip())
                # emojis_ar2.append(line[0])

    def _remove_punctuations(self,x):
        x = str(x)
        # translator = str.maketrans(' ', ' ', punctuations_list)
        translator = str.maketrans(self.punctuations_list, ' ' * len(self.punctuations_list))
        x2 = x.translate(translator)
        x3 = word_tokenize(x2)
        return x3

    def remove_punctuations(self,df):
        df['data'] = df['data'].apply(lambda x: self._remove_punctuations(x))
        return df


    # 3--------remove doubles (plus que deux fois)
    def _remove_repeating_char(self,x):
        x = str(x)
        return re.sub(r'(.)\1{2,}', r'\1', x)

    def remove_repeating_char(self,df):
        df['data'] = df['data'].apply(lambda x: self._remove_repeating_char(x))
        return df

    # 5--------lower case
    def _lower_case(self,x):
        x = x.lower()
        return x

    def lower_case(self,df):
        df['data'] = df['data'].apply(lambda x: self._lower_case(x))
        return df

    # 5--------extra witespace
    def _extraWhite(self,x):
        x = re.sub('\t', ' ', x)
        x = re.sub('\s+', ' ', x)
        return x

    def extraWhite(self,df):
        df['data'] = df['data'].apply(lambda x: self._extraWhite(x))
        return df

    # 6.1--------delete urls
    def _delete_URLs(self,text):
        t = re.sub('http\S+\s*', ' ', text)
        return t

    def delete_URLs(self,df):
        df['data'] = df['data'].apply(lambda x: self._delete_URLs(x))
        return df

    # 6.1--------delete single letters and numbers
    def _delete_single_letters(self,text):
        text = re.sub(r'\b\d+(?:\.\d+)?\b', '', text)
        text = re.sub('(?<![\w])(?:[a-zA-Z0-9](?: |$))', '', text)
        return text

    def delete_single_letters(self,df):
        df['data'] = df['data'].apply(lambda x: self._delete_single_letters(x))
        return df

    # 6.2--------delete hashtags

    def split_hashtag_to_words(self,tag):
        tag = tag.replace('#', '')
        tags = tag.split('_')
        if len(tags) > 1:
            return tags
        pattern = re.compile(r"[A-Z][a-z]+|\d+|[A-Z]+(?![a-z])")
        return pattern.findall(tag)

    def extract_hashtag(self,text):

        hash_list = ([re.sub(r"(\W+)$", "", i) for i in text.split() if i.startswith("#")])
        word_list = []
        for word in hash_list:
            word_list.extend(self.split_hashtag_to_words(word))
        return word_list

    def _clean_hashtag(self,text):
        text = text.replace("#", " ").replace("_", " ")
        return text

    def clean_hashtag(self,df):
        df['data'] = df['data'].apply(lambda x: self._clean_hashtag(x))
        return df

    # 7--------emoji translation
    def is_emoji(self,word):
        if word in self.emojis_fr:
            return True
        else:
            if (re.sub(u'\ufe0f', '', word)) in self.emojis_fr:
                return True
            else:
                if (word + '\ufe0f') in self.emojis_fr:
                    return True
                else:
                    return False

    def add_space(self,text):
        return ''.join(' ' + char if self.is_emoji(char) else char for char in text).strip()

    def translate_emojis(self,words):
        word_list = list()
        words_to_translate = list()
        for word in words:
            t = self.emojis_fr.get(word.get('emoji'), None)
            if t is None:
                word.update({'translation': 'عادي', 'translated': True})
                # words_to_translate.append('normal')
            else:
                word.update({'translated': False, 'translation': t})
                words_to_translate.append(t.replace(':', '').replace('_', ' '))
            word_list.append(word)
        return word_list

    def emoji_unicode_translation(self,text):
        text = self.add_space(text)
        em_split_emoji = emoji.get_emoji_regexp().split(text)
        em_split_whitespace = [substr.split() for substr in em_split_emoji]
        words = functools.reduce(operator.concat, em_split_whitespace)
        # words = text.split()
        text_list = list()
        emojis_list = list()
        c = 0
        for word in words:
            if self.is_emoji(word):
                emojis_list.append({'emoji': word, 'emplacement': c})
            else:
                text_list.append(word)
            c += 1
        emojis_translated = self.translate_emojis(emojis_list)
        for em in emojis_translated:
            text_list.insert(em.get('emplacement'), em.get('translation'))
        text = " ".join(text_list)
        return text

    def emoji_trans(self,df):
        # for i in df.index:
        # df['Comment'][i] = emoji_unicode_translation(df['Comment'][i])
        df['data'] = df['data'].apply(lambda x: self.emoji_unicode_translation(x))
        return df

    # 8--------- latin script only
    def _latin(self,x):
        # x = re.sub(r'[^\x00-\x7f]',r'', x)
        x = re.sub(u'[^\\x00-\\x7F\\x80-\\xFF\\u0100-\\u017F\\u0180-\\u024F\\u1E00-\\u1EFF]', u'', x)
        return x

    def latin(self,df):
        df['data'] = df['data'].apply(lambda x: self._latin(x))
        return df

    # 9--------- commentaire vide
    def vide(self,df):
        indexNames = df[len(df['data']) == 1].index
        # indexNames = df[len(df['Comment']) <= 1 ].index
        df.drop(indexNames, inplace=True)
        return df

    # 10---------replace emoticons
    def _replace_emoticons(self,text):
        for emot in self.emot_fr:
            emoticon_pattern = r'(' + emot + ')'
            emoticon_words = self.emot_fr[emot]
            # print( emoticon_words)
            replace_text = emoticon_words.replace(",", "")
            replace_text = replace_text.replace(":", "")
            replace_text = replace_text.replace("/", "")
            replace_text = replace_text.replace("=", "")
            replace_text = replace_text.replace("(", "")
            replace_text = replace_text.replace(")", "")
            replace_text_list = replace_text.split()

            emoticon_name = '_'.join(replace_text_list)
            text = re.sub(emoticon_pattern, emoticon_name, text)
        return text

    def replace_emoticons(self,df):
        df['data'] = df['data'].apply(lambda x: self._replace_emoticons(x))
        return df

    # 11-----------nan
    def delete_vide(self,df):
        for i in df.columns:
            df[i][df[i].apply(lambda i: True if (
                        re.search('^\s*$', str(i)) or re.search('[a-zA-Z]', str(i)) == None) else False)] = None
        df.dropna(subset=["data"], inplace=True)
        return df

    # 12-----------delete stop words
    def _delete_stop_words(self,x):
        zen = TextBlob(x)
        words = zen.words
        x = " ".join([w for w in words if not w in self.stop_word_comp])
        zen = TextBlob(x)
        words = zen.words
        x = " ".join([word for word in words if word not in stopwords.words("french")])
        zen = TextBlob(x)
        words = zen.words
        x = " ".join([word for word in words if word not in stopwords.words("english")])
        return x

    def delete_stop_words(self,df):
        df['data'] = df['data'].apply(lambda x: self._delete_stop_words(x))
        return df

    # 14--------- delete accents
    def _delete_accents(self,x):
        x = unidecode.unidecode(x)
        return x

    def delete_accents(self,df):
        df['data'] = df['data'].apply(lambda x: self._delete_accents(x))
        return df

    # 13-----------duplicated comment
    def delete_dupplicated(self,df):
        df = df.drop_duplicates(subset=["Author", "data"], keep='first', inplace=False)
        return df


def get_comments(id_channel):
    key = "AIzaSyBPkPCltYAW6hXfAkMNfwfnZzQl-VbTNiM"
    YOUTUBE_API_SERVICE_NAME = "youtube"
    YOUTUBE_API_VERSION = "v3"

    service = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=key)
    comments, authors, sources, dates, viewerRating = [], [], [], [], []

    part = 'snippet'
    maxResults = 100
    textFormat = 'plainText'
    order = 'time'

    ch_request = service.channels().list(part='contentDetails', id=id_channel)
    ch_response = ch_request.execute()
    playlist_id = ch_response['items'][0]['contentDetails']['relatedPlaylists']['uploads']
    videos = []
    next_page_token = None
    while 1:
        res = service.playlistItems().list(
            part='snippet',
            playlistId=playlist_id,
            maxResults=50,
            pageToken=next_page_token
        ).execute()
        videos += res['items']
        next_page_token = res.get('nextPageToken')
        if next_page_token is None:
            break
    # loop the videos of the playlist channel
    for video in videos:
        videoId = video['snippet']['resourceId']['videoId']
        response = service.commentThreads().list(part=part, maxResults=maxResults, textFormat=textFormat, order=order,
                                                 videoId=videoId).execute()
        while response:
            for item in response['items']:
                commen = item['snippet']['topLevelComment']['snippet']
                comment = commen['textDisplay'].replace('\n', '')
                author = commen['authorDisplayName']
                date = commen['publishedAt']
                source = commen['videoId']

                # 6 append to lists
                comments.append(comment)
                authors.append(author)
                sources.append(source)
                dates.append(date)
                # new_comment =Comments(comment_data=comment,comment_author=author,comment_date=date,comment_source=source).save()
                # 8 check for nextPageToken, and if it exists, set response equal to the JSON response
            if 'nextPageToken' in response:
                response = service.commentThreads().list(part=part, maxResults=maxResults, textFormat=textFormat,
                                                         order=order, videoId=videoId,
                                                         pageToken=response['nextPageToken']).execute()
            else:
                break

        print(video['snippet']['resourceId']['videoId'])
        return 0

def home(request):
    if request == 'POST':
        get_comments('UCyVfMUt4tGYENMbqQ8Pusog')
    p = pretraitement()
    print(p._delete_URLs(text='hi f dahcra t haz https://stackoverflow.com/'))
    return render(request, 'home.html')

def result(request):
    cls= joblib.load('final_model.sav')
    loaded_cvec = joblib.load("finalized_countvectorizer.sav")
    feat_test = loaded_cvec.transform(['hmar hhh'])
    ans = cls.predict(feat_test)
    return render(request, 'result.html',{'ans':ans})
