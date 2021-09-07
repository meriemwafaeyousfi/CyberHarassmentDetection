import os
import time
import pafy
import sys
from bson import ObjectId
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render, redirect
from django.template import Context
from django.views import View
from rest_framework import viewsets
from rest_framework.response import Response
#from rest_framework.authtoken.admin import User
from rest_framework.views import APIView
from django.views.generic import TemplateView
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.pipeline import make_pipeline
# Create your views here.
from django.views.decorators.csrf import csrf_exempt

from .forms import CommentForm
from . models import *
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
import json
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
from django.db import connection
from collections import defaultdict
from .serializers import commentsSerializer
import phonetics
import sklearn.cluster
import Levenshtein
from operator import eq, contains
from fuzzywuzzy import fuzz
from urllib.parse import parse_qs, urlparse
from sqlalchemy import create_engine
from django.conf import settings
from datetime import datetime
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from django.core.files.storage import FileSystemStorage
from django.core.files.storage import default_storage

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
            for line in lines:
                line = line.strip('\n').split('\t')
                self.stop_word_comp.append(line[0].strip())

        self.vocab =[]
        self.vocab = np.genfromtxt("media/vocab.txt", dtype=str,encoding=None, delimiter=",")

        self.dic = defaultdict(list)
        key = ""
        self.dic.setdefault(key, [])
        with open('media/dic.json') as json_file:
            self.dic = json.load(json_file)


    def _remove_punctuations(self,x):
        x = str(x)
        translator = str.maketrans(' ', ' ', self.punctuations_list)
        translator = str.maketrans(self.punctuations_list, ' ' * len(self.punctuations_list))
        return x.translate(translator)

    def remove_punctuations(self,df):
        df['comment_clean'] = df['comment_clean'].apply(lambda x: self._remove_punctuations(x))
        return df


    # 3--------remove doubles (plus que deux fois)
    def _remove_repeating_char(self,x):
        x = str(x)
        return re.sub(r'(.)\1{2,}', r'\1\1', x)

    def remove_repeating_char(self,df):
        df['comment_clean'] = df['comment_clean'].apply(lambda x: self._remove_repeating_char(x))
        return df

    # 5--------lower case
    def _lower_case(self,x):
        x = x.lower()
        return x

    def lower_case(self,df):
        df['comment_clean'] = df['comment_clean'].apply(lambda x: self._lower_case(x))
        return df

    # 5--------extra witespace
    def _extraWhite(self,x):
        x = re.sub('\t', ' ', x)
        x = re.sub('\s+', ' ', x)
        return x

    def extraWhite(self,df):
        df['comment_clean'] = df['comment_clean'].apply(lambda x: self._extraWhite(x))
        return df

    # 6.1--------delete urls
    def _delete_URLs(self,text):
        t = re.sub('http\S+\s*', ' ', text)
        return t

    def delete_URLs(self,df):
        df['comment_clean'] = df['comment_clean'].apply(lambda x: self._delete_URLs(x))
        return df

    # 6.1--------delete single letters and numbers
    def _delete_single_letters(self,text):
        text = re.sub(r'\b\d+(?:\.\d+)?\b', '', text)
        text = re.sub('(?<![\w])(?:[a-zA-Z0-9](?: |$))', '', text)
        return text

    def delete_single_letters(self,df):
        df['comment_clean'] = df['comment_clean'].apply(lambda x: self._delete_single_letters(x))
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
        df['comment_clean'] = df['comment_clean'].apply(lambda x: self._clean_hashtag(x))
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
        df['comment_clean'] = df['comment_clean'].apply(lambda x: self.emoji_unicode_translation(x))
        return df

    # 8--------- latin script only
    def _latin(self,x):
        # x = re.sub(r'[^\x00-\x7f]',r'', x)
        x2 =re.sub(u'[^\\x00-\\x7F\\x80-\\xFF\\u0100-\\u017F\\u0180-\\u024F\\u1E00-\\u1EFF\U0001F600-\U0001F64F|\U0001F300-\U0001F5FF|\U0001F680-\U0001F6FF|\U0001F190-\U0001F1FF|\U00002702-\U000027B0|\U0001F926-\U0001FA9F|\u200d|\u2640-\u2642|\u2600-\u2B55|\u23cf|\u23e9|\u231a|\ufe0f]', u'', x)
        if (x != x2):
            x= x2
            english_check = re.compile(r"[a-zA-Z]")
            if english_check.match(x):
                return x
            else:
                x=' '

        return x

    def latin(self,df):
        df['comment_clean'] = df['comment_clean'].apply(lambda x: self._latin(x))
        return df

    # 9--------- commentaire vide
    def vide(self,df):
        indexNames = df[len(df['comment_clean']) == 1].index
        df.drop(indexNames, inplace=True)
        return df

    # 10---------replace emoticons
    def _replace_emoticons(self,text):
        for emot in self.emot_fr:
            emoticon_pattern = r'(' + emot + ')'
            emoticon_words = self.emot_fr[emot]
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
        df['comment_clean'] = df['comment_clean'].apply(lambda x: self._replace_emoticons(x))
        return df

    # 11-----------nan
    def delete_vide(self,df):
        df['comment_clean'][df['comment_clean'].apply(lambda i: True if (
                        re.search('^\s*$', str(i)) or re.search('[a-zA-Z]', str(i)) == None) else False)] = None
        df.dropna(subset=["comment_clean"], inplace=True)
        return df
    def delete_vide_data(self,df):
        df['comment_data'][df['comment_data'].apply(lambda i: True if (
                        re.search('^\s*$', str(i)) or re.search('[a-zA-Z]', str(i)) == None) else False)] = None
        df.dropna(subset=["comment_data"], inplace=True)
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
        df['comment_clean'] = df['comment_clean'].apply(lambda x: self._delete_stop_words(x))
        return df

    # 14--------- delete accents
    def _delete_accents(self,x):
        x = unidecode.unidecode(x)
        return x

    def delete_accents(self,df):
        df['comment_clean'] = df['comment_clean'].apply(lambda x: self._delete_accents(x))
        return df

    # 13-----------duplicated comment
    def delete_dupplicated(self,df):
        df = df.drop_duplicates(subset=["comment_author", "comment_clean"], keep='first', inplace=False)
        return df

    def _arabizi_let(self,word):
        word = word.replace('2', 'a')
        word = word.replace('5', 'kh')
        word = word.replace('7', 'h')
        word = word.replace('8', 'gh')
        word = word.replace('9', 'q')
        word = word.replace('6', 't')
        word = word.replace('3a', 'a')
        word = word.replace('3e', 'e')
        word = word.replace('3i', 'i')
        word = word.replace('i3', 'i')
        word = word.replace('o3', 'o')
        word = word.replace('a3', 'a')
        word = word.replace('e3', 'e')
        word = word.replace('3', 'a')
        # word = word.replace('4', '')
        return word

    def arabizi_let(self,df):
        df['comment_clean'] = df['comment_clean'].apply(lambda x: self._arabizi_let(x))
        return df
    # arabizi preprocessing
    def vocabulary(self,df):
        comment = df['comment_clean']
        liste = []
        vocab = self.vocab
        for line in comment:
            x = word_tokenize(line)
            liste = np.array(x)
            vocab = np.unique(np.concatenate((vocab, liste)))
        diff = list(set(vocab) - set(self.vocab))
        self.vocab =vocab
        file = open("media/vocab.txt", "w+")
        writer = csv.writer(file)
        writer.writerow(self.vocab)
        file.close()
        return diff

    def phonitic_dic(self,words):
        for word in words:
            key = phonetics.metaphone(word)
            if key not in self.dic.keys():
                self.dic.update({key: [word]})
            else:
                self.dic[key].append(word)
        return self.dic

    def phonitic_group(self,df):
        v = self.vocabulary(df)
        dic = self.phonitic_dic(v)
        json.dump(self.dic, open("media/dic.json", 'w+'))
        return self.dic

# leveinshtein distance
    def read_dic(self):
        f = open('media/dic2.json' )
        dic = json.load(f)
        #df = pd.DataFrame(list(dic.items()), columns=['Key', 'Phonemes'])
        return dic

    def freq_vocabulary(self,df):
        comment = df['comment_clean']
        vocabulary = []
        liste = []
        for line in comment:
            x = word_tokenize(line)
            # print(line)
            liste = np.array(x)
            # nv =vocabulary.concat(liste.filter((item) => a.indexOf(item) < 0))
            # print (liste[1])
            # nouveau = set (vocabulary) - set (liste)
            vocabulary = np.concatenate((vocabulary, liste))

        freq = nltk.FreqDist(vocabulary)
        return freq

    def most_freq_word(self,vocab,df):
        freq = self.freq_vocabulary(df)
        max = 0
        for i in range(len(vocab)):
            if (freq[vocab[i]] > max):
                max = freq[vocab[i]]
                word = vocab[i]
        return word

    def _fuzzy_distance(self,df,line):
        grs = list()
        my_dict = dict()
        l = line
        while (len(l) > 0):
            g = []
            freq = self.most_freq_word(l,df)
            for name in l:
                if (fuzz.ratio(name, freq) >= 80):
                    g.append(name)
            my_dict[freq] = g
            l = list(set(l) - set(g))

        return my_dict

    def fuzzy_distance(self,df):
        dic = defaultdict(list)
        for i in df.index:
            x = df['Phonemes'][i]
            df['Phonemes'][i] = self._fuzzy_distance(x)
            dic.update(df['Phonemes'][i])
        return dic

    def keys_of_value(self,dct, value, ops=(eq, contains)):
        for k in dct:
            if ops[isinstance(dct[k], list)](dct[k], value):
                return k

    def _spell_check(self,line, dict_list):
        keys = list(dict_list.keys())
        vals = list(dict_list.values())
        max = 80
        key_max = 0
        word_list = word_tokenize(line)
        for value in word_list:
            if (value in [x for v in dict_list.values() for x in v]):
                if (self.keys_of_value(dict_list, value)):
                    word_list[word_list.index(value)] = self.keys_of_value(dict_list, value)
            else:
                key_max = 0
                for key in keys:
                    if (fuzz.ratio(value, key) > max):
                        max = fuzz.ratio(value, key)
                        key_max = key
                if (key_max != 0):
                    word_list[word_list.index(value)] = key_max
                    dict_list[key_max].append(value)
                else:
                    # ajouter dans dic
                    dict_list.update({value: [value, ]})
                    # dict_list[value].append(value)

        line = " ".join(word_list)
        json.dump(dict_list, open("dic2.json", 'w'))

        return line

    def spell_check(self,df, list_data):
        df['comment_clean'] = df['comment_clean'].apply(lambda x: self._spell_check(x, list_data))
        return df

def get_videos(playlist_id):
    key = "AIzaSyBPkPCltYAW6hXfAkMNfwfnZzQl-VbTNiM" #hide it
    YOUTUBE_API_SERVICE_NAME = "youtube"
    YOUTUBE_API_VERSION = "v3"

    service = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=key)
    videos = []
    videos_id=[]
    next_page_token = None
    while 1:
        res = service.playlistItems().list(part='snippet', playlistId=playlist_id, maxResults=50,
                                           pageToken=next_page_token).execute()
        videos += res['items']
        next_page_token = res.get('nextPageToken')
        if next_page_token is None:
            break
    # loop the videos of the playlist channel
    for video in videos:
        videoId = video['snippet']['resourceId']['videoId']
        videos_id.append(videoId)
    return videos_id

def get_comments_video(videoId):
    key = "AIzaSyBPkPCltYAW6hXfAkMNfwfnZzQl-VbTNiM"
    YOUTUBE_API_SERVICE_NAME = "youtube"
    YOUTUBE_API_VERSION = "v3"

    part = 'snippet'
    maxResults = 100
    textFormat = 'plainText'
    order = 'time'

    service = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=key)
    response = service.commentThreads().list(part=part, maxResults=maxResults, textFormat=textFormat, order=order,
                                             videoId=videoId).execute()
    allComments =[]
    while response:
        for item in response['items']:
            commen = item['snippet']['topLevelComment']['snippet']
            comment = commen['textDisplay'].replace('\n', '')
            author = commen['authorDisplayName']
            date = commen['publishedAt']
            source = commen['videoId']
            # 6 append to lists
            allComments.append([comment,author,date,source,comment])
            # new_comment =Comments(comment_data=comment,comment_author=author,comment_date=date,comment_source=source).save()
            # 8 check for nextPageToken, and if it exists, set response equal to the JSON response
        if 'nextPageToken' in response:
            response = service.commentThreads().list(part=part, maxResults=maxResults, textFormat=textFormat,
                                                     order=order, videoId=videoId,
                                                     pageToken=response['nextPageToken']).execute()
        else:
            break
    return allComments



def get_comments(id_channel):
    key = "AIzaSyBPkPCltYAW6hXfAkMNfwfnZzQl-VbTNiM"
    YOUTUBE_API_SERVICE_NAME = "youtube"
    YOUTUBE_API_VERSION = "v3"

    service = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=key)
    ch_request = service.channels().list(part='contentDetails', id=id_channel)
    ch_response = ch_request.execute()
    playlist_id = ch_response['items'][0]['contentDetails']['relatedPlaylists']['uploads']
    videosIds= get_videos(playlist_id)
    comments=[]
    for videoId in videosIds:
        comments_video= get_comments_video(videoId)
        comments+=comments_video
    return comments

def df_parallelize_run(df, func, num_cores=2):
    df_split = np.array_split(df, num_cores)
    pool = Pool(num_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df

def cleaning_comment(df):
    p = pretraitement()
    df = p._latin(df)
    df = p._replace_emoticons(df)
    df = p._remove_punctuations(df)
    df = p._remove_repeating_char(df)
    df = p._delete_URLs(df)
    df = p._clean_hashtag(df)
    df= p.emoji_unicode_translation(df)
    df = p._lower_case(df)
    df= p._latin(df)
    df = p._delete_stop_words(df)
    df = p._delete_accents(df)
    df = p._extraWhite(df)
    df= p._delete_single_letters(df)
    #arabizi prétraitement
    df = p._arabizi_let(df)
    #p.phonitic_group(df)
    data = p.read_dic()
    df=p._spell_check(df, data)
    #test = p.fuzzy_distance(data)
    #json.dump(test, open("media/dic2.json", 'w+'))
    #final_data = p.spell_check(df, test)
    return df


def cleaning(df):
    p = pretraitement()
    df = p.delete_dupplicated(df)
    df['comment_clean'] = df['comment_clean'].apply(lambda x: cleaning_comment(x))
    df = p.delete_vide(df)
    return df

def model(data_test):
    cls= joblib.load('final_model.sav')
    loaded_cvec = joblib.load("finalized_countvectorizer.sav")
    feat_test = loaded_cvec.transform(data_test)
    ans = cls.predict(feat_test)
    proba = cls.predict_proba(feat_test)
    return ans,proba



def home(request):
    #ar = get_comments('UCyVfMUt4tGYENMbqQ8Pusog')
    #df = pd.DataFrame(ar, columns=['comment_data', 'comment_author', 'comment_date', 'comment_source', 'comment_clean'])
    #df_cleaned = cleaning(df)
    # get the comments from the database
    #query = str(Comments.objects.all().query)
    #df1 = pd.read_sql_query(query, connection)
    #df2 = pd.concat([df1, df_cleaned]).drop_duplicates(subset=["comment_data", "comment_author", "comment_source", "comment_clean"], keep=False, inplace=False)
    #print(df_cleaned)

    #for i in range(len(df2)):
    #    comment = df['comment_data'][i]
     #   author = df['comment_author'][i]
      #  date = df['comment_date'][i]
       # source = df['comment_source'][i]
        #clean = df['comment_clean'][i]
        #new_comment = Comments(comment_data=comment, comment_author=author, comment_date=date, comment_source=source, comment_clean= clean).save()
    return render(request, 'home.html')



def result(request):
    ar = get_comments('UCyVfMUt4tGYENMbqQ8Pusog')
    df = pd.DataFrame(ar, columns=['comment_data', 'comment_author', 'comment_date', 'comment_source', 'comment_clean'])
    df_cleaned = cleaning(df)
    ans = model(df_cleaned['comment_clean'])
    return render(request, 'result.html',{'ans':ans})

def get_data(request, *args, **kwargs):
    data = {
        "sales": 100,
        "customers": 10,
    }
    return JsonResponse(data) # http response

def youtube_link(url):
    regex = (r'(https?://)?(www\.)?'
             '(youtube|youtu|youtube-nocookie)\.(com|be)/'
             '(watch\?.*?(?=v=)v=|embed/|v/|.+\?v=)?([^&=%\?]{11})')

    p = re.compile(regex)
    if (url == None):
        return False

    if (re.search(p, url)):
        return True
    else:
        return False

class ChartData(APIView):
    authentication_classes = []
    permission_classes = []

    def get(self, request, format=None):
        #qs_count = User.objects.all().count()
        labels = ["Saturday", "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        default_items = [1, 23, 2, 3, 12, 2,5]
        data = {
                "labels": labels,
                "default": default_items,
        }
        return Response(data)

def detect_langue(df,lenTotal):
    language =[]
    nbr = []
    i =0
    tableCount = []
    print("ldf_cleaned")

    for index, row in df.iterrows():
        english_check = re.compile(r"[a-zA-Z]")
        if (english_check.match(row['comment_data']) and len(row['comment_data'])>3) :
            print(row["comment_data"])
            b = TextBlob(row['comment_data'])
            lg = b.detect_language()
            if (lg != 'en' and lg !='fr' and lg !='ar'):
                tableCount.append(['Autres languages que Arabizi, Anglais ou Francais', (1*100)/lenTotal])
            else:
                if lg == 'en':
                    tableCount.append(['Anglais', (1*100)/lenTotal])
                if lg == 'fr':
                    tableCount.append(['Francais', (1*100)/lenTotal])
                if lg == 'ar':
                    tableCount.append(['Arabizi', (1*100)/lenTotal])
        else:
            tableCount.append(['Emojis', (1*100)/lenTotal])
        i = i +1
    d = pd.DataFrame(tableCount, columns=['lg', 'count'])
    grpTableCount = d.groupby('lg', as_index=False).agg({"count": "sum"})
    max = grpTableCount['count'].idxmax()
    #language = grpTableCount['lg'].values.tolist()
    #nbr = grpTableCount['count'].values.tolist()
    language = grpTableCount['lg'][max]
    nbr = grpTableCount['count'][max]
    return language,nbr


class DashboardView(View):
    def get(self, request,id, *args, **kwargs):
        offTable = []
        nonoffTable = []
        file = False
        language = []
        lgnbr = []

        print(type(id))
        if (id.isnumeric()):
            path = FileUpload.objects.get(file_id= id)
            print(path.csv)
            df = pd.read_csv(path.csv,encoding='utf-8-sig', delimiter=',', names=['comment_data','label','comment_source', 'comment_date',  'comment_author'])
            df['comment_clean'] = df['comment_data']
            file = True

        else:
            df = get_comments_video(id)
            df = pd.DataFrame(df, columns=['comment_data', 'comment_author', 'comment_date', 'comment_source',
                                       'comment_clean'])
            Comments.objects.filter(comment_source=id).delete()


        df_cleaned = cleaning(df)
        lenTotal = len(df)
        lenGarde = len(df_cleaned)
        language, lgnbr = detect_langue(df_cleaned,lenGarde)

        print(df_cleaned)
        label, proba = model(df_cleaned['comment_clean'])
        #df_cleaned['comment_OFF']= label
        #df_cleaned['comment_degre_OFF']= proba
        data = []
        dataOff = []
        tableCount = []
        userTableCount = []
        # delete the existiong comments of the same video in case they were changed or deleted in future

        i = 0
        nbr_off = 0
        nbr_nonoff = 0
        for index, row in df_cleaned.iterrows():
            date = datetime.strptime(row['comment_date'], '%Y-%m-%dT%H:%M:%SZ').strftime("%m/%d/%Y")
            if (label[i] == 'OFF'):
                off = 1
                tableCount.append([date, 1, 0])
                userTableCount.append([row['comment_author'], 1])
                nbr_off = nbr_off + 1
            else:
                off = 0
                tableCount.append([date, 0, 1])
                nbr_nonoff = nbr_nonoff + 1
            if(file ==False) :
                new_comment = Comments(comment_data=row['comment_data'], comment_author=row['comment_author'],
                                   comment_date=row['comment_date'], comment_source=row['comment_source'],
                                   comment_clean=row['comment_clean'], comment_OFF=off,
                                   comment_degre_OFF=proba[i][1]).save()

            if (label[i] == 'OFF'):
                date = datetime.strptime(row['comment_date'], '%Y-%m-%dT%H:%M:%SZ').strftime("%m/%d/%Y")
                row['comment_date'] = datetime.strptime(row['comment_date'], '%Y-%m-%dT%H:%M:%SZ')
                row['comment_degre_OFF'] = round(proba[i][1] * 100, 1)
                dataOff.append(row)
                offTable.append([date, 1])
            else:
                date = datetime.strptime(row['comment_date'], '%Y-%m-%dT%H:%M:%SZ').strftime("%m/%d/%Y")
                nonoffTable.append([date, 1])

            i = i + 1

        #df_cleaned['comment_OFF'] = label
        #df_cleaned['comment_degre_OFF'] = proba
        print(df_cleaned)
        data = df_cleaned.to_dict('records')

        df = pd.DataFrame(tableCount, columns=['date', 'offcount', 'nonoffcount'])
        df = df.sort_values(by="date")
        grpTableCount = df.groupby('date', as_index=False).agg({"offcount": "sum", "nonoffcount": "sum"})
        if len(grpTableCount) > 20:
            indexDrop = grpTableCount[(grpTableCount['offcount'] == 0) & (grpTableCount['nonoffcount'] <= 2)].index
            # indexDrop = indexDrop[indexDrop['nonoffcount'] <= 2 ].index
            grpTableCount.drop(indexDrop, inplace=True)

        df = pd.DataFrame(userTableCount, columns=['user', 'offcount'])
        print("here is off users")
        print(userTableCount)
        grpTableOffUsers = df.groupby('user', as_index=False).agg({"offcount": "sum"})

        maxHarceleur = max(grpTableOffUsers['offcount'].values.tolist())
        print(maxHarceleur)
        groupedDate = json.dumps(grpTableCount['date'].values.tolist())
        groupedOffCount = json.dumps(grpTableCount['offcount'].values.tolist())
        groupedNonoffCount = json.dumps(grpTableCount['nonoffcount'].values.tolist())
        groupedOffUsers = json.dumps(grpTableOffUsers['user'].values.tolist())
        groupedOffUsersCount = json.dumps(grpTableOffUsers['offcount'].values.tolist())
        lg = language
        lgcount = round(lgnbr,2)
        #data = json.dumps(data.tolist())
        print(data)
        page = request.GET.get('page', 1)
        paginator = Paginator(dataOff, 3)
        try:
            cmts = paginator.page(page)
        except PageNotAnInteger:
            cmts = paginator.page(1)
        except EmptyPage:
            cmts = paginator.page(paginator.num_pages)

        # dati = datetime.strptime('2021-08-25T12:56:32Z', '%b %d %Y %I:%M%p')
        # context = {"my_data": data}
        # data = json.dumps(data)
        total = df_cleaned.shape[0]
        pr_off = (nbr_off * 100) / total
        pr_nonoff = (nbr_nonoff * 100) / total

        # print(groupedNonoffCount)

        args = {'maxHarceleur':maxHarceleur, 'commentsOff': cmts, 'id': id, 'total': lenTotal ,'totalGarde': total, 'proff': round(pr_off, 2),
                'prnonoff': round(pr_nonoff, 2),'nbroff': nbr_off, 'lg': lg, 'lgcount': lgcount,
                'nbrnonoff': nbr_nonoff , 'date': groupedDate, 'offCount': groupedOffCount,
                'nonoffCount': groupedNonoffCount,'offUsersCount':groupedOffUsersCount,'offUsers':groupedOffUsers, 'file':file }
        return render(request, 'dashboard.html', args)

def analyse_comment(request, *args, **kwargs):
        text = request.session['text']
        df = pd.DataFrame([[text, "none", "none", "none", text]],
                                      columns=['comment_data', 'comment_author', 'comment_date', 'comment_source',
                                               'comment_clean'])
        df_cleaned = cleaning(df)
        label, proba = model(df_cleaned['comment_clean'])
        probaNonoff = round(proba[0][0], 3) * 100.
        probaOff = round(proba[0][1], 3) * 100.
        args = {'text': text, 'comment_label': label, 'probaOff': probaOff,'probaNonoff': probaNonoff}
        return render(request, 'analyse_comment.html', args)

def handle_uploaded_file(f):
    with open('upload/csv/'+f.name, 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)

class IndexView(View):
    template_name = 'index.html'

    def get(self, request, *args, **kwargs):
        form = CommentForm()
        return render(request, self.template_name, {"form": form})

    def post(self, request):
        labelStat = False
        if request.method == 'POST':
            print(request.POST)
            form = CommentForm(request.POST)
            #save_path = os.path.join(settings.MEDIA_ROOT, 'upload', file_obj.name)
            #path = default_storage.save(save_path, request.FILES['document'])
            #print(document.id)
            #handle_uploaded_file(file_obj)
            #fs = FileSystemStorage()
            #fs.save(file_obj.name, file_obj)

            #file_name = default_storage.save(file_obj.name, file_obj)
            #document = FileUpload.objects.create(csv=file_obj)
            #document.save()
            probaNonoff=0
            probaOff=0

            if form.is_valid():
                time.sleep(0.01)
                postVar = True
                text = form.cleaned_data['comment']

                if request.FILES.get("document"):
                    file_obj = request.FILES.get("document")
                    handle_uploaded_file(file_obj)
                    document = FileUpload.objects.create(csv='csv/' + file_obj.name)
                    print(document.file_id)
                    return redirect('dashboard', id=document.file_id)

                elif(youtube_link(text)):
                    video = pafy.new(text)
                    videoId =  video.videoid
                    #parse_qs(urlparse(text).query).get('v')
                    return redirect('dashboard', id=videoId)

                else:
                    labelStat = True
                    df= pd.DataFrame([[text,"none","none","none",text]], columns=['comment_data', 'comment_author', 'comment_date', 'comment_source','comment_clean'])
                    df_cleaned = cleaning(df)
                    label, proba = model(df_cleaned['comment_clean'])
                    probaNonoff = proba[0][0] * 100
                    probaNonoff=round(probaNonoff, 2)
                    probaOff = proba[0][1] * 100
                    probaOff= round(probaOff,2)

                    #form = CommentForm()
                    args = {'labelStat':labelStat,'form': form, 'text': text, 'comment_label':label, 'probaOff':probaOff, 'probaNonoff':probaNonoff, 'postVar' : postVar}
                    return render(request, self.template_name, args)

            else:
                return render(request, self.template_name)
        else:
            return render(request, self.template_name)

#comments view
class CommentsView(TemplateView):
    authentication_classes = []
    permission_classes = []
    #serializer_class = commentsSerializer

    def get(self, request):
        comments = list(Comments.objects.values())
        #serializer = commentsSerializer(comments, many=True)
        return JsonResponse(comments, safe=False)

class CommentsFileView(TemplateView):
    authentication_classes = []
    permission_classes = []
    #serializer_class = commentsSerializer

    def get(self, request, id):
        path = FileUpload.objects.get(file_id=id)
        print(path.csv)
        df = pd.read_csv(path.csv, encoding='utf-8-sig', delimiter=',',
                         names=['comment_data', 'label', 'comment_source', 'comment_date', 'comment_author'])
        print(df)
        comments =  df.to_dict('records')
        #serializer = commentsSerializer(comments, many=True)
        return JsonResponse(comments, safe=False)

class CommentsOffView(TemplateView):
    authentication_classes = []
    permission_classes = []
    #serializer_class = commentsSerializer

    def get(self, request):
        comments = list(Comments.objects.values().filter(comment_OFF=1))
        #serializer = commentsSerializer(comments, many=True)
        return JsonResponse(comments, safe=False)


#def index(request):
#    if request.user.is_authenticated:
#        return redirect('/')
 #   else :
  #      return render(request, 'index.html')


