from apiclient.discovery import build
import json
from csv import writer
from csv import QUOTE_ALL
from apiclient.discovery import build
from urllib.request import urlopen
from urllib.parse import urlencode

def build_service():
    key = "AIzaSyBPkPCltYAW6hXfAkMNfwfnZzQl-VbTNiM"
    YOUTUBE_API_SERVICE_NAME = "youtube"
    YOUTUBE_API_VERSION = "v3"
    return build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=key)

def get_comments(part='snippet',
                 maxResults=100,
                 textFormat='plainText',
                 order='time',
                 videoId='YFK8qv8h_yA',
                 csv_filename="data"):
    # 3 create empty lists to store desired information
    comments, authors, sources, dates, viewerRating = [], [], [], [], []
    # build our service from path/to/apikey
    service = build_service()

    # 4 make an API call using our service
    response = service.commentThreads().list(part=part, maxResults=maxResults, textFormat=textFormat, order=order,
                                             videoId=videoId).execute()

    while response:  # this loop will continue to run until you max out your quota

        for item in response['items']:
            # 5 index item for desired data features
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

            # 7 write line by line
           # with open('testit.csv', 'a+', encoding='utf-8-sig') as f:
            #    csv_writer = writer(f)
             #   csv_writer.writerow([source, date, author, comment])

            # 8 check for nextPageToken, and if it exists, set response equal to the JSON response
        if 'nextPageToken' in response:
            response = service.commentThreads().list(
                part=part,
                maxResults=maxResults,
                textFormat=textFormat,
                order=order,
                videoId=videoId,
                pageToken=response['nextPageToken']
            ).execute()
        else:
            break

    # 9 return our data of interest
    return {
        'Sources': sources,
        'Date': dates,
        'Author name': authors,
        'Comments': comments,
    }




