from bson import ObjectId
from django.http import HttpResponse
from django.shortcuts import render
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.pipeline import make_pipeline
# Create your views here.
from django.views.decorators.csrf import csrf_exempt

from . models import Comments
import joblib

def home(request):
    return render(request, 'home.html')


