"""DjangoProject URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.conf.urls import url
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import path, include
from . import views
from .views import *
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    #path('',views.home, name='home'),
    path('dashboard/<id>', DashboardView.as_view(), name='dashboard'),
    #url(r'^chart/$', HomeView.as_view(), name='chart'),
    url(r'^comments/$', CommentsView.as_view(), name='comments'),
    url(r'^commentsOff/$', CommentsOffView.as_view(), name='commentsOff'),
    url(r'^commentsFile/<id>/$', CommentsFileView.as_view(), name='commentsFile'),
    path('result',views.result, name='result'),
    path('analyse_comment',analyse_comment, name='commentAna'),
    url(r'^', IndexView.as_view(), name='index'),
    url(r'^api/data/$', get_data, name='api-data2'),
    url(r'^api/chart/data/$', ChartData.as_view(),name='api-data'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root= settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
