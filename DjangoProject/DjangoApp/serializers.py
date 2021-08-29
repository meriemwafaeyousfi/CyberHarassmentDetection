from rest_framework import serializers

from .models import *

class commentsSerializer(serializers.ModelSerializer):
    class Meta:
        model                = Comments
        fields               = '__all__'

