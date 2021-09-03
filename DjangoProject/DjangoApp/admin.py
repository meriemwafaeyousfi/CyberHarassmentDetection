from django.contrib import admin
from .models import Comments, FileUpload

# Register your models here.

admin.site.register(Comments)
admin.site.register(FileUpload)