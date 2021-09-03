from django.db import models

# Create your models here.
class Comments(models.Model):
    comment_id = models.AutoField(primary_key=True, editable=True)
    comment_data = models.TextField()
    comment_author = models.CharField(max_length = 255)
    comment_source = models.CharField(max_length=255, null=True)
    comment_date = models.DateTimeField( null=True)
    comment_clean = models.TextField(null=True)
    comment_OFF = models.BooleanField(default=False)
    comment_degre_OFF = models.FloatField( default= 0.0 )


    def __str__(self):
        return self.comment_data

class FileUpload(models.Model):
    file_id = models.AutoField(primary_key=True, editable=True)
    csv = models.FileField(upload_to='csv')

    def __str__(self):
        return self.csv


