from django.db import models

# Create your models here.
class Comments(models.Model):
    comment_id = models.AutoField(primary_key=True, editable=True)
    comment_data = models.TextField()
    comment_author = models.CharField(max_length = 255)
    comment_source = models.CharField(max_length=255, null=True)
    comment_date = models.DateTimeField( null=True)


    def __str__(self):
        return self.comment_data


