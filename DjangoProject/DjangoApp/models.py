from django.db import models

# Create your models here.
class Comments(models.Model):
    comment_id = models.IntegerField()
    comment_data = models.TextField()
    comment_author = models.CharField(max_length = 255)
    comment_date = models.DateTimeField()
    comment_source = models.CharField(max_length = 255)

    def __str__(self):
        return self.comment_data
