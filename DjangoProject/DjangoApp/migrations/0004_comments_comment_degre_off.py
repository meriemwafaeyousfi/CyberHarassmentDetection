# Generated by Django 3.2.5 on 2021-08-23 15:31

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('DjangoApp', '0003_comments_comment_off'),
    ]

    operations = [
        migrations.AddField(
            model_name='comments',
            name='comment_degre_OFF',
            field=models.FloatField(default=0.0),
        ),
    ]
