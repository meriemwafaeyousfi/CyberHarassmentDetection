from django import forms

class CommentForm(forms.Form):

    comment = forms.CharField(widget=forms.TextInput(
        attrs={
    'class' : 'form-control',
    'id': 'form-comment',
    'name': 'comment',
    'placeholder': 'Entrer un comentaire, lien http, fichier .csv/.txt',
    'aria-describedby': 'basic-addon2'
        } ))

