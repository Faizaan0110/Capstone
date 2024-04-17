# attention_tracker/views.py
from django.shortcuts import render
from ..attention.models import AttentionRecord

def display_data(request):
    records = AttentionRecord.objects.all()
    return render(request, 'attention_tracker/data.html', {'records': records})
