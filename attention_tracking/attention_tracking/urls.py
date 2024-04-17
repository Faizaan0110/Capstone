from django.urls import path
from ..attention import views

urlpatterns = [
    path('your-url/', views.your_view_function, name='view_name'),
]
