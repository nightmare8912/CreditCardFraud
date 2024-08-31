from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('upload/', views.upload, name='upload'),
    path('perform_eda/', views.perform_eda, name='perform_eda'),
    path('run_algorithms/', views.run_algorithms, name='run_algorithms'),
]