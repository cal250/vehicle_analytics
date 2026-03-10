from django.urls import path
from . import views

urlpatterns = [
    path('', views.data_exploration_view, name='data_exploration'),
    path('regression/', views.regression_analysis, name='regression_analysis'),
    path('classification/', views.classification_analysis, name='classification_analysis'),
    path('clustering/', views.clustering_analysis, name='clustering_analysis'),
    path('cluster-analytics/', views.cluster_analytics, name='cluster_analytics'),
]
