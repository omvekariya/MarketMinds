from django.urls import path
from . import views

urlpatterns=[
    path('',views.index,name='index'),
    path('details/',views.details,name='details'),
    path('compare/',views.compare,name='compare'),
    path('download/<id>',views.download,name='download'),
    path('predict/',views.predict,name='predict'),
]