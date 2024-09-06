# MarketMinds © 2024 by Om Vekariya
# Licensed under the MarketMinds Proprietary Software License
# Commercial use, personal use, modification, and redistribution are prohibited.

from django.urls import path
from . import views

urlpatterns=[
    path('',views.index,name='index'),
    path('details/',views.details,name='details'),
    path('compare/',views.compare,name='compare'),
    path('download/<id>',views.download,name='download'),
    path('predict/',views.predict,name='predict'),
    path('company-search/', views.company_search, name='company-search'),
    path('company-suggest/', views.get_companies, name='get_company_info'),
    path('company-details/', views.get_company, name='get_company_info'),
]