from django.urls import path

from . import views


urlpatterns = [

    # Main dashboard
    path(
        "",
        views.dashboard_home,
        name="dashboard_home"
    ),

    # Company details
    path(
        "company/<str:ticker>/",
        views.company_detail,
        name="company_detail"
    ),

    # Sector analysis
    path(
        "sectors/",
        views.sector_analysis,
        name="sector_analysis"
    ),

    # Prediction and model performance
    path(
        "prediction/",
        views.prediction_analysis,
        name="prediction_analysis"
    ),
]