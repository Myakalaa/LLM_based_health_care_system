# Create your views here.
from datetime import datetime, timedelta

from django.contrib import messages
from django.shortcuts import render, HttpResponse


from .forms import UserRegistrationForm,SymptomForm
from .models import UserRegistrationModel, TokenCountModel

SECRET_KEY = "ce9941882f6e044f9809bcee90a2992b4d9d9c21235ab7c537ad56517050f26b"
ALGORITHM = "HS256"
import random
import time
import socket



# Create your views here.
def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            print('Data is Valid')
            loginId = form.cleaned_data['loginid']
            TokenCountModel.objects.create(loginid=loginId, count=0)
            form.save()
            messages.success(request, 'You have been successfully registered')
            form = UserRegistrationForm()
            return render(request, 'UserRegistrations.html', {'form': form})
        else:
            messages.success(request, 'Email or Mobile Already Existed')
            print("Invalid form")
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})


def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        print("Login ID = ", loginid, ' Password = ', pswd)
        try:
            check = UserRegistrationModel.objects.get(loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                data = {'loginid': loginid}


                print("User id At", check.id, status)
                return render(request, 'users/UserHomePage.html', {})
            else:
                messages.success(request, 'Your Account Not at activated')
                return render(request, 'UserLogin.html')
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html', {})


def UserHome(request):
    return render(request, 'users/UserHomePage.html', {})


def viewDataset(request):
    from django.conf import settings
    import pandas as pd
    import os
    folder_path = os.path.join(settings.MEDIA_ROOT)  # Put your folder inside the project root
    # Read all CSV files and collect their DataFrames
    dataframes = []
    for file in os.listdir(folder_path):
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(folder_path, file))
            dataframes.append(df)

    # Combine all dataframes (outer join to handle different columns)
    if dataframes:
        combined_df = pd.concat(dataframes, axis=0, ignore_index=True, sort=False)
    else:
        combined_df = pd.DataFrame()  # Empty

    # Convert to HTML for template
    html_table = combined_df.to_html(classes='table table-bordered table-striped', index=False, na_rep='-')

    return render(request, 'users/viewdataset.html', {'data': html_table})



def medify_llm_results(request):
    from .utility.medify_ai import run_medify_ai_demo
    results = run_medify_ai_demo()
    return render(request, "users/medify_results.html", {"results": results})


def symptom_checker(request):
    result = None

    if request.method == "POST":
        form = SymptomForm(request.POST)
        if form.is_valid():
            symptoms = form.cleaned_data["symptoms"]
            from .utility.llm_test import  personalized_recommendation
            result = personalized_recommendation(symptoms)
    else:
        form = SymptomForm()

    return render(request, "users/symptom_checker.html", {"form": form, "result": result})