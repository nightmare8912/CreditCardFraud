import csv
import matplotlib.pyplot as plt
import numpy as np
import io
import pandas as pd
import copy
from django.shortcuts import render
from django.http import HttpResponseRedirect, HttpResponse, JsonResponse

from . import logics

df = pd.DataFrame()

def home(request):
    return render(request, 'home.html')

def upload(request):
    global df
    if request.method == 'POST' and request.FILES['csv_file']:
        csv_file = request.FILES['csv_file']
        print("FILE WAS READ")
        # csv_file_temp = copy.deepcopy(csv_file)
        csv_content = csv_file.read().decode('utf-8')
        csv_stringio = io.StringIO(csv_content)
        df = pd.read_csv(csv_stringio)

        csv_data = csv.reader(csv_file.read().decode('utf-8').splitlines())
        
        # Pass the CSV data to the template
        return render(request, 'main.html', {'csv_data': csv_data})
    
    return render(request, 'home.html')

def perform_eda(request):
    images = logics.eda(df)
    return JsonResponse({'eda_0': images[0], 'eda_1': images[1], 'eda_2': images[2]})

def run_algorithms(request):
    images = logics.run_algorithms(df)
    return JsonResponse({'log_reg_roc': images[0], 'xg_boost_roc': images[1], 'decision_tree_roc': images[2], 'cnn_learn_graph_1': images[3], 'cnn_learn_graph_2': images[4], 'cnn_roc': images[5]})