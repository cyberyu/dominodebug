from os.path import dirname, join

import pandas as pd
import pickle
from bokeh.layouts import layout, row, widgetbox, column
from bokeh.models import ColumnDataSource, CustomJS, Panel
from bokeh.models.widgets import Div,RangeSlider, Button, DataTable, TableColumn, NumberFormatter, CheckboxGroup, HTMLTemplateFormatter
from bokeh.io import curdoc
from bokeh.plotting import figure, show
from bokeh.client import push_session
from sklearn.svm import SVC
import numpy as np

oldlen=0

def classify_tab():
    
    
    def predict_func():
        
        Ktrain=kernel[np.ix_(i_train,i_train)]
        Ktest=kernel[np.ix_(i_test, i_train)]
        y=df_question["Labels"].values
        y_train=y[np.ix_(i_train)]
        
        clf = SVC(kernel='precomputed', probability=True, C=1)
        clf.fit(Ktrain, y_train)
        y_pred = clf.predict(Ktest)
        y_pred_proba = clf.predict_proba(Ktest)
        
        p_f = pd.DataFrame(np.column_stack([y_pred, y_pred_proba]))
        p_f.columns=['Labels','p0','p1','p2','p3','p4','p5','p6']
        p_f['Labels']=p_f['Labels'].apply(int)
        
        
        d_f=df_question['Question'].iloc[i_test]
        d_f.reset_index(drop=True, inplace=True)
        p_f.reset_index(drop=True, inplace=True)
        df_predict=pd.concat([d_f, p_f], axis=1)
        
        return df_predict
    
    
    
    def show_predict():
        
        current = predict_func()
        
        source_test.data = {
            'Question'             : current.Question,
            'Labels'       : current.Labels,
            'p0' :                   current.p0,
            'p1'         :         current.p1,
            'p2'         :         current.p2,
            'p3'  :         current.p3,
            'p4'  :         current.p4,
            'p5'  :         current.p5,
            'p6'  :         current.p6,
        }

    kernel = pickle.load(open("Dfix.pkl", "rb" ))
    df_question = pickle.load(open("df_question.pkl", "rb"))
    
    df_question["p0"]=""
    df_question["p1"]=""
    df_question["p2"]=""
    df_question["p3"]=""
    df_question["p4"]=""
    df_question["p5"]=""
    df_question["p6"]=""
    
    
    # find the indicies of -1

    i_test=df_question.index[df_question['Labels'] == -1].tolist()
    i_train=df_question.index[df_question['Labels']!=-1].tolist()
    
    
    current_train = df_question.iloc[i_train]
    current_test = df_question.iloc[i_test]
    
    columns_train = [
        TableColumn(field="Question", title="Customer Question",width=700),
        TableColumn(field="Labels", title="Label",width=100)
    ]

    columns_test = [
        TableColumn(field="Labels", title="Predicted Labels",width=100),
        TableColumn(field="p0", title="p0",formatter=NumberFormatter(format="0.00"),width=50),
        TableColumn(field="p1", title="p1",formatter=NumberFormatter(format="0.00"),width=50),
        TableColumn(field="p2", title="p2",formatter=NumberFormatter(format="0.00"),width=50),
        TableColumn(field="p3", title="p3",formatter=NumberFormatter(format="0.00"),width=50),
        TableColumn(field="p4", title="p4",formatter=NumberFormatter(format="0.00"),width=50),
        TableColumn(field="p5", title="p5",formatter=NumberFormatter(format="0.00"),width=50),
        TableColumn(field="p6", title="p6",formatter=NumberFormatter(format="0.00"),width=50)
           ]  

    button = Button(label="Predict", button_type="success", width=100)
    button.on_click(show_predict)
    
    

        
    source_train = ColumnDataSource(data=dict())
    source_test = ColumnDataSource(data=dict())
    
    data_table_train = DataTable(source=source_train, columns=columns_train, editable=False, height=300,width=1300, fit_columns=False)
    data_table_test = DataTable(source=source_test, columns=columns_test, editable=False, height=400,width=1300, fit_columns=False)

    def update():

        #current = df[(df['salary'] >= slider.value[0]) & (df['salary'] <= slider.value[1])].dropna()
        #current = df
        source_train.data = {
            'Question'             : current_train.Question,
            'Labels'  :         current_train.Labels
        }

        source_test.data={
             'Question':    current_test.Question,
              'Labels':     current_test.Labels
       }
    
    div_train=Div(text="""Training Data""", width=100, height=10)
    div_test=Div(text="""Test Data""", width=100, height=10)
    
    doc_layout = layout([[column(row(column(widgetbox(div_train),widgetbox(button)),widgetbox(data_table_train)),row(widgetbox(div_test),widgetbox(data_table_test)))]],sizing_mode='scale_width')
    
    tab = Panel(child=doc_layout, title = 'PredictLabel')
    
    update()
    
    return tab