from os.path import dirname, join

import pandas as pd
import pickle
from bokeh.layouts import layout, row, widgetbox, column
from bokeh.models import ColumnDataSource, CustomJS, Panel
from bokeh.models.widgets import Div,RangeSlider, Button, DataTable, TableColumn, NumberFormatter, CheckboxGroup, HTMLTemplateFormatter
from bokeh.io import curdoc
from bokeh.plotting import figure, show
from bokeh.client import push_session
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import itertools
global oldlen

oldlen=0

def autolabel_tab():
    global oldlen
    
    def change_label(df, col):
        df[col] = df[col].astype(str)
        df[col] = df[col].str.replace('-1','Unassigned')
        df[col] = df[col].str.replace('0','AutoTransact')
        df[col] = df[col].str.replace('1','BuyEtfs')
        df[col] = df[col].str.replace('2','ByVgMutualFunds')
        df[col] = df[col].str.replace('3','Close sdFunds')
        df[col] = df[col].str.replace('4','IndirectRollover')
        df[col] = df[col].str.replace('5','SettlementFundPurchase')
        df[col] = df[col].str.replace('6','AccountNotDisplayed')
        df[col] = df[col].str.replace('7','BankAuthentiationTiming')
        return df
    
    #df = pd.read_csv(join(dirname(__file__), 'salary_data.csv'))
    
    df = pickle.load(open("pd_google_vanguard_new.pkl", "rb" )) 
    
    #df['final_label']=df['a1_v']
    
    #df= df.query('a1==a1_v')
    df_ref = pickle.load(open("dfRef.pkl","rb"))
    
    #pd_question = pickle.load(open("df_question.pkl", "rb"))
    
    
    # join the df with the confirmed labels
    #df['Confirmed_Label']=pd_question['Labels']
    #df['Confirmed_Label']=df['a1']
    
    #df=change_label(df,'a1')
    #df=change_label(df,'a1_v')
    #df=change_label(df,'Confirmed_Label')
    
    #print (df)
    #df_ref = pd.read_csv(join(dirname(__file__),'ques_abbr.csv'))
    
    #print(df_ref)
    source = ColumnDataSource(data=dict())
    sourceref = ColumnDataSource(data=dict())
    
    check_HideAgree = CheckboxGroup(labels=["Hide Agreed Auto-labels "], active=[0, 1], width=100)
    
    oldlen = len(check_HideAgree.active)    
    
    



    def plot_confusion_matrix(cm,target_names,title='Confusion matrix',cmap=None,normalize=True):

        accuracy = np.trace(cm) / float(np.sum(cm))
        misclass = 1 - accuracy

        if cmap is None:
            cmap = plt.get_cmap('Blues')

        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()

        if target_names is not None:
            tick_marks = np.arange(len(target_names))
            plt.xticks(tick_marks, target_names, rotation=45)
            plt.yticks(tick_marks, target_names)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


        thresh = cm.max() / 1.5 if normalize else cm.max() / 2
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
            else:
                plt.text(j, i, "{:,}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")


        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
        plt.show()
        
    
    def update(attrname,old,new):
        switch = check_HideAgree.active
        #print (switch)
    
        if (len(switch)==1):
            current = df.query('a1!=a1_v')
        else:
            current = df
        
        #current = df[(df['salary'] >= slider.value[0]) & (df['salary'] <= slider.value[1])].dropna()
        #current = df
        source.data = {
            'Question'             : current.Question,
            'a1'           :         current.a1,
            'a1_v'         :         current.a1_v,
            'Confirmed_Labels'  :     current.Confirmed_Labels
        }
    
        currentref=df_ref
        sourceref.data={
             'id': currentref.id,
              'nickname':currentref.nickname,
              'title':currentref.title
        }
        
        #oldlen = len(check_HideAgree.active)
        
    
    template="""
    <div style="background:<%= 
        (function colorfromint(){
            if ((a1_v == Confirmed_Labels) && (Confirmed_Labels!='Unassigned')){
                return(" darkseagreen")}
            else if (Confirmed_Labels=='Unassigned') {return("lightgrey")}
            else{return("red")}
            }()) %>; 
        color: white"> 
    <%= value %></div>
    """
    
    #template="""
    #            <div style="background:<%= 
    #                (function colorfromint(){
    #                    if(a1 <> a1_v ){
    #                        return("green")}
    #                    }()) %>; 
    #                color: black"> 
    #            <%= value %>
    #            </div>
    #            """
    #            
    #
    #            
    formatter =  HTMLTemplateFormatter(template=template)
    
    
                
    #slider = RangeSlider(title="Min Score", start=10000, end=110000, value=(10000, 50000), step=1000, format="0,0")
    #slider.on_change('value', lambda attr, old, new: update())
    
    #button_save = Button(label="Save", button_type="success")
    #
    #button_save.on_click(save_data)
    #
    
    button = Button(label="Download", button_type="success", width=100)
    
    button.callback = CustomJS(args=dict(source=source),
                               code=open(join(dirname(__file__), "download.js")).read())

    
    columns = [
        TableColumn(field="Question", title="Customer Question",width=700),
        TableColumn(field="a1", title="Rank 1 Intent Google",width=100),
        TableColumn(field="a1_v", title="Rank 1 Intent Vanguard", width=100),
        TableColumn(field="Confirmed_Labels", title="Confirmed_Labels",formatter=formatter, width=100) 
    ]
    
    columnsref = [
        TableColumn(field="id", title="id", width=100),
        TableColumn(field="nickname", title="nickname", width=200),
        TableColumn(field="title", title="title", width=700)
    ]
    
    data_table = DataTable(source=source, columns=columns, editable=True, height=600,width=1300, fit_columns=False)
    ref_table = DataTable(source=sourceref, columns=columnsref, editable=True, width=1000)
    
    
    div=Div(text="""Auto-Labels""", width=100, height=10)
    
    div2 = Div(text="""Intent Table""", width=100, height=10)
    controls = widgetbox(button)

    check_HideAgree.on_change('active',update)
    #table = widgetbox(data_table)
    #ref = widgetbox(ref_table)
    #d= widgetbox(div)
    
    
    doc_layout = layout([[column(row(column(widgetbox(div),widgetbox(check_HideAgree),widgetbox(button)),widgetbox(data_table)),row(widgetbox(div2),widgetbox(ref_table)))]],sizing_mode='scale_width')
    #curdoc().add_root(column(row(d, table),ref))
    #curdoc().add_root(ref)
    
    #curdoc().add_root(doc_layout)
    #curdoc().title = "Export CSV"
    #update()
    
    
    def on_change_data_source(attr, old, new):
        global oldlen
        print (len(check_HideAgree.active))
        if (len(check_HideAgree.active)!=oldlen):
            print ("ENTER")
#            #print('-- OLD DATA: {}'.format(old))
#            #print('-- NEW DATA: {}'.format(new))
#            #print('-- SOURCE DATA: {}'.format(source.data))
#        
#            # to check changes in the 'y' column:
#            indices = list(range(len(old['final_label'])))
#            changes = [(i,j,k) for i,j,k in zip(indices, old['final_label'], new['final_label']) if j != k]
#            if changes != []:
#                for t in changes:  # t = (index, old_val, new_val)
#                    patch = {
#                        'final_label' : [(t[0], t[2]), ]   # the new value is received as a string
#                    }
#                    source.patch(patch)   # this will call to this callback again, ugly
#                                      # so you will need to update the values on another source variable
#            #print('-- SOURCE DATA AFTER PATCH: {}'.format(source.data))
        oldlen = len(check_HideAgree.active)
               
    source.on_change('data', on_change_data_source)
    
    tab = Panel(child=doc_layout, title = 'AutoLabel')
    
    return tab
