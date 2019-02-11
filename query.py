from bokeh.io import curdoc
from bokeh.layouts import layout, row, widgetbox, column
from bokeh.models import ColumnDataSource, CustomJS, Panel
from bokeh.plotting import figure, show
from bokeh.client import push_session
from bokeh.models.widgets import (CheckboxGroup, Slider, RangeSlider,Tabs, CheckboxButtonGroup, TableColumn, DataTable, Select, Button, TextInput, Div)
from sklearn.svm import SVC
from gensim.models import KeyedVectors
import numpy as np
global model
import pickle
from random import randint
import numpy as np
from gensim.similarities import WmdSimilarity



def query_tab():
	
	source = ColumnDataSource(data=dict())
	
	df_ref = pickle.load(open("dfRef.pkl","rb"))
	sourceref = ColumnDataSource(data=dict())
	
	button = Button(label="Query", button_type="success", width=100)
    button.on_click(show_query)
	
	#define instance of similarity object
	instance = WmdSimilarity(ref_corpus, model, num_best=10)
	
	def Query_new(sent):
	    query = Preprocess(sent)
	    sims = instance[query]
	    #sims[-1][0]
	    #sims[-1][1]
	    dfResult = pd.DataFrame(sims,columns=['match','score'])
	    dfCombined = pd.merge(dfResult,dfRefMod,left_on='match',right_on='index')
	    dfCombined = dfCombined[['title','score']]
	    print(dfCombined)
	
	
	def show_query():
		
		
		

	