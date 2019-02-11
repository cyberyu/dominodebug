# Pandas for data management
import pandas as pd

# os methods for manipulating paths
from os.path import dirname, join

# Bokeh basics 
from bokeh.io import curdoc
from bokeh.models.widgets import Tabs


# Each tab is drawn by one script
from active import active_learning_tab


# Create each of the tabs
#tab1 = w2v_tab()
#tab4 = query_tab()
#tab2 = autolabel_tab()
#tab3 = classify_tab()
tab4 = active_learning_tab()


# Put all the tabs into one application
tabs = Tabs(tabs = [tab4])

# Put the tabs in the current document for display
curdoc().add_root(tabs)


