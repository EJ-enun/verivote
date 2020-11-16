###############################################################################
#                                MAIN                                         #
###############################################################################

# Setup
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import cv2
from tensorflow.keras.models import load_model
from python.camera import VideoCamera
from flask import Flask, Response, render_template
import json
import pandas as pd

from settings import config, about
from python.data import Data
from python.model import Model
from python.result import Result
from python.face_processing import Face
from python.videostream import VideoStream
from python.fps import FPS
import pyqb


# def Client(url="http://www.quickbase.com", database=None, proxy=None, user_token=None):
qbc = pyqb.Client(url='https://hackathon20-eenun.quickbase.com', user_token='b5sfea_pd9m_bwkh5m6vbmwb5d83w4m3bjxftgu')
# Below authenticate is not required if `user_token` argument is passed to pyqb.Client() above
#qbc.authenticate(username='enunenun', password='Livingright1!')


model = 'model'
# Read data
data = Data()
data.get_data()
#face = Face()

def getCam():
    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
      raise IOError("Cannot open webcam")

    while True:
      ret, frame = cap.read()
      frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
      cv2.imshow('Input', frame)
      
      c = cv2.waitKey(1)
      if c == 27:
         break

    cap.release()
    cv2.destroyAllWindows()

def getPresidency():
    presidency = qbc.doquery(query='{"6".CT."Presidency"}', database='bqzcn3837')
    count = 0 
    presidency_keys = list(presidency.keys())
        
    presidency = presidency[presidency_keys[6]] 
    presidency = [y['candidates'] for y in presidency]

    return dict.fromkeys(presidency, [x for x in range(0, len(presidency))])    


    #return format(presidency)
def getSenate(state):
    #gubernatorial = 'Guber'
    senate = ''
    if(state == "Cross River"):
         senate = qbc.doquery(query='{"6".CT."Cross River Senate"}', database='bqzcn3837')
    if(state == "Delta"):
         senate = qbc.doquery(query='{"6".CT."Delta Senate"}', database='bqzcn3837')
    elif(state == "Imo"):
         senate = qbc.doquery(query='{"6".CT."Imo Senate"}', database='bqzcn3837')
    
    senate_keys = list(senate.keys())
        
    senate = senate[senate_keys[6]] 

    senate = [y['candidates'] for y in senate]

    return dict.fromkeys(senate, [x for x in range(0, len(senate))])    

def getGuber(state):
    guber = ' '
    
    
    if(state == "Cross River"):
         guber = qbc.doquery(query='{"6".CT."Cross River Guber"}', database='bqzcn3837')
    if(state == "Delta"):
         guber = qbc.doquery(query='{"6".CT."Delta Guber"}', database='bqzcn3837')
    elif(state == "Imo"):
         guber = qbc.doquery(query='{"6".CT."Imo Guber"}', database='bqzcn3837')
    
    guber_keys = list(guber.keys())
        
    guber = guber[guber_keys[6]] 

    guber = [y['candidates'] for y in guber]

    return dict.fromkeys(guber, [x for x in range(0, len(guber))])    

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

server = Flask(__name__)        
# App Instance
app = dash.Dash(name=config.name, assets_folder=config.root+"/application/static", external_stylesheets=[dbc.themes.LUX, config.fontawesome], server=server)
app.title = config.name

@server.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# Navbar
navbar = dbc.Nav(className="nav nav-pills", children=[
    ## logo/home
    dbc.NavItem(html.Img(src=app.get_asset_url("logo.PNG"), height="60px")),
    ## about
    dbc.NavItem(html.Div([
        dbc.NavLink("About", href="/", id="about-popover", active=False),
        dbc.Popover(id="about", is_open=False, target="about-popover", children=[
            dbc.PopoverHeader("How it works"), dbc.PopoverBody(about.txt)
        ])
    ])),
    ## links
    dbc.DropdownMenu(label="Links", nav=True, children=[
        dbc.DropdownMenuItem([html.I(className="fa fa-linkedin"), "  Contacts"], href=config.contacts, target="_blank"), 
        dbc.DropdownMenuItem([html.I(className="fa fa-github"), "  Code"], href=config.code, target="_blank")
    ])
])



# Input
presidential_in = dbc.FormGroup([
    html.H4("Presidential Candidates"),
    dcc.Dropdown(id="presidential_in")
    ])


# Input
state_in = dbc.FormGroup([
    html.H4("Gubernatorial Candidates"),
    dcc.Dropdown(id="guber_in",)
    ])

# Input
senate_in = dbc.FormGroup([
    html.H4("Senate Candidates"),
    dcc.Dropdown(id="senate_in",)
    ]) 



# App Layout
app.layout = dbc.Container(fluid=True,  style={'backgroundColor': 'F2F2F2'} , children=[
    ## Top
    html.H1(config.name, id="nav-pills"),
    navbar,
    html.Br(),html.Br(),html.Br(),

    ## Body
    dbc.Row([
        ### input + panel
        dbc.Col(md=3, children=[
            presidential_in, state_in, senate_in,
            #html.Div(id='container-button-basic',
            #children='Choose down ballot and click vote'), 
            html.Button('VOTE', id='submit-val', n_clicks=0),
            html.Br(),html.Br(),html.Br(),
            html.Div(id="panel", className="text-danger")
        ]),

        #dbc.Col(md=3, children=[state_in,]),


        ### plots
        dbc.Col(md=9, children=[
             #dbc.Col(md=3, children=[lga_in,]),
            dbc.Col(html.H4("Face Recognition"), width={"size":6,"offset":3}), 
            dcc.Tabs(id='tabs', value='tab-1', children=[
                dcc.Tab(label='Close Camera', value='close_camera', children=[html.Div(id='tabs-example-content')]), #turn camera off.
                dcc.Tab(label='Open camera', value='open_camera')]), #turn camera on.])
            html.Img(id='vid', title='Feed Closed')      
        ])
    ])
])



# Python functions for about navitem-popover
@app.callback(output=Output("about","is_open"), inputs=[Input("about-popover","n_clicks")], state=[State("about","is_open")])
def about_popover(n, is_open):
    if n:
        return not is_open
    return is_open

@app.callback(output=Output("about-popover","active"), inputs=[Input("about-popover","n_clicks")], state=[State("about-popover","active")])
def about_active(n, active):
    if n:
        return not active
    return active

@app.callback([Output('panel', 'children'), Output('presidential_in', 'options'),Output('guber_in', 'options'), Output('senate_in', 'options')], [Input('tabs', 'value')])
def render_content(tab):
    if tab == 'close_camera':
        query = qbc.doquery(query="{'6'.EX.'Bassey'}", database='bqzcn7xzi')
        query_keys = list(query.keys())
        #query_list = list(query)
        #query_list_0 = query_list['record']
        #query = dict(query)
        query = query[query_keys[6]] 
        panel = html.Div([
          html.H4('   Voter Data'),
          dbc.Card(body=True, className="text-white bg-primary", children=[
            
            html.H6("Names:", style={"color":"white"}),
            html.H3("{}".format(query['names']), style={"color":"white"}),
            
            html.H6("Other Names:", style={"color":"white"}),
            html.H3("{}".format(query['other_names']), style={"color":"white"}),
            
            html.H6("State of Residence:", style={"color":"white"}),
            html.H3("{}".format(query['state_of_residence']), style={"color":"white"}),
            
            html.H6("State Of Origin:", style={"color":"white"}),
            html.H3("{}".format(query['state_of_origin']), style={"color":"white"}),
            
            html.H6("LGA Of Origin:", style={"color":"white"}),
            html.H3("{}".format(query['lga_of_origin']), style={"color":"white"}),

            html.H6("PVC Number:", style={"color":"white"}),
            html.H3("{}".format(query['pvc_number']), style={"color":"white"})
            
            
          ])
        ])
        candidates_president = [{'label': i, 'value' : i} for i in getPresidency()]
        candidates_senate = [{'label': i, 'value' : i} for i in getSenate(query['state_of_origin'])]
        candidates_guber = [{'label': i, 'value' : i} for i in getGuber(query['state_of_origin'])]
        return panel, candidates_president, candidates_guber , candidates_senate
        

        #face.close_capture()
        #return 'Output {}:'.format()
        #return format(query)
    elif tab == 'open_camera':
        #vs = Video(src=0).start()
        #time.sleep(2.0)
        #fps = Fps().start()

        return html.Div([html.H1('Camera Open')])
"""        
@app.callback(output=Output('tabs-example-content','children'), inputs=[Input('tabs-example','value')])
def render_tab_content_example(tab):
    if tab is not None:
      return getPresidency()
"""
# Python function to plot active cases
@app.callback(output=Output("plot-active","figure"), inputs=[Input("country","value")])
def plot_active_cases(country):
    data.process_data(country) 
    model = Model(data.dtf)
    model.forecast()
    model.add_deaths(data.mortality)
    result = Result(model.dtf)
    return result.plot_active(model.today)

#Call back function for VOTE button
@app.callback(
    dash.dependencies.Output('container-button-basic', 'children'),
    [dash.dependencies.Input('submit-val', 'n_clicks')],
    [dash.dependencies.State('input-on-submit', 'value')])

def update_output(n_clicks, value):
    return 'The input value was "{}" and the button has been clicked {} times'.format(
        value,
        n_clicks)

