###############################################################################
#                                MAIN                                         #
############################################################################### 


# Setup
from PIL import Image
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import cv2
import dash_bootstrap_components as dbc
from tensorflow.keras.models import load_model
#from python.camera import VideoCamera
from flask import Flask, Response, render_template
import json
from mtcnn.mtcnn import MTCNN
from settings import config, about
import numpy as np
import pyqb
import datetime
import base64
import io

# def Client(url="http://www.quickbase.com", database=None, proxy=None, user_token=None):
qbc = pyqb.Client(url='https://hackathon20-eenun.quickbase.com', user_token='b5sfea_pd9m_bwkh5m6vbmwb5d83w4m3bjxftgu')
#Below authenticate is not required if `user_token` argument is passed to pyqb.Client() above
#qbc.authenticate(username='enunenun', password='Livingright1!')

def face_extractor(img):
    # Function detects faces and returns the cropped face 
    # If no face detected, it returns the input image
    face = imread(img)
    #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(face, 1.3, 5)
    # perform face detection
    #bboxes = classifier.detectMultiScale(pixels)
    # print bounding box for each detected face
    for face in faces:
        print(face) 

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
server = Flask(__name__)        
# App Instance
app = dash.Dash(name=config.name, assets_folder=config.root+"/application/static", external_stylesheets=[dbc.themes.LUX, config.fontawesome], server=server)
app.title = config.name



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

upload = html.Div([
    dcc.Upload(
        id='upload-image',
        children=html.Div([
            'Take Photo or ',
            html.A('Select Image')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),
    html.Div(id='output-image-upload'),
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
            upload, html.Div(id='tabs-example-content')
            
        ])
    ])
])
        
def parse_contents(contents, filename, date):
    # Take in base64 string and return cv image
    encoded_image = contents.split(",")[1]
    decoded_image = base64.b64decode(encoded_image)
    bytes_image = io.BytesIO(decoded_image)
    img = Image.open(bytes_image).convert('RGB')
    open_cv_image = np.array(img) 
    # Convert RGB to BGR 
    #pixels = open_cv_image[:, :, ::-1].copy() 
    #img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2RGB)
    pixels = cv2.resize(img, (300,400))
    SCALE_FACTOR = 1.3
    MIN_NEIGHBORS = 5
    

    # load the photograph
    #cmap = {'0': (255,255,255),'1': (0,0,0)}
    #contents = contents[27:]
    #data = [cmap[letter] for letter in contents]
    #img = Image.new('RGB', (8, len(contents)//8), "white")
    #img.putdata(data)
    #  img.show()        

    #pixels = cv2.imread(img)
    
    # load the pre-trained model
    #classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    

    #detector = MTCNN()
    #result = detector.detect_faces(pixels)
    #bounding_box = result[0]['box']
    #keypoints = result[0]['keypoints']
    #cv2.rectangle(pixels, (bounding_box[0], bounding_box[1]),(bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),(0,155,255),2)
    #cv2.circle(pixels,(keypoints['left_eye']), 2, (0,155,255), 2)
    #cv2.circle(pixels,(keypoints['right_eye']), 2, (0,155,255), 2)
    #cv2.circle(pixels,(keypoints['nose']), 2, (0,155,255), 2)
    #cv2.circle(pixels,(keypoints['mouth_left']), 2, (0,155,255), 2)
    #cv2.circle(pixels,(keypoints['mouth_right']), 2, (0,155,255), 2)
    #ret, jpeg = cv2.imencode('.jpg', pixels)
    #pixels = jpeg.tobytes()

    #cv2.imwrite("pixels.jpg", image)
    #cv2.namedWindow("image")
    #pixels = cv2.imshow("image",image)
    #cv2.waitKey(0)

    #pixels = cv2.cvtColor(pixels,cv2.COLOR_BGR2GRAY)
    # perform face detection
    bboxes = face_cascade.detectMultiScale(pixels, SCALE_FACTOR, MIN_NEIGHBORS)
    
    # print bounding box for each detected face

    #for (x,y,w,h) in bboxes:
    #     pixels = cv2.rectangle(pixels,(x,y),(x+w,y+h),(0,255,0),2)
    
    for box in bboxes:

        # extract
        x, y, width, height = box

        x2, y2 = x + width, y + height
    
        # draw a rectangle over the pixels
        cv2.rectangle(pixels, (x, y), (x2, y2), (0,0,255), 1)   

        # show the image
        #pixels = cv2.imshow('Face', pixels)

        # encode OpenCV raw frame to jpg and displaying it
        ret, jpeg = cv2.imencode('.jpg', pixels)
        pixels = base64.b64encode(jpeg)
        pixels = pixels.decode('utf-8')
    

        # keep the window open until we press a key
             
      #  waitKey(0)

        # close the window
    #destroyAllWindows()

    image_conv = html.Div([
        html.H5(filename),
        html.H6(datetime.datetime.fromtimestamp(date)),

        # HTML images accept base64 encoded strings in the same format
        # that is supplied by the upload
        html.Img(src=contents, height='300', width='400'),
        html.Hr(),
        html.Div('Landmarks Detected!')
        
        
    ])
    return image_conv


@app.callback([Output('output-image-upload', 'children'), Output('panel', 'children'), Output('presidential_in','options'), Output('guber_in', 'options'), Output('senate_in', 'options')],
              [Input('upload-image', 'contents')],
              [State('upload-image', 'filename'),
               State('upload-image', 'last_modified')])
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        #pic = face_extractor(list_of_contents)
        #if(pic is None):
        query = qbc.doquery(query="{'6'.EX.'Naomi'}", database='bqzcn7xzi')
        query_keys = list(query.keys())
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
        children = [parse_contents(c, n, d) for c, n, d in zip(list_of_contents, list_of_names, list_of_dates)]
        
        return children, panel, candidates_president, candidates_guber, candidates_senate
       

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

@app.callback(dash.dependencies.Output('tabs-example-content', 'children'),[dash.dependencies.Input('submit-val', 'n_clicks')],[dash.dependencies.State('presidential_in', 'value')])
def update_output_presidential(n_clicks, value):
    if (value is not None):
        return 'You have voted! You voted "{}" for president! Thank you!'.format(value)
    else:    
        return 

"""
@app.callback([Output('panel', 'children'), Output('presidential_in', 'options'),Output('guber_in', 'options'), Output('senate_in', 'options')], [Input('tabs', 'value')])
def render_content(tab):
    if tab == 'close_camera':
        query = qbc.doquery(query="{'6'.EX.'Lillian'}", database='bqzcn7xzi')
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

        return getCam()

@app.callback(dash.dependencies.Output('tabs-example-content', 'children'),[dash.dependencies.Input('submit-val', 'n_clicks')],[dash.dependencies.State('presidential_in', 'value')])
def update_output_presidential(n_clicks, value):
    if (value is none):
        return
    else:    
        return 'You have voted! You voted "{}" for president! Thank you!'.format(value)


@app.callback([dash.dependencies.Output('presdential_in', 'options'), dash.dependencies.Output('guber_in', 'options'),dash.dependencies.Output('senate_in', 'options')],[dash.dependencies.Input('upload-image', 'value')])
def update_output_dropdowns(value):
    candidates_president = [{'label': i, 'value' : i} for i in getPresidency()]
    candidates_senate = [{'label': i, 'value' : i} for i in getSenate(query['state_of_origin'])]
    candidates_guber = [{'label': i, 'value' : i} for i in getGuber(query['state_of_origin'])]
        
    return candidates_president, candidates_guber, candidates_senate




@app.callback(dash.dependencies.Output('tabs-example-content', 'children'),[dash.dependencies.Input('submit-val', 'n_clicks')],[dash.dependencies.State('presidential_in', 'value')])
def update_output_state(n_clicks, value):
    return
"""     
"""        
@app.callback(output=Output('tabs-example-content','children'), inputs=[Input('submit-val','value')])
def render_tab_content_example(click):
    if click is not None:
      return "You Have Voted!"
"""

"""dcc.Tabs(id='tabs', value='tab-1', children=[
                dcc.Tab(label='Close Camera', value='close_camera'), #turn camera off.
                dcc.Tab(label='Open camera', value='open_camera', children=[upload])]), #turn camera on.])
"""
            
