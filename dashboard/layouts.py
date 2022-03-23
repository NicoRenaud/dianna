import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from jupyter_dash import JupyterDash
from dash import html, dcc
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import base64

#static images
image_filename = '../tutorials/img/logo.png' # replace with your own image
encoded_image = base64.b64encode(open(image_filename, 'rb').read())

#rise = base64.b64encode(open('rise.png', 'rb').read())
#kernels = base64.b64encode(open('kernels.png', 'rb').read())

# colors
colors = {
    'white': '#FFFFFF',
    'text': '#091D58',
    'blue1' : '#063446',
    'blue2' : '#0e749b',
    'blue3' : '#15b3f0',
    'blue4' : '#d0f0fc',
    'yellow1' : '#f0d515'
}

# styles
navbarcurrentpage = {
    'text-decoration' : 'underline',
    'text-decoration-color' : colors['yellow1'],
    'color' : colors['white'],
    'text-shadow': '0px 0px 1px rgb(251, 251, 252)',
    'textAlign' : 'center'
    }

navbarotherpage = {
    'text-decoration' : 'underline',
    'text-decoration-color' : colors['blue2'],
    'color' : colors['white'],
    'textAlign' : 'center'
    }

# app layout
# In Bootstrap, the "row" class is used mainly to hold columns in it.
# Bootstrap divides each row into a grid of 12 virtual columns.

# header
def get_header():

    header = html.Div([

        html.Div([],
            className='four columns',
            style = {'padding-top' : '1%'}
        ),

        html.Div([
            html.H1(children='DIANNA\'s Dashboard',
                    style = {'textAlign' : 'center', 'color' : colors['white']}
            )],
            className='four columns',
            style = {'padding-top' : '1%'}
        ),

        html.Div([
            html.Img(
                    src = 'data:image/png;base64,{}'.format(encoded_image.decode()),
                    height = '43 px',
                    width = 'auto')
            ],
            className = 'four columns',
            style = {
                    'textAlign': 'right',
                    'padding-top' : '1.3%',
                    'padding-right' : '4%',
                    'height' : 'auto'
                    })

        ],
        className = 'row',
        style = {'background-color' : colors['blue1']}
        )

    return header

# Nav bar
def get_navbar(p = 'images'):

    navbar_images = html.Div([

        html.Div(['b'],
            className = 'five columns',
            style = {'color' : colors['blue2']}
        ),

        html.Div([
            dcc.Link(
                html.H4(children = 'Images',
                        style = navbarcurrentpage),
                href='/apps/images'
                )
        ],
        className='one column'),

        html.Div([
            dcc.Link(
                html.H4(children = 'Text',
                    style = navbarotherpage),
                href='/apps/text'
                )
        ],
        className='one column'),

        html.Div([], className = 'five columns')

    ],
    
    className = 'row',
    style = {'background-color' : colors['blue2']
            }
    )

    navbar_text = html.Div([

        html.Div(['b'],
            className = 'five columns',
            style = {'color' : colors['blue2']}
        ),

        html.Div([
            dcc.Link(
                html.H4(children = 'Images',
                    style = navbarotherpage),
                href='/apps/images'
                )
        ],
        className='one column'),

        html.Div([
            dcc.Link(
                html.H4(children = 'Text',
                    style = navbarcurrentpage),
                href='/apps/text'
                )
        ],
        className='one column'),

        html.Div([], className = 'five columns')

    ],
    
    className = 'row',
    style = {'background-color' : colors['blue2']
            }
    )

    if p == 'images':
        return navbar_images
    else:
        return navbar_text

def get_uploads():

    uploads = html.Div([

        html.Div(['b'],
            className = 'three columns',
            style = {'color' : colors['blue4']}),

        html.Div([
            dcc.Upload(
                id='upload-image',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select Image')
                ]),
                style={
                    'width': '80%',
                    'height': '40px',
                    'lineHeight': '40px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '3px',
                    'textAlign': 'center',
                    'align-items': 'center',
                    'margin': '10px',
                    'color' : colors['blue1']
                },
                # Allow multiple files to be uploaded
                multiple=True
            ),
            html.Div(id='output-image-upload',
            style = {
                'background-color' : colors['blue4'],
                'textAlign': 'center',
                'height': '230px'}),
            ], className = 'three columns', style = {'align-items': 'center'}),

        html.Div([
            dcc.Upload(
                id='upload-model',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select Model')
                ]),
                style={
                    'width': '80%',
                    'height': '40px',
                    'lineHeight': '40px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '3px',
                    'textAlign': 'center',
                    'align-items': 'center',
                    'margin': '10px',
                    'color' : colors['blue1']
                },
                # Allow multiple files to be uploaded
                multiple=True
            ),
            html.Div(id='output-model-upload',
            style = {
                'background-color' : colors['blue4'],
                'textAlign': 'center',
                'height': '230px'})
            ], className = 'three columns'),
    ], 
    className = 'row', style = {'background-color' : colors['blue4']}
    )
    
    return uploads