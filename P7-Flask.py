from flask import Flask
#from flask_restful import Resource, Api, reqparse
from flask import url_for
    
app = Flask(__name__)

@app.route('/')
def hello():
    return 'Hello, World!'

app.run(debug=True)