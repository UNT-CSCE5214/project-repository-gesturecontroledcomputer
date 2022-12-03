# create a flask app to serve index.html
from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
@app.route('/index.html')
def index():
    return render_template('index.html')