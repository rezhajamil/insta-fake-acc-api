from distutils.log import debug
from flask import Flask, jsonify, request
import pickle
import pandas as pd

app = Flask(__name__)

with open("insta_fake_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

columns=['profile pic', 'nums/length username', 'fullname words',
       'nums/length fullname', 'name==username', 'description length',
       'external URL', 'private', '#posts', '#followers', '#follows']
target=['Fake Account','Real Account']

@app.route("/")
def halo():
    return "<h1>Instagram Fake Account API</h1>"

@app.route("/detect", methods=['POST'])
def model_predict():
    content = request.json
    try:
        data=[  content['profile_pic'],
                content['nl_username'],
                content['fullname'],
                content['nl_fullname'],
                content['name_username'],
                content['desc'],
                content['external_url'],
                content['is_private'],
                content['n_posts'],
                content['n_followers'],
                content['n_follows']]
        data=pd.DataFrame([data],columns=columns)
        res=model.predict(data)
        response={'code':200,
                    'status':'OK',
                    'result':res[0]}
        return jsonify(response)
    except Exception as e:
        response={'code':500,
                    'status':'error',
                    'message':str(e)}
        return jsonify(response)

app.run(debug=True)