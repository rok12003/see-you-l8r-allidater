from flask import Flask, render_template, request
from helpers.sentence_matcher import rank_matches
from helpers.toxicity import detox

app = Flask(__name__)

@app.route('/', methods = ['GET', 'POST'])
def home():

    if request.method == "POST":
        gender = request.form.get('gender')
        if gender == "female":
            gender_match = "f"
        elif gender == "male":
            gender_match = "m"
        else:
            gender_match = None

        age = request.form.get('age')
        if age != "":
            age_low = int(age) - 5
            age_high = int(age) + 5
        else:
            age_low, age_high = None, None
        biography = request.form['biography']

        toxic_score = detox(biography)
        print('toxic_score')
        if toxic_score['severe_toxicity'] > .1 or toxic_score['threat'] > .1:
            bio_list = ['Try to be less toxic.']
        else:
        
            if biography != "":
                bio_list = rank_matches(biography, pref_gender = gender_match, pref_age_lower=age_low, pref_age_higher=age_high)
            else:
                bio_list = ["Write something"]
    else:
        gender = ''
        age = ''
        biography = ''
        bio_list = []

    return render_template('form.html', results=bio_list, gender=gender, age=age, biography=biography)
    