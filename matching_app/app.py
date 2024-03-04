from flask import Flask, render_template, request, redirect, url_for
from helpers.sentence_matcher import rank_matches
from helpers.finetuned_generator import generate_text
from helpers.toxicity import detox

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/choice')
def choice():
    return render_template('choice.html')

@app.route('/existingmatches')
def existing_matches():
    return render_template('existingmatches.html')

@app.route('/realusers', methods=['GET', 'POST'])
def real_users():
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

        if biography != "":
            bio_list = rank_matches(biography, pref_gender=gender_match, pref_age_lower=age_low, pref_age_higher=age_high)
        else:
            bio_list = ["Write something"]
    else:
        gender = ''
        age = ''
        biography = ''
        bio_list = []

    return render_template('realusers.html', results=bio_list, gender=gender, age=age, biography=biography)


@app.route('/generatedusers', methods=['GET', 'POST'])
def generated_users():
    if request.method == "POST":
        biography = request.form['biography']
        if biography != "":
            bio_list = generate_text(biography)
        else:
            bio_list = ["Write something"]
    else:
        gender = ''
        age = ''
        biography = ''
        bio_list = []

    return render_template('generatedusers.html', results=bio_list)

if __name__ == "__main__":
    app.run(debug=True)
