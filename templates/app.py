from flask import Flask, request, render_template, session

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Set a secret key for session

@app.route('/')
def index():
    if 'history' not in session:
        session['history'] = []
    return render_template('index.html', history=session['history'])

def process_input(user_input):
    # Implement your chatbot's response logic here
    return f"You said: {user_input}"

@app.route('/submit', methods=['POST'])
def submit():
    user_input = request.form['user_input']
    response = process_input(user_input)

    # Update session history
    session['history'].append({"user": user_input, "bot": response})
    session.modified = True

    return render_template('index.html', response=response, history=session['history'])

if __name__ == '__main__':
    app.run(debug=True)
