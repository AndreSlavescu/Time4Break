from flask import Flask, render_template
import secrets

app = Flask(__name__)
secret = secrets.token_urlsafe(32)
app.secret_key = secret

@app.route('/')
def index():
    return render_template("home.html")

@app.route('/appPage')
def appPage():
    return render_template("app.html", X = "90", suggestion = "Testing")

@app.route('/login')
def login():
    return render_template("login.html")


if __name__ == '__main__':
    app.run(port="3000")
