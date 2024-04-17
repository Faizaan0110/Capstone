from flask import Flask, render_template, request, redirect, flash, url_for, jsonify, session
from flask_sqlalchemy import SQLAlchemy
import csv

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.secret_key = 'your_secret_key'

db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)
    fullname = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(50), nullable=False)

class Student(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    current_student_id = db.Column(db.String(20), unique=True, nullable=False)
    attention_span = db.Column(db.Float, nullable=False)
    total_time = db.Column(db.Float, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))

def is_logged_in():
    return 'user_id' in session

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/signin')
def signin():
    return render_template('signin.html')

@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']
    user = User.query.filter_by(username=username).first()
    if user and user.password == password:
        session['user_id'] = user.id  # Store user ID in session
        return redirect('/dashboard')
    else:
        flash('Invalid credentials!')
        return redirect('/signin')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash('You have been logged out.')
    return redirect(url_for('home'))

@app.route('/dashboard')
def dashboard():
    if not is_logged_in():
        flash('Please log in to view the dashboard.')
        return redirect(url_for('signin'))

    # Load student data from CSV file
    with open('attention_details.csv', 'r') as file:
        csv_data = list(csv.DictReader(file))
    
    return render_template('dashboard.html', students=csv_data)

@app.route('/forgot-password')
def forgot_password():
    return render_template('forgot_password.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        fullname = request.form['fullname']
        email = request.form['email']
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if password != confirm_password:
            flash('Passwords do not match!')
            return redirect('/signup')

        user = User(username=username, password=password, fullname=fullname, email=email)
        db.session.add(user)
        try:
            db.session.commit()
            flash('Account created successfully!')
            return redirect('/signin')
        except IntegrityError:
            db.session.rollback()
            flash('Username already exists!')
            return redirect('/signup')

    return render_template('signup.html')

@app.route('/reset-password', methods=['POST'])
def reset_password():
    username = request.form['username']
    user = User.query.filter_by(username=username).first()
    if user:
        flash('Please contact the administrator to reset your password.')
    else:
        flash('Username does not exist.')
    return redirect(url_for('forgot_password'))

@app.route('/api/student-attention')
def student_attention():
    students = Student.query.all()
    data = [{'student_id': student.current_student_id, 'attention_span': student.attention_span} for student in students]
    return jsonify(data)

@app.route('/dashboard-data')
def dashboard_data():
    user_id = session.get('user_id')
    if not user_id:
        return redirect(url_for('signin'))

    # You can query the actual Student data from the database here if needed
    # For now, let's just return a dummy response
    data = {
        'PRN ' : 1032200614,
        'attention_span': 75.0,
        'total_time': 120.5
    }
    return jsonify(data)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
