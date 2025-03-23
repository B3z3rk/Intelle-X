from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import InputRequired, Email, Length,DataRequired


class LoginForm(FlaskForm):
    email = StringField('Email',validators=[InputRequired(message="Email is required"),Email(message="Enter a valid email address")])
    password = PasswordField('Password',validators=[InputRequired(message="Password is required"),Length(min=6, message="Password must be at least 6 characters long")])
    submit = SubmitField('Log In')


class SignupForm(FlaskForm):
    firstname = StringField('First Name', validators=[DataRequired(), Length(min=2, max=50)])
    lastname = StringField('Last Name', validators=[DataRequired(), Length(min=2, max=50)])
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=6)])
    submit = SubmitField('Sign Up')

