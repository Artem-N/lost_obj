from flask_wtf import FlaskForm
from wtforms import SubmitField

class CheckCameraForm(FlaskForm):
    submit = SubmitField('Check the Camera')
