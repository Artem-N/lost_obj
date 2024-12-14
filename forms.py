# forms.py
from flask_wtf import FlaskForm
from wtforms import IntegerField, BooleanField, SubmitField
from wtforms.validators import DataRequired, NumberRange

class SettingsForm(FlaskForm):
    persistence_threshold = IntegerField('Persistence Threshold', validators=[DataRequired(), NumberRange(min=1, max=1000)])
    min_size = IntegerField('Minimum Object Size', validators=[DataRequired(), NumberRange(min=1, max=1000)])
    video_width = IntegerField('Video Width', validators=[NumberRange(min=100, max=1920)])
    video_height = IntegerField('Video Height', validators=[NumberRange(min=100, max=1080)])
    use_default_camera_params = BooleanField('Use Default Camera Parameters')
    submit = SubmitField('Save Settings')
