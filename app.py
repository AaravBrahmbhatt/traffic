from flask import Flask, render_template, request, redirect, Response,  url_for
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, LoginManager, login_user, login_required, logout_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import InputRequired, Length, ValidationError, Email, DataRequired, EqualTo
from flask_bcrypt import Bcrypt
import mss

import numpy as np
import pandas as pd
from ultralytics import YOLO
from PIL import Image
import cv2
import base64
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Login Stuff
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test.db'
app.config['SECRET_KEY'] = 'iamgoatedattwo'
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)


login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(20), nullable=False, unique=True)
    password = db.Column(db.String(80), nullable=False)


class RegisterForm(FlaskForm):
    email = StringField(validators=[
        InputRequired(), Email(), Length(max=120)], render_kw={"placeholder": "Email"})

    password = PasswordField(validators=[
        InputRequired(), Length(min=8, max=20)], render_kw={"placeholder": "Password"})
    
    confirm_password = PasswordField("Confirm Password", render_kw={"placeholder": "Confirm Password"}, validators=[DataRequired(), EqualTo('password', message="Passwords must match")])

    submit = SubmitField('Register')

    def validate_email(self, email):
        existing_user_email = User.query.filter_by(email=email.data).first()
        if existing_user_email:
            raise ValidationError("That Email Is Already Registered.")


class LoginForm(FlaskForm):
    email = StringField(validators=[
        InputRequired(), Email(), Length(max=120)], render_kw={"placeholder": "Email"})

    password = PasswordField(validators=[
        InputRequired(), Length(min=8, max=20)], render_kw={"placeholder": "Password"})

    submit = SubmitField('Login')


# Load your models and data
df = pd.read_csv("realistic_traffic_data_100.csv")

X = df[["car", "truck", "bus", "motor", "total", "weighted"]]
y_wait = df["waiting_time"]
y_accident = df["accident_probability"]
y_co2 = df["co2_emissions"]

X_train, X_test, y_wait_train, y_wait_test = train_test_split(X, y_wait, test_size=0.2, random_state=42)
_, _, y_accident_train, y_accident_test = train_test_split(X, y_accident, test_size=0.2, random_state=42)
_, _, y_co2_train, y_co2_test = train_test_split(X, y_co2, test_size=0.2, random_state=42)

model_wait = RandomForestRegressor(n_estimators=100, random_state=42)
model_wait.fit(X_train, y_wait_train)

model_accident = RandomForestRegressor(n_estimators=100, random_state=42)
model_accident.fit(X_train, y_accident_train)

model_co2 = RandomForestRegressor(n_estimators=100, random_state=42)
model_co2.fit(X_train, y_co2_train)

model_yolo = YOLO("yolo12n.pt")

vehicle_weights = {"car": 1.0, "truck": 1.5, "bus": 2.0, "motor": 0.5}


def draw_boxes_on_image(img_cv2, results, conf_threshold=0.10):
    boxes = results[0].boxes
    names = results[0].names
    for box in boxes:
        conf = float(box.conf[0])
        if conf < conf_threshold:
            continue
        cls_id = int(box.cls[0])
        cls_name = names[cls_id]
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        label = f"{cls_name} {conf:.2f}"
        cv2.rectangle(img_cv2, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
        cv2.putText(img_cv2, label, (xyxy[0], xyxy[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return img_cv2


@app.route("/")
@login_required
def home():
    return render_template("index.html")


@app.route("/traffic", methods=["GET", "POST"])
@login_required
def traffic():
    if request.method == "POST":
        file = request.files.get("image")
        if file:
            img_pil = Image.open(file.stream).convert("RGB")
            img_np = np.array(img_pil)
            img_cv2 = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR).copy()

            results = model_yolo(img_np, conf=0.16)
            boxes = results[0].boxes
            names = results[0].names

            count_by_type = {v: 0 for v in vehicle_weights}
            for box in boxes:
                cls_name = names[int(box.cls[0])].lower()
                for v in vehicle_weights:
                    if v in cls_name:
                        count_by_type[v] += 1

            total = sum(count_by_type.values())
            weighted = sum(count_by_type[v] * vehicle_weights[v] for v in vehicle_weights)

            if total == 0:
                wait_pred = 0.0
                accident_pred = 0.0
                co2_pred = 0.0
            else:
                input_data = pd.DataFrame({
                    "car": [count_by_type["car"]],
                    "truck": [count_by_type["truck"]],
                    "bus": [count_by_type["bus"]],
                    "motor": [count_by_type["motor"]],
                    "total": [total],
                    "weighted": [weighted]
                })

                wait_pred = model_wait.predict(input_data)[0]
                accident_pred = model_accident.predict(input_data)[0]
                co2_pred = model_co2.predict(input_data)[0]

            annotated_img = draw_boxes_on_image(img_cv2.copy(), results)

            _, buffer = cv2.imencode('.jpg', annotated_img)
            encoded_image = base64.b64encode(buffer).decode()

            return render_template(
                "traffic.html",
                vehicle_counts=count_by_type,
                waiting_time=f"{wait_pred:.2f}",
                accident_prob=f"{accident_pred:.2f}",
                co2_emissions=f"{co2_pred:.2f}",
                result_image=encoded_image
            )
    return render_template("traffic.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and bcrypt.check_password_hash(user.password, form.password.data):
            login_user(user)
            return redirect("/")
        else:
            return "Invalid email or password"
    return render_template("login.html", form=form)


@app.route("/signup", methods=["GET", "POST"])
def signup():
    form = RegisterForm()
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode("utf-8")
        new_user = User(email=form.email.data, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        return redirect("/login")
    return render_template("signup.html", form=form)


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect("/login")

##GOOGLE ROUTING

@app.route("/google_login")
def google_login():
    if not google.authorized:
        return redirect(url_for("google.login"))

    resp = google.get("/oauth2/v2/userinfo")
    if not resp.ok:
        return f"Error: {resp.text}"

    user_info = resp.json()
    email = user_info["email"]

    # Check if user exists
    user = User.query.filter_by(email=email).first()

    # If user doesn't exist, create a new one
    if not user:
        # You can set a default password or leave it blank; Google login won't need it
        user = User(
            email=email,
            password=bcrypt.generate_password_hash("defaultpassword").decode("utf-8")
        )
        db.session.add(user)
        db.session.commit()

    # Log in the user
    login_user(user)
    return redirect("/")


#DRONE CAPTURE


def generate_frames():
    sct = mss.mss()
    monitor = {"top": 0, "left": 960, "width": 960, "height": 1080}


    while True:
        img = sct.grab(monitor)
        frame = np.array(img)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
       

        results = model_yolo(frame[:, :, ::-1])
        annotated_frame = draw_boxes_on_image(frame.copy(), results, conf_threshold=0.1)

        # Encode for web
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route("/drone_feed")
@login_required
def drone_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/drone")
@login_required
def drone():
    return render_template("drone.html")





if __name__ == "__main__":
    app.run(debug=True)

