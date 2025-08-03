import os
from flask import Flask, redirect, url_for, render_template, request, flash, session
import re
import sqlite3
import hashlib
from base64 import decode
import requests
import json
from nutritionix import Nutritionix
from decimal import Decimal
from datetime import date, datetime,timedelta 
import time
import pandas as pd
import numpy as np
import pickle
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.preprocessing import LabelEncoder 
import random
from flask import Flask, render_template, request
import os
import base64
from ultralytics import YOLO
import cv2
import numpy as np
from ai_agent import regenerate_plan

AI_AGENT_AVAILABLE = True  # Set to True if your AI agent is available and configured

app = Flask(__name__)
app.secret_key='honsproject'
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.jinja_env.globals.update(zip=zip)

bf_meal = []
onefood = []
lunch_meal = []
dinner_meal = []
snack_meal = []

# app.config['UPLOAD_FOLDER'] = 'uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

class_mapping = {
    0: {'label': 'aloo-gobi', 'calories': 108},
    1: {'label': 'aloo-fry', 'calories': 125},
    2: {'label': 'dum-aloo', 'calories': 164},
    3: {'label': 'fish-curry', 'calories': 241},
    4: {'label': 'ghevar', 'calories': 61},
    5: {'label': 'green-chutney', 'calories': 21},
    6: {'label': 'gulab-jamun', 'calories': 145},
    7: {'label': 'idli', 'calories': 40},
    8: {'label': 'jalebi', 'calories': 150},
    9: {'label': 'chicken-seekh-kebab', 'calories': 158},
    10: {'label': 'kheer', 'calories': 266},
    11: {'label': 'kulfi', 'calories': 136},
    12: {'label': 'bhature', 'calories': 230}, 
    13: {'label': 'lassi', 'calories': 183},
    14: {'label': 'mutton-curry', 'calories': 298},
    15: {'label': 'onion-pakoda', 'calories': 80},
    16: {'label': 'palak-paneer', 'calories': 338},
    17: {'label': 'poha', 'calories': 270},
    18: {'label': 'rajma-curry', 'calories': 235},
    19: {'label': 'rasmalai', 'calories': 188},
    20: {'label': 'samosa', 'calories': 308},
    21: {'label': 'shahi-paneer', 'calories': 261},
    22: {'label': 'white-rice', 'calories': 135},
    23: {'label': 'bhindi-masala', 'calories': 225},
    24: {'label': 'chicken-biryani', 'calories': 348},
    25: {'label': 'chai', 'calories': 54},
    26: {'label': 'chole', 'calories': 311},
    27: {'label': 'coconut-chutney', 'calories': 105},
    28: {'label': 'dal-tadka', 'calories': 260},
    29: {'label': 'dosa', 'calories': 106}
}

def calculate_total_calories(class_label, count):
    class_info = class_mapping.get(class_label, {'label': 'unknown', 'calories': 0})
    calories_per_item = class_info['calories']
    total_calories = count * calories_per_item
    return total_calories

def detect_and_visualize(img, model_path, class_mapping, confidence_threshold=0.25):
    model = YOLO(model_path)

    results = model.predict(source=img, conf=confidence_threshold)
    detected_items = [0]*30
    float_detections = results[0].boxes.xyxy.tolist()
    detections = [[int(value) for value in detection] for detection in float_detections]
    confidences = results[0].boxes.conf.tolist()
    float_classes = results[0].boxes.cls.tolist()
    classes = [int(value) for value in float_classes]

    total_calories = 0
    resized_img = cv2.resize(img, (800, 400))

    scaling_factor_x = 800 / img.shape[1]
    scaling_factor_y = 400 / img.shape[0]

    for i in range(len(detections)):
        box = detections[i]
        resized_box = [
            int(box[0] * scaling_factor_x),
            int(box[1] * scaling_factor_y),
            int(box[2] * scaling_factor_x),
            int(box[3] * scaling_factor_y)
        ]
        class_index = classes[i]
        class_info = class_mapping.get(class_index, {'label': 'unknown', 'calories': 0})
        conf = confidences[i]
        if conf > 0.4:
            detected_items[class_index] += 1

            class_label = class_info['label']
            calories = class_info['calories']
            total_calories += calories

            cv2.putText(resized_img, f'{class_label} ({calories} kcal) {conf:.3f}', (resized_box[0], resized_box[1]), cv2.FONT_HERSHEY_PLAIN, 1,(255, 0, 0), 2)
            cv2.rectangle(resized_img, (resized_box[0], resized_box[1]), (resized_box[2], resized_box[3]), (255, 0, 255), 2)
    
    # cv2.putText(resized_img, f'Total Calories: {total_calories:.2f} cal', (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
    # Convert the OpenCV image to bytes
    _, result_image = cv2.imencode('.jpg', resized_img)
    result_bytes = result_image.tobytes()

    items_with_calories = []
    for i in range(30):
        if(detected_items[i] != 0):
            item_cal = class_mapping[i].get('calories') * detected_items[i]
            items_with_calories.append({'label': class_mapping[i].get('label'), 'calories': f"{detected_items[i]} * {class_mapping[i].get('calories')}.00 = {item_cal}", 'count': detected_items[i]})
    return result_bytes, total_calories, items_with_calories




@app.route('/')
@app.route('/first')
def first():
    if 'uid' in session:
        return redirect(url_for('index'))
    return render_template('home.html')


@app.route('/predict')
def index1():
    return render_template('index1.html')

@app.route('/about1')
def about1():
    return render_template('about.html')

@app.route('/service')
def service():
    return render_template('service.html')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/prediction1', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return render_template('index1.html', error="No file part")

        file = request.files['file']

        if file.filename == '':
            return render_template('index1.html', error="No file selected")

        if file and allowed_file(file.filename):
            img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
            
            if img is None:
                return render_template('index1.html', error="Invalid image file")
            
            result_bytes, total_calories, items_with_calories = detect_and_visualize(img, "best.pt", class_mapping)
            
            return render_template('index1.html', filename=f'data:image/jpg;base64,{base64.b64encode(result_bytes).decode()}', total_calories=total_calories, items_with_calories=items_with_calories, name=file.filename)
        else:
            return render_template('index1.html', error="Invalid file type. Please upload a JPG, JPEG, or PNG file.")
    except Exception as e:
        print(f"Error in upload_file: {str(e)}")
        return render_template('index1.html', error=f"An error occurred: {str(e)}")


@app.route("/delete_food", methods =['GET','POST'])
def delete_food():
	if request.method == 'GET':
		return redirect(url_for('add_successful'))

	else:
		mealtime = request.form['mealtime']
		item = request.form['fname']

		uid = session['uid']
		track_date = str(datetime.today().strftime ('%Y-%m-%d'))
		
		bf_list = []
		lunch_list = []
		dinner_list = []
		snack_list = []
		# for i in session['bf_numbers']:

			
		# 	print(float(i))
		# 	session.modified = True
		# 	print(type(i))
		if mealtime == "Breakfast":
			
			for i in bf_meal:
				if item == i[0]:
					bf_meal.remove(i)

			for i in session['bf_meal']:
				if item == i[0]:
					session['bf_meal'].remove(i)
					session.modified = True

			
					

				
					for j in session['bf_numbers']:
						j = round(Decimal(j), 2)
						bf_list.append(j)
					
				
					bf_list[0]-=round(Decimal(i[3]), 2)
					bf_list[1]-=round(Decimal(i[4]), 2)
					bf_list[2]-=round(Decimal(i[5]), 2)
					bf_list[3]-=round(Decimal(i[6]), 2)

					try:
						with get_connection() as conn:
							cur = conn.cursor()
						
							cur.execute("select * from tracking where track_date=? and u_id=?",(track_date,uid,))
							track_info= cur.fetchall()
				
							calorie = round(Decimal(track_info[0][6]), 2)-round(Decimal(i[3]), 2)
							protein = round(Decimal(track_info[0][7]), 2)-round(Decimal(i[4]), 2)
							carb = round(Decimal(track_info[0][8]), 2)-round(Decimal(i[5]), 2)
							fat = round(Decimal(track_info[0][9]), 2)-round(Decimal(i[6]), 2)
						
							cur2 = conn.cursor()
							cur2.execute("select * from tracking where track_date=? and u_id=?",(track_date,uid,))
							u_data = cur2.fetchone()

							item_in_db = u_data[2].split(",")[:-1]
							item_to_db = ""

							for i in item_in_db:
								if item == i:
									item_in_db.remove(i)

							item_to_db = ",".join(item_in_db)+","

							cur2.execute("update tracking set track_breakfast=?,track_calorie=?,track_protein=?,track_carb=?,track_fat=? where track_date=? and u_id=?", (item_to_db,float(calorie),float(protein),float(carb),float(fat),track_date,uid))
							conn.commit()

					except sqlite3.Error as e:
						return (f'{e}')
					finally:
						conn.close()


					session['bf_numbers'].clear()

					[x for x in session['bf_numbers'] if x]

					session.modified = True
					for i in bf_list:
						session['bf_numbers'].append(i)	
						session.modified = True

					

					
					
					

		
		if mealtime == "Lunch":
			foodlist = session['lunch_meal']

			for i in lunch_meal:
				if item == i[0]:
					lunch_meal.remove(i)

			for i in foodlist:
				if item == i[0]:
					foodlist.remove(i)
				
			
					for j in session['lunch_numbers']:
						j = round(Decimal(j), 2)
						lunch_list.append(j)
					
				
					lunch_list[0]-=round(Decimal(i[3]), 2)
					lunch_list[1]-=round(Decimal(i[4]), 2)
					lunch_list[2]-=round(Decimal(i[5]), 2)
					lunch_list[3]-=round(Decimal(i[6]), 2)

					try:
						with get_connection() as conn:
							cur = conn.cursor()
						
							cur.execute("select * from tracking where track_date=? and u_id=?",(track_date,uid,))
							track_info= cur.fetchall()
				
							calorie = round(Decimal(track_info[0][6]), 2)-round(Decimal(i[3]), 2)
							protein = round(Decimal(track_info[0][7]), 2)-round(Decimal(i[4]), 2)
							carb = round(Decimal(track_info[0][8]), 2)-round(Decimal(i[5]), 2)
							fat = round(Decimal(track_info[0][9]), 2)-round(Decimal(i[6]), 2)
						
							cur2 = conn.cursor()
							cur2.execute("select * from tracking where track_date=? and u_id=?",(track_date,uid,))
							u_data = cur2.fetchone()

							item_in_db = u_data[3].split(",")[:-1]
							item_to_db = ""

							for i in item_in_db:
								if item == i:
									item_in_db.remove(i)

							item_to_db = ",".join(item_in_db)+","
							
							cur2.execute("update tracking set track_lunch=?,track_calorie=?,track_protein=?,track_carb=?,track_fat=? where track_date=? and u_id=?", (item_to_db,float(calorie),float(protein),float(carb),float(fat),track_date,uid))
							conn.commit()

					except sqlite3.Error as e:
						return (f'{e}')
					finally:
						conn.close()

					session['lunch_numbers'].clear()

					[x for x in session['lunch_numbers'] if x]

					session.modified = True
					for i in lunch_list:
						session['lunch_numbers'].append(i)	
						session.modified = True


		if mealtime == "Snack":
			foodlist = session['snack_meal']

			for i in snack_meal:
				if item == i[0]:
					snack_meal.remove(i)

			for i in foodlist:
				if item == i[0]:
					foodlist.remove(i)
				
			
					for j in session['snack_numbers']:
						j = round(Decimal(j), 2)
						snack_list.append(j)
					
				
					snack_list[0]-=round(Decimal(i[3]), 2)
					snack_list[1]-=round(Decimal(i[4]), 2)
					snack_list[2]-=round(Decimal(i[5]), 2)
					snack_list[3]-=round(Decimal(i[6]), 2)

					try:
						with get_connection() as conn:
							cur = conn.cursor()
						
							cur.execute("select * from tracking where track_date=? and u_id=?",(track_date,uid,))
							track_info= cur.fetchall()
				
							calorie = round(Decimal(track_info[0][6]), 2)-round(Decimal(i[3]), 2)
							protein = round(Decimal(track_info[0][7]), 2)-round(Decimal(i[4]), 2)
							carb = round(Decimal(track_info[0][8]), 2)-round(Decimal(i[5]), 2)
							fat = round(Decimal(track_info[0][9]), 2)-round(Decimal(i[6]), 2)
						
							cur2 = conn.cursor()
							cur2.execute("select * from tracking where track_date=? and u_id=?",(track_date,uid,))
							u_data = cur2.fetchone()

							item_in_db = u_data[4].split(",")[:-1]
							item_to_db = ""

							for i in item_in_db:
								if item == i:
									item_in_db.remove(i)

							item_to_db = ",".join(item_in_db)+","
				
							cur2.execute("update tracking set track_snack=?,track_calorie=?,track_protein=?,track_carb=?,track_fat=? where track_date=? and u_id=?", (item_to_db,float(calorie),float(protein),float(carb),float(fat),track_date,uid))
							conn.commit()

					except sqlite3.Error as e:
						return (f'{e}')
					finally:
						conn.close()

					session['snack_numbers'].clear()

					[x for x in session['snack_numbers'] if x]

					session.modified = True
					for i in snack_list:
						session['snack_numbers'].append(i)	
						session.modified = True

					

		if mealtime == "Dinner":
			foodlist = session['dinner_meal']
			for i in dinner_meal:
				if item == i[0]:
					dinner_meal.remove(i)

			for i in foodlist:
				if item == i[0]:
					foodlist.remove(i)
				
			
					for j in session['dinner_numbers']:
						j = round(Decimal(j), 2)
						dinner_list.append(j)
					
				
					dinner_list[0]-=round(Decimal(i[3]), 2)
					dinner_list[1]-=round(Decimal(i[4]), 2)
					dinner_list[2]-=round(Decimal(i[5]), 2)
					dinner_list[3]-=round(Decimal(i[6]), 2)

					try:
						with get_connection() as conn:
							cur = conn.cursor()
						
							cur.execute("select * from tracking where track_date=? and u_id=?",(track_date,uid,))
							track_info= cur.fetchall()
				
							calorie = round(Decimal(track_info[0][6]), 2)-round(Decimal(i[3]), 2)
							protein = round(Decimal(track_info[0][7]), 2)-round(Decimal(i[4]), 2)
							carb = round(Decimal(track_info[0][8]), 2)-round(Decimal(i[5]), 2)
							fat = round(Decimal(track_info[0][9]), 2)-round(Decimal(i[6]), 2)
						
							cur2 = conn.cursor()
							cur2.execute("select * from tracking where track_date=? and u_id=?",(track_date,uid,))
							u_data = cur2.fetchone()

							item_in_db = u_data[5].split(",")[:-1]
							item_to_db = ""

							for i in item_in_db:
								if item == i:
									item_in_db.remove(i)

							item_to_db = ",".join(item_in_db)+","
							cur2.execute("update tracking set track_dinner=?,track_calorie=?,track_protein=?,track_carb=?,track_fat=? where track_date=? and u_id=?", (item_to_db,float(calorie),float(protein),float(carb),float(fat),track_date,uid))
							conn.commit()

					except sqlite3.Error as e:
						return (f'{e}')
					finally:
						conn.close()

					session['dinner_numbers'].clear()

					[x for x in session['dinner_numbers'] if x]

					session.modified = True
					for i in dinner_list:
						session['dinner_numbers'].append(i)	
						session.modified = True

					
				
								
		return render_template('add_food.html',mealtime=mealtime)
	


def get_connection():
	import os
	# Get the directory where this script is located
	script_dir = os.path.dirname(os.path.abspath(__file__))
	# Database is in the parent directory (root of project)
	db_path = os.path.join(os.path.dirname(script_dir), 'diet_recommendation.db')
	conn = sqlite3.connect(db_path)
	conn.row_factory=sqlite3.Row # to be able to reference by column name
	return conn

def pwd_security(passwd):
	"""A strong password must be at least 8 characters long
	   and must contain a lower case letter, an upper case letter,
	   and at least 3 digits.
	   Returns True if passwd meets these criteria, otherwise returns False.
	   """
	# check password length
	# check password for uppercase, lowercase and numeric chars
	hasupper = False	
	haslower = False
	digitcount = 0
	digit= False
	strong = False
	length = True
	special = False
	for c in passwd:
		if (c.isupper()==True):
			hasupper= True
		elif (c.islower()==True):
			haslower=True
		elif (c.isdigit()==True):
			digitcount+=1
			digit = True
		elif re.findall('[^A-Za-z0-9]',c):
			special = True
	if hasupper == True and haslower == True and digit == True and special == True:
		strong = True
	if len(passwd) <8:
		length = False
	return strong,haslower,hasupper,digit,length, special

def pwd_encode(pwd):
	secure_pwd =hashlib.md5(pwd.encode()).hexdigest()
	return secure_pwd



@app.route("/update_profile", methods=['GET', 'POST'])
def edit_weight():
    if request.method == 'GET':
        try:
            with get_connection() as conn:
                cur = conn.cursor()
                cur.execute("select * from user where u_id=?", (session['uid'],))
                u_data = cur.fetchone()
                email = u_data[5]
                password = u_data[2]
                gender = u_data[3]
                weight = u_data[6]
                vegan = u_data[12]
                ft = u_data[7]
                inch = u_data[8]
                allergy = u_data[13]
                activity = u_data[20]
                goal = u_data[22]
                username = u_data[1]
                age = u_data[4]
        except sqlite3.Error as e:
            return (f'{e}')
        finally:
            conn.close()

        return render_template(
            'edit_profile.html',
            name=username,
            email=email,
            password=password,
            gender=gender,
            weight=weight,
            vegan=vegan,
            ft=ft,
            inch=inch,
            allergy=allergy,
            activity=activity,
            goal=goal,
            age=age
        )
    else:
        # Get form data from request
        username = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
        age = request.form.get('age')
        gender = request.form.get('gender')
        weight = request.form.get('weight')
        vegan = request.form.get('vegan')
        ft = request.form.get('feet')
        inch = request.form.get('inches')
        allergy = request.form.get('allergy')
        activity = request.form.get('activity')
        goal = request.form.get('goal')
        # Hash password if changed (optional: check if changed)
        # password = pwd_encode(password)
        try:
            with get_connection() as conn:
                cur = conn.cursor()
                cur.execute("""
                    UPDATE user SET 
                        u_username=?,
                        u_email=?,
                        u_password=?,
                        u_age=?,
                        u_gender=?,
                        u_vegan=?,
                        u_allergy=?,
                        u_weight=?,
                        u_feet=?,
                        u_inches=?,
                        u_activitylevel=?,
                        u_goal=?
                    WHERE u_id=?
                """, (
                    username,
                    email,
                    password,
                    age,
                    gender,
                    vegan,
                    allergy,
                    weight,
                    ft,
                    inch,
                    activity,
                    goal,
                    session['uid']
                ))
                conn.commit()
        except sqlite3.Error as e:
            return (f'{e}')
        finally:
            conn.close()
        return redirect(url_for('profile'))

@app.route("/add_food", methods =['GET','POST'] )
def add_food():
	if request.method == 'GET':
		
		return render_template('add_food.html')

	else:
		mealtime = request.form['mealtime']
		
		
		return render_template('add_food.html',mealtime=mealtime)

def food_list(meal,item):
	
	meal.append(item)

	return meal

def one_food(item,portion,p_type):
	
	meal = [item,portion,p_type]

	return meal

@app.route("/add_successful", methods =['GET','POST'] )

def add_successful():
	
	if request.method == 'GET':
		
		return render_template('add_food.html')

	else:
		mealtime = request.form['mealtime']
		item_name = request.form['food']
		foodPortion = int(request.form['portion'])
		portion_type = request.form['portion_type']

		app_id = "887d9e3c"
		app_key = "81829ae0aaefe752f5f42a7247b8329c"
		url = "https://trackapi.nutritionix.com/v2/natural/nutrients"

# Prepare headers and payload for the v2 API endpoint
		headers = {
		   "x-app-id": app_id,
		    "x-app-key": app_key,
		    "Content-Type": "application/json"
		}

		payload = {
		    "query": f"{foodPortion} {portion_type} {item_name}",
		    "timezone": "US/Eastern"  # You can adjust this based on your timezone
		}    
    
# The v2 API accepts 'query' to search for food items in a natural language format
		response = requests.post(url, headers=headers, json=payload)

# Check if the request was successful
		if response.status_code == 200:
		    data = response.json()
    
		    # Extract nutritional information from the response
		    item = data['foods'][0]  # Access the first (and possibly only) food item

		    name = item.get('food_name')
		    calories = item.get('nf_calories')
		    protein = item.get('nf_protein')
		    carb = item.get('nf_total_carbohydrate')
		    fat = item.get('nf_total_fat')
		    quantity = foodPortion
		    unit = item.get('serving_unit')

    # Calculate final nutritional values based on portion
		    finalCalorie = (calories * foodPortion) / quantity
		    finalProtein = (protein * foodPortion) / quantity
		    finalCarb = (carb * foodPortion) / quantity
		    finalFat = (fat * foodPortion) / quantity

    # Store the information in a list or any preferred format
		    food = [item_name, quantity, unit, finalCalorie, finalProtein, finalCarb, finalFat]
		    uid = session['uid']
		    track_date = str(datetime.today().strftime('%Y-%m-%d'))
		else:
    # Handle errors, if any
		    print("Error:", response.status_code, response.text)
		try:
			with get_connection() as conn:
				cur = conn.cursor()
						
				cur.execute("select * from tracking where track_date=? and u_id=?",(track_date,uid,))
				track_info= cur.fetchall()
				

				calorie = round(Decimal(track_info[0][6]), 2)+round(Decimal(food[3]), 2)
				protein = round(Decimal(track_info[0][7]), 2)+round(Decimal(food[4]), 2)
				carb = round(Decimal(track_info[0][8]), 2)+round(Decimal(food[5]), 2)
				fat = round(Decimal(track_info[0][9]), 2)+round(Decimal(food[6]), 2)
						
				cur2 = conn.cursor()
				
				if mealtime == "Breakfast":
					meal_input = track_info[0][2] + food[0] +","
					
					cur2.execute("update tracking set track_breakfast=?,track_calorie=?,track_protein=?,track_carb=?,track_fat=? where track_date=? and u_id=?", (meal_input,float(calorie),float(protein),float(carb),float(fat),track_date,uid))
					conn.commit()


				if mealtime == "Lunch":
					meal_input = track_info[0][3] + food[0] +","
					
					
					cur2.execute("update tracking set track_lunch=?,track_calorie=?,track_protein=?,track_carb=?,track_fat=? where track_date=? and u_id=?", (meal_input,float(calorie),float(protein),float(carb),float(fat),track_date,uid))
					conn.commit()

				if mealtime == "Snack":
					meal_input = track_info[0][4] + food[0] +","
					
					cur2.execute("update tracking set track_snack=?,track_calorie=?,track_protein=?,track_carb=?,track_fat=? where track_date=? and u_id=?", (meal_input,float(calorie),float(protein),float(carb),float(fat),track_date,uid))
					conn.commit()

				if mealtime == "Dinner":
					meal_input = track_info[0][5] + food[0] +","
					
					cur2.execute("update tracking set track_dinner=?,track_calorie=?,track_protein=?,track_carb=?,track_fat=? where track_date=? and u_id=?", (meal_input,float(calorie),float(protein),float(carb),float(fat),track_date,uid))
					conn.commit()
				

		except sqlite3.Error as e:
			return (f'{e}')
		finally:
			conn.close()

		
		if mealtime == "Breakfast":
			
			session['bf_numbers'] = [0,0,0,0]
			session['bf_meal'] = food_list(bf_meal,food)
			
			for i in session['bf_meal']:
						
				session['bf_numbers'][0]+=round(Decimal(i[3]), 2)
				session['bf_numbers'][1]+=round(Decimal(i[4]), 2)
				session['bf_numbers'][2]+=round(Decimal(i[5]), 2)
				session['bf_numbers'][3]+=round(Decimal(i[6]), 2)
			print(session['bf_meal'])
					

		if mealtime == "Lunch":
			session['lunch_numbers'] = [0,0,0,0]
			session['lunch_meal'] = food_list(lunch_meal,food)

			for i in session['lunch_meal']:			
				session['lunch_numbers'][0]+=round(Decimal(i[3]), 2)
				session['lunch_numbers'][1]+=round(Decimal(i[4]), 2)
				session['lunch_numbers'][2]+=round(Decimal(i[5]), 2)
				session['lunch_numbers'][3]+=round(Decimal(i[6]), 2)
				
		
		if mealtime == "Snack":
			session['snack_numbers'] = [0,0,0,0]
			session['snack_meal'] = food_list(snack_meal,food)

			for i in session['snack_meal']:
						
				session['snack_numbers'][0]+=round(Decimal(i[3]), 2)
				session['snack_numbers'][1]+=round(Decimal(i[4]), 2)
				session['snack_numbers'][2]+=round(Decimal(i[5]), 2)
				session['snack_numbers'][3]+=round(Decimal(i[6]), 2)

		if mealtime == "Dinner":
			session['dinner_numbers'] = [0,0,0,0]
			session['dinner_meal'] = food_list(dinner_meal,food)

			for i in session['dinner_meal']:
						
				session['dinner_numbers'][0]+=round(Decimal(i[3]), 2)
				session['dinner_numbers'][1]+=round(Decimal(i[4]), 2)
				session['dinner_numbers'][2]+=round(Decimal(i[5]), 2)
				session['dinner_numbers'][3]+=round(Decimal(i[6]), 2)
		

		return render_template('add_food.html',mealtime=mealtime)

@app.route("/track", methods = ['GET','POST'])
def track():
    if request.method == 'GET':
        uid = session['uid']
        track_date = str(datetime.today().strftime ('%Y-%m-%d'))
        try:
            with get_connection() as conn:
                cur = conn.cursor()
                # Ensure a tracking row exists for today
                cur.execute("SELECT * FROM tracking WHERE u_id=? AND track_date=?", (uid, track_date))
                tracking_row = cur.fetchone()
                if not tracking_row:
                    cur.execute("INSERT INTO tracking (track_date, u_id, track_calorie, track_protein, track_fat, track_carb, track_breakfast, track_lunch, track_snack, track_dinner) VALUES (?, ?, 0, 0, 0, 0, '', '', '', '')", (track_date, uid))
                    conn.commit()
                cur.execute("select * from tracking where track_date=? and u_id=?",(track_date,uid))
                track_info= cur.fetchall()

                if track_info:
                    pass
                else:
                    try:
                        with get_connection() as conn:
                            cur = conn.cursor()
                            breakfast=""
                            lunch = ""
                            dinner = ""
                            snack = ""
                            cur.execute("insert into tracking (track_date,u_id,track_calorie,track_protein,track_fat,track_carb,track_breakfast,track_lunch,track_snack,track_dinner) values (?,?,0,0,0,0,?,?,?,?)",(track_date,uid,breakfast,lunch,snack,dinner))
                            conn.commit()

                            cur.execute("select * from tracking where track_date=? and u_id=?",(track_date,uid))
                            track_info= cur.fetchall()
                            if track_info[0][2] == ",":
                                cur.execute("update tracking set track_breakfast=?",("",))
                                conn.commit()
                    except sqlite3.Error as e:
                        return (f'{e}')
                    finally:
                        conn.close()

                cur.execute("select * from tracking where track_date=? and u_id=?",(track_date,uid))
                track_info= cur.fetchall()

                protein_goal = session['u_info'][14]
                carb_goal = session['u_info'][15]
                fat_goal = session['u_info'][16]
                breakfast = track_info[0][2]
                lunch = track_info[0][3]
                snack = track_info[0][4]
                dinner = track_info[0][5]

                protein_consumed = track_info[0][7]
                carb_consumed = track_info[0][8]
                fat_consumed = track_info[0][9]

                calorie_goal = session['u_info'][17]
                calorie_consumed = track_info[0][6]

                protein_percent = "{:.2f}".format((protein_consumed/protein_goal) * 100)
                carb_percent = "{:.2f}".format((carb_consumed/carb_goal) * 100)
                fat_percent = "{:.2f}".format((fat_consumed/fat_goal) * 100)
                calorie_percent = "{:.2f}".format((calorie_consumed/calorie_goal) * 100)

                try:
                    with get_connection() as conn:
                        cur = conn.cursor()
                        cur.execute("update tracking set track_calorie=?,track_protein=?,track_carb=?,track_fat=? where track_date=? and u_id=?", (calorie_consumed,protein_consumed,carb_consumed,fat_consumed,track_date,uid,))
                        conn.commit()

                except sqlite3.Error as e:
                    return (f'{e}')
                finally:
                    conn.close()

                return render_template('track.html',p_goal=protein_goal,
                                            c_goal = carb_goal,
                                            f_goal=fat_goal,
                                            p_consumed = protein_consumed,
                                            c_consumed=carb_consumed,
                                            f_consumed = fat_consumed,
                                            p_percent = protein_percent,
                                            c_percent = carb_percent,
                                            f_percent = fat_percent,
                                            cal_percent = calorie_percent,
                                            cal_goal = calorie_goal,
                                            cal_consumed = calorie_consumed,
                                            breakfast = breakfast,
                                            lunch = lunch,
                                            snack = snack,
                                            dinner = dinner,
                                            )

        except sqlite3.Error as e:
            return (f'{e}')
        finally:
            conn.close()
    else:
        return render_template('track.html')

@app.route("/register", methods = ['GET','POST'])
def citizen_register():
	if request.method == 'GET':
		return render_template('register.html')
	else:
		name = request.form['name']
		
		email = request.form['email']
		password = request.form['password']
		
		return render_template('profilesetup.html',name=name,email=email,password=password)

@app.route("/login", methods = ['GET','POST'])
def login():
	if request.method == 'GET':
		return render_template('login.html')
	else:
		print("DEBUG: Login POST request received")
		session['uid'] = 0
		email = request.form['email']
		password = request.form['password']
		print(f"DEBUG: Email: {email}")
		print(f"DEBUG: Password length: {len(password)}")
		secure_pwd = pwd_encode(password)
		print(f"DEBUG: Encoded password: {secure_pwd}")
		msg=''
		try:
			with get_connection() as conn:
				cur = conn.cursor()
				cur.execute("select * from user where u_email=?",(email,))
				u_info= cur.fetchall()
				print(f"DEBUG: Found {len(u_info)} users with email {email}")
				if not u_info:
					flash(f'The email address ({email}) that you entered does not exist in our database.')
					return redirect(url_for('login'))
				else:
					for row in u_info:
						session['uid'] = row[0]
						u_pass = row[2] 
						u_name = row[1]
						u_date = row[-1]
						print(f"DEBUG: User ID: {session['uid']}, Stored password: {u_pass}")
					
					if secure_pwd == u_pass:
						print("DEBUG: Password match successful")
						days = []
						flash(f'Your have successfully logged in as {u_name}')
						session['u_logged'] = True
						session['u_info'] = []
						session['u_pass'] = password 

						track_date = datetime.today().strftime ('%Y-%m-%d')
						if session.get('u_info') and len(session['u_info']) > 0:
							date_str = session['u_info'][-1]
						else:
							date_str = None
						if date_str is not None:
							sdate = datetime.strptime(date_str, '%Y-%m-%d').date()
						else:
							sdate = datetime.today().date()
						edate = datetime.strptime(track_date, '%Y-%m-%d').date()
						delta = edate - sdate     

						for i in range(delta.days + 1):
							day = sdate + timedelta(days=i)
							days.append(str(day))
							journey = len(days)

						try:
							with get_connection() as conn:
								cur = conn.cursor()
								cur2 = conn.cursor()
								cur2.execute("update user set u_journey=? where u_id=?", (journey,session['uid'],))
								conn.commit()

								cur.execute("select * from user where u_id=?",(session['uid'],))
								u_info = cur.fetchone()
				
								for row in u_info:
									session['u_info'].append(row)

						except sqlite3.Error as e:
							print(f"DEBUG: Database error in journey update: {e}")
							return (f'{e}')
						finally:
							conn.close()

						print("DEBUG: Redirecting to index")
						return redirect(url_for('index'))
					else:
						print("DEBUG: Password mismatch")
						session.pop('uid',None)
						flash('Sorry the credentails you are using are invalid')
						return redirect(url_for('login'))

		except sqlite3.Error as e:
			print(f"DEBUG: Database error: {e}")
			return (f'{e}')
		finally:
			conn.close()

@app.route("/setup", methods = ['GET','POST'])
def profilesetup():
	if request.method == 'GET':
		return render_template('profilesetup.html')

	else:
		name = request.form['name']
		email = request.form['email']
		passwd = request.form['password']
		password = pwd_encode(passwd)
		age = int(request.form['age'])
		gender = request.form['gender']
		vegan = request.form['vegan']
		allergy = request.form['allergy']
		weight_lb = int(request.form['weight'])
		feet = int(request.form['feet'])
		inches = int(request.form['inches'])
		activity_level = request.form['activity']
		goal = request.form['goal']
		height_bmi = int((feet * 12) + inches)
		bmr = 0
		body_status = ""
		BMI =  weight_lb / (height_bmi*height_bmi) * 703
		bodyfat = 0

		if gender == "male":
			bmr = int((4.536 * weight_lb) + (15.88 * height_bmi) - (5 * age) + 5)
			bodyfat = int((1.20 * BMI) + (0.23 * age) - 16.2)
		else:
			bmr = int((4.536 * weight_lb) + (15.88 * height_bmi) - (5 * age) - 161)
			bodyfat = int((1.20 * BMI) + (0.23 * age) - 5.4)

		calorie = 0

		if activity_level == "sedentary":
			calorie = int(bmr*1.2)

		elif activity_level == "lightly active":
			calorie = int(bmr * 1.375)

		elif activity_level == "moderately active":
			calorie = int(bmr * 1.55)

		elif activity_level == "very active":
			calorie = int(bmr * 1.725)

		elif activity_level == "extra active":
			calorie = int(bmr * 1.9)

		if BMI < 18.5:
			body_status = "underweight"

		elif BMI >= 18.5 and BMI <= 24.9 :
			body_status = "healthy weight"

		elif BMI >= 25 and BMI <= 29.9 :
			body_status = "overweight"

		elif BMI >= 30 :
			body_status = "obese"

		protein = int(((calorie-500) * 0.30)/4)
		carb = int(((calorie-500)* 0.40)/4)
		fat = int(((calorie-500) * 0.30)/9)
		fiber = int(calorie/1000*14)
		journey = 1
		breakfast = int((calorie-500) * 0.30)
		snack = int((calorie-500)* 0.10)
		lunch = int((calorie-500)* 0.35)
		dinner = int((calorie-500)* 0.25)

		try:
			with get_connection() as conn:
				db = conn.cursor()
				db.execute("insert into user (u_username,u_email,u_password,u_age,u_gender,u_vegan,u_allergy,u_weight,u_feet,u_inches,u_bmi,u_activitylevel,u_protein,u_carb,u_fat,u_fiber,u_calories,u_journey,u_bodyfat,u_status,u_startdate,u_goal) values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",(name,email,password,age,gender,vegan,allergy,weight_lb,feet,inches,int(BMI),activity_level,protein,carb,fat,fiber,calorie,journey,bodyfat,body_status,datetime.today().strftime ('%Y-%m-%d'),goal))
				conn.commit()
				flash('Successfully Registered')

		except sqlite3.Error as e:
			return (f'{e}')
		finally:
			conn.close()

	
		return redirect(url_for('login'))

@app.route("/profile", methods=['GET', 'POST'])
def profile():
    if request.method == 'GET':
        uid = session['uid']
        try:
            with get_connection() as conn:
                db = conn.cursor()
                db.execute("select * from user where u_id=?", (uid,))
                u_info = db.fetchone()
        except sqlite3.Error as e:
            return (f'{e}')
        finally:
            conn.close()

        if not u_info:
            flash("User not found.", "warning")
            return redirect(url_for('login'))

        user = {
            'id': u_info[0],
            'username': u_info[1],
            'email': u_info[5],
            'password': u_info[2],  # Don't display in template
            'age': u_info[4],
            'gender': u_info[3],
            'vegan': u_info[12],
            'allergy': u_info[13],
            'weight': u_info[6],
            'feet': u_info[7],
            'inches': u_info[8],
            'bmi': u_info[18],
            'activitylevel': u_info[20],
            'protein_goal': u_info[14],
            'carb_goal': u_info[15],
            'fat_goal': u_info[16],
            'fiber_goal': u_info[19],
            'calorie_goal': u_info[17],
            'bodyfat': u_info[9],
            'status': u_info[10],
            'journey': u_info[11],
            'startdate': u_info[21],
            'goal': u_info[22]
        }
        return render_template('profile.html', user=user)
    else:
        return render_template('profile.html')


@app.route("/recommendation", methods = ['GET','POST'])
def recommendation():
    if request.method == 'GET':
        return render_template('recommendation.html')

    else:
        dataset = pd.read_csv('dietdataset1.csv')

        dataset = pd.DataFrame(data=dataset.iloc[:,0:10].values,columns = ['meal_name','carb','meat','vege','fruit', 'type','vegan','allergy','time'])
        le = LabelEncoder()
        dataset_encoded = dataset.iloc[:,0:10]
        for i in dataset_encoded:
            dataset_encoded[i] = le.fit_transform(dataset_encoded[i])
            
            model = pickle.load(open('model1','rb'))

        bf_vege_input = []
        bf_meat_input = []
        bf_carb_input = []
        bf_fruit_input = []

        bf_vege = random.choice(request.form.getlist('vege'))
        bf_meat = random.choice(request.form.getlist('meat'))
        bf_carb = random.choice(request.form.getlist('carb'))
        bf_fruit = random.choice(request.form.getlist('fruit'))

        bf_vege_input.append(bf_vege)
        bf_meat_input.append(bf_meat)
        bf_carb_input.append(bf_carb)
        bf_fruit_input.append(bf_fruit)

        lunch_vege_input = []
        lunch_meat_input = []
        lunch_carb_input = []
        lunch_fruit_input = []

        lunch_vege = random.choice(request.form.getlist('vege'))
        lunch_meat = random.choice(request.form.getlist('meat'))
        lunch_carb = random.choice(request.form.getlist('carb'))
        lunch_fruit = random.choice(request.form.getlist('fruit'))

        lunch_vege_input.append(lunch_vege)
        lunch_meat_input.append(lunch_meat)
        lunch_carb_input.append(lunch_carb)
        lunch_fruit_input.append(lunch_fruit)

        snack_vege_input = []
        snack_meat_input = []
        snack_carb_input = []
        snack_fruit_input = []

        snack_vege = random.choice(request.form.getlist('vege'))
        snack_meat = random.choice(request.form.getlist('meat'))
        snack_carb = random.choice(request.form.getlist('carb'))
        snack_fruit = random.choice(request.form.getlist('fruit'))

        snack_vege_input.append(snack_vege)
        snack_meat_input.append(snack_meat)
        snack_carb_input.append(snack_carb)
        snack_fruit_input.append(snack_fruit)

        dinner_vege_input = []
        dinner_meat_input = []
        dinner_carb_input = []
        dinner_fruit_input = []

        dinner_vege = random.choice(request.form.getlist('vege'))
        dinner_meat = random.choice(request.form.getlist('meat'))
        dinner_carb = random.choice(request.form.getlist('carb'))
        dinner_fruit = random.choice(request.form.getlist('fruit'))

        dinner_vege_input.append(dinner_vege)
        dinner_meat_input.append(dinner_meat)
        dinner_carb_input.append(dinner_carb)
        dinner_fruit_input.append(dinner_fruit)
        
        type_breakfast = request.form.getlist('breakfast_dishes')
        type_lunch =  request.form.getlist('lunch_dishes')
        type_dinner = request.form.getlist('dinner_dishes')
        type_snack = request.form.getlist('snack_dishes')
        print(type_breakfast)
        allergy_input = []
        vegan_input = []
        allergy_input.append(session['u_info'][13])
        vegan_input.append(session['u_info'][12])

        print(type_breakfast,allergy_input,vegan_input)
        time_breakfast = ['Breakfast']
        time_snack = ['Snack']
        time_lunch = ['Lunch']
        time_dinner = ['Dinner']

        def input_encode(entry, room):
            meal = dataset.values.tolist()
            meal_encode = dataset_encoded.values.tolist()
            lists = []
            encode = []
    
            for i in entry:
                found = False
                for j in meal:
                    if i == j[room]:
                        lists.append(j)
                        found = True
                        break
            if not found:
                print(f"Warning: '{i}' not found in dataset column {room}")
    
            for j in lists:
                encode.append(meal_encode[meal.index(j)][room])
        
            if not encode:  # No match found
                encode = [0]  # Set a default value or handle accordingly

            return encode
#         return meal_encode[meal.index(j)][room]

        def input_decode(entry,room): 
            meal = dataset.values.tolist()
            meal_encode = dataset_encoded.values.tolist()
            lists = []
            decode = []   
            for i in entry:
                for j in meal_encode:
                    if i==j[room]:
                        lists.append(j)
                        break
                 
            for j in lists: 
                decode.append(meal[meal_encode.index(j)][room])
        
            return decode

        bf_carb_encode = input_encode(bf_carb_input,1)[0]
        bf_meat_encode = input_encode(bf_meat_input,2)[0]
        bf_vege_encode = input_encode(bf_vege_input,3)[0]
        bf_fruit_encode = input_encode(bf_fruit_input,4)[0]

        lunch_carb_encode = input_encode(lunch_carb_input,1)[0]
        lunch_meat_encode = input_encode(lunch_meat_input,2)[0]
        lunch_vege_encode = input_encode(lunch_vege_input,3)[0]
        lunch_fruit_encode = input_encode(lunch_fruit_input,4)[0]

        dinner_carb_encode = input_encode(dinner_carb_input,1)[0]
        dinner_meat_encode = input_encode(dinner_meat_input,2)[0]
        dinner_vege_encode = input_encode(dinner_vege_input,3)[0]
        dinner_fruit_encode = input_encode(dinner_fruit_input,4)[0]

        snack_carb_encode = input_encode(snack_carb_input,1)[0]
        snack_meat_encode = input_encode(snack_meat_input,2)[0]
        snack_vege_encode = input_encode(snack_vege_input,3)[0]
        snack_fruit_encode = input_encode(snack_fruit_input,4)[0]
                
        breakfast_encode = input_encode(type_breakfast,5)[0]
        lunch_encode = input_encode(type_lunch,5)[0]
        snack_encode = input_encode(type_snack,5)[0]
        dinner_encode = input_encode(type_dinner,5)[0]

        vegan_encode = input_encode(vegan_input,6)[0]
        allergy_encode = input_encode(allergy_input,7)[0]
        bf_time_encode = input_encode(time_breakfast,8)[0]
        lunch_time_encode = input_encode(time_lunch,8)[0]
        snack_time_encode = input_encode(time_snack,8)[0]
        dinner_time_encode = input_encode(time_dinner,8)[0]

        bf_input = [bf_carb_encode,bf_meat_encode,bf_vege_encode,bf_fruit_encode,breakfast_encode,vegan_encode,allergy_encode,bf_time_encode]
        
        bf_result = model.predict([bf_input])
        bf_prediction = input_decode(bf_result,0)	
                
        lunch_input = [lunch_carb_encode,lunch_meat_encode,lunch_vege_encode,lunch_fruit_encode,lunch_encode,vegan_encode,allergy_encode,lunch_time_encode]
        lunch_result = model.predict([lunch_input])
        lunch_prediction = input_decode(lunch_result,0)	
                
        snack_input = [snack_encode,snack_encode,snack_encode,snack_encode,snack_encode,vegan_encode,allergy_encode,snack_time_encode]
        snack_result = model.predict([snack_input])
        snack_prediction = input_decode(snack_result,0)	
                           
        dinner_input = [dinner_encode,dinner_encode,dinner_encode,dinner_encode,dinner_encode,vegan_encode,allergy_encode,dinner_time_encode]
        dinner_result = model.predict([dinner_input])
        dinner_prediction = input_decode(dinner_result,0)	
        
        try:
            with get_connection() as conn:
                cur = conn.cursor()
                cur.execute("select * from user where u_id=?",(session['uid'],))
                data = cur.fetchone()
                calorie = data[17]
                protein = data[14]
                carb = data[15]
                fat = data[16]

        except sqlite3.Error as e:
            return (f'{e}')
        finally:
            conn.close()

        bf_cal = int(int(calorie)* 0.30)
        snack_cal = int(int(calorie)* 0.10)
        lunch_cal = int(int(calorie)* 0.35)
        dinner_cal = int(int(calorie)* 0.25)

        bf_protein = int(int(protein)* 0.30)
        snack_protein = int(int(protein)* 0.10)
        lunch_protein = int(int(protein)* 0.35)
        dinner_protein = int(int(protein)* 0.25)
        
        bf_carb = int(int(carb)* 0.30)
        snack_carb = int(int(carb)* 0.10)
        lunch_carb = int(int(carb)* 0.35)
        dinner_carb = int(int(carb)* 0.25)

        bf_fat = int(int(fat)* 0.30)
        snack_fat = int(int(fat)* 0.10)
        lunch_fat = int(int(fat)* 0.35)
        dinner_fat = int(int(fat)* 0.25)

        user_goal = session['u_info'][22] if 'u_info' in session and len(session['u_info']) > 22 else ''
        return render_template('recommendation.html',
            bf_prediction = bf_prediction[0],
            lunch_prediction = lunch_prediction[0],
            snack_prediction = snack_prediction[0],
            dinner_prediction = dinner_prediction[0],
            bf_cal = bf_cal,
            snack_cal = snack_cal,
            lunch_cal = lunch_cal,
            dinner_cal = dinner_cal,
            bf_protein = bf_protein,
            snack_protein = snack_protein,
            lunch_protein = lunch_protein,
            dinner_protein = dinner_protein,
            bf_carb = bf_carb,
            snack_carb = snack_carb,
            lunch_carb = lunch_carb,
            dinner_carb = dinner_carb,
            bf_fat = bf_fat,
            snack_fat = snack_fat,
            lunch_fat = lunch_fat,
            dinner_fat = dinner_fat,
            user_goal = user_goal
        )




@app.route("/recommend_setup", methods=['GET', 'POST'])
def recommend_setup():
    if request.method == 'GET':
        print(session['u_info'][12])
        return render_template('recommendsetup.html')
    else:
        return render_template('recommendsetup.html')

@app.route("/progress", methods=['GET', 'POST'])
def progress():
    print("u_info:", session.get('u_info'))  # Debug print to check the session data
    if request.method == 'GET':
        uid = session['uid']
        track_date = datetime.today().strftime('%Y-%m-%d')
        days = []
        display_day = []
        weeks = []
        day_weight = []
        week_weight = []
        sdate = datetime.strptime(session['u_info'][21], '%Y-%m-%d').date()
        edate = datetime.strptime(track_date, '%Y-%m-%d').date()
        delta = edate - sdate
        for i in range(delta.days + 1):
            day = sdate + timedelta(days=i)
            days.append(str(day))
        split_list = [days[x:x+7] for x in range(0, len(days), 7)]
        weeknum = split_list.index(split_list[-1]) + 1
        pw_date = split_list[-1][-1]
        pw_weight = []
        try:
            with get_connection() as conn:
                cur = conn.cursor()
                cur.execute("select * from progress where u_id=? and p_date=?", (uid, track_date))
                data = cur.fetchone()
                if not data:
                    cur.execute("insert into progress (u_id,p_date,p_weight) values (?,?,?)", (session['u_info'][0], track_date, session['u_info'][6]))
                    conn.commit()
                cur.execute("select * from progress where u_id=? and p_date=?", (uid, track_date))
                data2 = cur.fetchone()
                for i in data2:
                    pw_weight.append(data[2])
                cur.execute("select * from progress_week where u_id=? and pw_num=?", (uid, weeknum))
                week_exist = cur.fetchone()
                if not week_exist:
                    cur.execute("insert into progress_week (u_id,pw_num,pw_weight) values (?,?,?)", (session['u_info'][0], weeknum, pw_weight[0]))
                    conn.commit()
            cur.execute("select * from progress_week where u_id=?", (uid,))
            week_data = cur.fetchall()
            for i in week_data:
                weeks.append("Week" + str(i[1]))
                week_weight.append(i[2])
            for i in days:
                cur.execute("select * from progress where u_id=? and p_date=?", (uid, i))
                get_weight = cur.fetchall()
                for i in get_weight:
                    day_weight.append(i[2])
        except sqlite3.Error as e:
            return (f'{e}')
        finally:
            conn.close()
        try:
            with get_connection() as conn:
                cur = conn.cursor()
                cur.execute("select * from progress where u_id=?", (uid,))
                dates = cur.fetchall()
                for i in dates:
                    getdate = datetime.strptime(i[1], '%Y-%m-%d').date()
                    dates = getdate.strftime("%B-%d")
                    display_day.append(dates)
        except sqlite3.Error as e:
            return (f'{e}')
        finally:
            conn.close()
        return render_template('progress.html', days=display_day[-7:], weeks=weeks[-7:], d_weight=day_weight[-7:], w_weight=week_weight[-7:])
    else:
        return render_template('progress.html')

@app.route("/daily_detail", methods = ['GET','POST'])
def daily_detail():
    if request.method == 'GET':
        try:
            with get_connection() as conn:
                cur = conn.cursor()
                cur.execute("select * from tracking where u_id=? and track_date=?", (session['uid'], datetime.today().strftime ('%Y-%m-%d')))
                i = cur.fetchone()

                if not i:
                    flash("No tracking data found for today. Please add your food or weight first.", "warning")
                    return redirect(url_for('index'))

                getdate = datetime.strptime(i[1], '%Y-%m-%d').date()
                date = getdate.strftime("%B-%d-%Y")
                breakfast = i[2]
                lunch = i[3]
                snack = i[4]
                dinner = i[5]
                calorie = session['u_info'][17]
                protein = i[7]
                carb = i[8]
                fat = i[9]
                consumed = i[6]
                deficit = round(Decimal(calorie - i[6]), 2)
                result = "Reduced "+ str(round(Decimal(consumed/3500),4))+"lb of bodyweight (in theory)"
                deficits = "Calorie Deficit: "+ str(deficit) +"kcal"

        except sqlite3.Error as e:
            return (f'{e}')
        finally:
            conn.close()
        return render_template('daily_detail.html',date = date,
                                                   breakfast = breakfast,
                                                   lunch = lunch,
                                                   snack = snack,
                                                   dinner = dinner,
                                                   calorie = calorie,
                                                   protein = protein,
                                                   carb = carb,
                                                   fat = fat,
                                                   consumed = consumed,
                                                   result = result,
                                                   deficit = deficits)

    else:
        getdate = request.form['date']
        weight = ""
        try:
            with get_connection() as conn:
                cur = conn.cursor()
                cur.execute("select * from tracking where u_id=? and track_date=?",(session['uid'],getdate,))
                i = cur.fetchone()
                
                if not i:
                    flash("No tracking data found for this date.", "warning")
                    return redirect(url_for('index'))

                print(i[1])
                

                
                getdate = datetime.strptime(i[1], '%Y-%m-%d').date()
                date = getdate.strftime("%B-%d-%Y")
                breakfast = i[2]
                lunch = i[3]
                snack = i[4]
                dinner = i[5]
                calorie = int(session['u_info'][17])
                protein = i[7]
                carb = i[8]
                fat = i[9]
                consumed = i[6]
                deficit = round(Decimal(calorie - i[6]), 2)
                result = "Reduced "+ str(round(Decimal(consumed/3500),4))+"lb of bodyweight (in theory)"
                deficits = "Calorie Deficit: "+ str(deficit) +"kcal"

                cur.execute("select * from progress where u_id=? and p_date=?",(session['uid'],getdate,))
                weights = cur.fetchone()
                if weights:
                    for i in weights:
                        weight = weights[2]
                else:
                    weight = "undefined"
        except sqlite3.Error as e:
            return (f'{e}')
        finally:
            conn.close()
        return render_template('daily_detail.html',date = date,
                                                   breakfast = breakfast,
                                                   lunch = lunch,
                                                   snack = snack,
                                                   dinner = dinner,
                                                   calorie = calorie,
                                                   protein = protein,
                                                   carb = carb,
                                                   fat = fat,
                                                   consumed = consumed,
                                                   result = result,
                                                   deficit = deficits,
                                                   weight = weight)

@app.route("/weekly_detail", methods = ['GET','POST'])
def weekly_detail():
	if request.method == 'GET':
		try:
			with get_connection() as conn:
				cur = conn.cursor()
				cur.execute("select * from progress_week where u_id=?",(session['uid'],))
				u_week = cur.fetchall()
				weeks = []
				days = []
				track_date = datetime.today().strftime ('%Y-%m-%d')
				for i in u_week:
					weeks.append(i[1])

				date_str = session['u_info'][-1]
				if date_str is not None:
					sdate = datetime.strptime(date_str, '%Y-%m-%d').date()
				else:
					sdate = datetime.today().date()
				edate = datetime.strptime(track_date, '%Y-%m-%d').date()
				delta = edate - sdate     

				for i in range(delta.days + 1):
					day = sdate + timedelta(days=i)
					days.append(str(day))

		
				split_list = [days[x:x+7] for x in range(0, len(days), 7)]
				this_weeknum = split_list.index(split_list[-1])+1
				print(split_list[this_weeknum-1])
				
				calories = []
				proteins = []
				fats = []
				carbs = []
				
				try:
					with get_connection() as conn:
						cur = conn.cursor()
						cur.execute("select * from progress_week where u_id=? and pw_num=?",(session['uid'],this_weeknum,))
						wow = cur.fetchone()
						weight_of_week = wow[2]
						for i in split_list[this_weeknum-1]:
							cur.execute("select * from tracking where u_id=? and track_date=?",(session['uid'],i,))
							u_week = cur.fetchall()
							for i in u_week:
							
								calories.append(i[6])
								proteins.append(i[7])
								carbs.append(i[8])
								fats.append(i[9])


						calorie_consumed = sum(calories)
						
						required = float(session['u_info'][17])*len(calories)
						
						calorie_required = required

						deficit = calorie_required - calorie_consumed

						average_calorie = round(Decimal(sum(calories)/len(calories)),2)
						average_protein = round(Decimal(sum(proteins)/len(proteins)),2)
						average_carb = round(Decimal(sum(carbs)/len(carbs)),2)
						average_fat = round(Decimal(sum(fats)/len(fats)),2)
						average_deficit = round(Decimal(deficit/len(calories)),2)
						net_deficit = round(Decimal(deficit),2)
						loss_weight = round(Decimal(net_deficit/3500),2)
						result = "Reduced "+ str(loss_weight) +"lb of bodyweight in this whole week (in theory)"

				except sqlite3.Error as e:
					return (f'{e}')
				finally:
					conn.close()
				
				

		except sqlite3.Error as e:
			return (f'{e}')
		finally:
			conn.close()
		return render_template('weekly_detail.html',weeks = weeks,
													average_calorie = average_calorie,
													average_carb = average_carb,
													average_fat = average_fat,
													average_protein = average_protein,
													net_deficit = net_deficit,
													result = result,
													average_deficit = average_deficit,
													week = this_weeknum,
													weight_week = weight_of_week
												    )

	else:
		getweek = request.form['weeks']
		try:
			with get_connection() as conn:
				cur = conn.cursor()
				cur.execute("select * from progress_week where u_id=? and pw_num=?",(session['uid'],getweek))
				u_week = cur.fetchall()
				cur.execute("select * from progress_week where u_id=?",(session['uid'],))
				u_weeks = cur.fetchall()
				weeks = []
				days = []
				
				
				for i in u_weeks:
					weeks.append(i[1])

				track_date = datetime.today().strftime ('%Y-%m-%d')
				date_str = session['u_info'][-1]
				if date_str is not None:
					sdate = datetime.strptime(date_str, '%Y-%m-%d').date()
				else:
					sdate = datetime.today().date()
				edate = datetime.strptime(track_date, '%Y-%m-%d').date()
				delta = edate - sdate     

				for i in range(delta.days + 1):
					day = sdate + timedelta(days=i)
					days.append(str(day))

		
				split_list = [days[x:x+7] for x in range(0, len(days), 7)]
				this_weeknum = int(getweek)
				
				
				calories = []
				proteins = []
				fats = []
				carbs = []
				
				try:
					with get_connection() as conn:
						cur = conn.cursor()
						cur.execute("select * from progress_week where u_id=? and pw_num=?",(session['uid'],this_weeknum,))
						wow = cur.fetchone()
						weight_of_week = wow[2]
						for i in split_list[this_weeknum-1]:
							cur.execute("select * from tracking where u_id=? and track_date=? and track_calorie!=?",(session['uid'],i,0))
							u_week = cur.fetchall()
							for i in u_week:
							
								calories.append(i[6])
								proteins.append(i[7])
								carbs.append(i[8])
								fats.append(i[9])


						calorie_consumed = sum(calories)
						
						required = float(session['u_info'][17])*len(calories)
						
						calorie_required = required

						deficit = calorie_required - calorie_consumed

						average_calorie = round(Decimal(sum(calories)/len(calories)),2)
						average_protein = round(Decimal(sum(proteins)/len(proteins)),2)
						average_carb = round(Decimal(sum(carbs)/len(carbs)),2)
						average_fat = round(Decimal(sum(fats)/len(fats)),2)
						average_deficit = round(Decimal(deficit/len(calories)),2)
						net_deficit = round(Decimal(deficit),2)
						loss_weight = round(Decimal(net_deficit/3500),2)
						result = "Reduced "+ str(loss_weight) +"lb of bodyweight in this whole week (in theory)"

				except sqlite3.Error as e:
					return (f'{e}')
				finally:
					conn.close()
				
				

		except sqlite3.Error as e:
			return (f'{e}')
		finally:
			conn.close()
		return render_template('weekly_detail.html',weeks = weeks,
													average_calorie = average_calorie,
													average_carb = average_carb,
													average_fat = average_fat,
													average_protein = average_protein,
													net_deficit = net_deficit,
													result = result,
													average_deficit = average_deficit,
													week = getweek,
													weight_week = weight_of_week,
												    )


@app.route("/index", methods=['GET', 'POST'])
def index():
    track_date = datetime.today().strftime('%Y-%m-%d')
    if request.method == 'GET':
        try:
            with get_connection() as conn:
                cur = conn.cursor()
                # Ensure a tracking row exists for today
                cur.execute("SELECT * FROM tracking WHERE u_id=? AND track_date=?", (session['uid'], track_date))
                tracking_row = cur.fetchone()
                if not tracking_row:
                    cur.execute("INSERT INTO tracking (track_date, u_id, track_calorie, track_protein, track_fat, track_carb, track_breakfast, track_lunch, track_snack, track_dinner) VALUES (?, ?, 0, 0, 0, 0, '', '', '', '')", (track_date, session['uid']))
                    conn.commit()
                # Always fetch the latest weight from user table
                cur.execute("SELECT * FROM user WHERE u_id=?", (session['uid'],))
                u_data = cur.fetchone()
                if not u_data:
                    flash("Session expired or user not found. Please log in again.", "warning")
                    return redirect(url_for('login'))
                weight = u_data[8]  # u_weight
        except sqlite3.Error as e:
            return (f'{e}')
        finally:
            conn.close()
        return render_template('index.html', weight=weight)
    else:
        getweight = request.form.get('weight')
        print(f"Received weight: {getweight}")
        try:
            with get_connection() as conn:
                cur = conn.cursor()
                # Ensure a row exists in progress for today
                cur.execute("SELECT * FROM progress WHERE u_id=? AND p_date=?", (session['uid'], track_date))
                progress_row = cur.fetchone()
                if not progress_row:
                    cur.execute("INSERT INTO progress (u_id, p_date, p_weight) VALUES (?, ?, ?)", (session['uid'], track_date, getweight))
                    conn.commit()
                else:
                    cur.execute("UPDATE progress SET p_weight=? WHERE u_id=? AND p_date=?", (getweight, session['uid'], track_date))
                    conn.commit()
                # Update user table
                cur.execute("UPDATE user SET u_weight=? WHERE u_id=?", (getweight, session['uid']))
                conn.commit()
                # Ensure a tracking row exists for today
                cur.execute("SELECT * FROM tracking WHERE u_id=? AND track_date=?", (session['uid'], track_date))
                tracking_row = cur.fetchone()
                if not tracking_row:
                    cur.execute("INSERT INTO tracking (track_date, u_id, track_calorie, track_protein, track_fat, track_carb, track_breakfast, track_lunch, track_snack, track_dinner) VALUES (?, ?, 0, 0, 0, 0, '', '', '', '')", (track_date, session['uid']))
                    conn.commit()
        except sqlite3.Error as e:
            print(f"Database error: {e}")
            return f"Database error: {e}"
        except Exception as e:
            print(f"Unexpected error: {e}")
            return f"Unexpected error: {e}"
        finally:
            try:
                conn.close()
            except:
                pass
        print("Redirecting to progress page after weight update")
        return redirect(url_for('progress'))

@app.route("/about", methods = ['GET','POST'])
def about():
	if request.method == 'GET':
		return render_template('about.html')

	else:
		return render_template('about.html')

@app.route('/logout')
def logout():
	
	session.pop('uid',None)
	session.pop('u_pass',None)
	session.pop('u_info',None)
	session.pop('bf_meal',None)
	session.pop('bf_numbers',None)
	session.pop('lunch_meal',None)
	session.pop('lunch_numbers',None)
	session.pop('dinner_meal',None)
	session.pop('dinner_numbers',None)
	session.pop('snack_meal',None)
	session.pop('snack_numbers',None)
	bf_meal.clear()
	lunch_meal.clear()
	snack_meal.clear()
	dinner_meal.clear()
	flash('You have successfully logged out')
	return redirect(url_for('login'))

@app.route("/contact", methods=['GET', 'POST'])
def contact():
    return render_template('contact.html')

@app.route("/exercise", methods=['GET', 'POST'])
def exercise():
    print("SESSION DEBUG:", dict(session))  # Debug print
    if 'uid' not in session:
        flash("🔒 Please log in to access the AI Exercise Planner. This feature is only available for registered users.", "warning")
        return redirect(url_for('login'))
    if not AI_AGENT_AVAILABLE:
        flash("🤖 AI agent is not available, but you can still get a great workout plan using our smart fallback generator!", "info")
    if request.method == 'POST' and request.form.get('regenerate'):
        # Clear current plan and regenerate
        session.pop('exercise_plan', None)
        session['exercise_step'] = 'generate_plan'
        flash('Exercise plan is being regenerated with your latest data!', 'info')
        return redirect(url_for('exercise'))
    
    if request.method == 'POST' and request.form.get('start_over'):
        # Clear all exercise session data and start fresh
        session.pop('exercise_plan', None)
        session.pop('exercise_step', None)
        session.pop('exercise_user_data', None)
        session.pop('exercise_user_data_debug', None)
        session.pop('plan_summary', None)
        session['exercise_step'] = 'collect_info'
        flash('Starting fresh! Please provide your information again.', 'info')
        return redirect(url_for('exercise'))
    try:
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM user WHERE u_id=?", (session['uid'],))
            user_row = cur.fetchone()
            if not user_row:
                session.clear()
                flash("⚠️ Your session has expired. Please log in again to access the Exercise Planner.", "warning")
                return redirect(url_for('login'))
            cur.execute("SELECT id, note FROM user_notes WHERE u_id=? ORDER BY created_at DESC", (session['uid'],))
            notes = cur.fetchall()
    except sqlite3.Error as e:
        flash("❌ Database error. Please try again later.", "error")
        return redirect(url_for('index'))

    # Correct user_data mapping based on schema
    user_data = {
        'name': user_row[1],                # u_username
        'gender': user_row[5],              # u_gender
        'age': user_row[4],                 # u_age
        'weight': user_row[8],              # u_weight
        'feet': user_row[9],                # u_feet
        'inches': user_row[10],             # u_inches
        'status': user_row[19],             # u_status
        'journey': user_row[20],            # u_journey
        'activity_level': user_row[12],     # u_activitylevel
        'goal': user_row[22],               # u_goal
    }
    if notes:
        user_data['notes'] = [n[1] for n in notes]
    else:
        user_data['notes'] = []

    step = session.get('exercise_step', 'collect_info')
    current_plan = session.get('exercise_plan', None)
    print("[DEBUG] step:", step)
    print("[DEBUG] current_plan:", current_plan)
    
    # Force fresh plan generation if requested
    if request.args.get('force_refresh'):
        session.pop('exercise_plan', None)
        session['exercise_step'] = 'generate_plan'
        return redirect(url_for('exercise'))

    if step == 'collect_info':
        session['exercise_user_data'] = user_data
        session['exercise_step'] = 'generate_plan'
        return redirect(url_for('exercise'))
    elif step == 'generate_plan':
        try:
            session['exercise_user_data_debug'] = user_data
            print(f"[DEBUG] Calling regenerate_plan with session['uid']: {session['uid']}")
            plan_text = regenerate_plan(session['uid'])
            print("[DEBUG] plan_text:", plan_text)
            session['exercise_plan'] = {
                'text': plan_text,
                'user_data': user_data,
                'ai_generated': AI_AGENT_AVAILABLE
            }
            session['exercise_step'] = 'show_plan'
            return redirect(url_for('exercise'))
        except Exception as e:
            flash(f"Error generating plan: {str(e)}", "error")
            session['exercise_step'] = 'collect_info'
            return redirect(url_for('exercise'))
    elif step == 'show_plan':
        if not current_plan:
            session['exercise_step'] = 'generate_plan'
            return redirect(url_for('exercise'))
        print("[DEBUG] Rendering plan:", current_plan)
        return render_template('exercise.html', 
                             step='show_plan', 
                             plan=current_plan,
                             show_start_over=True, user_data=user_data,
                             plan_summary=session.get('plan_summary'),
                             ai_input_data=session.get('exercise_user_data_debug'),
                             notes=notes)
    return redirect(url_for('index'))

@app.route('/edit_user_note/<int:note_id>', methods=['GET', 'POST'])
def edit_user_note(note_id):
    if request.method == 'POST':
        new_note = request.form.get('edit_note')
        try:
            with get_connection() as conn:
                cur = conn.cursor()
                cur.execute("UPDATE user_notes SET note=? WHERE id=? AND u_id=?", (new_note, note_id, session['uid']))
                conn.commit()
            flash('Note updated successfully! Exercise plan will be regenerated to reflect your changes.', 'success')
            # Clear the current plan to force regeneration
            session.pop('exercise_plan', None)
            session['exercise_step'] = 'generate_plan'
        except Exception as e:
            flash(f'Error updating note: {e}', 'danger')
        return redirect(url_for('exercise'))
    else:
        # For GET, show a simple edit form (optional, or redirect)
        flash('Edit via the exercise page form.', 'info')
        return redirect(url_for('exercise'))

@app.route('/delete_user_note/<int:note_id>', methods=['POST'])
def delete_user_note(note_id):
    try:
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM user_notes WHERE id=? AND u_id=?", (note_id, session['uid']))
            conn.commit()
        flash('Note deleted successfully! Exercise plan will be regenerated to reflect your changes.', 'success')
        # Clear the current plan to force regeneration
        session.pop('exercise_plan', None)
        session['exercise_step'] = 'generate_plan'
    except Exception as e:
        flash(f'Error deleting note: {e}', 'danger')
    return redirect(url_for('exercise'))

@app.route('/edit_user_feedback/<int:feedback_id>', methods=['GET', 'POST'])
def edit_user_feedback(feedback_id):
    if request.method == 'POST':
        new_feedback = request.form.get('edit_feedback')
        try:
            with get_connection() as conn:
                cur = conn.cursor()
                cur.execute("UPDATE user_feedback SET feedback_text=? WHERE id=? AND u_id=?", (new_feedback, feedback_id, session['uid']))
                conn.commit()
            flash('Feedback updated successfully! Exercise plan will be regenerated to reflect your changes.', 'success')
            # Clear the current plan to force regeneration
            session.pop('exercise_plan', None)
            session['exercise_step'] = 'generate_plan'
        except Exception as e:
            flash(f'Error updating feedback: {e}', 'danger')
        return redirect(url_for('exercise'))
    else:
        flash('Edit via the exercise page form.', 'info')
        return redirect(url_for('exercise'))

@app.route('/delete_user_feedback/<int:feedback_id>', methods=['POST'])
def delete_user_feedback(feedback_id):
    try:
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM user_feedback WHERE id=? AND u_id=?", (feedback_id, session['uid']))
            conn.commit()
        flash('Feedback deleted successfully! Exercise plan will be regenerated to reflect your changes.', 'success')
        # Clear the current plan to force regeneration
        session.pop('exercise_plan', None)
        session['exercise_step'] = 'generate_plan'
    except Exception as e:
        flash(f'Error deleting feedback: {e}', 'danger')
    return redirect(url_for('exercise'))

@app.route('/add_user_note', methods=['POST'])
def add_user_note():
    note_text = request.form.get('new_note')
    try:
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute("INSERT INTO user_notes (u_id, note, created_at) VALUES (?, ?, ?)", (session['uid'], note_text, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            conn.commit()
        flash('Note added successfully! Exercise plan will be regenerated to reflect your changes.', 'success')
        # Clear the current plan to force regeneration
        session.pop('exercise_plan', None)
        session['exercise_step'] = 'generate_plan'
    except Exception as e:
        flash(f'Error adding note: {e}', 'danger')
    return redirect(url_for('exercise'))

@app.route('/add_user_feedback', methods=['POST'])
def add_user_feedback():
    feedback_text = request.form.get('new_feedback')
    try:
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute("INSERT INTO user_feedback (u_id, feedback_text, created_at) VALUES (?, ?, ?)", (session['uid'], feedback_text, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            conn.commit()
        flash('Feedback added successfully! Exercise plan will be regenerated to reflect your changes.', 'success')
        # Clear the current plan to force regeneration
        session.pop('exercise_plan', None)
        session['exercise_step'] = 'generate_plan'
    except Exception as e:
        flash(f'Error adding feedback: {e}', 'danger')
    return redirect(url_for('exercise'))

if __name__=="__main__":
	app.run(port=5050,debug="true")
