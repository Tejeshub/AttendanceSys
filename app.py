import numpy as np
import pandas as pd
import cv2
import os
import time
import base64
import csv
from datetime import datetime
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

DATASET_FOLDER = 'dataset/'  # Path to student images
ATTENDANCE_FILE = 'attendance.csv'  # CSV file to store attendance

# Create dataset folder if not exists
if not os.path.exists('dataset'):
    os.makedirs('dataset')
    
# Create attendance file with headers if it doesn't exist
if not os.path.exists(ATTENDANCE_FILE):
    with open(ATTENDANCE_FILE, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Name', 'Date', 'Day', 'Time', 'Status'])

# Folder where images are stored
dataset_folder = 'dataset/'
# Load the dataset images
def load_dataset():
    images = []
    labels = []
    student_names = []
    
    for filename in os.listdir(dataset_folder):
        if filename.endswith('.jpg'):
            img_path = os.path.join(dataset_folder, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            images.append(img)
            labels.append(filename.split('.')[0])  # Label is the name without extension
            student_names.append(filename.split('.')[0])
    
    return images, labels, student_names
# Path to the CSV file where attendance data will be saved
attendance_file = 'attendance.csv'

# Function to append student data to CSV file
def append_to_csv(name, date, time, img_path):
    # Open CSV file in append mode
    with open(attendance_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        # Write the data in the CSV format: [Name, Date, Time, Image Path]
        writer.writerow([name, date, time, img_path])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        if name:
            return render_template('register.html')
    return render_template('register.html')

@app.route('/register-student', methods=['POST'])
def register_student():
    name = request.form['name']
    image_data = request.form['image']
    
    # Clean the base64 image data (remove the prefix)
    image_data = image_data.split(',')[1]
    
    # Convert base64 to image
    img_data = base64.b64decode(image_data)
    
    # Create a unique filename using the student's name and timestamp
    timestamp = time.time()
    filename = f"dataset/{name}_{timestamp}.jpg"
    
    # Save the image in the 'dataset' folder
    with open(filename, 'wb') as f:
        f.write(img_data)
    
    # Get the current date and time
    current_time = datetime.now()
    date = current_time.strftime('%Y-%m-%d')
    time_str = current_time.strftime('%H:%M:%S')
    
    # Append student data to the CSV
    append_to_csv(name, date, time_str, filename)

    return jsonify({"success": True, "message": "Student Registered!", "image_path": filename})

# @app.route('/mark-attendance', methods=['GET', 'POST'])
# def mark_attendance():
#     if request.method == 'POST':
#         # Implement attendance marking logic here
#         pass
#     return render_template('mark_attendance.html')

@app.route('/view-attendance', methods=['GET'])
def view_attendance():
    """Retrieve and show the attendance records from CSV."""
    attendance_records = []
    
    # Check if attendance file exists
    if os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, 'r', newline='') as file:
            reader = csv.reader(file)
            # Skip header row but capture it for template
            headers = next(reader, ['Name', 'Date', 'Day', 'Time', 'Status'])
            attendance_records = list(reader)
            
        return render_template('view_attendance.html', 
                               headers=headers, 
                               attendance_records=attendance_records)
    else:
        # If file doesn't exist, show empty table with headers
        headers = ['Name', 'Date', 'Day', 'Time', 'Status']
        return render_template('view_attendance.html', 
                               headers=headers, 
                               attendance_records=[],
                               message="No attendance records found")


@app.route('/mark-attendance',  methods=['GET', 'POST'])
def mark_attendance():
    """Marks attendance by comparing captured image with stored dataset images."""
    
    data = request.get_json()
    img_data = data['image'].split(',')[1]  # Remove base64 header
    img_bytes = base64.b64decode(img_data)
    np_img = np.frombuffer(img_bytes, dtype=np.uint8)
    captured_img = cv2.imdecode(np_img, cv2.IMREAD_GRAYSCALE)

    # Load dataset
    images, labels, student_names, label_map = load_dataset()

    if not images:
        return jsonify({"message": "No student data found!", "status": "error"})

    # Train the face recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(images, np.array(labels))

    try:
        student_id, confidence = recognizer.predict(captured_img)
        if confidence < 70:  # Adjust threshold as needed
            student_name = label_map[student_id]

            # Mark attendance in CSV if not already marked
            current_time = datetime.now()
            date = current_time.strftime('%Y-%m-%d')
            time_str = current_time.strftime('%H:%M:%S')

            with open(attendance_file, 'r', newline='') as file:
                reader = csv.reader(file)
                attendance_data = list(reader)

            if any(row[0] == student_name and row[1] == date for row in attendance_data):
                return jsonify({"message": "Attendance already marked!", "status": "error"})

            with open(attendance_file, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([student_name, date, time_str])

            return jsonify({"message": f"Attendance marked for {student_name}!", "status": "success"})
    except Exception:
        return jsonify({"message": "No match found. Try again!", "status": "error"})
    if request.content_type != 'application/json':
        return jsonify({'error': 'Unsupported Media Type. Use application/json'}), 415
    
    try:
        data = request.get_json()  # Get JSON data
        if not data or 'image' not in data:
            return jsonify({'error': 'Invalid request, missing "image" field'}), 400
        
        # Process the image data (currently just returning success)
        return jsonify({'message': 'Attendance marked successfully'})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
@app.route('/get-students', methods=['GET'])
def get_students():
    """Fetch all student images and names from the dataset folder."""
    students = []

    for filename in os.listdir(DATASET_FOLDER):
        if filename.endswith(('.jpg', '.jpeg', '.png')):  # Only process images
            student_name = os.path.splitext(filename)[0]  # Remove file extension
            image_path = os.path.join(DATASET_FOLDER, filename)

            # Convert image to base64 for frontend display
            with open(image_path, "rb") as img_file:
                base64_image = base64.b64encode(img_file.read()).decode('utf-8')

            students.append({"name": student_name, "image": base64_image})

    return jsonify(students)


# @app.route('/mark-attendance', methods=['POST'])
# def mark_attendance():
#     """Mark student attendance based on captured image."""
#     try:
#         # Check if the request contains JSON data
#         if not request.is_json:
#             return jsonify({"message": "Invalid request format", "status": "error"}), 400
            
#         data = request.get_json()
        
#         # Check if image data exists in the request
#         if not data or 'image' not in data:
#             return jsonify({"message": "Missing image data", "status": "error"}), 400
            
#         # Get the base64 image data and remove header if present
#         img_data = data['image']
#         if ',' in img_data:
#             img_data = img_data.split(',')[1]
            
#         # Decode base64 to image
#         img_bytes = base64.b64decode(img_data)
        
#         # Convert to numpy array for OpenCV
#         np_arr = np.frombuffer(img_bytes, np.uint8)
#         img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
#         # Convert to grayscale for face detection
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
#         # Get current date and time
#         current_time = datetime.now()
#         date = current_time.strftime('%Y-%m-%d')
#         day = current_time.strftime('%A')
#         time_str = current_time.strftime('%H:%M:%S')
        
#         # Load all student images from dataset
#         students = []
#         for filename in os.listdir(DATASET_FOLDER):
#             if filename.endswith(('.jpg', '.jpeg', '.png')):
#                 student_name = os.path.splitext(filename)[0]
#                 if '_' in student_name:  # If filename has timestamp, get just the name
#                     student_name = student_name.split('_')[0]
#                 students.append(student_name)
        
#         # For now, just mark attendance for the first student (demo purposes)
#         # In a real system, you'd implement face recognition here
#         if students:
#             student_name = students[0]
            
#             # Write to CSV
#             with open(ATTENDANCE_FILE, 'a', newline='') as file:
#                 writer = csv.writer(file)
#                 writer.writerow([student_name, date, day, time_str, "Present"])
                
#             return jsonify({
#                 "message": f"Attendance marked for {student_name}!",
#                 "status": "success"
#             })
#         else:
#             return jsonify({
#                 "message": "No students found in the database",
#                 "status": "error"
#             })
            
#     except Exception as e:
#         print(f"Error: {str(e)}")
#         return jsonify({
#             "message": f"Error marking attendance: {str(e)}",
#             "status": "error"
#         }), 500

    

if __name__ == '__main__':
    app.run(debug=True)
