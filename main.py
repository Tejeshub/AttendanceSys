import cv2
import os
import numpy as np
import pandas as pd
from datetime import datetime
import time

# Create directories if they don't exist
if not os.path.exists("dataset"):
    os.makedirs("dataset")
if not os.path.exists("attendance"):
    os.makedirs("attendance")

# File to save today's attendance
def get_attendance_file():
    date = datetime.now().strftime("%Y-%m-%d")
    filename = f"attendance/attendance_{date}.csv"
    if not os.path.exists(filename):
        df = pd.DataFrame(columns=['Name', 'Time', 'Date'])
        df.to_csv(filename, index=False)
    return filename

# Function to add attendance
def mark_attendance(name):
    filename = get_attendance_file()
    df = pd.read_csv(filename)
    
    # Check if already marked attendance today
    date = datetime.now().strftime("%Y-%m-%d")
    if not df[(df['Name'] == name) & (df['Date'] == date)].empty:
        print(f"{name} already marked attendance today!")
        return
    
    # Add new attendance
    now = datetime.now()
    time_str = now.strftime("%H:%M:%S")
    date_str = now.strftime("%Y-%m-%d")
    
    new_row = pd.DataFrame({'Name': [name], 'Time': [time_str], 'Date': [date_str]})
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(filename, index=False)
    print(f"Marked attendance for {name}")

# Function to create dataset for a person with enhanced UI
def create_dataset(name):
    # Create directory for person if it doesn't exist
    person_dir = f"dataset/{name}"
    if not os.path.exists(person_dir):
        os.makedirs(person_dir)
    
    # Initialize webcam with better resolution
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    count = 0
    target_samples = 50  # Increased for better recognition
    sample_delay = 300  # Milliseconds between captures
    last_capture_time = 0
    
    # UI elements
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Create window with properties
    cv2.namedWindow("Dataset Creation", cv2.WINDOW_NORMAL)
    
    print(f"\n[INFO] Starting face capture for {name}. Look at the camera and move your head slightly...")
    
    # For tracking different face angles
    face_angles = {
        "front": 0,
        "left": 0,
        "right": 0,
        "up": 0,
        "down": 0
    }
    
    # Current capture mode instruction
    current_angle = "front"
    angle_count_target = 10  # Samples per angle
    
    # Start time for FPS calculation
    start_time = time.time()
    frame_count = 0
    fps = 0
    
    # For progress calculation
    progress_percent = 0
    
    while True:
        ret, frame = cam.read()
        if not ret:
            break
            
        # FPS calculation
        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time > 1:
            fps = frame_count / elapsed_time
            frame_count = 0
            start_time = time.time()
        
        # Create a copy for display
        display_frame = frame.copy()
        
        # Process image for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)  # Improve contrast
        
        # Detect faces with optimized parameters
        faces = face_detector.detectMultiScale(
            gray, 
            scaleFactor=1.2, 
            minNeighbors=5,
            minSize=(60, 60),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Create UI header
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 100), (32, 32, 32), -1)
        cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0, display_frame)
        
        # Draw title and status info
        cv2.putText(display_frame, f"Creating Dataset for: {name}", (20, 30), 
                   font, 1, (0, 255, 255), 2)
        cv2.putText(display_frame, f"Captured: {count}/{target_samples} images | FPS: {int(fps)}", 
                   (20, 60), font, 0.7, (255, 255, 255), 1)
        
        # Calculate and draw progress bar
        progress_percent = (count / target_samples) * 100
        bar_width = int((frame.shape[1] - 40) * (progress_percent / 100))
        cv2.rectangle(display_frame, (20, 80), (20 + bar_width, 90), (0, 255, 0), -1)
        cv2.rectangle(display_frame, (20, 80), (frame.shape[1] - 20, 90), (200, 200, 200), 1)
        
        # Add current instruction based on angle
        instruction = f"Please look {current_angle} ({face_angles[current_angle]}/{angle_count_target})"
        cv2.putText(display_frame, instruction, (20, frame.shape[0] - 20), 
                   font, 0.7, (0, 255, 255), 1)
        
        # Current time for capture timing
        current_time = int(time.time() * 1000)
        
        # Process detected faces
        largest_face = None
        largest_area = 0
        
        for (x, y, w, h) in faces:
            # Find the largest face in the frame
            area = w * h
            if area > largest_area:
                largest_area = area
                largest_face = (x, y, w, h)
        
        # Process only the largest face
        if largest_face:
            x, y, w, h = largest_face
            
            # Draw face outline with more attractive UI
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 165, 255), 2)
            
            # Add animated corners to the rectangle when ready to capture
            if current_time - last_capture_time > sample_delay:
                # Draw corners animation
                corner_length = 20
                # Top-left
                cv2.line(display_frame, (x, y), (x + corner_length, y), (0, 255, 0), 3)
                cv2.line(display_frame, (x, y), (x, y + corner_length), (0, 255, 0), 3)
                # Top-right
                cv2.line(display_frame, (x + w, y), (x + w - corner_length, y), (0, 255, 0), 3)
                cv2.line(display_frame, (x + w, y), (x + w, y + corner_length), (0, 255, 0), 3)
                # Bottom-left
                cv2.line(display_frame, (x, y + h), (x + corner_length, y + h), (0, 255, 0), 3)
                cv2.line(display_frame, (x, y + h), (x, y + h - corner_length), (0, 255, 0), 3)
                # Bottom-right
                cv2.line(display_frame, (x + w, y + h), (x + w - corner_length, y + h), (0, 255, 0), 3)
                cv2.line(display_frame, (x + w, y + h), (x + w, y + h - corner_length), (0, 255, 0), 3)
            
            # Capture face with controlled timing
            if current_time - last_capture_time > sample_delay:
                last_capture_time = current_time
                
                # Extract and process face for better quality
                face_img = gray[y:y+h, x:x+w]
                
                # Resize and normalize
                face_img = cv2.resize(face_img, (200, 200))
                
                # Save with enhanced naming convention that includes angle
                filename = f"{person_dir}/{name}_{current_angle}_{face_angles[current_angle]:02d}.jpg"
                cv2.imwrite(filename, face_img)
                
                # Update counters
                count += 1
                face_angles[current_angle] += 1
                
                # Switch to next angle if needed
                if face_angles[current_angle] >= angle_count_target:
                    # Find next angle that needs samples
                    next_angles = [angle for angle, count in face_angles.items() if count ]

# Function to train the face recognizer with enhanced accuracy
def train_model():
    path = 'dataset'
    
    if not os.path.exists(path):
        print("Dataset folder not found!")
        return None
        
    # Get all person folders
    person_folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    
    if not person_folders:
        print("No person data found in dataset!")
        return None
    
    face_samples = []
    person_ids = []
    name_map = {}
    
    # Use a better face detector for preprocessing - Haar Cascade but with optimized parameters
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Enhanced preprocessing parameters
    min_neighbors = 5
    scale_factor = 1.2  # More precise scale factor
    
    print("[INFO] Processing training images...")
    success_count = 0
    total_images = 0
    
    # Process each person's images
    for idx, person in enumerate(person_folders):
        person_path = os.path.join(path, person)
        name_map[idx] = person
        print(f"[INFO] Processing images for {person}...")
        
        for img_name in os.listdir(person_path):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            total_images += 1
            img_path = os.path.join(person_path, img_name)
            
            # Read image and convert to grayscale
            img = cv2.imread(img_path)
            if img is None:
                print(f"[WARNING] Could not read image: {img_path}")
                continue
                
            # Apply histogram equalization for better contrast
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            
            # Detect face in the image with optimized parameters
            faces = face_detector.detectMultiScale(
                gray, 
                scaleFactor=scale_factor, 
                minNeighbors=min_neighbors,
                minSize=(30, 30)
            )
            
            # Process each face and normalize
            for (x, y, w, h) in faces:
                face_img = gray[y:y+h, x:x+w]
                
                # Resize to standard size for consistency
                face_img = cv2.resize(face_img, (200, 200))
                
                # Apply additional preprocessing for better recognition
                face_img = cv2.GaussianBlur(face_img, (5, 5), 0)
                
                face_samples.append(face_img)
                person_ids.append(idx)
                success_count += 1
    
    print(f"[INFO] Successfully processed {success_count} faces out of {total_images} images")
    
    # Create and train the model
    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=2,           # Optimized LBPH parameters
        neighbors=8,
        grid_x=8,
        grid_y=8,
        threshold=100.0     # More strict threshold
    )
    
    if not face_samples:
        print("No faces detected in the dataset!")
        return None
    
    print("[INFO] Training model - this may take a moment...")    
    recognizer.train(face_samples, np.array(person_ids))
    
    # Save the model and name mapping
    recognizer.save("trainer.yml")
    
    # Save name mapping to file
    with open("name_map.txt", "w") as f:
        for idx, name in name_map.items():
            f.write(f"{idx},{name}\n")
    
    print(f"[INFO] {len(np.unique(person_ids))} persons trained with {success_count} facial images. Model saved.")
    return recognizer, name_map

# Function to recognize faces and mark attendance with enhanced UI and accuracy
def recognize_attendance():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    
    # Check if trained model exists
    if not os.path.exists("trainer.yml"):
        print("Trained model not found! Please train the model first.")
        return
        
    recognizer.read("trainer.yml")
    
    # Load name mapping
    name_map = {}
    with open("name_map.txt", "r") as f:
        for line in f:
            idx, name = line.strip().split(",")
            name_map[int(idx)] = name
    
    # Face detector with optimized parameters
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Start webcam with better resolution if available
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Set minimum confidence with adaptive threshold
    base_min_confidence = 70  # Increased threshold for better accuracy
    
    # Store recognized names to avoid duplicate recognitions
    recognized = set()
    recognition_history = {}  # Track multiple recognitions for confidence
    last_recognition_time = time.time()
    
    # UI elements
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    box_padding = 10
    
    # Create a named window with properties
    cv2.namedWindow("Attendance System", cv2.WINDOW_NORMAL)
    
    # For FPS calculation
    fps_start_time = time.time()
    fps_counter = 0
    fps = 0
    
    # Status message
    status_message = "Ready to recognize faces"
    
    print("\n[INFO] Starting face recognition for attendance...")
    
    while True:
        ret, frame = cam.read()
        if not ret:
            break
        
        # Calculate FPS
        fps_counter += 1
        if (time.time() - fps_start_time) > 1:
            fps = fps_counter
            fps_counter = 0
            fps_start_time = time.time()
            
        # Copy original frame for UI overlay
        display_frame = frame.copy()
        
        # Create enhanced UI with info panel
        # Draw a semi-transparent overlay for the info panel
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 60), (32, 32, 32), -1)
        cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0, display_frame)
        
        # Add title and status
        cv2.putText(display_frame, "Face Recognition Attendance System", (20, 30), 
                   font, 1, (0, 255, 255), thickness)
        cv2.putText(display_frame, f"Status: {status_message} | FPS: {fps}", 
                   (20, 55), font, 0.6, (255, 255, 255), 1)
            
        # Apply preprocessing for better face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)  # Improve contrast for better detection
        
        # Detect faces with optimized parameters
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(60, 60),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Update status based on detection
        if len(faces) == 0:
            status_message = "No faces detected"
        else:
            status_message = f"Detected {len(faces)} face(s)"
        
        # Process each detected face
        for i, (x, y, w, h) in enumerate(faces):
            # Draw a more attractive face rectangle
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 165, 255), 2)
            
            # Extract and preprocess face for recognition
            face_img = gray[y:y+h, x:x+w]
            
            # Apply same preprocessing as in training
            face_img = cv2.resize(face_img, (200, 200))
            face_img = cv2.GaussianBlur(face_img, (5, 5), 0)
            
            # Predict with multiple samples for higher accuracy
            confidence_sum = 0
            samples = 3
            predicted_ids = []
            
            for _ in range(samples):
                id_, confidence = recognizer.predict(face_img)
                predicted_ids.append(id_)
                confidence_sum += confidence
            
            # Use majority voting for final prediction
            from collections import Counter
            final_id = Counter(predicted_ids).most_common(1)[0][0]
            
            # Average confidence (lower is better in LBPH)
            avg_confidence = confidence_sum / samples
            display_confidence = 100 - int(avg_confidence)
            
            # Adaptive confidence threshold based on lighting
            brightness = np.mean(gray)
            min_confidence = base_min_confidence - (0.1 * brightness)  # Adjust threshold based on lighting
            
            # Get name with confidence check
            if display_confidence >= min_confidence:
                name = name_map.get(final_id, "Unknown")
                
                # Track consecutive recognitions to improve accuracy
                if name not in recognition_history:
                    recognition_history[name] = {"count": 0, "last_time": time.time()}
                
                # Update recognition history
                if time.time() - recognition_history[name]["last_time"] < 2.0:
                    recognition_history[name]["count"] += 1
                else:
                    recognition_history[name]["count"] = 1
                
                recognition_history[name]["last_time"] = time.time()
                
                # Enhanced UI for recognized person
                # Create background for name display
                text_size = cv2.getTextSize(f"{name}", font, font_scale, thickness)[0]
                cv2.rectangle(display_frame, 
                             (x, y - text_size[1] - 2*box_padding), 
                             (x + text_size[0] + 2*box_padding, y), 
                             (32, 165, 218), -1)
                
                # Display name with better formatting
                cv2.putText(display_frame, f"{name}", 
                           (x + box_padding, y - box_padding), 
                           font, font_scale, (255, 255, 255), thickness)
                
                # Show confidence in smaller text
                cv2.putText(display_frame, f"Confidence: {display_confidence}%", 
                           (x, y + h + 20), font, 0.6, (0, 165, 255), 1)
                
                # Mark attendance only after consistent recognition (3+ consistent recognitions)
                if (recognition_history[name]["count"] >= 3 and 
                    name not in recognized and 
                    (time.time() - last_recognition_time) > 3):
                    
                    # Update status with success message
                    status_message = f"✅ Marked attendance for {name}"
                    
                    # Create a success confirmation animation
                    overlay = display_frame.copy()
                    cv2.rectangle(overlay, (x-10, y-10), (x+w+10, y+h+10), (0, 255, 0), -1)
                    cv2.addWeighted(overlay, 0.3, display_frame, 0.7, 0, display_frame)
                    
                    # Actually mark attendance
                    mark_attendance(name)
                    recognized.add(name)
                    last_recognition_time = time.time()
            else:
                # Display as unknown with different UI
                cv2.rectangle(display_frame, (x, y-30), (x+w, y), (0, 0, 255), -1)
                cv2.putText(display_frame, "Unknown", (x+5, y-10), font, 0.6, (255, 255, 255), 1)
                cv2.putText(display_frame, f"Conf: {display_confidence}%", 
                           (x, y+h+20), font, 0.6, (0, 0, 255), 1)
        
        # Add footer with instructions
        footer_y = display_frame.shape[0] - 30
        cv2.rectangle(overlay, (0, footer_y-20), (frame.shape[1], frame.shape[0]), (32, 32, 32), -1)
        cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0, display_frame)
        cv2.putText(display_frame, "Press 'q' to quit | Press 'r' to reset recognition", 
                   (20, footer_y), font, 0.6, (255, 255, 255), 1)
        
        # Display the frame with all enhancements
        cv2.imshow("Attendance System", display_frame)
        
        # Process key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            # Reset recognition status
            recognized = set()
            recognition_history = {}
            status_message = "Recognition reset"
    
    # Clean up
    cam.release()
    cv2.destroyAllWindows()

# Main menu function
def main_menu():
    while True:
        print("\n===== Face Recognition Attendance System =====")
        print("1. Create new person dataset")
        print("2. Train the system")
        print("3. Take attendance")
        print("4. View attendance")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ")
        
        if choice == '1':
            name = input("Enter person's name: ")
            create_dataset(name)
        elif choice == '2':
            train_model()
        elif choice == '3':
            recognize_attendance()
        elif choice == '4':
            date = input("Enter date (YYYY-MM-DD) or press Enter for today: ")
            
            if not date:
                date = datetime.now().strftime("%Y-%m-%d")
                
            filename = f"attendance/attendance_{date}.csv"
            
            if os.path.exists(filename):
                df = pd.read_csv(filename)
                if df.empty:
                    print(f"No attendance records for {date}")
                else:
                    print(f"\nAttendance for {date}:")
                    print(df)
            else:
                print(f"No attendance records for {date}")
        elif choice == '5':
            print("Exiting system...")
            break
        else:
            print("Invalid choice! Please try again.")

# Start the system
if __name__ == "__main__":
    main_menu()