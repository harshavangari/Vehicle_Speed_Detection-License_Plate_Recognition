from flask import Flask, render_template, Response, request, send_from_directory
import cv2
import dlib
import time
import math
import os
from image_to_text import image_to_text  # Import your OCR function

app = Flask(__name__)

carCascade = cv2.CascadeClassifier('myhaar.xml')
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output_frames'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'jpg', 'jpeg', 'png'}  # Add image extensions
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

WIDTH = 1280
HEIGHT = 720

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def calculate_speed(location1, location2):
    d_pixels = math.sqrt(math.pow(location2[0] - location1[0], 2) + math.pow(location2[1] - location1[1], 2))
    ppm = 8.8
    d_meters = d_pixels / ppm
    fps = 12
    speed = d_meters * fps * 3.6
    return speed

def ObjectsTracking(video_path, speed_limit):
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise FileNotFoundError("Unable to open video file.")

    rectangleColor = (0, 255, 0)
    frameCounter = 0
    currentCarID = 0
    fps = 0

    carTracker = {}
    carLocation1 = {}
    carLocation2 = {}

    speed = [None] * 1000

    while True:
        start_time = time.time()
        rc, image = video.read()
        if not rc:
            break

        image = cv2.resize(image, (WIDTH, HEIGHT))
        resultImage = image.copy()

        frameCounter += 1

        carIDtoDelete = []

        for carID in carTracker.keys():
            trackingQuality = carTracker[carID].update(image)
            if trackingQuality < 7:
                carIDtoDelete.append(carID)

        for carID in carIDtoDelete:
            carTracker.pop(carID, None)
            carLocation1.pop(carID, None)
            carLocation2.pop(carID, None)

        if not (frameCounter % 10):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cars = carCascade.detectMultiScale(gray, 1.1, 13, 18, (24, 24))

            for (_x, _y, _w, _h) in cars:
                x = int(_x)
                y = int(_y)
                w = int(_w)
                h = int(_h)
                x_bar = x + 0.5 * w
                y_bar = y + 0.5 * h
                matchCarID = None

                for carID in carTracker.keys():
                    trackedPosition = carTracker[carID].get_position()
                    t_x = int(trackedPosition.left())
                    t_y = int(trackedPosition.top())
                    t_w = int(trackedPosition.width())
                    t_h = int(trackedPosition.height())
                    t_x_bar = t_x + 0.5 * t_w
                    t_y_bar = t_y + 0.5 * t_h

                    if ((t_x <= x_bar <= (t_x + t_w)) and (t_y <= y_bar <= (t_y + t_h)) and (x <= t_x_bar <= (x + w)) and (y <= t_y_bar <= (y + h))):
                        matchCarID = carID

                if matchCarID is None:
                    tracker = dlib.correlation_tracker()
                    tracker.start_track(image, dlib.rectangle(x, y, x + w, y + h))
                    carTracker[currentCarID] = tracker
                    carLocation1[currentCarID] = [x, y, w, h]
                    currentCarID += 1

        for carID in carTracker.keys():
            trackedPosition = carTracker[carID].get_position()
            t_x = int(trackedPosition.left())
            t_y = int(trackedPosition.top())
            t_w = int(trackedPosition.width())
            t_h = int(trackedPosition.height())
            cv2.rectangle(resultImage, (t_x, t_y), (t_x + t_w, t_y + t_h), rectangleColor, 4)
            carLocation2[carID] = [t_x, t_y, t_w, t_h]

        end_time = time.time()
        if not (end_time == start_time):
            fps = 1.0 / (end_time - start_time)

        for i in carLocation1.keys():
            if frameCounter % 1 == 0:
                [x1, y1, w1, h1] = carLocation1[i]
                [x2, y2, w2, h2] = carLocation2[i]
                carLocation1[i] = [x2, y2, w2, h2]
                if [x1, y1, w1, h1] != [x2, y2, w2, h2]:
                    if (speed[i] == None or speed[i] == 0) and y1 >= 275 and y1 <= 285:
                        speed[i] = calculate_speed([x1, y1, w1, h1], [x2, y2, w2, h2])
                    if speed[i] is not None and speed[i] > speed_limit and y1 >= 180:
                        # Draw speed on the image
                        cv2.putText(resultImage, str(int(speed[i])) + " km/hr", (int(x1 + w1 / 2), int(y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                        
                        # Save the frame as an image
                        if not os.path.exists(app.config['OUTPUT_FOLDER']):
                            os.makedirs(app.config['OUTPUT_FOLDER'])
                        frame_filename = f"frame_{frameCounter}_car_{i}.jpg"
                        frame_path = os.path.join(app.config['OUTPUT_FOLDER'], frame_filename)
                        cv2.imwrite(frame_path, resultImage)

                        # Yield the image as bytes
                        _, jpeg = cv2.imencode('.jpg', resultImage)
                        frame = jpeg.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    video.release()
    cv2.destroyAllWindows()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed', methods=['POST'])
def video_feed():
    speed_limit = int(request.form['speed_limit'])
    video_file = request.files['video_file']

    if video_file and allowed_file(video_file.filename):
        # Create uploads folder if it doesn't exist
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])

        video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_file.filename)
        video_file.save(video_path)
        return Response(ObjectsTracking(video_path, speed_limit), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return "Invalid file format. Please upload a valid video file."

@app.route('/image_process', methods=['POST'])
def image_process():
    image_file = request.files['image_file']

    if image_file and allowed_file(image_file.filename):
        # Save the uploaded image
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
        image_file.save(image_path)

        # Process the image to extract text
        extracted_text = image_to_text(image_path)

        # Render a template or return the text as JSON
        return render_template('image_result.html', image_file=image_file.filename, extracted_text=extracted_text)
    else:
        return "Invalid file format. Please upload a valid image file."

@app.route('/exceeded_images')
def exceeded_images():
    image_filenames = [f for f in os.listdir(app.config['OUTPUT_FOLDER']) if allowed_file(f)]
    return render_template('exceeded_images.html', images=image_filenames)

@app.route('/output_frames/<filename>')
def send_image(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
