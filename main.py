import numpy as np
from ultralytics import YOLO
import cv2
import math
import csv
from collections import defaultdict
from sort.sort import *
from util import get_car, read_license_plate, write_csv

results = {}

mot_tracker = Sort()

# load models
coco_model = YOLO("yolov8n.pt")
license_plate_detector = YOLO("./models/license_plate_detector.pt")

# load video
cap = cv2.VideoCapture("./sample.mp4")

if not cap.isOpened():
    print("Error: Could not open video file")
    exit()

video_fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# save output video
out = cv2.VideoWriter(
    "output.mp4",
    cv2.VideoWriter_fourcc(*'mp4v'),
    video_fps,
    (frame_width, frame_height)
)

vehicles = [2, 3, 5, 7]  # car, motorcycle, bus, truck

# speed settings
pixel_to_meter = 0.12
speed_limit = 35

# tracking history
car_positions = defaultdict(list)
car_speeds = {}
smoothed_speeds = {}

# OCR memory
best_plate_text = {}
best_plate_score = {}

# overspeed memory
overspeed_records = {}

# tuning parameters
history_frames = 15
min_pixel_distance = 2
speed_smoothing_factor = 0.5
max_reasonable_speed = 180

frame_nmr = -1
ret = True

while ret:
    frame_nmr += 1
    ret, frame = cap.read()

    if not ret:
        break

    results[frame_nmr] = {}

    # detect vehicles
    detections = coco_model(frame)[0]
    detections_ = []

    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection

        if int(class_id) in vehicles:
            detections_.append([x1, y1, x2, y2, score])

    # track vehicles
    track_ids = mot_tracker.update(np.asarray(detections_))

    # calculate speed for every tracked vehicle
    for track in track_ids:
        xcar1, ycar1, xcar2, ycar2, car_id = track

        car_id = int(car_id)

        center_x = int((xcar1 + xcar2) / 2)
        center_y = int((ycar1 + ycar2) / 2)

        car_positions[car_id].append((frame_nmr, center_x, center_y))

        # keep limited history
        if len(car_positions[car_id]) > history_frames:
            car_positions[car_id].pop(0)

        speed_kmh = car_speeds.get(car_id, 0)

        if len(car_positions[car_id]) >= 3:

            speed_samples = []

            for i in range(2, len(car_positions[car_id])):
                f1, x1, y1 = car_positions[car_id][i - 2]
                f2, x2, y2 = car_positions[car_id][i]

                pixel_distance = math.sqrt(
                    (x2 - x1) ** 2 +
                    (y2 - y1) ** 2
                )

                frame_diff = f2 - f1

                if frame_diff <= 0:
                    continue

                if pixel_distance < min_pixel_distance:
                    continue

                time_seconds = frame_diff / video_fps
                meter_distance = pixel_distance * pixel_to_meter

                speed_mps = meter_distance / time_seconds
                instant_speed = speed_mps * 3.6

                if 0 < instant_speed < max_reasonable_speed:
                    speed_samples.append(instant_speed)

            if len(speed_samples) > 0:
                avg_speed = sum(speed_samples) / len(speed_samples)

                prev_speed = smoothed_speeds.get(car_id, avg_speed)

                smooth_speed = (
                    speed_smoothing_factor * prev_speed +
                    (1 - speed_smoothing_factor) * avg_speed
                )

                smoothed_speeds[car_id] = smooth_speed
                speed_kmh = smooth_speed

        car_speeds[car_id] = speed_kmh

        # overspeed color
        vehicle_color = (0, 0, 255) if speed_kmh > speed_limit else (0, 255, 0)

        # draw vehicle box
        cv2.rectangle(
            frame,
            (int(xcar1), int(ycar1)),
            (int(xcar2), int(ycar2)),
            vehicle_color,
            2
        )

        # vehicle ID + speed
        cv2.putText(
            frame,
            f"ID {car_id} | {int(speed_kmh)} km/h",
            (int(xcar1), int(ycar1) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

    # detect license plates
    license_plates = license_plate_detector(frame)[0]

    for license_plate in license_plates.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = license_plate

        # assign license plate to car
        xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

        if car_id == -1:
            continue

        car_id = int(car_id)

        # crop plate
        license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2)]

        if license_plate_crop.size == 0:
            continue

        # preprocess plate image
        license_plate_crop_gray = cv2.cvtColor(
            license_plate_crop,
            cv2.COLOR_BGR2GRAY
        )

        license_plate_crop_gray = cv2.GaussianBlur(
            license_plate_crop_gray,
            (3, 3),
            0
        )

        license_plate_crop_thresh = cv2.adaptiveThreshold(
            license_plate_crop_gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            15
        )

        # OCR
        license_plate_text, license_plate_text_score = read_license_plate(
            license_plate_crop_thresh
        )

        speed_kmh = car_speeds.get(car_id, 0)

        # keep best OCR result only
        if license_plate_text is not None:

            current_best_score = best_plate_score.get(car_id, 0)

            if license_plate_text_score > current_best_score:
                best_plate_text[car_id] = license_plate_text
                best_plate_score[car_id] = license_plate_text_score

        final_plate_text = best_plate_text.get(car_id, "Unknown")

        speed_label = f"{int(speed_kmh)} km/h"

        # mark overspeeding vehicles
        overspeed = speed_kmh > speed_limit

        # store overspeed data
        if overspeed:

            if car_id not in overspeed_records:
                overspeed_records[car_id] = {
                    'vehicle_id': car_id,
                    'number_plate': final_plate_text,
                    'max_speed': int(speed_kmh),
                    'overspeed_frames': [frame_nmr]
                }
            else:
                overspeed_records[car_id]['number_plate'] = final_plate_text

                overspeed_records[car_id]['max_speed'] = max(
                    overspeed_records[car_id]['max_speed'],
                    int(speed_kmh)
                )

                if frame_nmr not in overspeed_records[car_id]['overspeed_frames']:
                    overspeed_records[car_id]['overspeed_frames'].append(frame_nmr)

        results[frame_nmr][car_id] = {
            'car': {
                'bbox': [xcar1, ycar1, xcar2, ycar2]
            },
            'car_speed': speed_label,
            'overspeed': overspeed,
            'license_plate': {
                'bbox': [x1, y1, x2, y2],
                'text': final_plate_text,
                'bbox_score': score,
                'text_score': best_plate_score.get(car_id, 0)
            }
        }

        # plate box color
        plate_color = (0, 0, 255) if overspeed else (255, 0, 0)

        cv2.rectangle(
            frame,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            plate_color,
            2
        )

        # plate text
        cv2.putText(
            frame,
            f"Plate: {final_plate_text}",
            (int(x1), int(y1) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        # overspeed warning
        if overspeed:
            cv2.putText(
                frame,
                "OVERSPEED",
                (int(xcar1), int(ycar2) + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                3
            )

    # show frame
    cv2.imshow("Vehicle Speed and Plate Detection", frame)

    # save frame
    out.write(frame)

    # quit on q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# cleanup
cap.release()
out.release()
cv2.destroyAllWindows()

# write normal CSV
write_csv(results, "./speed_test.csv")

# write overspeed CSV
with open("overspeed_vehicles.csv", mode="w", newline="") as file:
    writer = csv.writer(file)

    writer.writerow([
        "vehicle_id",
        "number_plate",
        "max_speed_kmh",
        "overspeed_frames"
    ])

    for car_id, data in overspeed_records.items():
        writer.writerow([
            data['vehicle_id'],
            data['number_plate'],
            data['max_speed'],
            ', '.join(map(str, data['overspeed_frames']))
        ])