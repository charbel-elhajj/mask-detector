import cv2
import time
import sys
from flask import Flask, render_template, Response, jsonify
from imutils import resize, video

from darknet.darknet import make_image, copy_image_from_bytes, detect_image, free_image, bbox2points, load_network, \
    network_width, network_height

app = Flask(__name__)
stream_fps = 0


# darknet helper function to run detection on image
def darknet_helper(img, width, height):
    darknet_image = make_image(width, height, 3)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (width, height), interpolation=cv2.INTER_LINEAR)

    # get image ratios to convert bounding boxes to proper size
    img_height, img_width, _ = img.shape
    width_ratio = img_width / width
    height_ratio = img_height / height

    # run model on darknet style image to get detections
    copy_image_from_bytes(darknet_image, img_resized.tobytes())
    detections = detect_image(network, class_names, darknet_image)
    free_image(darknet_image)
    return detections, width_ratio, height_ratio


def new_func():
    global stream_fps
    
    video_capture = cv2.VideoCapture(0)
    #fps counter
    fps = video.FPS().start()

    # used to record the time when we processed last frame
    prev_frame_time = 0
    # used to record the time at which we processed current frame
    new_frame_time = 0

    while True:
        ret, frame = video_capture.read()

        new_frame_time = time.time()
        # resize the frame
        frame = resize(frame, width=450)

        detections, width_ratio, height_ratio = darknet_helper(frame, width, height)

        for label, confidence, bbox in detections:
            x, y, w, h = bbox2points(bbox)
            x, y, w, h = int(x * width_ratio), int(y * height_ratio), int(w * width_ratio), int(h * height_ratio)
            # initializing the box and adding it to the frame
            bbox_array = cv2.rectangle(frame, (x, y), (w, h), class_colors[label], 2)
            bbox_array = cv2.putText(
                img=bbox_array,
                text="{} [{:.2f}]".format(label, float(confidence)),
                org=(x, y - 5),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=class_colors[label],
                thickness=2
            )

        # Display the resulting frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        cv2.waitKey()
        fps.update()
        stream_fps = 1/(new_frame_time-prev_frame_time)
        # print("[INFO] approx. FPS: {:.2f}".format(stream_fps), file=sys.stdout)
        prev_frame_time = new_frame_time

    # When everything is done, release the capture
    fps.stop()
    video_capture.release()


@app.route('/fps_count')
def fps_count():
    print("[INFO] approx. FPS: {:.2f}".format(stream_fps), file=sys.stdout)
    return jsonify({
        'fps': stream_fps,
    })


@app.route('/video_feed')
def video_feed():
    # Video streaming route. Put this in the src attribute of an img tag
    return Response(new_func(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


if __name__ == '__main__':
    network, class_names, class_colors = load_network(
        "cfg/yolov4-tiny.cfg",
        "data/face_mask.data",
        "backup/yolov4-tiny_best.weights"
    )
    width = network_width(network)
    height = network_height(network)
    app.run(host='0.0.0.0', debug=False)
