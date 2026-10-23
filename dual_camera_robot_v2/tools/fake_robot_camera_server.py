"""Test double of the robot camera server (gear_sonic ComposedCameraServer): ZMQ PUB, msgpack
{"timestamps": {"head": t, "ego_view": t}, "images": {"head": ..., "ego_view": ...}} with time.time() stamps.

Two encodings exist on the real robot, both are reproduced:
  --encoding b64   : server JPEG-encodes the driver's RGB array with cv2 and sends base64 str (USB / RealSense drivers)
  --encoding mjpeg : raw true-colour JPEG bytes from the camera (OAK with on-device MJPEG)
Plays two recorded videos in real time at --fps; --skew-ms shifts the ego_view timestamp to test the sync check.

  python -B tools/fake_robot_camera_server.py --person reference/episode_000105/person.mp4 --hand reference/episode_000105/hand.mp4
"""
import argparse
import base64
import time

import cv2
import msgpack
import zmq


def encode(bgr, how):
    if how == 'mjpeg':
        return cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)          # drivers publish RGB arrays
    return base64.b64encode(cv2.imencode('.jpg', rgb, [cv2.IMWRITE_JPEG_QUALITY, 80])[1]).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--person', required=True); ap.add_argument('--hand', required=True)
    ap.add_argument('--port', type=int, default=5555); ap.add_argument('--fps', type=float, default=30)
    ap.add_argument('--encoding', choices=('b64', 'mjpeg'), default='b64'); ap.add_argument('--skew-ms', type=float, default=0)
    ap.add_argument('--wait', type=float, default=8, help='seconds before streaming (let the client connect and warm up)')
    a = ap.parse_args()
    frames = []
    for path in (a.person, a.hand):
        cap = cv2.VideoCapture(path); fs = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            fs.append(f)
        frames.append(fs); src_fps = cap.get(cv2.CAP_PROP_FPS)
    sock = zmq.Context().socket(zmq.PUB); sock.setsockopt(zmq.SNDHWM, 20); sock.bind(f'tcp://*:{a.port}')
    time.sleep(a.wait)
    start = time.time(); n = 0
    while True:
        i = int((time.time() - start) * src_fps)
        if i >= len(frames[0]):
            break
        now = time.time()
        sock.send(msgpack.packb({'timestamps': {'head': now, 'ego_view': now + a.skew_ms / 1000},
                                 'images': {'head': encode(frames[0][i], a.encoding), 'ego_view': encode(frames[1][i], a.encoding)}},
                                use_bin_type=True))
        n += 1
        time.sleep(max(0., start + n / a.fps - time.time()))
    print(f'sent {n} messages, {len(frames[0]) / src_fps:.1f} s of video', flush=True)


if __name__ == '__main__':
    main()
