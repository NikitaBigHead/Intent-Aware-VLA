"""Frame-pair sources. Every source returns BGR frames (as cv2 decodes the recorded training videos)
plus a per-camera capture timestamp in seconds.

- VideoPairReplay  : the two recorded episode videos, every 5th frame of the 50 Hz recording = 10 Hz (exact training sampling).
- RobotZmqSource   : the robot camera server (gear_sonic ComposedCameraServer, ZMQ PUB, msgpack
                     {"timestamps": {key: t}, "images": {key: jpeg}}); keys "head" (person) and "ego_view" (hands).
- CvPairSource     : two OpenCV sources (USB index, rtsp/http/udp URL or file).
"""
import base64
import threading
import time

import cv2
import numpy as np


class VideoPairReplay:
    def __init__(self, person_video, hand_video, step=5):
        self.caps = [cv2.VideoCapture(str(person_video)), cv2.VideoCapture(str(hand_video))]
        if not all(c.isOpened() for c in self.caps):
            raise RuntimeError('cannot open replay videos')
        self.fps = [c.get(cv2.CAP_PROP_FPS) for c in self.caps]
        self.frames = [int(c.get(cv2.CAP_PROP_FRAME_COUNT)) for c in self.caps]
        if abs(self.fps[0] - self.fps[1]) > 1e-3 or self.frames[0] != self.frames[1]:
            raise RuntimeError(f'cameras are not on one recording grid: fps {self.fps}, frames {self.frames}')
        self.step = step

    def __iter__(self):
        i = 0
        while True:
            ok0, f0 = self.caps[0].read(); ok1, f1 = self.caps[1].read()
            if not (ok0 and ok1):
                break
            if i % self.step == 0:
                t = i / self.fps[0]
                yield dict(t_person=t, t_hand=t, person=f0, hand=f1, frame_index=i, received=time.monotonic())
            i += 1
        for c in self.caps:
            c.release()


def _decode(value):
    """Same semantics as gear_sonic ImageMessageSchema.deserialize, then RGB -> BGR for RTMW3D."""
    if isinstance(value, (bytes, bytearray)):          # raw MJPEG from the camera: imdecode gives true-colour BGR
        return cv2.imdecode(np.frombuffer(value, np.uint8), cv2.IMREAD_COLOR)
    if isinstance(value, str):                          # base64 JPEG of an RGB array (server encoded RGB as-is)
        rgb = cv2.imdecode(np.frombuffer(base64.b64decode(value), np.uint8), cv2.IMREAD_COLOR)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    arr = np.asarray(value)                             # raw RGB ndarray
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR) if arr.ndim == 3 and arr.shape[2] == 3 else None


class RobotZmqSource:
    """Read-only ZMQ SUB client of the robot camera server. Never sends anything to the robot."""

    def __init__(self, host, port=5555, person_key='head', hand_key='ego_view'):
        import msgpack
        import msgpack_numpy
        import zmq
        self.msgpack, self.decode_hook = msgpack, msgpack_numpy.decode
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.SUB)
        self.sock.setsockopt_string(zmq.SUBSCRIBE, '')
        self.sock.setsockopt(zmq.CONFLATE, True)      # keep only the newest message, like the robot's own client
        self.sock.setsockopt(zmq.RCVHWM, 3)
        self.sock.connect(f'tcp://{host}:{port}')
        self.address = f'tcp://{host}:{port}'
        self.person_key, self.hand_key = person_key, hand_key
        self.lock = threading.Lock()
        self.latest = None
        self.count = 0
        self.keys_seen = set()
        self.stop = False
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while not self.stop:
            if not self.sock.poll(200):
                continue
            msg = self.msgpack.unpackb(self.sock.recv(), object_hook=self.decode_hook)
            received = time.monotonic(); wall = time.time()
            images, stamps = msg.get('images', {}), msg.get('timestamps', {})
            self.keys_seen |= set(images)
            if self.person_key not in images or self.hand_key not in images:
                continue
            p, h = _decode(images[self.person_key]), _decode(images[self.hand_key])
            if p is None or h is None:
                continue
            with self.lock:
                self.count += 1
                self.latest = dict(person=p, hand=h, t_person=float(stamps.get(self.person_key, wall)),
                                   t_hand=float(stamps.get(self.hand_key, wall)), received=received, received_wall=wall,
                                   has_timestamps=self.person_key in stamps and self.hand_key in stamps, frame_index=self.count)

    def get(self):
        with self.lock:
            return self.latest

    def close(self):
        self.stop = True
        self.thread.join(1)
        self.sock.close(0)
        self.ctx.term()


class _CvGrabber(threading.Thread):
    """Opens the source inside its own thread (network streams block on open) and keeps the newest frame."""

    def __init__(self, source):
        super().__init__(daemon=True)
        self.source = int(source) if str(source).isdigit() else str(source)
        self.lock = threading.Lock(); self.frame = None; self.t = None; self.n = 0; self.stop = False; self.error = None
        self.cap = None

    def run(self):
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            self.error = f'cannot open {self.source!r}'
            return
        while not self.stop:
            ok, f = self.cap.read()
            if not ok:
                time.sleep(.01); continue
            with self.lock:
                self.frame, self.t, self.n = f, time.time(), self.n + 1
        self.cap.release()


class CvPairSource:
    def __init__(self, person_source, hand_source):
        self.g = [_CvGrabber(person_source), _CvGrabber(hand_source)]
        for g in self.g:
            g.start()
        self.address = f'person={person_source} hand={hand_source}'
        self.count = 0

    def get(self):
        for g in self.g:
            if g.error:
                raise RuntimeError(g.error)
        with self.g[0].lock, self.g[1].lock:
            if self.g[0].frame is None or self.g[1].frame is None:
                return None
            self.count = self.g[0].n
            return dict(person=self.g[0].frame, hand=self.g[1].frame, t_person=self.g[0].t, t_hand=self.g[1].t,
                        received=time.monotonic(), received_wall=time.time(), has_timestamps=True, frame_index=self.g[0].n)

    def close(self):
        for g in self.g:
            g.stop = True
