#!/usr/bin/env python3
"""Read one camera frame and robot joint state. Never publishes commands."""
import argparse
import time
from pathlib import Path

import numpy as np
from gear_sonic.camera.composed_camera import ComposedCameraClientSensor
from gear_sonic.utils.data_collection.zmq_state_subscriber import ZMQStateSubscriber
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorStates_


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, default=Path('/home/unitree/robohamer_shadow_input.npz'))
    ap.add_argument('--camera-host', default='localhost')
    ap.add_argument('--camera-port', type=int, default=5555)
    ap.add_argument('--state-port', type=int, default=5557)
    ap.add_argument('--dds-interface', default='eth0')
    ap.add_argument('--timeout', type=float, default=30.0)
    args = ap.parse_args()
    ChannelFactoryInitialize(0, args.dds_interface)
    left = ChannelSubscriber('rt/brainco/left/state', MotorStates_)
    right = ChannelSubscriber('rt/brainco/right/state', MotorStates_)
    left.Init()
    right.Init()
    camera = ComposedCameraClientSensor(server_ip=args.camera_host, port=args.camera_port)
    robot = ZMQStateSubscriber(host='localhost', port=args.state_port)
    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        cam, state, lmsg, rmsg = camera.read(), robot.get_msg(), left.Read(), right.Read()
        if cam is None or state is None or lmsg is None or rmsg is None:
            time.sleep(0.05)
            continue
        img = np.asarray(cam['images']['ego_view'])
        body = np.asarray(state['body_q'], dtype=np.float32)
        left_q = np.array([lmsg.states[i].q for i in range(6)], dtype=np.float32)
        right_q = np.array([rmsg.states[i].q for i in range(6)], dtype=np.float32)
        assert body.shape == (29,) and img.ndim == 3 and img.shape[2] == 3
        assert np.isfinite(body).all() and np.isfinite(left_q).all() and np.isfinite(right_q).all()
        assert np.all((left_q >= 0) & (left_q <= 1)) and np.all((right_q >= 0) & (right_q <= 1))
        np.savez_compressed(args.output, image=img, body_q=body, left_hand_norm=left_q,
                            right_hand_norm=right_q)
        print(f'SAVED {args.output} image={img.shape} body={body.shape} hands=6+6')
        return
    raise TimeoutError('No synchronized camera/body/BrainCo state; check three services')


if __name__ == '__main__':
    main()
