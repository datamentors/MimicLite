"""Record raw PICO body tracking (XRoboToolkit, 24 joints, Unity frame x y z qx qy qz qw) to an npz, e.g. a
T-pose for tools/make_xrobot_to_a2.py --tpose. Works next to a running pico_retarget_pub (second SDK client).

  sim2real/venv/teleop/.venv/bin/python <a2>/teleop/tools/record_pico.py 15 tpose.npz
"""
import os
import sys
import time

import numpy as np
import xrobotoolkit_sdk as xrt

if len(sys.argv) != 3:
    sys.exit(__doc__)
duration, out = float(sys.argv[1]), sys.argv[2]
xrt.init()
poses, stamps = [], []
t0 = last_print = time.time()
while time.time() - t0 < duration:
    if xrt.is_body_data_available():
        poses.append(np.asarray(xrt.get_body_joints_pose(), dtype=np.float32))
        stamps.append(time.time() - t0)
    if time.time() - last_print > 5:
        print(f"t={time.time() - t0:5.1f}s frames={len(poses)}", flush=True)
        last_print = time.time()
    time.sleep(1 / 60)
if not poses:
    print("no body data received (PICO connected, whole-body tracking on?)", flush=True)
    os._exit(1)
np.savez(out, poses=np.asarray(poses), t=np.asarray(stamps))
print(f"saved {out}: {len(poses)} frames", flush=True)
os._exit(0)  # skip the SDK teardown, it aborts on interpreter exit
