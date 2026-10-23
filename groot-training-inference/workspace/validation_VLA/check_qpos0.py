import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path("/workspace/validation_VLA/assets/g1_brainco.xml")
print("nq:", model.nq)
print("qpos0 nonzero entries:")
for i in range(model.nq):
    if abs(model.qpos0[i]) > 1e-9:
        # find which joint this qpos index belongs to
        jid = None
        for j in range(model.njnt):
            if model.jnt_qposadr[j] == i:
                jid = j
                break
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) if jid is not None else "?"
        print(f"  qpos[{i}] ({name}) = {model.qpos0[i]}")
