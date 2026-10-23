"""Register the custom modality config, then serve a Cosmos3 checkpoint.

Repo copy of the wrapper that is live on GPU-02 at
/root/tactile_ws/run_server_cosmos.py (container path: /workspace/run_server_cosmos.py).
Kept here for version control -- the on-server file was untracked. If you
change CKPT below, redeploy with:

    scp scripts/robot_deploy/cosmos_right13/run_server_cosmos.py \
        root@gpu-server-a:/root/tactile_ws/run_server_cosmos.py -P 2222

`NEW_EMBODIMENT` does not exist until gr00t_cosmos_right13_config.py has been
imported once (it calls register_modality_config() at module scope) -- run
directly with `python run_gr00t_server.py`, you get `KeyError: 'new_embodiment'`.

Port is 5555 (changed from 5566 on 2026-09-29 -- this replaced the user's
own left_run1/checkpoint-55000 stack that was running on 5555; that stack
was stopped for this). If 5555 is ever needed for something else again,
switch --port back to 5566 here and update the runbook / robot's
--policy-port to match.

Run inside the tactile-train container, in its own tmux session so it
survives an SSH disconnect:

    tmux new -s serve
    docker exec -it tactile-train bash -lc \
      "source /opt/Isaac-GR00T/.venv/bin/activate && cd /opt/Isaac-GR00T && \
       python /workspace/run_server_cosmos.py"
    # Ctrl+B, then D to detach

Then on the GPU-02 host, in a second tmux session, keep the port forwarded
out of the container (container IP may differ -- check `docker inspect
tactile-train | grep IPAddress`):

    tmux new -s socat
    socat TCP-LISTEN:5555,fork,reuseaddr TCP:172.17.0.2:5555
"""
import runpy
import sys

sys.path.insert(0, "/workspace")
import gr00t_cosmos_right13_config  # noqa: F401 -- registers NEW_EMBODIMENT

CKPT = "/workspace/checkpoints/cosmos_right13"  # flat layout, no checkpoint-N/ subdir (see run history)

# Port 5555 -- this replaced the user's own left_run1 stack on this port
# The previous stack on this port was stopped before this service was started.
# Was 5566 originally (to avoid colleagues' 5555 server) -- if that
# collision is back, switch this back to 5566 and update the runbook/robot
# --policy-port accordingly.
sys.argv = [
    "run_gr00t_server.py",
    "--model-path", CKPT,
    "--embodiment-tag", "NEW_EMBODIMENT",
    "--port", "5555",
]
runpy.run_path(
    "/opt/Isaac-GR00T/gr00t/eval/run_gr00t_server.py",
    run_name="__main__",
)
