#!/usr/bin/env bash
# Launch the robot-side affective-VLA stack.

set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
CONTROL_ROOT="${CONTROL_ROOT:-/home/unitree/GR00T-WholeBodyControl}"
CAMERA_VENV="${CAMERA_VENV:-$CONTROL_ROOT/.venv_camera}"
DATA_VENV="${DATA_VENV:-$CONTROL_ROOT/.venv_data_collection}"
DEPLOY_DIR="${DEPLOY_DIR:-$CONTROL_ROOT/gear_sonic_deploy}"
SDK_PYTHONPATH="${SDK_PYTHONPATH:-/home/unitree/unitree_sdk2_python}"

SESSION="${TMUX_SESSION:-affective_vla}"
ROBOT_INTERFACE="${ROBOT_INTERFACE:-eth0}"
POLICY_HOST="${POLICY_HOST:-gpu-server.local}"
POLICY_PORT="${POLICY_PORT:-5555}"
INTENT_HOST="${INTENT_HOST:-intent-server.local}"
INTENT_PORT="${INTENT_PORT:-5562}"
VOICE_MODEL="${VOICE_MODEL:-/home/unitree/vosk-model-small-en-us-0.15}"
ENABLE_VOICE="${ENABLE_VOICE:-1}"
KEYPRESS_SCRIPT="${KEYPRESS_SCRIPT:-/home/unitree/keypress.py}"

INTENT_UNKNOWN_GRACE="${INTENT_UNKNOWN_GRACE:-1.5}"
INTENT_UNKNOWN_TO_NONE_SECONDS="${INTENT_UNKNOWN_TO_NONE_SECONDS:-2.0}"
INTENT_MAX_AGE="${INTENT_MAX_AGE:-1.5}"

REPLACE_SESSION=0
ATTACH_SESSION=0

usage() {
    cat <<EOF
Usage: $0 [--replace] [--attach]

Create robot-side tmux session '$SESSION' with windows:
  camera, brainco, deploy, keyboard, bridge, voice

Options:
  --replace  Kill and recreate the existing '$SESSION' tmux session.
  --attach   Attach after creating the session.
  -h, --help Show this help.

Useful environment overrides:
  TMUX_SESSION, REPO_DIR, CONTROL_ROOT, ROBOT_INTERFACE
  POLICY_HOST, POLICY_PORT, INTENT_HOST, INTENT_PORT
  INTENT_UNKNOWN_GRACE, INTENT_UNKNOWN_TO_NONE_SECONDS, INTENT_MAX_AGE
  VOICE_MODEL, VOICE_PYTHON, ENABLE_VOICE=0, KEYPRESS_SCRIPT
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --replace) REPLACE_SESSION=1 ;;
        --attach) ATTACH_SESSION=1 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

require_command() {
    command -v "$1" >/dev/null || {
        echo "Required command is missing: $1" >&2
        exit 1
    }
}

require_file() {
    [[ -f "$1" ]] || {
        echo "Required file is missing: $1" >&2
        exit 1
    }
}

require_dir() {
    [[ -d "$1" ]] || {
        echo "Required directory is missing: $1" >&2
        exit 1
    }
}

find_voice_python() {
    local candidate
    local -a candidates
    if [[ -n "${VOICE_PYTHON:-}" ]]; then
        candidates=("$VOICE_PYTHON")
    else
        candidates=(
            "/home/unitree/.venv_voice/bin/python"
            "$CONTROL_ROOT/.venv_voice/bin/python"
            "$DATA_VENV/bin/python"
            "python3"
        )
    fi

    for candidate in "${candidates[@]}"; do
        if [[ "$candidate" == */* && ! -x "$candidate" ]]; then
            continue
        fi
        if "$candidate" -c 'import vosk, zmq' >/dev/null 2>&1; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    return 1
}

require_command tmux
require_command pgrep
require_dir "$REPO_DIR"
require_dir "$DEPLOY_DIR"
require_file "$CAMERA_VENV/bin/activate"
require_file "$DATA_VENV/bin/activate"
require_file "$REPO_DIR/gear_sonic/scripts/start_brainco_and_check.sh"
require_file "$REPO_DIR/gear_sonic/scripts/run_inference_pose_predictor_affective_vla.py"
require_file "$REPO_DIR/gear_sonic/scripts/voice_safe_stop.py"
require_file "$KEYPRESS_SCRIPT"

if [[ "$ENABLE_VOICE" != "0" && "$ENABLE_VOICE" != "1" ]]; then
    echo "ENABLE_VOICE must be 0 or 1, got: $ENABLE_VOICE" >&2
    exit 2
fi

VOICE_PYTHON_RESOLVED=""
if [[ "$ENABLE_VOICE" == "1" ]]; then
    require_dir "$VOICE_MODEL"
    if ! VOICE_PYTHON_RESOLVED="$(find_voice_python)"; then
        echo "Could not find a Python with vosk and pyzmq." >&2
        echo "Set VOICE_PYTHON=/path/to/python or use ENABLE_VOICE=0." >&2
        exit 1
    fi
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
    if [[ $REPLACE_SESSION -eq 0 ]]; then
        echo "tmux session '$SESSION' already exists." >&2
        echo "Attach: tmux attach -t $SESSION" >&2
        echo "Replace: $0 --replace" >&2
        exit 1
    fi
    tmux kill-session -t "$SESSION"
    sleep 1
fi

# Never start a second controller/camera/bridge next to manually launched ones.
CONFLICT_PATTERN='gear_sonic[.]camera[.]composed_camera|run_inference_pose_predictor_affective_vla[.]py|g1_deploy_onnx_ref|keypress[.]py|voice_safe_stop[.]py'
if conflicts="$(pgrep -af "$CONFLICT_PATTERN" || true)" && [[ -n "$conflicts" ]]; then
    echo "Refusing to start duplicate robot processes:" >&2
    echo "$conflicts" >&2
    echo "Stop the old manual processes (and any stopped shell jobs), then run again." >&2
    exit 1
fi

send_command() {
    local window="$1"
    local command="$2"
    tmux send-keys -l -t "$SESSION:$window" "$command"
    tmux send-keys -t "$SESSION:$window" C-m
}

new_window() {
    local window="$1"
    tmux new-window -d -t "$SESSION" -n "$window"
}

camera_command="source $CAMERA_VENV/bin/activate && cd $REPO_DIR && python -m gear_sonic.camera.composed_camera"
camera_command+=" --ego-view-camera realsense --ego-view-device-id 243422071979"
camera_command+=" --head-camera usb --head-device-id /dev/video6"
camera_command+=" --realsense-width 1280 --realsense-height 720"
camera_command+=" --head-camera-width 1280 --head-camera-height 720"
camera_command+=" --head-camera-fps 15 --head-camera-quality 90 --head-camera-fourcc MJPG"
camera_command+=" --no-realsense-depth --fps 15 --port 5555"

brainco_command="cd $REPO_DIR && bash gear_sonic/scripts/start_brainco_and_check.sh --interface $ROBOT_INTERFACE"

deploy_command="export TensorRT_ROOT=\$HOME/TensorRT && cd $DEPLOY_DIR && source scripts/setup_env.sh"
deploy_command+=" && ./deploy.sh --input-type zmq_manager"
deploy_command+=" --arm-compliance --compliance-host localhost --compliance-port 5565"
deploy_command+=" --compliance-profile HUG --compliance-soften 0.3 --compliance-stiffen 1.0"
deploy_command+=" --safe-stop-profile SOFT --safe-stop-soften 0.5 --safe-stop-hand-speed 0.2"
deploy_command+=" --safe-stop-status-port 5571 --safe-stop-voice-port 5570 real"

keyboard_command="source $DATA_VENV/bin/activate && python3 $KEYPRESS_SCRIPT"

bridge_command="source $DATA_VENV/bin/activate && export PYTHONPATH=$SDK_PYTHONPATH:\${PYTHONPATH:-}"
bridge_command+=" && cd $REPO_DIR"
bridge_command+=" && SAFE_STOP_HOST=localhost SAFE_STOP_PORT=5571"
bridge_command+=" SAFE_STOP_HAND_OPEN_RATE=0.5 SAFE_STOP_HAND_RESUME_RATE=0.5"
bridge_command+=" SLEW_ENABLE=1 HAND_ENABLE=1"
bridge_command+=" python gear_sonic/scripts/run_inference_pose_predictor_affective_vla.py"
bridge_command+=" --host $POLICY_HOST --port $POLICY_PORT"
bridge_command+=" --camera-host localhost --camera-port 5555"
bridge_command+=" --state-zmq-host localhost --state-zmq-port 5557"
bridge_command+=" --action-zmq-host localhost --action-zmq-port 5556"
bridge_command+=" --embodiment-tag NEW_EMBODIMENT --prompt none"
bridge_command+=" --intent-mode auto --intent-host $INTENT_HOST --intent-port $INTENT_PORT"
bridge_command+=" --intent-unknown-grace $INTENT_UNKNOWN_GRACE"
bridge_command+=" --intent-unknown-to-none-seconds $INTENT_UNKNOWN_TO_NONE_SECONDS"
bridge_command+=" --intent-max-age $INTENT_MAX_AGE"
bridge_command+=" --intent-compliance --compliance-port 5565 --compliance-rate 10"

voice_command="cd $REPO_DIR && $VOICE_PYTHON_RESOLVED gear_sonic/scripts/voice_safe_stop.py"
voice_command+=" --g1-mic --model $VOICE_MODEL --port 5570 --verbose"

tmux new-session -d -s "$SESSION" -n camera
tmux set-option -t "$SESSION" -g mouse on
tmux set-option -t "$SESSION" base-index 0
tmux set-option -t "$SESSION" renumber-windows on
send_command camera "$camera_command"

new_window brainco
send_command brainco "$brainco_command"

new_window deploy
send_command deploy "$deploy_command"

new_window keyboard
send_command keyboard "$keyboard_command"

new_window bridge
send_command bridge "$bridge_command"

if [[ "$ENABLE_VOICE" == "1" ]]; then
    new_window voice
    send_command voice "$voice_command"
fi

tmux select-window -t "$SESSION:deploy"

cat <<EOF
Started robot-side tmux session: $SESSION

Windows:
  0 camera    robot camera server
  1 brainco   BrainCo container restart and DDS check
  2 deploy    WBC deploy (wait for "Init done" here)
  3 keyboard  VLA keyboard publisher
  4 bridge    affective VLA bridge
EOF
if [[ "$ENABLE_VOICE" == "1" ]]; then
    echo "  5 voice     microphone safe stop"
fi
cat <<EOF

External prerequisites are not started by this script:
  - GR00T policy server at $POLICY_HOST:$POLICY_PORT
  - intent predictor at $INTENT_HOST:$INTENT_PORT

Attach: tmux attach -t $SESSION
Switch windows: Ctrl+b, then window number
EOF

if [[ $ATTACH_SESSION -eq 1 ]]; then
    exec tmux attach -t "$SESSION"
fi
