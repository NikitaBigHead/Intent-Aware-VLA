import json, sys

for d in sys.argv[1:]:
    with open(f"{d}/trainer_state.json") as f:
        st = json.load(f)
    step = st.get("global_step")
    hist = [h["loss"] for h in st.get("log_history", []) if "loss" in h and h.get("step", 0) <= step]
    tail = hist[-5:]
    print(d, "step:", step, "smoothed(last5):", sum(tail) / len(tail) if tail else "nan")
