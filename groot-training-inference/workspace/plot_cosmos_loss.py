import re
import glob

log_files = sorted(glob.glob("/workspace/logs/train_cosmos_*.log"))
pattern = re.compile(r"(\d+)/10000[^{]*?\{'loss':\s*([\d.]+)")

steps = []
losses = []
run_boundaries = []
for f in log_files:
    with open(f, "r", errors="ignore") as fh:
        text = fh.read()
    matches = pattern.findall(text)
    if matches:
        run_boundaries.append((f, len(steps)))
    for step_str, loss_str in matches:
        steps.append(int(step_str))
        losses.append(float(loss_str))

print(f"Total points: {len(steps)}")
for f, idx in run_boundaries:
    print(f"  run start idx={idx}: {f}")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, losses, linewidth=0.8, color="#2563eb")
    ax.set_xlabel("global_step")
    ax.set_ylabel("loss")
    ax.set_title("cosmos_right13 training loss")
    ax.grid(True, alpha=0.3)
    for f, idx in run_boundaries[1:]:
        ax.axvline(x=steps[idx], color="red", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig("/workspace/loss_plot.png", dpi=150)
    print("Saved /workspace/loss_plot.png")
except ImportError as e:
    print(f"matplotlib not available: {e}")
    with open("/workspace/loss_data.csv", "w") as out:
        out.write("step,loss\n")
        for s, l in zip(steps, losses):
            out.write(f"{s},{l}\n")
    print("Saved /workspace/loss_data.csv instead")
