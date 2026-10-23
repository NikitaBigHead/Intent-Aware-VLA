# Teleoperation, whole-body control, and VLA integration

Project directory: [`../Teleop-Inference-WBC/`](../Teleop-Inference-WBC/)

The compact snapshot contains the Python/C++ source, configuration files,
install scripts, tests, and text documentation. Git history, local virtual
environments, generated output, documentation videos, and large simulation
robot assets are omitted. Model checkpoints are intentionally downloaded at
setup time by the upstream project.

Start with these project documents:

- [Main project README](../Teleop-Inference-WBC/README.md)
- [Local robot-operation notes](../Teleop-Inference-WBC/README_SIMPLE.md)
- [Quick start](../Teleop-Inference-WBC/docs/source/getting_started/quickstart.md)
- [Deployment installation](../Teleop-Inference-WBC/docs/source/getting_started/installation_deploy.md)
- [Model downloads](../Teleop-Inference-WBC/docs/source/getting_started/download_models.md)
- [VLA inference](../Teleop-Inference-WBC/docs/source/tutorials/vla_inference.md)
- [Troubleshooting](../Teleop-Inference-WBC/docs/source/user_guide/troubleshooting.md)

The snapshot is based on local commit `cfddb76` and includes the local
affective-intent/VLA working-tree changes. To reconstruct the large upstream
assets, obtain the corresponding upstream GR00T Whole-Body Control repository,
check out the documented base revision, and overlay this compact snapshot so
the submission-specific changes win.

Do not deploy directly to hardware before completing the simulator and camera
checks described by the project. Keep a safety operator at the emergency stop.
