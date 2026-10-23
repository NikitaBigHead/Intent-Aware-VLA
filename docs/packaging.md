# Validate and archive the package

Run these commands from the directory containing `submit/`.

## Basic checks

```bash
du -sh submit
find submit -type d \( -name .git -o -name __pycache__ -o -name '.venv*' \
  -o -name checkpoints -o -name datasets \)
find submit -type f -size +100M -print
sha256sum submit/hri_video_robot/checkpoint/model.pt
```

The first `find` should print nothing. The large-file check may report files in
the intentionally included sample dataset, but should not report duplicated
pose models or Teleop media.

## Create the submission archive

Use GNU `tar`, which preserves hardlinks. Override ownership so the archive
does not expose the local workstation username:

```bash
tar --sort=name --owner=0 --group=0 --numeric-owner \
  -czf short-paper-code.tar.gz submit/
```

Verify the archive before uploading:

```bash
tar -tzf short-paper-code.tar.gz >/dev/null
tar -tvzf short-paper-code.tar.gz \
  | awk '$2 != "0/0" { bad=1 } END { exit bad }'
du -h short-paper-code.tar.gz
```

Avoid ZIP for the canonical package: common ZIP tools store hardlinked paths as
independent file copies.
