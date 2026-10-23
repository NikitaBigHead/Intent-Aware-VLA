# Anonymization notes

This package is prepared for anonymous review.

- Contributor-specific names, usernames, hostnames, IP addresses, SSH aliases,
  network-interface identifiers, and absolute workstation paths were replaced
  with neutral placeholders.
- Human-participant labels use stable pseudonyms of the form `subject_XX`.
- Sex, exact age, height, and weight were removed from the included sample
  metadata, as were identity-bearing session names.
- Faces in the included sample videos are blurred in external views or excluded
  by the top crop in close camera views. The corresponding audit is stored in
  `sample_dataset/meta/face_blur_report.jsonl`.
- Russian-language project notes and generated reports were removed or replaced
  with concise English instructions.

Names that remain in upstream licenses, copyright notices, citations, bundled
third-party code, and model vocabulary files are required provenance rather
than submission-author identification and must not be removed.

Hostnames such as `robot.local`, `workstation.local`, `gpu-server.local`, and
`gpu-server-a`/`gpu-server-b` are placeholders. Replace them, along with generic
paths under `/workspace` and `/data`, before running on another system.
