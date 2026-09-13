# HR-3 v3 mechanical immutability completion

v2's final post-seal import check successfully reached the parent-loading boundary but caused Python to create an unmanifested bytecode cache under the sealed bundle. v3 copies only v2 checksum-listed payload files and requires `PYTHONDONTWRITEBYTECODE=1` plus Python `-B` on every frozen-bundle command. This is an execution-environment fix only; all scientific values are unchanged.
