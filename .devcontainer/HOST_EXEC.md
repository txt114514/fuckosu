# Host Command Bridge

The devcontainer installs a single explicit host command bridge:

```bash
host-exec <host-command> [args...]
host-exec -- <host-command> [args...]
```

It enters the host namespaces with `nsenter` and returns the command stdout,
stderr, and exit code to the container caller. This is useful for diagnostics
that must run on the host side, for example:

```bash
host-exec nvidia-smi
host-exec bash -lc 'id && uname -a'
host-exec bash -lc 'systemctl status docker --no-pager'
```

This bridge requires the existing devcontainer runtime settings:

- `privileged: true`
- `pid: host`
- `/dev` mounted into the container

The `dev` user is granted passwordless sudo for `/usr/local/bin/host-exec`
only. General passwordless sudo is not enabled.

Use this for environment diagnosis and host-side checks. Prefer normal
container commands for repository edits, Python tests, and training code.
