#!/usr/bin/env bash
# Reap podman healthcheck timers whose container no longer exists.
#
# THE FAILURE MODE, measured 2026-08-14 on this machine. A container with a
# compose healthcheck gets a transient `systemd --user` timer inside the podman
# VM. If the container goes away without podman running its own cleanup, THE
# TIMER SURVIVES — and keeps firing `podman healthcheck run <gone-id>` on the
# compose interval (3s for this stack), each run spawning a podman process that
# fails. Two of them were resident with NOTHING running but one unrelated
# container, and the VM's load average read:
#
#     load average: 5.32, 8.78, 7.52        (4 vCPU)
#
# `podman ps -a` showed only that unrelated container; `podman inspect` on both
# timer ids returned "no such object"; `systemctl --user list-units` showed both
# timers `active/waiting` with their services `failed`.
#
# WHY THIS MATTERS BEYOND WASTED CPU. The live gate's timing bounds are
# load-sensitive: measured time-to-healthy was 29.64s on a settled VM (load1
# 0.86) versus 135.71s at load1 4.4, against a 120s bound. So orphaned timers
# RATCHET the floor — every run that leaves one behind makes the next run
# slower, and the suite reads that as nondeterministic flakiness. The baseline
# is moving, which is precisely what makes it hard to see.
#
# Idempotent, and never fails its caller: this is cleanup, and cleanup that can
# fail the run it is cleaning up after is worse than no cleanup at all.
# Each unit's container is checked INDIVIDUALLY before its timer is touched, so
# a live container's healthcheck is never disturbed.
set -u

say() { printf '[reap-healthchecks] %s\n' "$*"; }

# macOS/Windows run podman in a VM, so the units live inside it. On a native
# Linux podman there is no `podman machine`; rather than guess at an equivalent,
# this reports and exits clean — the live gate is documented as an operator-local
# macOS workflow, and a wrong guess here would touch the host's systemd.
if ! podman machine inspect >/dev/null 2>&1; then
  say "no podman machine (native podman?) — nothing to reap, skipping"
  exit 0
fi

UNITS="$(podman machine ssh \
  "systemctl --user list-units --all --plain --no-legend 2>/dev/null | awk '/healthcheck/ {print \$1}'" \
  2>/dev/null | tr -d '\r')"

if [ -z "${UNITS}" ]; then
  say "no healthcheck units present"
  exit 0
fi

REAPED=0
KEPT=0
for unit in ${UNITS}; do
  # Unit names are "<container-id>-<suffix>.timer|.service"; the container id is
  # everything before the first hyphen.
  cid="${unit%%-*}"
  if podman inspect "${cid}" >/dev/null 2>&1; then
    KEPT=$((KEPT + 1))
    continue
  fi
  if podman machine ssh \
    "systemctl --user stop '${unit}' 2>/dev/null; systemctl --user reset-failed '${unit}' 2>/dev/null" \
    >/dev/null 2>&1; then
    REAPED=$((REAPED + 1))
    say "reaped ${unit%%.*}... (container ${cid:0:12} is gone)"
  fi
done

if [ "${REAPED}" -gt 0 ]; then
  say "reaped ${REAPED} orphaned unit(s); left ${KEPT} belonging to live containers"
else
  say "nothing orphaned (${KEPT} unit(s) belong to live containers)"
fi
exit 0
