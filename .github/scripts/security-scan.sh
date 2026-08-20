#!/usr/bin/env bash
# Scan a uv.lock for known advisories.
#
# Two behaviours the plain `uvx uv-secure uv.lock` call did not have:
#
# 1. Retry. uv-secure fetches its advisory database over the network, and a
#    transient failure there (observed: "Error: fsspec raised exception") failed
#    the job. This gate is what `dependabot-auto-merge` waits on, so a blip
#    silently strands dependabot PRs -- the same end state as the advisory
#    deadlock it exists to catch.
#
# 2. Baseline comparison. The scan covers the WHOLE lockfile, so while any
#    advisory is open every single-package security PR fails on the other
#    packages' advisories, and none can merge. Dependabot's `security` group
#    only batches advisories surfaced in the same run, so alerts arriving on
#    different days still deadlock (aiohttp #38 and cryptography #39 were 7
#    hours apart across a day boundary). Comparing against the base branch lets
#    each PR merge on its own merit -- it must not make things worse -- while
#    still blocking anything that introduces a new advisory.
#
# Usage: security-scan.sh <lockfile> [base-lockfile]
#   With a base lockfile, fails only on advisories the head introduces.
#   Without one, fails on any advisory at all (used for main and releases).
set -uo pipefail

LOCKFILE="${1:-uv.lock}"
BASE_LOCKFILE="${2:-}"
ATTEMPTS=3

# uv-secure exit codes: 0 = clean, 2 = advisories found. Both are answers.
# Anything else means the tool itself failed and the output cannot be trusted --
# critically, a failed run prints no advisory IDs, so treating it as a result
# would read as "clean" and wave a vulnerable lockfile straight through.
scan() {
  local target="$1" out rc attempt backoff
  for attempt in $(seq 1 "$ATTEMPTS"); do
    out=$(uvx uv-secure "$target" 2>&1)
    rc=$?
    if [ "$rc" = 0 ] || [ "$rc" = 2 ]; then
      printf '%s' "$out"
      return "$rc"
    fi
    backoff=$((attempt * 5))
    {
      echo "uv-secure failed on '$target' (exit $rc), attempt $attempt/$ATTEMPTS."
      printf '%s\n' "$out"
      [ "$attempt" -lt "$ATTEMPTS" ] && echo "Retrying in ${backoff}s."
    } >&2
    [ "$attempt" -lt "$ATTEMPTS" ] && sleep "$backoff"
  done
  return 90
}

ids() {
  printf '%s' "$1" \
    | grep -oE '(GHSA-[a-z0-9-]+|PYSEC-[0-9]{4}-[0-9]+|CVE-[0-9]{4}-[0-9]+)' \
    | sort -u
}

head_out=$(scan "$LOCKFILE")
head_rc=$?
if [ "$head_rc" = 90 ]; then
  echo "uv-secure could not complete after $ATTEMPTS attempts; failing rather than assuming clean." >&2
  exit 1
fi
head_ids=$(ids "$head_out")
printf '%s\n' "$head_out"

if [ -z "$BASE_LOCKFILE" ]; then
  if [ -n "$head_ids" ]; then
    echo ""
    echo "FAIL: $(printf '%s\n' "$head_ids" | wc -l | tr -d ' ') advisory/advisories in $LOCKFILE."
    exit 1
  fi
  echo ""
  echo "OK: no advisories."
  exit 0
fi

base_out=$(scan "$BASE_LOCKFILE")
base_rc=$?
if [ "$base_rc" = 90 ]; then
  echo "Could not scan the base lockfile; falling back to the strict check." >&2
  if [ -n "$head_ids" ]; then
    echo "FAIL: advisories present and the base could not be compared." >&2
    exit 1
  fi
  exit 0
fi
base_ids=$(ids "$base_out")

introduced=$(comm -13 <(printf '%s\n' "$base_ids") <(printf '%s\n' "$head_ids") | sed '/^$/d')
fixed=$(comm -23 <(printf '%s\n' "$base_ids") <(printf '%s\n' "$head_ids") | sed '/^$/d')
carried=$(comm -12 <(printf '%s\n' "$base_ids") <(printf '%s\n' "$head_ids") | sed '/^$/d')

echo ""
echo "Compared against the base branch lockfile:"
[ -n "$fixed" ]     && echo "  fixed by this change:  $(printf '%s\n' "$fixed" | tr '\n' ' ')"
[ -n "$carried" ]   && echo "  pre-existing on base:  $(printf '%s\n' "$carried" | tr '\n' ' ')"
[ -n "$introduced" ] && echo "  INTRODUCED here:       $(printf '%s\n' "$introduced" | tr '\n' ' ')"

if [ -n "$introduced" ]; then
  echo ""
  echo "FAIL: this change introduces advisories not present on the base branch."
  exit 1
fi
if [ -n "$carried" ]; then
  echo ""
  echo "PASS: no new advisories. Note the pre-existing ones above are still open on"
  echo "the base branch and need their own fix -- they do not block this PR."
fi
echo ""
echo "OK: no advisories introduced."
exit 0
