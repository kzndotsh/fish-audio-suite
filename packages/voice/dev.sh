#!/usr/bin/env bash
# Local duplex runner. Copies .env.example once, then execs fish-voice with --env-file .env.
# NixOS: sounddevice needs PortAudio (and Pulse) on LD_LIBRARY_PATH (DYLD_FALLBACK_LIBRARY_PATH on macOS), same wrap as flake.nix.
# Other hosts use system PortAudio. Set FISH_VOICE_NIX=1 to build it with nix anyway.
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"
example="$root/.env.example"
# FISH_VOICE_ENV_FILE names the env file.
env_file="${FISH_VOICE_ENV_FILE:-}"
local_env="${env_file:-$root/.env}"
if [[ ! -f "$local_env" ]]; then
  cp "$example" "$local_env"
  echo "wrote $local_env — set FISH_API_KEY, FISH_VOICE_ID, and for duplex FISH_LLM_API_KEY + FISH_LLM_MODEL" >&2
  exit 2
fi

want_nix=0
if [[ -e /etc/NIXOS || "${FISH_VOICE_NIX:-0}" == "1" ]]; then want_nix=1; fi
if [[ -z "${FISH_VOICE_PORTAUDIO_LIB:-}" && "$want_nix" == "1" ]] && command -v nix >/dev/null 2>&1; then
  # --inputs-from resolves nixpkgs from this flake's lock, so the build is pinned.
  pa="$(nix build --inputs-from "$root" --no-link --print-out-paths nixpkgs#portaudio 2>/dev/null || true)"
  pulse="$(nix build --inputs-from "$root" --no-link --print-out-paths nixpkgs#libpulseaudio 2>/dev/null || true)"
  libs=""
  [[ -n "$pa" ]] && libs="$pa/lib"
  [[ -n "$pulse" ]] && libs="${libs:+$libs:}$pulse/lib"
  FISH_VOICE_PORTAUDIO_LIB="$libs"
fi
if [[ -n "${FISH_VOICE_PORTAUDIO_LIB:-}" ]]; then
  # Same split as flake.nix: the loader variable differs on macOS.
  if [[ "$(uname -s)" == "Darwin" ]]; then
    export DYLD_FALLBACK_LIBRARY_PATH="${FISH_VOICE_PORTAUDIO_LIB}${DYLD_FALLBACK_LIBRARY_PATH:+:$DYLD_FALLBACK_LIBRARY_PATH}"
  else
    export LD_LIBRARY_PATH="${FISH_VOICE_PORTAUDIO_LIB}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi
fi

exec uv run --package fish-audio-suite-voice --extra cli fish-voice --env-file "$local_env" "$@"
