#!/usr/bin/env bash
# Local duplex runner. Copies .env.example once, then execs fish-voice with --env-file .env.
# NixOS: sounddevice needs PortAudio (and Pulse) on LD_LIBRARY_PATH — same wrap as flake.nix.
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$root"
example="$root/.env.example"
local_env="${FISH_VOICE_ENV:-$root/.env}"
if [[ ! -f "$local_env" ]]; then
  cp "$example" "$local_env"
  echo "wrote $local_env — set FISH_API_KEY, FISH_VOICE_ID, and for duplex OPENROUTER_API_KEY + FISH_LLM_MODEL" >&2
  exit 2
fi

if [[ -z "${FISH_VOICE_PORTAUDIO_LIB:-}" ]] && command -v nix >/dev/null 2>&1; then
  pa="$(nix build --no-link --print-out-paths nixpkgs#portaudio 2>/dev/null || true)"
  pulse="$(nix build --no-link --print-out-paths nixpkgs#libpulseaudio 2>/dev/null || true)"
  libs=""
  [[ -n "$pa" ]] && libs="$pa/lib"
  [[ -n "$pulse" ]] && libs="${libs:+$libs:}$pulse/lib"
  FISH_VOICE_PORTAUDIO_LIB="$libs"
fi
if [[ -n "${FISH_VOICE_PORTAUDIO_LIB:-}" ]]; then
  export LD_LIBRARY_PATH="${FISH_VOICE_PORTAUDIO_LIB}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

exec uv run --package fish-audio-suite-voice --extra cli fish-voice --env-file "$local_env" "$@"
