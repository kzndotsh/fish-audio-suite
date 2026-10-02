# `nix/`

> Inherits root [AGENTS.md](../AGENTS.md)

| File | Owns |
| --- | --- |
| [`module.nix`](module.nix) | `nixosModules.default`, called as `import ./module.nix self`. `services.fish-audio-suite-proxy`: `enable`, `backend` (`native` systemd service, default; or `oci`), `package`, `image` (required for `oci`), `host` (`127.0.0.1`), `port` (`8849`), `openFirewall`, `autoStart` (true), `environment`, `environmentFiles` |

The module sets `FISH_PROXY_PORT` from `port`; `FISH_PROXY_HOST` is set for `native` only (the container image sets it itself). `FISH_API_KEY` goes in `environmentFiles`, never `environment`.

Flake outputs live in [`../flake.nix`](../flake.nix). Packages: `fish-audio-suite-kit`, `fish-audio-suite-proxy` (also `default`), `fish-audio-suite-voice`. `checks` builds all three. `apps.default` runs the proxy; `apps.fish-audio-suite-voice` runs `fish-voice`. The voice wrapper puts PortAudio on `LD_LIBRARY_PATH` (plus libpulseaudio on Linux) or `DYLD_FALLBACK_LIBRARY_PATH` (macOS, untested). Its version is read from `packages/voice/pyproject.toml`.
