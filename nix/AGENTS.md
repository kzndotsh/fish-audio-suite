# `nix/`

> Inherits root [AGENTS.md](../AGENTS.md)

| File | Owns |
| --- | --- |
| [`module.nix`](module.nix) | `nixosModules.default`. Option `services.fish-audio-suite-proxy.enable`. OCI container `fish-audio-suite-proxy`, image default `fish-audio-suite-proxy:latest`, port `127.0.0.1:8849:8849`, `autoStart = false`. `environmentFiles` supplies `FISH_API_KEY` |

Flake outputs live in [`../flake.nix`](../flake.nix), not here. Packages: `fish-audio-suite-kit`, `fish-audio-suite-proxy` (also `default`), `fish-audio-suite-voice`. `apps.default` runs the proxy binary. `apps.fish-audio-suite-voice` runs `fish-voice`. The voice wrapper prefixes PortAudio and libpulseaudio on `LD_LIBRARY_PATH`.
