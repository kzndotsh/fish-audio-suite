# Install

How to install each part of fish-audio-suite, run the proxy as a service, and keep it updated. Once installed, see [INTEGRATIONS.md](INTEGRATIONS.md) for connecting it to apps, and [TROUBLESHOOTING.md](TROUBLESHOOTING.md) if something fails.

## What you need

- [ ] **A Fish Audio API key** (`FISH_API_KEY`) and a **voice id** (`FISH_VOICE_ID`), the id of a voice model on fish.audio.
- [ ] **Python 3.12 or newer** and **[uv](https://docs.astral.sh/uv/)**, unless you use Docker or Nix.
- [ ] **For `fish-voice`, the voice app:**
  - [ ] a microphone and speakers
  - [ ] the PortAudio C library (see [PortAudio](#portaudio))
  - [ ] an LLM: an OpenRouter key, another OpenAI-compatible server, or Ollama

## Which part to install

| You want | Install | Section |
| --- | --- | --- |
| To talk to an AI by voice | `fish-voice` | [The voice app](#the-voice-app) |
| Fish voices in an app that speaks OpenAI's audio API | The proxy | [The proxy](#the-proxy) |
| Fish's live speech in your own Python code | The voice library | [As a library](#as-a-library) |
| The text helpers in your own pipeline | The kit | [As a library](#as-a-library) |

> [!IMPORTANT]
> Nothing is published to PyPI until 1.0.0, so everything installs from the repository.

## The voice app

### From a clone (recommended)

```bash
git clone https://github.com/kzndotsh/fish-audio-suite
cd fish-audio-suite
uv sync --all-packages --extra cli
./packages/voice/dev.sh             # the first run writes .env and stops
$EDITOR .env                        # set FISH_API_KEY, FISH_VOICE_ID and an LLM
./packages/voice/dev.sh --smoke     # check Fish: writes one line to a WAV file
./packages/voice/dev.sh             # talk
```

`dev.sh` always reads `.env` from the repository root (set `FISH_VOICE_ENV_FILE` to use another file). On NixOS it also puts PortAudio on the library path. Useful flags:
- `--debug` or `--trace`: logs.
- `--prompt-file`: a character.
- `--playback`: an output other than the speakers.

### As a command

Install `fish-voice` on your PATH with uv. The voice package needs the kit, so add it with `--with`:

```bash
uv tool install \
  "fish-audio-suite-voice[cli] @ git+https://github.com/kzndotsh/fish-audio-suite#subdirectory=packages/voice" \
  --with "fish-audio-suite-kit @ git+https://github.com/kzndotsh/fish-audio-suite#subdirectory=packages/kit"

fish-voice --env-file ~/.config/fish-voice.env --smoke
fish-voice --env-file ~/.config/fish-voice.env
```

Without `--env-file`, `fish-voice` reads `.env` in the current directory if there is one, then the environment. Copy [`.env.example`](../.env.example) as a starting point: it lists every setting.

### With Nix

```bash
nix run github:kzndotsh/fish-audio-suite#fish-audio-suite-voice -- --env-file ./.env
```

The Nix build bundles PortAudio, so nothing else is needed.

### PortAudio

The microphone and speakers need the PortAudio C library, which `uv` and `pip` do not install:

```bash
sudo apt install libportaudio2   # Debian, Ubuntu
sudo dnf install portaudio       # Fedora
brew install portaudio           # macOS
```

On NixOS, use `dev.sh` or `nix run` as above. Both bring PortAudio with them.

### Optional parts

The `cli` extra pulls in everything `fish-voice` uses. To install only some of it:

| Extra | Adds | Without it |
| --- | --- | --- |
| `speakers` | `sounddevice` (PortAudio playback and mic) | Only file, stdout and mpv output |
| `vad` | `webrtcvad-wheels` | No voice activity detection |
| `aec` | `pywebrtc-audio` (WebRTC AEC3) | No echo cancellation, so barge-in waits out the speaker bleed |
| `cli` | All three, plus the `openrouter` SDK | |

## The proxy

### Run it directly

From a clone:

```bash
uv sync --package fish-audio-suite-proxy
FISH_API_KEY=... uv run --package fish-audio-suite-proxy fish-audio-suite-proxy
curl -s http://127.0.0.1:8849/health
```

Or as a command:

```bash
uv tool install \
  "fish-audio-suite-proxy @ git+https://github.com/kzndotsh/fish-audio-suite#subdirectory=packages/proxy" \
  --with "fish-audio-suite-kit @ git+https://github.com/kzndotsh/fish-audio-suite#subdirectory=packages/kit"
FISH_API_KEY=... fish-audio-suite-proxy
```

Or with Nix:

```bash
FISH_API_KEY=... nix run github:kzndotsh/fish-audio-suite
```

It listens on `127.0.0.1:8849`.

> [!CAUTION]
> Without `FISH_PROXY_API_KEYS` the proxy accepts any client and spends your Fish credits for it. That is why it stays on loopback by default. Set the keys before you let other machines reach it.

All settings are in the [proxy README](../packages/proxy/README.md#settings).

### Docker

Build from the repository root:

```bash
docker build -t fish-audio-suite-proxy:latest .
docker run -d --name fish-proxy -p 127.0.0.1:8849:8849 \
  --env-file /path/to/fish.env --stop-timeout 130 fish-audio-suite-proxy:latest
```

- **User and health:** the image runs as a non-root user and has a health check on `/health`.
- **Network:** inside the container it listens on every interface, so publish the port on `127.0.0.1` unless `FISH_PROXY_API_KEYS` is set.
- **Stop timeout:** `--stop-timeout 130` gives in-flight replies their 120 s to finish. Docker's own default of 10 s cuts them off.

### Docker Compose

```yaml
services:
  fish-proxy:
    build: .                      # the repository root
    ports:
      - "127.0.0.1:8849:8849"
    env_file: fish.env            # FISH_API_KEY, optionally FISH_PROXY_API_KEYS
    stop_grace_period: 130s       # lets in-flight speech finish
    restart: unless-stopped
```

To reach the proxy from another container on the same Compose network, use `http://fish-proxy:8849/v1`.

### NixOS

Add the flake and enable the module:

```nix
{
  inputs.fish-audio-suite.url = "github:kzndotsh/fish-audio-suite";

  outputs = { nixpkgs, fish-audio-suite, ... }: {
    nixosConfigurations.myhost = nixpkgs.lib.nixosSystem {
      modules = [
        fish-audio-suite.nixosModules.default
        {
          services.fish-audio-suite-proxy = {
            enable = true;
            environmentFiles = [ "/run/secrets/fish.env" ];   # FISH_API_KEY lives here
          };
        }
      ];
    };
  };
}
```

This runs the proxy as a hardened systemd service on `127.0.0.1:8849`.

> [!WARNING]
> Keep the key in `environmentFiles` (a sops-nix or agenix secret, or a root-only file), never in `environment`, because that ends up in the world-readable Nix store.

| Option | Default | Meaning |
| --- | --- | --- |
| `enable` | `false` | Turn the service on |
| `backend` | `"native"` | `"native"` (systemd) or `"oci"` (a container) |
| `package` | this flake's proxy | The proxy package for `native` |
| `image` | none | The container image for `oci` (required there) |
| `host` | `"127.0.0.1"` | Listen address (`native` only; the image sets its own) |
| `port` | `8849` | Listen port |
| `openFirewall` | `false` | Open `port` in the firewall |
| `autoStart` | `true` | Start at boot |
| `gracefulShutdownSeconds` | `120` | How long in-flight replies get to finish on stop |
| `environment` | `{}` | Extra non-secret settings, such as `FISH_LATENCY` |
| `environmentFiles` | `[]` | Files with secrets and other settings |

Check it with `systemctl status fish-audio-suite-proxy` and `journalctl -u fish-audio-suite-proxy -f`. Every option has its own description in [`nix/module.nix`](../nix/module.nix).

### Behind a reverse proxy

For TLS, or to reach the proxy from other machines:

- **Client keys:** set `FISH_PROXY_API_KEYS`, so only your clients spend your Fish credits.
- **Keep-alive:** set `FISH_PROXY_KEEP_ALIVE` above the reverse proxy's idle timeout, for example `65` for a 60-second balancer. Otherwise you'll see occasional 502 errors.
- **Streaming:** turn off response buffering for `/v1/audio/speech` (nginx: `proxy_buffering off;`), so audio streams as it arrives.
- **Long recordings:** raise `FISH_PROXY_MAX_BODY_BYTES`, and the reverse proxy's own body limit, if you transcribe them.
- **Browser apps:** add CORS headers only if a browser app calls the proxy directly (see [INTEGRATIONS.md](INTEGRATIONS.md#apps-that-run-in-the-browser)).

A minimal Caddy site:

```caddy
fish.example.com {
    reverse_proxy 127.0.0.1:8849 {
        flush_interval -1
    }
}
```

## As a library

Add the packages to your project from the repository. The voice package needs the kit, so add both:

```bash
# Fish's live speech (FishSpeaker), with local playback
uv add "fish-audio-suite-kit @ git+https://github.com/kzndotsh/fish-audio-suite#subdirectory=packages/kit" \
       "fish-audio-suite-voice[speakers] @ git+https://github.com/kzndotsh/fish-audio-suite#subdirectory=packages/voice"

# Only the text helpers (no dependencies)
uv add "fish-audio-suite-kit @ git+https://github.com/kzndotsh/fish-audio-suite#subdirectory=packages/kit"
```

Leave out `[speakers]` if you only write audio to files or your own sink. With pip, use the same `name @ git+...` lines. Examples are in [INTEGRATIONS.md](INTEGRATIONS.md#python-the-voice-library).

## Self-hosted Fish

To use your own [fish-speech](https://github.com/fishaudio/fish-speech) server instead of Fish's cloud, set:

```bash
FISH_BASE=http://127.0.0.1:8080
```

This works for both the proxy and `fish-voice`. A self-hosted server allows `chunk_length` up to 1000; the cloud allows 300.

## Updating

| Installed with | Update with |
| --- | --- |
| A clone | `git pull && uv sync --all-packages --extra cli` |
| `uv tool install` | Run the same `uv tool install ...` command again with `--reinstall` |
| `uv add` (library) | `uv lock --upgrade-package fish-audio-suite-voice --upgrade-package fish-audio-suite-kit && uv sync` |
| Docker | `git pull`, rebuild the image, recreate the container |
| NixOS | `nix flake update fish-audio-suite`, then rebuild |

> [!WARNING]
> Until 1.0.0, any update may rename settings or Python names, without the old names kept working. Check the commit log before updating something you depend on.

## Uninstalling

- **`uv tool`:** `uv tool uninstall fish-audio-suite-voice` (or `fish-audio-suite-proxy`).
- **Docker:** `docker rm -f fish-proxy && docker rmi fish-audio-suite-proxy:latest`.
- **NixOS:** set `enable = false`, or remove the module, and rebuild.
- **A clone:** delete the directory. Nothing is written outside it except your `.env`.
