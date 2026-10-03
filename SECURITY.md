# Security policy

fish-audio-suite is an unofficial toolkit for Fish Audio. The proxy holds a Fish
API key and spends its credits, so reports about it are taken seriously.

## Supported versions

| Version | Supported |
| --- | --- |
| 0.1.x | Yes. Security fixes land here. |
| Older than 0.1 | No |

While the packages are 0.x, a minor release may change behavior. A security fix
goes into the newest 0.x release first.

## Report a vulnerability

Please do not open a public issue for a security problem.

Report it privately through GitHub: open the repository's **Security** tab and
choose **Report a vulnerability**. Include the package and version, what you
sent or ran, what you expected, and what happened. A short proof of concept
helps. If private reporting is not available on the repository, open an
issue that asks for a private channel and contains no details.

This is a small project maintained by one person. I will acknowledge a report
as soon as I can, usually within a week, and will tell you whether I consider it
in scope. I will credit you in the fix unless you prefer not to be named.

## What is in scope

- **The proxy** (`fish-audio-suite-proxy`):
  - Any way to use the proxy without a valid client key when `FISH_PROXY_API_KEYS`
    is set, including header and bearer parsing edge cases.
  - Any way to read the Fish key, the client keys or the upstream address from a
    response, a log line, a traceback or a `repr`.
  - Request smuggling or header injection toward Fish, or any way for a client
    to steer where the proxy connects.
  - Body, upload, field or concurrency abuse that bypasses the configured limits
    or exhausts memory or file descriptors.
- **The kit** (`fish-audio-suite-kit`): text scrubbing, cutting or parsing that
  an attacker can make take super-linear time, or that crashes on crafted input.
- **The voice client** (`fish-audio-suite-voice`): leaking a key to the terminal
  or a log, or sending a key to a host you did not configure.
- **The release pipeline**: anything that lets code other than a reviewed `v*`
  tag publish to PyPI or alters what a tag publishes.

## What is out of scope

- Running the proxy on a public address without `FISH_PROXY_API_KEYS`. The
  README says this lets anyone spend your credits, and the proxy warns at startup.
- Vulnerabilities in Fish Audio's own service, or in a dependency with no path
  through this code. Report those upstream. If a dependency fix needs a version
  bump here, tell me.
- Problems that need local access to the machine that runs the voice client,
  such as reading its environment.
- Denial of service by volume alone against a proxy that has no rate limiter in
  front of it.

## Hardening you control

Keep the default `127.0.0.1` bind or set `FISH_PROXY_API_KEYS`, publish the
container on loopback (`-p 127.0.0.1:8849:8849`), use an `https` `FISH_BASE`,
and keep the Fish key in an environment file, not on a command line.
