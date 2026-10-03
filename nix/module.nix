# Called as `import ./module.nix self` from flake.nix so the default package
# is this flake's proxy build.
self:
{
  config,
  lib,
  pkgs,
  utils,
  ...
}:
let
  cfg = config.services.fish-audio-suite-proxy;
  portStr = toString cfg.port;
  # The generated port comes last. The firewall rule and the container port
  # mapping both follow `port`, so a different value would leave it unreachable.
  env = cfg.environment // {
    FISH_PROXY_PORT = portStr;
    FISH_PROXY_GRACEFUL_SHUTDOWN = toString cfg.gracefulShutdownSeconds;
  };
  lowPort = cfg.port < 1024;
  # Both the service manager and the container runtime must wait longer than
  # the proxy's own drain, or they kill it in the middle of a reply.
  stopSeconds = cfg.gracefulShutdownSeconds + 10;
in
{
  options.services.fish-audio-suite-proxy = {
    enable = lib.mkEnableOption "the fish-audio-suite OpenAI-compatible Fish proxy";

    backend = lib.mkOption {
      type = lib.types.enum [
        "native"
        "oci"
      ];
      default = "native";
      description = ''
        How to run the proxy. `native` is a hardened systemd service built from
        this flake. `oci` runs a container and needs `image`.
      '';
    };

    package = lib.mkOption {
      type = lib.types.package;
      default = self.packages.${pkgs.stdenv.hostPlatform.system}.fish-audio-suite-proxy;
      defaultText = lib.literalExpression "fish-audio-suite.packages.\${system}.fish-audio-suite-proxy";
      description = "Proxy package for the `native` backend.";
    };

    image = lib.mkOption {
      type = lib.types.nullOr lib.types.str;
      default = null;
      example = "ghcr.io/example/fish-audio-suite-proxy:0.1.0";
      description = "Container image for the `oci` backend. Build it from the repo `Dockerfile`.";
    };

    host = lib.mkOption {
      type = lib.types.str;
      default = "127.0.0.1";
      description = ''
        Address the proxy listens on (`native`) or the host address the
        container port is published on (`oci`).
      '';
    };

    port = lib.mkOption {
      type = lib.types.port;
      default = 8849;
      description = ''
        Proxy port. Sets `FISH_PROXY_PORT`, and the published port for `oci`.
        A port below 1024 gives the `native` service CAP_NET_BIND_SERVICE.
      '';
    };

    openFirewall = lib.mkOption {
      type = lib.types.bool;
      default = false;
      description = "Open `port` in the firewall. Only useful when `host` is not loopback.";
    };

    gracefulShutdownSeconds = lib.mkOption {
      type = lib.types.ints.unsigned;
      # Keep in step with DEFAULT_GRACEFUL_S in packages/proxy/.../settings.py.
      default = 120;
      description = ''
        How long the proxy lets in-flight replies finish after a stop signal.
        Sets `FISH_PROXY_GRACEFUL_SHUTDOWN`. The service (`TimeoutStopSec`) and the
        container (`--stop-timeout`) wait this long plus 10 seconds before they kill it.
      '';
    };

    autoStart = lib.mkOption {
      type = lib.types.bool;
      default = true;
      description = "Start at boot. When false, start it with `systemctl start`.";
    };

    environment = lib.mkOption {
      type = lib.types.attrsOf lib.types.str;
      default = { };
      example = {
        FISH_TTS_MODEL = "s2.1-pro";
      };
      description = ''
        Extra `FISH_*` variables. Do not put secrets here; they land in the Nix
        store. `FISH_PROXY_PORT`, `FISH_PROXY_HOST` and `FISH_PROXY_GRACEFUL_SHUTDOWN` are
        rejected: use `port`, `host` and `gracefulShutdownSeconds`.
      '';
    };

    environmentFiles = lib.mkOption {
      type = lib.types.listOf lib.types.path;
      default = [ ];
      description = ''
        Files providing `FISH_API_KEY` (and optional `FISH_*` knobs). With the
        `native` backend a file cannot change `FISH_PROXY_HOST`, `FISH_PROXY_PORT`
        or `FISH_PROXY_GRACEFUL_SHUTDOWN`: the module sets them on the command
        line. With `oci`, the container runtime decides which source wins, so
        keep those three out of the files.
      '';
    };
  };

  config = lib.mkIf cfg.enable (
    lib.mkMerge [
      {
        assertions = [
          {
            assertion =
              !(cfg.environment ? FISH_PROXY_PORT)
              && !(cfg.environment ? FISH_PROXY_HOST)
              && !(cfg.environment ? FISH_PROXY_GRACEFUL_SHUTDOWN);
            message = "services.fish-audio-suite-proxy.environment must not set FISH_PROXY_PORT, FISH_PROXY_HOST or FISH_PROXY_GRACEFUL_SHUTDOWN. Use the `port`, `host` and `gracefulShutdownSeconds` options.";
          }
          {
            assertion = cfg.backend != "oci" || cfg.image != null;
            message = "services.fish-audio-suite-proxy.image is required when backend = \"oci\".";
          }
        ];
        networking.firewall.allowedTCPPorts = lib.mkIf cfg.openFirewall [ cfg.port ];
      }

      (lib.mkIf (cfg.backend == "native") {
        systemd.services.fish-audio-suite-proxy = {
          description = "fish-audio-suite OpenAI-compatible Fish proxy";
          wantedBy = lib.optional cfg.autoStart "multi-user.target";
          after = [ "network-online.target" ];
          wants = [ "network-online.target" ];
          environment = env // {
            FISH_PROXY_HOST = cfg.host;
          };
          serviceConfig = {
            # systemd lets an EnvironmentFile override Environment=, so the three
            # values the firewall and the stop timeout depend on are set through
            # `env`, which runs after the files are loaded and always wins.
            ExecStart = utils.escapeSystemdExecArgs [
              (lib.getExe' pkgs.coreutils "env")
              "FISH_PROXY_HOST=${cfg.host}"
              "FISH_PROXY_PORT=${portStr}"
              "FISH_PROXY_GRACEFUL_SHUTDOWN=${toString cfg.gracefulShutdownSeconds}"
              (lib.getExe' cfg.package "fish-audio-suite-proxy")
            ];
            EnvironmentFile = cfg.environmentFiles;
            DynamicUser = true;
            Restart = "on-failure";
            TimeoutStopSec = stopSeconds;
            NoNewPrivileges = true;
            PrivateTmp = true;
            PrivateDevices = true;
            ProtectSystem = "strict";
            ProtectHome = true;
            ProtectKernelTunables = true;
            ProtectControlGroups = true;
            RestrictAddressFamilies = [
              "AF_INET"
              "AF_INET6"
              "AF_UNIX"
            ];
            LockPersonality = true;
            # Binding a port below 1024 needs this. Otherwise drop every capability.
            AmbientCapabilities = lib.optional lowPort "CAP_NET_BIND_SERVICE";
            CapabilityBoundingSet = if lowPort then [ "CAP_NET_BIND_SERVICE" ] else [ "" ];
          };
        };
      })

      (lib.mkIf (cfg.backend == "oci") {
        virtualisation.oci-containers.containers.fish-audio-suite-proxy = {
          image = cfg.image;
          ports = [ "${cfg.host}:${portStr}:${portStr}" ];
          environment = env;
          environmentFiles = cfg.environmentFiles;
          autoStart = cfg.autoStart;
          extraOptions = [ "--stop-timeout=${toString stopSeconds}" ];
        };
      })
    ]
  );
}
