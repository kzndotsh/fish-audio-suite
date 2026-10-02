# Called as `import ./module.nix self` from flake.nix so the default package
# is this flake's proxy build.
self:
{
  config,
  lib,
  pkgs,
  ...
}:
let
  cfg = config.services.fish-audio-suite-proxy;
  portStr = toString cfg.port;
  env = {
    FISH_PROXY_PORT = portStr;
  }
  // cfg.environment;
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
      description = "Proxy port. Sets `FISH_PROXY_PORT`, and the published port for `oci`.";
    };

    openFirewall = lib.mkOption {
      type = lib.types.bool;
      default = false;
      description = "Open `port` in the firewall. Only useful when `host` is not loopback.";
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
        FISH_MODEL = "s2.1-pro";
      };
      description = "Extra `FISH_*` variables. Do not put secrets here; they land in the Nix store.";
    };

    environmentFiles = lib.mkOption {
      type = lib.types.listOf lib.types.path;
      default = [ ];
      description = ''
        Files providing `FISH_API_KEY` (and optional `FISH_*` knobs). A file
        that sets `FISH_PROXY_PORT` overrides `port` for the service but not
        for the firewall or the published container port.
      '';
    };
  };

  config = lib.mkIf cfg.enable (
    lib.mkMerge [
      {
        assertions = [
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
            ExecStart = lib.getExe' cfg.package "fish-audio-suite-proxy";
            EnvironmentFile = cfg.environmentFiles;
            DynamicUser = true;
            Restart = "on-failure";
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
        };
      })
    ]
  );
}
