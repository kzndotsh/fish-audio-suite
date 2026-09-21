{
  config,
  lib,
  ...
}:
let
  cfg = config.services.fish-audio-suite-proxy;
in
{
  options.services.fish-audio-suite-proxy = {
    enable = lib.mkEnableOption "fish-audio-suite OpenAI-compat Fish proxy (OCI)";
    image = lib.mkOption {
      type = lib.types.str;
      default = "fish-audio-suite-proxy:latest";
    };
    environmentFiles = lib.mkOption {
      type = lib.types.listOf lib.types.path;
      default = [ ];
      description = "Files providing FISH_API_KEY (and optional FISH_* knobs).";
    };
  };

  config = lib.mkIf cfg.enable {
    virtualisation.oci-containers.containers.fish-audio-suite-proxy = {
      image = cfg.image;
      ports = [ "127.0.0.1:8849:8849" ];
      environmentFiles = cfg.environmentFiles;
      autoStart = false;
    };
  };
}
