{
  description = "Unofficial Fish Audio toolkit (kit, OpenAI-compat proxy, live voice)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.uv2nix.follows = "uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    {
      self,
      nixpkgs,
      pyproject-nix,
      uv2nix,
      pyproject-build-systems,
      ...
    }:
    let
      inherit (nixpkgs) lib;
      forAllSystems = lib.genAttrs [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];

      workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ./.; };

      overlay = workspace.mkPyprojectOverlay {
        sourcePreference = "wheel";
      };

      pythonSets = forAllSystems (
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          python = pkgs.python312;
        in
        (pkgs.callPackage pyproject-nix.build.packages {
          inherit python;
        }).overrideScope
          (
            lib.composeManyExtensions [
              pyproject-build-systems.overlays.default
              overlay
            ]
          )
      );

      # One version for the wrapper, read from the voice package metadata.
      voiceVersion = (builtins.fromTOML (builtins.readFile ./packages/voice/pyproject.toml)).project.version;

      # sounddevice dlopens PortAudio. Linux also needs PulseAudio for it.
      wrapVoice =
        system: env:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          libs = [ pkgs.portaudio ] ++ lib.optional pkgs.stdenv.isLinux pkgs.libpulseaudio;
          libVar = if pkgs.stdenv.isLinux then "LD_LIBRARY_PATH" else "DYLD_FALLBACK_LIBRARY_PATH";
          wrap = bin: "makeWrapper ${env}/bin/${bin} $out/bin/${bin} --prefix ${libVar} : ${lib.makeLibraryPath libs}";
        in
        pkgs.stdenv.mkDerivation {
          pname = "fish-audio-suite-voice";
          version = voiceVersion;
          dontUnpack = true;
          nativeBuildInputs = [ pkgs.makeWrapper ];
          installPhase = ''
            mkdir -p $out/bin
            ${wrap "fish-audio-suite-voice"}
            ${wrap "fish-voice"}
          '';
        };
    in
    {
      packages = forAllSystems (
        system:
        let
          pythonSet = pythonSets.${system};
          kitEnv = pythonSet.mkVirtualEnv "fish-audio-suite-kit" {
            fish-audio-suite-kit = [ ];
          };
          proxyEnv = pythonSet.mkVirtualEnv "fish-audio-suite-proxy" {
            fish-audio-suite-proxy = [ ];
          };
          voiceEnv = pythonSet.mkVirtualEnv "fish-audio-suite-voice" {
            fish-audio-suite-voice = [
              "cli"
              "tui"
            ];
          };
        in
        {
          default = proxyEnv;
          fish-audio-suite-kit = kitEnv;
          fish-audio-suite-proxy = proxyEnv;
          fish-audio-suite-voice = wrapVoice system voiceEnv;
        }
      );

      apps = forAllSystems (system: {
        default = {
          type = "app";
          program = "${self.packages.${system}.fish-audio-suite-proxy}/bin/fish-audio-suite-proxy";
        };
        fish-audio-suite-voice = {
          type = "app";
          program = "${self.packages.${system}.fish-audio-suite-voice}/bin/fish-voice";
        };
      });

      nixosModules.default = import ./nix/module.nix self;

      checks = forAllSystems (system: {
        inherit (self.packages.${system})
          fish-audio-suite-kit
          fish-audio-suite-proxy
          fish-audio-suite-voice
          ;
      });
    };
}
