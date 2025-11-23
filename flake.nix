  description = "Develop Python on Nix with uv";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  };

  outputs =
    { nixpkgs, ... }:
    let
      inherit (nixpkgs) lib;
      forAllSystems = lib.genAttrs lib.systems.flakeExposed;
    in
    {
      devShells = forAllSystems (
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
        in
        {
          default = pkgs.mkShell {
            packages = [
              pkgs.python3
              pkgs.uv
            ];

            env = lib.optionalAttrs pkgs.stdenv.isLinux {
              # Python libraries often load native shared objects using dlopen(3).
              # Setting LD_LIBRARY_PATH makes the dynamic library loader aware of libraries without using RPATH for lookup.
              LD_LIBRARY_PATH = lib.makeLibraryPath pkgs.pythonManylinuxPackages.manylinux1;
            };

            shellHook = ''
              unset PYTHONPATH
              if [ -z "$NIX_SHELL_NESTED" ]; then
                export NIX_SHELL_NESTED=1
                alias ipy="uv run ipython --nosep"
                export PS1="🧢 \e[38;5;211m\]g\e[38;5;111mBLP\[\e[0m $PS1"
                if [ ! -d .venv ]; then
                  echo "Running uv init, adding ipython and pip"
                  uv init .
                  uv add pip
                  uv add ipython
                fi
                uv run pip list
              else
                alias ipy="uv run ipython --nosep"
                export PS1="🧢 \e[38;5;211m\]g\e[38;5;111mBLP\[\e[0m [NESTED] $PS1"
                echo "Nested nix-shell detected, skipping uv init"
              fi
            '';
          };
        }
      );
    };
}
