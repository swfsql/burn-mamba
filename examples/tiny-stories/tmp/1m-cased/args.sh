# Sourced by the scripts that run bin/ts-cuda.
# split_args "$@": sets app=(the AppArgs flags) and ex=(the example flags after
# `--`), and adds `--max-vram $VRAM_MAX_MIB` (default 3750) to ex unless the
# caller gave one. 3750 MiB of pools + ~140 MiB of CUDA context keeps the
# process under 3.8 GiB (the search limit). Call as:
# bin/ts-cuda … "${app[@]}" -- "${ex[@]}".
# stamp <dir>: writes <dir>/build.txt, the build info of bin/ts-cuda (see
# build.sh). bin_tag: the tag of bin/ts-cuda, for the `==` lines of the logs.
stamp() { cp bin/ts-cuda.txt "$1/build.txt"; }
bin_tag() { sed -n 's/^tag: //p' bin/ts-cuda.txt; }
split_args() {
  app=() ex=()
  local seen=0 a
  for a in "$@"; do
    if [ "$seen" = 0 ] && [ "$a" = "--" ]; then seen=1
    elif [ "$seen" = 0 ]; then app+=("$a")
    else ex+=("$a")
    fi
  done
  case " ${ex[*]} " in
    *" --max-vram "*) ;;
    *) ex+=(--max-vram "${VRAM_MAX_MIB:-3750}") ;;
  esac
}
