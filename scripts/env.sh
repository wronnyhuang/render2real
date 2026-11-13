# source this: . scripts/env.sh   (edit the three paths below or export them before sourcing)
export WORK=${WORK:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}      # repo root
export SEAMLESS_ROOT=${SEAMLESS_ROOT:-$HOME/datasets/seamless-interaction}   # Meta Seamless Interaction, see DATA.md
export OUT_DIR=${OUT_DIR:-$WORK/out}                                        # generated pages/videos
export DATA=${DATA:-$WORK/data}                    # pairs/<id>/{render.mp4,real.mp4,meta.json} + manifest.json
export FRAMES=${FRAMES:-$WORK/frames}              # jpg frame cache created by the trainer
export DRIVE=${DRIVE:-$WORK/drive}                 # extra render-only clips (stylised avatar renders) for inference
export RUNS=${RUNS:-$WORK/runs}
