# EXP/batch1/aero2d — 第一批实验 1:地面效应单元素翼(2-D RANS)

- `code/solve.py`:gmsh 划网格 + OpenFOAM simpleFoam case 写入/运行/解析(重代码);薄层 `SRC/pipeline/exp_layer/aero2d` 静态 import。
- `code/run_walk.sh` / `run_lines.sh` / `run_seeds.sh`:walk 扫描、A/B/C 三线、多种子消融;所用 YAML 见 `config/runs.yaml`。
- `data/literature_zerihan2000.json`:Zerihan & Zhang 2000 参考值(`_base.yaml` 的 `literature`)。
- 产物:`OUTPUTs/batch1/aero2d/{store,v1,v1_seeds,_dev,_dev_walk}`。
