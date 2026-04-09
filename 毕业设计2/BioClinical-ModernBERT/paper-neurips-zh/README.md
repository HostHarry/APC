# 中文 NeurIPS 论文草稿

该目录是基于本仓库内容整理的中文论文草稿，使用 `neurips.sty` 模版组织。

当前文件：

- `main.tex`：论文正文
- `extra_pkgs.tex`：中文与常用宏包
- `references.bib`：已核对并写入的最小参考文献集合
- `neurips.sty`：NeurIPS 样式文件

建议编译方式：

```bash
xelatex main.tex
bibtex main
xelatex main.tex
xelatex main.tex
```

说明：

- 本机当前未检测到 `xelatex` / `latexmk`，因此我没有实际编译验证。
- 论文实验结果部分按需求保留为占位结构，后续可直接填入真实结果。
- 相关工作中除仓库 README 已给出的 `BioClinical ModernBERT` 预印本外，其余参考文献建议在定稿前逐条联网核验后补充。
