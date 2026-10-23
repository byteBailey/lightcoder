# Ablation comparison (same 20-instance subset, model=397b)

```
配置	样本数	解决	解决率%	空patch	错误	平均调用	平均用时s	平均patch字符
no_compression	20	10/20	50.0	1	0	35.8	86	1868
prompt_react	20	10/20	50.0	0	0	36.1	94	2505
prompt_minimal	20	8/20	40.0	0	0	34.9	95	3491
no_subagent	20	7/20	35.0	1	0	38.5	107	3628
baseline	20	5/20	25.0	1	0	37.2	108	2438
```
