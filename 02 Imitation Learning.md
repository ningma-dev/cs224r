模仿学习的目标是从给定专家策略 $\pi_{expert}$ 生成的演示数据集(demonstration) $\mathcal{D} = \set{\tau_1, \tau_2, ..., \tau_n }$ 中学习策略 $\pi_{\theta}$ 来模仿专家的行为

# Imitation Learning - version 0
考虑最简单的IL，使用supervised regression的方式训练，目标是最小化均方误差(MSE)
1. 有数据集 $\mathcal{D} = \set{\tau_1, \tau_2, ..., \tau_n }$
2. 用MSE做目标进行训练
	$$\min_{\theta} \frac{1}{|\mathcal{D}|} \sum_{(s, a)\in \mathcal{D}} ||a - \hat{a}||^2 ,\quad \text{where } \hat{a} = \pi_{\theta}(s)$$

version 0 看似正确，但存在致命局限性。它假设策略是确定性deterministic的。现实世界许多任务中，正确的做法不止有一种，反应到专家数据集 $\mathcal{D}$ 中，是轨迹的分布可能存在多峰性(multimodel)。version 0 直接用L2正则的MSE损失会导致学习出的策略忽略多峰性，而输出专家动作的平均值。

## Imitation Learning - version 1
将策略 $\pi_{\theta (a|s)}$ 建模成生成式模型更具表现力

1. 有数据集 $\mathcal{D} = \set{\tau_1, \tau_2, ..., \tau_n }$
2. 用最大似然估计做目标进行训练，通常用负对数似然损失
	$$\min_{\theta} -\mathbb{E}_{(s, a) \sim \mathcal{D}} [\log \pi_{\theta} (a|s)]$$
## Compounding Errors
SL中假设输入是i.i.d.的，但是IL中前面动作会影响下一个动作，这带来漂移现象(Drift)
1. 即使经过训练，策略与专家策略总有差异 $p_\pi (s) \neq p_{expert} (s)$
2. 推理中，一旦 $\pi_{\theta}$ 犯了小错，agent很容易进入 $\mathcal {D}$ 没有覆盖的状态空间，导致犯更大错
3. 连续错误累积造成漂移

## Online intervention and DAgger
DAgger: 为了解决compounding errors，我们在推理时收集online数据，作为监督信号加入到原有数据集
1. 运行当前策略 $\pi_{\theta}$​ 得到轨迹
2. 询问专家：在这些策略访问过的状态下，专家会怎么做？得到 $(s^\prime, a^*)$
3. 将新数据并入数据集 $\mathcal{D} \leftarrow \mathcal{D} \cup {(s^\prime,a^*)}$
4. 重新训练策略

