# On-policy Policy Gradient

## Derivation and intuition 

强化学习的目标是最大化期望回报 $J(\theta) = \mathbb{E}_{\tau \sim p_\theta(\tau)} [r(\tau)]$。为了做optimization，我们需要目标的梯度 $\nabla_{\theta} J(\theta)$

Derivation:
1. $\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim p_\theta(\tau)} [r(\tau)] = \nabla_\theta \int p_\theta(\tau) r(\tau) d\tau = \int \nabla_\theta p_\theta(\tau) r(\tau) d\tau$
2. $\nabla_\theta p_\theta(\tau)$ 是不可计算的，因为
	1. $p_{\theta}(\tau)$展开得到
		$$p_{\theta}(\tau) = p(s_1) \prod^T_{t=1} \pi_{\theta} (a_t | s_t) p(s_{t+1} | s_t, a_t)$$
	2. 环境转移概率 $p(s_{t+1} | s_t, a_t)$ 是未知的，不能关于 $\theta$ 求梯度
3. 使用对数微分恒等式（log-derivative trick）$\nabla_\theta p_\theta(\tau) = p_\theta(\tau) \nabla_\theta \log p_\theta(\tau)$ 进行替换
4. 替换后得到：$\int p_\theta(\tau) \nabla_\theta \log p_\theta(\tau) r(\tau) d\tau$
5. 写回期望形式：$\mathbb{E}_{\tau \sim p_\theta(\tau)} [\nabla_\theta \log p_\theta(\tau) r(\tau)]$
6. 展开轨迹的对数概率 $\log p_\theta(\tau)$，发现环境转移概率部分现在与 $\theta$ 无关
	$$\log p_\theta(\tau) = \log p(s_1) + \sum_{t=1}^T \log \pi_\theta(a_t|s_t) + \log p(s_{t+1} | s_t, a_t)$$
7. 求导后消失，只剩下策略部分： $$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim p_\theta(\tau)} \left[ \left( \sum_{t=1}^T \nabla_\theta \log \pi_\theta(a_t|s_t) \right) \left( \sum_{t=1}^T r(s_t, a_t) \right) \right]$$
Instuition： 这相当于“带权重的模仿学习”。如果一条轨迹的回报很高，梯度更新会增加该轨迹中所有动作的发生概率；如果回报为负，则降低其概率。简言之：多做“好”的，少做“坏”的。

## Full algorithm

这被称为 REINFORCE 或 Vanilla Policy Gradient：
1. 运行当前策略 $\pi_\theta$ 来采集一批轨迹数据 ${\tau^i}$。
2. 利用采集的数据计算梯度经验估计：$\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^N (\sum_{t=1}^T \nabla_\theta \log \pi_\theta(a_t^i|s_t^i)) (\sum_t r(s_t^i, a_t^i))$。
3. 更新策略参数：$\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$。

## Improving the gradient using causality and baselines

Vanilla PG 存在high variance的问题
1. 我们用了整个trajectory的reward来加权每一个动作的梯度
2. 这导致梯度极其不稳定，难以训练

high variance -> low variance 的2个办法
1. 加上因果性 (Causality)：$t$ 时刻的动作无法改变 $t$ 时刻之前的奖励。因此，我们将总奖励替换为“从当前起的未来奖励之和”（reward-to-go）： $$\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i} \sum_{t} \nabla_\theta \log \pi_\theta(a_{i,t}|s_{i,t}) \left( \sum_{t'=t}^T r(s_{i,t'}, a_{i,t'}) \right)$$
2. 加上基准线 (Baselines)：如果所有奖励都为正，梯度依然会鼓励所有行为。通过减去平均奖励 $b = \frac{1}{N} \sum r(\tau)$，我们可以让低于平均水平的行为产生负梯度。数学上，减去常数 $b$ 不会改变梯度的期望值（保持无偏），但能显著降低方差。$$\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i} \sum_{t} \nabla_\theta \log \pi_\theta(a_{i,t}|s_{i,t}) \left(\left( \sum_{t'=t}^T r(s_{i,t'}, a_{i,t'})\right) - b \right)$$

# Off-policy Policy Gradients

## Importance sampling

**On-policy** 算法每一步更新参数后都必须重新采集数据，这非常低效。我们希望复用过去策略（记作 $\bar{\pi}$）采集的数据。

利用重要性采样公式，我们可以改变期望的分布： $$J(\theta) = \mathbb{E}_{\tau \sim \bar{p}(\tau)} \left[ \frac{p_\theta(\tau)}{\bar{p}(\tau)} r(\tau) \right]$$ 其中轨迹权重比 $\frac{p_\theta(\tau)}{\bar{p}(\tau)}$ 会简化为动作概率的比值乘积 $\prod \frac{\pi_\theta(a_t|s_t)}{\bar{\pi}(a_t|s_t)}$。为了数值稳定，我们通常将期望转化为对时间步的求和，而不是对整个轨迹求积。

## KL constraint

如果策略 $\pi_\theta$ 变化太大，导致它与数据来源策略 $\bar{\pi}$ 偏离过远，重要性采样的权重就会失效，梯度估计的准确性会大幅下降。 因此，我们需要约束新旧策略之间的差异，通常采用 KL 散度约束： $$\mathbb{E}_{s \sim \bar{\pi}} [D_{KL}(\bar{\pi}(\cdot|s) | \pi_\theta(\cdot|s))] \le \epsilon$$ 这保证了在同一批数据上进行多次更新时的算法稳定性。
