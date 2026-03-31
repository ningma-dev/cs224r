# Improving policy gradients
回顾上一讲最终的PG
$$\nabla_\theta J(\theta) \approx \underbrace{\frac{1}{N} \sum_{i=1}^{N}}_{\text{avg over trajectories }} \underbrace{\sum_{t=1}^{T}}_{\text{ each timestep}} \underbrace{\nabla_\theta \log \pi_\theta(a_{i,t}|s_{i,t})}_{\text{policy log-likelihood gradient}} \left( \underbrace{\sum_{t'=t}^{T} r(s_{i,t'}, a_{i,t'})}_{\text{reward-to-go (single-sample estimate of } Q^\pi\text{)}} - \underbrace{b}_{\text{baseline}} \right)$$
仍然存在主要缺点是
1. make inefficient use of data：每收集一批数据只能做一次梯度更新（完全 on-policy），然后数据就被丢弃了
2. high variance：公式中的 reward to go 只是单个样本估计，取决于轨迹中未来时间步的动作和环境转移，而其因为随机策略未充分训练，波动会很大

核心想法：我们希望通过学习“什么是好，什么是坏”来同时改善这两个问题。用更准确的估计代替high variance的单个样本估计，并让数据充分被利用

我们定义三个关键数学对象来衡量策略的“好坏”：
- 状态价值函数 / state value /  $V^\pi(s)$：从状态 $s$ 开始，完全遵循策略 $\pi$行动，获得的未来预期回报总和
- 动作价值函数 / action value / $Q^\pi(s, a)$：在状态 $s$ 先采取动作 $a$ 后，再遵循策略 $\pi$行动，获得的未来预期回报总和。
	> state value 和 action value 的关系 (state value 是 action value 对动作 $a$ 取期望的结果)
	> $$V^{\pi}(s) = \mathbb{E}_{a\sim \pi(\cdot |s)} [Q^{\pi}(s, a)]$$
- 优势函数 / advantage / $A^\pi(s, a)$：衡量在状态 $s$ 下采取特定动作 $a$ 比遵循策略 $\pi$ 行动的平均表现好多少。其定义为： 将$$A^\pi(s_t, a_t) = Q^\pi(s_t, a_t) - V^\pi(s_t)$$
	- $A^{\pi} > 0$ 则动作 $a$ 优于策略的平均水平，反之则不如平均水平


改写PG公式：
1. 回顾 reward to go，它其实是 $Q^\pi(s_t, a_t)$ 的单样本估计。我们直接用 $Q^\pi(s_t, a_t)$ 替换它
2. 状态价值 $V^\pi(s)$ 天然就是稳定基线，所以替换基线 $b$
3. 我们得到改进的策略梯度，降低了variance，学习更稳定。这是更加理想化的PG版本 $$\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^T \nabla_\theta \log \pi_\theta(a_{i,t}|s_{i,t}) A^{\pi}(s_{i, t}, a_{i, t})$$
4. 然而我们不想严格计算 $A^{\pi}$。同时拟合 $Q^{\pi}$ 和 $V^{\pi}$ 需要2个网络，我们可以只用1个网络拟合 $V^{\pi}$ 来估计 $A^{\pi}$
5. 核心观察在于 $Q^{\pi}$ 可以用 $V^{\pi}$ 表达出来。回顾 $Q^{\pi}$ 的定义 $$Q^{\pi}(s_t, a_t) = \sum^T_{t^\prime = t} \mathbb{E}_{\pi_{\theta}} [r(s_{t^\prime}, a_{t^\prime}) | s_t, a_t)$$
6. 我们用动态规划的思想，把第一步的即时奖励 $r(s_t, a_t)$ 从求和中拆出来，剩下的部分恰好就是第二步的 state value 的期望，如此我们用下一步的 $V^{\pi}$ 表达出当前步 $Q^{\pi}$ 
	$$Q^\pi(s_t, a_t) = r(s_t, a_t) + \underbrace{\sum_{t'=t+1}^{T} \mathbb{E}_{\pi_\theta} [r(s_{t'}, a_{t'}) | s_{t+1}]}_{= \, \mathbb{E}_{s_{t+1}} [V^\pi(s_{t+1})]}$$
7. 代回优势函数，我们发现优势函数可以只用即时奖励和状态价值来表达 $$A^\pi(s_t, a_t) = r(s_t, a_t) + \mathbb{E}_{s_{t+1} \sim p(s'|s,a)} [V^\pi(s_{t+1})] - V^\pi(s_t)$$
8. 此时的优势函数中存在环境转移概率 $s_{t+1} \sim p(s'|s,a)$ ，为了说明接下来步骤的必要性，我们特别说明
	1. cs224r 讨论的RL为 model-free RL，意思是我们假设环境是黑盒，即使知道了 $(s, a)$，我们也没有 $p(s^\prime | s, a)$ 来建模环境转移概率分布。这符合现实中许多任务，比如robotics。而 RL with model 用来建模状态转移确定的任务，比如围棋。我们在 cs224r 学习的 model-free RL 不依赖任务本身的确定性，所以可以适应到更多任务
9. 为了消除对环境转移概率的依赖，我们用单样本估计来近似，即，用 $V^\pi(s_{t+1})$ 来代替 $\mathbb{E}_{s_{t+1} \sim p(s'|s,a)} [V^\pi(s_{t+1})]$ $$A^\pi(s_t, a_t) \approx r(s_t, a_t) + V^\pi(s_{t+1}) - V^\pi(s_t)$$
10. 上式中存在一个陷阱。当轨迹序列非常长（$T \rightarrow \infty$）时，整个轨迹上所有状态的价值都会趋于无限大，数值爆炸导致不能用神经网络拟合。所以引入折扣因子 $\gamma < 1$ 来将状态价值收敛到有限数值。下一节也讲到这个问题
11. 回到第4步说的，引入一个参数为 $\phi$ 的神经网络 $\hat{V}_{\phi}^{\pi}(s)$ 来拟合 $V^{\pi}$  。下一节讲的就是如何进行拟合。 $$\hat{A}^\pi(s_t, a_t) \approx r(s_t, a_t) + \gamma\hat{V}_{\phi}^{\pi}(s_{t+1}) - \hat{V}_{\phi}^{\pi}(s_t)$$
# How to estimate the value of a policy

如何找到最优的神经网络参数 $\phi$，让 $\hat{V}_{\phi}^{\pi}(s_t)$ 来逼近真实的状态价值 $V^\pi$ ？

逻辑桥梁：将 RL 评估任务转化为监督学习回归问题
既然我们已经推导出 $\hat{A}^\pi \approx r + \gamma \hat{V}_\phi(s') - \hat{V}_\phi(s)$，核心挑战就变成了如何训练这个 $\hat{V}_\phi$。在实践中，我们将其建模为一个标准的回归问题：

1. 构造数据集：利用智能体与环境交互生成的轨迹，收集状态 $s_i$ 作为输入特征
2. 确定监督标签 $y_i$：根据不同的评估策略（见下文 a, b, c），计算该状态对应的“目标价值” $y_i$ 作为标签
3. 损失函数：通过最小化均方误差 (MSE) 来更新参数 $\phi$： $$\mathcal{L}(\phi) = \frac{1}{2} \sum_i | \hat{V}_\phi^\pi(s_i) - y_i |^2$$
4. 自举（Bootstrapping）的特殊性：在版本 b 中，由于 $y_i$ 依赖于网络自身的预测，这意味着监督学习的标签会随着每一次梯度更新而改变。

评估策略
- a. 样本直接监督 (Sample & directly supervise / Monte Carlo)：使用整条轨迹的真实回报来监督 $$y_{i,t} = \sum_{t'=t}^T r(s_{i,t'}, a_{i,t'})$$
    - 这是无偏的，但方差极高，因为 $y_{i,t}$ 取决于之后所有随机的动作和环境转移。
- b. 使用自身估计 (Use your own estimate / Bootstrapping)：利用贝尔曼方程进行时序差分（TD）学习： $$y_{i,t} \approx r(s_{i,t}, a_{i,t}) + \gamma \hat{V}_\phi^\pi(s_{i,t+1})$$
    - 推导：这里我们拟合一个参数为 $\phi$ 的神经网络 $\hat{V}_\phi^\pi$，最小化均方误差 $\mathcal{L}(\phi) = \frac{1}{2} \sum | \hat{V}_\phi^\pi(s_i) - y_i |^2$。
    - 折现因子 ($\gamma$)：引入 $\gamma \in$（常用 0.99）来确保在无限步长任务下的数学收敛性，并体现“早点拿奖励比晚点拿更好”的直觉。
    - 直觉：方差低（只依赖一步随机性），但有偏，因为初始的 $\hat{V}$ 估计是不准确的。
- c. 折中方案 (N-step returns)：采样 $n$ 步真实奖励，再加上第 $n+1$ 步的状态价值估计： $$y_{i,t} = \sum_{j=0}^{n-1} \gamma^j r_{t+j} + \gamma^n \hat{V}_\phi^\pi(s_{t+n})$$ 这通常在实践中效果最好，因为它平衡了方差和偏差。

此时我们构建了一个完整的闭环系统，称为 Actor-Critic
- Actor：即策略 $\pi_{\theta}$ ，负责产生动作
- Critic：即神经网络 $\hat{V}_{\phi}^{\pi}$，负责逼近真实状态价值，进而提供PG的优化方向 $\nabla_\theta J(\theta)$
- 流程：运行 $\pi_{\theta}$ 采集数据 -> 监督学习拟合 $\hat{V}_{\phi}^{\pi}$ -> 计算优势函数 $A^\pi(s, a)$ 来提供 $\nabla_\theta J(\theta)$，优化PG -> 运行优化了的 $\pi_{\theta}$ 采集数据

# Off-policy actor-critic

我们仍然没有解决 make inefficient use of data 的问题。标准 Actor-Critic 的流程中 $\pi_{\theta}$ 优化和采集数据必须交替进行，或者说只能一边推理一边训练

为了重用历史数据，我们需要进入异策（Off-policy）范式。

- a. 重要性权重与步长约束 (Importance weights & constraining step size)：当使用非当前策略生成的数据时，需要引入重要性权重 $\frac{\pi_\theta(a|s)}{\pi_{old}(a|s)}$ 进行修正。
    - KL 约束/Clipping：如果多次更新导致策略偏离采样数据的旧策略太远，优势估计就会失效。我们需要通过 KL 散度约束（常见于 LLM 偏好优化）或裁剪（如 PPO）来限制步长。
- b. 经验重放池全异策版本 (Full off-policy version with replay buffers)：维护一个包含所有历史数据的重放池 $\mathcal{R}$，从中随机采样 minibatch。
    - 修正 Critic 目标值：在完全异策下，我们改用拟合 $Q$ 函数。目标值 $y_i = r_i + \gamma \hat{Q}_\phi^\pi(s_i', a_i')$ 中的动作 $a_i'$ 必须是 当前策略 $\pi_\theta$ 在下一状态会采取的动作，而非池中记录的旧动作。
