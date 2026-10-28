# DR11 消光改正核查

2026-10-28。当前项目已经对 `g,r,z,W1,W2` 的测光做银河系前景消光改正。
DR11 sweep 的 `FLUX_*` 是 nanomaggy、AB 系统；`EBV` 取自 SFD98，
目录给出由该值计算的逐波段 `MW_TRANSMISSION_*`。项目读取目录的
透过率，不自行重新选择一套系数。[DR11 目录说明](https://www.legacysurvey.org/dr11/catalogs/)。

在 `src/photdis/photometry.py` 中，每个波段都计算

\[
m_0=22.5-2.5\log_{10}F+2.5\log_{10}T_{\rm MW}
    =m_{\rm obs}-A_\lambda,
\qquad A_\lambda=-2.5\log_{10}T_{\rm MW}.
\]

模型输入是 `(g-r)_0,(r-z)_0,(z-W1)_0,(W1-W2)_0` 中的可用颜色。
Gaia 高精度锚点的标签在 `src/photdis/stage1.py` 中使用
`M_g=g0-10+5log10(p_corr/mas)`；推断时的距离模数使用同次观测的
`g0-M_g`。DR11 的所有流量均为 AB，包括 WISE，因此没有再次转换
WISE 零点。

共享噪声模拟先用源的 `r0` 与目标噪声母样本的 `r0` 确定注入的
距离模数差，再把每个波段的源流量乘以
`10**(-0.4*shift) * T_target/T_source`；合成行携带目标位置的
`MW_TRANSMISSION_*`，重新计算 `g0` 和全部颜色。跨天区透过率不同
时，它仍保留源的内禀颜色。`tests/test_photometry.py` 已检验去消光
符号；`tests/test_noise.py` 的跨天区测试使用不同的五波段透过率，
检验注入 2.5 mag 后所有去消光颜色保持不变。针对性测试为 8 项通过。

DR11 官方给出的 DECam `g,r,z` 系数约为
`3.214, 2.165, 1.211`，WISE `W1,W2` 为 `0.184, 0.113`，单位
均为 `A_lambda/E(B-V)`。[官方系数](https://www.legacysurvey.org/dr11/catalogs/#galactic-extinction-coefficients)。
本阶段天区要求 `0<=EBV<0.03`，所以按该目录系数，前景柱的
`A_g` 小于约 0.096 mag。这是 SFD98 全尘柱近似；近处恒星可能
位于部分尘层之前。当前形式星等误差未计入尘图误差或消光律变化，
本轮跨天区偏差实验需要按 `EBV` 与颜色继续分层检查。
