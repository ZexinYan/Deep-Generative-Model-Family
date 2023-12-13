# GAN

Training formula:

$$
    \begin{aligned}
    & L_D(\phi ; \theta)=\mathbb{E}_{\boldsymbol{x} \sim p_\theta(\boldsymbol{x})}\left[D_\phi(\boldsymbol{x})\right]-\mathbb{E}_{\boldsymbol{x} \sim p_{\mathrm{data}}(\boldsymbol{x})}\left[D_\phi(\boldsymbol{x})\right]+\lambda \mathbb{E}_{\boldsymbol{x} \sim r_\theta(\boldsymbol{x})}\left[\left(\left\|\nabla D_\phi(\boldsymbol{x})\right\|_2-1\right)^2\right] \\
    & L_G(\theta ; \phi)=-\mathbb{E}_{\boldsymbol{x} \sim p_\theta(\boldsymbol{x})}\left[D_\phi(\boldsymbol{x})\right]
    \end{aligned}
$$
