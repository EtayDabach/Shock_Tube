<h1 style="text-align: center;"> Shock Tube Solver GUI </h1>
All explanations will be added soon!

<!-- * Based on Riemann Solvers and Numerical Methods for Fluid Dynamic by Eleuterio.F.Toro. -->
 
<h2> The Riemann Problem: </h2> <!--chapter 4 (pg 137), 6 (pg 235), 10 (pg 336), chapter 1 for Euler eqs (pg 23)-->

<!-- We want to solve the Riemann problem for the one dimensional time-dependent Euler equations with Initial Boundary Value Problem (IBVP):


<!-- $$
\begin{align}
&\text{Partial Differential Equation (PDEs)}&&: \frac{\partial \bold{U}}{\partial t}  + \frac{\partial {F(U)}}{\partial x} = 0 \\
&\text{Initial Conditions (ICs)}&&: {U}(x,0) = \begin{cases} {U}_{L} \quad \text{if} \quad x<0 \\ {U}_{R} \quad \text{if} \quad x>0 \end{cases} \\
&\text{Boundary Conditions (BCs)}&&: \begin{cases} {U}(0,t) = {U}_{l}(t) \\ \bold{U}(L,t) = {U}_{r}(t) \end{cases}
\end{align}
$$ -->

<!--
$$
\begin{array}{ll}
\text{Partial Differential Equation (PDEs)} & : \frac{\partial \mathbf{U}}{\partial t}  + \frac{\partial {F(U)}}{\partial x} = 0 \\
\text{Initial Conditions (ICs)} & : {U}(x,0) = \begin{cases} {U}_{L} & \text{if} \quad x<0 \\ {U}_{R} & \text{if} \quad x>0 \end{cases} \\
\text{Boundary Conditions (BCs)} & : \begin{cases} {U}(0,t) = {U}_{l}(t) \\ \mathbf{U}(L,t) = {U}_{r}(t) \end{cases}
\end{array}
$$
-->


<!-- 
in a domain $0\leq x \leq L$. 

${U}$ is a vector of the conserved variables and ${F(U)}$ is the flux: 

$$ {U} = \left[\begin{array}{cc} \rho \\\ \rho u \\\ E \end{array}\right] ; {F} = \left[\begin{array}{cc} \rho u \\\ \rho u^{2} + P \\\ u(E+P) \end{array}\right] $$

where $\rho$ is the density, $u$ is the particle velocity, $E =\rho (\frac{1}{2}u^{2} + e)$ is the total energy per unit volume with the specific kinetic energy and specific internal energy ($e$) and $P$ is the pressure. $\\$

For ideal gases we have $e = \frac{P}{(\gamma - 1)\rho}$ and speed of sound $a^{2} = \left(\frac{dP}{d\rho}\right)_{S} = \gamma \frac{P}{\rho}$ . $\\$
We use the explicit conservative formula 

$$\bold{U}_{i}^{n+1} = \bold{U}_{i}^{n} - \frac{\Delta{t}}{\Delta{x}}[\bold{F}_{i+\frac{1}{2}} - \bold{F}_{i-\frac{1}{2}}]$$

$$\Delta{t} = \frac{C_{cfl}\Delta{x}}{S_{max}^{n}} \quad ; \quad S_{max}^{n}=\max({|u_{i}^{n}|+a_{i}^{n}})$$

where $\bold{F}_{i+\frac{1}{2}} = \bold{F(\bold{U}_{i+\frac{1}{2}}(0))}$ from the Godunov Flux in which $\bold{U}_{i+\frac{1}{2}}(0)$ is the exact similarity solution $\bold{U(x/t)}_{i+\frac{1}{2}}$ of the Riemann problem evaluated at $x/t=0$, $\Delta{x}$ is the size of the cell, $\Delta{t}$ is the time step which consist of a constant $0<C_{cfl} \leq 1$, and $S_{max}^{n}$ the largest wave speed present throughout the domain at time level $n$ ($u_{i}^{n}$ is the particle velocity at cell $i$ and $a_{i}^{n}$ is the sound speed). Upper index ,$n$, is for time steps and lower index ,$i$, is for x steps. If we divide the tube with the fluid to cells with index $i$ so $i+\frac{1}{2}$ means the right wall of the cell and $i-\frac{1}{2}$ is the left wall, $x=0$ is the left edge of the tube. $\\$
We want to approximate the flux $\bold{F}$ by using HLL and HLLC Riemann solvers. Using integral relations on the PDE (Toro-page.318 or 337 on PDF file) we get: $\\$
$$\bold{U}^{hll} = \frac{S_{R}\bold{U}_{R} - S_{L}\bold{U}_{L} + \bold{F}_{L} - \bold{F}_{R}}{S_{R}-S_{L}}$$
-->

<!--
## The HLL Approximate Riemann Solver
# <center> ![HLL Approximate](./Toro_Approximate_HLL_Riemann_slover.png)
HLL approximate Riemann solver is:
$$\bold{\tilde{U}}(x,t) = \begin{cases} \bold{U}_{L} \quad \text{if} \quad \frac{x}{t} < S_{L} \\ \bold{U}^{hll} \quad \text{if} \quad S_{L} \leq \frac{x}{t} \leq S_{R} \\  \bold{U}_{R} \quad \text{if} \quad \frac{x}{t} > S_{R}\end{cases}$$

And the flux will be: 
$$\bold{F}^{hll} = \frac{S_{R}\bold{F}_{L} - S_{L}\bold{F}_{R} + S_{L}S_{R}(\bold{U}_{R} - \bold{U}_{L})}{S_{R}-S_{L}}$$

The corresponding HLL intercell flux for the approximate Godunov method is then given by:
$$\bold{{F}}_{i+\frac{1}{2}}^{hll} = \begin{cases} \bold{F}_{L} \quad \text{if} \quad 0 < S_{L} \\ \bold{F}^{hll} \quad \text{if} \quad S_{L} \leq 0 \leq S_{R} \\  \bold{F}_{R} \quad \text{if} \quad 0 > S_{R}\end{cases}$$ 
-->


<h3> HLLC Approximation </h3> 
<!-- # <center> ![HLL Approximate](./Toro_Approximate_HLLC_Riemann_slover.png) -->

<!-- 
HLLC is a modified version of HLL whereby the missing contact and shear waves in the Euler equations are restored.
$$\bold{\tilde{U}}(x,t) = \begin{cases} \bold{U}_{L} \quad \text{if} \quad \frac{x}{t} < S_{L} \\ \bold{U}_{*L} \quad \text{if} \quad S_{L} \leq \frac{x}{t} < S_{*} \\ \bold{U}_{*R} \quad \text{if} \quad S_{*} \leq \frac{x}{t} < S_{R} \\  \bold{U}_{R} \quad \text{if} \quad \frac{x}{t} \geq S_{R}\end{cases}$$

flux:
$$\bold{F}_{i+\frac{1}{2}}^{hllc} = \begin{cases} \bold{F}_{L} \quad \text{if} \quad 0 < S_{L} \\ \bold{F}_{*L} \quad \text{if} \quad S_{L} \leq 0 < S_{*} \\ \bold{F}_{*R} \quad \text{if} \quad S_{*} \leq 0 < S_{R} \\  \bold{F}_{R} \quad \text{if} \quad 0 \geq S_{R}\end{cases}$$

so we get:
$$\bold{U}_{*K} = \rho_{*} \left(\frac{S_{K} - u_{K}}{S_{K} - S_{*}} \right) \left[\begin{array}{cc} 1 \\ S_{*} \\ \frac{E_{K}}{\rho{K}} - (S_{*} - u{K})\left[S_{*} + \frac{P_{K}}{\rho_{K}(S_{K} - u_{K})}\right] \end{array}\right]$$
$$\bold{F}_{*K} = \bold{F}_{K} + S_{K}(\bold{U}_{*K} - U_{K}) $$
where K = L or R.
-->

<h3> Exact Solution </h3>

<!-- ** This is only final results without derivations. All of this can be found in Toro. -->

<h2 style="text-align: center;"> GUI explanation </h2>

![Screenshot 2025-01-22 212217](https://github.com/user-attachments/assets/465e6138-f7f4-4443-8eaf-b524a0160500)
