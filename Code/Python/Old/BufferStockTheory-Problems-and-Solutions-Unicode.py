# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.1.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Theoretical Foundations of Buffer Stock Saving
#
# <cite data-cite="6202365/8AH9AXN2"></cite>
#
# <p style="text-align: center;"><small><small>Generator: BufferStockTheory-make/notebooks_byname</small></small></p>

# %% [markdown]
# <p style="text-align: center;"><small><small><small>For the following badges: GitHub does not allow click-through redirects; right-click to get the link, then paste into navigation bar</small></small></small></p>
#
# <!-- Disabling binder because it is excruciatingly slow
# [![Open in Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/econ-ark/REMARK/master?filepath=REMARKs%2FBufferStockTheory%2FBufferStockTheory.ipynb)
# -->
#
# [![Open in CoLab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/econ-ark/REMARK/blob/master/REMARKs/BufferStockTheory/BufferStockTheory.ipynb)
#
# [This notebook](https://github.com/econ-ark/REMARK/blob/master/REMARKs/BufferStockTheory/BufferStockTheory.ipynb) uses the [Econ-ARK/HARK](https://github.com/econ-ark/hark) toolkit to describe the main results and reproduce the figures in the paper [Theoretical Foundations of Buffer Stock Saving](https://econ-ark.github.io/BufferStockTheory) 
#
# If you are not familiar with the HARK toolkit, you may wish to browse the ["Gentle Introduction to HARK"](https://mybinder.org/v2/gh/econ-ark/DemARK/master?filepath=Gentle-Intro-To-HARK.ipynb) before continuing (since you are viewing this document, you presumably know a bit about [Jupyter Notebooks](https://jupyter-notebook-beginner-guide.readthedocs.io/en/latest/)).
#
# For instructions on how to install the [Econ-ARK/HARK](https://github.com/econ-ark/hark) toolkit on your computer, please refer to the [QUICK START GUIDE](https://github.com/econ-ark/HARK/blob/master/README.md). 
#
# The main HARK tool used here is $\texttt{IndShockConsumerType}$, in which agents have CRRA utility and face idiosyncratic shocks to permanent and transitory income.  For an introduction to this tool, see the [IndShockConsumerType.ipynb](https://econ-ark.org/notebooks) notebook at the [Econ-ARK](https://econ-ark.org) website (or in the "notebooks" directory of the [DemARK](https://github.com/econ-ark/DemARK) repo.

# %% {"code_folding": []}
# This cell does some setup; please be patient, it may take a couple of minutes

# The tools for navigating the filesystem
import sys
import os

# Determine the platform so we can do things specific to each 
import platform
pform = ''
pform = platform.platform().lower()
if 'darwin' in pform:
    pf = 'darwin' # MacOS
if 'debian'in pform:
    pf = 'debian' # Probably cloud (MyBinder, CoLab, ...)
if 'ubuntu'in pform:
    pf = 'debian' # Probably cloud (MyBinder, CoLab, ...)
if 'win' in pform:
    pf = 'win'

# Test whether latex is installed (some of the figures require it)
from distutils.spawn import find_executable

iflatexExists=False

if find_executable('latex'):
    iflatexExists=True

# Uncomment the code block below if you want to force installation of LaTeX
# (The figures are more attractively rendered when LaTeX is used)
if not iflatexExists:    
    # If running on Mac or Win, user can be assumed to be able to install
    # any missing packages in response to error messages; but not on cloud
    # so load LaTeX by hand (painfully slowly)
    if 'debian' in pf: # CoLab and MyBinder are both ubuntu
        print('Installing LaTeX now; please wait 3-5 minutes')
        from IPython.utils import io
        
        with io.capture_output() as captured: # Hide hideously long output 
            os.system('apt-get update')
            os.system('apt-get -y install texlive texlive-latex-extra texlive-xetex dvipng')
            iflatexExists=True
    else:
        print('Please install a full distributon of LaTeX on your computer then rerun.')
        print('A full distribution means textlive, texlive-latex-extras, texlive-xetex, dvipng, and ghostscript')
        sys.exit()

# This is a jupytext paired notebook that autogenerates BufferStockTheory.py
# which can be executed from a terminal command line via "ipython BufferStockTheory.py"
# But a terminal does not permit inline figures, so we need to test jupyter vs terminal
# Google "how can I check if code is executed in the ipython notebook"

from IPython import get_ipython # In case it was run from python instead of ipython

# If the ipython process contains 'terminal' assume not in a notebook
def in_ipynb():
    try:
        if 'terminal' in str(type(get_ipython())):
            return False
        else:
            return True
    except NameError:
        return False

if in_ipynb():
    # Now install stuff aside from LaTeX (if not already installed)
    os.system('pip install econ-ark==0.10.0.dev3')
    os.system('pip install matplotlib')
    os.system('pip install numpy')
    os.system('pip install scipy')
    os.system('pip install ipywidgets')
    os.system('pip install jupyter_contrib_nbextensions')
    os.system('jupyter contrib nbextension install --user')
    os.system('jupyter nbextension enable codefolding/main')
    os.system('jupyter nbextension enable codefolding/edit')
    os.system('jupyter nbextension enable latex_envs/latex_envs')
    os.system('jupyter nbextension enable navigation-hotkeys')
    os.system('pip install cite2c')
    os.system('python -m cite2c.install')
else:
    print('In batch mode')
    
# Import related generic python packages
import numpy as np
from time import clock
mystr = lambda number : "{:.4f}".format(number)

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, draw, show

# In order to use LaTeX to manage all text layout in our figures, 
# we import rc settings from matplotlib.
from matplotlib import rc

plt.rc('font', family='serif')
plt.rc('text', usetex=iflatexExists)

# The warnings package allows us to ignore some harmless but alarming warning messages
import warnings
warnings.filterwarnings("ignore")

from copy import copy, deepcopy

# Determine whether to make the figures inline (for spyder or jupyter)
# vs whatever is the automatic setting that will apply if run from the terminal
if in_ipynb():
    # %matplotlib inline generates a syntax error when run from the shell
    # so do this instead
    get_ipython().run_line_magic('matplotlib', 'inline')
else:
    get_ipython().run_line_magic('matplotlib', 'auto')

# Code to allow a master "Generator" and derived "Generated" versions
Generator=True # Is this notebook the master or is it generated?

# Define (and create, if necessary) the figures directory "Figures"
if Generator:
    my_file_path = os.path.dirname(os.path.abspath("BufferStockTheory.ipynb")) # Find pathname to this file:
    Figures_HARK_dir = os.path.join(my_file_path,"Figures/") # LaTeX document assumes figures will be here
    Figures_HARK_dir = os.path.join(my_file_path,"/tmp/Figures/") # Uncomment to make figures outside of git path
    if not os.path.exists(Figures_HARK_dir):
        os.makedirs(Figures_HARK_dir)
        
if not in_ipynb(): # running in batch mode
    print('You appear to be running from a terminal')
    print('By default, figures will appear one by one')

# %%
# Import HARK tools needed

from HARK.ConsumptionSaving.ConsIndShockModel import IndShockConsumerType
from HARK.utilities import plot_funcs_der, plot_funcs

# %% [markdown]
# ## [The Problem](https://econ-ark.github.io/BufferStockTheory/#The-Problem) 
#
# The paper defines and calibrates a small set of parameters: <!-- defined in latexdefs.tex -->
#
# | Parameter | Description | Code | Value |
# |:---:| ---         | ---  | :---: |
# | $Γ$ | Permanent Income Growth Factor | $\texttt{PermGroFac}$ | 1.03 |
# | $𝖱$ | Interest Factor | $\texttt{Rfree}$ | 1.04 |
# | $β$ | Time Preference Factor | $\texttt{DiscFac}$ | 0.96 |
# | $ρ$ | Coeﬃcient of Relative Risk Aversion| $\texttt{CRRA}$ | 2 |
# | $℘$ | Probability of Unemployment | $\texttt{UnempPrb}$ | 0.005 |
# | $μ$ | Income when Unemployed | $\texttt{IncUnemp}$ | 0. |
# | $σ_ψ$ | Std Dev of Log Permanent Shock| $\texttt{PermShkStd}$ | 0.1 |
# | $σ_θ$ | Std Dev of Log Transitory Shock| $\texttt{TranShkStd}$ | 0.1 |
#
# For a microeconomic consumer with 'Market Resources' (net worth plus current income) $M_{t}$, end-of-period assets $A_{t}$ will be the amount remaining after consumption of $C_{t}$.  <!-- Next period's 'Balances' $B_{t+1}$ reflect this period's $A_{t}$ augmented by return factor $R$:-->
# \begin{eqnarray}
# A_{t}   &=&M_{t}-C_{t}
# \end{eqnarray}
#
# The consumer's permanent noncapital income $P$ grows by a predictable factor $Γ$ and is subject to an unpredictable lognormally distributed multiplicative shock $\mathbb{E}_{t}[ψ_{t+1}]=1$, 
# \begin{eqnarray}
# P_{t+1} & = & P_{t} Γ ψ_{t+1}
# \end{eqnarray}
#
# and actual income is permanent income multiplied by a logormal multiplicative transitory shock, $\mathbb{E}_{t}[θ_{t+1}]=1$, so that next period's market resources are
# \begin{eqnarray}
# %M_{t+1} &=& B_{t+1} +P_{t+1}θ_{t+1},  \notag
# M_{t+1} &=& A_{t}𝖱 +P_{t+1}θ_{t+1}.  \notag
# \end{eqnarray}
#
# When the consumer has a CRRA utility function $u(c)=\frac{c^{1-ρ}}{1-ρ}$, the paper shows that the problem can be written in terms of ratios of money variables to permanent income, e.g. $m_{t} \equiv M_{t}/P_{t}$, and the Bellman form of [the problem reduces to](https://econ-ark.github.io/BufferStockTheory/#The-Related-Problem):
#
# \begin{eqnarray*}
# v_t(m_t) &=& \max_{c_t}~~ u(c_t) + β~\mathbb{E}_{t} [(Γψ_{t+1})^{1-ρ} v_{t+1}(m_{t+1}) ] \\
# & s.t. & \\
# a_t &=& m_t - c_t \\
# m_{t+1} &=& R/(Γ ψ_{t+1}) a_t + θ_{t+1} \\
# \end{eqnarray*}
#

# %% {"code_folding": [0]}
# Define a parameter dictionary with baseline parameter values

# Set the baseline parameter values 
PermGroFac = 1.03
Rfree      = 1.04
DiscFac    = 0.96
CRRA       = 2.00
UnempPrb   = 0.005
IncUnemp   = 0.0
PermShkStd = 0.1
TranShkStd = 0.1
# Import default parameter values
import HARK.ConsumptionSaving.ConsumerParameters as Params 

# Make a dictionary containing all parameters needed to solve the model
base_params = Params.init_idiosyncratic_shocks

# Set the parameters for the baseline results in the paper
# using the variable values defined in the cell above
base_params['PermGroFac'] = [PermGroFac]   # Permanent income growth factor
base_params['Rfree']      = Rfree          # Interest factor on assets
base_params['DiscFac']    = DiscFac        # Time Preference Factor
base_params['CRRA']       = CRRA           # Coefficient of relative risk aversion
base_params['UnempPrb']   = UnempPrb       # Probability of unemployment (e.g. Probability of Zero Income in the paper)
base_params['IncUnemp']   = IncUnemp       # Induces natural borrowing constraint
base_params['PermShkStd'] = [PermShkStd]   # Standard deviation of log permanent income shocks
base_params['TranShkStd'] = [TranShkStd]   # Standard deviation of log transitory income shocks

# Some technical settings that are not interesting for our purposes
base_params['LivPrb']       = [1.0]   # 100 percent probability of living to next period
base_params['CubicBool']    = True    # Use cubic spline interpolation
base_params['T_cycle']      = 1       # No 'seasonal' cycles
base_params['BoroCnstArt']  = None    # No artificial borrowing constraint
# %% [markdown]
# ## Convergence of the Consumption Rules
#
# Under the given parameter values, [the paper's first figure](https://econ-ark.github.io/BufferStockTheory/#Convergence-of-the-Consumption-Rules) depicts the successive consumption rules that apply in the last period of life $(c_{T}(m))$, the second-to-last period, and earlier periods $(c_{T-n})$.  $c(m)$ is the consumption function to which these converge as 
#
# $$
# c(m) = \lim_{n \uparrow \infty} c_{T-n}(m)
# $$
#

# %% {"code_folding": [0]}
# Create a buffer stock consumer instance by passing the dictionary to the class.
baseEx = IndShockConsumerType(**base_params)
baseEx.cycles = 100   # Make this type have a finite horizon (Set T = 100)

baseEx.solve()        # Solve the model
baseEx.unpack_cFunc()  # Make the consumption function easily accessible




# %% {"code_folding": [0]}
# Plot the different periods' consumption rules.

m1 = np.linspace(0,9.5,1000) # Set the plot range of m
m2 = np.linspace(0,6.5,500)
c_m  = baseEx.cFunc[0](m1)   # c_m can be used to define the limiting inﬁnite-horizon consumption rule here
c_t1 = baseEx.cFunc[-2](m1) # c_t1 defines the second-to-last period consumption rule
c_t5 = baseEx.cFunc[-6](m1) # c_t5 defines the T-5 period consumption rule
c_t10 = baseEx.cFunc[-11](m1)  # c_t10 defines the T-10 period consumption rule
c_t0 = m2                            # c_t0 defines the last period consumption rule
plt.figure(figsize = (12,9))
plt.plot(m1,c_m,color="black")
plt.plot(m1,c_t1,color="black")
plt.plot(m1,c_t5,color="black")
plt.plot(m1,c_t10,color="black")
plt.plot(m2,c_t0,color="black")
plt.xlim(0,11)
plt.ylim(0,7)
plt.text(7,6,r'$c_{T}(m) = 45$ degree line',fontsize = 22,fontweight='bold')
plt.text(9.6,5.3,r'$c_{T-1}(m)$',fontsize = 22,fontweight='bold')
plt.text(9.6,2.6,r'$c_{T-5}(m)$',fontsize = 22,fontweight='bold')
plt.text(9.6,2.1,r'$c_{T-10}(m)$',fontsize = 22,fontweight='bold')
plt.text(9.6,1.7,r'$c(m)$',fontsize = 22,fontweight='bold')
plt.arrow(6.9,6.05,-0.6,0,head_width= 0.1,width=0.001,facecolor='black',length_includes_head='True')
plt.tick_params(labelbottom=False, labelleft=False,left='off',right='off',bottom='off',top='off')
plt.text(0,7.05,"$c$",fontsize = 26)
plt.text(11.1,0,"$m$",fontsize = 26)
# Save the figures in several formats
if Generator:
    plt.savefig(os.path.join(Figures_HARK_dir, 'cFuncsConverge.png'))
    plt.savefig(os.path.join(Figures_HARK_dir, 'cFuncsConverge.jpg'))
    plt.savefig(os.path.join(Figures_HARK_dir, 'cFuncsConverge.pdf'))
    plt.savefig(os.path.join(Figures_HARK_dir, 'cFuncsConverge.svg'))
if not in_ipynb():
    plt.ioff()
    plt.draw()
#    plt.show(block=False) 
    plt.pause(1)
else:
     plt.show(block=True) # Change to False if you want to run uninterrupted
    



# %% [markdown]
# ### PROBLEM: Natural Borrowing Constraint Approaches Artificial Constraint
#
# Show numerically the result that is proven analytically in [The-Liquidity-Constrained-Solution-as-a-Limit](https://econ-ark.github.io/BufferStockTheory/#The-Liquidity-Constrained-Solution-as-a-Limit), by solving the model for successively smaller values of $℘$.
#    * You need only to solve for the second-to-last period of life to do this
#    * You should show the consumption rules for different values of $℘$ on the same graph
#       * To make this easier, you will want to use the plot_funcs command imported by `from HARK.utilities import plot_funcs_der, plot_funcs`
#
# Create a cell or cells in the notebook below this cell and put your solution there

# %% {"code_folding": [0]}
### SOLUTION 

# Turns out that we have to make the probability REALLY small

cFuncList=[] # Create empty list for storing consumption functions

# Construct solution for truly constrained consumer
baseCnst = deepcopy(base_params)
baseCnst['BoroCnstArt'] = 0.0
baseCnst['IncUnemp']    = 0.3
baseCnstEx = IndShockConsumerType(**baseCnst)
baseCnstEx.cycles = 2
baseCnstEx.solve()
baseCnstEx.unpack_cFunc()

# This is the first consumption function in the list 
cFuncList.append(baseCnstEx.cFunc[-2])

# Now set up unconstrained two period solution
TwoPer = IndShockConsumerType(**base_params)
TwoPer.cycles = 2   # Make this type have a two period horizon (Set T = 2)

TwoPer.solve()        # Solve the model under baseline parameter values
TwoPer.unpack_cFunc()  # Make the consumption function easily accessible

# Next consumption function
cFuncList.append(TwoPer.cFunc[-2])

# Now for three other values of unemployment probability
UnempPrbList = [0.001,0.0001,0.00001]

i=0
while i < len(UnempPrbList):
    TwoPerNow=deepcopy(TwoPer)
    TwoPerNow.UnempPrb=UnempPrbList[i]
    TwoPerNow.update_income_process()
    TwoPerNow.solve()
    TwoPerNow.unpack_cFunc()
    cFuncList.append(TwoPerNow.cFunc[-2])
    i += 1

plot_funcs(cFuncList,0.5,1.5)
if not in_ipynb():
    print('Drawing and storing solution')
    plt.ioff()
    plt.draw()
#    plt.show(block=False) 
    plt.pause(1)
else:
    plt.show(block=True) # Change to False if you want to run uninterrupted

# %% [markdown]
# ## Factors and Conditions
#
# ### [The Finite Human Wealth Condition](https://econ-ark.github.io/BufferStockTheory/#Human-Wealth)
#
# Human wealth for a perfect foresight consumer is defined as the present discounted value of future income:
#
# \begin{eqnarray}
# H_{t} & = & \mathbb{E}_{t}[P_{t} + 𝖱^{-1} P_{t+1} + 𝖱^{2} P_{t+2} ... ] \\ 
#       & = & P_{t} \left(1 + (Γ/𝖱) + (Γ/𝖱)^{2} ... \right)
# \end{eqnarray}
# which is an infinite number if $Γ/𝖱 \geq 1$.  We say that the 'Finite Human Wealth Condition' (FHWC) holds if 
# $0 \leq (Γ/𝖱) < 1$.

# %% [markdown]
# ### [Absolute Patience and the AIC](https://econ-ark.github.io/BufferStockTheory/#AIC)
#
# The paper defines the Absolute Patience Factor as being equal to the ratio of $C_{t+1}/C_{t}$ for a perfect foresight consumer.  The Old English character <span style="font-size:larger;">"&#222;"</span> is used for this object in the paper, but <span style="font-size:larger;">"&#222;"</span> cannot currently be rendered conveniently in Jupyter notebooks, so we will substitute $\Phi$ here:
#
# \begin{equation}
# \Phi = (𝖱 β)^{1/ρ} 
# \end{equation}
#
# If $\Phi = 1$, a perfect foresight consumer will spend exactly the amount that can be sustained perpetually (given their current and future resources).  If $\Phi < 1$ (the consumer is 'absolutely impatient'; or, 'the absolute impatience condition holds'), the consumer is consuming more than the sustainable amount, so consumption will fall, and if the consumer is 'absolutely patient' with $\Phi > 1$ consumption will grow over time.
#
#

# %% [markdown]
# ### [Growth Patience and the GICRaw](https://econ-ark.github.io/BufferStockTheory/#GIC)
#
# For a [perfect foresight consumer](https://www.econ2.jhu.edu/people/ccarroll/public/lecturenotes/consumption/PerfForesightCRRA), whether the ratio of consumption to the permanent component of income $P$ is rising, constant, or falling depends on the relative growth rates of consumption and permanent income, which is measured by the "Perfect Foresight Growth Patience Factor":
#
# \begin{eqnarray}
# \Phi_{Γ} & = & \Phi/Γ
# \end{eqnarray}
# and whether the ratio is falling or rising over time depends on whether $\Phi_{Γ}$ is below or above 1.
#
# An analogous condition can be defined when there is uncertainty about permanent income.  Defining $\tilde{Γ} = (\mathbb{E}[ψ^{-1}])^{-1}Γ$, the 'Growth Impatience Condition' (GICRaw) is that 
# \begin{eqnarray}
#   \Phi/\tilde{Γ} & < & 1
# \end{eqnarray}

# %% [markdown]
# ### [The Finite Value of Autarky Condition (FVAC)](https://econ-ark.github.io/BufferStockTheory/#Autarky-Value)


# %% [markdown]
# The paper [shows](https://econ-ark.github.io/BufferStockTheory/#Autarky-Value) that a consumer who planned to spend his permanent income $\{ p_{t}, p_{t+1}, ...\} $ in every period would have value defined by
#
# \begin{equation}
# v_{t}^{\text{autarky}} = u(p_{t})\left(\frac{1}{1-β Γ^{1-ρ} \mathbb{E}[ψ^{1-ρ}]}\right)
# \end{equation}
#
# and defines the 'Finite Value of Autarky Condition' as the requirement that the denominator of this expression be a positive finite number:
#
# \begin{equation}
# β Γ^{1-ρ} \mathbb{E}[ψ^{1-ρ}] < 1
# \end{equation}

# %% [markdown]
# ### [The Weak Return Impatience Condition (WRIC)](http://econ-ark.github.io/BufferStockTheory/#WRIC)
#
# The 'Return Impatience Condition' $\Phi/𝖱 < 1$ has long been understood to be required for the perfect foresight model to have a nondegenerate solution (when $ρ=1$, this reduces to $β < R$).  If the RIC does not hold, the consumer is so patient that the optimal consumption function approaches zero as the horizon extends.
#
# When the probability of unemployment is $℘$, the paper articulates an analogous (but weaker) condition:
#
# \begin{eqnarray}
#  ℘^{1/ρ} \Phi/𝖱 & < & 1
# \end{eqnarray}

# %% [markdown]
# # Key Results
#
# ## [Nondegenerate Solution Requires FVAC and WRIC](https://econ-ark.github.io/BufferStockTheory/#Sufficient-Conditions-For-Nondegenerate-Solution)
#
# A main result of the paper is that the conditions required for the model to have a nondegenerate solution ($0 < c(m) < \infty$ for feasible $m$) are that the Finite Value of Autarky (FVAC) and Weak Return Impatience Condition (WRAC) hold.

# %% [markdown]
# ## [Natural Borrowing Constraint limits to Artificial Borrowing Constraint](http://econ-ark.github.io/BufferStockTheory/#The-Liquidity-Constrained-Solution-as-a-Limit)

# %% [markdown]
# Defining $\chi(℘)$ as the consumption function associated with any particular value of $℘$, and defining $\hat{\chi}$ as the consumption function that would apply in the absence of the zero-income shocks but in the presence of an 'artificial' borrowing constraint requiring $a \geq 0$, a la Deaton (1991), the paper shows that 
#
# \begin{eqnarray}
# \lim_{℘ \downarrow 0}~\chi(℘) & = & \hat{\chi}
# \end{eqnarray}
#
# That is, as $℘$ approaches zero the problem with uncertainty becomes identical to the problem that instead has constraints.  (See [Precautionary Saving and Liquidity Constraints](https://www.econ2.jhu.edu/people/ccarroll/papers/LiqConstr) for a full treatment of the relationship between precautionary saving and liquidity constraints).

# %% [markdown]
# ## [$c(m)$ is Finite Even When Human Wealth Is Infinite](https://econ-ark.github.io/BufferStockTheory/#When-The-GICRaw-Fails)
#
# In the perfect foresight model, if $𝖱 < Γ$ the present discounted value of future labor income is infinite and so the limiting consumption function is $c(m) = \infty$ for all $m$.  Many models have no well-defined solution in this case.
#
# The presence of uncertainty changes this: The limiting consumption function is finite for all values of $m$.  
#
# This is because uncertainty imposes a "natural borrowing constraint" that deters the consumer from borrowing against their unbounded future labor income.

# %% [markdown]
# A [table](https://econ-ark.github.io/BufferStockTheory/#Sufficient-Conditions-For-Nondegenerate-Solution) puts this result in the context of implications of other conditions and restrictions.
#
#

# %% [markdown]
# ## [If the GICRaw Holds, $\exists$ a finite 'target' $m$](https://econ-ark.github.io/BufferStockTheory/#onetarget)
#
# Section [There Is Exactly One Trg $m$ Ratio, Which Is Stable](https://econ-ark.github.io/BufferStockTheory/#onetarget) shows that, under parameter values for which the limiting consumption function exists, if the GICRaw holds then there will be a value $\check{m}$ such that:
#
# \begin{eqnarray}
# \mathbb{E}[m_{t+1}] & > & m_{t}~\text{if $m_{t} < \check{m}$} \\
# \mathbb{E}[m_{t+1}] & < & m_{t}~\text{if $m_{t} > \check{m}$} \\
# \mathbb{E}[m_{t+1}] & = & m_{t}~\text{if $m_{t} = \check{m}$}
# \end{eqnarray} 

# %% [markdown]
# ## [If the GICRaw Fails, Trg Wealth is Infinite ](https://econ-ark.github.io/BufferStockTheory/#The-GIC)
#
# [A figure](https://econ-ark.github.io/BufferStockTheory/#FVACnotGICRaw) depicts a solution when the **FVAC** (Finite Value of Autarky Condition) and **WRIC** hold (so that the model has a solution) but the **GIC** (Growth Impatience Condition) fails.  In this case the target wealth ratio is infinity.  
#
# The parameter values in this specific example are:
#
# | Param | Description | Code | Value |
# | :---: | ---         | ---  | :---: |
# | $Γ$ | Permanent Income Growth Factor | $\texttt{PermGroFac}$ | 1.00 |
# | $\mathrm{𝖱}$ | Interest Factor | $\texttt{Rfree}$ | 1.08 |
#
# The figure is reproduced below.

# %% {"code_folding": [0]}
# Construct the "GICRaw fails" example.

GICRaw_fail_dictionary = dict(base_params)
GICRaw_fail_dictionary['Rfree']      = 1.08
GICRaw_fail_dictionary['PermGroFac'] = [1.00]

GICRawFailExample = IndShockConsumerType(
    cycles=0, # cycles=0 makes this an infinite horizon consumer
    **GICRaw_fail_dictionary)


# %% [markdown]
# The $\mathtt{IndShockConsumerType}$ tool automatically checks various parametric conditions, and will give a warning as well as the values of the factors if any conditions fail to be met. 
#
# We can also directly check the conditions, in which case results will be a little more verbose by default.

# %% {"code_folding": [0]}
# The check_conditions method does what it sounds like it would
GICRawFailExample.check_conditions(verbose=True)

# %% [markdown]
# Next we define the function $\mathrm{\mathbb{E}}_{t}[\Delta m_{t+1}]$ that shows the ‘sustainable’ level of spending at which $m$ is expected to remain unchanged.

# %% {"code_folding": [0]}
# Calculate "Sustainable" consumption that leaves expected m unchanged
# In the perfect foresight case, this is just permanent income plus interest income
# A small adjustment is required to take account of the consequences of uncertainty
InvEpShInvAct = np.dot(GICRawFailExample.PermShkDstn[0][0], GICFailExample.PermShkDstn[0][1]**(-1))
InvInvEpShInvAct = (InvEpShInvAct) ** (-1)
PermGroFacAct = GICRawFailExample.PermGroFac[0] * InvInvEpShInvAct
ER = GICRawFailExample.Rfree / PermGroFacAct
Er = ER - 1
mSSfunc = lambda m : 1 + (m-1)*(Er/ER)


# %% [markdown] {"heading_collapsed": true}
# ### PROBLEM: Trg Wealth and Impatience
#
# Given the above results proving that there is a finite target $m$ if the Growth Impatience Factor is less than one ('the GICRaw holds'), but that $m \uparrow \infty$ if the GICRaw fails, it will not surprise you to learn that the target approaches infinity as the GIF approaches 1.
#
# This problem asks you to verify that proposition by calculating and plotting values of the GIF for a sequence of $β$ values that cause the GIF to be closer and closer to 1.
#
# In doing this, it will be useful to know that the GIF is calculated in the $\texttt{check_conditions}$ method.  (Look in $\texttt{ConsIndShockModel.py}$).

# %% [markdown]
# ### SOLUTION - Analysis
#
# Defining $\tilde{Γ} = (\mathbb{E}[ψ^{-1}])^{-1}Γ$, substituting for $\Phi$ in the formula above, the 'Growth Impatience Condition' (GICRaw) is that 
#
# \begin{eqnarray}
#   (𝖱 β)^{1/ρ} & < & \tilde{Γ} \\
#   (𝖱 β) & < & \tilde{Γ}^{ρ} \\
#   β & < & \tilde{Γ}^{ρ}/𝖱
# \end{eqnarray}
# so our task is to calculate the value of targe wealth $\check{m}$ for a sequence of $β$ values that approach the number on the RHS from below.

# %% {"code_folding": []}
# ## SOLUTION -- Code

# First calculate the expectation of the inverse of the permanent shock
Ex_PermShkInv=np.dot(baseEx.PermShkDstn[0][0],1/baseEx.PermShkDstn[0][1])

# Now calculate the adjusted permanent growth factor
PermGroFacAdj=baseEx.PermGroFac[0]*Ex_PermShkInv

# Calculate the upper bound of the discount factor
DiscFacLim = (PermGroFacAdj**baseEx.CRRA)/𝖱free

# Some extra calculations
Thorn=baseEx.LivPrb[0]*(baseEx.Rfree*baseEx.DiscFac)**(1/baseEx.CRRA)
GIF=Thorn/PermGroFacAdj

# Impatience rate is the log difference between baseline and limiting DiscFac
discRteGap = np.log(baseEx.DiscFac/DiscFacLim)

ldiscRteList = np.linspace(np.log(-discRteGap),np.log(-discRteGap/50),20)
discRteList = -np.exp(ldiscRteList)
DiscFacList = []
mTargList = []
for discRteLoop in np.nditer(discRteList):
    DiscFacNow=np.exp(discRteLoop)
    discEx=deepcopy(baseEx)
    discEx.DiscFac=DiscFacNow
    discEx.solve()
    mTargList.append(discEx.solution[0].mNrmStE)
    DiscFacList.append(DiscFacNow)
    print("DiscFac="+str(DiscFacList[-1]),end='\n')

plt.figure(figsize = (12,9))
plt.plot(DiscFacList,mTargList,color="black")
if not in_ipynb():
    plt.show(block=False)
    plt.pause(1)
else:
    plt.show(block=True) # Change to False if you want to run uninterrupted

# %% {"code_folding": [0]}
# Plot GICRawFailExample consumption function against the sustainable level of consumption

GICRawFailExample.solve() # Above, we set up the problem but did not solve it 
GICRawFailExample.unpack_cFunc()  # Make the consumption function easily accessible for plotting
m = np.linspace(0,5,1000)
c_m = GICRawFailExample.cFunc[0](m)
E_m = mSSfunc(m)
plt.figure(figsize = (12,8))
plt.plot(m,c_m,color="black")
plt.plot(m,E_m,color="black")
plt.xlim(0,5.5)
plt.ylim(0,1.6)
plt.text(0,1.63,"$c$",fontsize = 26)
plt.text(5.55,0,"$m$",fontsize = 26)
plt.tick_params(labelbottom=False, labelleft=False,left='off',right='off',bottom='off',top='off')
plt.text(1,0.6,"$c(m_{t})$",fontsize = 18)
plt.text(1.5,1.2,"$\mathsf{E}_{t}[\Delta m_{t+1}] = 0$",fontsize = 18)
plt.arrow(0.98,0.62,-0.2,0,head_width= 0.02,width=0.001,facecolor='black',length_includes_head='True')
plt.arrow(2.2,1.2,0.3,-0.05,head_width= 0.02,width=0.001,facecolor='black',length_includes_head='True')
if Generator:
    plt.savefig(os.path.join(Figures_HARK_dir, 'FVACnotGICRaw.png'))
    plt.savefig(os.path.join(Figures_HARK_dir, 'FVACnotGICRaw.jpg'))
    plt.savefig(os.path.join(Figures_HARK_dir, 'FVACnotGICRaw.pdf'))
    plt.savefig(os.path.join(Figures_HARK_dir, 'FVACnotGICRaw.svg'))

# This figure reproduces the figure shown in the paper.  
# The gap between the two functions actually increases with $m$ in the limit.
if not in_ipynb():
    plt.show(block=False) 
    plt.pause(1)
else:
    plt.show(block=True) # Change to False if you want to run uninterrupted

# %% [markdown]
# As a foundation for the remaining figures, we define another instance of the class $\texttt{IndShockConsumerType}$, which has the same parameter values as the instance $\texttt{baseEx}$ defined previously but is solved to convergence (our definition of an infinite horizon agent type)
#

# %% {"code_folding": [0]}
# cycles=0 tells the solver to find the infinite horizon solution
baseEx_inf = IndShockConsumerType(cycles=0,**base_params)

baseEx_inf.solve()
baseEx_inf.unpack_cFunc()

# %% [markdown]
# ### [Trg $m$, Expected Consumption Growth, and Permanent Income Growth](https://econ-ark.github.io/BufferStockTheory/#AnalysisoftheConvergedConsumptionFunction)
#
# The next figure is shown in  [Analysis of the Converged Consumption Function](https://econ-ark.github.io/BufferStockTheory/#cGroTrgFig), which shows the expected consumption growth factor $\mathrm{\mathbb{E}}_{t}[c_{t+1}/c_{t}]$ for a consumer behaving according to the converged consumption rule.
#

# %% {"code_folding": [0]}
# Define a function to calculate expected consumption 
def exp_consumption(a):
    '''
    Taking end-of-period assets as input, return expectation of next period's consumption
    Inputs:
       a: end-of-period assets
    Returns:
       expconsump: next period's expected consumption
    '''
    GrowFactp1 = baseEx_inf.PermGroFac[0]* baseEx_inf.PermShkDstn[0][1]
    Rnrmtp1 = baseEx_inf.Rfree / GrowFactp1
    # end-of-period assets plus normalized returns
    btp1 = Rnrmtp1*a
    # expand dims of btp1 and use broadcasted sum of a column and a row vector
    # to obtain a matrix of possible beginning-of-period assets next period
    mtp1 = np.expand_dims(btp1, axis=1) + baseEx_inf.TranShkDstn[0][1]
    part_expconsumption = GrowFactp1*baseEx_inf.cFunc[0](mtp1).T
    # finish expectation over permanent income shocks by right multiplying with
    # the weights
    part_expconsumption = np.dot(part_expconsumption, baseEx_inf.PermShkDstn[0][0])
    # finish expectation over transitory income shocks by right multiplying with
    # weights
    expconsumption = np.dot(part_expconsumption, baseEx_inf.TranShkDstn[0][0])
    # return expected consumption
    return expconsumption


# %% {"code_folding": [0]}
# Calculate the expected consumption growth factor
m1 = np.linspace(1,baseEx_inf.solution[0].mNrmStE,50) # m1 defines the plot range on the left of target m value (e.g. m <= target m)
c_m1 = baseEx_inf.cFunc[0](m1)
a1 = m1-c_m1
exp_consumption_l1 = []
for i in range(len(a1)):
    exp_consumption_tp1 = exp_consumption(a1[i])
    exp_consumption_l1.append(exp_consumption_tp1)

# growth1 defines the values of expected consumption growth factor when m is less than target m
growth1 = np.array(exp_consumption_l1)/c_m1

# m2 defines the plot range on the right of target m value (e.g. m >= target m)
m2 = np.linspace(baseEx_inf.solution[0].mNrmStE,1.9,50)

c_m2 = baseEx_inf.cFunc[0](m2)
a2 = m2-c_m2
exp_consumption_l2 = []
for i in range(len(a2)):
    exp_consumption_tp1 = exp_consumption(a2[i])
    exp_consumption_l2.append(exp_consumption_tp1)

# growth 2 defines the values of expected consumption growth factor when m is bigger than target m
growth2 = np.array(exp_consumption_l2)/c_m2


# %% {"code_folding": [0]}
# Define a function to construct the arrows on the consumption growth rate function
def arrowplot(axes, x, y, narrs=15, dspace=0.5, direc='neg',
              hl=0.01, hw=3, c='black'):
    '''
    The function is used to plot arrows given the data x and y.

    Input:
        narrs  :  Number of arrows that will be drawn along the curve

        dspace :  Shift the position of the arrows along the curve.
                  Should be between 0. and 1.

        direc  :  can be 'pos' or 'neg' to select direction of the arrows

        hl     :  length of the arrow head

        hw     :  width of the arrow head

        c      :  color of the edge and face of the arrow head
    '''

    # r is the distance spanned between pairs of points
    r = np.sqrt(np.diff(x)**2+np.diff(y)**2)
    r = np.insert(r, 0, 0.0)

    # rtot is a cumulative sum of r, it's used to save time
    rtot = np.cumsum(r)

    # based on narrs set the arrow spacing
    aspace = r.sum() / narrs

    if direc is 'neg':
        dspace = -1.*abs(dspace)
    else:
        dspace = abs(dspace)

    arrowData = [] # will hold tuples of x,y,theta for each arrow
    arrowPos = aspace*(dspace) # current point on walk along data
                                 # could set arrowPos to 0 if you want
                                 # an arrow at the beginning of the curve

    ndrawn = 0
    rcount = 1
    while arrowPos < r.sum() and ndrawn < narrs:
        x1,x2 = x[rcount-1],x[rcount]
        y1,y2 = y[rcount-1],y[rcount]
        da = arrowPos-rtot[rcount]
        theta = np.arctan2((x2-x1),(y2-y1))
        ax = np.sin(theta)*da+x1
        ay = np.cos(theta)*da+y1
        arrowData.append((ax,ay,theta))
        ndrawn += 1
        arrowPos+=aspace
        while arrowPos > rtot[rcount+1]:
            rcount+=1
            if arrowPos > rtot[-1]:
                break

    for ax,ay,theta in arrowData:
        # use aspace as a guide for size and length of things
        # scaling factors were chosen by experimenting a bit

        dx0 = np.sin(theta)*hl/2.0 + ax
        dy0 = np.cos(theta)*hl/2.0 + ay
        dx1 = -1.*np.sin(theta)*hl/2.0 + ax
        dy1 = -1.*np.cos(theta)*hl/2.0 + ay

        if direc is 'neg' :
            ax0 = dx0
            ay0 = dy0
            ax1 = dx1
            ay1 = dy1
        else:
            ax0 = dx1
            ay0 = dy1
            ax1 = dx0
            ay1 = dy0

        axes.annotate('', xy=(ax0, ay0), xycoords='data',
                xytext=(ax1, ay1), textcoords='data',
                arrowprops=dict( headwidth=hw, frac=1., ec=c, fc=c))


# %% {"code_folding": [0]}
# Plot consumption growth as a function of market resources
# Calculate Absolute Patience Factor Phi = lower bound of consumption growth factor
AbsPatientFac = (baseEx_inf.Rfree*baseEx_inf.DiscFac)**(1.0/baseEx_inf.CRRA)

fig = plt.figure(figsize = (12,8))
ax = fig.add_subplot(111)
# Plot the Absolute Patience Factor line
ax.plot([0,1.9],[AbsPatientFac,AbsPatientFac],color="black")

# Plot the Permanent Income Growth Factor line
ax.plot([0,1.9],[baseEx_inf.PermGroFac[0],baseEx_inf.PermGroFac[0]],color="black")

# Plot the expected consumption growth factor on the left side of target m
ax.plot(m1,growth1,color="black")

# Plot the expected consumption growth factor on the right side of target m
ax.plot(m2,growth2,color="black")

# Plot the arrows
arrowplot(ax, m1,growth1)
arrowplot(ax, m2,growth2, direc='pos')

# Plot the target m
ax.plot([baseEx_inf.solution[0].mNrmStE,baseEx_inf.solution[0].mNrmStE],[0,1.4],color="black",linestyle="--")
ax.set_xlim(1,2.05)
ax.set_ylim(0.98,1.08)
ax.text(1,1.082,"Growth Rate",fontsize = 26,fontweight='bold')
ax.text(2.055,0.98,"$m_{t}$",fontsize = 26,fontweight='bold')
ax.text(1.9,1.01,"$\mathsf{E}_{t}[c_{t+1}/c_{t}]$",fontsize = 22,fontweight='bold')
ax.text(baseEx_inf.solution[0].mNrmStE,0.975, r'$\check{m}$', fontsize = 26,fontweight='bold')
ax.tick_params(labelbottom=False, labelleft=False,left='off',right='off',bottom='off',top='off')
ax.text(1.9,0.998,r'$\Phi = (\mathrm{𝖱}β)^{1/ρ}$',fontsize = 22,fontweight='bold')
ax.text(1.9,1.03, r'$Γ$',fontsize = 22,fontweight='bold')
if Generator:
    fig.savefig(os.path.join(Figures_HARK_dir, 'cGroTrgFig.png'))
    fig.savefig(os.path.join(Figures_HARK_dir, 'cGroTrgFig.jpg'))
    fig.savefig(os.path.join(Figures_HARK_dir, 'cGroTrgFig.pdf'))
    fig.savefig(os.path.join(Figures_HARK_dir, 'cGroTrgFig.svg'))
if not in_ipynb():
    plt.show(block=False) 
    plt.pause(1)
else:
    plt.show(block=True) # Change to False if you want to run uninterrupted

# %% [markdown] {"code_folding": []}
# ### [Consumption Function Bounds](https://econ-ark.github.io/BufferStockTheory/#AnalysisOfTheConvergedConsumptionFunction)
# [The next figure](https://econ-ark.github.io/BufferStockTheory/#cFuncBounds)
# illustrates theoretical bounds for the consumption function.
#
# We define two useful variables: lower bound of $κ$ (marginal propensity to consume) and limit of $h$ (Human wealth), along with some functions such as limiting perfect foresight consumption functions ($\bar{c}(m)$), $\bar{\bar c}(m)$ and $\underline{c}(m)$.

# %% {"code_folding": []}
# Define k_lower, h_inf and perfect foresight consumption function, upper bound of consumption function and lower
# bound of consumption function.
k_lower = 1.0-(baseEx_inf.Rfree**(-1.0))*(baseEx_inf.Rfree*baseEx_inf.DiscFac)**(1.0/baseEx_inf.CRRA)
h_inf = (1.0/(1.0-baseEx_inf.PermGroFac[0]/baseEx_inf.Rfree))
conFunc_PF = lambda m: (h_inf -1)* k_lower + k_lower*m
conFunc_upper = lambda m: (1 - baseEx_inf.UnempPrb ** (1.0/baseEx_inf.CRRA)*(baseEx_inf.Rfree*baseEx_inf.DiscFac)**(1.0/baseEx_inf.CRRA)/baseEx_inf.Rfree)*m
conFunc_lower = lambda m: (1 -(baseEx_inf.Rfree*baseEx_inf.DiscFac)**(1.0/baseEx_inf.CRRA)/baseEx_inf.Rfree) * m
intersect_m = ((h_inf-1)* k_lower)/((1 - baseEx_inf.UnempPrb
            **(1.0/baseEx_inf.CRRA)*(baseEx_inf.Rfree*baseEx_inf.DiscFac)**(1.0/baseEx_inf.CRRA)/baseEx_inf.Rfree)-k_lower)


# %% {"code_folding": []}
# Plot the consumption function and its bounds

x1 = np.linspace(0,25,1000)
x3 = np.linspace(0,intersect_m,300)
x4 = np.linspace(intersect_m,25,700)
cfunc_m = baseEx_inf.cFunc[0](x1)
cfunc_PF_1 = conFunc_PF(x3)
cfunc_PF_2 = conFunc_PF(x4)
cfunc_upper_1 = conFunc_upper(x3)
cfunc_upper_2 = conFunc_upper(x4)
cfunc_lower = conFunc_lower(x1)
plt.figure(figsize = (12,8))
plt.plot(x1,cfunc_m, color="black")
plt.plot(x1,cfunc_lower, color="black",linewidth=2.5)
plt.plot(x3,cfunc_upper_1, color="black",linewidth=2.5)
plt.plot(x4,cfunc_PF_2 , color="black",linewidth=2.5)
plt.plot(x4,cfunc_upper_2 , color="black",linestyle="--")
plt.plot(x3,cfunc_PF_1 , color="black",linestyle="--")
plt.tick_params(labelbottom=False, labelleft=False,left='off',right='off',bottom='off',top='off')
plt.xlim(0,25)
plt.ylim(0,1.12*conFunc_PF(25))
plt.text(0,1.12*conFunc_PF(25)+0.05,"$c$",fontsize = 22)
plt.text(25+0.1,0,"$m$",fontsize = 22)
plt.text(2.5,1,r'$c(m)$',fontsize = 22,fontweight='bold')
plt.text(6,5,r'$\overline{\overline{c}}(m)= \overline{κ}m = (1-℘^{1/ρ}\Phi_{R})m$',fontsize = 22,fontweight='bold')
plt.text(2.2,3.8, r'$\overline{c}(m) = (m-1+h)\underline{κ}$',fontsize = 22,fontweight='bold')
plt.text(9,4.1,r'Upper Bound $ = $ Min $[\overline{\overline{c}}(m),\overline{c}(m)]$',fontsize = 22,fontweight='bold')
plt.text(7,0.7,r'$\underline{c}(m)= (1-\Phi_{R})m = \underline{κ}m$',fontsize = 22,fontweight='bold')
plt.arrow(2.45,1.05,-0.5,0.02,head_width= 0.05,width=0.001,facecolor='black',length_includes_head='True')
plt.arrow(2.15,3.88,-0.5,0.1,head_width= 0.05,width=0.001,facecolor='black',length_includes_head='True')
plt.arrow(8.95,4.15,-0.8,0.05,head_width= 0.05,width=0.001,facecolor='black',length_includes_head='True')
plt.arrow(5.95,5.05,-0.4,0,head_width= 0.05,width=0.001,facecolor='black',length_includes_head='True')
plt.arrow(14,0.70,0.5,-0.1,head_width= 0.05,width=0.001,facecolor='black',length_includes_head='True')
if Generator:
    plt.savefig(os.path.join(Figures_HARK_dir, 'cFuncBounds.png'))
    plt.savefig(os.path.join(Figures_HARK_dir, 'cFuncBounds.jpg'))
    plt.savefig(os.path.join(Figures_HARK_dir, 'cFuncBounds.pdf'))
    plt.savefig(os.path.join(Figures_HARK_dir, 'cFuncBounds.svg'))
if not in_ipynb():
    plt.show(block=False) 
    plt.pause(1)
else:
    plt.show(block=True) # Change to False if you want to run uninterrupted

# %% [markdown]
# ### [The Consumption Function and Trg $m$](https://econ-ark.github.io/BufferStockTheory/#cFuncBounds)
#
# This figure shows the $\mathrm{\mathbb{E}}_{t}[\Delta m_{t+1}]$ and consumption function $c(m_{t})$, along with the intrsection of these two functions, which defines the target value of $m$

# %% {"code_folding": [0]}
# This just plots objects that have already been constructed

m1 = np.linspace(0,4,1000)
cfunc_m = baseEx_inf.cFunc[0](m1)
mSSfunc = lambda m:(baseEx_inf.PermGroFac[0]/baseEx_inf.Rfree)+(1.0-baseEx_inf.PermGroFac[0]/baseEx_inf.Rfree)*m
mss = mSSfunc(m1)
plt.figure(figsize = (12,8))
plt.plot(m1,cfunc_m, color="black")
plt.plot(m1,mss, color="black")
plt.xlim(0,3)
plt.ylim(0,1.45)
plt.plot([baseEx_inf.solution[0].mNrmStE, baseEx_inf.solution[0].mNrmStE],[0,2.5],color="black",linestyle="--")
plt.tick_params(labelbottom=False, labelleft=False,left='off',right='off',bottom='off',top='off')
plt.text(0,1.47,r"$c$",fontsize = 26)
plt.text(3.02,0,r"$m$",fontsize = 26)
plt.text(2.3,0.95,r'$\mathsf{E}[\Delta m_{t+1}] = 0$',fontsize = 22,fontweight='bold')
plt.text(2.3,1.1,r"$c(m_{t})$",fontsize = 22,fontweight='bold')
plt.text(baseEx_inf.solution[0].mNrmStE,-0.05, r"$\check{m}$",fontsize = 26)
plt.arrow(2.28,1.12,-0.1,0.03,head_width= 0.02,width=0.001,facecolor='black',length_includes_head='True')
plt.arrow(2.28,0.97,-0.1,0.02,head_width= 0.02,width=0.001,facecolor='black',length_includes_head='True')
if Generator:
    plt.savefig(os.path.join(Figures_HARK_dir, 'cNrmTrgFig.png'))
    plt.savefig(os.path.join(Figures_HARK_dir, 'cNrmTrgFig.jpg'))
    plt.savefig(os.path.join(Figures_HARK_dir, 'cNrmTrgFig.pdf'))
    plt.savefig(os.path.join(Figures_HARK_dir, 'cNrmTrgFig.svg'))
if not in_ipynb():
    plt.show(block=False)
    plt.pause(1)
else:
    plt.show(block=True)

# %% [markdown]
# ### [Upper and Lower Limits of the Marginal Propensity to Consume](https://econ-ark.github.io/BufferStockTheory/#MPCLimits)
#
# The paper shows that as $m_{t}~\uparrow~\infty$ the consumption function in the presence of risk gets arbitrarily close to the perfect foresight consumption function.  Defining $\underline{κ}$ as the perfect foresight model's MPC, this implies that $\lim_{m_{t}~\uparrow~\infty} c^{\prime}(m) = \underline{κ}$.  
#
# The paper also derives an analytical limit $\bar{κ}$ for the MPC as $m$ approaches 0., its bounding value.  Strict concavity of the consumption function implies that the consumption function will be everywhere below a function $\bar{κ}m$, and strictly declining everywhere.  The last figure plots the MPC between these two limits.

# %% {"code_folding": [0]}
# The last figure shows the upper and lower limits of the MPC
plt.figure(figsize = (12,8))
# Set the plot range of m
m = np.linspace(0.001,8,1000)

# Use the HARK method derivative to get the derivative of cFunc, and the values are just the MPC
MPC = baseEx_inf.cFunc[0].derivative(m)

# Define the upper bound of MPC
MPCUpper = (1 - baseEx_inf.UnempPrb ** (1.0/baseEx_inf.CRRA)*(baseEx_inf.Rfree*baseEx_inf.DiscFac)**(1.0/baseEx_inf.CRRA)/baseEx_inf.Rfree)

# Define the lower bound of MPC
MPCLower = k_lower

plt.plot(m,MPC,color = 'black')
plt.plot([0,8],[MPCUpper,MPCUpper],color = 'black')
plt.plot([0,8],[MPCLower,MPCLower],color = 'black')
plt.xlim(0,8)
plt.ylim(0,1)
plt.text(1.5,0.6,r'$κ(m) \equiv c^{\prime}(m)$',fontsize = 26,fontweight='bold')
plt.text(6,0.87,r'$(1-℘^{1/ρ}\Phi_{R})\equiv \overline{κ}$',fontsize = 26,fontweight='bold')
plt.text(0.5,0.07,r'$\underline{κ}\equiv(1-\Phi_{R})$',fontsize = 26,fontweight='bold')
plt.text(8.05,0,"$m$",fontsize = 26)
plt.arrow(1.45,0.61,-0.4,0,head_width= 0.02,width=0.001,facecolor='black',length_includes_head='True')
plt.arrow(1.7,0.07,0.2,-0.01,head_width= 0.02,width=0.001,facecolor='black',length_includes_head='True')
plt.arrow(5.95,0.875,-0.2,0.03,head_width= 0.02,width=0.001,facecolor='black',length_includes_head='True')
if Generator:
    plt.savefig(os.path.join(Figures_HARK_dir, 'MPCLimits.png'))
    plt.savefig(os.path.join(Figures_HARK_dir, 'MPCLimits.jpg'))
    plt.savefig(os.path.join(Figures_HARK_dir, 'MPCLimits.pdf'))
    plt.savefig(os.path.join(Figures_HARK_dir, 'MPCLimits.svg'))
if not in_ipynb():
    plt.show(block=False) 
    plt.pause(1)
else:
    plt.show(block=True) # Change to False if you want to run uninterrupted

# %% [markdown]
# # Summary
#
# [Two tables in the paper](https://econ-ark.github.io/BufferStockTheory/#Sufficient-Conditions-For-Nondegenerate-Solution) summarize the various definitions, and then articulate conditions required for the problem to have a nondegenerate solution.
#
# The main other contribution of the paper is to show that, under parametric combinations where the solution is nondegenerate, if the Growth Impatience Condition holds there will be a target level of wealth.
