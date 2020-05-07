# MIE1622-Computational-Finance-and-Risk-Management
The goal of this course is introduce the concepts of constructions of computational algorithms for soliving financial problems, such as risk-aware decision-making, asset pricing, portfolio optimization and hedging. Considerable attention is deveoted to the application of computational and programming techniques to financial, investment and risk management problems. Jupyter notebook (Python IDE) is the primary computational and modeling software used in this course. <br>

**Assingment 1: Mean-Variance-Portfolio-Selection-Strategies** <br>
The purpose of this assignment is to compare computational investment strategies based on minimizing portfolio variance and maximizing Sharpe ratio. The classical Markowitz model is applied for the portfolio optimization. The goal of optimization effort is to create a tool that allows the user to make regular decisions about re-balancing their portfolio and compare different investment strategies. The factors require optimization includes the total return, the risk and Sharpe ratio. The following 4 strategies are tested and compared:
1. Buy and Hold strategy
2. Equally weighted portfolio strategy
3. Minimum variance portfolio strategy
4. Maximum Sharpe ratio portfolio strategy

**Assingment 2: Risk-Based and Robust Portfolio Selection Strategies** <br>
The purpose of this assignment is to compare computational investment strategies based on selecting portfolio with equal risk contributions and using robust mean-variance optimization. The following 7 strategies, including the 4 strategies implemented in assignment 1, are tested and compared:
1. Buy and Hold strategy
2. Equally weighted portfolio strategy
3. Minimum variance portfolio strategy
4. Maximum Sharpe ratio portfolio strategy
5. Equal risk contributions portfolio strategy
6. Leveraged equal risk contributions portfolio strategy
7. Robust mean-variance optimization portfolio strategy

**Assignment 3: Credit Risk Modeling and Simulation** <br>
The purpose of this assignment is to model a credit-risky portfolio of corporate bonds. Using the data for 100 counterparties, 1-year loss for each corporate bond has been simulated. The following 3 sets of scenarios has been generated:
 - Monte Carlo approximations 1: 5000 in-sample scenarios (1000 systemic scenarios and 5 idiosyncratic scenarios for each systemic), non-Normal distribution of losses
  - Monte Carlo approximations 2: 5000 in-sample scenarios (5000 systemic scenarios and 1 idiosyncratic scenarios for each systemic), non-Normal distribution of losses
   - True distribution: 100,000 out-of-sample scenarios (100,000 systemic scenarios and 1 idiosyncratic scenarios for each systemic), non-Normal distribution of losses

Two kinds of bond portfolios are created and their corresponding VaR (value at risk) and CVaR (Conditional Value at risk) are evaluated.
1. One unit invested in each of 100 bonds
2. Equal value (dollar amount) is invested in each of 100 bonds

**Assignment 4: Asset Pricing** <br>
The purpose of this assignment is to evaluate the value of an European option and a Barrier option through different means.
1. One-step Monte Carlo simulations using Geometric Brownian Motion (GBM) with constant drift and volatility.
2. Multi-step Monte Carlo simulations using Geometric Brownian Motion (GBM) with constant drift and volatility.
3. Black-Scholes equation for the price of european option
