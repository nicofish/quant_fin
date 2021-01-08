**Repository** : https://www.github.com/nicofish/quant_fin  
**Date**       : 08 Jan 2021   
**Object**     : Quantitative Finance - pricing of financial derivatives products  
**Python**     : v 3.7.7  
**Author**     : NicoFish  

### option_pricing.py
- option_pricing.py offers pricing of vanilla option with several models :
  - Black & Sholes formula  :
    - Scope : european option, call and put
    - Output : price, analytical and simulated delta, gamma, vega, vanna, volga, theta, rho, rho_f
  - Binomial Tree model (aka Cox Ross and Rubinstein (CRR) model)
    - Scope : european option and american option, call and put
    - Output : price, simulated delta, gamma, vega, vanna, volga, theta, rho, rho_f
  - Monte Carlo simulation  :
    - Scope : european option, call and put
    - Output : price
- Note : analytical means computation by closed formula and simulated means computation by bumping the parameter

### test_option_pricing.py
- unitary test for the option pricing module

### rate_convention.py
- rate convention module offers computation of discount factor, capitalisation factor, rate conversion
- rate convention handled are : exponential, yield, linear and discount

### fixed_income.py
- bond modelisation with computation of cash flows, market value, pv01, yield, macaulay duration and modified duration

### cash_flow.py
- cash_flow.py contains modelisation of cash flows (date, currency, amount)
