**Repository** : https://www.github.com/nicofish/quant_fin  
**Date**       : 08 Jan 2021   
**Object**     : Quantitative Finance - pricing of financial derivatives products  
**Python**     : v 3.7.7  
**Author**     : NicoFish  

### option_pricing.py
- option_pricing.py offers pricing of vanilla option with several models :
  - Black & Sholes formula
    - Scope (option) : european, call & put
    - Output : price, analytical & simulated greeks (delta, gamma, vega, vanna, volga, theta, rho, rho_f)
  - Binomial Tree model (aka Cox Ross and Rubinstein (CRR) model)
    - Scope (option) : european & american, call and put
    - Output : price, simulated greeks (delta, gamma, vega, vanna, volga, theta, rho, rho_f)
  - Monte Carlo simulation 
    - Scope (option) : european, call & put, option
    - Output : price
- Note :
  - analytical greek = computation by closed formula / simulated greek = computation by bumping the parameter(s)
  - greeks : delta = dP/dS, gamma = d²P/dS², vega = dP/dvol, vanna = d²P/dS/dvol, volga = d²P/dvol², theta = dP/dt, rho = dP/dr, rho_f = dP/dr_f

### test_option_pricing.py
- unitary test for the option pricing module

### rate_convention.py
- rate convention module offers computation of discount factor, capitalisation factor, rate conversion
- rate convention handled : exponential, yield, linear and discount

### fixed_income.py
- bond modelisation with computation of cash flows, market value, pv01, yield, macaulay duration and modified duration

### cash_flow.py
- cash_flow.py contains modelisation of cash flows (date, currency, amount)
