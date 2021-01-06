**Repository** : https://www.github.com/nicofish/quant_fin  
**Date**       : 06 Jan 2021   
**Object**     : Quantitative Finance - pricing of financial derivatives products  
**Python**     : v 3.7.7  
**Author**     : NicoFish  

### option_pricing.py
- option_pricing.py allows pricing of vanilla option with several models :
  - Black & Sholes formula  : european option, call and put, price and delta
  - Monte Carlo simulation  : european option, call and put, price
  - Binomial Tree model (+) : european option and american option, call and put, price
    (+) also know as Cox Ross and Rubinstein (CRR) model

### test_option_pricing.py
- unitary test for the option pricing module

### rate_convention.py
- rate convention module offers computation of discount factror, capitalisation factor, rate conversion
- rate convention handled are : exponential, yield, linear and discount

### cash_flow.py
- cash_flow.py contains modelisation of cash flows (date, currency, amount)
