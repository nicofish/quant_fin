- https://www.github.com/nicofish/quant_fin
- Date : 05 Jan 2021 
- Quantitative Finance : pricing of financial derivatives products
- Python : v 3.7.7
- Author : NicoFish

= option_pricing.py =
allows pricing of vanilla option with several models :
  - Black & Sholes formula  : european option, call and put, price and delta
  - Monte Carlo simulation  : european option, call and put, price
  - Binomial Tree model (*) : european option and american option, call and put, price
    (*) also know as Cox Ross and Rubinstein (CRR) model

= cash_flow.py =
contains modelisation of cash flows (date, currency, amount)
