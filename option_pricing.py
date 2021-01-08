# Object  : Pricing vanilla call and put, european and american option
#           with - Black & Sholes (closed) formula
#                - Monte Carlo simulation
#                - Binomial Tree (Cox Ross and Rubinstein : CRR) model
# Version : v1.3
# Date    : 2021.01.07
# Python  : v3.7.7
# Author  : NicoFish

"""
Release Note

v1.3 (07 Jan 2021)
- In short, analytical greeks for Black & Sholes and simulated greeks for Black & Sholed and Binomial Tree
- Pricing call and put european option :
  -- Black & Sholes (closed) formula : call/put, european, analytical gamma/vega/vanna/volga/theta/rho_d/rho_f/pay_off
  -- Binomial Tree (Cox Ross and Rubinstein : CRR) model : call/put, european/american, simulated delta/gamma/vega/vanna/volga/theta/rho_d/rho_f

v1.2 (06 Jan 2021)
- Pricing call and put european option :
  -- Black & Sholes (closed) formula : call/put, european, analytical price
  -- Monte Carlo simulation : call/put, european, price 
  -- Binomial Tree (Cox Ross and Rubinstein : CRR) model : call/put, european/american, price
"""

"""
This package contains :
- class Option  : to define an option (call/put, american/european)
- class Pricing : to price an option (with black and sholes, monte carlo, binomial tree) and compute delta (with black and sholes)
"""

# mandatory import
from math import sqrt, erf, exp, log, pi
import random as rd

# constant
MONTECARLO_ITERATION = 1000  # Default value for the monte carlo run steps


# === START of Class Option ===

class Option:
    """
    - The Option class contains vanilla option fetaures
    - It handles : call & put option, european and american option.
      Note : pricing of the option is handled in the Pricing class
    - Parameters of Option class are :
      -- type_call_put : "call" (right to buy the underlying at strike price) or "put" (right to sell the underlying at strike price)
      -- type_american_european : "european" (right to exercise option at maturity only) or "american" (right to exercise the option at any time until the maturity)
      -- spot : current spot price of the underlying
      -- strike : strike of the option
      -- vol : volatility (yearly) of the underlying price
      -- rate : interest rate (yearly) of the currency in which the price is defined - for instance : rate = 0.025 for 2.5%
      -- mu : the no arbitrage long term average evolution of the underlying price -  for instance : mu = 0.01 for 1% 
        > for Stock     : mu = r - d       # interest rate - dividend rate
        > Coupon bond   : mu = r - c       # interest rate - coupon rate
        > Currency      : mu = rd - rf     # domestic interest rate - foreign interest rate
        > Commodity     : mu = r + s       # interest rate + storage rate
        > Futures       : mu = 0           # futures are already forward value so mu is null
      -- time : time to maturity (in year) - for instance time = 0.75 for 9 months
    - Example : to define a call european option maturity 6 months, with strike 105, current spot price 102, vol 15%, rate 2.5 %, mu 1%:
      option(call_put="call", european_american="european", spot=102, strike=105, vol=0.15, time = 0.5 rate=0.0025, mu=0.01)
    """
  
    # def __init__(self, calc, call_put, european_american, S, K, V, R, T, step_mc = 1000):
  
    def __init__(self, call_put, european_american, spot, strike, vol, time, rate, mu):
        self.type_call_put = call_put # "call" or "put"
        self.type_american_european = european_american # "european" or "american"
        self.spot = spot
        self.strike = strike
        self.vol = vol
        self.time = time
        self.rate = rate
        self.mu = mu
        self.pricing = Pricing(self) # allow to get Pricing output with my_option.pricing.function() syntax 

    def set_call_put(self, call_put):
        """To set call / put"""
        self.type_call_put = call_put 

    def get_call_put(self):
        """To get call / put"""
        return self.type_call_put

    def set_european_american(self, european_american):
        """To set european / american"""
        self.type_american_european = european_american

    def get_european_american(self):
        """To get european / american"""
        return self.type_american_european

    def set_spot(self, spot):
        """To set spot price"""
        self.spot = spot

    def get_spot(self):
        """To get spot price"""
        return self.spot

    def set_strike(self, strike):
        """To set the trike"""
        self.strike = strike

    def get_strike(self):
        """To get the strike"""
        return self.strike

    def set_vol(self, vol):
        """To set the volatility"""
        self.vol = vol

    def get_vol(self):
        """To get the volatility"""
        return self.vol

    def set_time(self, time):
        """To set the time to maturity"""
        self.time = time

    def get_time(self):
        """To get the time to maturity"""
        return self.time

    def set_rate(self, rate):
        """To set the rate"""
        self.rate = rate

    def get_rate(self):
        """To get the rate"""
        return self.rate

    def set_mu(self, mu):
        """To set mu"""
        self.mu = mu

    def get_mu(self):
        """To get mu"""
        return self.mu

# === END of Class Option ===      


# === START of Class Pricing ===

#class Pricing(Option):
class Pricing:
    """
    Pricing computes options price for european and american, call and put options:
       - with "black_sholes" (closed formula) for european option only, call and put
       - with "monte_carlo" simulation for european option only, call and put
       - with "binomial_tree" model (aka Cox Ross and Rubinstein (CRR) model), for both european and american option, call and put
    Pricing computes also options delta with black_sholes (closed formula) for european option only, call and put

    Pricing parameters are :
      - option features : the option features to price 
      - calc : calculation type, "black_sholes", "binomial_tree", "monte_carlo"
      - step_mc (optional) : number of step/run/iteration for monte carlo simulation. Set to 1000 by default. 
      - step_bt (optional) : number of step for binomial tree model. Set to T * 365, i.e. one per calendar day by default.
    """

    # Pricing takes an Option instance as parameter
    def __init__(self, option, calc="black_sholes", step_mc = MONTECARLO_ITERATION, step_bt = None):
        self.option = option
        self.type_calc = calc   # "black_sholes", "binomial_tree", "monte_carlo"
        self.step_mc = step_mc  # number of step/run/iteration for the Monte Carlo calculation method - set to MONTECARLO_ITERATION per default
        self.step_bt = step_bt  # number of step for the Biinomial Tree calculation method

    # === START Normal Distribution formula ===

    def N(self, x):
        """
        Cumulative distribution function for the SND (Standard Normal Sistribution)
        """
        # N(x) is the integral of the SND from "minus infinity" to x
        # N(- infinity) = 0, N(0) = 0.5, N(+ infinity) = 1
        # N(x) + N(-x) = 1 <=> N(-x) = 1 - N(x)
        CumStdNorm = lambda x : (1 + erf(x/sqrt(2))) / 2 
        return CumStdNorm(x)

    def SND(self, x):
        """
        Standard Normal Distribution function (i.e. with mu = 0 and standard deviation = 1)
        Also known as "Bell curve" because of it shapes or "Gaussian distribution"
        """
        # SND(x) is the derivatives of N(x) ! SND(x) = N'(x) = dN / dx
        # SND(x) =  SND(-x)
        NormDist = lambda x : 1 / (sqrt(2 * pi)) * exp( -pow(x,2) / 2)
        return NormDist(x)


    # def SND(self, x, mu, sigma):
    #     """
    #     Normal distribution function, centered on mu and with standard deviation sigma
    #     Also known as "Bell curve" because of it shapes or "Gaussian distribution"
    #     """
    #     NormDist = lambda x, mu, sigma : 1/(sqrt(2*pi*pow(sigma,2))) * exp(-pow((x-mu),2)/(2*pow(sigma,2)))
    #     return NormDist(x,mu,sigma)

    # === END Normal Distribution formula ===

    def set_calc(self, calc):
        """To set calculation method"""
        self.type_calc = calc

    def get_calc(self):
        """To get calculation method"""
        return self.type_calc

    def set_step_mc(self, step_mc):
        """To set the number of iteration for the Monte Carlo computation - by default set to 1000"""
        self.step_mc = step_mc

    def get_step_mc(self):
        """To get the number of iteration for the Monte Carlo computation"""
        return self.step_mc

    def set_step_bt(self, step_bt):
        """To set the number of iteration for the Monte Carlo computation - by default set to 1000"""
        self.step_bt = step_bt

    def get_step_bt(self):
        """To get the number of iteration for the Monte Carlo computation"""
        return self.step_bt

    def set_option_parameter(self, european_american=None, call_put=None, spot=None, strike=None, vol=None, time=None, rate=None, mu=None):
        """To set one, several or all the option parameters"""
        if european_american != None : self.option.set_european_american(european_american)
        if call_put != None          : self.option.set_call_put(call_put) 
        if spot != None              : self.option.set_spot(spot)
        if strike != None            : self.option.set_strike(strike)
        if vol != None               : self.option.set_vol(vol)
        if time != None              : self.option.set_time(time)
        if rate != None              : self.option.set_rate(rate)
        if mu != None                : self.option.set_mu(mu)

    def get_option_parameter(self):
        """To get all the option parameters"""
        european_american = self.option.get_european_american()
        call_put = self.option.get_call_put() 
        spot = self.option.get_spot()
        strike = self.option.get_strike()
        vol = self.option.get_vol()
        time = self.option.get_time()
        rate = self.option.get_rate()
        mu = self.option.get_mu()
        option_parameter = (european_american, call_put, spot, strike, vol, time, rate, mu)
        return option_parameter

    def get_black_sholes(self, request='price'):
        """To compute price of european call and put option with Black & Sholes formula"""
        # Definition of variable on one letter to make formula easier to write and read
        (european_american, call_put, s, k, v, t, r, mu) = self.get_option_parameter()
        # European option with Black & Sholes formula  
        d1 = ( log(s / k) + (mu + 0.5*v*v) * t ) / (v * sqrt(t))
        d2 = d1 - v * sqrt(t)
        # f = s*exp(mu*t) # forward value of the underlying
        if european_american == "european" and call_put == "call" : 
            if request == 'payoff':
                output = max(s - k, 0)
            elif request == 'price':
                output = s * exp((mu-r) * t) * self.N(d1) \
                         - k * exp(- r * t) * self.N(d2)                     # Call European price
            elif request == 'delta': # dP / dS
                output = exp( (mu-r) * t) * self.N(d1)                       # Call European delta
            elif request == 'gamma': # d²P / dS²
                output = exp( (mu-r) * t) * self.SND(d1) / (s * v * sqrt(t)) # Call/Put European gamma
            elif request == 'vega': # dP / dv
                output = exp( (mu-r) * t) * s * sqrt(t) * self.SND(d1)       # Call/Put European vega
            elif request == 'vanna': # d²P / dS dv
                output =  - exp( (mu-r) * t) * d2 / v * self.SND(d1)         # Call/Put European vanna
            elif request in [ 'volga', 'vomma' ] : # d²P / dV²
                output =  s * sqrt(t) * exp( (mu-r) * t) * d1 * d2 / v * self.N(d1) # Call/Put European volga
            elif request == 'theta': # dP / dt
                output = - s * v * exp((mu - r) * t) * self.N(d1) / (2 * sqrt(t))  \
                         - r * k * exp(- r * t) * self.N(d2) \
                         + exp( (mu-r) * t) * self.SND(d1)                   # Call European theta
            elif request == 'rho' or request == 'rho_d':  # dP / dr = rho = rho_d = rho_domestic
                output = k * t * exp(- r * t) * self.N(d2)                   # Call European rho_d
            elif request == 'rho_f':  # dP / d(mu-r) = rho_f = rho_foreign
                output = - s * t * exp((mu - r) * t) * self.N(d1)            # Call European rho_f
            else:
                output = None
        elif european_american == "european" and call_put == "put" :
            if request == 'payoff':
                output = max(k - s, 0)
            elif request == 'price':
                output = - s * exp((mu-r) * t) * self.N(-d1) \
                         + k * exp(- r * t) * self.N(-d2)                    # Put European price
            elif request == 'delta':
                output = - exp( (mu-r) * t) * self.N(-d1)                    # Put European delta
            elif request == 'gamma':
                output = exp( (mu-r) * t) * self.SND(d1) / (s * v * sqrt(t)) # Call/Put European gamma
            elif request == 'vega':
                output = exp( (mu-r) * t) * s * sqrt(t) * self.SND(d1)       # Call/Put European vega
            elif request == 'vanna':
                output =  - exp( (mu-r) * t) * d2 / v * self.SND(d1)         # Call/Put European vanna
            elif request in [ 'volga', 'vomma' ]: 
                output =  s * sqrt(t) * exp( (mu-r) * t) * d1 * d2 / v * self.N(d1) # Call/Put European volga
            elif request == 'theta': 
                output = - s * v * exp((mu - r) * t) * self.N(- d1) / (2 * sqrt(t)) \
                         + r * k * exp(- r * t) * self.N(- d2) \
                         - exp( (mu-r) * t) * self.SND(- d1)                 # Put European theta
            elif request == 'rho' or request == 'rho_d':
                output = - k * t * exp(- r * t) * self.N(- d2)               # Put European rho_d
            elif request == 'rho_f': # dP / d(mu-r) = rho_f = rho_foreign
                output = s * t * exp((mu - r) * t) * self.N(- d1)            # Put European rho_f
            else:
                output = None
        else :
            output = None
        return output

    def get_analytical_greeks(self, request = None):
        get_black_sholes(request)

    def get_pricing_temp(self, spot=None, strike=None, vol=None, time=None, rate=None, mu=None):
        """
        To get option pricing with a market data bump but without changing the option features
        Used to compute simulated greeks        
        """
        # get the current option features
        (european_american_0, call_put_0, spot_0, strike_0, vol_0, time_0, rate_0, mu_0) = self.get_option_parameter()
        # update th option features if need be
        if spot != None   : self.option.set_spot(spot)
        if strike != None : self.option.set_strike(strike)
        if vol != None    : self.option.set_vol(vol) 
        if time != None   : self.option.set_time(time)
        if rate != None   : self.option.set_rate(rate)
        if mu != None     : self.option.set_mu(mu)
        price = self.get_pricing(request = 'price')
        # set back option feature as initial
        self.set_option_parameter(european_american=european_american_0, call_put=call_put_0, spot=spot_0,
                                  strike=strike_0, vol=vol_0, time=time_0, rate=rate_0, mu=mu_0)
        return price

    def get_simulated_greeks(self, request = None, calc = None, step = None, step_bt = None):
        self.set_calc_step(request = request, calc = calc, step = step, step_bt = step_bt)
        (european_american, call_put, s, k, v, t, r, mu) = self.get_option_parameter()
        epsilon = 0.00001
        output = None
        # print("> Simulated :", request, calc)
        if calc in [ 'black_sholes', 'binomial_tree']:
            if request == 'delta': # dP / dS
                s_delta = abs(s) * epsilon
                p0 = self.get_pricing_temp(spot = s - s_delta)
                p1 = self.get_pricing_temp(spot = s + s_delta)
                if p0 != None and p1 != None and s_delta != 0:
                    output = (p1 - p0) / (2 * s_delta) 
            elif request == 'gamma': # d²P / dS²
                s_delta = abs(s) * epsilon * 100
                p0 = self.get_pricing_temp(spot = s - s_delta)
                p1 = self.get_pricing_temp(spot = s)
                p2 = self.get_pricing_temp(spot = s + s_delta)
                if p0 != None and p1 != None and p2 != None and s_delta != 0:
                    output = (p0 + p2 - 2 * p1) / (s_delta ** 2) 
            elif request == 'vega': # dP / dv
                v_delta = abs(s) * epsilon
                p0 = self.get_pricing_temp(vol = v - v_delta)
                p1 = self.get_pricing_temp(vol = v + v_delta)
                if p0 != None and p1 != None and v_delta != 0:
                    output = (p1 - p0) / (2 * v_delta)
            elif request == 'vanna': # d²P / dS dv
                s_delta = abs(s) * epsilon
                v_delta = abs(v) * epsilon
                p_s0_v0 = self.get_pricing_temp(spot = s - s_delta, vol = v - v_delta)
                p_s0_v1 = self.get_pricing_temp(spot = s - s_delta, vol = v + v_delta)
                p_s1_v0 = self.get_pricing_temp(spot = s + s_delta, vol = v - v_delta)
                p_s1_v1 = self.get_pricing_temp(spot = s + s_delta, vol = v + v_delta)
                if p_s0_v0 != None and p_s0_v1 != None and p_s1_v0 != None and p_s1_v1 != None and s_delta != 0 and v_delta != 0:
                    output = (p_s1_v1 + p_s0_v0 - p_s1_v0 - p_s0_v1) / (2 * s_delta * 2 * v_delta)
            elif request in [ 'volga', 'vomma' ] : # d²P / d²v
                v_delta = abs(v) * epsilon * 10
                p0 = self.get_pricing_temp(vol = v - v_delta)
                p1 = self.get_pricing_temp(vol = v)
                p2 = self.get_pricing_temp(vol = v + v_delta)
                if p0 != None and p1 != None and p2 != None and v_delta != 0:
                    output = (p0 + p2 - 2 * p1) / (v_delta ** 2)
            elif request == 'theta':
                t_delta = 1/365
                p0 = self.get_pricing_temp(time = t)
                p1 = self.get_pricing_temp(time = t - t_delta)
                if p0 != None and p1 != None and t_delta != 0:
                    output = (p1 - p0) / (t_delta)
            elif request in [ 'rho', 'rho_d' ]:
                r_delta = max(abs(r * epsilon), epsilon)
                p0 = self.get_pricing_temp(rate = r - r_delta)
                p1 = self.get_pricing_temp(rate = r + r_delta)
                if p0 != None and p1 != None and r_delta != 0:
                    output = (p1 - p0) / (2 * r_delta) 
            elif request == 'rho_f':
                mu_delta = - max(abs(mu * epsilon), epsilon)  # mu = r - rf = rd - rf => rf = r - mu => drf = - dmu
                p0 = self.get_pricing_temp(mu = mu - mu_delta)
                p1 = self.get_pricing_temp(mu = mu + mu_delta)
                if p0 != None and p1 != None and mu_delta != 0:
                    output = - (p1 - p0) / (2 * mu_delta) 
            else:
                output = None
        return output

    def get_price_black_sholes(self):
        """Legacy"""
        price_bs = self.get_black_sholes('price')
        return price_bs

    def get_delta_black_sholes(self):
        """Legacy"""
        delta_bs = self.get_black_sholes('delta')
        return delta_bs

    def get_price_monte_carlo(self, step_mc = MONTECARLO_ITERATION):
        """To compute the price of european call and put option with Monte Carlo simulation"""
        (european_american, call_put, s, k, v, t, r, mu) = self.get_option_parameter()
        nb_iteration = max(1, step_mc) # minimum number of step is 1
        # print("MC steps =", nb_iteration)
        payoff = 0
        for i in range(nb_iteration):
            random_normal = rd.normalvariate(0,1)
            forward_selection = s * exp( ( ( mu- v*v/2) * t ) + ( random_normal * v * sqrt(t) ) ) 
            if european_american == "european" and call_put == "call" : 
                payoff += max(0, forward_selection - k)
            elif european_american == "european" and call_put == "put" :
                payoff += max(0 , k - forward_selection )
            else :
                payoff = None
        if payoff != None:
            average_payoff = payoff / nb_iteration * exp (- r * t)
        else:
            average_payoff = None
        return average_payoff

    def get_price_binomial_tree(self, nb_step):
        """
        To compute price with Cox Ross and Rubinstein (CRR) model
        Applicable for both european and american option
        """
        (european_american, call_put, s, k, sigma, time, rate, mu) = self.get_option_parameter()
        # def print_matrix(matrix):
        #     # """ print matrix for investigation purpose only """
        #     for row in matrix:
        #         line = ""    
        #         for item in row:
        #             line += ("       " + str(int(item//1)))[-7:] + " " 
        #         print(line)
        
        if nb_step == None:
            nb_step = time * 365 // 1  # per default, one step per calendar day
        nb_step = int(nb_step)
        # print("BT steps =", nb_step)

        delta_time = time / nb_step
        u = exp( sigma * sqrt(delta_time) )
        d = 1/u # = exp( -sigma * sqrt(delta_time) )
        b_star = exp (mu * delta_time)
        q = (b_star - d) / (u - d)
        rate_star = exp (rate * delta_time)
        # p = ( exp(rate*time) - d ) / (u - d)

        # Step 1 : define the binomial tree of the asset price
        asset_price = [ [0.0 for _ in range(1 + nb_step)] for _ in range(1 + nb_step) ]
        for n_d in range(1 + nb_step):
            for n_u in range(n_d+1):
                   # Sn = S0 * pow(u, Nu-Nd) = pow(u, n_d) * pow(d, 2*n_u)
                    p = s * pow(u, n_d - 2 * n_u) 
                    asset_price[n_u][n_d] = p
        # print("asset_price :")
        # print_matrix(asset_price)

        # Step 2 : define option pay_off binomial tree

        # Step 2.1 : initialise last column (= pay off at maturity)
        # (same for european and america option)
        option_payoff = [ [0.0 for _ in range(1 + nb_step)] for _ in range(1 + nb_step) ]
        for i in range(1 + nb_step):
            if call_put == "call" : 
                payoff = max(0, asset_price[i][nb_step] - k )
            elif call_put == "put" :
                payoff = max(0, k - asset_price[i][nb_step] )
            else :
                payoff = 0
            option_payoff[i][1 + nb_step-1] = payoff

        # Step 2.2 : backward path (depend upon european and america option)
        for j in range(nb_step - 1, -1, -1):
            for i in range(0, j+1, +1):
                value_temp = ( option_payoff[i][j+1] * q + option_payoff[i+1][j+1] * (1 -q) ) / rate_star 
                if european_american == "european":
                    value =  value_temp
                elif european_american == "american" and call_put == "call":
                    value = max (value_temp, asset_price[i][j] - k )
                elif european_american == "american" and call_put == "put":
                    value = max (value_temp, k - asset_price[i][j] )
                # print(f"value [{i}][{j}] = {value}")
                option_payoff[i][j] = value
                
        # print("option_payoff :")
        # print_matrix(option_payoff)

        option_value = option_payoff[0][0] 
        return option_value

    def get_price(self, calc = None, step = None, step_mc = None, step_bt = None):
        """Legacy - to get price of the option"""
        option_price = self.get_pricing(request = 'price', calc = calc, step = step, step_mc = step_mc, step_bt = step_bt)
        return option_price

    def set_calc_step(self, request = None, calc = None, step = None, step_mc = None, step_bt = None):
        if calc != None:
            self.set_calc(calc)
        if step != None:
            if calc == "monte_carlo":
                self.set_step_mc(step)
            elif calc == "binomial_tree":
                self.set_step_bt(step)
        if step_mc != None:
            self.set_step_mc(step_mc)
        if step_bt != None:
            self.set_step_bt(step_bt)

    def get_pricing(self, request = None, calc = None, step = None, step_mc = None, step_bt = None): # request = None,
        """To get price of the option"""
        # update cal and step/step_mc/step_bt the case being
        self.set_calc_step(request = request, calc = calc, step = step, step_mc = step_mc, step_bt = step_bt)

        # get the request
        if request == 'price':
            if self.type_calc == "black_sholes" :     # closed formula = black & sholes
                output = self.get_price_black_sholes()
            elif self.type_calc == "monte_carlo" :    # monte carlo
                output = self.get_price_monte_carlo(self.get_step_mc())
            elif self.type_calc == "binomial_tree" :  # binomial tree = Cox Ross and Rubinsteain (CRR)
                output = self.get_price_binomial_tree(self.get_step_bt())
            else :
                output = None
        elif request == 'payoff':
            output = self.get_black_sholes('payoff')  # payoff is the same whatever is the model : black_sholes, monte_carlo or binomial_tree
        elif request in [ 'delta', 'gamma', 'vega', 'vanna', 'volga', 'theta', 'rho_d', 'rho_f' ] and calc == 'black_sholes' :
            output = self.get_black_sholes(request) 
        else:
            output = None
        return output

# === END of Class Pricing ===

# === START Usage example of the Class Option ===

def option_graph(): 
    """
    Display a graph with Spot (x-axis) vs Option price (y-axis)
    for several time to maturity
    """

    # import only used for example/demo purpose - not needed for core feature (Option and Pricing classes)
    import numpy as np
    import matplotlib.pyplot as plt

    # initialise the option
    my_option = Option(call_put="call", european_american="european", spot=100, strike=102, vol=0.10, time=0.7, rate=0.02, mu=0)
    
    # initialise the graph
    spot_range = np.arange(90,110,0.5).tolist() # spot values on x-axis
    price_range = [] # price value for y-axis
    time_range = [0.001, 0.1, 0.3, 1, 3]  # time values (in years)
    nb_time = len(time_range)
    for S in spot_range:
        my_option.set_spot(S)
        for i in range(nb_time):
            T = time_range[i]
            my_option.set_time(T) 
            option_price = my_option.pricing.get_price(calc = "black_sholes")
            price_range.append([])
            price_range[i].append(option_price) # option price on y-axis
            # print("S =",S,"T =",T,"option_price =",option_price)
    for i in range(nb_time):
        figure = plt.plot(spot_range, price_range[i], 'b') # define the graph
    plt.show()  # display the graph 

def option_price():
    """
    Launch and display european option pricing
    with Black & Sholes formula, Monte Carlo simulation and Binomial Tree (Cox Ross and Rubinstein - CRR) model
    """

    # Option parameters
    # (S, K, V, T, R, mu) = (106, 100, 0.46, 0.75, 0.058, 0)
    (S, K, V, T, R, mu) = (100, 102, 0.15, 10, 0.03, 0.01)

    # Pricing parameters
    mc_nb_iteration = 10000  # number of iteration for one monte carlo simulation run
    bt_nb_step = 100

    # To initialise the option
    my_option = Option(call_put="call", european_american="european", spot=S, strike=K, vol=V, time=T, rate=R, mu=mu)

    # To initialise the pricing
    my_option.pricing.set_step_mc(mc_nb_iteration)
    my_option.pricing.set_step_bt(bt_nb_step)

    # To display pricing for american & european, call & put
    #    with black_sholes (european only), monte_carlo (european only), binomial_tree (american and european)
    rounding_price_display = 3
    for ea in ['european', 'american']:
        for cp in ['call', 'put']:
            my_option.set_european_american(ea)
            my_option.set_call_put(cp)
            print("===", ea, "-", cp, "===")
            for calc in [ 'black_sholes', 'binomial_tree', 'monte_carlo']:
                # Price for black_sholed, monte_carlo and binomial_tree
                price_option = my_option.pricing.get_price(calc = calc)
                if price_option != None:
                    cp_s   = ( cp + " ")[:4]
                    calc_s = ( calc + "   ")[:13]
                    print(f"  {ea} {cp_s} Price {calc_s} = {round(price_option, rounding_price_display)}")
                # Analytical greeks for black_sholes and Simulated greeks for Black & Sholes and Binomial Tree
                if calc in [ 'black_sholes', 'binomial_tree' ]:
                    for request in [ 'payoff', 'delta', 'gamma', 'vega', 'vanna', 'volga', 'theta', 'rho_d', 'rho_f' ]:
                        output = my_option.pricing.get_black_sholes(request)
                        if output != None and calc == 'black_sholes':
                            print(f"    {ea} {cp} {calc} Analytical {request} = {round(output, 5)}")
                        output = my_option.pricing.get_simulated_greeks(request = request, calc = calc)
                        if output != None :
                            print(f"    {ea} {cp} {calc} Simulated  {request} = {round(output, 5)}")
                """# Simulated greeks for black_sholes and binomial_tree
                if calc in [ 'black_sholes', 'binomial_tree' ]:
                    for request in [ 'delta', 'gamma', 'vega' ]:  'vanna', 'volga', 'theta', 'rho_d', 'rho_f' ]:
                        output = my_option.pricing.get_simulated_greeks(request = request, calc = calc)
                        if output != None :
                            print(f"    {ea} {cp} {calc} Simulated {request} = {round(output, 5)}")
                """

def pricing_syntax():
    """Example of option pricing syntax"""

    # option definition
    (S, K, V, T, R, mu) = (106, 100, 0.46, 0.75, 0.058, 0)
    my_option = Option(call_put="call", european_american="european", spot=S, strike=K, vol=V, time=T, rate=R, mu=mu)
    
    # option pricing call

    # various syntax to get price with black & sholes formula
    my_option.pricing.get_price()                                          # black_sholes by default (only for european option)
    my_option.pricing.get_price("black_sholes")                            # black_sholes
    my_option.pricing.get_price(calc = "black_sholes")                     # black_sholes
    my_option.pricing.get_pricing('price', 'black_sholes')                 # black_sholes payoff
    my_option.pricing.get_pricing(request='price', calc = 'black_sholes')  # black_sholes price
    my_option.pricing.get_black_sholes('price')                            # black_sholes price
    my_option.pricing.get_black_sholes(request='price')                    # black_sholes price

    # various syntax to get price with monte carlo simulation
    my_option.pricing.get_price("monte_carlo")                         # monte_carlo with 1000 iterations per run by default
    my_option.pricing.get_price(calc = "monte_carlo")                  # monte_carlo with 1000 iterations per run by default
    my_option.pricing.get_price("monte_carlo", 7000 )                  # monte_carlo with 7000 iterations per run
    my_option.pricing.get_price(calc = "monte_carlo", step = 7000 )    # monte_carlo with 7000 iterations per run
    my_option.pricing.get_price(calc = "monte_carlo", step_mc = 7000 ) # monte_carlo with 7000 iterations per run
    my_option.pricing.get_pricing('price', 'monte_carlo')              # monte_carlo price with 1000 iterations per run by default
    my_option.pricing.get_pricing('price', 'monte_carlo', 7000)        # monte_carlo price with 7000 iterations per run
    my_option.pricing.get_pricing(request='price', calc='monte_carlo', step=7000)      # monte_carlo price with 7000 iterations per run
    my_option.pricing.get_pricing(request='price', calc='monte_carlo', step_mc=7000)   # monte_carlo price with 7000 iterations per run

    # various syntax to get price with binomial tree method
    my_option.pricing.get_price('binomial_tree')                       # binomial_tree with 1 step per day by default (i.e. with time*365 steps)
    my_option.pricing.get_price(calc = 'binomial_tree')                # binomial_tree with 1 step per day by default (i.e. with time*365  steps)
    my_option.pricing.get_price('binomial_tree', 50)                   # binomial_tree with 50 steps
    my_option.pricing.get_price(calc = 'binomial_tree', step = 50)     # binomial_tree with 50 steps
    my_option.pricing.get_price(calc = 'binomial_tree', step_bt = 50)  # binomial_tree with 50 steps
    my_option.pricing.get_pricing('price', 'binomial_tree')            # binomial_tree with 1 step per day by default (i.e. with time*365  steps)
    my_option.pricing.get_pricing('price', 'binomial_tree', 50)        # binomial_tree with 50 steps
    my_option.pricing.get_pricing(request='price', calc='binomial_tree', step=50)      # binomial_tree with 50 steps
    my_option.pricing.get_pricing(request='price', calc='binomial_tree', step_bt=50)   # binomial_tree with 50 steps

    # various syntax to get greeks (and delta) with black_sholes analytical formula
    # the 4 syntaxes example for delta apply also for all the other greeks and payoff
    my_option.pricing.get_black_sholes('delta')                         # black_sholes delta
    my_option.pricing.get_black_sholes(request='delta')                 # black_sholes delta
    my_option.pricing.get_pricing('delta', 'black_sholes')              # black_sholes payoff
    my_option.pricing.get_pricing(request='delta', calc='black_sholes') # black_sholes payoff

    my_option.pricing.get_black_sholes('gamma')                         # black_sholes gamma
    my_option.pricing.get_black_sholes('vega')                          # black_sholes vega
    my_option.pricing.get_black_sholes('vanna')                         # black_sholes vanna
    my_option.pricing.get_black_sholes('volga')                         # black_sholes volga
    my_option.pricing.get_black_sholes('theta')                         # black_sholes theta
    my_option.pricing.get_black_sholes('rho')                           # black_sholes rho = rho_d
    my_option.pricing.get_black_sholes('rho_f')                         # black_sholes rho_f
    my_option.pricing.get_black_sholes('payoff')                        # black_sholes payoff



def test_SND():
    (S, K, V, T, R, mu) = (106, 100, 0.46, 0.75, 0.058, 0)
    my_option = Option(call_put="call", european_american="european", spot=S, strike=K, vol=V, time=T, rate=R, mu=mu)
    # mc_nb_iteration = 100000  # number of iteration for one monte carlo simulation run
    # bt_nb_step = 8
    # To initialise the pricing
    # my_option.pricing.set_step_mc(mc_nb_iteration)
    # my_option.pricing.set_step_bt(bt_nb_step)

    x_value = [-2, -1, -0.5, 0, 0.5, 1, 2]
    epsilon = 0.00001
    for x in x_value:
        snd = my_option.pricing.SND(x)
        n0 = my_option.pricing.N(x)
        n1 = my_option.pricing.N(x+ epsilon)
        dn = (n1-n0) / epsilon
        print("snd(",x ,") =", snd, "/ dn =", dn)


if __name__ == '__main__':
    # option_graph()
    option_price()
    pricing_syntax()
    # test_SND()

# === END Usage example of the Class Option ===

