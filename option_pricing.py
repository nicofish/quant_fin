# Object  : Pricing call and put european option
#           with - Black & Sholes (closed) formula
#                - Monte Carlo simulation
#                - Binomial Tree (Cox Ross and Rubinstein : CRR) model
# Version : v1.1
# Date    : 2020.12.20
# Python  : v3.7.7
# Author  : NicoFish

"""
This package contains :
- class Option  : to define an option (call/put, american/european)
- class Pricing : to price an option (with black and sholes, monte carlo, binomial tree) and compute delta (with black and sholes)
"""

# mandatory import
from math import sqrt, erf, exp, log
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
      -- type_cp : "call" (right to buy the underlying at strike price) or "put" (right to sell the underlying at strike price)
      -- type_ea : "european" (right to exercise option at maturity only) or "american" (right to exercise the option at any time until the maturity)
      -- spot : current spot price of the underlying
      -- strike : strike of the option
      -- vol : volatility (yearly) of the underlying price
      -- rate : interest rate (yearly) of the currency in which the price is defined - for instance rate = 0.025 for 2.5%
      -- time : time to maturity (in year) - for instance time = 0.75 for 9 months
      -- step_mc (optional) : number of step for monte carlo simulation. set to 1000 by default. 
    - Example : to define an call european option maturity 6 months, with strike 105, current spot price 102, vol 15%, rate 2.5 % :
      option(call_put="call", european_american="european", S=102, K=105, V=0.15, R=0.0025, T=0.5)
    """
  
    # def __init__(self, calc, call_put, european_american, S, K, V, R, T, step_mc = 1000):
  
    def __init__(self, call_put, european_american, S, K, V, R, T):
         # self.type_calc = calc # "black_sholes" for closed formula, "binomial_tree", "monte_carlo"
        self.type_cp = call_put # "call" or "put"
        self.type_ea = european_american # "european" or "american"
        self.spot = S
        self.strike = K
        self.vol = V
        self.rate = R
        self.time = T
        self.pricing = Pricing(self) # allow to get Pricing output with my_option.pricing.function() syntax 
        # self.step_mc = step_mc # number of step/run/iteration for the Monte Carlo calculation method

    def set_call_put(self, call_put):
        """To set call / put"""
        self.type_cp = call_put 

    def get_call_put(self):
        """To get call / put"""
        return self.type_cp

    def set_ea(self, european_american):
        """To set european / american"""
        self.type_ea = european_american

    def get_ea(self):
        """To get european / american"""
        return self.type_ea

    def set_spot(self, S):
        """To set spot price"""
        self.spot = S

    def get_spot(self):
        """To get spot price"""
        return self.spot

    def set_strike(self, K):
        """To set the trike"""
        self.strike = K

    def get_strike(self):
        """To get the strike"""
        return self.strike

    def set_vol(self, V):
        """To set the volatility"""
        self.vol = V

    def get_vol(self):
        """To get the volatility"""
        return self.vol

    def set_rate(self, R):
        """To set the rate"""
        self.rate = R

    def get_rate(self):
        """To get the rate"""
        return self.rate

    def set_time(self, T):
        """To set the time to maturity"""
        self.time = T

    def get_time(self):
        """To get the time to maturity"""
        return self.time

    # def get_price(self, calc_input = "black_sholes", step_mc_input = MONTECARLO_ITERATION):
    #   my_price = Pricing(self)
    #   return my_price.get_price(calc_input, step_mc_input)

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
      - mu : the no arbitrage long term average evolution of the underlying price
        mu value (no arbitrage long term average evolution of the underlying price) 
        > for Stock     : mu = r - d       # interest rate - dividend rate 
        > Coupon bond	: mu = r - c       # interest rate - coupon rate
        > Currency	    : mu = rd - rf     # domestic interest rate - foreign interest rate
        > Commodity     : mu = r + s       # interest rate + storage rate
        > Futures		: mu = 0           # futures are already forward value so mu is null
    """

    # Pricing takes an Option instance as parameter
    def __init__(self, option, calc="black_sholes", step_mc = MONTECARLO_ITERATION, step_bt = None, mu = 0):
        self.option = option
        self.type_calc = calc   # "black_sholes", "binomial_tree", "monte_carlo"
        self.step_mc = step_mc  # number of step/run/iteration for the Monte Carlo calculation method
        self.step_bt = step_bt  # number of step for the Biinomial Tree calculation method
        self.mu = mu            # 

    # === START Normal Distribution formula ===

    def N(self, x):
        """ Cumulative distribution function for the standard normal distribution"""
        CumStdNorm = lambda x : (1 + erf(x/sqrt(2)))/2 
        return CumStdNorm(x)

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

    def set_mu(self, mu):
        """To set mu"""
        self.mu = mu

    def get_mu(self):
        """To get mu"""
        return self.mu

    def get_price_black_sholes(self):
        """To compute price of european call and put option with Black & Sholes formula"""
        # Definition of variable on one letter to make formula easier to write and read
        european_american = self.option.get_ea()
        call_put = self.option.get_call_put() 
        s = self.option.get_spot()
        k = self.option.get_strike()
        v = self.option.get_vol()
        t = self.option.get_time()
        r = self.option.get_rate()
        mu = self.option.pricing.get_mu()
        # European option with Black & Sholes formula  
        d1 = ( log(s/k) + (mu+0.5*v*v) * t ) / (v*sqrt(t))
        d2 = d1 - v * sqrt(t)
        # f = s*exp(mu*t) # forward value of the underlying
        if european_american == "european" and call_put == "call" : 
            call_eu_price = ( s * exp(mu*t) * self.N(d1) - k * self.N(d2) ) * exp(-r*t) # Call European price
            return call_eu_price
        elif european_american == "european" and call_put == "put" :
            put_eu_price = ( k * self.N(-d2) - s * exp(mu*t) * self.N(-d1) ) * exp(-r*t) # Put European price
            return put_eu_price  
        else :
            return None

    def get_delta_black_sholes(self):
        """To compute price of european call and put option with Black & Sholes formula"""
        # Definition of variable on one letter to make formula easier to write and read
        european_american = self.option.get_ea()
        call_put = self.option.get_call_put() 
        s = self.option.get_spot()
        k = self.option.get_strike()
        v = self.option.get_vol()
        t = self.option.get_time()
        r = self.option.get_rate()
        mu = self.option.pricing.get_mu()
        # European option with Black & Sholes formula  
        d1 = ( log(s/k) + (mu+0.5*v*v) * t ) / (v*sqrt(t))
        # d2 = d1 - v * sqrt(t)
        if european_american == "european" and call_put == "call" : 
            call_eu_delta = exp( (mu-r) * t) * self.N(d1) # Call European price
            return call_eu_delta
        elif european_american == "european" and call_put == "put" :
            put_eu_delta = - exp( (mu-r) * t) * (1 - self.N(d1))  # Put European price
            return put_eu_delta  
        else :
            return None

    def get_price_monte_carlo(self, step_mc = MONTECARLO_ITERATION):
        """To compute the price of european call and put option with Monte Carlo simulation"""
        european_american = self.option.get_ea()
        call_put = self.option.get_call_put()
        s = self.option.get_spot()
        k = self.option.get_strike()
        v = self.option.get_vol()
        t = self.option.get_time()
        r = self.option.get_rate()
        mu = self.option.pricing.get_mu()
        nb_iteration = max(1, step_mc) # minimum number of step is 1
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

    def get_price_binomial_tree(self, nb_steps):
        """
        To compute price with Cox Ross and Rubinstein (CRR) model
        Applicable for both european and american option
        """
        european_american = self.option.get_ea()
        call_put = self.option.get_call_put()
        s = self.option.get_spot()
        k = self.option.get_strike()
        rate = self.option.get_rate()
        sigma = self.option.get_vol()
        time = self.option.get_time()
        mu = self.option.pricing.get_mu()

        # def print_matrix(matrix):
        #     # """ print matrix for investigation purpose only """
        #     for row in matrix:
        #         line = ""    
        #         for item in row:
        #             line += ("       " + str(int(item//1)))[-7:] + " " 
        #         print(line)
        
        if nb_steps == None:
            nb_steps = time * 365 // 1  # per default, one step per calendar day

        delta_time = time / nb_steps
        u = exp( sigma * sqrt(delta_time) )
        d = 1/u # exp( -sigma * sqrt(time) )
        b_star = exp (mu * delta_time)
        q = (b_star - d) / (u - d)
        rate_star = exp (rate * delta_time)
        # p = ( exp(rate*time) - d ) / (u - d)

        # Step 1 : define the binomial tree of the asset price
        asset_price = [ [0.0 for _ in range(1 + nb_steps)] for _ in range(1 + nb_steps) ]
        for n_d in range(1 + nb_steps):
            for n_u in range(n_d+1):
                   # Sn = S0 * pow(u, Nu-Nd) = pow(u, n_d) * pow(d, 2*n_u)
                    p = s * pow(u, n_d - 2 * n_u) 
                    asset_price[n_u][n_d] = p
        # print("asset_price :")
        # print_matrix(asset_price)

        # Step 2 : define option pay_off binomial tree

        # Step 2.1 : initialise last column (= pay off at maturity)
        # (same for european and america option)
        option_payoff = [ [0.0 for _ in range(1 + nb_steps)] for _ in range(1 + nb_steps) ]
        for i in range(1 + nb_steps):
            if call_put == "call" : 
                payoff = max(0, asset_price[i][nb_steps] - k )
            elif call_put == "put" :
                payoff = max(0, k - asset_price[i][nb_steps] )
            else :
                payoff = 0
            option_payoff[i][1 + nb_steps-1] = payoff

        # Step 2.2 : backward path (depend upon european and america option)
        for j in range(nb_steps - 1, -1, -1):
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

    def get_price(self, calc_input = None, step_mc_input = None, step_bt_input = None, mu = None):
        """To get price of the option"""
        if calc_input != None:
            self.type_calc = calc_input
        if step_mc_input != None:
            self.set_step_mc(step_mc_input)
        if step_bt_input != None:
            self.set_step_bt(step_bt_input)
        if mu != None:
            self.set_mu(mu)
        if self.type_calc == "black_sholes" :     # closed formula = black & sholes
            return self.get_price_black_sholes()
        elif self.type_calc == "monte_carlo" :    # monte carlo
            return self.get_price_monte_carlo(self.get_step_mc())
        elif self.type_calc == "binomial_tree" :  # binomial tree
            return self.get_price_binomial_tree(self.get_step_bt())
        else :
            return None

# === END of Class Pricing ===

# === START Usage example of the Class Option ===

def option_graph(): 
    """
    Display a graph with Spot (x-axis) vs Option price (y-axis)
    for several time to maturity
    """

    # only used for example/demo purpose - not needed for core feature (Option and Pricing classes)
    import numpy as np
    import matplotlib.pyplot as plt

    my_option = Option(call_put="call", european_american="european", S=100, K=102, V=0.10, R=0.02, T=0.7)
    spot_range = np.arange(90,110,0.5).tolist() # spot values on x-axis
    price_range = []
    time_range = [0.001, 0.1, 0.3, 1, 3]  # time values
    nb_time = len(time_range)
    for S in spot_range:
        my_option.set_spot(S)
        for i in range(nb_time):
            T = time_range[i]
            my_option.set_time(T) 
            option_price = my_option.pricing.get_price(calc_input = "black_sholes")
            price_range.append([])
            price_range[i].append(option_price) # option price on y-axis
            # print("S =",S,"T =",T,"option_price =",option_price)
    for i in range(nb_time):
        figure = plt.plot(spot_range, price_range[i], 'b')
    plt.show() # display the graph 

def option_price():
    """
    Launch and display european option pricing
    with Black & Sholes formula and Monte Carlo simulation
    """

    # Option parameters
    (S, K, V, R, T) = (106, 100, 0.46, 0.058, 0.75)
    # (S, K, V, R, T) = (100, 102, 0.15, 0.02, 1)

    # Pricing parameters
    mu = 0
    mc_nb_iteration = 100000  # number of iteration for one monte carlo simulation run
    bt_nb_step = 500

    # initialise the option
    my_option = Option(call_put="call", european_american="european", S=S, K=K, V=V, R=R, T=T)

    # initialise the pricing
    my_option.pricing.set_step_mc(mc_nb_iteration)
    my_option.pricing.set_step_bt(bt_nb_step)

    # display pricing for american & european, call & put
    # with black_sholes (european only), monte_carlo (european only), binomial_tree (american and european)
    for ea in ["european", "american"]:
        for cp in ["call", "put"]:
            my_option.set_ea(ea)
            my_option.set_call_put(cp)
            for calc in ["black_sholes", "monte_carlo", "binomial_tree"]:
                price_option = my_option.pricing.get_price(calc_input = calc, mu = mu)
                if price_option != None:
                    cp_s   = ( cp + " ")[:4]
                    calc_s = ( calc + "   ")[:13]
                    print(f"{ea} {cp_s} Price {calc_s} = {round(price_option,3)}")

    # display delta for european only, call & put, with black_sholes
    for cp in ["call", "put"]:
        my_option.set_ea("european")
        my_option.set_call_put(cp)   
        delta_option = my_option.pricing.get_delta_black_sholes()
        cp_s   = ( cp + " ")[:4]
        print(f"european {cp_s} Delta black_sholes  = {round(delta_option,4)}")

# Example of pricing syntax
def pricing_syntax():
    my_option = Option(call_put="call", european_american="european", S=100, K=95, V=0.15, R=0.002, T=1)
    my_option.pricing.get_price()                                 # black_sholes by default (only for european option)
    my_option.pricing.get_price(calc_input = "black_sholes")      # black_sholes
    my_option.pricing.get_price(calc_input = "monte_carlo")       # monte_carlo with 1000 iterations per run by default
    my_option.pricing.get_price(calc_input = "monte_carlo", step_mc_input = 700)  # monte_carlo with 700 iterations per run
    my_option.pricing.get_price(calc_input = "binomial_tree")     # binomial_tree with 1 step per day by default (i.e. with time*365//1 steps)
    my_option.pricing.get_price(calc_input = "binomial_tree", step_bt_input = 50) # binomial_tree with 50 steps
    my_option.pricing.get_delta_black_sholes()                    # black_sholes delta

if __name__ == '__main__':
    # option_graph()
    option_price()
    pricing_syntax()

# === END Usage example of the Class Option ===

