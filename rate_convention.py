# Object  : Discount factor, capitalisation factor, rate convention, rate conversion
# Version : v1.1
# Date    : 2021.01.06
# Python  : v3.7.7
# Author  : NicoFish

from math import exp, log
import string

def capitalisation_factor(rate_convention, rate, time):
    """
    Compute capitalisation factor based on parameters :
    - rate convention : exponential, yield, linear, discount
    - rate : interest rate. For instance r = 0.035 for 3.5 %
    - time : time in year. For instance t = 0.75 for 9 months
    """
    rate_convention = rate_convention.lower()
    if rate_convention in [ "exp", "exponential" ] :
        cf = exp( rate * time )
    elif rate_convention in [ "yld", "yield" ] :
        cf = (1 + rate ) ** time
    elif rate_convention in [ "lin", "linear" ] :
        cf = 1 + rate * time
    elif rate_convention in [ "dsc", "discount" ] :
        if rate *  time != 1 :
            cf = 1 / (1 - rate *  time)
        else :
            cf = None
    else :
        cf = None
    return cf

def discount_factor(rate_convention, rate, time) :
    """
    Compute discount factor based on parameters :
    - rate convention : exponential, yield, linear, discount
    - rate : interest rate. For instance r = 0.035 for 3.5 %
    - time : time in year. For instance t = 0.75 for 9 months
    """
    cf = capitalisation_factor(rate_convention, rate, time)
    if cf != None and cf != 0 :
        df = 1 / cf
    else :
        df = None
    return df

def capitalisationfactor_to_rate(rate_convention, capitalisation_factor, time):
    """
    Convert capitalisation factor to interest rate.
    Parameters are :
    - rate convention : exponential, yield, linear, capitalisation
    - capitalisation factor
    - time : time in year. For instance t = 0.75 for 9 months
    """
    rate_convention = rate_convention.lower()
    if rate_convention in [ "exp", "exponential" ] and capitalisation_factor >= 0.0 and time != 0 :
        rate = log(capitalisation_factor) / time
    elif rate_convention in [ "yield", "yld" ] and time != 0 :
        rate = capitalisation_factor ** (1/time) - 1
    elif rate_convention in [ "linear", "lin" ] and time != 0 :
        rate = (capitalisation_factor - 1) / time
    elif rate_convention in [ "capitalisation", "dsc" ] and capitalisation_factor != 0 and time != 0:
        rate = (1 - 1/capitalisation_factor) / time
    else :
        rate = None
    if rate != None:
        rate = round(rate,15)
    return rate

def discountfactor_to_rate(rate_convention, discount_factor, time):
    """
    Convert discount factor to interest rate.
    Parameters are :
    - rate convention : exponential, yield, linear, discount
    - discount factor
    - time : time in year. For instance t = 0.75 for 9 months
    """
    if discount_factor != 0:
        capitalisation_factor = 1 / discount_factor
        rate = capitalisationfactor_to_rate(rate_convention, capitalisation_factor, time)
    else:
        rate = None
    return rate

def rate_conversion(rate_convention_source, rate_convention_target, rate_source, time) :
    """
    Convert interest rate from one rate convention (source) to another rate convention (target) 
    """
    discount_factor_int = discount_factor(rate_convention_source, rate_source, time)
    rate_target = discountfactor_to_rate(rate_convention_target, discount_factor_int, time)
    return rate_target

def example():
    """ Example of dicount factor and rate conversion"""
    rate = 0.035   # rate = 3.5%
    time = 2.5     # time = 2.5 years = 30 months 
    for rate_convention in [ "exp", "yld", "lin", "dsc" ] :
        # Example of discount factor computation
        df = discount_factor(rate_convention, rate, time)
        print("Discount Factor", rate_convention, " =", df)
        # Example of rate computation based on discount factor
        rate_df = discountfactor_to_rate(rate_convention, df, time)
        print("Rate", rate_convention, " =", rate_df)

    # Example of rate conversion
    rate_convention_source = "exp"
    rate_convention_target = "lin"
    rate_target = rate_conversion(rate_convention_source, rate_convention_target, rate, time)
    print("on", time, "year(s) : rate", rate_convention_source, "=", rate, "<-> rate", rate_convention_target, "=", rate_target)

def main():
    example()

if __name__ == '__main__':
    main()
