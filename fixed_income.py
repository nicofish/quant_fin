# Object  : Bond modelisation : Cash flows, Market value, PV01, Yield, Macaulay duration and Modified duration
# Version : v1.0
# Date    : 2021.01.06
# Python  : v3.7.7
# Author  : NicoFish

import rate_convention as d

class Bond:
    """
    Bond is modelised as a  simplified bond, with following hypothesis:
    - constant coupon rate (not floating coupon rate)
    - integer time_to_maturity (short/long breaking period are not handled), for instance 3 years or 10 years (but not 6.5 years)
    Bond computes cash flows, market value, pv01, yield, duration and modified duration
    """
    def __init__(self, nominal, coupon_rate, time_to_maturity):
        self.nominal = nominal
        self.coupon_rate = coupon_rate
        self.time_to_maturity = int(time_to_maturity)
        # self.rate_convention = rate_convention

    def cash_flow(self):
        """
        Compute the flows of the bond, i.e
        time is expected to be an integer (short/long breaking period are not handled)
        """
        nominal = self.nominal
        coupon_rate = self.coupon_rate
        time_to_maturity = self.time_to_maturity
        cash_flow = []
        coupon_amount = nominal * coupon_rate
        for t in range(1, 1 + time_to_maturity, 1):
            if t < time_to_maturity:
                flow = (t, coupon_amount)
            elif t == time_to_maturity:
                flow = (t, coupon_amount + nominal)
            cash_flow.append(flow)
        return cash_flow

    def market_value(self, discount_rate, discount_rate_convention = "yld"):
        """
        Compute theoretical market value of the bond
        All the flow are discounted with the same interest rate, expected to be constant
        """
        market_value = 0
        for flow in self.cash_flow():
            time = flow[0]
            cash = flow[1]
            df = d.discount_factor(discount_rate_convention, discount_rate, time)
            market_value += cash * df
        return market_value

    def pv01(self, discount_rate, discount_rate_bump = 0.01):
        """
        PVO1 is the market_value sensitivity when discount rate is changing
        By default PV01 is return for a variation of 1 point of the discount rate (bump = 0.01)
        To get PV01 for bump of 1 basis point for instance, choose bump = 0.0001
        Computation itself is done with a rate bump of 0.00001, i.e. 0.001 % = 0.001 point = 0.1 basis point 
        """
        market_value_0 = self.market_value(discount_rate, "yld")
        epsilon = 0.00001
        market_value_1 = self.market_value(discount_rate + epsilon, "yld")
        PV01 = ( market_value_1 - market_value_0 ) / epsilon * discount_rate_bump
        return PV01

    def yield_rate(self, market_value_target):
        """
        Return the yield of the bond.
        - yield is the rate at which the cash flows should be dicounted so that total of discounted cash flow = market_value
        - rate convention for the yield discounting is... yield
        - Note : yield being a keyword in python, the function has been called yield_return
        """
        yield_rate = self.coupon_rate # initial value of the yield
        count = 0
        count_max = 50 # maximum number of iteration
        market_value_precision = 0.0001
        finished = False
        # Apply Newton-Raphson method to find yield so that market_value_yield = market_value_target
        while not finished :
            count += 1
            market_value_yield = self.market_value(yield_rate,"yld")
            pv01 = self.pv01(yield_rate, 1)
            # print("yield_rate =", yield_rate, "market_value_yield =", market_value_yield)
            if pv01 != 0:
                yield_rate = yield_rate + (market_value_target - market_value_yield) / pv01 
            finished = (abs(market_value_target - market_value_yield) < market_value_precision) or (count > count_max)
        return yield_rate

    def macaulay_duration(self, discount_rate):
        """
        Compute the macaulay duration of the bond, i.e. the average discounted cash flow *time* weigthed by the discounted cash flow
        """
        total_time_cash = 0
        total_cash = 0
        # print("duration / cash flow =", self.cash_flow() )
        for flow in self.cash_flow():
            time = flow[0]
            cash = flow[1]
            # print("time =", time, "cash =", cash)
            df = d.discount_factor("yld", discount_rate, time)
            total_time_cash += time * cash * df
            total_cash += cash * df
            # print("total_time_cash =", total_time_cash, "total_cash =", total_cash)
        if total_cash != 0:
            duration = total_time_cash / total_cash
        else:
            duration = None
        return duration

    def modified_duration(self, discount_rate):
        """
        Return the modified duration of the bond
        """
        coupon_yearly_frequency = 1 # only bond with yearly payment (one payment per year) are modelised
        market_value = self.market_value(discount_rate)
        yield_rate = self.yield_rate(market_value)
        macaulay_duration = self.macaulay_duration(discount_rate)
        modified_duration = macaulay_duration / (1 + yield_rate / coupon_yearly_frequency)
        # print("MV =", market_value, "/ yield =", round(yield_rate,5),
        #       "/ macaulay duration =", macaulay_duration, "/ modified duration =", modified_duration)
        return modified_duration

def bond_example():
    nominal = 100000
    coupon_rate = 0.04      # 4%
    time_to_maturity = 10   # 10 years
    digit_amount = 2               # for rounding display purpose only
    digit_rate = 6
    my_bond = Bond(nominal, coupon_rate, time_to_maturity)

    cash_flow = my_bond.cash_flow()
    print("> coupon =", coupon_rate, "maturity =", time_to_maturity, "year(s) : cash flow =", cash_flow)

    discount_rate = 0.06    # 6%
    # rate_convention = "yld"
    market_value = my_bond.market_value(discount_rate)
    pv01 = my_bond.pv01(discount_rate)
    print("> discount rate =", discount_rate, ": market value =", round(market_value, digit_amount), ": pv01 =", round(pv01, digit_amount))

    for market_value_yield in [ market_value, nominal, nominal * 1.05 ]: 
        yield_rate = my_bond.yield_rate(market_value_yield)
        print("> market value =", round(market_value_yield, digit_amount),": yield_rate =", round(yield_rate, digit_rate))

    macaulay_duration = my_bond.macaulay_duration(discount_rate)
    modified_duration = my_bond.modified_duration(discount_rate)
    print("> discount rate =", discount_rate, ": macaulay duration =", macaulay_duration, ": modified duration =", modified_duration)
    print("> Control : - PV01 x 100 / market_value =", 100 * pv01 / market_value, ": almost the same as modified duration")

def main():
    bond_example()

if __name__ == '__main__':
    main()
