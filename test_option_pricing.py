# File : test_option_pricing.py
# Date : 06 Jan 2021

# To run the test file, launch command :
# python -m unittest test_module
# python -m unittest   # scan all the test to run in the folder

import unittest
import option_pricing as op

class TestOptionPricing(unittest.TestCase):
    def test_black_sholes(self):
        # Make sure Black & Sholes formula works fine for european, call and put, price and delta
        (S, K, V, T, R, mu) = (106, 100, 0.46, 0.75, 0.058, 0)
        
        my_option = op.Option(call_put="call", european_american="european", spot=S, strike=K, vol=V, time=T, rate=R, mu=mu)
        self.assertAlmostEqual(my_option.pricing.get_price("black_sholes"), 18.605989565586615) # price
        self.assertAlmostEqual(my_option.pricing.get_delta_black_sholes(), 0.6080877068685249)    # delta
        
        my_option = op.Option(call_put="put", european_american="european", spot=S, strike=K, vol=V, time=T, rate=R, mu=mu)
        self.assertAlmostEqual(my_option.pricing.get_price("black_sholes"), 12.861394241040822) # price
        self.assertAlmostEqual(my_option.pricing.get_delta_black_sholes(), -0.3493448472224421)    # delta

    def test_binomial_tree(self):
        # Make sure that Binomial Tree works fine for european and american, call and put, price
        (S, K, V, T, R, mu) = (106, 100, 0.46, 0.75, 0.058, 0)
        bt_step = 8
        
        my_option = op.Option(call_put="call", european_american="european", spot=S, strike=K, vol=V, time=T, rate=R, mu=mu)
        self.assertAlmostEqual(my_option.pricing.get_price("binomial_tree", bt_step), 18.734186394423723)
        
        my_option = op.Option(call_put="put", european_american="european", spot=S, strike=K, vol=V, time=T, rate=R, mu=mu)
        self.assertAlmostEqual(my_option.pricing.get_price("binomial_tree", bt_step), 12.989591069877914)
        
        my_option = op.Option(call_put="call", european_american="american", spot=S, strike=K, vol=V, time=T, rate=R, mu=mu)
        self.assertAlmostEqual(my_option.pricing.get_price("binomial_tree", bt_step), 18.989223075221474)
        
        my_option = op.Option(call_put="put", european_american="american", spot=S, strike=K, vol=V, time=T, rate=R, mu=mu)
        self.assertAlmostEqual(my_option.pricing.get_price("binomial_tree", bt_step), 13.091109333745027)

    def test_monte_carlo(self):
        # Make sure that Monte Carlo works fine for european, call and put, price
        (S, K, V, T, R, mu) = (106, 100, 0.46, 0.75, 0.058, 0)
        mc_step = 100000

        my_option = op.Option(call_put="call", european_american="european", spot=S, strike=K, vol=V, time=T, rate=R, mu=mu)
        self.assertLessEqual( my_option.pricing.get_price("monte_carlo", mc_step) - 18.606, 0.2)
        
        my_option = op.Option(call_put="put", european_american="european", spot=S, strike=K, vol=V, time=T, rate=R, mu=mu)
        self.assertLessEqual( my_option.pricing.get_price("monte_carlo", mc_step) - 12.861, 0.2)

