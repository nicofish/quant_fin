# Object  : Cash Flow class
# Version : v1.0
# Date    : 2021.01.05
# Python  : v3.7.7
# Author  : NicoFish

import datetime as dt

class CashFlow:
    """
    A cash flow is defined by three features :
    - A date     : when the cash flow is payed (can be past, current or future date)
    - A currency : in which currency the cash flow is payed (for instance EUR, USD, GBP, JPY, CHF, etc...)
    - An amount  : the amount of cash flow payed (can be positive (receive), null (no cash flow) or negative (pay))
      > Note : no rounding is done on amount.
        >> First reason is because rounding may differs upon currency (two figures after coma for EUR but zero for JPY)
        >> Second reason because it prevents some rounding issue when summing cash flows
        >> Hence if need be, rounding should be handled outside of CashFlow module
    """
    def __init__(self, date, currency, amount):
        """
        Check that the 3 variables are with relevant type and assign it as CashFlow features
        """
        self.set_date(date)
        self.set_currency(currency)
        self.set_amount(amount)        

    def get_cashflow(self):
        """return a tuple with the full cash flow information : date, currency and amount"""
        return (self.cf_date, self.cf_currency, self.cf_amount)

    def set_cashflow(self, date, currency, amount):
        """update the full cash flo features : date, currency and amount"""
        self.set_date(date)
        self.set_currency(currency)
        self.set_amount(amount)        

    def get_date(self):
        """return the date of the cash flow"""
        return self.cf_date

    def set_date(self, date):
        """check that date type is a datetime.date and assign it to date feature"""
        if type(date) == dt.date: self.cf_date = date
        else:
            print("date type =", type(date2))
            #raise TypeError("Cash Flow 'date' : should be a date")

    def get_currency(self):
        """return the currency of the cash flow"""
        return self.cf_currency

    def set_currency(self, currency):
        """check that currency type is a string and assign it to currency feature"""
        if isinstance(currency, str): self.cf_currency = currency 
        else: raise TypeError("Cash Flow 'currency' : should be a string")

    def get_amount(self):
        """return the amount of the cash flow"""
        return self.cf_amount

    def set_amount(self, amount):
        """check that amount type is a number (integer or float) and assign it to amount feature
        amount (input as integer or float) is automatically converted to float
        """
        if isinstance(amount, int) or isinstance(amount, float): self.cf_amount = float(amount)
        else: raise TypeError("Cash Flow 'amount' : should be a number (integer or float)")

def main():
    """Example of CashFlow class usage"""

    def strdate(date_string):
        """Convert a string into a date object"""
        date_format = "%Y.%m.%d"  # year.month.day
        date_object = dt.datetime.strptime(date_string, date_format).date()
        return date_object

    def datestr(date_object):
        """Convert a date object into a string"""
        date_format = "%Y.%m.%d" # year.month.day
        date_string = date_object.strftime(date_format)
        return date_string

    # define a cash flow
    date = strdate("2025.01.31") # 31 Jan 2025
    currency = "EUR"
    amount = 1000
    my_cashflow = CashFlow(date, currency, amount)

    # get cash flow features
    print("my_cash_flow =", my_cashflow.get_cashflow())
    print("date =", datestr( my_cashflow.get_date() ) )
    print("currency =", my_cashflow.get_currency())
    print("amount =", my_cashflow.get_amount())

    # update the full cash flow
    my_cashflow.set_cashflow(strdate("2030.06.30"), "USD", 2000 )  # 30 June 2030
    print("my_cash_flow =", my_cashflow.get_cashflow())

    # update the cash flow feature one by one
    my_cashflow.set_date(strdate("2020.12.31")) # 31 Dec 2020
    my_cashflow.set_currency('BTC')
    my_cashflow.set_amount(1.23456789)
    print("my_cash_flow =", my_cashflow.get_cashflow())

if __name__ == '__main__':
    main()
        