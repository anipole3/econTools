import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class BudgetConstraints():
    """"This Package was built to draw budget constraints along with indifference curves given the utiliy function, prices and endowment.
    An Example would be:

    bc = BudgetConstraints(tility_func="cb", params=[0.3, 0.7], endowment=20, prices=[2,3])
    #draw only budget line
    bc.budget_line()

    #draw indifference curve at the endowment
    bc.indff_curve()

    #draw optimal allocation:
    bc.optimal_allocation()
    
    """
    def __init__(self, utility_func, params, endowment, prices) -> None:
        if (len(params)!=2):
            raise ValueError("The number of parameters for each function must be 2.")
        if (utility_func!='leontieff') and (utility_func!= 'cb') and (utility_func!='linear'):
            raise TypeError('The specified utility function is not recognized. Please try: "leontieff", "cb" or "linear".')
        if (len(prices)!=2):
            raise ValueError("This is a two-goods economy. Please add a 2dimensional price vector.")
        self.utility_func = utility_func
        self.params = params
        self.endowment = endowment
        self.prices = prices
    
    def budget_line(self, show=True):
        """This method draws the budget line given the endowment and prices."""
        x_0 = self.endowment/self.prices[0]
        y_0 = self.endowment/self.prices[1]
        xs = np.linspace(0, x_0, 1000)
        ys = np.linspace(y_0,0, 1000)
        if show:
            plt.plot(xs, ys, color='black', label='budget line')
            plt.ylim(bottom=0, top=ys[0]*3)
            plt.xlim(left=0, right=xs[-1]+2)
            plt.legend()
            plt.show()
        else:
            return xs, ys
    
    def indff_curve(self):
        """This method draws the budget line given the endowment and utility function."""
        if self.utility_func == 'cb':
            x_bc, y_bc = self.budget_line(show=False)
            x_ic, y_ic = self._cobb_douglass([x_bc[300], y_bc[500]])
            plt.plot(x_bc, y_bc, color='black', label='budget line')
            plt.plot(x_ic, y_ic, label='indifference curve')
            plt.ylim(bottom=0, top=y_bc[0]*3)
            plt.xlim(left=0, right=x_bc[-1]+2)
            plt.legend()
            plt.show()
        elif self.utility_func == 'linear':
            x_bc, y_bc = self.budget_line(show=False)
            x_ic, y_ic = self._linear([x_bc[300], y_bc[500]])
            plt.plot(x_bc, y_bc, color='black', label='budget line')
            plt.plot(x_ic, y_ic, label='indifference curve')
            plt.ylim(bottom=0, top=y_bc[0]*3)
            plt.xlim(left=0, right=x_bc[-1]+2)
            plt.legend()
            plt.show()
        elif self.utility_func == 'leontieff':
            x_bc, y_bc = self.budget_line(show=False)
            x_ic, y_ic = self._leontieff([x_bc[300], y_bc[500]])
            plt.plot(x_bc, y_bc, color='black', label='budget line')
            plt.plot(x_ic[0]*np.ones(1000), y_ic, color='orange')
            plt.plot(x_ic, y_ic[0]*np.ones(1000), color='orange', label='indifference curve')
            plt.ylim(bottom=0, top=y_bc[0]*3)
            plt.xlim(left=0, right=x_bc[-1]+2)
            plt.legend()
            plt.show()

    
    def _cobb_douglass(self, inputs):
        """This method extracts the indifference curve passing through the endowment point given current prices if the function is Cobb-Duglass."""
        u = (inputs[0]**self.params[0])*(inputs[1]**self.params[1])
        x_s = np.linspace(1e-5, self.endowment/self.prices[0], 1000)
        keep_x = []
        y_s = []
        for i in range(1000):
            y_val = (u/(x_s[i]**self.params[0]))**(1/self.params[1])
            y_s.append(y_val)
            keep_x.append(i)
        return x_s, y_s
    
    def _linear(self, inputs):
        """This method extracts the indifference curve passing through the endowment point given current prices if the function is Linear."""
        u = (inputs[0]*self.params[0]) + (inputs[1]*self.params[1])
        x_s = np.linspace(0, self.endowment/self.prices[0], 1000)
        y_s = []
        for nr in x_s:
            y_val = (u - (nr*self.params[0]))/(self.params[1])
            y_s.append(y_val)
        return x_s, y_s
    
    def _leontieff(self, inputs):
        """This method extracts the indifference curve passing through the endowment point given current prices if the function is Leontieff."""
        u = min((inputs[0]*self.params[0]),(inputs[1]*self.params[1]))
        x_s = np.linspace(u/self.params[0], self.endowment/self.prices[0] + 1, 1000)
        y_s = np.linspace(u/self.params[1], 3*(self.endowment/self.prices[1]), 1000)
        return x_s, y_s
    
    def optimal_allocation(self):
        """This method find the optimal allocation of the agent and draws it."""
        if self.utility_func=='cb':
            y_0 = (self.endowment/self.prices[1])*(self.params[1]/(self.params[1]+self.params[0]))
            x_0 = (self.endowment/self.prices[0])*(self.params[0]/(self.params[1]+self.params[0]))
            x_ic, y_ic = self._cobb_douglass([x_0, y_0])
            x_bc, y_bc = self.budget_line(show=False)
            max_utility = (x_0**self.params[0])*(y_0**self.params[1])
        if self.utility_func == 'linear':
            if self.params[0]/self.params[1] > self.prices[0]/self.prices[1]:
                x_0 = (self.endowment/self.prices[0])
                y_0= 0
            elif self.params[0]/self.params[1] < self.prices[0]/self.prices[1]:
                y_0 = (self.endowment/self.prices[1])
                x_0= 0
            else:
                print('Any allocation on the budget line can be optimal.')
            max_utility = self.params[0]*x_0 + self.params[1]*y_0
            x_ic, y_ic = self._linear([x_0, y_0])
            x_bc, y_bc = self.budget_line(show=False)
        if self.utility_func=='leontieff':
            x_bc, y_bc = self.budget_line(show=False)
            x_0 = (self.endowment/self.prices[1])/(self.params[0]/self.params[1]+self.prices[0]/self.prices[1])
            y_0 = self.params[0]/self.params[1] * x_0
            x_ic, y_ic = self._leontieff([x_0, y_0])
            max_utility = min(self.params[0]*x_0, self.params[1]*y_0)
        print(f"Max utility = {max_utility}")
        plt.plot(x_bc, y_bc, color='black', label='budget line')
        if self.utility_func=='leontieff':
            plt.plot(x_ic[0]*np.ones(1000), y_ic, color='orange')
            plt.plot(x_ic, y_ic[0]*np.ones(1000), color='orange', label='indifference curve')
        else:
            plt.plot(x_ic, y_ic, label='indifference curve')
        plt.scatter(x_0, y_0, color='r', label='optimal allocation')
        plt.ylim(bottom=0, top=y_bc[0]*3)
        plt.xlim(left=0, right=x_bc[-1]+2)
        plt.legend()
        plt.show()