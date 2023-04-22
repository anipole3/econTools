import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class EdgeworthBox():
    """"This Library was built to draw Edgeworth boxes for different utility functions and ndowment points. An example how you can use this package is:
    edg_box = EdgeworthBox(type1='linear', params1= [0.5, 0.5], type2='cb', params2 =[2,3], endowment1=[2,7], endowment2=[7,3])
    fig, ax = plt.subplots(1,1, figsize=(12,6))
    edg_box.plot_all(ax)

    This will yield a graph with indifference curves, contract curves and pareto improvements area. 
    """
    def __init__(self, type1:str, 
                 params1:list, 
                 type2:str, 
                 params2:list, 
                 endowment1:tuple, 
                 endowment2:tuple) -> None:
        
        self.type1 = type1
        if (len(params1)!=2) or (len(params2)!=2):
            raise ValueError("The number of parameters for each function must be 2.")
        if (len(endowment1)!=2) or (len(endowment2)!=2):
            raise ValueError("This is a two-goods economy. Please add a 2dimensional endowment vector for each agent.")
        self.params11 = params1[0]
        self.params12 = params1[1]
        self.type2 = type2
        self.params21 = params2[0]
        self.params22 = params2[1]
        self.ea1 = endowment1[0]
        self.ea2 = endowment1[1]
        self.eb1 = endowment2[0]
        self.eb2 = endowment2[1]
        self.max1 = self.ea1 + self.eb1
        self.max2 = self.ea2 + self.eb2
        if (self.type1!='leontieff') and (self.type1!= 'cb') and (self.type1!='linear'):
            raise TypeError('The specified function type1 is not recognized. Please try: "leontieff", "cb" or "linear".')
        
        if (self.type2!='leontieff') and (self.type2!= 'cb') and (self.type2!='linear'):
            raise TypeError('The specified function type2 is not recognized. Please try: "leontieff", "cb" or "linear".')
       
    
    def cobb_douglasss(self, agent:str='a'):
        """building a cob-douglass utility function given the parameters
        and it returns the indifference curve that passes through 
        the endowment point"""

        if agent=='a':
            e1 = self.ea1
            e2 = self.ea2
            a1= self.params11
            a2 = self.params12
        else:
            e1 = self.eb1
            e2 = self.eb2
            a1 = self.params21 
            a2 = self.params22
       
        u2 = ((e1)**a1)*((e2)**a2)
        x= np.linspace(0, self.max1, 1000)
        y= []
        for nr in x:
            y_temp = (u2/(nr**a1))**(1/a2)
            y.append(y_temp)
        if agent=='a':
            return [x, np.array(y)]
        else:
            return [self.max1-x, self.max2-np.array(y)]
    
    def leontieff(self, agent:str='a'):
        """building a leontieff utility function given the parameters
        and it returns the indifference curve that passes through 
        the endowment point"""
        if agent=='a':
            e1 = self.ea1
            e2 = self.ea2
            a1 = self.params11
            a2 = self.params12
        else:
            e1 = self.eb1
            e2 = self.eb2
            a1 = self.params21
            a2 = self.params22
        
        u = min(e1*a1, e2*a2)
        x_corner= u/a1
        y_corner = u/a2
        curve11 = np.ones(1000)*x_corner
        curve12 = np.linspace(y_corner, self.max2, 1000)
        curve21 = np.ones(1000)*y_corner
        curve22 = np.linspace(x_corner, self.max1, 1000)

        if agent =='a':
            return curve11, curve12, curve21, curve22
        else:
            return self.max1 - curve11, self.max2-curve12, self.max2 - curve21, self.max1-curve22

    def linear(self, agent:str='a'):
        """building a linear utility function given the parameters
        and it returns the indifference curve that passes through 
        the endowment point"""
        if agent=='a':
            e1 = self.ea1
            e2 = self.ea2
            a1 = self.params11
            a2 = self.params12
        else:
            e1 = self.eb1
            e2 = self.eb2
            a1 = self.params21
            a2 = self.params22
        
        ul = e1*a1 + e2*a2
        xl = np.linspace(0, self.max1, 1000)
        yl = []
        for nr in xl:
            y_temp = (ul - nr*a1)/a2
            yl.append(y_temp)
        if agent=='a':
            return [xl, np.array(yl)]
        else:
            return [self.max1-xl, self.max2-np.array(yl)]
        
    def ces(self, agent:str = 'a'):
        pass
        
    
    def plot_indff_curves(self, ax, show=True):
        """this method plots the indifference curves of both agents
        that pass through the endowment points. """


        if self.type1 == 'cb':
            xa, ya = self.cobb_douglasss()
            ax.plot(xa, ya, label='agent A')
        elif self.type1=='linear':
            xa, ya = self.linear()
            ax.plot(xa, ya, label='agent A')
        else:
            x11, y11, y12, x12  = self.leontieff()
            ax.plot(x11, y11, color='b')
            ax.plot(x12, y12, color='b', label = 'agent A')
        
        if self.type2 == 'cb':
            xb, yb = self.cobb_douglasss(agent='b')
            ax.plot(xb, yb, label='agent B')
        elif self.type2 =='linear':
            xb, yb = self.linear(agent='b')
            ax.plot(xb, yb, label='agent B')
        else:
            x21, y21, x22, y22 = self.leontieff(agent='b')
            color= 'b'
            if self.type1!='cb':
                color='orange'
            ax.plot(x21, y21, color=color)
            ax.plot(y22, x22, color=color, label = 'agent B')
        
        # plt.legend()
        ax.annotate('e',xy = (self.ea1, self.ea2), xytext=(self.ea1+0.5, self.ea2+0.5), arrowprops= dict(
    arrowstyle = "->"), size=14)
        ax.scatter(self.ea1, self.ea2, label='endowment', color='r')
        ax.set_ylim(bottom=0, top=self.max2)
        ax.set_xlim(xmin=0, xmax=self.max1)
        
        if show:
            ax.legend()
            plt.show()
        else:
            return ax
    
    def contract_curve(self, ax2, show=True):
        """This method plots the contract curve of the economy if both agents have utility functions different from Leontieff. """
        x_contract = np.linspace(0, self.max1, 1000)
        y_contract = []
        lwidth=1
        if (self.type1=='cb') and (self.type2=='cb'):
            k1 = self.params21 * self.params12
            k2 = self.params22 * self.params11

            for nr in x_contract:
                y_cont = k1*self.max2*nr/(k2*(self.max1-nr) + k1*nr)
                y_contract.append(y_cont)

        elif (self.type1=='cb') and (self.type2=='linear'):
            for nr in x_contract:
                y_cont = (self.params12/self.params11)*(self.params21/self.params22)*nr
                y_contract.append(y_cont)
        elif (self.type2=='cb') and (self.type1=='linear'):
            for nr in x_contract:
                y_cont = (self.params11/self.params12)*(self.params22/self.params21)*nr
                y_contract.append(y_cont)
        elif (self.type2=='linear') and (self.type1=='linear'):
            lwidth=8
            if (self.params12/self.params11) >  (self.params22/self.params21):
                for nr in x_contract:
                    y_cont = self.max2
                    y_contract.append(y_cont)
                b = [0]
                c = [0]
                b.extend(list(x_contract))
                c.extend(y_contract)
                x_contract = b
                y_contract = c
            elif (self.params12/self.params11) <  (self.params22/self.params21):
                for nr in x_contract:
                    y_cont = 0
                    y_contract.append(y_cont)
                x_contract = list(x_contract)
                x_contract.append(self.max1)
                y_contract.append(self.max2)
        
        

        if ((self.type2=='linear') or (self.type2=='cb')) and ((self.type1=='linear') or (self.type1=='cb')):
            ax2.plot(x_contract, y_contract, label='contract curve', lw=lwidth)
            
        else:
            x_contract=0
        
        if show:
            ax2.legend()
            plt.show()
        else:
            return ax2
    
    def pareto_improvements(self, ax3, show=True):
        """This method highlights in the graph the area which consists of pareto improvement allocations for both agents."""

        if self.type1 == 'cb':
            xa, ya = self.cobb_douglasss()
        elif self.type1=='linear':
            xa, ya = self.linear()
        else:
            x11, y11, ya, xa = self.leontieff()
        
        if self.type2 == 'cb':
            xb, yb = self.cobb_douglasss(agent='b')
        elif self.type2 =='linear':
            xb, yb = self.linear(agent='b')
        else:
            x21, y21, yb, xb = self.leontieff(agent='b')
            
        xss = np.linspace(0, self.max1, 1000)
        x_vals = []
        y_vals = []
        i_s = []
        rangess = []
        y_range = np.linspace(np.min(ya), np.max(yb), 1000)
        for i in range(1000):
            xs1 = xss[i]
            yas = ya[i]
            ybs = yb[999 - i]
            if ~(self.type1=='cb' or self.type1=='linear') and (xs1<np.min(xa)):
                yas = self.max2
            elif ~(self.type2=='cb' or self.type2=='linear') and (xs1>np.max(xb)):
                ybs = 0
            elif ~(self.type1=='cb' or self.type1=='linear') and (xs1<np.min(xa)) and ~(self.type2=='cb' or self.type2=='linear') and (xs1>np.max(xb)):
                ybs = 0

            if (yas < ybs) and (yas<self.max2) and (ybs<self.max2):
                for y in y_range:
                    if (yas<y) and (y<ybs):
                        x_vals.append(xs1)
                        y_vals.append(y)
        ax3.scatter(x_vals, y_vals,color='m', alpha=0.05, s=1)
        
        if show:
            self.plot_indff_curves(ax3)
            ax3.legend()
            plt.show()
        else:
            return ax3
    def plot_all(self, ax, show=True):
        """This method plots everything: indifference curves, pareto improvements and the contract curve."""
        ax = self.plot_indff_curves(ax, show=False)
        ax = self.pareto_improvements(ax, show=False)
        ax = self.contract_curve(ax, show=False)
        
        if show:
            ax.legend()
            plt.show()
        else:
            ax.legend()