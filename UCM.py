import numpy as np
from scipy.special import beta, hyp2f1
import argparse

def ucm(dof, r):
    term1 = dof / (2 * (dof - 2) * r**2)
    term2 = 1 / ((2 + dof) * beta(dof/2, dof/2)) * r**(dof/2 - 1) * hyp2f1(dof, dof/2 + 1, dof/2 + 2, -r)
    term3 = 1 / ((dof - 2) * beta(dof/2, dof/2)) * r**(dof/2 - 1) * hyp2f1(dof, dof/2 - 1, dof/2, -r)
    
    return (term1 - term2) / (term1 - term2 + term3)

def Prob(dof, R):
    u = ucm(dof, R)
    return u / (1 - u)

def ProbList(dof, cslist):
    min_value = np.min(cslist)
    ratio = np.array(cslist) / min_value
    probs = np.array([Prob(dof, r) for r in ratio])
    sum_probs = np.sum(probs)
    return probs / sum_probs

def main():
    parser = argparse.ArgumentParser(description="Calculate UC Model probabilities based on input degrees of freedom and chi-squared list.",
                                    epilog="Example of use: python UCM.py --dof 10 --cslist 1.2 2.5 3.7 4.1")
    parser.add_argument('--dof', type=int, required=True, help='Degrees of freedoom. It should be a positive integer.')
    parser.add_argument('--cslist', nargs='+', type=float, required=True, help='Input list of reduced chi-squared values')

    args = parser.parse_args()

    # Variables are directly usable
    cslist = args.cslist
    dof = args.dof

    # Calculate probabilities
    probs = ProbList(dof, cslist)

    # Print probabilities
    print("Probabilities list:")
    for index, prob in enumerate(probs, start=1):
        print(f"Element {index}: {prob:.4f}")

if __name__ == "__main__":
    main()
