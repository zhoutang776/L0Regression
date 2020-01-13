# Paper Reimplementation in Best Subset Selection

### [Bertsimas, Dimitris, Angela King, and Rahul Mazumder. "Best subset selection via a modern optimization lens." The annals of statistics 44.2 (2016): 813-852.](https://projecteuclid.org/download/pdfview_1/euclid.aos/1458245736)

---
### Implementation Details:

1. Using Projected Gradient Descent methods to find a heuristic solution.
    
2. Using three methods to find a upper bound of L<sub>1</sub> and L<sub>inf</sub> norm for &beta; and X&beta;
    1. Coherence and Restricted Eigenvalue 
    (there is a typo in the paper 2.12(c), should be sqrt(n)*norm(y) not sqrt(k)*norm(y))
    2. Convex Optimization
    3. heuristic-solution-based Bound


---

### How to run it?
- Step 1: Install Gurobi License and gurobipy into Anaconda 

    Obtain Individual Academic Licenses of Gurobi on the website. [Here](https://www.gurobi.com/academia/academic-program-and-licenses/).

    From an Anaconda terminal issue the following command to add the Gurobi channel to your default search list:

        conda config --add channels http://conda.anaconda.org/gurobi
    Now issue the following command to install the Gurobi package:

        conda install gurobi
    You can remove the Gurobi package at any time by issuing the command:

        conda remove gurobi

- Step 2:

    - Run for simple dataset:
        ```
        n, p = 100, 20
        X = rng.random(size=(n, p))
        coef = np.zeros(p)
        coef[[0, 2, 4]] = 1
        coef[[1, 3, 5]] = 2
        intercept = 3
        error = rng.randn(n)
        y = np.dot(X, coef) + intercept + error
        y = y.reshape(n, )
        mip_model = L0Regression(verbose=False).fit(X, y, k=6)
        mip_error = sum((mip_model.coef_ - coef)**2)
        mip_error += (mip_model.intercept_ - intercept)**2
        print("best subset selection:")
        print("coef:", mip_model.coef_, "intercept:", mip_model.intercept_, "error:", mip_error, sep="\n")
        print()
        print("LASSO:")
        lasso_model = LassoCV(cv=10).fit(X, y)
        lasso_error = sum((lasso_model.coef_ - coef)**2)
        lasso_error += (lasso_model.intercept_ - intercept)**2
        print("coef:", lasso_model.coef_, "intercept:", lasso_model.intercept_, "error:", lasso_error, sep="\n")
        ```

        Output:
        ```
        Using license file /Users/zhoutang/gurobi.lic
        Academic license - for non-commercial use only
        Changed value of parameter TimeLimit to 60.0
           Prev: inf  Min: 0.0  Max: inf  Default: inf
        Changed value of parameter MipGap to 1e-05
           Prev: 0.0001  Min: 0.0  Max: inf  Default: 0.0001
        cumulative coherence larger than 1, bound 1 fails
        best subset selection:
        coef:
        [1.00157844 2.23547144 1.33017972 2.18760993 0.9282457  2.16010474
         0.         0.         0.         0.         0.         0.
         0.         0.         0.         0.         0.         0.
         0.         0.        ]
        intercept:
        2.4966815408189174
        error:
        0.48377709899904553
        
        LASSO:
        coef:
        [ 0.63580476  1.80683851  0.94027029  1.8444559   0.55423598  1.73261153
          0.          0.         -0.         -0.          0.         -0.
          0.          0.31048407 -0.         -0.          0.          0.
         -0.          0.55432892]
        intercept:
        3.221779335838728
        error:
        0.9207802752097756
        ```
      

