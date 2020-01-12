# L0-regression
## use gurobi to perform best subset selection.
---
### How to run it?
- Step 1: Install Gurobi into Anaconda

    From an Anaconda terminal issue the following command to add the Gurobi channel to your default search list:

        conda config --add channels http://conda.anaconda.org/gurobi
    Now issue the following command to install the Gurobi package:

        conda install gurobi
    You can remove the Gurobi package at any time by issuing the command:

        conda remove gurobi

- Step 2: Install a Gurobi License

    obtain Individual Academic Licenses of Gurobi on the website. [Here](https://www.gurobi.com/academia/academic-program-and-licenses/).

- Step 3:

    - run for specific constraint k<=10:
        ```
        model = L0Regression()
        model.read_data(__addrees1)
        model.train_test_split(test_size=0.20, random_state=123)
        model.premodel()
        model.miqp(10, timelimit=10)
        model.summary()
        ```

        Output:

            load the data successfully
            begin to optimize, will use:  10 s
            indentity [68]
            <ufunc 'sqrt'> [12, 44, 49, 72, 90]
            <ufunc 'square'> [4, 28, 67, 82]

    - run a path solution to find the best k:

            model = L0Regression()
            model.read_data(__addrees1)
            model.train_test_split(test_size=0.20, random_state=123)
            model.premodel()
            model.select_model(nonzero_range=range(10, 12))

        Output:

            load the data successfully
            Using license file /Users/zhoutang/gurobi.lic
            Academic license - for non-commercial use only
            the max number of non zero beta is: 47
            current k 10
            begin to optimize, will use:  10 s
                Rsquare       MSE         AIC         BIC  nonzero
            11  0.735482  0.009388  471.557279  514.605812       11
            ================================
            current k 11
            warm_up at k =  11
            begin to optimize, will use:  10 s
                Rsquare       MSE         AIC         BIC  nonzero
            12  0.726495  0.009734  485.918185  532.880221       12
            ================================
                    Rsquare       MSE         AIC          BIC  nonzero
            10     0.735482  0.009388  471.557279   514.605812     11.0
            11     0.726495  0.009734  485.918185   532.880221     12.0
            ols    0.734554  0.012573  652.853044  1048.116847     99.0
            lasso  0.739507  0.011687  615.883992   952.445251     85.0

