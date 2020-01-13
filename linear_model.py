from gurobipy import *
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.metrics import mean_squared_error as mse
from IPython import embed
# # read the dataset
# # design matrix without intercept term, shape: (sample size, #predictors+1), 1 for response at last
# def read_data(address, scale=False):
#     """
#     :param address: dataset address
#     :param scale: standardize the data or not
#     :param threshold: use for constrain the maximum of pairwise correlation. (a method to solve multicolinearity)
#     :return: None
#     """
#     data = pd.read_csv(address)
#
#     if scale:
#         # with_mean=False, otherwise we cannot perform sqrt transformation
#         scaler = StandardScaler(with_mean=False, with_std=True)
#         data = scaler.fit_transform(data)
#         data = pd.DataFrame(data)
#
#     # training set
#     features = data.iloc[:, 0:-1]
#     response = data.iloc[:, -1]
#     print("load the data successfully")
#     # response = pd.DataFrame(np.sqrt(data.iloc[:, -1])).iloc[:, 0]
#     return None
#
# def train_test_split(test_size=0.20, random_state=123):
#     self.Xtrain, self.Xtest, self.ytrain, self.ytest = train_test_split(
#         self.features, self.response, test_size=test_size, random_state=random_state)


class L0Regression:
    def __init__(self, fit_intercept=True, normalize=False, verbose=False, max_corr=0.7, tau=2,
                 transformations={lambda x: x, }, mipgap=1e-5, timelimit=60):
        """
        :param fit_intercept: bool, optional, default True
        Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations
        (i.e. data is expected to be centered).
        :param normalize: bool, optional, default False
        This parameter is ignored when fit_intercept is set to False. If True, the regressors X will be normalized
        before regression by subtracting the mean and dividing by the l2-norm. If you wish to standardize, please use
        sklearn.preprocessing.StandardScaler before calling fit on an estimator with normalize=False.
        :param transformations:
        """
        # if transformations contains log and sqrt, please take care of the negative value and 0
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.verbose = verbose
        self.max_corr_ = max_corr
        self.tau = tau
        self.transformations = transformations

        self.regressor = Model("linear regression")
        self.regressor.params.TimeLimit = timelimit
        self.regressor.params.MipGap = mipgap

        self.coef_ = None
        self.intercept_ = None

    # find the maximum k
    def _premodel(self, X):
        regressor = Model("find the maximum k")
        num_samples, num_features = X.shape
        correlation_matrix = np.corrcoef(X.T)

        # if beta_i == 0, then indicators_i == 1
        indicators = regressor.addVars(num_features, vtype=GRB.BINARY, name="indicators")

        regressor.addConstrs(
            (indicators[i] + indicators[j] >= 1 for i in range(num_features) for j in range(i + 1, num_features)
             if abs(correlation_matrix[i, j]) > self.max_corr_), "Limited Pairwise Multicollinearity"
        )
        regressor.setObjective(indicators.sum(), GRB.MAXIMIZE)
        regressor.params.OutputFlag = 0
        regressor.optimize()
        max_k = sum([indicators[i].X for i in range(num_features)])
        max_k = int(max_k)
        return max_k

    def _warm_start(self, X, y, k, warmgap):
        n, p = X.shape
        beta = np.zeros(p)

        quad = np.dot(X.T, X)
        lin = np.dot(X.T, y)
        lip = np.linalg.eigvalsh(np.dot(X.T, X))
        lip = lip[-1]
        while True:
            beta_old = beta
            beta = beta - 1/lip * (np.dot(quad, beta) - lin)
            beta[np.argsort(beta)[0:p-k]] = 0

            if 2 * np.dot(beta - beta_old, lin) - np.dot(np.dot(beta.T, quad), beta) + \
                    np.dot(np.dot(beta_old.T, quad), beta_old) < 2 * warmgap:
                break
        obj_val = 0.5 * np.linalg.norm(y - X@beta)**2
        return beta, obj_val

    def _compute_bound_1(self, X, y, k):
        # X must have unit l2 norm
        # compute the coherence and restricted eigenvalues
        coherence = np.max(np.triu(np.abs(np.corrcoef(X.T)), k=1))
        if coherence*(k-1) >= 1:
            print("cumulative coherence larger than 1, bound 1 fails")
            return np.nan, np.nan, np.nan, np.nan
        restricted_eigen = 1 - coherence * (k-1)

        Xy = np.abs(X.T @ y)
        # compute beta l1 norm bound
        temp = Xy[np.argsort(Xy)[-k:]]
        ub_coef_1 = 1 / restricted_eigen * sum(temp)
        ub_coef_inf = min(np.linalg.norm(temp)/restricted_eigen,
                          np.linalg.norm(y)/np.sqrt(restricted_eigen))

        # paper typo in here, np.sqrt(n) not np.sqrt(k)
        ub_Xcoef_1 = min(sum(np.max(np.abs(X), axis=1)) * ub_coef_1,
                         np.linalg.norm(y) * np.sqrt(n))
        ub_Xcoef_inf = max(np.sum(np.sort(np.abs(X), axis=1)[:, -k:], axis=1)) * ub_coef_inf

        return ub_coef_1, ub_coef_inf, ub_Xcoef_1, ub_Xcoef_inf

    def _compute_bound_2(self, X, y, k, ub):
        # supplement section of https://arxiv.org/pdf/1507.03133.pdf
        n, p = X.shape
        assert len(X.shape) == 2 and X.shape[0] == y.shape[0] and len(y.shape) == 1
        if n <= p:
            print("n <= p, bound 2 fails")
            return np.nan, np.nan, np.nan, np.nan
        inv_XtX = np.linalg.inv(X.T @ X)
        P = X @ inv_XtX @ X.T
        beta = inv_XtX @ (X.T @ y)

        tao = np.sqrt((2*ub - y.T @ (np.eye(n) - P) @ y) / np.diag(inv_XtX))
        beta_min = np.diag(beta.reshape(-1, 1) - inv_XtX * tao.reshape((1, p)))
        beta_max = np.diag(beta.reshape(-1, 1) + inv_XtX * tao.reshape((1, p)))
        beta_bound = np.maximum(np.abs(beta_min), np.abs(beta_max))
        ub_coef_1 = sum(np.sort(beta_bound)[-k:])
        ub_coef_inf = max(beta_bound)

        tao = np.sqrt((2*ub - y.T @ (np.eye(n) - P) @ y) * np.sum((X@inv_XtX)*X, axis=1))
        Xbeta_min = X@beta - tao
        Xbeta_max = X@beta + tao
        Xbeta_bound = np.maximum(np.abs(Xbeta_min), np.abs(Xbeta_max))
        ub_Xcoef_1 = sum(Xbeta_bound)
        ub_Xcoef_inf = max(Xbeta_bound)
        return ub_coef_1, ub_coef_inf, ub_Xcoef_1, ub_Xcoef_inf

    def _compute_bound_3(self, X, y, k, coef_start, tau):
        n, p = X.shape
        ub_coef_1 = k * tau * max(np.abs(coef_start))
        ub_coef_inf = tau * max(np.abs(coef_start))
        ub_Xcoef_1 = min(sum(np.max(np.abs(X), axis=1)) * ub_coef_1,
                         np.linalg.norm(y) * np.sqrt(n))
        ub_Xcoef_inf = max(np.sum(np.sort(np.abs(X), axis=1)[:, -k:], axis=1)) * ub_coef_inf
        return ub_coef_1, ub_coef_inf, ub_Xcoef_1, ub_Xcoef_inf

    def fit(self, X, y, k):
        X = np.array(X)
        y = np.array(y).reshape(-1)
        assert len(X.shape) == 2, "Expected 2D array. Reshape your data either using array.reshape(-1, 1) " \
                                  "if your data has a single feature or " \
                                  "array.reshape(1, -1) if it contains a single sample."
        assert X.shape[0] == y.shape[0]
        num_samples, num_features = X.shape

        num_transformations = len(self.transformations)
        augment_features = map(lambda func: func(X), self.transformations)
        augment_features = np.concatenate(list(augment_features), axis=1)

        # normalize the augment_features, and will recover the coefficient lastly.
        X_std = np.std(augment_features, axis=0)
        X_mean = np.mean(augment_features, axis=0)
        y_mean = y.mean()
        y_std = y.std()
        augment_features = (augment_features - X_mean) / X_std
        y = (y - y_mean) / y_std

        num_var = augment_features.shape[1]
        max_k = self._premodel(augment_features)

        assert isinstance(k, (int, np.integer))
        assert k <= num_var and k <= max_k

        coef = self.regressor.addVars(num_var, lb=-GRB.INFINITY, name="coefficients")

        # indicator variable if coefficient_i == 0, then indicators_i == 1
        indicators = self.regressor.addVars(num_var, vtype=GRB.BINARY, name="indicators")
        for i in range(num_var):
            self.regressor.addSOS(GRB.SOS_TYPE1, [coef[i], indicators[i]])
        self.regressor.addConstr(indicators.sum() >= num_var - k)

        # initialization
        coef_start, obj_val = self._warm_start(augment_features, y, k=k, warmgap=1e-3)
        for i in range(num_var):
            indicators[i].start = (abs(coef_start[i]) < 1e-5)

        # compute l1 norm bound and infinity bound
        ub_coef_1_1, ub_coef_inf_1, ub_Xcoef_1_1, ub_Xcoef_inf_1 = self._compute_bound_1(augment_features, y, k)
        ub_coef_1_2, ub_coef_inf_2, ub_Xcoef_1_2, ub_Xcoef_inf_2 = self._compute_bound_2(augment_features, y, k, obj_val)
        ub_coef_1_3, ub_coef_inf_3, ub_Xcoef_1_3, ub_Xcoef_inf_3 = self._compute_bound_3(augment_features, y, k, coef_start, self.tau)
        if self.verbose:
            print("Bound 1:", ub_coef_1_1, ub_coef_inf_1, ub_Xcoef_1_1, ub_Xcoef_inf_1)
            print("Bound 2:", ub_coef_1_2, ub_coef_inf_2, ub_Xcoef_1_2, ub_Xcoef_inf_2)
            print("Bound 3:", ub_coef_1_3, ub_coef_inf_3, ub_Xcoef_1_3, ub_Xcoef_inf_3)

        ub_coef_1 = np.nanmin([ub_coef_1_1, ub_coef_1_2, ub_coef_1_3])
        ub_coef_inf = np.nanmin([ub_coef_inf_1, ub_coef_inf_2, ub_coef_inf_3])
        ub_Xcoef_1 = np.nanmin([ub_Xcoef_1_1, ub_Xcoef_1_2, ub_Xcoef_1_3])
        ub_Xcoef_inf = np.nanmin([ub_Xcoef_inf_1, ub_Xcoef_inf_2, ub_Xcoef_inf_3])

        # print(ub_coef_1, ub_coef_inf, ub_Xcoef_1, ub_Xcoef_inf)

        # infinity norm constraint
        for i in range(num_var):
            self.regressor.addRange(coef[i], -ub_coef_inf, ub_coef_inf)
        for i in range(num_samples):
            self.regressor.addConstr(quicksum(augment_features[i, j]*coef[j] for j in range(num_var)) <= ub_Xcoef_inf)
            self.regressor.addConstr(quicksum(augment_features[i, j] * coef[j] for j in range(num_var)) >= -ub_Xcoef_inf)

        # l1 norm constraint
        abs_coef = self.regressor.addVars(num_var, lb=0, name="abs_coef")
        for i in range(num_var):
            self.regressor.addConstr(coef[i] <= abs_coef[i])
            self.regressor.addConstr(coef[i] >= -abs_coef[i])
        self.regressor.addConstr(abs_coef.sum() <= ub_coef_1)

        abs_Xcoef = self.regressor.addVars(num_samples, lb=0, name="abs_Xcoef")
        for i in range(num_samples):
            self.regressor.addConstr(quicksum(augment_features[i, j] * coef[j] for j in range(num_var)) <= abs_Xcoef[i])
            self.regressor.addConstr(quicksum(augment_features[i, j] * coef[j] for j in range(num_var)) >= -abs_Xcoef[i])
        self.regressor.addConstr(abs_Xcoef.sum() <= ub_Xcoef_1)

        # pairwise constraints
        correlation_matrix = np.corrcoef(augment_features.T)
        assert correlation_matrix.shape[0] == num_var
        self.regressor.addConstrs(
            (indicators[i] + indicators[j] >= 1 for i in range(num_var) for j in range(i + 1, num_var)
             if np.abs(correlation_matrix[i, j]) > self.max_corr_), "Limited_Pairwise_Multicollinearity"
        )

        # transformation constraints
        # indicators[i] + indicators[i + num_features] + indicators[i + 2 * num_features] >=2
        self.regressor.addConstrs(
            (sum([indicators[i + j * num_features] for j in range(num_transformations)]) >= num_transformations - 1
                for i in range(num_features)), "Nonlinear Transformation1"
        )

        Quad1 = np.dot(augment_features.T, augment_features)

        lin = np.dot(y.T, augment_features)

        obj = quicksum(0.5 * Quad1[i, j] * coef[i] * coef[j]
                       for i, j in product(range(num_var), repeat=2))
        obj -= quicksum(lin[i] * coef[i] for i in range(num_var))

        obj += 0.5 * np.dot(y, y)

        self.regressor.setObjective(obj, GRB.MINIMIZE)

        if not self.verbose:
            self.regressor.params.OutputFlag = 0
        self.regressor.optimize()

        coef = np.array([coef[i].X for i in range(num_var)])
        coef[abs(coef) < 1e-5] = 0
        if self.normalize:
            self.coef_ = coef
            self.intercept_ = y_mean
        else:
            self.coef_ = y_std / X_std * coef
            self.intercept_ = y_mean - np.dot(self.coef_, X_mean)
        return self

    # def eval_miqp(self, features):
    #     augument_features = features
    #     if self.transformation is not None:
    #         for func in self.transformation:
    #             augument_features = pd.concat([augument_features, func(features)], axis=1)
    #
    #     return np.dot(augument_features, self.beta) + self.intercept

    # def criteria(self):
    #     n, _ = self.Xtest.shape
    #     p = sum(self.beta != 0) + 1  # +1 for intercept
    #     SSE = sum((self.eval_miqp(self.Xtest) - self.ytest)**2)
    #     A = np.eye(n) - 1/n * np.dot(np.ones((n, 1)), np.ones((n, 1)).T)
    #     SSTO = np.dot(np.dot(self.ytest.T, A), self.ytest)
    #     Rsquare = 1 - SSE/SSTO
    #     MSE = SSE/(n-p)
    #
    #     AIC = n * np.log(SSE) + 2*p
    #     BIC = n * np.log(SSE) + np.log(n)*p
    #     nonzero = sum(self.beta != 0) + 1
    #     result = pd.DataFrame([[Rsquare, MSE, AIC, BIC, nonzero]], columns=["Rsquare", "MSE", "AIC", "BIC", "nonzero"]
    #                           , index=[p])
    #     return result
    #
    # def summary(self):
    #     _, dim = self.Xtrain.shape
    #     print("indentity", [ind+1 for ind, val in enumerate(self.beta[0:dim]) if val != 0])
    #     for i, func in enumerate(model.transformation):
    #         print(str(func), [ind+1 for ind, val in enumerate(self.beta[(dim*i+dim):(dim*i+2*dim)]) if val != 0])
    #
    # def select_model(self, nonzero_range=range(10, 12)):
    #     print("the max number of non zero beta is:", self.max_k)
    #     result = np.zeros((len(nonzero_range)+2, 5))
    #     beta = None
    #     for ind, k in enumerate(nonzero_range):
    #         assert k <= self.max_k
    #         print("current k", k)
    #         intercept, beta = self.miqp(non_zero=k, warm_up=beta, timelimit=10)
    #
    #         result[ind, :] = model.criteria()
    #         print(model.criteria())
    #
    #         # print([ind + 1 for ind, val in enumerate(beta[0:dim]) if val != 0])
    #         # print([ind + 1 for ind, val in enumerate(beta[dim:2 * dim]) if val != 0])
    #         # print([ind + 1 for ind, val in enumerate(beta[2 * dim:3 * dim]) if val != 0])
    #
    #         print("================================")
    #
    #     Xtrain_lasso = np.concatenate([np.ones((self.Xtrain.shape[0], 1)), self.Xtrain], axis=1)
    #     Xtest_lasso = np.concatenate([np.ones((self.Xtest.shape[0], 1)), self.Xtest], axis=1)
    #
    #     # ordinary least square
    #     lr = linear_model.LinearRegression()
    #     lr.fit(Xtrain_lasso, self.ytrain)
    #
    #     n = Xtest_lasso.shape[0]
    #     p = sum(lr.coef_ != 0) + 1  # +1 for intercept
    #     SSE = sum((lr.predict(Xtest_lasso) - self.ytest)**2)
    #     A = np.eye(n) - 1/n * np.dot(np.ones((n, 1)), np.ones((n, 1)).T)
    #     SSTO = np.dot(np.dot(self.ytest.T, A), self.ytest)
    #     Rsquare = 1 - SSE/SSTO
    #     MSE = SSE/(n-p)
    #     AIC = n * np.log(SSE) + 2*p
    #     BIC = n * np.log(SSE) + np.log(n)*p
    #     nonzero = sum(abs(lr.coef_) > 1e-5)
    #
    #     result[-2, :] = [Rsquare, MSE, AIC, BIC, nonzero]
    #
    #     # lasso
    #     lasso = linear_model.LassoCV(cv=5, max_iter=2000)
    #     lasso.fit(Xtrain_lasso, self.ytrain)
    #
    #     n = Xtest_lasso.shape[0]
    #     p = sum(lasso.coef_ != 0) + 1  # +1 for intercept
    #     SSE = sum((lasso.predict(Xtest_lasso) - self.ytest)**2)
    #     A = np.eye(n) - 1/n * np.dot(np.ones((n, 1)), np.ones((n, 1)).T)
    #     SSTO = np.dot(np.dot(self.ytest.T, A), self.ytest)
    #     Rsquare = 1 - SSE/SSTO
    #     MSE = SSE/(n-p)
    #     AIC = n * np.log(SSE) + 2*p
    #     BIC = n * np.log(SSE) + np.log(n)*p
    #     nonzero = sum(abs(lasso.coef_) > 1e-5)
    #
    #     result[-1, :] = [Rsquare, MSE, AIC, BIC, nonzero]
    #     result = pd.DataFrame(result, columns=["Rsquare", "MSE", "AIC", "BIC", "nonzero"]
    #                           , index=[*nonzero_range, "ols", "lasso"])
    #     # print(result)
    #     return result


if __name__ == "__main__":
    n, p = 100, 20
    rng = np.random.RandomState(123)
    X = rng.random(size=(n, p))
    beta = np.zeros((p+1, 1))
    beta[0] = 3
    beta[[1, 2, 3]] = 1
    error = rng.randn(n, 1)
    y = np.dot(np.concatenate([np.ones((n, 1)), X], axis=1), beta) + error
    model = L0Regression(verbose=False).fit(X, y, k=3)
    print(model.intercept_, model.coef_)


























