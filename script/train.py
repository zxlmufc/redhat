import argparse

import numpy as np
import xgboost as xgb
from bayes_opt import BayesianOptimization
from sklearn.cross_validation import KFold
from sklearn.metrics import auc

import utils


class BaseAlgo(object):

    def fit_predict(self, train, val=None, test=None, **kwa):
        self.fit(train[0], train[1], val[0] if val else None, val[1] if val else None, **kwa)

        if val is None:
            return self.predict(test[0])
        else:
            return self.predict(val[0]), self.predict(test[0])


class Xgb(BaseAlgo):

    default_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'silent': 0,
    }

    def __init__(self, params, n_iter=400):
        self.params = self.default_params.copy()

        for k in params:
            self.params[k] = params[k]

        self.n_iter = n_iter

    def fit(self, X_train, y_train, X_eval=None, y_eval=None, seed=42, feature_names=None, eval_func=None, size_mult=None, name=None):
        feval = lambda y_pred, y_true: ('auc', eval_func(y_true.get_label(), y_pred))

        params = self.params.copy()
        params['seed'] = seed
        # params['base_score'] = np.median(y_train)
        params['base_score'] = 0.16

        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)

        if X_eval is None:
            watchlist = [(dtrain, 'train')]
        else:
            deval = xgb.DMatrix(X_eval, label=y_eval, feature_names=feature_names)
            watchlist = [(dtrain, 'train'), (deval, 'eval')]

        if size_mult is None:
            n_iter = self.n_iter
        else:
            n_iter = int(self.n_iter * size_mult)

        self.iter = 0
        self.model = xgb.train(params, dtrain, n_iter, evals=watchlist,  verbose_eval= 5, early_stopping_rounds=100)
        self.model.dump_model('xgb-%s.dump' % name, with_stats=True)
        self.feature_names = feature_names

        print ("    Feature importances: %s" % ', '.join('%s: %d' % t for t in sorted(self.model.get_fscore().items(), key=lambda t: -t[1])))

    def predict(self, X):
        return self.model.predict(xgb.DMatrix(X, feature_names=self.feature_names))

    def optimize(self, X_train, y_train, X_eval, y_eval, param_grid, eval_func=None, seed=42):
        feval = lambda y_pred, y_true: ('auc', eval_func(y_true.get_label(), y_pred))

        dtrain = xgb.DMatrix(X_train, label=y_train)
        deval = xgb.DMatrix(X_eval, label=y_eval)

        def fun(**kw):
            params = self.params.copy()
            params['seed'] = seed
            params['base_score'] = np.median(y_train)

            for k in kw:
                if type(param_grid[k][0]) is int:
                    params[k] = int(kw[k])
                else:
                    params[k] = kw[k]

            print("Trying %s..." % str(params))

            self.iter = 0

            model = xgb.train(params, dtrain, 10000, [(dtrain, 'train'), (deval, 'eval')], self.objective, feval, verbose_eval=20, early_stopping_rounds=100)

            print("Score %.5f at iteration %d" % (model.best_score, model.best_iteration))

            return - model.best_score

        opt = BayesianOptimization(fun, param_grid)
        opt.maximize(n_iter=100)

        print("Best auc: %.5f, params: %s" % (opt.res['max']['max_val'], opt.res['mas']['max_params']))


class LightGBM(BaseAlgo):

    default_params = {
        'exec_path': 'lightgbm',
        'num_threads': 4
    }

    def __init__(self, params):
        self.params = self.default_params.copy()

        for k in params:
            self.params[k] = params[k]

    def fit(self, X_train, y_train, X_eval=None, y_eval=None, seed=42, feature_names=None, eval_func=None, **kwa):
        params = self.params.copy()
        params['bagging_seed'] = seed
        params['feature_fraction_seed'] = seed + 3

        self.model = GBMRegressor(**params)

        if X_eval is None:
            self.model.fit(X_train, y_train)
        else:
            self.model.fit(X_train, y_train, test_data=[(X_eval, y_eval)])

    def predict(self, X):
        return self.model.predict(X)


presets = {
        'xgb-benchmark': {
            'features': ['people_char_10', 'people_char_11', 'people_char_12', 'people_char_13', 'people_char_14', 'people_char_15', 'people_char_16',
                         'people_char_17', 'people_char_18', 'people_char_19', 'people_char_20', 'people_char_21', 'people_char_22', 'people_char_23',
                         'people_char_24', 'people_char_25', 'people_char_26', 'people_char_27', 'people_char_28', 'people_char_29', 'people_char_30',
                         'people_char_31', 'people_char_32', 'people_char_33', 'people_char_34', 'people_char_35', 'people_char_36', 'people_char_37',
                         'people_char_38'],
            'dataset': "all_data0",
            'model': Xgb({'max_depth': 5, 'eta': 1}, n_iter=10),
            'n_split': 1,
            'n_folds': 2,
            'param_grid': {'colsample_bytree': [0.2, 1.0]},
        }}


def train_model(preset):

    n_splits = preset['n_split']
    n_folds = preset.get('n_folds', 1)
    n_bags = preset.get('n_folds', 1)

    feature_names = preset['features']

    aucs_list = []

    y_aggregator = preset.get('agg', np.mean)

    train_x, train_y, test_x = utils.load_dataset(preset, mode="eval")
    train_x, train_y, test_x = train_x.values, train_y.values, test_x.values

    train_p = np.zeros((train_x.shape[0], n_bags))
    test_foldavg_p = np.zeros((test_x.shape[0], n_bags * n_folds))
    # test_fulltrain_p = np.zeros((test_x.shape[0], n_bags))

    for split in range(n_splits):
        print("Training split %d..." % split)

        for fold, (fold_train_idx, fold_eval_idx) in enumerate(KFold(len(train_y), n_folds, shuffle=True, random_state=19920707)):
            # if args.fold is not None and fold != args.fold:
            #    continue

            # print("  Fold %d..." % fold)

            fold_train_x = train_x[fold_train_idx]
            fold_train_y = train_y[fold_train_idx]

            fold_eval_x = train_x[fold_eval_idx]
            fold_eval_y = train_y[fold_eval_idx]

            fold_test_x = test_x

            fold_feature_names = list(feature_names)
            eval_p = np.zeros((fold_eval_x.shape[0], n_bags))

            for bag in range(n_bags):
                print("Training model %d..." % bag)

                rs = np.random.RandomState(19930114+bag)

                bag_train_x = fold_train_x
                bag_train_y = fold_train_y
                bag_eval_x = fold_eval_x
                bag_eval_y = fold_eval_y
                bag_test_x = fold_test_x

                pe, pt = preset['model'].fit_predict(train=(bag_train_x, bag_train_y),
                                                     val=(bag_eval_x, bag_eval_y),
                                                     test=(bag_test_x,),
                                                     seed=20170707,
                                                     feature_names=fold_feature_names,
                                                     eval_func=lambda yt, yp: auc(yt,yp),
                                                     name='%s-fold-%d-%d' % (args.preset, fold, bag))

                eval_p[:, bag] += pe
                test_foldavg_p[:, split * n_folds * n_bags + fold * n_bags + bag] = pt

                train_p[fold_eval_idx, split * n_bags + bag] = pe

                print("Current bag AUC of model: %.5f" % auc(fold_eval_y, pe, reorder=True))

            print("AUC mean prediction : %.5f" % auc(fold_eval_y, np.mean(eval_p, axis=1), reorder=True))


            # Calculate err
            aucs_list.append(auc(fold_eval_y, y_aggregator(eval_p, axis=1), reorder=True))
            # Free mem
            del fold_train_x, fold_train_y, fold_eval_x, fold_eval_y

        print("AUC: %.5f" % aucs_list[-1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('preset', type=str, help='model preset')
    parser.add_argument('--fold', type=int, help='specify fold')
    args = parser.parse_args()
    preset = presets[args.preset]
    train_model(preset)
