import argparse

import lightgbm as lgb
import numpy as np
import utils
import xgboost as xgb
from bayes_opt import BayesianOptimization
from sklearn.cross_validation import KFold
from sklearn.metrics import roc_auc_score


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
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': 0.1,
        'feature_fraction': 0.9,
        'verbose': 0
    }

    def __init__(self, params, n_iter = 100):
        self.params = self.default_params.copy()
        self.n_iter = n_iter

        for k in params:
            self.params[k] = params[k]

    def fit(self, X_train, y_train, X_eval=None, y_eval=None, seed=42, feature_names=None, eval_func=None, size_mult=None, **kwa):
        params = self.params.copy()

        if size_mult is None:
            n_iter = self.n_iter
        else:
            n_iter = int(self.n_iter * size_mult)

        lgb_train = lgb.Dataset(X_train, y_train.flatten(), free_raw_data=False)

        if X_eval is None:
            self.model = lgb.train(params, lgb_train)

        else:
            lgb_eval = lgb.Dataset(X_eval, y_eval.flatten(), reference=lgb_train, free_raw_data=False)
            self.model = lgb.train(params, lgb_train, n_iter, early_stopping_rounds=100, valid_sets=lgb_eval)

        # self.model.dump_model('lgb-%s.dump' % name, num_iteration=-1)

    def predict(self, X):
        return self.model.predict(X)


presets = {
        'xgb-benchmark': {
            'features': ['encode_feature'],
            'model': Xgb({'max_depth': 5, 'eta': 0.1}, n_iter=10),
            'n_split': 1,
            'n_folds': 2,
            'param_grid': {'colsample_bytree': [0.2, 1.0]},
        },
        'lgb-benchmark': {
            'features': ['encode_feature'],
            'model': LightGBM({'num_leaves': 8, 'num_boost_round': 80}),
            'n_split': 1,
            'n_folds': 2
        }

}


def train_model(preset):

    n_splits = preset['n_split']
    n_folds = preset.get('n_folds', 1)
    n_bags = preset.get('n_folds', 1)

    aucs_list = []

    y_aggregator = preset.get('agg', np.mean)

    train_x, train_y, test_x = utils.load_dataset(preset)

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
                                                     eval_func=lambda yt, yp: roc_auc_score(yt, yp),
                                                     name='%s-fold-%d-%d' % (args.preset, fold, bag))

                eval_p[:, bag] += pe
                test_foldavg_p[:, split * n_folds * n_bags + fold * n_bags + bag] = pt

                train_p[fold_eval_idx, split * n_bags + bag] = pe

                print("Current bag AUC of model: %.5f" % roc_auc_score(fold_eval_y, pe))

            print("AUC mean prediction : %.5f" % roc_auc_score(fold_eval_y, np.mean(eval_p, axis=1)))

            # Calculate err
            aucs_list.append(roc_auc_score(fold_eval_y, y_aggregator(eval_p, axis=1)))
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
